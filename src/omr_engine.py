"""
High-level OpenCV OMR engine used by the Flask webapp.

The rest of the app should not need to know about thresholding, perspective
warping, or answer-box masks. It should send this module image bytes plus an
answer key and receive a plain Python dictionary with scores and details.
"""

from __future__ import annotations

import cv2

from bubble_reader import interpret_answers, read_all_boxes
from grader import grade_answers
from preprocessing import (
    compute_blur_score,
    compute_brightness,
    convert_to_gray,
    detect_corner_markers,
    load_image_from_bytes,
    threshold_dark_marks,
    warp_to_sheet,
)
from sheet_layout import ANSWER_BOX_SIZE, OPTIONS, SHEET_HEIGHT, SHEET_WIDTH
from sheet_layout import generate_answer_box_centers
from visualization import draw_box_debug


def _quality_warnings(blur_score: float, brightness: float) -> list[str]:
    """
    Turn raw image-quality numbers into teacher-friendly warnings.

    These thresholds are intentionally conservative. A warning does not stop
    grading; it simply tells the teacher that a retake may be better if the
    detected answers look suspicious.
    """

    warnings = []

    if blur_score < 60:
        warnings.append("The photo may be blurry. Retake if the answers look wrong.")

    if brightness < 65:
        warnings.append("The photo is quite dark. Try better light if detection is poor.")

    if brightness > 245:
        warnings.append("The photo is very bright. Avoid glare or flash reflection.")

    return warnings


def _normalize_answer_key(answer_key: dict, question_count: int) -> dict[str, str]:
    """
    Keep only the answers that belong to the selected test size.

    Form submissions and JSON values may arrive with integer or string keys.
    The grader expects string keys, so this function normalizes the shape before
    scoring.
    """

    normalized = {}

    for question_number in range(1, question_count + 1):
        key = str(question_number)
        value = answer_key.get(key, answer_key.get(question_number))

        if value:
            normalized[key] = value

    return normalized


def process_scan(
    image_bytes: bytes,
    answer_key: dict,
    question_count: int,
    options: tuple[str, ...] = OPTIONS,
) -> dict:
    """
    Process one uploaded sheet image and return grading data.

    Steps:
    1. Decode the uploaded phone image.
    2. Check blur and brightness.
    3. Detect the four corner markers.
    4. Warp the camera photo into the fixed 1000 x 1400 sheet coordinate space.
    5. Threshold dark marks.
    6. Measure every answer box.
    7. Interpret marks and grade against the saved answer key.
    8. Produce an annotated image that can be shown on the preview page.
    """

    original = load_image_from_bytes(image_bytes)
    original_gray = convert_to_gray(original)

    blur_score = compute_blur_score(original_gray)
    brightness = compute_brightness(original_gray)
    warnings = _quality_warnings(blur_score, brightness)

    detected_markers, marker_warnings = detect_corner_markers(original)
    warnings.extend(marker_warnings)

    if not detected_markers:
        # For a real OMR workflow, grading a scan without all four orientation
        # markers is dangerous because the answer boxes may be sampled from the
        # wrong places. Rejecting bad scans is better than saving a confident
        # but wrong score.
        raise ValueError(" ".join(marker_warnings))

    warped, warp_quality = warp_to_sheet(original, detected_markers)
    used_perspective_correction = True

    if warp_quality["homography_inliers"] < 12:
        warnings.append("The sheet alignment is weak. Retake from a less extreme angle if answers look wrong.")

    if warp_quality["max_reprojection_error"] > 12:
        warnings.append("The page perspective is very distorted. Retake the photo with the page flatter in view.")

    warped_gray = convert_to_gray(warped)
    thresholded = threshold_dark_marks(warped_gray)

    answer_box_centers = generate_answer_box_centers(question_count, options)
    raw_box_scores = read_all_boxes(
        thresholded_image=thresholded,
        box_centers=answer_box_centers,
        box_size=ANSWER_BOX_SIZE,
    )

    interpreted_answers = interpret_answers(raw_box_scores)
    normalized_key = _normalize_answer_key(answer_key, question_count)
    summary, detailed_results = grade_answers(interpreted_answers, normalized_key)

    annotated = draw_box_debug(
        image=warped,
        box_centers=answer_box_centers,
        interpreted_answers=interpreted_answers,
        box_size=ANSWER_BOX_SIZE,
    )

    return {
        "summary": summary,
        "details": detailed_results,
        "detected_answers": interpreted_answers,
        "raw_scores": raw_box_scores,
        "quality": {
            "blur_score": round(float(blur_score), 2),
            "brightness": round(float(brightness), 2),
            "used_perspective_correction": used_perspective_correction,
            **warp_quality,
        },
        "warnings": warnings,
        "warped_image": warped,
        "thresholded_image": thresholded,
        "annotated_image": annotated,
    }


def build_manual_interpretation(
    final_answers: dict[str, str | None],
    question_count: int,
) -> dict[str, dict]:
    """
    Convert teacher-edited answers back into the scanner result shape.

    The preview page lets the teacher correct uncertain detections before
    saving. To reuse the same grader, we represent those manual selections as
    high-confidence valid answers or blanks.
    """

    interpreted = {}

    for question_number in range(1, question_count + 1):
        key = str(question_number)
        selected = final_answers.get(key)

        if selected:
            interpreted[key] = {
                "selected": selected,
                "status": "valid",
                "confidence": 1.0,
                "scores": {},
            }
        else:
            interpreted[key] = {
                "selected": None,
                "status": "blank",
                "confidence": 1.0,
                "scores": {},
            }

    return interpreted
