import cv2
import numpy as np


def create_rectangular_mask(shape, center, box_size, inner_scale=0.68):
    """
    Create a filled rectangular mask for one square answer box.

    The printed answer box has a dark border even when the student leaves it
    blank. If we measured the full box, that border would increase the mark
    score for every answer. To avoid that, the mask covers only the inner area
    of the square. A real student mark should darken this inner area, while the
    printed border mostly stays outside the measurement zone.
    """
    height, width = shape[:2]
    x, y = center

    inner_size = int(box_size * inner_scale)
    half = max(1, inner_size // 2)

    left = max(0, x - half)
    right = min(width, x + half)
    top = max(0, y - half)
    bottom = min(height, y + half)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255

    return mask


def read_single_box(thresholded_image, center, box_size):
    """
    Read one square answer box by measuring dark pixels inside its center.

    The thresholded image is expected to use THRESH_BINARY_INV, so black ink
    becomes white pixels and white paper becomes black pixels. This means
    countNonZero gives us a simple "how much ink is here?" score.
    """
    x, y = center

    mask = create_rectangular_mask(
        thresholded_image.shape,
        center=(x, y),
        box_size=box_size,
    )

    answer_pixels = cv2.bitwise_and(thresholded_image, thresholded_image, mask=mask)

    total_area = np.count_nonzero(mask)
    marked_area = cv2.countNonZero(answer_pixels)

    mark_ratio = marked_area / total_area if total_area > 0 else 0

    return mark_ratio


def read_all_boxes(thresholded_image, box_centers, box_size):
    """
    Read every square answer box for every question.

    The returned values are ratios from 0.0 to 1.0. A blank box should be near
    0.0. A fully filled box should be much higher. The interpretation function
    turns those raw ratios into statuses like valid, blank, multiple, or unclear.
    """
    results = {}

    for question, options in box_centers.items():
        results[question] = {}

        for option, center in options.items():
            mark_ratio = read_single_box(
                thresholded_image=thresholded_image,
                center=center,
                box_size=box_size
            )

            results[question][option] = mark_ratio

    return results


def interpret_answers(
    box_results,
    blank_threshold=0.08,
    marked_threshold=0.18,
    ambiguity_margin=0.05
):
    """
    Convert raw answer-box mark ratios into teacher-readable answer statuses.

    The scanner produces one numeric mark ratio for every answer box. This
    function turns those numbers into decisions:

    - valid
    - blank
    - multiple
    - unclear

    These thresholds are deliberately separated from the box-reading code so
    they can be tuned later without changing the image sampling logic.
    """
    interpreted = {}

    for question, options in box_results.items():
        sorted_options = sorted(
            options.items(),
            key=lambda item: item[1],
            reverse=True
        )

        best_option, best_score = sorted_options[0]
        second_option, second_score = sorted_options[1]

        marked_options = [
            option for option, score in options.items()
            if score >= marked_threshold
        ]

        if best_score < blank_threshold:
            interpreted[question] = {
                "selected": None,
                "status": "blank",
                "confidence": round(best_score, 3),
                "scores": options
            }

        elif len(marked_options) > 1:
            interpreted[question] = {
                "selected": marked_options,
                "status": "multiple",
                "confidence": round(best_score, 3),
                "scores": options
            }

        elif best_score - second_score < ambiguity_margin:
            interpreted[question] = {
                "selected": best_option,
                "status": "unclear",
                "confidence": round(best_score - second_score, 3),
                "scores": options
            }

        else:
            interpreted[question] = {
                "selected": best_option,
                "status": "valid",
                "confidence": round(best_score, 3),
                "scores": options
            }

    return interpreted
