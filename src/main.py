import os
import json
import argparse
from pathlib import Path
import cv2

from preprocessing import (
    load_image,
    align_to_template,
    resize_to_template,
    preprocess_for_thresholding,
    compute_blur_score,
    compute_brightness
)

from template_loader import (
    load_template,
    generate_bubble_centers
)

from bubble_reader import (
    read_all_bubbles,
    interpret_answers
)

from grader import (
    load_answer_key,
    grade_answers,
    build_review_items
)

from visualization import draw_bubble_debug


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_existing_path(path):
    path = Path(path).expanduser()

    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path

    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def resolve_output_dir(path):
    path = Path(path).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read and grade a bubble answer sheet image."
    )
    parser.add_argument(
        "--image",
        default="data/samples/test_sheet.png",
        help="Path to the photographed/scanned answer sheet."
    )
    parser.add_argument(
        "--template",
        default="templates/template_20q_abcd.json",
        help="Path to the template JSON file."
    )
    parser.add_argument(
        "--answer-key",
        default="answer_key.json",
        help="Path to the answer key JSON file."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where debug images and result JSON are written."
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip marker alignment and resize the image directly into template space."
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir):
    paths = {
        "debug": os.path.join(output_dir, "debug"),
        "annotated": os.path.join(output_dir, "annotated"),
        "warped": os.path.join(output_dir, "warped"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def write_report(output_dir, payload):
    report_path = os.path.join(output_dir, "results.json")

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return report_path


def print_summary(summary, detailed_results):
    print("\n===== RESULT SUMMARY =====")
    print(f"Correct: {summary['correct']}")
    print(f"Wrong: {summary['wrong']}")
    print(f"Blank: {summary['blank']}")
    print(f"Multiple: {summary['multiple']}")
    print(f"Unclear: {summary['unclear']}")
    print(f"Needs review: {summary['needs_review']}")
    print(f"Score: {summary['score_percent']}%")

    print("\n===== DETAILED RESULTS =====")
    for question, result in detailed_results.items():
        print(
            f"Q{question}: "
            f"student={result['student_answer']} | "
            f"correct={result['correct_answer']} | "
            f"status={result['status']} | "
            f"correct={result['is_correct']}"
        )


def print_review_items(review_items):
    print("\n===== HUMAN REVIEW =====")

    if not review_items:
        print("No uncertain or multiple-mark answers found.")
        return

    for item in review_items:
        print(item["message"])


def main():
    args = parse_args()
    image_path = resolve_existing_path(args.image)
    template_path = resolve_existing_path(args.template)
    answer_key_path = resolve_existing_path(args.answer_key)
    output_dir = resolve_output_dir(args.output_dir)
    output_paths = ensure_output_dirs(output_dir)

    print(f"[INFO] Loading image: {image_path}")
    image = load_image(str(image_path))

    print(f"[INFO] Loading template: {template_path}")
    template = load_template(str(template_path))
    bubble_centers = generate_bubble_centers(template)

    print("[INFO] Aligning image to template space...")
    if args.no_align:
        warped = resize_to_template(image, template)
        alignment_info = {
            "status": "skipped",
            "reason": "--no-align was used"
        }
    else:
        warped, alignment_info = align_to_template(image, template)

    print("[INFO] Preprocessing image...")
    gray, thresholded = preprocess_for_thresholding(warped)

    blur_score = compute_blur_score(gray)
    brightness = compute_brightness(gray)

    print(f"[INFO] Blur score: {blur_score:.2f}")
    print(f"[INFO] Brightness: {brightness:.2f}")

    print("[INFO] Reading bubbles...")
    threshold_settings = template.get("thresholds", {})
    bubble_results = read_all_bubbles(
        thresholded_image=thresholded,
        bubble_centers=bubble_centers,
        radius=template["bubble_radius"],
        inner_radius_ratio=template.get("inner_radius_ratio", 0.55)
    )

    print("[INFO] Interpreting answers...")
    interpreted_answers = interpret_answers(
        bubble_results,
        blank_threshold=threshold_settings.get("blank", 0.08),
        marked_threshold=threshold_settings.get("marked", 0.35),
        ambiguity_margin=threshold_settings.get("ambiguity_margin", 0.12)
    )

    print(f"[INFO] Loading answer key: {answer_key_path}")
    answer_key = load_answer_key(str(answer_key_path))

    print("[INFO] Grading...")
    summary, detailed_results = grade_answers(
        interpreted_answers=interpreted_answers,
        answer_key=answer_key
    )
    review_items = build_review_items(detailed_results)

    print_summary(summary, detailed_results)
    print_review_items(review_items)

    print("[INFO] Saving debug images...")

    warped_path = os.path.join(output_paths["warped"], "warped_sheet.png")
    thresholded_path = os.path.join(output_paths["debug"], "thresholded.png")
    annotated_path = os.path.join(output_paths["annotated"], "annotated_result.png")

    cv2.imwrite(warped_path, warped)
    cv2.imwrite(thresholded_path, thresholded)

    annotated = draw_bubble_debug(
        image=warped,
        bubble_centers=bubble_centers,
        interpreted_answers=interpreted_answers,
        radius=template["bubble_radius"],
        answer_key=answer_key
    )

    cv2.imwrite(annotated_path, annotated)

    report_path = write_report(
        output_dir,
        {
            "input_image": str(image_path),
            "template": str(template_path),
            "answer_key": str(answer_key_path),
            "alignment": alignment_info,
            "quality": {
                "blur_score": round(float(blur_score), 2),
                "brightness": round(float(brightness), 2),
            },
            "summary": summary,
            "review": {
                "required": len(review_items) > 0,
                "count": len(review_items),
                "items": review_items
            },
            "answers": interpreted_answers,
            "details": detailed_results,
            "outputs": {
                "warped": warped_path,
                "thresholded": thresholded_path,
                "annotated": annotated_path,
            }
        }
    )

    print(f"[INFO] Report saved: {report_path}")
    print("[DONE] OMR pipeline finished.")


if __name__ == "__main__":
    main()
