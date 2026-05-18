import os
import cv2

from preprocessing import (
    load_image,
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
    grade_answers
)

from visualization import draw_bubble_debug


def main():
    image_path = "data/samples/test_sheet.png"
    template_path = "templates/template_20q_abcd.json"
    answer_key_path = "answer_key.json"

    os.makedirs("outputs/debug", exist_ok=True)
    os.makedirs("outputs/annotated", exist_ok=True)

    print("[INFO] Loading image...")
    image = load_image(image_path)

    print("[INFO] Loading template...")
    template = load_template(template_path)
    bubble_centers = generate_bubble_centers(template)

    print("[INFO] Preprocessing image...")
    gray, thresholded = preprocess_for_thresholding(image)

    blur_score = compute_blur_score(gray)
    brightness = compute_brightness(gray)

    print(f"[INFO] Blur score: {blur_score:.2f}")
    print(f"[INFO] Brightness: {brightness:.2f}")

    print("[INFO] Reading bubbles...")
    bubble_results = read_all_bubbles(
        thresholded_image=thresholded,
        bubble_centers=bubble_centers,
        radius=template["bubble_radius"]
    )

    print("[INFO] Interpreting answers...")
    interpreted_answers = interpret_answers(bubble_results)

    print("[INFO] Loading answer key...")
    answer_key = load_answer_key(answer_key_path)

    print("[INFO] Grading...")
    summary, detailed_results = grade_answers(
        interpreted_answers=interpreted_answers,
        answer_key=answer_key
    )

    print("\n===== RESULT SUMMARY =====")
    print(f"Correct: {summary['correct']}")
    print(f"Wrong: {summary['wrong']}")
    print(f"Blank: {summary['blank']}")
    print(f"Multiple: {summary['multiple']}")
    print(f"Unclear: {summary['unclear']}")
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

    print("[INFO] Saving debug images...")

    cv2.imwrite("outputs/debug/thresholded.png", thresholded)

    annotated = draw_bubble_debug(
        image=image,
        bubble_centers=bubble_centers,
        interpreted_answers=interpreted_answers,
        radius=template["bubble_radius"]
    )

    cv2.imwrite("outputs/annotated/annotated_result.png", annotated)

    print("[DONE] OMR pipeline finished.")


if __name__ == "__main__":
    main()