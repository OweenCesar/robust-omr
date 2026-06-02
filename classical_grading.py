# Classical answer interpretation and grading for the OMR project.
#
# This stage starts after bubble candidates have already been detected and scored.
# It follows the same idea as the reference paper: order the option circles, group
# them into answer rows, assign a selected value, and compare with an answer key.

import json
from pathlib import Path

import numpy as np

from preprocessing import resize_image, image_to_grayscale
from counter_detection import (
    build_warped_rectangle_sections,
    find_external_contours,
    prepare_rectangle_detection_image,
    select_largest_rectangular_contours,
)
from bubble_detection import (
    detect_bubble_candidates_hough,
    filter_bubbles_by_color_presence,
    score_bubble_marks,
)


DEFAULT_OPTIONS = ("A", "B", "C", "D")


def cluster_items_by_axis(items, axis, tolerance):
    """
    Group dictionaries by a numeric axis such as x or y.

    Parameters:
    items (list): Items containing the selected axis.
    axis (str): Dictionary key to group by, usually "x" or "y".
    tolerance (float): Maximum distance from the current group center.

    Returns:
    list: Groups with center and items.
    """
    if not items:
        return []

    groups = []

    for item in sorted(items, key=lambda current_item: current_item[axis]):
        if not groups:
            groups.append([item])
            continue

        current_center = np.mean(
            [grouped_item[axis] for grouped_item in groups[-1]]
        )

        if abs(item[axis] - current_center) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])

    grouped_items = []

    for group in groups:
        grouped_items.append({
            "center": round(float(np.mean([item[axis] for item in group])), 3),
            "items": group,
        })

    return grouped_items


def split_option_column_groups(x_groups, options_per_question=4):
    """
    Split detected x-axis bubble columns into answer sections.

    In the Tamaulipas-style sheets used by the paper, the answer section is formed
    by three vertical blocks. Each block has four option columns.
    """
    sections = []

    for start_index in range(0, len(x_groups), options_per_question):
        sections.append(x_groups[start_index:start_index + options_per_question])

    return sections


def build_answer_rows(
    scored_bubbles,
    options=DEFAULT_OPTIONS,
    x_tolerance=10,
    y_tolerance=10,
):
    """
    Convert scored bubble candidates into ordered answer rows.

    Returns rows in question order: top-to-bottom inside each answer block, then
    left-to-right across the blocks.
    """
    options = tuple(options)
    options_per_question = len(options)
    x_groups = cluster_items_by_axis(scored_bubbles, axis="x", tolerance=x_tolerance)
    option_column_groups = split_option_column_groups(
        x_groups=x_groups,
        options_per_question=options_per_question,
    )

    rows = []
    warnings = []
    question_number = 1

    if len(x_groups) % options_per_question != 0:
        warnings.append(
            "The number of detected option columns is not divisible by the "
            f"number of options ({options_per_question})."
        )

    for section_index, section_columns in enumerate(option_column_groups, start=1):
        if len(section_columns) != options_per_question:
            warnings.append(
                f"Section {section_index} has {len(section_columns)} option "
                f"columns instead of {options_per_question}."
            )
            continue

        section_bubbles = [
            bubble
            for column_group in section_columns
            for bubble in column_group["items"]
        ]
        y_groups = cluster_items_by_axis(
            section_bubbles,
            axis="y",
            tolerance=y_tolerance,
        )

        for row_index, y_group in enumerate(y_groups, start=1):
            row_bubbles = y_group["items"]
            option_cells = []
            missing_candidates = 0
            duplicate_candidates = 0

            for option_label, column_group in zip(options, section_columns):
                x_center = column_group["center"]
                candidates = [
                    bubble
                    for bubble in row_bubbles
                    if abs(bubble["x"] - x_center) <= x_tolerance
                ]

                if len(candidates) == 0:
                    missing_candidates += 1
                    chosen_bubble = None
                else:
                    if len(candidates) > 1:
                        duplicate_candidates += len(candidates) - 1

                    chosen_bubble = max(
                        candidates,
                        key=lambda bubble: (
                            bubble.get("dark_ratio", 0.0),
                            -abs(bubble["x"] - x_center),
                        ),
                    )

                option_cells.append({
                    "label": option_label,
                    "x_center": x_center,
                    "bubble": chosen_bubble,
                    "dark_ratio": 0.0 if chosen_bubble is None else float(
                        chosen_bubble.get("dark_ratio", 0.0)
                    ),
                })

            rows.append({
                "question": question_number,
                "section": section_index,
                "row_in_section": row_index,
                "y_center": y_group["center"],
                "options": option_cells,
                "missing_candidates": missing_candidates,
                "duplicate_candidates": duplicate_candidates,
            })
            question_number += 1

    layout = {
        "option_column_count": len(x_groups),
        "answer_section_count": len(option_column_groups),
        "questions_detected": len(rows),
        "warnings": warnings,
    }

    return rows, layout


def interpret_answer_row(
    row,
    mark_threshold=0.18,
    weak_threshold=0.08,
    ambiguity_margin=0.08,
):
    """
    Assign one answer value to a row of option bubbles.

    Output values:
    - A/B/C/D: one selected option
    - X: no option selected
    - M: multiple options selected
    - ?: unclear answer that should be reviewed
    """
    option_scores = {
        option["label"]: float(option["dark_ratio"])
        for option in row["options"]
    }
    sorted_scores = sorted(
        option_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    selected_options = [
        label
        for label, score in option_scores.items()
        if score >= mark_threshold
    ]
    weak_options = [
        label
        for label, score in option_scores.items()
        if weak_threshold <= score < mark_threshold
    ]

    quality_flags = []

    if row["missing_candidates"] > 0:
        quality_flags.append("missing bubble candidate")

    if row["duplicate_candidates"] > 0:
        quality_flags.append("duplicate bubble candidate")

    if len(selected_options) > 1:
        answer = "M"
        status = "multiple"
        needs_review = True
        review_reason = "multiple marks detected"
        confidence = 0.0
    elif len(selected_options) == 0:
        if weak_options or row["missing_candidates"] > 0:
            answer = "?"
            status = "unclear"
            needs_review = True
            review_reason = "unclear: please review"
            confidence = 0.0
        else:
            answer = "X"
            status = "blank"
            needs_review = False
            review_reason = ""
            confidence = round(1.0 - sorted_scores[0][1], 3)
    else:
        answer = selected_options[0]
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        if second_score >= weak_threshold and (top_score - second_score) <= ambiguity_margin:
            answer = "?"
            status = "unclear"
            needs_review = True
            review_reason = "unclear: please review"
            confidence = round(max(0.0, top_score - second_score), 3)
        else:
            status = "single"
            needs_review = False
            review_reason = ""
            confidence = round(max(0.0, top_score - second_score), 3)

    interpreted_row = row.copy()
    interpreted_row.update({
        "answer": answer,
        "status": status,
        "selected_options": selected_options,
        "weak_options": weak_options,
        "scores": option_scores,
        "confidence": confidence,
        "needs_review": needs_review,
        "review_reason": review_reason,
        "quality_flags": quality_flags,
    })

    return interpreted_row


def interpret_answer_rows(
    rows,
    mark_threshold=0.18,
    weak_threshold=0.08,
    ambiguity_margin=0.08,
):
    """
    Interpret every grouped answer row.
    """
    return [
        interpret_answer_row(
            row=row,
            mark_threshold=mark_threshold,
            weak_threshold=weak_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        for row in rows
    ]


def normalize_answer_key(answer_key_data):
    """
    Normalize answer-key JSON data to a dictionary keyed by question number.
    """
    if isinstance(answer_key_data, list):
        return {
            question_number: str(answer).upper()
            for question_number, answer in enumerate(answer_key_data, start=1)
        }

    answers = answer_key_data.get("answers", answer_key_data)

    if isinstance(answers, list):
        return {
            question_number: str(answer).upper()
            for question_number, answer in enumerate(answers, start=1)
        }

    return {
        int(question_number): str(answer).upper()
        for question_number, answer in answers.items()
    }


def load_answer_key(answer_key_path):
    """
    Load an answer key from JSON.
    """
    answer_key_path = Path(answer_key_path)

    with answer_key_path.open("r", encoding="utf-8") as file:
        answer_key_data = json.load(file)

    return normalize_answer_key(answer_key_data)


def grade_answers(interpreted_rows, answer_key):
    """
    Compare interpreted answers with a teacher answer key.
    """
    answer_key = normalize_answer_key(answer_key)
    graded_rows = []

    for row in interpreted_rows:
        question = int(row["question"])
        correct_answer = answer_key.get(question)
        student_answer = row["answer"]

        if correct_answer is None:
            is_correct = None
            grading_status = "not_in_answer_key"
        elif row["status"] == "single" and student_answer == correct_answer:
            is_correct = True
            grading_status = "correct"
        else:
            is_correct = False
            grading_status = "incorrect"

        graded_row = row.copy()
        graded_row.update({
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "grading_status": grading_status,
        })
        graded_rows.append(graded_row)

    keyed_rows = [
        row
        for row in graded_rows
        if row["correct_answer"] is not None
    ]
    correct_count = sum(1 for row in keyed_rows if row["is_correct"] is True)
    incorrect_count = sum(1 for row in keyed_rows if row["is_correct"] is False)

    summary = {
        "questions_detected": len(interpreted_rows),
        "questions_in_answer_key": len(keyed_rows),
        "correct": correct_count,
        "incorrect": incorrect_count,
        "blank": sum(1 for row in interpreted_rows if row["status"] == "blank"),
        "multiple": sum(1 for row in interpreted_rows if row["status"] == "multiple"),
        "unclear": sum(1 for row in interpreted_rows if row["status"] == "unclear"),
        "needs_review": sum(1 for row in interpreted_rows if row["needs_review"]),
    }

    if keyed_rows:
        summary["score_percent"] = round(100 * correct_count / len(keyed_rows), 2)
    else:
        summary["score_percent"] = None

    return {
        "summary": summary,
        "answers": graded_rows,
    }


def build_review_messages(interpreted_rows):
    """
    Create short human-review messages.
    """
    messages = []

    for row in interpreted_rows:
        if not row["needs_review"]:
            continue

        question = row["question"]

        if row["status"] == "multiple":
            messages.append(f"Q{question}: multiple marks detected")
        else:
            messages.append(f"Q{question}: unclear, please review")

    return messages


def run_classical_omr_pipeline(
    image,
    answer_key=None,
    working_size=(900, 1200),
    min_rectangle_area=5000,
    rectangle_count=2,
    x_tolerance=10,
    y_tolerance=10,
    mark_threshold=0.18,
    weak_threshold=0.08,
    ambiguity_margin=0.08,
    use_color_filter=True,
):
    """
    Run the current classical OMR method from sheet image to interpreted answers.

    The largest detected rectangle is treated as the answer section. The smaller
    rectangle is preserved in the returned sections for later ID processing.
    """
    resized_image = resize_image(image, size=working_size)
    grayscale_image = image_to_grayscale(resized_image)
    rectangle_binary_image = prepare_rectangle_detection_image(grayscale_image)
    contours = find_external_contours(rectangle_binary_image)
    rectangular_contours = select_largest_rectangular_contours(
        contours=contours,
        count=rectangle_count,
        min_area=min_rectangle_area,
    )
    sections = build_warped_rectangle_sections(
        image=resized_image,
        rectangular_contours=rectangular_contours,
        count=rectangle_count,
    )

    if not sections:
        raise ValueError("No answer-sheet rectangles were detected.")

    answer_section = max(sections, key=lambda section: section["area"])
    answer_section_image = answer_section["warped_image"]

    raw_bubbles = detect_bubble_candidates_hough(
        answer_section_image,
        dp=1.2,
        min_distance=10,
        param1=50,
        param2=12,
        min_radius=3,
        max_radius=12,
    )
    if use_color_filter:
        filtered_bubbles = filter_bubbles_by_color_presence(
            answer_section_image,
            raw_bubbles,
        )
    else:
        filtered_bubbles = raw_bubbles
    scored_bubbles = score_bubble_marks(
        answer_section_image,
        filtered_bubbles,
    )
    answer_rows, layout = build_answer_rows(
        scored_bubbles=scored_bubbles,
        x_tolerance=x_tolerance,
        y_tolerance=y_tolerance,
    )
    interpreted_rows = interpret_answer_rows(
        rows=answer_rows,
        mark_threshold=mark_threshold,
        weak_threshold=weak_threshold,
        ambiguity_margin=ambiguity_margin,
    )

    grading = None

    if answer_key is not None:
        grading = grade_answers(
            interpreted_rows=interpreted_rows,
            answer_key=answer_key,
        )

    return {
        "resized_image": resized_image,
        "rectangle_binary_image": rectangle_binary_image,
        "sections": sections,
        "answer_section": answer_section,
        "raw_bubbles": raw_bubbles,
        "filtered_bubbles": filtered_bubbles,
        "scored_bubbles": scored_bubbles,
        "layout": layout,
        "answers": interpreted_rows,
        "review_messages": build_review_messages(interpreted_rows),
        "grading": grading,
    }


def compact_answer_rows(interpreted_rows):
    """
    Keep the answer data needed for reports without storing full bubble objects.
    """
    compact_rows = []

    for row in interpreted_rows:
        compact_rows.append({
            "question": row["question"],
            "answer": row["answer"],
            "status": row["status"],
            "scores": row["scores"],
            "confidence": row["confidence"],
            "needs_review": row["needs_review"],
            "review_reason": row["review_reason"],
            "quality_flags": row["quality_flags"],
            "correct_answer": row.get("correct_answer"),
            "is_correct": row.get("is_correct"),
            "grading_status": row.get("grading_status"),
        })

    return compact_rows


def save_grading_report(report_path, interpreted_rows, grading=None, layout=None):
    """
    Save a compact JSON grading report.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "layout": layout,
        "summary": None if grading is None else grading["summary"],
        "review_messages": build_review_messages(interpreted_rows),
        "answers": compact_answer_rows(interpreted_rows),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report
