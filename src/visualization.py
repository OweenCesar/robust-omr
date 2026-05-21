import cv2


def _selected_contains(selected, option):
    if selected is None:
        return False

    if isinstance(selected, list):
        return option in selected

    return selected == option


def _label_color(status, selected, correct_answer):
    if status == "valid" and correct_answer is not None:
        return (0, 160, 0) if selected == correct_answer else (0, 0, 255)

    if status == "blank":
        return (120, 120, 120)

    if status == "unclear":
        return (0, 165, 255)

    if status == "multiple":
        return (0, 100, 255)

    return (0, 0, 255)


def draw_bubble_debug(image, bubble_centers, interpreted_answers, radius, answer_key=None):
    """
    Draws circles around bubbles and labels the detected answers.
    """
    annotated = image.copy()

    for question, options in bubble_centers.items():
        result = interpreted_answers.get(question, {})
        selected = result.get("selected")
        status = result.get("status")
        correct_answer = answer_key.get(question) if answer_key else None

        for option, center in options.items():
            x, y = center

            color = (180, 180, 180)

            if status == "valid" and selected == option:
                color = (0, 180, 0) if selected == correct_answer else (0, 0, 255)

            elif status == "unclear" and selected == option:
                color = (0, 255, 255)

            elif status == "multiple" and _selected_contains(selected, option):
                color = (0, 165, 255)

            cv2.circle(annotated, (x, y), radius, color, 2)

        first_x, first_y = list(options.values())[0]
        cv2.putText(
            annotated,
            f"Q{question}: {status}",
            (first_x - 170, first_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            _label_color(status, selected, correct_answer),
            1,
            cv2.LINE_AA
        )

    return annotated
