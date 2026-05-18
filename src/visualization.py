import cv2


def draw_bubble_debug(image, bubble_centers, interpreted_answers, radius):
    """
    Draws circles around bubbles and labels the detected answers.
    """
    annotated = image.copy()

    for question, options in bubble_centers.items():
        result = interpreted_answers.get(question, {})
        selected = result.get("selected")
        status = result.get("status")

        for option, center in options.items():
            x, y = center

            color = (180, 180, 180)

            if status == "valid" and selected == option:
                color = (0, 255, 0)

            elif status == "unclear" and selected == option:
                color = (0, 255, 255)

            elif status == "multiple" and option in selected:
                color = (0, 165, 255)

            cv2.circle(annotated, (x, y), radius, color, 2)

        first_x, first_y = list(options.values())[0]
        cv2.putText(
            annotated,
            f"Q{question}: {status}",
            (first_x - 170, first_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    return annotated