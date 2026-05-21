import cv2


def draw_box_debug(image, box_centers, interpreted_answers, box_size):
    """
    Draw detected answer boxes and status labels on a warped sheet image.

    The annotated image is not required for grading, but it is very useful in a
    real marking workflow. If a teacher asks "why did it mark this answer?",
    the colored overlay shows exactly where the detector looked.
    """
    annotated = image.copy()
    half = box_size // 2

    for question, options in box_centers.items():
        result = interpreted_answers.get(question, {})
        selected = result.get("selected")
        status = result.get("status")

        for option, center in options.items():
            x, y = center

            color = (180, 180, 180)

            if status == "valid" and selected == option:
                color = (0, 180, 0)

            elif status == "unclear" and selected == option:
                color = (0, 180, 255)

            elif status == "multiple" and isinstance(selected, list) and option in selected:
                color = (0, 120, 255)

            cv2.rectangle(
                annotated,
                (x - half, y - half),
                (x + half, y + half),
                color,
                2,
            )

        first_x, first_y = list(options.values())[0]
        cv2.putText(
            annotated,
            f"Q{question}: {status}",
            (first_x - 95, first_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 220),
            1,
            cv2.LINE_AA,
        )

    return annotated
