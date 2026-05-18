import json


def load_template(template_path: str) -> dict:
    """
    Loads the OMR template configuration from a JSON file.
    """
    with open(template_path, "r", encoding="utf-8") as file:
        template = json.load(file)

    return template


def generate_bubble_centers(template: dict) -> dict:
    """
    Generates bubble center coordinates based on the template layout.
    Returns:
        {
            "1": {"A": (x, y), "B": (x, y), ...},
            "2": {"A": (x, y), "B": (x, y), ...}
        }
    """
    options = template["options"]
    questions = template["questions"]

    layout = template["bubble_layout"]

    start_x = layout["start_x"]
    start_y = layout["start_y"]
    option_gap_x = layout["option_gap_x"]
    question_gap_y = layout["question_gap_y"]

    bubble_centers = {}

    for q in range(1, questions + 1):
        y = start_y + (q - 1) * question_gap_y

        bubble_centers[str(q)] = {}

        for i, option in enumerate(options):
            x = start_x + i * option_gap_x
            bubble_centers[str(q)][option] = (x, y)

    return bubble_centers