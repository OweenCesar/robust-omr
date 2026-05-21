import json


def _point_tuple(point):
    return (int(round(point[0])), int(round(point[1])))


def load_template(template_path: str) -> dict:
    """
    Loads the OMR template configuration from a JSON file.
    """
    with open(template_path, "r", encoding="utf-8") as file:
        template = json.load(file)

    required_fields = ["template_name", "warped_width", "warped_height", "options", "bubble_radius"]

    for field in required_fields:
        if field not in template:
            raise ValueError(f"Template is missing required field: {field}")

    if "bubble_centers" not in template and "bubble_blocks" not in template and "bubble_layout" not in template:
        raise ValueError(
            "Template must define one of: bubble_centers, bubble_blocks, bubble_layout."
        )

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
    if "bubble_centers" in template:
        return {
            str(question): {
                option: _point_tuple(center)
                for option, center in options.items()
            }
            for question, options in template["bubble_centers"].items()
        }

    if "bubble_blocks" in template:
        return generate_bubble_centers_from_blocks(template)

    return generate_bubble_centers_from_legacy_layout(template)


def generate_bubble_centers_from_blocks(template: dict) -> dict:
    """
    Generates bubble centers from one or more answer blocks/columns.

    Each block can either define:
    - start_question + questions, for sequential numbering
    - question_numbers, for custom numbering
    """
    template_options = template["options"]
    bubble_centers = {}

    for block in template["bubble_blocks"]:
        options = block.get("options", template_options)
        start_x = block["start_x"]
        start_y = block["start_y"]
        option_gap_x = block["option_gap_x"]
        question_gap_y = block["question_gap_y"]

        if "question_numbers" in block:
            question_numbers = block["question_numbers"]
        else:
            start_question = int(block["start_question"])
            question_count = int(block["questions"])
            question_numbers = range(start_question, start_question + question_count)

        for row_index, question in enumerate(question_numbers):
            question_key = str(question)

            if question_key in bubble_centers:
                raise ValueError(f"Duplicate question in template: {question_key}")

            y = start_y + row_index * question_gap_y
            bubble_centers[question_key] = {}

            for option_index, option in enumerate(options):
                x = start_x + option_index * option_gap_x
                bubble_centers[question_key][option] = _point_tuple((x, y))

    return dict(
        sorted(
            bubble_centers.items(),
            key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0])
        )
    )


def generate_bubble_centers_from_legacy_layout(template: dict) -> dict:
    """
    Supports the first single-column template format.
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
            bubble_centers[str(q)][option] = _point_tuple((x, y))

    return bubble_centers
