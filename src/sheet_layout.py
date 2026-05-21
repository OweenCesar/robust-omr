"""
Shared OMR sheet layout.

The web page that prints the answer sheet and the OpenCV code that reads the
phone photo must agree on the exact same coordinate system. This file is the
single source of truth for that geometry.

The printable sheet is rendered as an SVG with a 1000 x 1400 viewBox. When a
teacher scans the printed page, OpenCV finds the four black corner markers and
warps the photo back into this same 1000 x 1400 coordinate system. After that,
reading answers is just a matter of checking fixed box centers.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache

import cv2


SHEET_WIDTH = 1000
SHEET_HEIGHT = 1400

OPTIONS = ("A", "B", "C", "D")

# The production scanner uses four unique ArUco markers instead of identical
# black squares. ArUco markers encode an ID and orientation, so OpenCV can tell
# which printed corner is which even when the teacher takes the photo from a
# side angle or the phone rotates the image.
MARKER_SIZE = 88
MARKER_CENTERS = {
    "top_left": (74, 74),
    "top_right": (926, 74),
    "bottom_left": (74, 1326),
    "bottom_right": (926, 1326),
}
MARKER_IDS = {
    "top_left": 10,
    "top_right": 11,
    "bottom_right": 12,
    "bottom_left": 13,
}

# The answer boxes are square because the user's LaTeX design uses square
# boxes. The reader samples the inner part of each square so the printed border
# itself is not mistaken for a student mark.
ANSWER_BOX_SIZE = 26

# The first column can hold questions 1-20. The second column is used only when
# a test has more than 20 questions.
MAX_ROWS_PER_COLUMN = 20
FIRST_QUESTION_Y = 390
QUESTION_ROW_GAP = 42

LEFT_COLUMN_LABEL_X = 118
LEFT_COLUMN_OPTION_XS = (190, 245, 300, 355)

RIGHT_COLUMN_LABEL_X = 558
RIGHT_COLUMN_OPTION_XS = (630, 685, 740, 795)


@dataclass(frozen=True)
class SheetColumn:
    """Describes one visual question column on the printed sheet."""

    label_x: int
    option_xs: tuple[int, ...]
    start_question: int
    end_question: int


def validate_question_count(question_count: int) -> None:
    """Raise a helpful error if a requested test size is outside the supported range."""

    if question_count < 10 or question_count > 40:
        raise ValueError("Question count must be between 10 and 40.")


def get_columns(question_count: int) -> list[SheetColumn]:
    """
    Return the columns needed for a test.

    Tests with 10-20 questions use one column. Tests with 21-40 questions use
    two columns, with questions 1-20 on the left and the remaining questions on
    the right. This keeps spacing large enough for low-end phone cameras.
    """

    validate_question_count(question_count)

    columns = [
        SheetColumn(
            label_x=LEFT_COLUMN_LABEL_X,
            option_xs=LEFT_COLUMN_OPTION_XS,
            start_question=1,
            end_question=min(question_count, MAX_ROWS_PER_COLUMN),
        )
    ]

    if question_count > MAX_ROWS_PER_COLUMN:
        columns.append(
            SheetColumn(
                label_x=RIGHT_COLUMN_LABEL_X,
                option_xs=RIGHT_COLUMN_OPTION_XS,
                start_question=MAX_ROWS_PER_COLUMN + 1,
                end_question=question_count,
            )
        )

    return columns


def question_y(question_number: int) -> int:
    """Return the y-coordinate for a question inside its column."""

    row_index = (question_number - 1) % MAX_ROWS_PER_COLUMN
    return FIRST_QUESTION_Y + row_index * QUESTION_ROW_GAP


def generate_answer_box_centers(
    question_count: int,
    options: tuple[str, ...] = OPTIONS,
) -> dict[str, dict[str, tuple[int, int]]]:
    """
    Build the fixed answer-box centers used by the detector.

    The return shape matches the existing project style:

    {
        "1": {"A": (190, 390), "B": (245, 390), ...},
        "2": {"A": (190, 432), "B": (245, 432), ...}
    }
    """

    validate_question_count(question_count)

    centers: dict[str, dict[str, tuple[int, int]]] = {}

    for column in get_columns(question_count):
        for question_number in range(column.start_question, column.end_question + 1):
            y = question_y(question_number)
            centers[str(question_number)] = {}

            for option, x in zip(options, column.option_xs):
                centers[str(question_number)][option] = (x, y)

    return centers


def marker_rects() -> list[dict[str, int | str]]:
    """
    Return marker rectangles for SVG rendering.

    The detector uses marker centers, while the SVG renderer needs top-left
    positions and sizes. Keeping this helper here prevents small coordinate
    differences between print and scan.
    """

    half = MARKER_SIZE // 2

    return [
        {
            "name": name,
            "id": MARKER_IDS[name],
            "x": center[0] - half,
            "y": center[1] - half,
            "size": MARKER_SIZE,
            "data_url": aruco_marker_data_url(MARKER_IDS[name]),
        }
        for name, center in MARKER_CENTERS.items()
    ]


def marker_corner_points() -> dict[str, tuple[tuple[int, int], ...]]:
    """
    Return destination corner points for each printed ArUco marker.

    OpenCV returns the detected corners of each ArUco marker in marker-local
    order: top-left, top-right, bottom-right, bottom-left. Matching those to
    these exact printed destination corners gives a stronger homography than
    using marker centers alone.
    """

    half = MARKER_SIZE // 2
    points = {}

    for name, center in MARKER_CENTERS.items():
        x, y = center
        points[name] = (
            (x - half, y - half),
            (x + half, y - half),
            (x + half, y + half),
            (x - half, y + half),
        )

    return points


def marker_id_to_name() -> dict[int, str]:
    """Return the reverse lookup used by the ArUco detector."""

    return {marker_id: name for name, marker_id in MARKER_IDS.items()}


@lru_cache(maxsize=None)
def aruco_marker_data_url(marker_id: int) -> str:
    """
    Generate a printable PNG data URL for one ArUco marker.

    The marker is embedded directly into the SVG sheet so there are no external
    image files to lose when the app is moved. The browser scales this crisp
    black-and-white bitmap to the marker rectangle.
    """

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, 256)
    success, encoded = cv2.imencode(".png", marker_image)

    if not success:
        raise RuntimeError(f"Could not generate ArUco marker {marker_id}.")

    marker_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{marker_base64}"


def sheet_context(question_count: int, options: tuple[str, ...] = OPTIONS) -> dict:
    """
    Create a template-friendly representation of the sheet.

    Jinja templates are easier to read when they receive simple dictionaries
    rather than doing geometry calculations inside the HTML file.
    """

    centers = generate_answer_box_centers(question_count, options)
    columns = []

    for column in get_columns(question_count):
        rows = []

        for question_number in range(column.start_question, column.end_question + 1):
            rows.append(
                {
                    "number": question_number,
                    "y": question_y(question_number),
                    "label_x": column.label_x,
                    "options": [
                        {
                            "label": option,
                            "x": centers[str(question_number)][option][0],
                        }
                        for option in options
                    ],
                }
            )

        columns.append(
            {
                "label_x": column.label_x,
                "option_xs": column.option_xs,
                "header_y": FIRST_QUESTION_Y - 42,
                "rows": rows,
            }
        )

    return {
        "width": SHEET_WIDTH,
        "height": SHEET_HEIGHT,
        "markers": marker_rects(),
        "marker_centers": MARKER_CENTERS,
        "marker_ids": MARKER_IDS,
        "box_size": ANSWER_BOX_SIZE,
        "options": options,
        "columns": columns,
    }
