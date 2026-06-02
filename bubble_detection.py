# Bubble and mark detection helpers for warped OMR sections.

import cv2
import numpy as np


def convert_to_grayscale(image):
    """
    Convert RGB/BGR or grayscale input to grayscale.
    """
    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def detect_bubble_candidates_hough(
    section_image,
    dp=1.2,
    min_distance=10,
    param1=50,
    param2=12,
    min_radius=3,
    max_radius=12
):
    """
    Detect circular bubble candidates in a warped section.

    This is a first classical baseline. It gives us candidate bubbles that can
    later be evaluated by either image-processing rules or an ML model.
    """
    grayscale_image = convert_to_grayscale(section_image)
    blurred_image = cv2.medianBlur(grayscale_image, 5)

    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_distance,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return []

    height, width = grayscale_image.shape[:2]
    candidates = []

    for x, y, radius in np.round(circles[0]).astype(int):
        if x - radius < 0 or y - radius < 0:
            continue

        if x + radius >= width or y + radius >= height:
            continue

        candidates.append({
            "x": int(x),
            "y": int(y),
            "radius": int(radius),
        })

    return candidates


def score_bubble_marks(
    section_image,
    bubbles,
    inner_radius_ratio=0.75,
    dark_threshold=130
):
    """
    Estimate how strongly each bubble is marked.

    The score is the ratio of dark pixels inside the inner part of the bubble.
    """
    grayscale_image = convert_to_grayscale(section_image)
    scored_bubbles = []

    for bubble in bubbles:
        x = bubble["x"]
        y = bubble["y"]
        radius = bubble["radius"]
        inner_radius = max(2, int(round(radius * inner_radius_ratio)))

        mask = np.zeros_like(grayscale_image, dtype=np.uint8)
        cv2.circle(mask, (x, y), inner_radius, 255, cv2.FILLED)

        pixels = grayscale_image[mask > 0]

        if len(pixels) == 0:
            dark_ratio = 0.0
            mean_intensity = 255.0
        else:
            dark_ratio = float(np.mean(pixels < dark_threshold))
            mean_intensity = float(np.mean(pixels))

        scored_bubble = bubble.copy()
        scored_bubble["dark_ratio"] = round(dark_ratio, 3)
        scored_bubble["mean_intensity"] = round(mean_intensity, 2)
        scored_bubbles.append(scored_bubble)

    return scored_bubbles


def filter_bubbles_by_color_presence(
    section_image,
    bubbles,
    saturation_threshold=35,
    value_threshold=90,
    min_color_ratio=0.08,
    radius_scale=1.4
):
    """
    Keep bubbles that contain enough colored pixels around the candidate.

    The Tamaulipas samples use orange option circles. This simple filter removes
    many false Hough detections on black row numbers while keeping the real
    answer bubbles. It is intentionally optional because other datasets may use
    black bubbles or square boxes instead of colored circles.
    """
    if section_image.ndim == 2:
        return bubbles

    hsv_image = cv2.cvtColor(section_image, cv2.COLOR_RGB2HSV)
    filtered_bubbles = []

    for bubble in bubbles:
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        radius = max(3, int(round(bubble["radius"] * radius_scale)))
        cv2.circle(mask, (bubble["x"], bubble["y"]), radius, 255, cv2.FILLED)

        pixels = hsv_image[mask > 0]

        if len(pixels) == 0:
            color_ratio = 0.0
        else:
            colored_pixels = (
                (pixels[:, 1] > saturation_threshold)
                & (pixels[:, 2] > value_threshold)
            )
            color_ratio = float(np.mean(colored_pixels))

        if color_ratio >= min_color_ratio:
            filtered_bubble = bubble.copy()
            filtered_bubble["color_ratio"] = round(color_ratio, 3)
            filtered_bubbles.append(filtered_bubble)

    return filtered_bubbles


def draw_bubbles(image, bubbles, color=(0, 255, 0), thickness=2):
    """
    Draw bubble candidate circles on a copy of the image.
    """
    image_copy = image.copy()

    for bubble in bubbles:
        center = (bubble["x"], bubble["y"])
        cv2.circle(image_copy, center, bubble["radius"], color, thickness)

    return image_copy


def draw_mark_scores(
    image,
    bubbles,
    mark_threshold=0.25,
    candidate_color=(0, 255, 0),
    marked_color=(255, 0, 0),
    thickness=2
):
    """
    Draw all candidates and highlight bubbles whose dark ratio is high enough.
    """
    image_copy = image.copy()

    for bubble in bubbles:
        dark_ratio = bubble.get("dark_ratio", 0.0)
        color = marked_color if dark_ratio >= mark_threshold else candidate_color
        center = (bubble["x"], bubble["y"])

        cv2.circle(image_copy, center, bubble["radius"], color, thickness)

    return image_copy


def threshold_dark_marks(section_image, dark_threshold=130):
    """
    Produce a binary image where dark marks become white.
    """
    grayscale_image = convert_to_grayscale(section_image)

    _, binary_image = cv2.threshold(
        grayscale_image,
        dark_threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    return binary_image


def sort_bubbles_by_position(bubbles, row_tolerance=12):
    """
    Sort bubbles from top to bottom and left to right.

    This prepares candidates for later grouping into answer rows.
    """
    sorted_bubbles = sorted(bubbles, key=lambda bubble: bubble["y"])
    rows = []

    for bubble in sorted_bubbles:
        if not rows:
            rows.append([bubble])
            continue

        row_center = np.mean([item["y"] for item in rows[-1]])

        if abs(bubble["y"] - row_center) <= row_tolerance:
            rows[-1].append(bubble)
        else:
            rows.append([bubble])

    ordered = []

    for row in rows:
        ordered.extend(sorted(row, key=lambda bubble: bubble["x"]))

    return ordered
