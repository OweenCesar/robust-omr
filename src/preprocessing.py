import cv2
import numpy as np


CORNER_KEYS = ("top_left", "top_right", "bottom_left", "bottom_right")


def load_image(image_path: str):
    """
    Loads an image from disk.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def resize_for_display(image, max_width: int = 900):
    """
    Resizes image for easier visualization/debugging.
    """
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height))


def convert_to_gray(image):
    """
    Converts BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess_for_thresholding(image):
    """
    Converts image to grayscale, blurs it, and applies adaptive thresholding.
    This is more robust against shadows than simple global thresholding.
    """
    gray = convert_to_gray(image)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    return gray, thresholded


def _marker_candidate(contour, image_area):
    area = cv2.contourArea(contour)

    if area < image_area * 0.00015:
        return None

    x, y, width, height = cv2.boundingRect(contour)

    if width == 0 or height == 0:
        return None

    aspect_ratio = width / height
    fill_ratio = area / (width * height)

    if not 0.65 <= aspect_ratio <= 1.35:
        return None

    if fill_ratio < 0.55:
        return None

    return {
        "area": float(area),
        "bbox": (int(x), int(y), int(width), int(height)),
        "center": (float(x + width / 2), float(y + height / 2)),
        "fill_ratio": float(fill_ratio),
    }


def detect_corner_markers(image):
    """
    Finds the four filled square registration markers on a sheet image.
    Returns marker centers keyed by template corner names.
    """
    gray = convert_to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresholded = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = image.shape[0] * image.shape[1]
    candidates = []

    for contour in contours:
        candidate = _marker_candidate(contour, image_area)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        raise ValueError("Could not find any square registration markers.")

    largest_area = max(candidate["area"] for candidate in candidates)
    marker_candidates = [
        candidate for candidate in candidates
        if candidate["area"] >= largest_area * 0.45
    ]

    if len(marker_candidates) < 4:
        raise ValueError(
            "Could not find four registration markers. "
            "Make sure all black corner squares are visible."
        )

    centers = np.array(
        [candidate["center"] for candidate in marker_candidates],
        dtype=np.float32
    )

    ordered = {
        "top_left": tuple(centers[np.argmin(centers.sum(axis=1))]),
        "top_right": tuple(centers[np.argmax(centers[:, 0] - centers[:, 1])]),
        "bottom_left": tuple(centers[np.argmax(centers[:, 1] - centers[:, 0])]),
        "bottom_right": tuple(centers[np.argmax(centers.sum(axis=1))]),
    }

    if len({tuple(np.round(ordered[key], 1)) for key in CORNER_KEYS}) != 4:
        raise ValueError(
            "Registration marker detection was ambiguous. "
            "Try a clearer photo with all four corner squares visible."
        )

    return ordered


def align_to_template(image, template):
    """
    Warps a photographed/scanned sheet into the template coordinate space.
    """
    if "corner_markers" not in template:
        return resize_to_template(image, template), {
            "status": "skipped",
            "reason": "template has no corner_markers"
        }

    detected_markers = detect_corner_markers(image)
    width = int(template["warped_width"])
    height = int(template["warped_height"])

    source_points = np.array(
        [detected_markers[key] for key in CORNER_KEYS],
        dtype=np.float32
    )
    destination_points = np.array(
        [template["corner_markers"][key] for key in CORNER_KEYS],
        dtype=np.float32
    )

    transform = cv2.getPerspectiveTransform(source_points, destination_points)
    warped = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderValue=(255, 255, 255)
    )

    return warped, {
        "status": "aligned",
        "detected_markers": {
            key: [round(float(value), 2) for value in detected_markers[key]]
            for key in CORNER_KEYS
        }
    }


def resize_to_template(image, template):
    """
    Resizes an already-normalized image into template dimensions.
    Useful when the input is not a camera photo and alignment is skipped.
    """
    width = int(template["warped_width"])
    height = int(template["warped_height"])

    if image.shape[1] == width and image.shape[0] == height:
        return image.copy()

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def compute_blur_score(gray_image):
    """
    Computes a blur score using the variance of the Laplacian.
    Lower values usually mean the image is blurrier.
    """
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def compute_brightness(gray_image):
    """
    Computes the average brightness of the image.
    """
    return float(np.mean(gray_image))
