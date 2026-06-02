# Contour detection functions for the OMR project.
# These functions are used after preprocessing, when the Canny image is already available.

import cv2
import numpy as np


def prepare_rectangle_detection_image(grayscale_image, blur_kernel_size=3):
    """
    Prepare a binary image for detecting the main sheet rectangles.

    This follows the spirit of the paper's ID/answer section detection stage:
    smooth small noise, threshold the dark rectangle borders and close tiny gaps.

    Parameters:
    grayscale_image (numpy array): Grayscale image.
    blur_kernel_size (int): Median blur kernel size. It must be odd.

    Returns:
    numpy array: Binary image where dark lines become white regions.
    """
    blurred_image = cv2.medianBlur(grayscale_image, blur_kernel_size)

    _, binary_image = cv2.threshold(
        blurred_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return binary_image


def find_all_contours(canny_image):
    """
    Find all external contours from a Canny edge image.

    Parameters:
    canny_image (numpy array): The image after Canny edge detection.

    Returns:
    list: Detected contours.
    """
    contours, hierarchy = cv2.findContours(
        canny_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    return contours


def find_external_contours(binary_image, approximation=cv2.CHAIN_APPROX_SIMPLE):
    """
    Find external contours from a binary image.

    This is useful after Otsu/adaptive thresholding, while find_all_contours is
    kept for the Canny-based experiments already used in the notebook.
    """
    contours, hierarchy = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        approximation
    )

    return contours


def draw_all_contours(image, contours, color=(0, 255, 0), thickness=10):
    """
    Draw all detected contours on a copy of the image.
    This is mainly for debugging and visualization.

    Parameters:
    image (numpy array): Original or resized image where contours will be drawn.
    contours (list): List of detected contours.
    color (tuple): Color of the contours in BGR format.
    thickness (int): Thickness of the contour lines.

    Returns:
    numpy array: Image with all contours drawn.
    """
    image_copy = image.copy()
    cv2.drawContours(image_copy, contours, -1, color, thickness)

    return image_copy


def filter_rectangular_contours(contours, min_area=1000):
    """
    Filter contours and keep only rectangle-like contours.

    A contour is considered rectangular if:
    - its area is larger than min_area
    - its approximated polygon has 4 corner points

    Parameters:
    contours (list): List of contours.
    min_area (int): Minimum contour area to be considered.

    Returns:
    list: Rectangular contours sorted by area from biggest to smallest.
    """
    rectangular_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_area:
            perimeter = cv2.arcLength(contour, True)

            # Approximate the contour shape.
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Rectangles should have 4 corner points.
            if len(approx) == 4:
                rectangular_contours.append(contour)

    # Sort by area, biggest first.
    rectangular_contours = sorted(
        rectangular_contours,
        key=cv2.contourArea,
        reverse=True
    )

    return rectangular_contours


def select_largest_rectangular_contours(contours, count=2, min_area=1000):
    """
    Select the largest rectangle-like contours.

    For the current answer sheets, the two largest rectangles normally correspond
    to the answer section and the ID section.
    """
    rectangular_contours = filter_rectangular_contours(
        contours=contours,
        min_area=min_area
    )

    return rectangular_contours[:count]


def get_corner_points(contour):
    """
    Get the 4 corner points of a rectangular contour.

    Parameters:
    contour (numpy array): A rectangular contour.

    Returns:
    numpy array: 4 corner points of the contour.
    """
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    return approx


def reorder_corner_points(points):
    """
    Reorder 4 corner points into a consistent order:
    top-left, top-right, bottom-left, bottom-right.

    This is important before applying perspective transform.

    Parameters:
    points (numpy array): Corner points with shape usually (4, 1, 2).

    Returns:
    numpy array: Reordered points with shape (4, 2).
    """
    points = points.reshape((4, 2))

    reordered_points = np.zeros((4, 2), dtype=np.float32)

    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1)

    reordered_points[0] = points[np.argmin(point_sum)]   # top-left
    reordered_points[3] = points[np.argmax(point_sum)]   # bottom-right
    reordered_points[1] = points[np.argmin(point_diff)]  # top-right
    reordered_points[2] = points[np.argmax(point_diff)]  # bottom-left

    return reordered_points


def get_warp_size(points):
    """
    Estimate the width and height of a warped rectangle from ordered points.
    """
    points = points.reshape((4, 2)).astype(np.float32)
    top_left, top_right, bottom_left, bottom_right = points

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    width = int(round(max(width_top, width_bottom)))
    height = int(round(max(height_left, height_right)))

    return max(width, 1), max(height, 1)


def warp_rectangle_from_points(image, ordered_points, output_size=None):
    """
    Apply a perspective transform using ordered rectangle points.

    Parameters:
    image (numpy array): Image to warp.
    ordered_points (numpy array): Points ordered as top-left, top-right,
        bottom-left, bottom-right.
    output_size (tuple | None): Optional output size as (width, height).

    Returns:
    numpy array: Warped rectangular region.
    """
    ordered_points = ordered_points.reshape((4, 2)).astype(np.float32)

    if output_size is None:
        width, height = get_warp_size(ordered_points)
    else:
        width, height = output_size

    destination_points = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
        ],
        dtype=np.float32
    )

    transform = cv2.getPerspectiveTransform(
        ordered_points,
        destination_points
    )

    warped_image = cv2.warpPerspective(
        image,
        transform,
        (width, height)
    )

    return warped_image


def warp_rectangular_contour(image, contour, output_size=None):
    """
    Extract and warp one rectangle-like contour.

    Returns the warped image and the ordered corner points used for the transform.
    """
    corner_points = get_corner_points(contour)
    ordered_points = reorder_corner_points(corner_points)
    warped_image = warp_rectangle_from_points(
        image=image,
        ordered_points=ordered_points,
        output_size=output_size
    )

    return warped_image, ordered_points


def build_warped_rectangle_sections(image, rectangular_contours, count=2):
    """
    Warp the largest rectangle-like sections and keep useful metadata.
    """
    sections = []

    for index, contour in enumerate(rectangular_contours[:count], start=1):
        warped_image, ordered_points = warp_rectangular_contour(
            image=image,
            contour=contour
        )

        sections.append({
            "index": index,
            "name": f"rectangle_{index}",
            "area": float(cv2.contourArea(contour)),
            "contour": contour,
            "corner_points": ordered_points,
            "warped_image": warped_image,
        })

    return sections


def draw_corner_points(image, points, color=(0, 0, 255), radius=10):
    """
    Draw corner points on an image for debugging.

    Parameters:
    image (numpy array): Image where points will be drawn.
    points (numpy array): Corner points.
    color (tuple): Point color in BGR format.
    radius (int): Radius of each point.

    Returns:
    numpy array: Image with corner points drawn.
    """
    image_copy = image.copy()

    points = points.reshape((4, 2))

    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_copy, (x, y), radius, color, cv2.FILLED)

    return image_copy
