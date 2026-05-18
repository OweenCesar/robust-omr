import cv2
import numpy as np


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