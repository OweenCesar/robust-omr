# Preprocessing functions to be used mainly for the first stage of the project.
# These functions can also be imported in notebooks.

import numpy as np
from PIL import Image
import cv2


# Predefined dimensions for resizing the images.

width_Img = 700
height_Img = 700


def resize_image(image, size=(width_Img, height_Img)):
    """
    Resize the input image using PIL.

    Parameters:
    image (numpy array): The input image to be resized.
    size (tuple): Desired output size in the format (width, height).

    Returns:
    numpy array: The resized image.
    """
    img = Image.fromarray(image)
    img_resized = img.resize(size) # PIL resize expects size as (width, height).

    return np.array(img_resized)


def image_to_grayscale(resized_image):
    """
    Convert the input image to grayscale using PIL.
    It is supposed to be used after the image was already resized.

    Parameters:
    resized_image (numpy array): The resized input image.

    Returns:
    numpy array: The grayscale image.
    """
    img = Image.fromarray(resized_image)
    img_gray = img.convert("L")
    return np.array(img_gray)


def add_gaussian_blur(grayscale_image, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian blur to the input grayscale image using OpenCV.

    Parameters:
    grayscale_image (numpy array): The input grayscale image.
    kernel_size (tuple): Size of the Gaussian kernel.
    sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
    numpy array: The blurred image.
    """
    blurred_image = cv2.GaussianBlur(grayscale_image, kernel_size, sigma)
    return blurred_image


def apply_canny_edge_detection(blurred_image, low_threshold=10, high_threshold=70):
    """
    Apply Canny edge detection to the input blurred image.

    Parameters:
    blurred_image (numpy array): The input blurred image.
    low_threshold (int): Lower threshold for edge detection.
    high_threshold (int): Upper threshold for edge detection.

    Returns:
    numpy array: The image with detected edges.
    """
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges


# Debugging functions

def create_blank_image(size=(width_Img, height_Img)):
    """
    Create a blank image for debugging purposes.

    The size parameter follows image resizing convention: (width, height).
    NumPy creates images using shape: (height, width, channels).

    Parameters:
    size (tuple): Desired image size in the format (width, height).

    Returns:
    numpy array: A blank black image.
    """
    width, height = size
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)
    return blank_image