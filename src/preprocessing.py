import cv2
import numpy as np

from sheet_layout import SHEET_HEIGHT, SHEET_WIDTH
from sheet_layout import MARKER_IDS, marker_corner_points, marker_id_to_name


def load_image_from_bytes(image_bytes: bytes):
    """
    Decode an uploaded image into an OpenCV BGR image.

    Flask receives uploaded photos as bytes. OpenCV works with NumPy arrays, so
    this helper is the bridge between the web layer and the image-processing
    layer. It accepts JPEG, PNG, and other formats supported by OpenCV.
    """
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode the uploaded image.")

    return image


def convert_to_gray(image):
    """
    Converts BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def threshold_dark_marks(gray_image):
    """
    Convert dark ink into white pixels on a black background.

    OMR reading is easier when the image contains only two meanings: mark or no
    mark. Adaptive thresholding is used because classroom photos often have
    uneven light, shadows, or a brighter area near the phone flash.
    """
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )


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


def detect_corner_markers(image):
    """
    Find the four coded ArUco corner markers in a camera photo.

    The earlier prototype used four identical black squares and guessed their
    positions by image geometry. That breaks down when the sheet is photographed
    from a side angle. ArUco markers solve that because each printed corner has
    an ID and OpenCV returns the marker corners in a known orientation.
    """
    gray = convert_to_gray(image)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # These parameters make detection less fragile when a phone photo has a
    # steep perspective angle, light blur, or modest marker size. They still
    # reject random black text and filled answer boxes because the marker must
    # decode to one of our expected IDs.
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 8
    parameters.minMarkerPerimeterRate = 0.015
    parameters.maxMarkerPerimeterRate = 0.6
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(ids) == 0:
        return None, [
            "Could not find the coded corner markers. Make sure the full page is visible and retake the photo."
        ]

    expected_ids = marker_id_to_name()
    detected = {}

    for marker_corners, marker_id_array in zip(corners, ids):
        marker_id = int(marker_id_array[0])

        if marker_id not in expected_ids:
            continue

        name = expected_ids[marker_id]
        corner_points = marker_corners.reshape(4, 2).astype("float32")
        marker_area = cv2.contourArea(corner_points)

        # If the same marker is detected twice, keep the larger detection. That
        # usually corresponds to the clearer candidate.
        if name not in detected or marker_area > detected[name]["area"]:
            center = np.mean(corner_points, axis=0)
            detected[name] = {
                "id": marker_id,
                "center": (float(center[0]), float(center[1])),
                "corners": corner_points,
                "area": float(marker_area),
            }

    missing_names = [name for name in MARKER_IDS if name not in detected]

    if missing_names:
        readable = ", ".join(name.replace("_", " ") for name in missing_names)
        return None, [
            f"Could not find these coded corner markers: {readable}. Keep every marker inside the photo."
        ]

    return detected, []


def warp_to_sheet(image, detected_markers):
    """
    Correct the camera photo into the fixed SVG sheet coordinate system.

    The homography is built from all four corners of all four ArUco markers,
    giving up to 16 point correspondences. This is more stable than using only
    four marker centers, especially when the photo is taken from an angle.
    """
    destination_marker_corners = marker_corner_points()
    source_points = []
    destination_points = []

    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        source_points.extend(detected_markers[name]["corners"])
        destination_points.extend(destination_marker_corners[name])

    source_points = np.array(source_points, dtype="float32")
    destination_points = np.array(destination_points, dtype="float32")

    transform, inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=6.0,
    )

    if transform is None:
        raise ValueError("Could not calculate sheet perspective. Retake the photo with all four markers clear.")

    projected_points = cv2.perspectiveTransform(source_points.reshape(-1, 1, 2), transform)
    reprojection_errors = np.linalg.norm(
        projected_points.reshape(-1, 2) - destination_points,
        axis=1,
    )

    inlier_count = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
    quality = {
        "marker_count": len(detected_markers),
        "homography_inliers": inlier_count,
        "mean_reprojection_error": round(float(np.mean(reprojection_errors)), 2),
        "max_reprojection_error": round(float(np.max(reprojection_errors)), 2),
    }

    warped = cv2.warpPerspective(
        image,
        transform,
        (SHEET_WIDTH, SHEET_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return warped, quality
