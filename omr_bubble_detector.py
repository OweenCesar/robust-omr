"""
OpenCV OMR bubble detector for phone photos.

Goal of this module:
- Detect the printed sheet/frame from a camera image.
- Correct perspective using ArUco markers when present, with the older
  printed-frame/page-contour logic kept as fallback.
- Detect ONLY bubbles inside the answer area, Student ID, and Test ID areas.
- Return bubble coordinates and simple fill scores. This is NOT grading yet.

Designed for answer sheets with:
- Answers: A, B, C, D, E bubbles per question.
- Student ID: 8 digit columns, 10 rows for digits 0..9.
- Test ID: 4 digit columns, 10 rows for digits 0..9.

After perspective correction, the detector finds the printed header and answer
rectangles from the sheet line work. Fixed normalized ROIs are only used as a
fallback when the photographed lines are too broken to recover.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

OPTIONS = ["A", "B", "C", "D", "E"]
WARP_W = 1000
WARP_H = 1414
FRAME_MARGIN = 70

# New sheet templates use four OpenCV ArUco markers from DICT_4X4_50.
# Their centers sit on the printed-frame corners, so mapping marker centers to
# FRAME_MARGIN produces the same warped coordinate system as the old frame path.
ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_FRAME_MARKERS = {
    0: "top_left",
    1: "top_right",
    2: "bottom_right",
    3: "bottom_left",
}
ARUCO_CORNER_ORDER = [0, 1, 2, 3]

# Fallback normalized ROIs on the warped page: (x1, y1, x2, y2).
# The primary path below detects these regions from the printed rectangles.
DEFAULT_ROIS = {
    "student_id": (0.075, 0.165, 0.355, 0.315),
    "test_id": (0.335, 0.165, 0.525, 0.315),
    "answers": (0.075, 0.300, 0.950, 0.900),
}

ROI = Tuple[int, int, int, int]


@dataclass
class CircleCandidate:
    x: float
    y: float
    r: float
    score: float = 0.0


# -----------------------------------------------------------------------------
# Basic geometry helpers
# -----------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _line_y_from_points(points: Sequence[Tuple[float, float]]) -> Optional[Tuple[str, float, float]]:
    """Fit y = m*x + b."""
    if len(points) < 2:
        return None
    pts = np.asarray(points, dtype=np.float32)
    m, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
    return "h", float(m), float(b)


def _line_x_from_points(points: Sequence[Tuple[float, float]]) -> Optional[Tuple[str, float, float]]:
    """Fit x = m*y + b. This is stable for vertical-ish lines."""
    if len(points) < 2:
        return None
    pts = np.asarray(points, dtype=np.float32)
    m, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    return "v", float(m), float(b)


def _intersect_hv(
    h_line: Tuple[str, float, float], v_line: Tuple[str, float, float]
) -> np.ndarray:
    """Intersection of y=m*x+b and x=m*y+b."""
    _, mh, bh = h_line
    _, mv, bv = v_line
    denom = 1.0 - mv * mh
    if abs(denom) < 1e-8:
        raise ValueError("Nearly parallel fitted frame lines")
    x = (mv * bh + bv) / denom
    y = mh * x + bh
    return np.array([x, y], dtype=np.float32)


def _clip_point_to_image(pt: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.array([np.clip(pt[0], 0, width - 1), np.clip(pt[1], 0, height - 1)], dtype=np.float32)


def _roi_pixels(shape: Tuple[int, int], roi_norm: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = roi_norm
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def _clip_roi(roi: Sequence[float], shape: Tuple[int, int], min_size: int = 8) -> ROI:
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in roi]
    x1 = int(np.clip(x1, 0, w - min_size))
    y1 = int(np.clip(y1, 0, h - min_size))
    x2 = int(np.clip(x2, x1 + min_size, w))
    y2 = int(np.clip(y2, y1 + min_size, h))
    return x1, y1, x2, y2


def _pad_roi(roi: ROI, shape: Tuple[int, int], pad_x: int = 0, pad_y: int = 0) -> ROI:
    x1, y1, x2, y2 = roi
    return _clip_roi((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), shape)


def _resolve_roi(
    shape: Tuple[int, int],
    roi: Optional[ROI],
    roi_norm: Tuple[float, float, float, float],
) -> ROI:
    if roi is not None:
        return _clip_roi(roi, shape)
    return _clip_roi(_roi_pixels(shape, roi_norm), shape)


# -----------------------------------------------------------------------------
# Perspective correction
# -----------------------------------------------------------------------------

def _get_aruco_dictionary() -> Optional[Any]:
    """Return the ArUco dictionary used by the generated LaTeX templates."""
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, ARUCO_DICT_NAME):
        return None

    dictionary_id = getattr(aruco, ARUCO_DICT_NAME)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(aruco, "Dictionary_get"):
        return aruco.Dictionary_get(dictionary_id)
    return None


def _detect_aruco_frame_corners(image_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Detect the four ArUco markers and return frame-corner points.

    The LaTeX template places marker centers exactly at the intended printed
    frame corners. Using centers instead of marker outer corners makes the warp
    independent of each marker's internal rotation/corner ordering.
    """
    dictionary = _get_aruco_dictionary()
    aruco = getattr(cv2, "aruco", None)
    if dictionary is None or aruco is None:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    elif hasattr(aruco, "DetectorParameters_create"):
        parameters = aruco.DetectorParameters_create()
    else:
        parameters = None

    if parameters is not None and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    elif hasattr(aruco, "detectMarkers"):
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    else:
        return None

    if ids is None or len(ids) == 0:
        return None

    found: Dict[int, Dict[str, Any]] = {}
    for marker_corners, marker_id_value in zip(corners, ids.flatten()):
        marker_id = int(marker_id_value)
        if marker_id not in ARUCO_FRAME_MARKERS:
            continue

        pts = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        perimeter = float(sum(np.linalg.norm(pts[(idx + 1) % 4] - pts[idx]) for idx in range(4)))
        center = pts.mean(axis=0)

        # If a duplicate ID is detected, keep the larger instance. It is usually
        # the real printed marker rather than a tiny false positive.
        if marker_id not in found or perimeter > float(found[marker_id]["perimeter"]):
            found[marker_id] = {
                "center": center,
                "corners": pts,
                "perimeter": perimeter,
            }

    if any(marker_id not in found for marker_id in ARUCO_CORNER_ORDER):
        return None

    frame_corners = np.array([found[marker_id]["center"] for marker_id in ARUCO_CORNER_ORDER], dtype=np.float32)
    marker_centers = {
        ARUCO_FRAME_MARKERS[marker_id]: found[marker_id]["center"].astype(float).round(2).tolist()
        for marker_id in ARUCO_CORNER_ORDER
    }
    marker_corners = {
        ARUCO_FRAME_MARKERS[marker_id]: found[marker_id]["corners"].astype(float).round(2).tolist()
        for marker_id in ARUCO_CORNER_ORDER
    }
    meta = {
        "orientation_source": "aruco_marker_ids",
        "aruco_dictionary": ARUCO_DICT_NAME,
        "aruco_marker_ids": {
            ARUCO_FRAME_MARKERS[marker_id]: marker_id
            for marker_id in ARUCO_CORNER_ORDER
        },
        "aruco_marker_centers_original": marker_centers,
        "aruco_marker_corners_original": marker_corners,
    }
    return frame_corners, meta


def _is_plausible_aruco_frame(corners: np.ndarray, image_shape: Tuple[int, int]) -> bool:
    """Reject impossible ArUco quadrilaterals before applying the warp."""
    h, w = image_shape[:2]
    tl, tr, br, bl = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    top_w = float(np.linalg.norm(tr - tl))
    bottom_w = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))
    avg_w = (top_w + bottom_w) / 2.0
    avg_h = (left_h + right_h) / 2.0
    area = float(cv2.contourArea(np.asarray([tl, tr, br, bl], dtype=np.float32).reshape(-1, 1, 2)))

    # Use the ID-defined edge order here. A sideways sheet is still a valid
    # portrait sheet semantically, even though its geometric bounding box is wide.
    if area < 0.04 * h * w:
        return False
    if min(top_w, bottom_w) / (max(top_w, bottom_w) + 1e-6) < 0.42:
        return False
    if min(left_h, right_h) / (max(left_h, right_h) + 1e-6) < 0.42:
        return False
    return 0.90 <= avg_h / (avg_w + 1e-6) <= 2.20


def _warp_via_aruco_markers(image_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Warp using the ArUco marker IDs when the new templates are present.

    IDs 0, 1, 2, and 3 define top-left, top-right, bottom-right, and
    bottom-left. That gives both perspective correction and true sheet
    orientation before the rest of the OMR pipeline runs.
    """
    detected = _detect_aruco_frame_corners(image_bgr)
    if detected is None:
        return None

    frame_corners, aruco_meta = detected
    if not _is_plausible_aruco_frame(frame_corners, image_bgr.shape[:2]):
        return None

    warped, meta = _warp_from_corners(
        image_bgr,
        frame_corners,
        method="aruco_markers",
        map_printed_frame_to_margin=True,
    )
    meta.update(aruco_meta)
    return warped, meta


def _find_page_by_contour(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Fallback: detect the whole white paper as a quadrilateral."""
    h0, w0 = image_bgr.shape[:2]
    scale = 900.0 / max(h0, w0)
    small = cv2.resize(image_bgr, (int(w0 * scale), int(h0 * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = small.shape[:2]
    min_area = 0.18 * h * w
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if best is None or area > best[0]:
            best = (area, approx, c)

    if best is None:
        return None

    _, approx, contour = best
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32) / scale
    else:
        pts = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32) / scale
    return _order_points(pts)


def _cluster_segments_by_mid(segments: List[Dict[str, Any]], gap: float) -> List[List[Dict[str, Any]]]:
    segments = sorted(segments, key=lambda s: s["mid"])
    groups: List[List[Dict[str, Any]]] = []
    for seg in segments:
        if not groups:
            groups.append([seg])
            continue
        group_mid = float(np.mean([s["mid"] for s in groups[-1]]))
        if abs(seg["mid"] - group_mid) > gap:
            groups.append([seg])
        else:
            groups[-1].append(seg)
    return groups


def _points_from_long_segments(group: List[Dict[str, Any]], max_segments: int = 8) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for seg in sorted(group, key=lambda s: s["length"], reverse=True)[:max_segments]:
        points.extend(seg["points"])
    return points


def _find_printed_frame_corners(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the printed frame/corner guide using long black-ish line segments.

    This is preferred over pure paper contour detection, because phone photos can show
    a lot of background while the printed frame remains a stable OMR reference.
    """
    h0, w0 = image_bgr.shape[:2]
    scale = 1000.0 / max(h0, w0)
    small = cv2.resize(image_bgr, (int(w0 * scale), int(h0 * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = small.shape[:2]
    min_line = int(0.25 * min(h, w))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=min_line,
        maxLineGap=30,
    )
    if lines is None:
        return None

    horizontal: List[Dict[str, Any]] = []
    vertical: List[Dict[str, Any]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_line:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        item = {
            "points": [(x1, y1), (x2, y2)],
            "length": length,
            "angle": angle,
        }
        if abs(angle) < 12 or abs(abs(angle) - 180) < 12:
            item["mid"] = (y1 + y2) / 2.0
            horizontal.append(item)
        elif abs(abs(angle) - 90) < 12:
            item["mid"] = (x1 + x2) / 2.0
            vertical.append(item)

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    h_groups = _cluster_segments_by_mid(horizontal, gap=30)
    v_groups = _cluster_segments_by_mid(vertical, gap=30)

    # Keep only substantial side groups. This usually removes text/table fragments.
    h_groups = [g for g in h_groups if sum(s["length"] for s in g) > 0.45 * w]
    v_groups = [g for g in v_groups if sum(s["length"] for s in g) > 0.45 * h]
    if len(h_groups) < 2 or len(v_groups) < 2:
        return None

    top_group = h_groups[0]
    bottom_group = h_groups[-1]
    left_group = v_groups[0]
    right_group = v_groups[-1]

    top = _line_y_from_points(_points_from_long_segments(top_group))
    bottom = _line_y_from_points(_points_from_long_segments(bottom_group))
    left = _line_x_from_points(_points_from_long_segments(left_group))
    right = _line_x_from_points(_points_from_long_segments(right_group))
    if not all([top, bottom, left, right]):
        return None

    try:
        corners_small = np.array(
            [
                _intersect_hv(top, left),
                _intersect_hv(top, right),
                _intersect_hv(bottom, right),
                _intersect_hv(bottom, left),
            ],
            dtype=np.float32,
        )
    except ValueError:
        return None

    # Reject strange quadrilaterals.
    area = cv2.contourArea(corners_small.reshape(-1, 1, 2))
    if area < 0.20 * h * w:
        return None

    # Small safety clipping before rescaling.
    corners_small = np.array([_clip_point_to_image(p, w, h) for p in corners_small], dtype=np.float32)
    return _order_points(corners_small / scale)


def _quad_metrics(corners: np.ndarray) -> Dict[str, float]:
    corners = _order_points(corners)
    tl, tr, br, bl = corners
    top_w = float(np.linalg.norm(tr - tl))
    bottom_w = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))
    avg_w = (top_w + bottom_w) / 2.0
    avg_h = (left_h + right_h) / 2.0
    return {
        "top_width": top_w,
        "bottom_width": bottom_w,
        "left_height": left_h,
        "right_height": right_h,
        "width_ratio": min(top_w, bottom_w) / (max(top_w, bottom_w) + 1e-6),
        "height_ratio": min(left_h, right_h) / (max(left_h, right_h) + 1e-6),
        "aspect": avg_h / (avg_w + 1e-6),
        "area": float(cv2.contourArea(corners.reshape(-1, 1, 2))),
    }


def _is_plausible_printed_frame(
    corners: np.ndarray,
    image_shape: Tuple[int, int],
    min_width_ratio: float = 0.66,
) -> bool:
    h, w = image_shape[:2]
    metrics = _quad_metrics(corners)
    if metrics["area"] < 0.18 * h * w:
        return False
    if metrics["width_ratio"] < min_width_ratio or metrics["height_ratio"] < 0.62:
        return False
    return 0.95 <= metrics["aspect"] <= 2.05


def _warp_from_corners(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    method: str,
    map_printed_frame_to_margin: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if map_printed_frame_to_margin:
        dst = np.float32(
            [
                [FRAME_MARGIN, FRAME_MARGIN],
                [WARP_W - FRAME_MARGIN, FRAME_MARGIN],
                [WARP_W - FRAME_MARGIN, WARP_H - FRAME_MARGIN],
                [FRAME_MARGIN, WARP_H - FRAME_MARGIN],
            ]
        )
    else:
        dst = np.float32([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]])

    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(image_bgr, matrix, (WARP_W, WARP_H))
    meta = {
        "warp_method": method,
        "frame_corners_original": corners.astype(float).round(2).tolist(),
        "warp_size": [WARP_W, WARP_H],
    }
    return warped, meta


def _warp_via_paper_then_frame(image_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    paper_corners = _find_page_by_contour(image_bgr)
    if paper_corners is None:
        return None

    full_dst = np.float32([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]])
    paper_matrix = cv2.getPerspectiveTransform(paper_corners.astype(np.float32), full_dst)
    paper_warp = cv2.warpPerspective(image_bgr, paper_matrix, (WARP_W, WARP_H))

    frame_on_paper = _find_printed_frame_corners(paper_warp)
    if frame_on_paper is None or not _is_plausible_printed_frame(
        frame_on_paper,
        paper_warp.shape[:2],
        min_width_ratio=0.54,
    ):
        warped, meta = _warp_from_corners(
            image_bgr,
            paper_corners,
            method="paper_contour",
            map_printed_frame_to_margin=False,
        )
        meta["paper_corners_original"] = paper_corners.astype(float).round(2).tolist()
        return warped, meta

    margin_dst = np.float32(
        [
            [FRAME_MARGIN, FRAME_MARGIN],
            [WARP_W - FRAME_MARGIN, FRAME_MARGIN],
            [WARP_W - FRAME_MARGIN, WARP_H - FRAME_MARGIN],
            [FRAME_MARGIN, WARP_H - FRAME_MARGIN],
        ]
    )
    refine_matrix = cv2.getPerspectiveTransform(frame_on_paper.astype(np.float32), margin_dst)
    refined = cv2.warpPerspective(paper_warp, refine_matrix, (WARP_W, WARP_H))

    inv_paper_matrix = np.linalg.inv(paper_matrix)
    original_frame = cv2.perspectiveTransform(
        frame_on_paper.reshape(1, -1, 2).astype(np.float32),
        inv_paper_matrix,
    ).reshape(-1, 2)

    meta = {
        "warp_method": "paper_contour_refined_frame",
        "frame_corners_original": original_frame.astype(float).round(2).tolist(),
        "paper_corners_original": paper_corners.astype(float).round(2).tolist(),
        "refined_frame_corners_paper_warp": frame_on_paper.astype(float).round(2).tolist(),
        "warp_size": [WARP_W, WARP_H],
    }
    return refined, meta


def warp_sheet(image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Warp the camera photo to a stable A4-like canvas.

    Returns:
        warped_bgr, metadata
    """
    aruco_recovered = _warp_via_aruco_markers(image_bgr)
    if aruco_recovered is not None:
        return aruco_recovered

    frame_corners = _find_printed_frame_corners(image_bgr)
    if frame_corners is not None and _is_plausible_printed_frame(frame_corners, image_bgr.shape[:2]):
        return _warp_from_corners(
            image_bgr,
            frame_corners,
            method="printed_frame",
            map_printed_frame_to_margin=True,
        )

    recovered = _warp_via_paper_then_frame(image_bgr)
    if recovered is not None:
        return recovered

    if frame_corners is None:
        raise RuntimeError("Could not detect sheet/page corners. Try a clearer photo with all corner marks visible.")

    # Last resort: a suspicious printed frame is still usually better than no
    # output, and downstream metadata will expose that the layout had to fall back.
    return _warp_from_corners(
        image_bgr,
        frame_corners,
        method="printed_frame_low_confidence",
        map_printed_frame_to_margin=True,
    )


# -----------------------------------------------------------------------------
# Circle detection and grid inference
# -----------------------------------------------------------------------------

def _normalize_lighting(gray: np.ndarray, sigma: int = 25) -> np.ndarray:
    """Reduce shadows/brightness gradients from phone camera images."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.divide(gray, bg, scale=255)


def _cluster_centers(values: Iterable[float], gap: float) -> List[float]:
    values = sorted(float(v) for v in values)
    groups: List[List[float]] = []
    for value in values:
        if not groups or value - float(np.mean(groups[-1])) > gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.mean(group)) for group in groups]


def _merge_parallel_lines(lines: List[Dict[str, Any]], gap: float = 8.0) -> List[Dict[str, Any]]:
    """Merge pieces of the same printed line component."""
    if not lines:
        return []

    groups: List[List[Dict[str, Any]]] = []
    for line in sorted(lines, key=lambda item: item["center"]):
        if not groups:
            groups.append([line])
            continue
        group_center = float(np.mean([item["center"] for item in groups[-1]]))
        if abs(float(line["center"]) - group_center) <= gap:
            groups[-1].append(line)
        else:
            groups.append([line])

    merged: List[Dict[str, Any]] = []
    for group in groups:
        span_total = sum(float(item["span"]) for item in group) + 1e-6
        center = sum(float(item["center"]) * float(item["span"]) for item in group) / span_total
        x1 = min(int(item["x1"]) for item in group)
        y1 = min(int(item["y1"]) for item in group)
        x2 = max(int(item["x2"]) for item in group)
        y2 = max(int(item["y2"]) for item in group)
        orientation = group[0]["orientation"]
        merged.append(
            {
                "orientation": orientation,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center": float(center),
                "span": float((x2 - x1) if orientation == "h" else (y2 - y1)),
                "area": int(sum(int(item.get("area", 0)) for item in group)),
                "pieces": len(group),
            }
        )
    return merged


def _component_lines(mask: np.ndarray, orientation: str) -> List[Dict[str, Any]]:
    """Convert a horizontal/vertical line mask into coarse printed-line segments."""
    h, w = mask.shape[:2]
    labels_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    lines: List[Dict[str, Any]] = []

    for idx in range(1, labels_count):
        x, y, ww, hh, area = [int(v) for v in stats[idx]]
        cx, cy = centroids[idx]
        if orientation == "h":
            if ww < 0.25 * w or hh > max(30, h // 38) or area < ww * 0.25:
                continue
            lines.append(
                {
                    "orientation": "h",
                    "x1": x,
                    "y1": y,
                    "x2": x + ww,
                    "y2": y + hh,
                    "center": float(cy),
                    "span": float(ww),
                    "area": area,
                }
            )
        else:
            if hh < 0.08 * h or ww > max(30, w // 32) or area < hh * 0.20:
                continue
            lines.append(
                {
                    "orientation": "v",
                    "x1": x,
                    "y1": y,
                    "x2": x + ww,
                    "y2": y + hh,
                    "center": float(cx),
                    "span": float(hh),
                    "area": area,
                }
            )

    return _merge_parallel_lines(lines, gap=8.0)


def _extract_printed_lines(warped_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract the printed table/frame lines after perspective correction.

    Text and bubbles are deliberately suppressed by directional morphology. The
    remaining components are good enough to recover the big sheet sections even
    when the phone photo leaves the warp slightly imperfect.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    norm = _normalize_lighting(gray, sigma=35)
    th = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(45, w // 22), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, h // 35)))
    horizontal = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vertical = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)

    horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)), iterations=1)
    vertical = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)), iterations=1)

    return _component_lines(horizontal, "h"), _component_lines(vertical, "v")


def _nearest_horizontal_line(
    h_lines: Sequence[Dict[str, Any]],
    y_hint: float,
    max_distance: float,
    min_span: float = 0.0,
) -> Optional[Dict[str, Any]]:
    candidates = [
        line for line in h_lines
        if float(line["span"]) >= min_span and abs(float(line["center"]) - y_hint) <= max_distance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda line: abs(float(line["center"]) - y_hint))


def _fallback_layout(shape: Tuple[int, int]) -> Dict[str, Any]:
    student = _roi_pixels(shape, DEFAULT_ROIS["student_id"])
    test = _roi_pixels(shape, DEFAULT_ROIS["test_id"])
    answers = _roi_pixels(shape, DEFAULT_ROIS["answers"])
    header = (
        min(student[0], test[0]),
        min(student[1], test[1]),
        int(0.925 * shape[1]),
        max(student[3], test[3]),
    )
    return {
        "method": "fallback_normalized_roi",
        "header": list(_clip_roi(header, shape)),
        "student_id": list(_clip_roi(student, shape)),
        "test_id": list(_clip_roi(test, shape)),
        "answers": list(_clip_roi(answers, shape)),
        "source": {
            "header": "fallback_normalized_roi",
            "student_id": "fallback_normalized_roi",
            "test_id": "fallback_normalized_roi",
            "answers": "fallback_normalized_roi",
        },
        "line_counts": {"horizontal": 0, "vertical": 0},
    }


def _detect_answer_region_from_lines(
    h_lines: Sequence[Dict[str, Any]],
    v_lines: Sequence[Dict[str, Any]],
    shape: Tuple[int, int],
) -> Optional[ROI]:
    h, w = shape[:2]

    wide_lines = sorted(
        [line for line in h_lines if float(line["span"]) >= 0.45 * w],
        key=lambda line: float(line["center"]),
    )

    answer_top_line: Optional[Dict[str, Any]] = None
    for upper, lower in zip(wide_lines, wide_lines[1:]):
        gap = float(lower["center"]) - float(upper["center"])
        lower_y = float(lower["center"])
        if 10 <= gap <= 36 and 0.20 * h <= lower_y <= 0.58 * h:
            answer_top_line = lower

    if answer_top_line is None:
        return None

    y1 = float(answer_top_line["center"])
    x1 = float(answer_top_line["x1"])
    x2 = float(answer_top_line["x2"])

    bottom_candidates = [
        line for line in h_lines
        if float(line["center"]) > y1 + 0.30 * h
        and float(line["center"]) > 0.65 * h
        and float(line["span"]) >= 0.20 * w
    ]
    if bottom_candidates:
        answer_bottom_line = min(bottom_candidates, key=lambda line: float(line["center"]))
        y2 = float(answer_bottom_line["center"])
        # Use the top border for x when the lower border is partially shadowed.
        if float(answer_bottom_line["span"]) >= 0.60 * w:
            x1 = float(np.median([x1, float(answer_bottom_line["x1"])]))
            x2 = float(np.median([x2, float(answer_bottom_line["x2"])]))
    else:
        lower_verticals = [
            line for line in v_lines
            if float(line["span"]) >= 0.32 * h
            and float(line["y2"]) >= 0.68 * h
            and x1 - 35 <= float(line["center"]) <= x2 + 35
        ]
        y2 = float(np.median([float(line["y2"]) for line in lower_verticals])) if lower_verticals else 0.90 * h

    roi = _clip_roi((x1, y1, x2, y2), shape)
    if roi[2] - roi[0] < 0.50 * w or roi[3] - roi[1] < 0.25 * h:
        return None
    return roi


def _detect_header_region_from_lines(
    h_lines: Sequence[Dict[str, Any]],
    v_lines: Sequence[Dict[str, Any]],
    answer_roi: ROI,
    shape: Tuple[int, int],
) -> Optional[ROI]:
    h, w = shape[:2]
    ax1, ay1, ax2, _ = answer_roi
    answer_width = ax2 - ax1

    bottom_candidates = [
        line for line in h_lines
        if float(line["center"]) < ay1 - 6
        and ay1 - float(line["center"]) <= 85
        and float(line["span"]) >= 0.35 * w
    ]
    bottom_line = max(bottom_candidates, key=lambda line: float(line["center"])) if bottom_candidates else None
    if bottom_line is None:
        cropped_separators = [
            line for line in v_lines
            if ax1 - 45 <= float(line["center"]) <= ax2 + 45
            and float(line["y1"]) <= 0.12 * h
            and 0.07 * h <= float(line["span"]) <= 0.28 * h
            and float(line["y2"]) <= 0.36 * h
        ]
        if len(cropped_separators) >= 2:
            header_top = max(0.0, float(np.median([float(line["y1"]) for line in cropped_separators])))
            header_bottom = min(float(ay1 - 10), max(float(line["y2"]) for line in cropped_separators) + 6.0)
            header_edges = [
                float(line["center"]) for line in v_lines
                if ax1 - 80 <= float(line["center"]) <= ax2 + 80
                and float(line["y1"]) <= header_bottom
                and float(line["y2"]) >= header_top + 0.45 * max(40.0, header_bottom - header_top)
            ]
            if len(header_edges) >= 2:
                roi = _clip_roi((min(header_edges), header_top, max(header_edges), header_bottom), shape)
                if roi[2] - roi[0] >= 0.45 * w and roi[3] - roi[1] >= 0.06 * h:
                    return roi
        header_bottom = float(ay1 - 18)
    else:
        header_bottom = float(bottom_line["center"])

    top_candidates = [
        line for line in h_lines
        if float(line["center"]) < header_bottom - 55
        and float(line["span"]) >= 0.45 * w
        and int(line["x1"]) <= ax1 + 0.25 * answer_width
        and int(line["x2"]) >= ax2 - 0.25 * answer_width
    ]
    top_line = max(top_candidates, key=lambda line: float(line["center"])) if top_candidates else None
    if top_line is None:
        # Header separators provide a useful top hint when the horizontal line is faint.
        separator_tops = [
            float(line["y1"]) for line in v_lines
            if ax1 - 25 <= float(line["center"]) <= ax2 + 25
            and float(line["y2"]) >= header_bottom - 12
            and float(line["y1"]) <= header_bottom - 70
            and 0.08 * h <= float(line["span"]) <= 0.35 * h
        ]
        if not separator_tops:
            return None
        header_top = float(np.median(separator_tops))
        x1 = float(ax1)
        x2 = float(ax2)
    else:
        header_top = float(top_line["center"])
        x1_values = [float(top_line["x1"])]
        x2_values = [float(top_line["x2"])]
        if bottom_line is not None:
            x1_values.append(float(bottom_line["x1"]))
            x2_values.append(float(bottom_line["x2"]))
        x1 = float(np.median(x1_values))
        x2 = float(np.median(x2_values))

    roi = _clip_roi((x1, header_top, x2, header_bottom), shape)
    if roi[2] - roi[0] < 0.50 * w or roi[3] - roi[1] < 0.08 * h:
        return None
    return roi


def _detect_header_cells_from_lines(
    header_roi: ROI,
    v_lines: Sequence[Dict[str, Any]],
    shape: Tuple[int, int],
) -> Optional[Tuple[ROI, ROI, List[float]]]:
    hx1, hy1, hx2, hy2 = header_roi
    header_h = hy2 - hy1
    header_w = hx2 - hx1
    raw_xs: List[float] = []

    for line in v_lines:
        x = float(line["center"])
        if not (hx1 - 30 <= x <= hx2 + 30):
            continue
        overlap = min(float(line["y2"]), hy2 + 10) - max(float(line["y1"]), hy1 - 10)
        if overlap >= max(45.0, 0.40 * header_h):
            raw_xs.append(x)

    if not raw_xs:
        return None

    boundaries = _cluster_centers(raw_xs, gap=max(14.0, 0.018 * shape[1]))
    if not any(abs(x - hx1) <= 24 for x in boundaries):
        boundaries.append(float(hx1))
    if not any(abs(x - hx2) <= 24 for x in boundaries):
        boundaries.append(float(hx2))

    boundaries = _cluster_centers(boundaries, gap=max(14.0, 0.018 * shape[1]))
    boundaries = sorted(x for x in boundaries if hx1 - 35 <= x <= hx2 + 35)

    # If an outer frame line survives near the left edge, it can create a tiny
    # false first column. Drop impossible slivers before assigning cells.
    while len(boundaries) >= 4 and boundaries[1] - boundaries[0] < 0.12 * header_w:
        boundaries.pop(0)
    while len(boundaries) >= 4 and boundaries[-1] - boundaries[-2] < 0.12 * header_w:
        boundaries.pop()

    if len(boundaries) < 3:
        return None

    student_roi = _clip_roi((boundaries[0], hy1, boundaries[1], hy2), shape)
    test_roi = _clip_roi((boundaries[1], hy1, boundaries[2], hy2), shape)

    if student_roi[2] - student_roi[0] < 0.15 * header_w or test_roi[2] - test_roi[0] < 0.08 * header_w:
        return None
    return student_roi, test_roi, boundaries


def detect_sheet_layout(warped_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Detect high-level printed regions on the warped sheet.

    The returned ROIs are pixel coordinates on the warped canvas. They are used
    both for visual overlays and as the search windows for bubble detection.
    """
    shape = warped_bgr.shape[:2]
    fallback = _fallback_layout(shape)
    h_lines, v_lines = _extract_printed_lines(warped_bgr)

    source = {
        "header": "fallback_normalized_roi",
        "student_id": "fallback_normalized_roi",
        "test_id": "fallback_normalized_roi",
        "answers": "fallback_normalized_roi",
    }

    answer_roi = _detect_answer_region_from_lines(h_lines, v_lines, shape)
    if answer_roi is None:
        answer_roi = tuple(fallback["answers"])  # type: ignore[assignment]
    else:
        source["answers"] = "printed_line_region"

    header_roi = _detect_header_region_from_lines(h_lines, v_lines, answer_roi, shape)
    if header_roi is None:
        header_roi = tuple(fallback["header"])  # type: ignore[assignment]
    else:
        source["header"] = "printed_line_region"

    cell_result = _detect_header_cells_from_lines(header_roi, v_lines, shape)
    boundaries: List[float] = []
    if cell_result is None:
        student_roi = tuple(fallback["student_id"])  # type: ignore[assignment]
        test_roi = tuple(fallback["test_id"])  # type: ignore[assignment]
    else:
        student_roi, test_roi, boundaries = cell_result
        source["student_id"] = "printed_header_cell"
        source["test_id"] = "printed_header_cell"

    method = "printed_line_layout" if all(value.startswith("printed") for value in source.values()) else "hybrid_layout"
    return {
        "method": method,
        "header": list(header_roi),
        "student_id": list(student_roi),
        "test_id": list(test_roi),
        "answers": list(answer_roi),
        "source": source,
        "header_boundaries_x": [round(float(x), 2) for x in boundaries],
        "line_counts": {"horizontal": len(h_lines), "vertical": len(v_lines)},
    }


def _dedupe_circles(circles: List[CircleCandidate], distance: float = 6.0) -> List[CircleCandidate]:
    """Remove near-duplicate circle detections."""
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: c.score, reverse=True)
    kept: List[CircleCandidate] = []
    for c in circles:
        if all(math.hypot(c.x - k.x, c.y - k.y) > distance for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: (c.y, c.x))


def _contour_circle_candidates(gray_roi: np.ndarray) -> List[CircleCandidate]:
    """
    Detect circular bubble-like contours.

    This is used for the answer area, because it gives very clean A-E lanes when
    restricted to the answer rectangle.
    """
    norm = _normalize_lighting(gray_roi, sigma=25)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[CircleCandidate] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if not (20 <= area <= 420):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (5 <= w <= 30 and 5 <= h <= 30):
            continue
        ratio = w / float(h)
        if ratio < 0.60 or ratio > 1.55:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circularity = float(4 * np.pi * area / (peri * peri))
        if circularity < 0.32:
            continue
        candidates.append(CircleCandidate(x + w / 2.0, y + h / 2.0, max(w, h) / 2.0, circularity))

    return _dedupe_circles(candidates, distance=5.0)


def _hough_circle_candidates(gray_roi: np.ndarray) -> List[CircleCandidate]:
    """
    Detect circles using Hough transform.

    This works better for the Student ID and Test ID grids, where the bubbles are
    small, regular, and sometimes too thin for contour filtering.
    """
    norm = _normalize_lighting(gray_roi, sigma=17)
    norm = cv2.equalizeHist(norm)
    blur = cv2.medianBlur(norm, 3)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=13,
        param1=80,
        param2=11,
        minRadius=5,
        maxRadius=12,
    )
    if circles is None:
        return []

    out: List[CircleCandidate] = []
    for x, y, r in np.round(circles[0]).astype(int):
        out.append(CircleCandidate(float(x), float(y), float(r), 1.0))
    return _dedupe_circles(out, distance=7.0)


def _cluster_1d(values: Iterable[float], gap: float) -> List[Dict[str, float]]:
    values = sorted(float(v) for v in values)
    groups: List[List[float]] = []
    for v in values:
        if not groups:
            groups.append([v])
            continue
        if v - float(np.mean(groups[-1])) > gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [{"center": float(np.mean(g)), "count": float(len(g))} for g in groups]


def _valid_regular_centers(centers: Sequence[float], min_gap: float, max_gap: float, max_cv: float) -> bool:
    if len(centers) < 2:
        return False
    gaps = np.diff(np.asarray(centers, dtype=float))
    median_gap = float(np.median(gaps))
    if median_gap < min_gap or median_gap > max_gap:
        return False
    if float(np.min(gaps)) <= 0:
        return False
    if float(np.max(gaps)) / float(np.min(gaps)) > 1.7:
        return False
    cv = float(np.std(gaps) / (median_gap + 1e-6))
    return cv <= max_cv


def _find_lane_groups(
    clusters: List[Dict[str, float]],
    lanes_per_group: int = 5,
    min_gap: float = 12,
    max_gap: float = 35,
    max_cv: float = 0.18,
    min_count: int = 3,
) -> List[List[float]]:
    """Find non-overlapping regular x-lane groups, e.g. A-B-C-D-E."""
    clusters = sorted(clusters, key=lambda c: c["center"])
    groups: List[List[float]] = []
    i = 0
    while i <= len(clusters) - lanes_per_group:
        window = clusters[i : i + lanes_per_group]
        centers = [c["center"] for c in window]
        counts = [c["count"] for c in window]
        if min(counts) >= min_count and _valid_regular_centers(centers, min_gap, max_gap, max_cv):
            groups.append([float(x) for x in centers])
            i += lanes_per_group
        else:
            i += 1
    return groups


def _best_regular_subset(
    clusters: List[Dict[str, float]],
    n: int,
    min_gap: float,
    max_gap: float,
) -> Optional[List[float]]:
    """Choose n cluster centers that form the best regular grid axis."""
    clusters = sorted(clusters, key=lambda c: c["center"])
    if len(clusters) < n:
        return None

    best_score = float("inf")
    best_centers: Optional[List[float]] = None
    max_combinations = 50000
    checked = 0

    for idxs in itertools.combinations(range(len(clusters)), n):
        checked += 1
        if checked > max_combinations:
            break
        chosen = [clusters[i] for i in idxs]
        centers = [c["center"] for c in chosen]
        gaps = np.diff(np.asarray(centers, dtype=float))
        med = float(np.median(gaps))
        if med < min_gap or med > max_gap:
            continue
        if float(np.min(gaps)) <= 0 or float(np.max(gaps)) / float(np.min(gaps)) > 3.0:
            continue
        cv = float(np.std(gaps) / (med + 1e-6))
        total_count = sum(float(c["count"]) for c in chosen)
        # Regular spacing is the priority; high support count is a tie-breaker.
        score = cv - 0.002 * total_count
        if score < best_score:
            best_score = score
            best_centers = [float(x) for x in centers]

    return best_centers


def _fill_score(gray: np.ndarray, x: float, y: float, r: float) -> float:
    """
    Estimate how strongly a bubble is marked, relative to nearby paper.

    A raw darkness score is sensitive to shadows: an empty bubble in a dark
    corner can look more "filled" than it should. This score compares the
    bubble interior with a small surrounding ring, so gradual lighting changes
    mostly cancel out.
    """
    h, w = gray.shape[:2]
    cx, cy = int(round(x)), int(round(y))

    inner_r = max(3, int(round(r * 0.72)))
    annulus_inner_r = max(inner_r + 1, int(round(r * 1.05)))
    annulus_outer_r = max(annulus_inner_r + 2, int(round(r * 1.55)))

    x1, y1 = max(0, cx - annulus_outer_r), max(0, cy - annulus_outer_r)
    x2, y2 = min(w, cx + annulus_outer_r + 1), min(h, cy + annulus_outer_r + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    patch = gray[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inner_mask = dist2 <= inner_r ** 2
    annulus_mask = (dist2 >= annulus_inner_r ** 2) & (dist2 <= annulus_outer_r ** 2)
    if inner_mask.sum() == 0 or annulus_mask.sum() == 0:
        return 0.0

    inner_vals = patch[inner_mask].astype(np.float32)
    background_vals = patch[annulus_mask].astype(np.float32)

    # Use a bright-ish percentile instead of the mean so neighboring bubble
    # outlines or tiny bits of text in the annulus do not pull the paper level
    # down too much.
    background_level = float(np.percentile(background_vals, 70))
    contrast = np.clip(background_level - inner_vals, 0.0, 255.0)
    contrast_mean = float(contrast.mean() / 255.0)

    # Filled marks darken many pixels, while printed letters usually darken only
    # a small part of the bubble. Mixing in dark-pixel coverage helps separate
    # a real fill from the printed A/B/C/D/E character.
    dark_pixel_fraction = float(np.mean(inner_vals <= background_level - 35.0))
    return float(np.clip(0.75 * contrast_mean + 0.25 * dark_pixel_fraction, 0.0, 1.0))


# -----------------------------------------------------------------------------
# Section detectors
# -----------------------------------------------------------------------------

def _infer_rows_for_answer_group(
    candidates: Sequence[CircleCandidate],
    xs: Sequence[float],
    cluster_gap: float = 10.0,
) -> List[float]:
    """
    Infer the real printed rows for one A-B-C-D-E answer block.

    Important: shorter sheets are not always rectangular full blocks. For example,
    a 30-question sheet can be printed as 25 rows in the first block and only
    5 rows in the second block. The previous implementation used one global
    y-axis and therefore generated fake rows for the short second block.

    This function solves that by checking row support inside each x-group
    independently. A row is accepted only if several of the 5 option lanes have
    a nearby circle candidate.
    """
    if not candidates or not xs:
        return []

    xs = sorted(float(x) for x in xs)
    if len(xs) >= 2:
        lane_gap = float(np.median(np.diff(np.asarray(xs, dtype=float))))
    else:
        lane_gap = 18.0

    x_tol = float(np.clip(lane_gap * 0.45, 6.0, 13.0))
    y_tol = 8.0

    # Keep only detections that belong to this A-E block, not to another block
    # or to question numbers/table text.
    group_candidates = [
        c for c in candidates
        if min(abs(float(c.x) - x) for x in xs) <= x_tol
    ]
    if not group_candidates:
        return []

    y_clusters = _cluster_1d([c.y for c in group_candidates], gap=cluster_gap)

    rows: List[float] = []
    for cluster in y_clusters:
        y = float(cluster["center"])
        lane_support = 0
        for x in xs:
            if any(abs(float(c.x) - x) <= x_tol and abs(float(c.y) - y) <= y_tol for c in group_candidates):
                lane_support += 1

        # Normal rows should have 5 bubbles. Allow 3+ because filled bubbles,
        # camera blur, or thresholding can occasionally hide one or two rings.
        if lane_support >= 3:
            rows.append(y)

    return sorted(rows)


def _detect_answer_grid(
    warped_bgr: np.ndarray,
    expected_questions: Optional[int] = None,
    roi: Optional[ROI] = None,
    roi_norm: Tuple[float, float, float, float] = DEFAULT_ROIS["answers"],
) -> Dict[str, Any]:
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = _resolve_roi(gray.shape, roi, roi_norm)
    roi = gray[y1:y2, x1:x2]

    local_candidates = _contour_circle_candidates(roi)
    candidates = [CircleCandidate(c.x + x1, c.y + y1, c.r, c.score) for c in local_candidates]

    x_clusters = _cluster_1d([c.x for c in candidates], gap=10)
    x_groups = _find_lane_groups(x_clusters, lanes_per_group=5, min_gap=12, max_gap=38, max_cv=0.18, min_count=3)

    supported_groups: List[Tuple[List[float], List[float]]] = []
    for xs in x_groups:
        rows = _infer_rows_for_answer_group(candidates, xs, cluster_gap=10.0)
        if rows:
            supported_groups.append((xs, rows))

    if not supported_groups:
        return {
            "roi": [x1, y1, x2, y2],
            "bubble_count": 0,
            "question_count": 0,
            "groups_detected": 0,
            "rows_per_group": [],
            "bubbles": [],
            "raw_circle_candidates": len(candidates),
            "warning": "Could not infer a regular A-E answer grid.",
        }

    # Keep only x-groups that actually had supported rows. This prevents a
    # text fragment from being reported as an empty answer block.
    x_groups = [group for group, _ in supported_groups]
    group_rows = [rows for _, rows in supported_groups]

    median_r = float(np.median([c.r for c in candidates])) if candidates else 8.0
    median_r = float(np.clip(median_r, 5.0, 12.0))

    bubbles: List[Dict[str, Any]] = []
    question = 1
    for group_idx, (xs, rows) in enumerate(zip(x_groups, group_rows)):
        for row_idx, y in enumerate(rows):
            if expected_questions is not None and question > expected_questions:
                break
            for option_idx, x in enumerate(xs):
                bubbles.append(
                    {
                        "section": "answers",
                        "question": int(question),
                        "option": OPTIONS[option_idx],
                        "option_index": int(option_idx),
                        "group_index": int(group_idx + 1),
                        "row_in_group": int(row_idx + 1),
                        "x": round(float(x), 2),
                        "y": round(float(y), 2),
                        "r": round(median_r, 2),
                        "fill_score": round(_fill_score(gray, x, y, median_r), 4),
                    }
                )
            question += 1
        if expected_questions is not None and question > expected_questions:
            break

    question_count = len({b["question"] for b in bubbles})
    return {
        "roi": [x1, y1, x2, y2],
        "bubble_count": len(bubbles),
        "question_count": question_count,
        "groups_detected": len(x_groups),
        "rows_per_group": [len(rows) for rows in group_rows],
        "x_groups": [[round(float(x), 2) for x in group] for group in x_groups],
        "y_rows_by_group": [[round(float(y), 2) for y in rows] for rows in group_rows],
        "bubbles": bubbles,
        "raw_circle_candidates": len(candidates),
    }


def _answer_detection_score(result: Dict[str, Any], expected_questions: Optional[int]) -> float:
    question_count = int(result.get("question_count", 0))
    bubble_count = int(result.get("bubble_count", 0))
    rows_per_group = [int(v) for v in result.get("rows_per_group", [])]
    known_counts = {10, 20, 30, 50, 100}

    if expected_questions is not None:
        score = -5000.0 * abs(question_count - expected_questions)
        score += 20.0 * question_count + 0.1 * bubble_count
    else:
        score = 1000.0 * question_count + 0.1 * bubble_count
        if question_count in known_counts:
            score += 2500.0

    if rows_per_group and max(rows_per_group) > 25:
        score -= 5000.0 * (max(rows_per_group) - 25)
    if question_count == 0:
        score -= 1000.0
    return score


def _tighten_retry_answer_roi(result: Dict[str, Any], search_roi: ROI, shape: Tuple[int, int]) -> None:
    rows: List[float] = []
    for group_rows in result.get("y_rows_by_group", []):
        rows.extend(float(y) for y in group_rows)
    if not rows:
        rows = [float(b["y"]) for b in result.get("bubbles", []) if "y" in b]
    if not rows:
        return

    sorted_rows = sorted(set(round(y, 2) for y in rows))
    gaps = np.diff(np.asarray(sorted_rows, dtype=float)) if len(sorted_rows) >= 2 else np.asarray([])
    row_gap = float(np.median(gaps)) if len(gaps) else 28.0
    top_pad = max(24.0, 1.25 * row_gap)
    x1, _, x2, y2 = search_roi
    refined_roi = _clip_roi((x1, min(sorted_rows) - top_pad, x2, y2), shape)
    result["search_roi"] = list(search_roi)
    result["roi"] = list(refined_roi)


def _detect_answer_grid_with_retries(
    warped_bgr: np.ndarray,
    layout: Dict[str, Any],
    expected_questions: Optional[int] = None,
) -> Dict[str, Any]:
    base_roi = _clip_roi(layout.get("answers", _roi_pixels(warped_bgr.shape[:2], DEFAULT_ROIS["answers"])), warped_bgr.shape[:2])
    candidates: List[ROI] = [base_roi]

    answer_source = layout.get("source", {}).get("answers")
    if answer_source != "printed_line_region":
        h, _ = warped_bgr.shape[:2]
        x1, _, x2, y2 = base_roi
        y2 = max(y2, int(0.90 * h))
        for y_norm in (0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30):
            candidates.append(_clip_roi((x1, int(y_norm * h), x2, y2), warped_bgr.shape[:2]))

    seen: set[ROI] = set()
    best: Optional[Dict[str, Any]] = None
    best_candidate: Optional[ROI] = None
    best_score = -float("inf")
    for candidate_roi in candidates:
        if candidate_roi in seen:
            continue
        seen.add(candidate_roi)
        result = _detect_answer_grid(warped_bgr, expected_questions=expected_questions, roi=candidate_roi)
        score = _answer_detection_score(result, expected_questions)
        if score > best_score:
            best_score = score
            best = result
            best_candidate = candidate_roi

    if best is None:
        return _detect_answer_grid(warped_bgr, expected_questions=expected_questions, roi=base_roi)
    if best_candidate is not None and best_candidate != base_roi and best.get("bubble_count", 0) > 0:
        _tighten_retry_answer_roi(best, best_candidate, warped_bgr.shape[:2])
    return best


def _detect_digit_grid(
    warped_bgr: np.ndarray,
    section_name: str,
    expected_columns: int,
    roi_norm: Tuple[float, float, float, float],
    roi: Optional[ROI] = None,
) -> Dict[str, Any]:
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = _resolve_roi(gray.shape, roi, roi_norm)
    roi = gray[y1:y2, x1:x2]

    local_candidates = _hough_circle_candidates(roi)
    candidates = [CircleCandidate(c.x + x1, c.y + y1, c.r, c.score) for c in local_candidates]

    x_clusters = _cluster_1d([c.x for c in candidates], gap=9)
    y_clusters = _cluster_1d([c.y for c in candidates], gap=9)

    x_lanes = _best_regular_subset(x_clusters, expected_columns, min_gap=16, max_gap=35)
    y_lanes = _best_regular_subset(y_clusters, 10, min_gap=9, max_gap=28)

    if not x_lanes or not y_lanes:
        return {
            "roi": [x1, y1, x2, y2],
            "bubble_count": 0,
            "columns_detected": 0,
            "rows_detected": 0,
            "bubbles": [],
            "raw_circle_candidates": len(candidates),
            "warning": f"Could not infer a regular grid for {section_name}.",
        }

    median_r = float(np.median([c.r for c in candidates])) if candidates else 7.0
    median_r = float(np.clip(median_r, 5.0, 10.0))

    bubbles: List[Dict[str, Any]] = []
    for col_idx, x in enumerate(x_lanes):
        for digit_value, y in enumerate(y_lanes):
            bubbles.append(
                {
                    "section": section_name,
                    "digit_position": int(col_idx + 1),
                    "value": int(digit_value),
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "r": round(median_r, 2),
                    "fill_score": round(_fill_score(gray, x, y, median_r), 4),
                }
            )

    return {
        "roi": [x1, y1, x2, y2],
        "bubble_count": len(bubbles),
        "columns_detected": len(x_lanes),
        "rows_detected": len(y_lanes),
        "x_columns": [round(float(x), 2) for x in x_lanes],
        "y_digits": [round(float(y), 2) for y in y_lanes],
        "bubbles": bubbles,
        "raw_circle_candidates": len(candidates),
    }


def detect_omr_bubbles(
    image_bgr: np.ndarray,
    expected_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main API for Streamlit or backend code.

    Args:
        image_bgr: OpenCV BGR image.
        expected_questions: Optional known question count: 10, 20, 30, 50, or 100.
            When provided, extra generated answer bubbles after that question are trimmed.

    Returns:
        dict with warped image metadata and detected bubble coordinates.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")

    warped, warp_meta = warp_sheet(image_bgr)
    layout = detect_sheet_layout(warped)

    student = _detect_digit_grid(
        warped,
        section_name="student_id",
        expected_columns=8,
        roi_norm=DEFAULT_ROIS["student_id"],
        roi=tuple(layout["student_id"]),
    )
    test = _detect_digit_grid(
        warped,
        section_name="test_id",
        expected_columns=4,
        roi_norm=DEFAULT_ROIS["test_id"],
        roi=tuple(layout["test_id"]),
    )
    answers = _detect_answer_grid_with_retries(
        warped,
        layout,
        expected_questions=expected_questions,
    )
    if answers.get("roi") and list(answers["roi"]) != list(layout.get("answers", [])):
        layout["answers"] = list(answers["roi"])
        layout.setdefault("source", {})["answers"] = "answer_grid_retry"
        layout["method"] = "hybrid_layout"

    return {
        "metadata": {
            **warp_meta,
            "expected_questions": expected_questions,
            "layout_method": layout.get("method", "unknown"),
            "note": "Detection only. fill_score is included for later grading but no answer is graded here.",
        },
        "layout": layout,
        "student_id": student,
        "test_id": test,
        "answers": answers,
    }


# -----------------------------------------------------------------------------
# Reading and grading
# -----------------------------------------------------------------------------

def _choice_groups(
    bubbles: Sequence[Dict[str, Any]],
    group_key: str,
) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for bubble in bubbles:
        if group_key not in bubble:
            continue
        groups.setdefault(int(bubble[group_key]), []).append(bubble)
    return groups


def _classify_marked_choice(
    bubbles: Sequence[Dict[str, Any]],
    value_key: str,
    min_delta: float,
    multi_delta: float,
) -> Dict[str, Any]:
    if not bubbles:
        return {
            "status": "missing",
            "selected": None,
            "selected_values": [],
            "confidence": 0.0,
            "top_fill_score": 0.0,
            "second_fill_score": 0.0,
            "baseline_fill_score": 0.0,
        }

    ranked = sorted(bubbles, key=lambda b: float(b.get("fill_score", 0.0)), reverse=True)
    scores = [float(b.get("fill_score", 0.0)) for b in ranked]
    baseline = float(np.median(scores))
    top = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0
    confidence = top - second
    relative_strength = top - baseline

    strong = [
        b for b in ranked
        if float(b.get("fill_score", 0.0)) >= baseline + min_delta
        and top - float(b.get("fill_score", 0.0)) <= max(multi_delta, min_delta * 0.85)
    ]

    if relative_strength < min_delta:
        status = "blank"
        selected = None
        selected_values: List[Any] = []
    elif len(strong) > 1 and confidence <= multi_delta:
        status = "multiple"
        selected = None
        selected_values = [b.get(value_key) for b in strong]
    elif confidence < max(0.045, multi_delta * 0.70):
        status = "unclear"
        selected = ranked[0].get(value_key)
        selected_values = [ranked[0].get(value_key)]
    else:
        status = "selected"
        selected = ranked[0].get(value_key)
        selected_values = [selected]

    return {
        "status": status,
        "selected": selected,
        "selected_values": selected_values,
        "confidence": round(float(confidence), 4),
        "top_fill_score": round(float(top), 4),
        "second_fill_score": round(float(second), 4),
        "baseline_fill_score": round(float(baseline), 4),
    }


def normalize_answer_key(answer_key: Any) -> Dict[int, str]:
    """
    Normalize a list/dict answer key to {question_number: option}.

    Accepted options are A-E. Invalid or empty entries are ignored so callers can
    pass partial keys while still reading every detected answer.
    """
    if answer_key is None:
        return {}

    normalized: Dict[int, str] = {}
    if isinstance(answer_key, dict):
        items = answer_key.items()
    elif isinstance(answer_key, (list, tuple)):
        items = enumerate(answer_key, start=1)
    else:
        raise TypeError("answer_key must be a dict, list, tuple, or None")

    for question, option in items:
        try:
            q = int(question)
        except (TypeError, ValueError):
            continue
        if q <= 0 or option is None:
            continue
        opt = str(option).strip().upper()
        if opt in OPTIONS:
            normalized[q] = opt
    return normalized


def read_answer_choices(answer_section: Dict[str, Any]) -> Dict[str, Any]:
    """Read selected A-E choices from detected answer bubbles."""
    question_groups = _choice_groups(answer_section.get("bubbles", []), "question")
    rows: List[Dict[str, Any]] = []

    for question in sorted(question_groups):
        bubbles = sorted(question_groups[question], key=lambda b: int(b.get("option_index", 0)))
        classification = _classify_marked_choice(
            bubbles,
            value_key="option",
            min_delta=0.075,
            multi_delta=0.070,
        )
        row = {
            "question": int(question),
            "detected_answer": classification["selected"],
            "selected_options": classification["selected_values"],
            "status": classification["status"],
            "confidence": classification["confidence"],
            "top_fill_score": classification["top_fill_score"],
            "second_fill_score": classification["second_fill_score"],
            "baseline_fill_score": classification["baseline_fill_score"],
        }
        rows.append(row)

    counts = {
        "selected": sum(1 for row in rows if row["status"] == "selected"),
        "blank": sum(1 for row in rows if row["status"] == "blank"),
        "multiple": sum(1 for row in rows if row["status"] == "multiple"),
        "unclear": sum(1 for row in rows if row["status"] == "unclear"),
        "missing": sum(1 for row in rows if row["status"] == "missing"),
    }
    return {
        "question_count": len(rows),
        "responses": rows,
        "counts": counts,
    }


def decode_digit_grid(section: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a Student ID or Test ID digit bubble grid."""
    position_groups = _choice_groups(section.get("bubbles", []), "digit_position")
    expected_columns = int(section.get("columns_detected", 0) or len(position_groups))
    positions: List[Dict[str, Any]] = []

    for position in range(1, expected_columns + 1):
        bubbles = sorted(position_groups.get(position, []), key=lambda b: int(b.get("value", 0)))
        classification = _classify_marked_choice(
            bubbles,
            value_key="value",
            min_delta=0.095,
            multi_delta=0.060,
        )
        selected = classification["selected"]
        digit = int(selected) if classification["status"] == "selected" and selected is not None else None
        positions.append(
            {
                "position": int(position),
                "digit": digit,
                "selected_values": classification["selected_values"],
                "status": classification["status"],
                "confidence": classification["confidence"],
                "top_fill_score": classification["top_fill_score"],
                "second_fill_score": classification["second_fill_score"],
                "baseline_fill_score": classification["baseline_fill_score"],
            }
        )

    value_chars: List[str] = []
    for item in positions:
        if item["status"] == "selected" and item["digit"] is not None:
            value_chars.append(str(item["digit"]))
        elif item["status"] == "blank":
            value_chars.append("_")
        elif item["status"] == "multiple":
            value_chars.append("*")
        else:
            value_chars.append("?")

    return {
        "value": "".join(value_chars),
        "complete": bool(positions) and all(item["status"] == "selected" for item in positions),
        "positions": positions,
        "counts": {
            "selected": sum(1 for item in positions if item["status"] == "selected"),
            "blank": sum(1 for item in positions if item["status"] == "blank"),
            "multiple": sum(1 for item in positions if item["status"] == "multiple"),
            "unclear": sum(1 for item in positions if item["status"] == "unclear"),
            "missing": sum(1 for item in positions if item["status"] == "missing"),
        },
    }


def grade_omr_result(
    result: Dict[str, Any],
    answer_key: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Read IDs and grade answers from an existing detection result.

    This is intentionally layered on top of detect_omr_bubbles(). It consumes the
    bubble coordinates/fill_score values and does not change detection behavior.
    """
    key = normalize_answer_key(answer_key)
    answer_reading = read_answer_choices(result.get("answers", {}))
    student_reading = decode_digit_grid(result.get("student_id", {}))
    test_reading = decode_digit_grid(result.get("test_id", {}))

    graded_questions: List[Dict[str, Any]] = []
    correct = incorrect = blank = multiple = unclear = unkeyed = 0

    for row in answer_reading["responses"]:
        question = int(row["question"])
        expected = key.get(question)
        detected = row.get("detected_answer")
        status = str(row.get("status", "missing"))
        is_correct = False
        outcome = "unkeyed"

        if expected is None:
            unkeyed += 1
        elif status == "selected":
            if detected == expected:
                correct += 1
                is_correct = True
                outcome = "correct"
            else:
                incorrect += 1
                outcome = "incorrect"
        elif status == "blank":
            blank += 1
            outcome = "blank"
        elif status == "multiple":
            multiple += 1
            outcome = "multiple"
        else:
            unclear += 1
            outcome = "unclear"

        graded_questions.append(
            {
                **row,
                "correct_answer": expected,
                "is_correct": is_correct,
                "outcome": outcome,
            }
        )

    keyed_questions = sum(1 for item in graded_questions if item.get("correct_answer") is not None)
    answered_questions = sum(1 for item in graded_questions if item.get("status") == "selected")
    score_percent = round((correct / keyed_questions) * 100.0, 2) if keyed_questions else None

    return {
        "identity": {
            "student_id": student_reading,
            "test_id": test_reading,
        },
        "answer_key": key,
        "answers": answer_reading,
        "questions": graded_questions,
        "summary": {
            "question_count": answer_reading["question_count"],
            "keyed_questions": keyed_questions,
            "answered_questions": answered_questions,
            "correct": correct,
            "incorrect": incorrect,
            "blank": blank,
            "multiple": multiple,
            "unclear": unclear,
            "unkeyed": unkeyed,
            "needs_review": blank + multiple + unclear,
            "score_percent": score_percent,
        },
    }


# -----------------------------------------------------------------------------
# Drawing / exports
# -----------------------------------------------------------------------------

def draw_detection_overlay(
    warped_bgr: np.ndarray,
    result: Dict[str, Any],
    draw_rois: bool = True,
    draw_labels: bool = False,
) -> np.ndarray:
    """Draw detected bubbles on the warped image."""
    overlay = warped_bgr.copy()

    if draw_rois:
        layout = result.get("layout", {})
        header_roi = layout.get("header")
        if header_roi:
            x1, y1, x2, y2 = map(int, header_roi)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 0, 180), 2)

        for section_key, color in [("student_id", (255, 0, 0)), ("test_id", (255, 170, 0)), ("answers", (0, 170, 255))]:
            roi = result.get(section_key, {}).get("roi")
            if roi:
                x1, y1, x2, y2 = map(int, roi)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    # BGR colors
    colors = {
        "answers": (0, 180, 0),
        "student_id": (255, 0, 0),
        "test_id": (255, 170, 0),
    }

    for section_key in ["answers", "student_id", "test_id"]:
        section = result.get(section_key, {})
        color = colors.get(section_key, (0, 255, 0))
        for b in section.get("bubbles", []):
            x, y, r = int(round(b["x"])), int(round(b["y"])), int(round(b["r"]))
            cv2.circle(overlay, (x, y), r, color, 2)
            # Small center dot makes visual debugging easier.
            cv2.circle(overlay, (x, y), 1, color, -1)
            if draw_labels and section_key == "answers" and b.get("option") == "A":
                cv2.putText(
                    overlay,
                    str(b.get("question", "")),
                    (x - 28, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    summary = (
        f"Answers: {result.get('answers', {}).get('question_count', 0)} questions / "
        f"{result.get('answers', {}).get('bubble_count', 0)} bubbles"
    )
    cv2.rectangle(overlay, (20, 18), (620, 52), (255, 255, 255), -1)
    cv2.putText(overlay, summary, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return overlay


def _draw_check_mark(image: np.ndarray, x: int, y: int, r: int, color: Tuple[int, int, int]) -> None:
    cv2.line(image, (x - r // 2, y), (x - r // 6, y + r // 3), color, 2, cv2.LINE_AA)
    cv2.line(image, (x - r // 6, y + r // 3), (x + r // 2, y - r // 3), color, 2, cv2.LINE_AA)


def _draw_cross_mark(image: np.ndarray, x: int, y: int, r: int, color: Tuple[int, int, int]) -> None:
    cv2.line(image, (x - r // 2, y - r // 2), (x + r // 2, y + r // 2), color, 2, cv2.LINE_AA)
    cv2.line(image, (x + r // 2, y - r // 2), (x - r // 2, y + r // 2), color, 2, cv2.LINE_AA)


def draw_grading_overlay(
    warped_bgr: np.ndarray,
    result: Dict[str, Any],
    draw_rois: bool = True,
    draw_labels: bool = False,
    show_correct_answers: bool = True,
) -> np.ndarray:
    """Draw a grading-focused overlay when result['grading'] is available."""
    grading = result.get("grading")
    if not grading:
        return draw_detection_overlay(warped_bgr, result, draw_rois=draw_rois, draw_labels=draw_labels)

    overlay = warped_bgr.copy()

    if draw_rois:
        layout = result.get("layout", {})
        header_roi = layout.get("header")
        if header_roi:
            x1, y1, x2, y2 = map(int, header_roi)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 0, 180), 2)
        for section_key, color in [("student_id", (255, 0, 0)), ("test_id", (255, 170, 0)), ("answers", (0, 170, 255))]:
            roi = result.get(section_key, {}).get("roi")
            if roi:
                x1, y1, x2, y2 = map(int, roi)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    for section_key, color in [("student_id", (255, 0, 0)), ("test_id", (255, 170, 0))]:
        for bubble in result.get(section_key, {}).get("bubbles", []):
            x, y, r = int(round(bubble["x"])), int(round(bubble["y"])), int(round(bubble["r"]))
            cv2.circle(overlay, (x, y), r, color, 1)

    answer_lookup: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for bubble in result.get("answers", {}).get("bubbles", []):
        question = int(bubble.get("question", 0))
        option = str(bubble.get("option", ""))
        answer_lookup.setdefault(question, {})[option] = bubble
        x, y, r = int(round(bubble["x"])), int(round(bubble["y"])), int(round(bubble["r"]))
        cv2.circle(overlay, (x, y), r, (150, 150, 150), 1)
        if draw_labels and option == "A":
            cv2.putText(
                overlay,
                str(question),
                (x - 28, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 180),
                1,
                cv2.LINE_AA,
            )

    key_color = (255, 90, 0)
    correct_color = (0, 170, 0)
    wrong_color = (0, 0, 230)
    review_color = (0, 165, 255)

    for item in grading.get("questions", []):
        question = int(item.get("question", 0))
        bubbles_by_option = answer_lookup.get(question, {})
        correct_answer = item.get("correct_answer")
        outcome = item.get("outcome")

        if show_correct_answers and correct_answer in bubbles_by_option:
            bubble = bubbles_by_option[correct_answer]
            x, y, r = int(round(bubble["x"])), int(round(bubble["y"])), int(round(bubble["r"]))
            cv2.circle(overlay, (x, y), r + 5, key_color, 2)

        selected_options = item.get("selected_options") or []
        if item.get("detected_answer") and item.get("detected_answer") not in selected_options:
            selected_options = [item.get("detected_answer")]

        for option in selected_options:
            if option not in bubbles_by_option:
                continue
            bubble = bubbles_by_option[option]
            x, y, r = int(round(bubble["x"])), int(round(bubble["y"])), int(round(bubble["r"]))
            if outcome == "correct":
                cv2.circle(overlay, (x, y), r + 7, correct_color, 3)
                _draw_check_mark(overlay, x, y, r + 4, correct_color)
            elif outcome == "incorrect":
                cv2.circle(overlay, (x, y), r + 7, wrong_color, 3)
                _draw_cross_mark(overlay, x, y, r + 4, wrong_color)
            elif outcome in {"multiple", "unclear"}:
                cv2.circle(overlay, (x, y), r + 7, review_color, 3)
            else:
                cv2.circle(overlay, (x, y), r + 5, review_color, 2)

    summary = grading.get("summary", {})
    identity = grading.get("identity", {})
    student_value = identity.get("student_id", {}).get("value", "")
    test_value = identity.get("test_id", {}).get("value", "")
    pct = summary.get("score_percent")
    if pct is None:
        summary_text = (
            f"Read: {summary.get('answered_questions', 0)} selected / "
            f"{summary.get('question_count', 0)} questions"
        )
    else:
        summary_text = (
            f"Score: {summary.get('correct', 0)}/{summary.get('keyed_questions', 0)} "
            f"({pct:.2f}%)"
        )
    if student_value or test_value:
        summary_text += f" | ID {student_value or '-'} | Test {test_value or '-'}"

    cv2.rectangle(overlay, (20, 18), (min(960, 35 + 18 * len(summary_text)), 52), (255, 255, 255), -1)
    cv2.putText(overlay, summary_text, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 2, cv2.LINE_AA)
    return overlay


def make_json_safe(result: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure output can be serialized by json.dumps."""
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return convert(result)


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Detect OMR bubbles in a phone photo.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--questions", type=int, default=None, help="Expected questions: 10, 20, 30, 50, or 100")
    parser.add_argument("--out-dir", default="omr_debug_output", help="Folder for warped/overlay/json outputs")
    parser.add_argument("--labels", action="store_true", help="Draw question labels on overlay")
    args = parser.parse_args()

    image = _read_image(args.image)
    result = detect_omr_bubbles(image, expected_questions=args.questions)
    warped, _ = warp_sheet(image)
    overlay = draw_detection_overlay(warped, result, draw_labels=args.labels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    cv2.imwrite(str(out_dir / f"{stem}_warped.jpg"), warped)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), overlay)
    with open(out_dir / f"{stem}_detections.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(result), f, indent=2)

    print(json.dumps(make_json_safe(result["metadata"]), indent=2))
    print(f"Answers: {result['answers']['question_count']} questions, {result['answers']['bubble_count']} bubbles")
    print(f"Student ID bubbles: {result['student_id']['bubble_count']}")
    print(f"Test ID bubbles: {result['test_id']['bubble_count']}")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_cli()
