"""Partial color filter - keep masked regions in color, rest in monochrome."""

from pathlib import Path

import cv2
import numpy as np


def detect_color_mask(
    img: np.ndarray,
    hsv_ranges: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Auto-detect colorful regions via HSV thresholding.

    Args:
        img: BGR image.
        hsv_ranges: List of (lower, upper) HSV bounds. Defaults to broad green.

    Returns:
        Binary mask (0 or 255).
    """
    if hsv_ranges is None:
        hsv_ranges = [(np.array([25, 30, 40]), np.array([90, 255, 255]))]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    combined = np.zeros(img.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        combined |= cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.GaussianBlur(combined, (15, 15), 0)


def generate_region_mask(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int]] | None = None,
    ellipses: list[tuple[int, int, int, int]] | None = None,
    feather: int = 0,
) -> np.ndarray:
    """Generate a spatial region mask from coordinates.

    Args:
        height: Image height.
        width: Image width.
        rects: List of (x, y, w, h) rectangles.
        ellipses: List of (cx, cy, rx, ry) ellipses.
        feather: Gaussian blur radius for soft edges (0 = hard edge).

    Returns:
        Grayscale mask (0=outside, 255=inside).
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if rects:
        for x, y, w, h in rects:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            mask[y1:y2, x1:x2] = 255

    if ellipses:
        for cx, cy, rx, ry in ellipses:
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    if feather > 0:
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    return mask


def apply_partial_color(
    input_path: Path,
    output_path: Path,
    mask_path: Path | None = None,
    rects: list[tuple[int, int, int, int]] | None = None,
    ellipses: list[tuple[int, int, int, int]] | None = None,
    feather: int = 0,
    auto_detect: bool = False,
) -> None:
    """Apply partial color effect.

    Three modes, combinable:
    - mask_path: Use an external mask image.
    - rects/ellipses: Spatial region selection.
    - auto_detect: Auto-detect colorful regions (HSV).

    When auto_detect is combined with rects/ellipses, the final mask
    is the intersection: only pixels that are BOTH detected as colorful
    AND inside the specified region are kept in color.
    """
    img = cv2.imread(str(input_path))
    if img is None:
        msg = f"Could not read image: {input_path}"
        raise ValueError(msg)

    h, w = img.shape[:2]

    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            msg = f"Could not read mask: {mask_path}"
            raise ValueError(msg)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
    elif auto_detect:
        color_mask = detect_color_mask(img)
        has_region = rects or ellipses
        if has_region:
            region_mask = generate_region_mask(h, w, rects, ellipses, feather)
            # Intersection: both colorful AND in region
            mask = cv2.min(color_mask, region_mask)
        else:
            mask = color_mask
    else:
        mask = generate_region_mask(h, w, rects, ellipses, feather)

    # Create monochrome version
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mono = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Blend based on mask (0=mono, 255=color)
    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    result = (img.astype(np.float32) * alpha + mono.astype(np.float32) * (1.0 - alpha))
    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(str(output_path), result)
