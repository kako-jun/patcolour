"""Partial color filter - keep masked regions in color, rest in monochrome."""

from pathlib import Path

import cv2
import numpy as np


def rel_to_abs_point(rx: float, ry: float, width: int, height: int) -> tuple[int, int]:
    """Convert relative (0.0–1.0) point to absolute pixel coordinates."""
    return (round(rx * width), round(ry * height))


def rel_to_abs_rect(
    rx: float, ry: float, rw: float, rh: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert relative rect to absolute (x, y, w, h).

    rx/rw are scaled by width; ry/rh are scaled by height.
    """
    return (round(rx * width), round(ry * height), round(rw * width), round(rh * height))


def rel_to_abs_ellipse(
    rcx: float, rcy: float, rrx: float, rry: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert relative ellipse to absolute (cx, cy, rx, ry).

    rcx/rrx are scaled by width; rcy/rry are scaled by height.
    """
    return (round(rcx * width), round(rcy * height), round(rrx * width), round(rry * height))


def _apply_feather(mask: np.ndarray, feather: int) -> np.ndarray:
    """Softly blur mask edges when requested."""
    if feather <= 0:
        return mask

    ksize = feather * 2 + 1
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


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


def detect_sample_color_mask(
    img: np.ndarray,
    sample_point: tuple[int, int],
    lab_radius: float = 18.0,
) -> np.ndarray:
    """Detect pixels near a sampled color using Lab chroma distance.

    This intentionally compares in the chroma plane first, so different brightness levels of
    roughly the same hue can still be kept in color.
    """
    x, y = sample_point
    height, width = img.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        msg = f"sample point out of bounds: {(x, y)} for image {width}x{height}"
        raise ValueError(msg)

    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    target = lab[y, x]

    chroma = lab[:, :, 1:3]
    target_chroma = target[1:3]
    distance = np.linalg.norm(chroma - target_chroma, axis=2)

    mask = np.where(distance <= lab_radius, 255, 0).astype(np.uint8)
    return mask


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

    return _apply_feather(mask, feather)


def apply_partial_color(
    input_path: Path,
    output_path: Path,
    mask_path: Path | None = None,
    rects: list[tuple[int, int, int, int]] | None = None,
    ellipses: list[tuple[int, int, int, int]] | None = None,
    feather: int = 0,
    auto_detect: bool = False,
    sample_point: tuple[int, int] | None = None,
    lab_radius: float = 18.0,
    sample_point_rel: tuple[float, float] | None = None,
    rects_rel: list[tuple[float, float, float, float]] | None = None,
    ellipses_rel: list[tuple[float, float, float, float]] | None = None,
) -> None:
    """Apply partial color effect.

    Three modes, combinable:
    - mask_path: Use an external mask image.
    - rects/ellipses: Spatial region selection.
    - auto_detect: Auto-detect colorful regions (HSV).
    - sample_point: Sample a reference pixel and keep nearby Lab chroma colors.

    When a color-selection mode (`auto_detect` or `sample_point`) is combined with
    rects/ellipses, the final mask is the intersection: only pixels that are BOTH selected by
    color AND inside the specified region are kept in color.
    """
    img = cv2.imread(str(input_path))
    if img is None:
        msg = f"Could not read image: {input_path}"
        raise ValueError(msg)

    h, w = img.shape[:2]

    # Convert relative coordinates to absolute and merge
    if sample_point_rel is not None:
        sample_point = rel_to_abs_point(sample_point_rel[0], sample_point_rel[1], w, h)
    if rects_rel:
        abs_rects = [rel_to_abs_rect(*r, w, h) for r in rects_rel]
        rects = list(rects) + abs_rects if rects else abs_rects
    if ellipses_rel:
        abs_ellipses = [rel_to_abs_ellipse(*e, w, h) for e in ellipses_rel]
        ellipses = list(ellipses) + abs_ellipses if ellipses else abs_ellipses

    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            msg = f"Could not read mask: {mask_path}"
            raise ValueError(msg)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
    elif auto_detect or sample_point is not None:
        if sample_point is not None:
            color_mask = detect_sample_color_mask(img, sample_point, lab_radius=lab_radius)
        else:
            color_mask = detect_color_mask(img)
        color_mask = _apply_feather(color_mask, feather)
        has_region = rects or ellipses
        if has_region:
            region_mask = generate_region_mask(h, w, rects, ellipses, feather=0)
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

    result = img.astype(np.float32) * alpha + mono.astype(np.float32) * (1.0 - alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(str(output_path), result)
