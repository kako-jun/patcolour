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


def detect_sample_color_mask_lab_full(
    img: np.ndarray,
    sample_point: tuple[int, int],
    lab_radius: float = 18.0,
) -> np.ndarray:
    """Lab full 3D distance (L + a + b)."""
    x, y = sample_point
    height, width = img.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        msg = f"sample point out of bounds: {(x, y)} for image {width}x{height}"
        raise ValueError(msg)

    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    target = lab[y, x]

    distance = np.linalg.norm(lab - target, axis=2)
    mask = np.where(distance <= lab_radius, 255, 0).astype(np.uint8)
    return mask


def detect_sample_color_mask_lch(
    img: np.ndarray,
    sample_point: tuple[int, int],
    lch_radius: float = 18.0,
    lightness_weight: float = 0.3,
) -> np.ndarray:
    """LCh distance with downweighted lightness."""
    x, y = sample_point
    height, width = img.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        msg = f"sample point out of bounds: {(x, y)} for image {width}x{height}"
        raise ValueError(msg)

    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    target = lab[y, x]

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    tL, ta, tb = target[0], target[1], target[2]

    C = np.sqrt(a**2 + b**2)
    h = np.degrees(np.arctan2(b, a))
    tC = np.sqrt(ta**2 + tb**2)
    th = np.degrees(np.arctan2(tb, ta))

    dL = L - tL
    dC = C - tC
    dh_raw = np.abs(h - th)
    dh_norm = np.minimum(dh_raw, 360.0 - dh_raw) / 180.0

    distance = np.sqrt((lightness_weight * dL) ** 2 + dC**2 + dh_norm**2)
    mask = np.where(distance <= lch_radius, 255, 0).astype(np.uint8)
    return mask


def detect_sample_color_mask_xyy(
    img: np.ndarray,
    sample_point: tuple[int, int],
    xyy_radius: float = 0.05,
) -> np.ndarray:
    """xyY chromaticity distance (ignores luminance Y)."""
    x, y = sample_point
    height, width = img.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        msg = f"sample point out of bounds: {(x, y)} for image {width}x{height}"
        raise ValueError(msg)

    img_f = img.astype(np.float32) / 255.0
    xyz = cv2.cvtColor(img_f, cv2.COLOR_BGR2XYZ)

    X = xyz[:, :, 0]
    Y = xyz[:, :, 1]
    Z = xyz[:, :, 2]
    total = X + Y + Z
    safe_total = np.where(total > 0, total, 1.0)  # avoid division by zero

    cx = np.where(total > 0, X / safe_total, 0.0)
    cy = np.where(total > 0, Y / safe_total, 0.0)

    target_xyz = xyz[y, x]
    tX, tY, tZ = target_xyz[0], target_xyz[1], target_xyz[2]
    t_total = tX + tY + tZ
    tcx = tX / t_total if t_total > 0 else 0.0
    tcy = tY / t_total if t_total > 0 else 0.0

    distance = np.sqrt((cx - tcx) ** 2 + (cy - tcy) ** 2)
    mask = np.where(distance <= xyy_radius, 255, 0).astype(np.uint8)
    return mask


COLOR_SPACES = ["lab-chroma", "lab-full", "lch", "xyy"]


def detect_sample_color_mask_by_space(
    img: np.ndarray,
    sample_point: tuple[int, int],
    color_space: str = "lab-chroma",
    lab_radius: float = 18.0,
) -> np.ndarray:
    """Dispatch to the appropriate color-distance function."""
    if color_space == "lab-chroma":
        return detect_sample_color_mask(img, sample_point, lab_radius=lab_radius)
    elif color_space == "lab-full":
        return detect_sample_color_mask_lab_full(img, sample_point, lab_radius=lab_radius)
    elif color_space == "lch":
        return detect_sample_color_mask_lch(img, sample_point, lch_radius=lab_radius)
    elif color_space == "xyy":
        xyy_radius = lab_radius / 360.0
        return detect_sample_color_mask_xyy(img, sample_point, xyy_radius=xyy_radius)
    else:
        msg = f"Unknown color_space: {color_space!r}. Must be one of {COLOR_SPACES}"
        raise ValueError(msg)


def _draw_regions(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int]] | None,
    ellipses: list[tuple[int, int, int, int]] | None,
) -> np.ndarray:
    """Draw rects and ellipses onto a blank mask. Internal helper."""
    mask = np.zeros((height, width), dtype=np.uint8)

    if rects:
        for x, y, w, h in rects:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            mask[y1:y2, x1:x2] = 255

    if ellipses:
        for cx, cy, rx, ry in ellipses:
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

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
    return _apply_feather(_draw_regions(height, width, rects, ellipses), feather)


def generate_exclude_mask(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int]] | None = None,
    ellipses: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Generate an exclusion mask from coordinate regions.

    Always applied with hard edges regardless of ``--feather``.

    Returns:
        Binary mask where 255 = exclude (suppress color here).
    """
    return _draw_regions(height, width, rects, ellipses)


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
    color_space: str = "lab-chroma",
    sample_point_rel: tuple[float, float] | None = None,
    rects_rel: list[tuple[float, float, float, float]] | None = None,
    ellipses_rel: list[tuple[float, float, float, float]] | None = None,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
    exclude_ellipses: list[tuple[int, int, int, int]] | None = None,
    exclude_rects_rel: list[tuple[float, float, float, float]] | None = None,
    exclude_ellipses_rel: list[tuple[float, float, float, float]] | None = None,
) -> None:
    """Apply partial color effect.

    Three modes, combinable:
    - mask_path: Use an external mask image.
    - rects/ellipses: Spatial region selection.
    - auto_detect: Auto-detect colorful regions (HSV).
    - sample_point: Sample a reference pixel and keep nearby colors (see color_space).

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
    if exclude_rects_rel:
        abs_excl_rects = [rel_to_abs_rect(*r, w, h) for r in exclude_rects_rel]
        exclude_rects = list(exclude_rects) + abs_excl_rects if exclude_rects else abs_excl_rects
    if exclude_ellipses_rel:
        abs_excl_ellipses = [rel_to_abs_ellipse(*e, w, h) for e in exclude_ellipses_rel]
        exclude_ellipses = (
            list(exclude_ellipses) + abs_excl_ellipses if exclude_ellipses else abs_excl_ellipses
        )

    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            msg = f"Could not read mask: {mask_path}"
            raise ValueError(msg)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
    elif auto_detect or sample_point is not None:
        if sample_point is not None:
            color_mask = detect_sample_color_mask_by_space(
                img, sample_point, color_space=color_space, lab_radius=lab_radius
            )
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

    # Apply exclude mask: exclude always wins over positive
    has_exclude = exclude_rects or exclude_ellipses
    if has_exclude:
        excl_mask = generate_exclude_mask(h, w, exclude_rects, exclude_ellipses)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(excl_mask))

    # Create monochrome version
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mono = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Blend based on mask (0=mono, 255=color)
    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    result = img.astype(np.float32) * alpha + mono.astype(np.float32) * (1.0 - alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(str(output_path), result)


def apply_color_space_comparison(
    input_path: Path,
    output_dir: Path,
    sample_point: tuple[int, int] | None = None,
    sample_point_rel: tuple[float, float] | None = None,
    lab_radius: float = 18.0,
    feather: int = 0,
) -> list[Path]:
    """Run all 4 color-space modes and save individual images plus a collage."""
    if sample_point is None and sample_point_rel is None:
        msg = "Either sample_point or sample_point_rel must be specified"
        raise ValueError(msg)

    img = cv2.imread(str(input_path))
    if img is None:
        msg = f"Could not read image: {input_path}"
        raise ValueError(msg)

    h, w = img.shape[:2]

    if sample_point_rel is not None:
        sample_point = rel_to_abs_point(sample_point_rel[0], sample_point_rel[1], w, h)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    saved_paths: list[Path] = []
    result_images: list[np.ndarray] = []

    for cs in COLOR_SPACES:
        out_path = output_dir / f"{stem}_cs_{cs}.png"
        apply_partial_color(
            input_path,
            out_path,
            sample_point=sample_point,
            lab_radius=lab_radius,
            color_space=cs,
            feather=feather,
        )
        saved_paths.append(out_path)
        result_img = cv2.imread(str(out_path))
        result_images.append(result_img)

    # Build collage (4 columns, max 4000px wide)
    max_collage_width = 4000
    col_width = min(w, max_collage_width // 4)
    col_height = round(h * col_width / w) if w > 0 else h

    cols = []
    for ri in result_images:
        resized = cv2.resize(ri, (col_width, col_height))
        cols.append(resized)

    collage = np.concatenate(cols, axis=1)
    collage_path = output_dir / f"{stem}_cs_compare.png"
    cv2.imwrite(str(collage_path), collage)
    saved_paths.append(collage_path)

    return saved_paths
