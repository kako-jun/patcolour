from pathlib import Path

import cv2
import numpy as np
import pytest

from patcolour.filter import (
    COLOR_SPACES,
    apply_color_space_comparison,
    apply_partial_color,
    detect_color_mask,
    detect_sample_color_mask,
    detect_sample_color_mask_by_space,
    detect_sample_color_mask_lab_full,
    detect_sample_color_mask_lch,
    detect_sample_color_mask_xyy,
    generate_exclude_mask,
    generate_region_mask,
    rel_to_abs_ellipse,
    rel_to_abs_point,
    rel_to_abs_rect,
)


def _write_image(path: Path, img: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), img)
    assert ok


def test_generate_region_mask_rect() -> None:
    mask = generate_region_mask(8, 8, rects=[(2, 2, 3, 3)])

    assert mask[3, 3] == 255
    assert mask[0, 0] == 0


def test_detect_color_mask_finds_green_patch() -> None:
    img = np.zeros((24, 24, 3), dtype=np.uint8)
    img[8:16, 8:16] = (0, 255, 0)

    mask = detect_color_mask(img)

    assert mask[12, 12] > 0


def test_apply_partial_color_keeps_masked_pixel_colored(tmp_path: Path) -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :] = (10, 40, 220)
    img[1, 1] = (0, 255, 0)

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 255

    input_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)
    _write_image(mask_path, mask)

    apply_partial_color(input_path, output_path, mask_path=mask_path)
    result = cv2.imread(str(output_path))

    assert result is not None
    assert tuple(int(v) for v in result[1, 1]) == (0, 255, 0)
    assert result[0, 0, 0] == result[0, 0, 1] == result[0, 0, 2]


def test_detect_sample_color_mask_matches_similar_hue() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :] = (80, 80, 80)
    img[1, 1] = (0, 255, 0)
    img[1, 2] = (10, 220, 10)

    mask = detect_sample_color_mask(img, (1, 1), lab_radius=20.0)

    assert mask[1, 1] == 255
    assert mask[1, 2] == 255
    assert mask[0, 0] == 0


# ---------------------------------------------------------------------------
# rel_to_abs_point / rel_to_abs_rect / rel_to_abs_ellipse unit tests
# ---------------------------------------------------------------------------


def testrel_to_abs_point_center() -> None:
    assert rel_to_abs_point(0.5, 0.5, 100, 200) == (50, 100)


def testrel_to_abs_rect_quarter() -> None:
    assert rel_to_abs_rect(0.25, 0.25, 0.5, 0.5, 100, 200) == (25, 50, 50, 100)


def testrel_to_abs_ellipse_half() -> None:
    assert rel_to_abs_ellipse(0.5, 0.5, 0.25, 0.25, 100, 200) == (50, 100, 25, 50)


def testrel_to_abs_point_origin() -> None:
    assert rel_to_abs_point(0.0, 0.0, 100, 200) == (0, 0)


def testrel_to_abs_point_max() -> None:
    assert rel_to_abs_point(1.0, 1.0, 100, 200) == (100, 200)


def testrel_to_abs_rect_full_image() -> None:
    result = rel_to_abs_rect(0.0, 0.0, 1.0, 1.0, 80, 60)
    assert result == (0, 0, 80, 60)


def testrel_to_abs_point_rounding() -> None:
    # 1/3 of 10 = 3.333... → rounds to 3
    x, y = rel_to_abs_point(1 / 3, 2 / 3, 10, 10)
    assert x == round(1 / 3 * 10)
    assert y == round(2 / 3 * 10)


def testrel_to_abs_point_resolution_independent() -> None:
    x1, y1 = rel_to_abs_point(0.5, 0.5, 100, 100)
    x2, y2 = rel_to_abs_point(0.5, 0.5, 200, 200)
    assert x1 / 100 == x2 / 200
    assert y1 / 100 == y2 / 200


def testrel_to_abs_rect_non_square_image() -> None:
    # rx and ry must scale independently
    x, y, w, h = rel_to_abs_rect(0.5, 0.5, 0.5, 0.5, 100, 50)
    assert x == 50
    assert y == 25
    assert w == 50
    assert h == 25


# ---------------------------------------------------------------------------
# apply_partial_color — rel argument integration tests
# ---------------------------------------------------------------------------


def _solid_color_image(height: int, width: int, bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def test_apply_partial_color_rect_rel_colors_region(tmp_path: Path) -> None:
    # Image is solid red; center 50% rect should remain red in output.
    img = _solid_color_image(100, 100, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_partial_color(input_path, output_path, rects_rel=[(0.25, 0.25, 0.5, 0.5)])
    result = cv2.imread(str(output_path))

    assert result is not None
    center_pixel = result[50, 50]
    # Center pixel should be colored (not gray), so R channel ≠ G channel
    assert int(center_pixel[2]) != int(center_pixel[1])


def test_apply_partial_color_ellipse_rel_colors_region(tmp_path: Path) -> None:
    img = _solid_color_image(100, 100, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_partial_color(input_path, output_path, ellipses_rel=[(0.5, 0.5, 0.25, 0.25)])
    result = cv2.imread(str(output_path))

    assert result is not None
    center_pixel = result[50, 50]
    assert int(center_pixel[2]) != int(center_pixel[1])


def test_apply_partial_color_sample_rel_keeps_similar_hue(tmp_path: Path) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (80, 80, 80)
    # Place a vivid green patch
    img[40:60, 40:60] = (0, 200, 0)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    # sample from center of green patch via rel
    apply_partial_color(input_path, output_path, sample_point_rel=(0.5, 0.5), lab_radius=30.0)
    result = cv2.imread(str(output_path))

    assert result is not None
    # Center of green patch should stay green (G > R)
    assert int(result[50, 50, 1]) > int(result[50, 50, 2])


def test_apply_partial_color_rel_none_does_not_change_behavior(tmp_path: Path) -> None:
    img = _solid_color_image(10, 10, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    # abs rect covers top-left 5×5
    apply_partial_color(
        input_path,
        output_path,
        rects=[(0, 0, 5, 5)],
        rects_rel=None,
        ellipses_rel=None,
        sample_point_rel=None,
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    # top-left corner colored
    assert int(result[2, 2, 2]) != int(result[2, 2, 1])
    # bottom-right corner gray
    assert result[8, 8, 0] == result[8, 8, 1] == result[8, 8, 2]


def test_apply_partial_color_rect_rel_and_abs_are_unioned(tmp_path: Path) -> None:
    img = _solid_color_image(100, 100, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    # abs covers top-left, rel covers bottom-right
    apply_partial_color(
        input_path,
        output_path,
        rects=[(0, 0, 20, 20)],
        rects_rel=[(0.75, 0.75, 0.25, 0.25)],
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    # top-left region colored
    assert int(result[10, 10, 2]) != int(result[10, 10, 1])
    # bottom-right region colored
    assert int(result[90, 90, 2]) != int(result[90, 90, 1])
    # middle region gray
    assert result[50, 50, 0] == result[50, 50, 1] == result[50, 50, 2]


def test_apply_partial_color_ellipse_rel_and_abs_are_unioned(tmp_path: Path) -> None:
    img = _solid_color_image(100, 100, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    # abs ellipse center top-left; rel ellipse center bottom-right
    apply_partial_color(
        input_path,
        output_path,
        ellipses=[(10, 10, 8, 8)],
        ellipses_rel=[(0.9, 0.9, 0.08, 0.08)],
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    assert int(result[10, 10, 2]) != int(result[10, 10, 1])
    assert int(result[90, 90, 2]) != int(result[90, 90, 1])


def test_apply_partial_color_rects_rel_does_not_mutate_original_rects_arg(tmp_path: Path) -> None:
    img = _solid_color_image(100, 100, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    original = [(0, 0, 50, 50)]
    rects_copy = list(original)
    apply_partial_color(input_path, output_path, rects=rects_copy, rects_rel=[(0.5, 0.5, 0.5, 0.5)])

    assert rects_copy == original


def test_apply_partial_color_sample_rel_overrides_sample_point(tmp_path: Path) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (80, 80, 80)
    # Green at center
    img[45:55, 45:55] = (0, 200, 0)
    # Red at top-left corner (will be passed as abs sample_point)
    img[5, 5] = (0, 0, 200)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    # sample_point_rel points to center green; abs sample_point points to red corner
    apply_partial_color(
        input_path,
        output_path,
        sample_point=(5, 5),
        sample_point_rel=(0.5, 0.5),
        lab_radius=30.0,
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    # Center (green) should be colored, so G channel dominant
    assert int(result[50, 50, 1]) > int(result[50, 50, 2])


def test_apply_partial_color_rect_rel_same_region_across_resolutions(tmp_path: Path) -> None:
    for size in (50, 100):
        img = _solid_color_image(size, size, (0, 0, 200))
        input_path = tmp_path / f"input_{size}.png"
        output_path = tmp_path / f"output_{size}.png"
        _write_image(input_path, img)
        apply_partial_color(input_path, output_path, rects_rel=[(0.25, 0.25, 0.5, 0.5)])
        result = cv2.imread(str(output_path))
        assert result is not None
        cx = size // 2
        center_pixel = result[cx, cx]
        assert int(center_pixel[2]) != int(center_pixel[1]), f"Failed for size={size}"


# ---------------------------------------------------------------------------
# generate_exclude_mask tests
# ---------------------------------------------------------------------------


def test_generate_exclude_mask_rect() -> None:
    mask = generate_exclude_mask(10, 10, rects=[(2, 2, 4, 4)])

    # Interior of rect should be excluded (255)
    assert mask[4, 4] == 255
    # Outside rect should be zero
    assert mask[0, 0] == 0
    assert mask[9, 9] == 0


def test_generate_exclude_mask_ellipse() -> None:
    mask = generate_exclude_mask(20, 20, ellipses=[(10, 10, 4, 4)])

    # Center of ellipse should be excluded
    assert mask[10, 10] == 255
    # Far corner should not be excluded
    assert mask[0, 0] == 0


def test_generate_exclude_mask_empty() -> None:
    mask = generate_exclude_mask(8, 8, rects=None, ellipses=None)

    assert mask.shape == (8, 8)
    assert np.all(mask == 0)


# ---------------------------------------------------------------------------
# apply_partial_color — exclude argument tests
# ---------------------------------------------------------------------------


def test_apply_partial_color_exclude_rect_suppresses_color(tmp_path: Path) -> None:
    # Solid red image; positive rect covers whole image; exclude rect covers center.
    img = _solid_color_image(10, 10, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    # positive: whole image; exclude: center 4x4
    apply_partial_color(
        input_path,
        output_path,
        rects=[(0, 0, 10, 10)],
        exclude_rects=[(3, 3, 4, 4)],
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    # Excluded center pixel → monochrome (all channels equal)
    assert result[5, 5, 0] == result[5, 5, 1] == result[5, 5, 2]
    # Non-excluded corner → colored (R channel > G/B for red image)
    assert int(result[0, 0, 2]) != int(result[0, 0, 1])


def test_apply_partial_color_exclude_ellipse_suppresses_color(tmp_path: Path) -> None:
    img = _solid_color_image(20, 20, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_partial_color(
        input_path,
        output_path,
        rects=[(0, 0, 20, 20)],
        exclude_ellipses=[(10, 10, 4, 4)],
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    # Center of ellipse → monochrome
    assert result[10, 10, 0] == result[10, 10, 1] == result[10, 10, 2]
    # Outside ellipse → still colored
    assert int(result[0, 0, 2]) != int(result[0, 0, 1])


def test_apply_partial_color_exclude_rect_rel_suppresses_color(tmp_path: Path) -> None:
    img = _solid_color_image(100, 100, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    # positive: full image; exclude_rects_rel: center quarter
    apply_partial_color(
        input_path,
        output_path,
        rects=[(0, 0, 100, 100)],
        exclude_rects_rel=[(0.25, 0.25, 0.5, 0.5)],
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    # Center (inside excluded rel rect) → monochrome
    assert result[50, 50, 0] == result[50, 50, 1] == result[50, 50, 2]
    # Corner (outside excluded) → colored
    assert int(result[0, 0, 2]) != int(result[0, 0, 1])


def test_apply_partial_color_exclude_does_not_affect_outside_positive(tmp_path: Path) -> None:
    # positive: top-left 5x5 only; exclude: bottom-right corner (outside positive).
    # The bottom-right should be monochrome regardless (it's outside positive already).
    img = _solid_color_image(10, 10, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_partial_color(
        input_path,
        output_path,
        rects=[(0, 0, 5, 5)],
        exclude_rects=[(7, 7, 3, 3)],
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    # Bottom-right (outside positive) → monochrome, same as without exclude
    assert result[9, 9, 0] == result[9, 9, 1] == result[9, 9, 2]
    # Top-left (inside positive, not excluded) → colored
    assert int(result[2, 2, 2]) != int(result[2, 2, 1])


def test_apply_partial_color_exclude_none_does_not_change_behavior(tmp_path: Path) -> None:
    img = _solid_color_image(10, 10, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path_with_none = tmp_path / "output_none.png"
    output_path_without = tmp_path / "output_without.png"
    _write_image(input_path, img)

    apply_partial_color(
        input_path,
        output_path_with_none,
        rects=[(0, 0, 5, 5)],
        exclude_rects=None,
        exclude_ellipses=None,
        exclude_rects_rel=None,
        exclude_ellipses_rel=None,
    )
    apply_partial_color(
        input_path,
        output_path_without,
        rects=[(0, 0, 5, 5)],
    )

    result_none = cv2.imread(str(output_path_with_none))
    result_without = cv2.imread(str(output_path_without))

    assert result_none is not None
    assert result_without is not None
    np.testing.assert_array_equal(result_none, result_without)


def test_apply_partial_color_exclude_wins_over_positive_at_overlap(tmp_path: Path) -> None:
    # positive and exclude both cover the entire image → exclude wins → full monochrome
    img = _solid_color_image(10, 10, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_partial_color(
        input_path,
        output_path,
        rects=[(0, 0, 10, 10)],
        exclude_rects=[(0, 0, 10, 10)],
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    # All pixels must be monochrome
    for y in (0, 5, 9):
        for x in (0, 5, 9):
            assert result[y, x, 0] == result[y, x, 1] == result[y, x, 2], (
                f"pixel [{y},{x}] is not monochrome: {result[y, x]}"
            )


def test_apply_partial_color_exclude_rel_and_abs_are_unioned(tmp_path: Path) -> None:
    # positive: full image; exclude_rects: top-left 3x3; exclude_rects_rel: bottom-right corner.
    img = _solid_color_image(10, 10, (0, 0, 200))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    apply_partial_color(
        input_path,
        output_path,
        rects=[(0, 0, 10, 10)],
        exclude_rects=[(0, 0, 3, 3)],
        exclude_rects_rel=[(0.7, 0.7, 0.3, 0.3)],
    )
    result = cv2.imread(str(output_path))

    assert result is not None
    # Top-left corner (abs exclude) → monochrome
    assert result[1, 1, 0] == result[1, 1, 1] == result[1, 1, 2]
    # Bottom-right corner (rel exclude) → monochrome
    assert result[9, 9, 0] == result[9, 9, 1] == result[9, 9, 2]
    # Middle (not excluded) → colored
    assert int(result[5, 5, 2]) != int(result[5, 5, 1])


# ---------------------------------------------------------------------------
# detect_sample_color_mask_lab_full
# ---------------------------------------------------------------------------


def _make_color_image(color_bgr: tuple[int, int, int]) -> np.ndarray:
    """50x50 image filled with color_bgr with a small colored rectangle."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def test_detect_sample_color_mask_lab_full_returns_mask() -> None:
    img = _make_color_image((0, 200, 0))
    mask = detect_sample_color_mask_lab_full(img, (25, 25))

    assert mask.dtype == np.uint8
    assert mask.shape == img.shape[:2]


def test_detect_sample_color_mask_lab_full_range() -> None:
    img = _make_color_image((0, 200, 0))
    img[10:20, 10:20] = (0, 0, 200)
    mask = detect_sample_color_mask_lab_full(img, (25, 25))

    unique = np.unique(mask)
    assert set(unique.tolist()).issubset({0, 255})


def test_detect_sample_color_mask_lab_full_selects_similar_color() -> None:
    img = _make_color_image((0, 200, 0))
    mask = detect_sample_color_mask_lab_full(img, (25, 25), lab_radius=5.0)

    # Solid same color → entire image should be selected
    assert np.all(mask == 255)


# ---------------------------------------------------------------------------
# detect_sample_color_mask_lch
# ---------------------------------------------------------------------------


def test_detect_sample_color_mask_lch_returns_mask() -> None:
    img = _make_color_image((0, 200, 0))
    mask = detect_sample_color_mask_lch(img, (25, 25))

    assert mask.dtype == np.uint8
    assert mask.shape == img.shape[:2]


def test_detect_sample_color_mask_lch_selects_same_hue_different_brightness() -> None:
    # Same green hue, two brightness levels
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[0:25, :] = (0, 200, 0)   # bright green
    img[25:, :] = (0, 150, 0)    # slightly darker green (same hue)

    # Sample from bright green region; with large radius, darker green should be selected
    mask = detect_sample_color_mask_lch(img, (25, 12), lch_radius=50.0, lightness_weight=0.3)

    # With low lightness_weight and large radius, darker green should also be selected
    assert mask[37, 25] == 255


# ---------------------------------------------------------------------------
# detect_sample_color_mask_xyy
# ---------------------------------------------------------------------------


def test_detect_sample_color_mask_xyy_returns_mask() -> None:
    img = _make_color_image((0, 200, 0))
    mask = detect_sample_color_mask_xyy(img, (25, 25))

    assert mask.dtype == np.uint8
    assert mask.shape == img.shape[:2]


def test_detect_sample_color_mask_xyy_range() -> None:
    img = _make_color_image((0, 200, 0))
    img[10:20, 10:20] = (0, 0, 200)
    mask = detect_sample_color_mask_xyy(img, (25, 25))

    unique = np.unique(mask)
    assert set(unique.tolist()).issubset({0, 255})


def test_detect_sample_color_mask_xyy_zero_division_safe() -> None:
    # Black pixel (0,0,0) causes total=0 in XYZ; must not crash
    img = _make_color_image((0, 200, 0))
    img[0, 0] = (0, 0, 0)

    # Should not raise
    mask = detect_sample_color_mask_xyy(img, (25, 25))

    assert mask is not None


# ---------------------------------------------------------------------------
# detect_sample_color_mask_by_space
# ---------------------------------------------------------------------------


def test_detect_sample_color_mask_by_space_lab_chroma() -> None:
    img = _make_color_image((0, 200, 0))
    expected = detect_sample_color_mask(img, (25, 25), lab_radius=18.0)
    result = detect_sample_color_mask_by_space(img, (25, 25), color_space="lab-chroma")

    np.testing.assert_array_equal(result, expected)


def test_detect_sample_color_mask_by_space_lab_full() -> None:
    img = _make_color_image((0, 200, 0))
    mask = detect_sample_color_mask_by_space(img, (25, 25), color_space="lab-full")

    assert mask.dtype == np.uint8
    assert mask.shape == img.shape[:2]


def test_detect_sample_color_mask_by_space_lch() -> None:
    img = _make_color_image((0, 200, 0))
    mask = detect_sample_color_mask_by_space(img, (25, 25), color_space="lch")

    assert mask.dtype == np.uint8
    assert mask.shape == img.shape[:2]


def test_detect_sample_color_mask_by_space_xyy() -> None:
    img = _make_color_image((0, 200, 0))
    mask = detect_sample_color_mask_by_space(img, (25, 25), color_space="xyy")

    assert mask.dtype == np.uint8
    assert mask.shape == img.shape[:2]


def test_detect_sample_color_mask_by_space_invalid_raises() -> None:
    img = _make_color_image((0, 200, 0))

    with pytest.raises(ValueError, match="Unknown color_space"):
        detect_sample_color_mask_by_space(img, (25, 25), color_space="unknown-mode")


# ---------------------------------------------------------------------------
# apply_partial_color — color_space argument
# ---------------------------------------------------------------------------


def test_apply_partial_color_color_space_default_is_lab_chroma(tmp_path: Path) -> None:
    img = _make_color_image((0, 200, 0))
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    _write_image(input_path, img)

    # No color_space specified → should use default "lab-chroma" without error
    apply_partial_color(input_path, output_path, sample_point=(25, 25))

    assert output_path.exists()


def test_apply_partial_color_color_space_all_modes_produce_output(tmp_path: Path) -> None:
    img = _make_color_image((0, 200, 0))
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    for cs in COLOR_SPACES:
        output_path = tmp_path / f"output_{cs}.png"
        apply_partial_color(input_path, output_path, sample_point=(25, 25), color_space=cs)
        assert output_path.exists(), f"Output missing for color_space={cs!r}"


# ---------------------------------------------------------------------------
# apply_color_space_comparison
# ---------------------------------------------------------------------------


def test_apply_color_space_comparison_returns_five_paths(tmp_path: Path) -> None:
    img = _make_color_image((0, 200, 0))
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_color_space_comparison(input_path, tmp_path / "out", sample_point=(25, 25))

    assert len(paths) == 5


def test_apply_color_space_comparison_all_files_exist(tmp_path: Path) -> None:
    img = _make_color_image((0, 200, 0))
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_color_space_comparison(input_path, tmp_path / "out", sample_point=(25, 25))

    for p in paths:
        assert p.exists(), f"File missing: {p}"


def test_apply_color_space_comparison_collage_name(tmp_path: Path) -> None:
    img = _make_color_image((0, 200, 0))
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_color_space_comparison(input_path, tmp_path / "out", sample_point=(25, 25))

    collage = paths[-1]
    assert collage.name == "input_cs_compare.png"


def test_apply_color_space_comparison_no_sample_raises(tmp_path: Path) -> None:
    img = _make_color_image((0, 200, 0))
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    with pytest.raises(ValueError, match="sample_point"):
        apply_color_space_comparison(input_path, tmp_path / "out")


def test_apply_color_space_comparison_collage_width_le_4000(tmp_path: Path) -> None:
    img = _make_color_image((0, 200, 0))
    input_path = tmp_path / "input.png"
    _write_image(input_path, img)

    paths = apply_color_space_comparison(input_path, tmp_path / "out", sample_point=(25, 25))

    collage = cv2.imread(str(paths[-1]))
    assert collage is not None
    assert collage.shape[1] <= 4000
