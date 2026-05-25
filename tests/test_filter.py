from pathlib import Path

import cv2
import numpy as np

from patcolour.filter import (
    apply_partial_color,
    detect_color_mask,
    detect_sample_color_mask,
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
        input_path, output_path,
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
        input_path, output_path,
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
        input_path, output_path,
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
        input_path, output_path,
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
