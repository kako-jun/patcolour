from pathlib import Path

import cv2
import numpy as np

from patcolour.filter import (
    apply_partial_color,
    detect_color_mask,
    detect_sample_color_mask,
    generate_region_mask,
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
