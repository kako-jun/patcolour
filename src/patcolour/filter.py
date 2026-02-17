"""Partial color filter - keep masked regions in color, rest in monochrome."""

from pathlib import Path

import cv2
import numpy as np


def apply_partial_color(
    input_path: Path,
    mask_path: Path,
    output_path: Path,
) -> None:
    """Apply partial color effect using a mask.

    White regions in the mask retain their original color.
    Black regions become monochrome.
    Gray regions blend between color and monochrome.
    """
    img = cv2.imread(str(input_path))
    if img is None:
        msg = f"Could not read image: {input_path}"
        raise ValueError(msg)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        msg = f"Could not read mask: {mask_path}"
        raise ValueError(msg)

    # Resize mask to match image if needed
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Create monochrome version
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mono = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Blend based on mask (0=mono, 255=color)
    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    result = (img.astype(np.float32) * alpha + mono.astype(np.float32) * (1.0 - alpha))
    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(str(output_path), result)
