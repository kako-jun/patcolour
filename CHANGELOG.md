# Changelog

## v0.1.0 — Initial release

- **`patcolour` CLI**: keep specific regions in color while the rest becomes monochrome
- **`--mask PATH`**: external grayscale mask image (white = color, black = mono, intermediate = blend)
- **`--rect x,y,w,h` / `--ellipse cx,cy,rx,ry`**: spatial region selection (repeatable)
- **`--rect-rel` / `--ellipse-rel`**: relative coordinate variants (0.0–1.0), resolution-independent
- **`--sample x,y` / `--sample-rel rx,ry`**: reference-pixel color sampling with Lab chroma distance
- **`--color-space {lab-chroma,lab-full,lch,xyy}`**: choose perceptual color-distance metric
  - `lab-chroma` (default): Lab a\*b\* chroma-only, ignores lightness — best for hue families across brightness
  - `lab-full`: full Lab 3D Euclidean distance
  - `lch`: LCh distance with downweighted lightness (weight=0.3) and hue wraparound
  - `xyy`: xyY chromaticity distance, luminance-invariant
- **`--compare-color-space`**: runs all four color-space modes and saves individual images plus a collage
- **`--exclude-rect` / `--exclude-ellipse` / `--exclude-rect-rel` / `--exclude-ellipse-rel`**: negative guides — exclusion always wins over positive selection
- **`--auto-detect`**: HSV-based automatic colorful-region detection
- **`--feather N`**: Gaussian blur for soft mask edges
- **`--lab-radius`**: distance threshold for `--sample` modes (default: 18.0)
- Batch directory mode supported with `--mask-dir`
