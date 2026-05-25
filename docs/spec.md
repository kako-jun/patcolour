# patcolour Spec

## Command

```bash
patcolour INPUT [OPTIONS]
```

`INPUT` may be a single image file or a directory.

Supported extensions in directory mode:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

## Output behavior

- single file mode: defaults to `<stem>_patcolour<suffix>`
- directory mode: defaults to `<input>/patcolour_out/`

## Region selection modes

### `--mask PATH`

Uses an external grayscale image.

- white = keep original color
- black = convert to monochrome
- intermediate values = blend between original and monochrome

If mask size does not match input size, the mask is resized.

### `--rect x,y,w,h`

Adds a rectangular region to keep in color.

Repeatable.

### `--ellipse cx,cy,rx,ry`

Adds an elliptical region to keep in color.

Repeatable.

### `--auto-detect`

Detects colorful regions through HSV thresholding and morphology.

Current default heuristic is broad and biased toward greenish or naturally saturated areas.

### `--sample x,y`

Samples a reference pixel from the input image and keeps nearby colors in Lab chroma space.

This is the current best-fit mode for "keep this hue family, even if brightness changes."

### `--lab-radius`

Threshold radius for `--sample` mode.

Smaller values are stricter. Larger values include more neighboring hues.

## Combining modes

- `--mask` is exclusive in practice because it directly defines the mask
- `--rect` and `--ellipse` are unioned together
- `--auto-detect` combined with coordinate regions produces an intersection
- `--sample` combined with coordinate regions also produces an intersection

This combination is intentional. A color-based candidate set often needs human spatial
guidance to become the intended subject.

## Feathering

`--feather N` applies Gaussian blur to coordinate-generated masks.

- `0` means hard edge
- larger values soften the transition

## Failure behavior

- directory mode without `--mask-dir` exits with status 1
- single-file mode without any selection mode exits with status 1
- unreadable input or mask raises an error
