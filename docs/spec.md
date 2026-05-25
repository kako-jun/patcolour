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

### `--sample-rel rx,ry`

Relative coordinate variant of `--sample`. Specifies the sample point in 0.0–1.0 coordinates,
making it reusable across different image resolutions.

When both `--sample` and `--sample-rel` are specified, `--sample-rel` takes priority.

### `--rect-rel rx,ry,rw,rh`

Relative coordinate variant of `--rect`. Specifies a rectangular region using 0.0–1.0
coordinates, making it reusable across different image resolutions.

Repeatable.

### `--ellipse-rel rcx,rcy,rrx,rry`

Relative coordinate variant of `--ellipse`. Specifies an elliptical region using 0.0–1.0
coordinates, making it reusable across different image resolutions.

Repeatable.

### `--exclude-rect x,y,w,h`

Adds a rectangular region to explicitly exclude from color output. Always applied with hard edges
regardless of `--feather`.

Repeatable.

### `--exclude-ellipse cx,cy,rx,ry`

Adds an elliptical region to explicitly exclude from color output. Always applied with hard edges
regardless of `--feather`.

Repeatable.

### `--exclude-rect-rel rx,ry,rw,rh`

Relative coordinate variant of `--exclude-rect`. Specifies an exclusion rectangle using 0.0–1.0
coordinates.

Repeatable.

### `--exclude-ellipse-rel rcx,rcy,rrx,rry`

Relative coordinate variant of `--exclude-ellipse`. Specifies an exclusion ellipse using 0.0–1.0
coordinates.

Repeatable.

### `--lab-radius`

Threshold radius for `--sample` mode.

Smaller values are stricter. Larger values include more neighboring hues.

## Combining modes

- `--mask` is exclusive in practice because it directly defines the mask
- `--rect` and `--ellipse` are unioned together
- `--rect-rel` and `--ellipse-rel` follow the same union rule and are merged with `--rect`/`--ellipse`
- `--auto-detect` combined with coordinate regions produces an intersection
- `--sample` combined with coordinate regions also produces an intersection
- `--sample-rel` follows the same intersection rule as `--sample`

Exclude regions override all positive selection. The final mask is computed as:

```
final = positive AND NOT exclude
```

Exclude regions always win over positive/color-distance/region selections, regardless of how the
positive mask was generated (`--mask`, `--rect`, `--ellipse`, `--auto-detect`, `--sample`, or any
combination thereof).

When `--mask` is used, the mask may contain soft/intermediate values (0–255). Exclude regions
still apply with hard edges; pixels inside an exclude region are set to fully monochrome (mask=0)
regardless of the original mask value at that location.

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
