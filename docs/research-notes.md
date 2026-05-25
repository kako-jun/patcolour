# Research Notes

Short notes gathered while refining `patcolour`.

## Selective color is not only a color problem

In many scenes, "keep the purple flower" does **not** mean "keep all purple pixels".
It means:

- find candidate pixels near the intended hue
- use human guidance to say which purple is the subject
- suppress similar colors in semantically wrong regions

So `patcolour` should be designed as a human-guided selector, not a fully automatic effect.

## Color space direction

For "keep this hue family" workflows, raw RGB distance is the wrong baseline.

Useful candidates:

- Lab full-distance
- Lab chroma-only distance
- LCh hue/chroma comparisons
- xyY chromaticity radius

## Practical OpenCV note

OpenCV supports Lab conversion through `cvtColor`.
When measuring perceptual distance, prefer float conversion paths rather than relying only on
8-bit round-tripped values, because 8-bit conversions lose information.

## Guidance direction

CLI guidance should probably prefer relative coordinates as the main path:

- `sample-rel`
- `rect-rel`
- `ellipse-rel`
- negative variants for exclusion

Why:

- users think in areas like "left half" more often than exact pixels
- the same guide should survive multiple resolutions
- pipeline callers can reuse the same guide contract

## Blob-level selection

After color-distance masking, a connected-component pass is a strong candidate.
Often the intended output is:

- not "all matching pixels"
- but "the blob near this guided subject"

That is why component scoring is a separate issue, not merely an implementation detail.

## Useful real-world input family

The committed night backgrounds from `ear-sky/public/bg/` are also good test inputs.

Why:

- neon scenes produce small but important saturated regions
- distractor colors are common
- dark surroundings make "keep only this accent color" tests meaningful
