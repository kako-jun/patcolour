# patcolour

Partial color processing — keep specific regions in color while the rest becomes monochrome.

## Install

```bash
uv tool install patcolour
```

## Usage

```bash
# Apply mask: colored region stays, rest becomes monochrome
patcolour photo.jpg --mask mask.png -o output.png

# Batch processing
patcolour ./photos/ --mask-dir ./masks/ -o ./output/
```
