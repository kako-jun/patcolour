# patcolour

[![CI](https://github.com/kako-jun/patcolour/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/patcolour/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/patcolour.svg)](https://pypi.org/project/patcolour/)

Partial color processing for visual-novel style artwork.

`patcolour` keeps selected regions in color while the rest becomes monochrome. It is meant
to sit between raw photo editing and the larger `skirts-colour` production pipeline, and to
be callable from other tools that need stable image-processing behavior.

## Why patcolour?

The target use case is not generic "selective color" photography. The real goal is more
specific: keep emotionally important parts in color while the rest falls back to memory-like
grayscale. That makes mask quality, edge softness, and region selection workflow more
important than a huge option surface.

Just as importantly, this tool is expected to become reusable infrastructure for other repos,
including `name-name`-based `skirts-colour` workflows and future image-processing utilities.

## Install

```bash
uv tool install patcolour
```

Or from source:

```bash
git clone https://github.com/kako-jun/patcolour.git
cd patcolour
uv sync --group dev
uv run patcolour --help
```

## Usage

### Mask-based

```bash
patcolour photo.jpg --mask mask.png -o output.png
```

White areas in the mask stay in color. Black areas become monochrome.

### Coordinate-based

```bash
patcolour photo.jpg \
  --rect 120,40,220,180 \
  --ellipse 540,260,110,80 \
  --feather 12 \
  -o output.png
```

### Auto-detect colorful regions

```bash
patcolour photo.jpg --auto-detect -o output.png
```

### Sample a target color from the image

```bash
patcolour photo.jpg --sample 420,180 --lab-radius 16 -o output.png
```

This mode samples one pixel from the input and keeps nearby colors using Lab chroma distance,
which is a better fit than raw RGB distance for "keep this hue family" workflows.

### Keep only the front flower, not every similar color in the scene

```bash
patcolour photo.jpg \
  --sample 420,180 \
  --lab-radius 16 \
  --ellipse 420,180,90,120 \
  --feather 10 \
  -o output.png
```

Color alone is not enough for many scenes. In practice, the user often needs to combine a
color sample with a spatial guide.

### Batch processing

```bash
patcolour ./photos/ --mask-dir ./masks/ -o ./output/
```

## Current behavior

- `--mask` uses an external grayscale mask
- `--rect` and `--ellipse` define keep-color regions
- `--auto-detect` keeps broad HSV-detected colorful areas
- `--sample` keeps colors near a sampled reference point in Lab chroma space
- `--feather` softens coordinate-based region edges

The CLI is the first interface, but not the only intended interface. The filter logic should
stay predictable enough to be reused from other tools.

Just as importantly, this is a human-guided tool. If the scene contains several similar
purples, reds, or greens with different semantic roles, the user must be able to tell the
tool which region is the intended subject.

If `--auto-detect` is combined with coordinate regions, the result is their intersection.

## Limitations

- Auto-detection is currently tuned for broad colorful regions and is easy to over-select
- There is no semantic selection yet such as "left flower only" or "girl's ribbon only"
- Batch mode currently assumes one mask per file and does not validate coverage quality

## Docs

- `docs/overview.md` — positioning, goals, non-goals
- `docs/spec.md` — CLI and processing behavior
- `docs/roadmap.md` — internal roadmap and quality targets
- `CLAUDE.md` — AI-facing internal guide

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
```
