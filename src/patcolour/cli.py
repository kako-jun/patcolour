"""CLI entry point for patcolour."""

import argparse
import sys
from pathlib import Path

from patcolour.filter import apply_partial_color


def _parse_rect(s: str) -> tuple[int, int, int, int]:
    """Parse 'x,y,w,h' string."""
    parts = s.split(",")
    if len(parts) != 4:
        msg = f"rect must be x,y,w,h: {s}"
        raise argparse.ArgumentTypeError(msg)
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _parse_ellipse(s: str) -> tuple[int, int, int, int]:
    """Parse 'cx,cy,rx,ry' string."""
    parts = s.split(",")
    if len(parts) != 4:
        msg = f"ellipse must be cx,cy,rx,ry: {s}"
        raise argparse.ArgumentTypeError(msg)
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep specific regions in color, rest becomes monochrome",
    )
    parser.add_argument("input", type=Path, help="Input image or directory")
    parser.add_argument("--mask", type=Path, help="Mask image (white=color, black=mono)")
    parser.add_argument("--mask-dir", type=Path, help="Directory of mask images (for batch)")
    parser.add_argument(
        "--rect", type=_parse_rect, action="append", metavar="x,y,w,h",
        help="Rectangle region to keep in color (repeatable)",
    )
    parser.add_argument(
        "--ellipse", type=_parse_ellipse, action="append", metavar="cx,cy,rx,ry",
        help="Ellipse region to keep in color (repeatable)",
    )
    parser.add_argument(
        "--auto-detect", action="store_true",
        help="Auto-detect colorful regions (HSV). Combine with --rect/--ellipse to limit where.",
    )
    parser.add_argument(
        "--feather", type=int, default=0,
        help="Edge softness in pixels (default: 0 = hard edge)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output path")

    args = parser.parse_args()
    has_coords = args.rect or args.ellipse

    if args.input.is_dir():
        if not args.mask_dir:
            print("--mask-dir is required for batch processing", file=sys.stderr)
            sys.exit(1)
        out_dir = args.output or args.input / "patcolour_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [f for f in args.input.iterdir() if f.suffix.lower() in extensions]
        if not files:
            print(f"No image files found in {args.input}", file=sys.stderr)
            sys.exit(1)
        for f in sorted(files):
            mask_path = args.mask_dir / f"{f.stem}.png"
            if not mask_path.exists():
                mask_path = args.mask_dir / f.name
            if not mask_path.exists():
                print(f"  skip {f.name} (no mask found)", file=sys.stderr)
                continue
            out_path = out_dir / f"{f.stem}_patcolour.png"
            apply_partial_color(f, out_path, mask_path=mask_path)
            print(f"{f.name} -> {out_path.name}")
    else:
        if not args.mask and not has_coords and not args.auto_detect:
            print("--mask, --rect/--ellipse, or --auto-detect is required", file=sys.stderr)
            sys.exit(1)
        out_path = args.output or args.input.with_stem(f"{args.input.stem}_patcolour")
        apply_partial_color(
            args.input, out_path,
            mask_path=args.mask,
            rects=args.rect,
            ellipses=args.ellipse,
            feather=args.feather,
            auto_detect=args.auto_detect,
        )
        print(f"{args.input.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
