# imgdiet/cli.py

import argparse
from pathlib import Path
from .core import save

def main():
    parser = argparse.ArgumentParser(
        description="Compress images to WebP, preserving folder structure."
    )
    parser.add_argument("--source", required=True, help="Path to an image or directory.")
    parser.add_argument("--target", required=True, help="Path to an image or directory.")
    parser.add_argument("--psnr", type=float, default=40.0, help="Target PSNR (0=lossless).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    save(
        source=Path(args.source),
        target=Path(args.target),
        target_psnr=args.psnr,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
