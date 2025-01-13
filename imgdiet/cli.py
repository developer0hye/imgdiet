# imgdiet/cli.py

import argparse
from pathlib import Path
from .core import save

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        self.exit(2, f"{self.prog}: error: {message}\n{self.epilog}\n")

def main():
    parser = CustomArgumentParser(
        description="Compress images to WebP, preserving folder structure.",
        epilog=(
            "Example usage:\n"
            "  imgdiet --source <input_path> --target <output_path> [--psnr <value>] [--verbose]\n"
            "  imgdiet --source image.png --target compressed_image.webp --psnr 40.0\n"
            "  imgdiet --source ./images/image.png --target ./compressed_images/ --psnr 40.0\n"
            "  imgdiet --source ./images --target ./compressed_images --psnr 40.0 --verbose"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--source", required=True, help="Path to an image or directory.")
    parser.add_argument("--target", required=True, help="Path to an image or directory.")
    parser.add_argument("--psnr", type=float, default=40.0, help="Target PSNR (0=lossless, higher=better quality, larger size).")
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
