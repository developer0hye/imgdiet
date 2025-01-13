# `imgdiet`

**A Python package for minimizing file size of images with negligible quality loss**

| PNG Image                                   | WEBP Image (Optimized by `imgdiet`)                                  |
|--------------------------------------------|--------------------------------------------|
| <img src="./assets/20250105_164724.png" alt="PNG Image" width="300"> | <img src="./assets/20250105_164724.webp" alt="WEBP Image" width="300"> |
| File Size: 26.9 MB                     | File Size : 4.1 MB, Target PSNR: 40 dB |

| JPG Image                                   | WEBP Image (Optimized by `imgdiet`)                                  |
|--------------------------------------------|--------------------------------------------|
| <img src="./assets/xp.jpg" alt="PNG Image" width="300"> | <img src="./assets/xp.webp" alt="WEBP Image" width="300"> |
| File Size: 1.5 MB                     | File Size : 0.3 MB, Target PSNR: 40 dB |

| JPG Image                                   | WEBP Image (Optimized by `imgdiet`)                                  |
|--------------------------------------------|--------------------------------------------|
| <img src="./assets/pexels-wildlittlethingsphoto-1388069.jpg" alt="PNG Image" width="300"> | <img src="./assets/pexels-wildlittlethingsphoto-1388069.webp" alt="WEBP Image" width="300"> |
| File Size: 2.4 MB                     | File Size : 1.9 MB, Target PSNR: 38 dB |

## Installation

To install `imgdiet`, use `pip`:

```bash
pip install imgdiet
```

Alternatively, you can install the package directly from the source repository:

```bash
git clone https://github.com/developer0hye/imgdiet.git
cd imgdiet
pip install .
```

## Features

- Compress images to the WebP format while preserving folder structure.
- Lossless and lossy compression options.
- Automatically optimizes compression level using PSNR (Peak Signal-to-Noise Ratio) targets.
- Supports multi-threaded processing for batch compression.
- Preserves ICC profiles and handles EXIF orientation.
- Supports various input formats including JPG, PNG, BMP, and TIFF.

## Usage

### Command-Line Interface

`imgdiet` provides a convenient CLI for compressing images:

```bash
imgdiet --source <input_path> --target <output_path> [--psnr <value>] [--verbose]
```

#### Arguments:

- `--source`: Path to an image or a directory containing images.
- `--target`: Path to the output directory or a single WebP file.
- `--psnr`: Target Peak Signal-to-Noise Ratio (default: 40.0). Use `0` for lossless compression.
- `--verbose`: Enable detailed logging for the process.

#### Examples:

**Compress a single image:**
```bash
imgdiet --source image.png --target compressed_image.webp --psnr 40.0
```

**Compress a single image and save a compressed version in a directory:**
```bash
imgdiet --source ./images/image.png --target ./compressed_images/ --psnr 40.0
```

**Compress all images in a directory:**
```bash
imgdiet --source ./images --target ./compressed_images --psnr 40.0 --verbose
```

### Python API

You can also use `imgdiet` programmatically in your Python projects:

```python
from imgdiet import save

# Compress a single image
save(
    source="image.png",
    target="compressed_image.webp",
    target_psnr=40.0,
    verbose=True
)

# Compress a single image and save a compressed version in a directory
save(
    source="./images/image.png",
    target="./compressed_images/",
    target_psnr=40.0,
    verbose=True
)

# Compress all images in a directory
save(
    source="./images",
    target="./compressed_images",
    target_psnr=40.0,
    verbose=False
)
```

## How It Works

1. **Lossless Compression:** If the target PSNR is `0`, the image is compressed in lossless WebP format.
2. **Lossy Compression:** If the target PSNR is greater than `0`, the package performs a binary search to find the optimal WebP quality that meets the PSNR target.
3. **PSNR and Size Trade-off:** Higher PSNR values result in better image quality but also larger file sizes. Conversely, lower PSNR values reduce file sizes at the cost of image quality. You can adjust the PSNR to balance quality and file size according to your needs.
4. **ICC Profile Preservation:** The package retains the ICC color profile of the input image for consistent color rendering.
5. **EXIF Handling:** Automatically rotates images based on their EXIF orientation metadata.
6. **Original Image Retention:** If the compressed WebP image is larger than the original, the original image is saved instead to ensure no increase in file size.

## Supported Formats

Input formats:

- JPG / JPEG
- PNG
- BMP
- TIFF / TIF

Output format:

- WebP

## Requirements

- Python 3.8+
- Pillow (for image processing)
- tqdm (for progress bars in multi-threaded tasks)

## Limitations

- The output format is currently limited to WebP.
- ICC profile conversion to sRGB requires the `ImageCms` module, which may not be available in some environments.

## Future Plans

- **AVIF and JPEG XL Support:** We plan to add support for AVIF and JPEG XL formats in future versions. These formats offer superior compression efficiency and image quality. However, their integration requires optional dependencies since they are not natively supported by OpenCV or Pillow at this time. Once these dependencies become more accessible, they will be incorporated into `imgdiet`.

## Contributing

Contributions are welcome! If you'd like to contribute, please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature-name"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

Enjoy compressing your images efficiently with `imgdiet`!

