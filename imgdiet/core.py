import time
import io
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import pillow_avif
from typing import Literal
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from imgdiet.utils import setup_logger, copy_original, calculate_psnr

try:
    from PIL import ImageCms
except ImportError:
    ImageCms = None

def measure_webp_pil(
    original_bgr: np.ndarray,
    pil_image: Image.Image,
    *,
    quality: Optional[int] = None,
    lossless: bool = False
) -> Tuple[float, int, bytes]:
    """
    Compresses the given PIL Image to WebP.
    - If `lossless` is True, performs lossless compression.
    - Otherwise, uses quality-based compression with the provided `quality`.
    
    Returns a tuple (psnr, compressed_size, compressed_data).
    """
    if not lossless and quality is None:
        raise ValueError("Either 'lossless' must be True or 'quality' must be provided for quality-based compression.")
    
    buffer = io.BytesIO()
    icc_profile = pil_image.info.get("icc_profile")
    
    # Set up parameters for saving
    save_kwargs = {
        "format": "WEBP",
        "icc_profile": icc_profile,
        "exact": True,
    }
    if lossless:
        save_kwargs["lossless"] = True
    else:
        save_kwargs["quality"] = quality

    pil_image.save(buffer, **save_kwargs)
    data = buffer.getvalue()
    size = len(data)

    buffer.seek(0)
    compressed_pil = Image.open(buffer)
    if compressed_pil.mode == 'RGBA':
        compressed_pil = compressed_pil.convert('RGB')
    compressed_bgr = np.array(compressed_pil)[:, :, ::-1]

    psnr_val = calculate_psnr(original_bgr, compressed_bgr)
    return psnr_val, size, data


def find_optimal_compression_binary_search(
    original_bgr: np.ndarray,
    pil_image: Image.Image,
    target_psnr: float = 40.0
) -> Optional[Dict[str, Union[int, float]]]:
    """
    Binary search to find a WebP quality that meets or exceeds target_psnr.
    Returns a dict { 'quality': int, 'psnr': float, 'size': int } or None.
    """
    left, right = 1, 100
    best_quality = None
    best_size = float("inf")
    best_psnr = 0.0

    while left <= right:
        mid = (left + right) // 2
        psnr_val, size, _ = measure_webp_pil(original_bgr, pil_image, quality=mid, lossless=False)

        if psnr_val >= target_psnr:
            if size < best_size:
                best_size = size
                best_quality = mid
                best_psnr = psnr_val
            right = mid - 1
        else:
            left = mid + 1

    if best_quality is None:
        return None

    return {
        "quality": best_quality,
        "psnr": best_psnr,
        "size": int(best_size)
    }


def process_single_image(
    img_path: Path,
    source_root: Path,
    target_dir: Path,
    target_psnr: float,
    verbose: bool
) -> Path:
    """
    Compress a single image to WebP under target_psnr rules.
    Returns the path of the saved file.
    """
    logger = setup_logger(verbose)
    
    try:
        # 1) Open image with context manager
        with Image.open(img_path) as pil_image:
            # Process EXIF rotation
            pil_image = ImageOps.exif_transpose(pil_image)
            
            # 알파 채널 확인
            has_alpha = pil_image.mode in ('RGBA', 'LA')
            
            # RGB 또는 RGBA 모드로 변환 (알파 채널 보존)
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA' if 'transparency' in pil_image.info else 'RGB')
            elif pil_image.mode not in ('RGB', 'RGBA'):
                pil_image = pil_image.convert('RGB')
            else:
                pil_image = pil_image.copy()

            # ICC 프로파일 변환을 여기로 이동
            if ImageCms is not None:
                icc_profile_bytes = pil_image.info.get("icc_profile", None)
                if icc_profile_bytes:
                    try:
                        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_bytes))
                        srgb_profile = ImageCms.createProfile("sRGB")
                        transform = ImageCms.buildTransform(
                            input_profile,
                            srgb_profile,
                            "RGB",
                            "RGB"
                        )
                        pil_image = ImageCms.applyTransform(pil_image, transform)
                        pil_image.info["icc_profile"] = srgb_profile.tobytes()
                    except Exception as e:
                        logger.warning(f"Failed to convert ICC profile: {e}")
            else:
                logger.warning("ImageCms module not available, skipping ICC conversion")

        # 2) PSNR 계산을 위해 알파 채널 제외하고 BGR로 변환
        if has_alpha:
            original_bgr = np.array(pil_image.convert('RGB'))[:, :, ::-1]
        else:
            original_bgr = np.array(pil_image)[:, :, ::-1]

        original_size = img_path.stat().st_size

        # 3) Keep folder structure
        if source_root.is_file():
            rel_path = img_path.relative_to(source_root.parent)
        else:
            rel_path = img_path.relative_to(source_root)

        webp_path = target_dir / rel_path.with_suffix(".webp")

        # Case 1: target_psnr == 0 => lossless
        if target_psnr == 0:
            try:
                psnr_val, compressed_size, data = measure_webp_pil(original_bgr, pil_image, lossless=True)
                if psnr_val == float("inf") and compressed_size < original_size:
                    webp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(webp_path, "wb") as f:
                        icc_profile = pil_image.info.get("icc_profile", None)
                        # lossless 모드로 저장 (원본 모드 유지)
                        pil_image.save(
                            f, 
                            format="WEBP", 
                            lossless=True, 
                            icc_profile=icc_profile,
                            exact=True  # 알파 채널의 정확한 보존을 위해 추가
                        )
                    saving_ratio = (1 - compressed_size / original_size) * 100
                    logger.info(f"Lossless WebP saved for {img_path}")
                    logger.info(f"PSNR: {psnr_val:.2f} dB")
                    logger.info(f"Size: {original_size:,} -> {compressed_size:,} bytes")
                    logger.info(f"Saved: {saving_ratio:.1f}%")
                    return webp_path
                else:
                    logger.warning(f"Lossless compression failed: output is not identical or larger")
                    logger.warning(f"Original size: {original_size:,} bytes")
                    logger.warning(f"Lossless WebP size: {compressed_size:,} bytes")
                    return copy_original(img_path, webp_path.with_suffix(img_path.suffix), verbose)
            except Exception as e:
                logger.warning(f"Failed lossless: {e}, copying original.")
                return copy_original(img_path, webp_path.with_suffix(img_path.suffix), verbose)
        
        # Case 2: target_psnr > 0 => binary search
        best_params = find_optimal_compression_binary_search(original_bgr, pil_image, target_psnr)
        if best_params is None:
            logger.warning(f"No quality meets {target_psnr} dB, copying original.")
            logger.warning(f"Original size: {original_size:,} bytes")
            return copy_original(img_path, webp_path.with_suffix(img_path.suffix), verbose)
        else:
            q = best_params["quality"]
            logger.info(f"Found best quality={q} for {img_path}")
            psnr_val, compressed_size, _ = measure_webp_pil(original_bgr, pil_image, quality=q, lossless=False)
            if compressed_size < original_size:
                webp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(webp_path, "wb") as f:

                    icc_profile = pil_image.info.get("icc_profile", None)
                    # quality 모드로 저장 (원본 모드 유지)
                    pil_image.save(
                        f, 
                        format="WEBP", 
                        quality=q, 
                        icc_profile=icc_profile,
                        exact=True  # 알파 채널의 정확한 보존을 위해 추가
                    )
                saving_ratio = (1 - compressed_size / original_size) * 100
                logger.info(f"WebP saved: {img_path} -> {webp_path}")
                logger.info(f"PSNR: {psnr_val:.2f} dB")
                logger.info(f"Size: {original_size:,} -> {compressed_size:,} bytes")
                logger.info(f"Saved: {saving_ratio:.1f}%")
                return webp_path
            else:
                logger.warning(f"Compressed >= original, copying original.")
                logger.warning(f"Original size: {original_size:,} bytes")
                logger.warning(f"Compressed size: {compressed_size:,} bytes")
                return copy_original(img_path, webp_path.with_suffix(img_path.suffix), verbose)
    except (UnidentifiedImageError, OSError) as e:
        logger.error(f"Failed to open image {img_path}: {str(e)}")
        return copy_original(img_path, 
                           target_dir / img_path.relative_to(source_root), 
                           verbose)


def save(
    source: Union[str, Path],
    target: Union[str, Path],
    codec: Literal["webp", "avif"] = "webp",
    target_psnr: float = 40.0,
    verbose: bool = False
) -> Tuple[list[Path], list[Path]]:
    """
    Main entry: compress images to WebP, preserving folder structure, 
    with a target PSNR or lossless if target_psnr=0. 
    Returns a tuple of (source_paths, target_paths).
    """
    
    if codec not in ['webp', 'avif']:
        raise ValueError("Invalid codec. Please choose 'webp' or 'avif'.")
    
    start_time = time.time()

    logger = setup_logger(verbose)
    src_path = Path(source)
    dst_path = Path(target)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".avif"}

    # Check extension is same with codec
    if dst_path.suffix and not dst_path.is_dir():
        if dst_path.suffix.lower() != f".{codec}":
            raise ValueError(f"Codec and target extension are not same. {dst_path.suffix} != {codec}")

    # Add extension check and warning
    if dst_path.suffix and dst_path.suffix.lower() in valid_exts and dst_path.suffix.lower() not in ['.webp', '.avif']:
        logger.warning("Currently only WebP or AVIF format is supported for output. Forcing .webp or .avif extension.")
        dst_path = dst_path.with_suffix('.webp') if codec == 'webp' else dst_path.with_suffix('.avif')

    source_paths = []
    saved_paths = []
    
    if src_path.is_file():
        source_paths.append(src_path)
        if src_path.suffix.lower() in ['.webp', '.avif']:
            if dst_path.suffix:  # target이 파일 경로인 경우
                saved_path = copy_original(src_path, dst_path, verbose)
            else:  # target이 디렉토리인 경우
                saved_path = copy_original(src_path, dst_path / src_path.name, verbose)
            saved_paths.append(saved_path)
        else:
            if dst_path.suffix:  # If target is a file path
                saved_path = process_single_image(src_path, src_path, dst_path.parent, target_psnr, verbose)
                # Rename the output file to match the target filename
                if saved_path.exists():
                    saved_path.rename(dst_path)
                    saved_paths.append(dst_path)
            else:  # If target is a directory
                saved_path = process_single_image(src_path, src_path, dst_path, target_psnr, verbose)
                saved_paths.append(saved_path)
    elif src_path.is_dir():
        # Check if target path has a media file extension
        if dst_path.suffix.lower() in valid_exts:
            raise ValueError("Target must be a directory when source is a directory")
        files = [
            f for f in src_path.rglob("*") 
            if f.is_file() and f.suffix.lower() in valid_exts
        ]
        source_paths.extend(files)
        logger.info(f"Found {len(files)} images. Starting processing...")

        with ThreadPoolExecutor() as executor:
            saved_paths = list(tqdm(
                executor.map(
                    lambda p: (
                        copy_original(p, dst_path / p.relative_to(src_path), verbose)
                        if p.suffix.lower() in ['.webp', '.avif']
                        else process_single_image(p, src_path, dst_path, target_psnr, verbose)
                    ),
                    files
                ),
                total=len(files),
                desc="Processing images"
            ))
    else:
        raise ValueError(f"Invalid source path: {source}")
    
    end_time = time.time()
    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    # Assert that source and target lists have same length
    assert len(source_paths) == len(saved_paths), "Source and target path lists must have same length"
    
    return source_paths, saved_paths
