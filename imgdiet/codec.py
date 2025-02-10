import io
import numpy as np
from imgdiet.utils import calculate_psnr
from PIL import Image
from typing import Optional, Tuple  


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