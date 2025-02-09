import logging
import numpy as np

def setup_logger(verbose: bool) -> logging.Logger:
    """Configure and return a logger with appropriate level"""
    logger = logging.getLogger("imgdiet")
    
    # 기존 logger가 있으면 level만 설정하고 반환
    if logger.hasHandlers():
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        return logger
        
    # 새로운 logger 설정
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[imgdiet] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger


def calculate_psnr(
    original_bgr: np.ndarray,
    compressed_bgr: np.ndarray
) -> float:
    """
    TODO: support various bit depth

    Calculates PSNR (Peak Signal-to-Noise Ratio) in dB.
    Returns float('inf') if images are identical.
    """
    mse = float(np.mean((original_bgr - compressed_bgr) ** 2))
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20.0 * np.log10(max_pixel / np.sqrt(mse))
