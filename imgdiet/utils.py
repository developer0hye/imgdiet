import logging
import numpy as np  
import shutil
import time
from pathlib import Path
from typing import Union

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

def measure_time(func):
    """
    Decorator to measure execution time of a function and log it.
    """
    def wrapper(*args, verbose=False, **kwargs):
        start_time = time.time()
        result = func(*args, verbose=verbose, **kwargs)
        end_time = time.time()
        
        logger = setup_logger(verbose)
        logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper

def copy_original(
    src: Union[str, Path],
    dst: Union[str, Path],
    verbose: bool = False
) -> Path:
    """
    Copies the original file from src to dst.
    If src and dst are the same, skip copying.
    Returns the destination path.
    """
    logger = setup_logger(verbose)
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        logger.info(f"Source and destination are same, skipping: {src}")
        return dst
    
    if dst.is_dir():
        dst = dst / src.name

    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Copying original: {src} -> {dst}")
    shutil.copy2(src, dst)
    return dst

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
