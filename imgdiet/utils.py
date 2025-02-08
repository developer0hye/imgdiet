import logging

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
