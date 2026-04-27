"""
Logging Utility for MEXC Trading System
"""
import logging
import os
from datetime import datetime
from config import settings


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logs directory
    log_dir = "./data/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create main logger
logger = setup_logger("mexc_trading", "trading.log")
