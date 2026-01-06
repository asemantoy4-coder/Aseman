# fast-scalp-crypto-bot/utils/logger.py
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str = "fast_scalp.log") -> logging.Logger:
    """تنظیم و ایجاد لاگر"""
    
    # ایجاد دایرکتوری لاگ‌ها
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # تنظیم فرمت
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ایجاد لاگر
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # هندلر فایل
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # هندلر کنسول
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
