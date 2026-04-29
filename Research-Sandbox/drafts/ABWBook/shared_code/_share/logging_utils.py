from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd

# logging_utils.py

def get_logger(name: str = 'geo_util', log_dir: Path = None) -> logging.Logger:
    '''
    Create or retrieve a named logger that writes to both the console and
    optionally a log file (if `log_dir` is provided).

    Parameters:
    - name: The name of the logger (used for filtering/config).
    - log_dir: Optional Path to a directory where logs will be stored.

    Returns:
    - A configured `logging.Logger` instance.
    '''
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        # Optional file handler
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f'{name}.log'
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

    return logger

def seconds2clock(nofSeconds: float) -> str:
    """Convert number of seconds (possibly float) to HH:MM:SS.sss clock format."""
    hours, remainder = divmod(nofSeconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours):02}:{int(minutes):02}:{seconds:06.3f}'
