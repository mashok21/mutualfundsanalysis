import random
import numpy as np
import logging
from pathlib import Path
from typing import Union

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def setup_logger(name: str, log_file: Union[str, Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger with console handler and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensures a directory exists, creating parent directories if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
