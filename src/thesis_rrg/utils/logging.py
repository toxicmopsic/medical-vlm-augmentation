from __future__ import annotations

import logging
from pathlib import Path

from thesis_rrg.utils.io import ensure_dir



def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("thesis_rrg")
    logger.setLevel(level)

    # Avoid duplicate handlers when called multiple times.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        file_path = Path(log_file)
        ensure_dir(file_path.parent)
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
