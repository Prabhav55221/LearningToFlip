"""
Logging setup for runs.

Call setup() once at the top of any entry-point script. All module-level
loggers (e.g. in solver.py) inherit the level automatically.

Levels:
  default  -- WARNING and above only (quiet; just final summary)
  verbose  -- INFO: per-instance results
  debug    -- DEBUG: per-try detail inside each solve call
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


FMT = "%(asctime)s | %(levelname)-5s | %(message)s"
DATEFMT = "%H:%M:%S"


def setup(verbose: bool = False, debug: bool = False, log_dir: Path | None = None) -> None:
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"run_{ts}.log")
        fh.setFormatter(logging.Formatter(FMT, datefmt=DATEFMT))
        handlers.append(fh)

    logging.basicConfig(level=level, format=FMT, datefmt=DATEFMT,
                        handlers=handlers, force=True)
