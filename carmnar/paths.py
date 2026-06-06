"""Repository-relative paths for data and results.

These can be overridden with the environment variables ``CARMNAR_DATA_DIR`` and
``CARMNAR_RESULTS_DIR`` (useful when the package is installed and run from
outside the repository).
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root: <repo>/carmnar/paths.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("CARMNAR_DATA_DIR", REPO_ROOT / "data" / "raw"))
RESULTS_DIR = Path(os.environ.get("CARMNAR_RESULTS_DIR", REPO_ROOT / "results"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
