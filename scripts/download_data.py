"""Re-download the three clinical datasets into data/raw/.

The datasets are bundled in the repository already; this script is provided for
provenance and to refresh them from their public sources. URLs may change over
time; see data/README.md for the canonical sources (UCI / Kaggle).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Mirror URLs for the raw CSVs. Update if a source moves.
SOURCES = {
    "Diabetes.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
    # Heart and Hepatitis: see data/README.md for the UCI sources; many community
    # mirrors exist. We keep the bundled copies as the reference of record.
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = DATA_DIR / name
        try:
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  -> {dest}")
        except Exception as e:  # pragma: no cover - network dependent
            print(f"  [skip] could not fetch {name}: {e}", file=sys.stderr)
    print("Done. The repository already ships working copies under data/raw/.")


if __name__ == "__main__":
    main()
