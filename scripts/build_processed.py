"""Run all three arrays through the load -> clean -> resample pipeline.

Saves each array to data/processed/<name>_hourly.parquet and prints row
count, date range, and NaN percentage per column.

Usage:
    python scripts/build_processed.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import clean_5min, load_array, resample_hourly
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Nameplate capacities in kW.
ARRAYS = {
    "array07_CdTe": (RAW_DIR / "array07_CdTe.csv", 7.0),
    "array11_polySi": (RAW_DIR / "array11_polySi.csv", 5.0),
    "array12_monoSi": (RAW_DIR / "array12_monoSi.csv", 5.1),
}


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, (path, nameplate_kw) in ARRAYS.items():
        print(f"\n{name} ({path.name}, nameplate {nameplate_kw} kW)")

        raw = load_array(path, start="2009-01-01", end="2015-12-31")
        cleaned, removed = clean_5min(raw, nameplate_kw)
        hourly = resample_hourly(cleaned)

        out_path = PROCESSED_DIR / f"{name}_hourly.parquet"
        hourly.to_parquet(out_path)

        print(f"  rows: {len(hourly)}")
        print(f"  date range: {hourly.index.min()} to {hourly.index.max()}")
        print("  NaN percentage per column:")
        nan_pct = 100.0 * hourly.isna().mean()
        for col, pct in nan_pct.items():
            print(f"    {col}: {pct:.2f}%")
        print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
