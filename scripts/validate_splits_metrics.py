"""Validate src/data/splits.py and src/eval/metrics.py.

1. Loads array11 processed parquet, runs the clear-sky pipeline, splits it
   with split_chronological, and prints row count / date range /
   daylight-hour count for each split. The internal assertions in
   split_chronological already run as part of calling it - if they fail,
   this script fails loudly instead of printing a result.
2. Tests the metrics functions on synthetic data where the answer is known
   by hand and prints PASS/FAIL for each check.

Usage:
    python scripts/validate_splits_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky import add_daylight_mask, add_solar_position
from src.data.splits import split_chronological
from src.eval.metrics import all_metrics, mbe, skill_score

PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def validate_splits() -> None:
    print("=" * 60)
    print("split_chronological on array11_polySi_hourly.parquet")
    print("=" * 60)

    df = pd.read_parquet(PROCESSED_DIR / "array11_polySi_hourly.parquet")
    df = add_solar_position(df)
    df = add_daylight_mask(df)

    # split_chronological runs its own AssertionErrors internally; getting
    # here means train < val < test with no overlap and lengths that sum
    # to len(df).
    train, val, test = split_chronological(df)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        n_daylight = int(split["is_daylight"].sum())
        print(f"\n{name}:")
        print(f"  rows: {len(split)}")
        print(f"  date range: {split.index.min()} to {split.index.max()}")
        print(f"  daylight hours: {n_daylight}")

    print("\nAll split_chronological assertions passed.")


def _print_check(label, got, expected, tol=1e-9) -> bool:
    ok = abs(got - expected) < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: got {got:.6f}, expected {expected:.6f}")
    return ok


def validate_metrics() -> None:
    print("\n" + "=" * 60)
    print("metrics.py synthetic checks")
    print("=" * 60)

    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 5, size=50)

    results = []

    # Perfect prediction -> skill 1.0 (rmse(pred) = 0, so 1 - 0/ref = 1).
    y_pred_perfect = y_true.copy()
    y_ref = y_true + rng.uniform(0.1, 1.0, size=50)  # non-zero reference error
    skill_perfect = skill_score(y_true, y_pred_perfect, y_ref)
    results.append(_print_check("perfect prediction -> skill 1.0", skill_perfect, 1.0))

    # Prediction equal to the reference -> skill 0.0.
    skill_equal_ref = skill_score(y_true, y_ref, y_ref)
    results.append(_print_check("prediction == reference -> skill 0.0", skill_equal_ref, 0.0))

    # Constant offset of +1.0 -> MBE +1.0.
    y_pred_offset = y_true + 1.0
    mbe_offset = mbe(y_true, y_pred_offset)
    results.append(_print_check("constant +1.0 offset -> MBE +1.0", mbe_offset, 1.0))

    # all_metrics sanity check: n_samples matches input length when there
    # are no NaNs, and skill/mbe agree with the standalone functions above.
    metrics = all_metrics(y_true, y_pred_offset, y_ref, nameplate_kw=5.0)
    results.append(_print_check("all_metrics n_samples", metrics["n_samples"], 50))
    results.append(_print_check("all_metrics mbe matches mbe()", metrics["mbe"], mbe_offset))

    if all(results):
        print("\nAll metrics checks PASSED.")
    else:
        print("\nSome metrics checks FAILED.")
        sys.exit(1)


def main() -> None:
    validate_splits()
    validate_metrics()


if __name__ == "__main__":
    main()
