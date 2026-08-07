"""Validate src/models/persistence.py (SmartPersistence) on all three
arrays, and give a first preview of RQ1 (how much reported accuracy comes
from including night hours).

Evaluated on the VALIDATION split (2014) only, NOT the test split (2015).
Per CLAUDE.md's research-integrity rules, the test set is touched once, at
the end of the whole project, after every modelling and evaluation choice
is frozen. This script is model/protocol development, not the final
number, so it must not look at 2015 at all - not even to "just check."

The clear-sky power model (temperature climatology + gain) is fit on
TRAINING data only (2012-2013), then applied to the validation split's
index to produce p_cs and k_p there. That is not a leak: gain and the
climatology table are training-only parameters, and using them to compute
a reference/index on other splits is exactly how they are meant to be
used downstream.

For horizons 1, 3, 6 hours, prints per array MAE/RMSE/nRMSE/MBE/n_samples/
fallback_fraction twice: once over daylight hours only, once over all 24
hours, plus the ratio of the two nRMSEs. If that ratio is well below 1,
it means the all-hours number looks much better than the daylight-only
number purely because smart persistence predicts near-zero at night and
is trivially "right" about it - free accuracy, not forecasting skill.

Usage:
    python scripts/validate_persistence.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky import (
    add_clearsky,
    add_clearsky_index_ghi,
    add_daylight_mask,
    add_solar_position,
)
from src.data.clearsky_power import (
    GAMMA_PDC_CDTE,
    GAMMA_PDC_SILICON,
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
from src.data.splits import split_chronological
from src.eval.metrics import mae, mbe, nrmse, rmse
from src.models.base import check_no_lookahead
from src.models.persistence import SmartPersistence

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# name, parquet filename, nameplate kW, gamma_pdc
ARRAYS = [
    ("array07_CdTe", "array07_CdTe_hourly.parquet", 7.0, GAMMA_PDC_CDTE),
    ("array11_polySi", "array11_polySi_hourly.parquet", 5.0, GAMMA_PDC_SILICON),
    ("array12_monoSi", "array12_monoSi_hourly.parquet", 5.1, GAMMA_PDC_SILICON),
]

HORIZONS = [1, 3, 6]


def evaluate_subset(y_true, y_pred, fallback_mask, nameplate_kw):
    """Metrics for one (array, horizon, daylight/all) cell."""
    valid = y_true.notna() & y_pred.notna()
    n_samples = int(valid.sum())
    fallback_fraction = float(fallback_mask.mean()) if len(fallback_mask) else float("nan")
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred, nameplate_kw),
        "mbe": mbe(y_true, y_pred),
        "n_samples": n_samples,
        "fallback_fraction": fallback_fraction,
    }


def print_row(label, r):
    print(
        f"  {label:10s}  MAE={r['mae']:.4f} kW  RMSE={r['rmse']:.4f} kW  "
        f"nRMSE={r['nrmse']:.2f}%  MBE={r['mbe']:+.4f} kW  n={r['n_samples']:>5d}  "
        f"fallback={100 * r['fallback_fraction']:.1f}%"
    )


def main() -> None:
    for name, filename, nameplate_kw, gamma_pdc in ARRAYS:
        print(f"\n{'=' * 70}")
        print(f"{name} (nameplate {nameplate_kw} kW)")
        print("=" * 70)

        df = pd.read_parquet(PROCESSED_DIR / filename)
        df = add_solar_position(df)
        df = add_clearsky(df)
        df = add_daylight_mask(df)
        df = add_clearsky_index_ghi(df)

        # test (2015) is intentionally unused below - see module docstring.
        train, val, _test = split_chronological(df)

        temp_clim = fit_temperature_climatology(train)
        p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
        gain, n_gain_hours, gain_iqr = fit_gain(train, p_cs_raw_train)
        print(f"gain (fit on 2012-2013): {gain:.4f}  (hours used: {n_gain_hours}, IQR: {gain_iqr:.4f})")

        # Apply the training-fit climatology and gain to the validation
        # split's own index - this is not a leak, see module docstring.
        p_cs_raw_val = model_clearsky_power(val.index, nameplate_kw, gamma_pdc, temp_clim)
        val = add_clearsky_power(val, p_cs_raw_val, gain, nameplate_kw)

        for horizon in HORIZONS:
            model = SmartPersistence()
            model.fit(train, horizon)  # no-op, see persistence.py
            preds = model.predict(val, horizon)
            check_no_lookahead(val, preds, horizon)

            y_true = val.loc[preds.index, "Active_Power"]
            is_daylight = val.loc[preds.index, "is_daylight"]
            fallback_mask = model._fallback_mask

            results = {}
            for label, mask in [
                ("daylight", is_daylight),
                ("all_hours", pd.Series(True, index=preds.index)),
            ]:
                results[label] = evaluate_subset(
                    y_true[mask], preds[mask], fallback_mask[mask], nameplate_kw
                )

            print(f"\n--- horizon = {horizon}h (validation split, 2014) ---")
            print_row("daylight", results["daylight"])
            print_row("all_hours", results["all_hours"])

            ratio = results["all_hours"]["nrmse"] / results["daylight"]["nrmse"]
            print(f"  all_hours nRMSE / daylight nRMSE = {ratio:.3f}")


if __name__ == "__main__":
    main()
