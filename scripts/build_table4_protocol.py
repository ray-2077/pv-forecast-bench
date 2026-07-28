"""Build the paper's Table 4 (protocol inflation) from scratch.

Motivation: paper/PROJECT_CHECKPOINT.md Finding 2 shows that including
night hours deflates nRMSE by roughly the closed-form factor
sqrt(N_daylight / N_all), and that skill vs a defensible reference is
essentially immune to the same choice. The run JSONs under results/
cannot demonstrate this cleanly: their "daylight"/"common_hours" (or,
pre-2026-07-28, "daylight"/"all_hours") blocks are both restricted to
the intersection where the model AND SmartPersistence AND Climatology
AND ConvexCombination all produced a prediction, and Climatology has no
prediction at (month, hour) cells that are never daylight in training
data - see src/eval/runner.py's module docstring CAVEAT. Neither block
in a run JSON spans a true 24-hour cycle, so neither can show what
happens when night hours are genuinely included.

This script computes each configuration from scratch, using ONLY the
prediction intersection each one actually needs (never the four-way
intersection above):

  C1  daylight only, skill vs convex reference   <- the correct protocol
      (CLAUDE.md rule 4): model + convex_reference only, restricted to
      is_daylight.
  C2  daylight only, skill vs smart persistence: model + persistence
      only, restricted to is_daylight.
  C3  ALL 24 HOURS, skill vs convex reference: model + convex_reference
      only, no daylight filter. Convex still inherits Climatology's
      coverage gap, so this does NOT span a true 24h cycle either - the
      point of including it is to show that gap directly, via n_samples.
  C4  ALL 24 HOURS, skill vs smart persistence: model + persistence
      only, no daylight filter. SmartPersistence forward-fills through
      the night (src.models.persistence.SmartPersistence,
      FFILL_LIMIT_HOURS=24), so this DOES span a true 24h cycle - it is
      the one configuration in this table that actually answers the
      night-inclusion question against a skill-score reference.
  C5  daylight only, raw nRMSE, no skill score: model only, restricted
      to is_daylight. No reference forecaster is touched.
  C6  ALL 24 HOURS, raw nRMSE, no skill score: model only, no daylight
      filter, no reference forecaster. This is N_all in the closed-form
      check below.

Fixed: model=xgboost, regime=lagged, seed=0. Swept: all three arrays
(array11, array12, array17 - array07 excluded, see CLAUDE.md "Data
window"), all three horizons (1, 3, 6), VALIDATION split (2014) only.
2015 is never loaded past split_chronological's mechanical partitioning
and its rows are never read, per CLAUDE.md's research-integrity rules.

Documented-outage hours (src.eval.exclusions.exclusion_mask) are
dropped from every configuration's index, same as every run_*_dev.py
script, so no configuration benefits/suffers from hours known to be
equipment-dead rather than a forecasting failure.

After the six configurations, prints the closed-form night-inclusion
check per (array, horizon):
  ratio    = nRMSE(C6) / nRMSE(C5)
  analytic = sqrt(n_samples(C5) / n_samples(C6))
  diff     = ratio - analytic
per Finding 2's derivation (RMSE_all = RMSE_day * sqrt(N_day / N_all),
assuming night errors are ~0 since both truth and a sane forecast are
~0 at night).

Writes results/table4_protocol.csv, one row per (array, horizon,
config): array, horizon, config_id, config_description, n_samples,
rmse, nrmse, mae, mbe, skill, reference_used, hours_included.

No plotting, no test-split access, no fabricated numbers - every row is
a real XGBoost fit and predict on real data (CLAUDE.md research
integrity rules).

Usage:
    python scripts/build_table4_protocol.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import ARRAYS, add_clearsky_power_per_split, load_and_prepare
from src.data.splits import split_chronological
from src.eval.exclusions import exclusion_mask
from src.eval.metrics import mae, mbe, nrmse, rmse, skill_score
from src.eval.runner import set_all_seeds
from src.models.base import check_no_lookahead
from src.models.climatology import Climatology, ConvexCombination
from src.models.persistence import SmartPersistence
from src.models.xgb import XGBForecaster

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

HORIZONS = (1, 3, 6)
SEED = 0
REGIME = "lagged"

# (config_id, description, hours_included, reference_key) - reference_key
# indexes into the reference_preds dict built per (array, horizon), or is
# None for the two no-skill-score configs.
CONFIGS = [
    ("C1", "daylight only, skill vs convex reference (correct protocol, CLAUDE.md rule 4)",
     "daylight", "convex_reference"),
    ("C2", "daylight only, skill vs smart persistence",
     "daylight", "smart_persistence"),
    ("C3", "all 24 hours, skill vs convex reference (model + convex only)",
     "all_24h", "convex_reference"),
    ("C4", "all 24 hours, skill vs smart persistence (model + persistence only, true 24h cycle)",
     "all_24h", "smart_persistence"),
    ("C5", "daylight only, raw nRMSE, no skill score",
     "daylight", None),
    ("C6", "all 24 hours, raw nRMSE, no skill score",
     "all_24h", None),
]


def restrict_daylight(val, idx):
    return idx[val.loc[idx, "is_daylight"].to_numpy()]


def drop_outages(array, idx):
    mask = exclusion_mask(array, idx)
    return idx[~mask.to_numpy()]


def fit_forecasters(train, val, horizon, seed):
    """Fit xgboost + the two reference forecasters on this (array,
    horizon)'s train/val, and return their predictions on val plus the
    fitted convex weight. Mirrors scripts/run_xgb_dev.py's pipeline
    exactly, minus the four-way common_idx restriction this script is
    specifically avoiding.
    """
    set_all_seeds(seed)

    model = XGBForecaster(seed=seed, regime=REGIME)
    model.fit(train, horizon, df_val=val)
    preds_xgb = model.predict(val, horizon)
    check_no_lookahead(val, preds_xgb, horizon)

    sp_model = SmartPersistence()
    sp_model.fit(train, horizon)
    preds_sp = sp_model.predict(val, horizon)
    check_no_lookahead(val, preds_sp, horizon)

    convex_model = ConvexCombination()
    convex_model.fit(train, horizon, val)
    preds_convex = convex_model.predict(val, horizon)
    check_no_lookahead(val, preds_convex, horizon)

    reference_preds = {
        "smart_persistence": preds_sp,
        "convex_reference": preds_convex,
    }
    return preds_xgb, reference_preds, convex_model.w


def build_config_row(array, horizon, config_id, description, hours_included,
                      reference_used, val, preds_xgb, preds_ref, nameplate_kw):
    idx = preds_xgb.index
    if preds_ref is not None:
        idx = idx.intersection(preds_ref.index)
    if hours_included == "daylight":
        idx = restrict_daylight(val, idx)
    idx = drop_outages(array, idx)

    y_true = val.loc[idx, "Active_Power"]
    y_pred = preds_xgb.loc[idx]
    skill = None
    if preds_ref is not None:
        y_ref = preds_ref.loc[idx]
        skill = skill_score(y_true, y_pred, y_ref)

    return {
        "array": array,
        "horizon": horizon,
        "config_id": config_id,
        "config_description": description,
        "n_samples": int(len(idx)),
        "rmse": rmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred, nameplate_kw),
        "mae": mae(y_true, y_pred),
        "mbe": mbe(y_true, y_pred),
        "skill": skill,
        "reference_used": reference_used if reference_used is not None else "none",
        "hours_included": hours_included,
    }


def print_config_row(row):
    skill_str = f"{row['skill']:+.4f}" if row["skill"] is not None else "    n/a"
    print(
        f"  {row['config_id']}  {row['hours_included']:9s}  ref={row['reference_used']:17s}  "
        f"n={row['n_samples']:5d}  nRMSE={row['nrmse']:6.2f}%  skill={skill_str}  "
        f"| {row['config_description']}"
    )


def main():
    all_rows = []
    closed_form_rows = []

    for array in sorted(ARRAYS):
        df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
        train, val, test = split_chronological(df)
        train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)

        for horizon in HORIZONS:
            print(f"\n{'=' * 90}")
            print(f"array = {array}  horizon = {horizon}h  model = xgboost  regime = {REGIME}  "
                  f"seed = {SEED}  (validation split, 2014)")
            print("=" * 90)

            preds_xgb, reference_preds, convex_w = fit_forecasters(train, val, horizon, SEED)
            print(f"convex_weight (fit on validation) = {convex_w:.2f}")

            cell_rows = {}
            for config_id, description, hours_included, ref_key in CONFIGS:
                preds_ref = reference_preds.get(ref_key) if ref_key is not None else None
                row = build_config_row(
                    array, horizon, config_id, description, hours_included,
                    ref_key, val, preds_xgb, preds_ref, nameplate_kw,
                )
                print_config_row(row)
                all_rows.append(row)
                cell_rows[config_id] = row

            n5, n6 = cell_rows["C5"]["n_samples"], cell_rows["C6"]["n_samples"]
            nrmse5, nrmse6 = cell_rows["C5"]["nrmse"], cell_rows["C6"]["nrmse"]
            ratio = nrmse6 / nrmse5
            analytic = np.sqrt(n5 / n6)
            diff = ratio - analytic
            print(
                f"\n  closed-form night-inclusion check: nRMSE(C6)/nRMSE(C5) = {ratio:.4f}  "
                f"sqrt(N_day/N_all) = {analytic:.4f}  diff = {diff:+.4f} "
                f"({100 * diff / analytic:+.2f}%)"
            )
            closed_form_rows.append({
                "array": array, "horizon": horizon,
                "n_daylight": n5, "n_all": n6,
                "nrmse_ratio": ratio, "analytic_prediction": analytic, "diff": diff,
            })

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "table4_protocol.csv"
    out_df = pd.DataFrame(all_rows, columns=[
        "array", "horizon", "config_id", "config_description", "n_samples",
        "rmse", "nrmse", "mae", "mbe", "skill", "reference_used", "hours_included",
    ])
    out_df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(out_df)} rows)")

    print(f"\n{'=' * 90}")
    print("closed-form night-inclusion summary (all array x horizon cells)")
    print("=" * 90)
    for r in closed_form_rows:
        print(
            f"  {r['array']:8s} h={r['horizon']}  N_day={r['n_daylight']:5d}  N_all={r['n_all']:5d}  "
            f"ratio={r['nrmse_ratio']:.4f}  analytic={r['analytic_prediction']:.4f}  "
            f"diff={r['diff']:+.4f}"
        )


if __name__ == "__main__":
    main()
