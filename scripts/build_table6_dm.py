"""Build the paper's Table 6 (Diebold-Mariano significance test).

Motivation: scripts/aggregate_seed_sweep.py's "exceeds 2 standard
deviations of either model's own 5-seed spread" heuristic is not a
hypothesis test - it has no p-value, no notion of a null distribution,
and does not account for the fact that daylight-hour forecast errors at
horizon h are autocorrelated up to lag h-1 (overlapping forecast
origins). This script replaces it, for the paper's significance claims,
with the Diebold-Mariano test (src/eval/dm.py): Diebold & Mariano (1995),
with the Harvey-Leybourne-Newbold (1997) small-sample correction. Seed
spread (Table 3) remains in the paper as a reproducibility statistic, not
a significance test.

Pipeline, per (array, horizon), VALIDATION split (2014) only - this
script never reads 2015 (CLAUDE.md research-integrity rule, test set
touched once at the end):
  load one array's processed parquet -> solar position / clear-sky
  irradiance / daylight mask / clear-sky index of GHI (src.data.clearsky)
  -> chronological split (src.data.splits) -> temperature climatology and
  gain fit on TRAIN only, clear-sky POWER added to train and val
  (src.data.clearsky_power) -> fit all 5 models (xgboost, lstm, cnn_lstm,
  lstm_residual, cnn_lstm_residual - residual models use
  residual_fit_split='oof', the corrected scheme, CLAUDE.md rule 6) AT
  SEED 0, PLUS smart_persistence and convex_reference as two more
  comparators (7 models total) -> restrict to the intersection of
  timestamps where ALL SEVEN produced a prediction, further restricted to
  is_daylight and with documented-outage hours dropped
  (src.eval.exclusions) -> error series e = Active_Power - p_hat per model
  -> src.eval.dm.dm_matrix over every pairwise comparison, with a
  Holm-Bonferroni correction applied across the 21 pairs within that one
  (array, horizon) cell.

Writes results/table6_dm.csv, one row per (array, horizon, model_1,
model_2): array, horizon, model_1, model_2, dbar, dm_stat, hln_stat,
p_raw, p_holm, n, better_model.

Prints, for each (array, horizon), the intersection size n and which
pairs are significant at p_holm < 0.05.

Runs src.eval.dm's self-test first (three checks - see that module and
run_self_test() below) and prints PASS/FAIL for each; aborts before
touching any data if a check fails, since a broken significance test is
worse than none.

No plotting, no test-split access, no fabricated numbers - every row is
a real model fit and predict on real data (CLAUDE.md research integrity
rules).

Usage:
    python scripts/build_table6_dm.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import ARRAYS, add_clearsky_power_per_split, load_and_prepare
from src.data.splits import split_chronological
from src.eval.dm import dm_matrix, dm_test
from src.eval.exclusions import exclusion_mask
from src.eval.runner import set_all_seeds
from src.models.base import check_no_lookahead
from src.models.climatology import ConvexCombination
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.lstm import LSTMForecaster
from src.models.persistence import SmartPersistence
from src.models.residual import ResidualCorrected
from src.models.xgb import XGBForecaster

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

HORIZONS = (1, 3, 6)
SEED = 0
REGIME = "lagged"
ALPHA = 0.05

CSV_COLUMNS = [
    "array", "horizon", "model_1", "model_2",
    "dbar", "dm_stat", "hln_stat", "p_raw", "p_holm", "n", "better_model",
]


# ---------------------------------------------------------------------------
# Self-test (src/eval/dm.py validation requirement)
# ---------------------------------------------------------------------------

def run_self_test():
    """Three checks on src.eval.dm before it is trusted with a real table.

    (a) DM of a series against itself: dbar == 0 exactly, p == 1.
    (b) DM of a good forecast against a deliberately degraded copy: a
        significant result (p < 0.05) with the correct sign (model 1,
        the good one, reported as better_model).
    (c) The HLN small-sample correction shrinks |statistic| relative to
        the raw DM statistic for h > 1 (see _bartlett_hac_variance's
        docstring - the correction factor is < 1 for any finite n).

    Prints PASS/FAIL for each and returns True iff all three passed.
    """
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="Australia/Darwin")

    print("=" * 90)
    print("self-test: src/eval/dm.py")
    print("=" * 90)

    all_passed = True

    # (a) identical series
    e_same = pd.Series(rng.normal(0, 1, n), index=idx)
    result_a = dm_test(e_same, e_same, h=3)
    passed_a = (result_a["dbar"] == 0.0) and (result_a["p_value"] == 1.0)
    print(
        f"(a) identical series: dbar={result_a['dbar']!r}  p={result_a['p_value']!r}  "
        f"{'PASS' if passed_a else 'FAIL'}"
    )
    all_passed &= passed_a

    # (b) good forecast vs a deliberately degraded copy (same errors plus
    # extra independent noise, so model 1 is unambiguously better)
    e_good = pd.Series(rng.normal(0, 0.1, n), index=idx)
    e_bad = e_good + rng.normal(0, 2.0, n)
    result_b = dm_test(e_good, e_bad, h=3)
    passed_b = (result_b["p_value"] < ALPHA) and (result_b["better_model"] == "model_1") and (result_b["dbar"] < 0)
    print(
        f"(b) good vs degraded: dbar={result_b['dbar']:+.4f}  hln_stat={result_b['hln_stat']:+.4f}  "
        f"p={result_b['p_value']:.6f}  better_model={result_b['better_model']}  "
        f"{'PASS' if passed_b else 'FAIL'}"
    )
    all_passed &= passed_b

    # (c) HLN correction shrinks the statistic for h > 1, same series as (b)
    result_c = dm_test(e_good, e_bad, h=6)
    passed_c = abs(result_c["hln_stat"]) < abs(result_c["dm_stat"])
    print(
        f"(c) HLN shrinkage (h=6): |dm_stat|={abs(result_c['dm_stat']):.4f}  "
        f"|hln_stat|={abs(result_c['hln_stat']):.4f}  {'PASS' if passed_c else 'FAIL'}"
    )
    all_passed &= passed_c

    print()
    return bool(all_passed)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_all_models(train, val, horizon, seed):
    """Fit all 5 benchmark models plus smart persistence and the convex
    reference at `seed`, on this (array, horizon)'s train/val, and return
    {model_name: predictions Series}. Mirrors scripts/run_xgb_dev.py /
    run_lstm_dev.py / run_cnn_lstm_dev.py / run_residual_dev.py exactly,
    minus the per-script result-JSON writing (this script's own output is
    results/table6_dm.csv, not one run JSON per model).

    Residual models use residual_fit_split='oof' (the default and only
    scheme that should ever produce a reported result, CLAUDE.md rule 6).
    """
    set_all_seeds(seed)
    preds = {}

    xgb_model = XGBForecaster(seed=seed, regime=REGIME)
    xgb_model.fit(train, horizon, df_val=val)
    preds["xgboost"] = xgb_model.predict(val, horizon)
    check_no_lookahead(val, preds["xgboost"], horizon)

    lstm_model = LSTMForecaster(seed=seed, regime=REGIME)
    lstm_model.fit(train, horizon, df_val=val)
    preds["lstm"] = lstm_model.predict(val, horizon)
    check_no_lookahead(val, preds["lstm"], horizon)

    cnn_lstm_model = CNNLSTMForecaster(seed=seed, regime=REGIME)
    cnn_lstm_model.fit(train, horizon, df_val=val)
    preds["cnn_lstm"] = cnn_lstm_model.predict(val, horizon)
    check_no_lookahead(val, preds["cnn_lstm"], horizon)

    lstm_residual_model = ResidualCorrected(
        LSTMForecaster(seed=seed, regime=REGIME), seed=seed, residual_fit_split="oof"
    )
    lstm_residual_model.fit(train, horizon, val)
    preds["lstm_residual"] = lstm_residual_model.predict(val, horizon)
    check_no_lookahead(val, preds["lstm_residual"], horizon)

    cnn_lstm_residual_model = ResidualCorrected(
        CNNLSTMForecaster(seed=seed, regime=REGIME), seed=seed, residual_fit_split="oof"
    )
    cnn_lstm_residual_model.fit(train, horizon, val)
    preds["cnn_lstm_residual"] = cnn_lstm_residual_model.predict(val, horizon)
    check_no_lookahead(val, preds["cnn_lstm_residual"], horizon)

    # ALSO includes smart persistence and the convex reference as
    # comparators - not just as the two skill-score denominators
    # (CLAUDE.md rule 4), but as models in their own right in this test.
    sp_model = SmartPersistence()
    sp_model.fit(train, horizon)
    preds["smart_persistence"] = sp_model.predict(val, horizon)
    check_no_lookahead(val, preds["smart_persistence"], horizon)

    convex_model = ConvexCombination()
    convex_model.fit(train, horizon, val)
    preds["convex_reference"] = convex_model.predict(val, horizon)
    check_no_lookahead(val, preds["convex_reference"], horizon)

    return preds


def build_cell(array, horizon, train, val, nameplate_kw):
    preds = fit_all_models(train, val, horizon, SEED)

    common_idx = preds["xgboost"].index
    for name, p in preds.items():
        if name == "xgboost":
            continue
        common_idx = common_idx.intersection(p.index)

    daylight_mask = val.loc[common_idx, "is_daylight"].to_numpy()
    daylight_idx = common_idx[daylight_mask]

    outage_mask = exclusion_mask(array, daylight_idx)
    eval_idx = daylight_idx[~outage_mask.to_numpy()]
    n_excluded_outage = int(outage_mask.sum())

    print(
        f"array={array}  horizon={horizon}  n_intersection(7 models)={len(common_idx)}  "
        f"n_daylight={len(daylight_idx)}  n_excluded_outage={n_excluded_outage}  "
        f"n_eval={len(eval_idx)}"
    )

    y_true = val.loc[eval_idx, "Active_Power"]
    errors = {name: (y_true - p.loc[eval_idx]) for name, p in preds.items()}

    hln_df, p_holm_df, pairs_df = dm_matrix(errors, h=horizon)

    pairs_df.insert(0, "horizon", horizon)
    pairs_df.insert(0, "array", array)

    sig = pairs_df[pairs_df["p_holm"] < ALPHA]
    if sig.empty:
        print(f"  no pairs significant at p_holm < {ALPHA}")
    else:
        print(f"  significant at p_holm < {ALPHA}:")
        for row in sig.itertuples():
            print(
                f"    {row.model_1} vs {row.model_2}: dbar={row.dbar:+.5f}  "
                f"hln_stat={row.hln_stat:+.3f}  p_holm={row.p_holm:.4g}  "
                f"better={row.better_model}"
            )

    return pairs_df, hln_df, p_holm_df


def main():
    if not run_self_test():
        print("self-test FAILED - aborting before touching any data")
        sys.exit(1)

    all_rows = []
    array11_matrices = {}

    for array in sorted(ARRAYS):
        df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
        train, val, test = split_chronological(df)
        train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)

        for horizon in HORIZONS:
            print(f"\n{'=' * 90}")
            print(f"array={array}  horizon={horizon}h  seed={SEED}  regime={REGIME}  (validation split, 2014)")
            print("=" * 90)

            start = time.perf_counter()
            pairs_df, hln_df, p_holm_df = build_cell(array, horizon, train, val, nameplate_kw)
            elapsed = time.perf_counter() - start
            print(f"  cell done in {elapsed:.1f}s")

            all_rows.append(pairs_df)
            if array == "array11":
                array11_matrices[horizon] = (hln_df, p_holm_df)

    out_df = pd.concat(all_rows, ignore_index=True)[CSV_COLUMNS]
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "table6_dm.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(out_df)} rows)")

    print(f"\n{'=' * 90}")
    print("array11 DM matrices (HLN statistic; row = model_1, column = model_2 in dm_test's sign")
    print("convention - hln_df.loc[a, b] < 0 means a has lower loss than b)")
    print("=" * 90)
    for horizon, (hln_df, p_holm_df) in sorted(array11_matrices.items()):
        print(f"\n--- array11, h={horizon} : HLN statistics ---")
        print(hln_df.round(3).to_string())
        print(f"\n--- array11, h={horizon} : Holm-adjusted p-values ---")
        print(p_holm_df.round(4).to_string())


if __name__ == "__main__":
    main()
