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

Pipeline, per (array, horizon), on the split selected by --eval-split
(default val, i.e. 2014; CLAUDE.md research-integrity rule, test set
touched once at the end - this script only reads 2015 if --eval-split
test is passed explicitly, see CHANGE LOG below):
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

Writes results/table6_dm_<regime>.csv (val) or results/table6_dm_
<regime>_test.csv (test), one row per (array, horizon, model_1, model_2):
array, horizon, model_1, model_2, dbar, dm_stat, hln_stat, p_raw, p_holm,
n, better_model.

--regime selects lagged (default) or oracle, mirroring run_seed_sweep.py
and aggregate_seed_sweep.py's flag exactly - CLAUDE.md rule 5. Output
path is regime-suffixed so the two can never shadow each other
(2026-08-07: see build_table4_protocol.py's docstring for why this
matters - this script had the same hardcoded-REGIME, no-flag,
un-suffixed-output shape until now).

Prints, for each (array, horizon), the intersection size n and which
pairs are significant at p_holm < 0.05.

Runs src.eval.dm's self-test first (three checks - see that module and
run_self_test() below) and prints PASS/FAIL for each; aborts before
touching any data if a check fails, since a broken significance test is
worse than none.

No plotting, no fabricated numbers - every row is a real model fit and
predict on real data (CLAUDE.md research integrity rules).

CHANGE LOG (2026-08-09): added --eval-split {val,test}, default val,
same strict-argparse shape as scripts/aggregate_seed_sweep.py. val is
UNCHANGED from before this flag existed: models are fit on train (val
used only for early stopping / the residual oof base refit / the convex
weight w, exactly as before) and evaluated on val, writing results/
table6_dm_<regime>.csv - the same path as always. test fits on the same
train (val still used ONLY for early stopping / oof / convex weight w,
NEVER for computing the error series - matching scripts/
run_final_test.py's own convention exactly) but evaluates on test (2015)
instead, writing a DISTINCT, _test-suffixed path (results/table6_dm_
<regime>_test.csv) that can never overwrite or shadow the val output.
Unlike scripts/run_final_test.py, this script does not read from or
write run JSONs at all - Diebold-Mariano needs a full paired error
series across all 7 comparators evaluated at the exact same timestamps,
which no run JSON stores (they hold aggregate metrics only) - so a
--eval-split test run here REFITS every model against test, independent
of and in addition to the 450 run JSONs scripts/run_final_test.py
already wrote. This is deliberately NOT run automatically by this
change: --eval-split test touches the test split and costs roughly the
same wall-clock as the existing val path (63 model fits: 3 arrays x 3
horizons x 7 comparators, at a single seed=0, matching the val path's
own cost) - run it only when you want DM significance against the held-
out year specifically.

CHANGE LOG (2026-08-09, later same day): --eval-split test crashed with
torch.AcceleratorError (CUDA out of memory) at array12 h=6, five cells
in, with another GPU process running concurrently - a contributing but
not sole cause. Nothing in this script freed GPU memory between fits,
and each residual-corrected model fits its LSTM/CNN-LSTM base THREE
TIMES within a single build_cell call (once per out-of-fold fold, once
on the full training period) - 7 models x up to 3 fits each, times 9
cells, with every intermediate model object kept alive until Python's
garbage collector got around to it. Fixed two ways:
  1. fit_all_models and build_cell now del each model object and call
     torch.cuda.empty_cache() (see _free_gpu_memory below) as soon as
     its predictions are collected, rather than waiting for the
     function to return and letting garbage collection reclaim it
     eventually. No statistical logic changed - this only affects when
     memory is freed, not what is computed.
  2. The crash also meant the run threw away all 5 already-completed
     cells' work - main() wrote the full output CSV only once, at the
     very end, via a single pd.concat + to_csv. It now appends each
     cell's rows to the output CSV as soon as that cell completes (mode
     "a", matching results/table6_dm_<regime>[_test].csv's existing
     path/naming - unchanged for val), and skips any (array, horizon)
     already present in that file on startup - the same skip-if-exists
     resumability scripts/run_seed_sweep.py already has for run JSONs.
     A crash now loses at most the one in-progress cell, not the whole
     run.
The val path was re-run after these changes and its output CSV is
byte-identical to before (see git history / commit message for that
verification).

Usage:
    python scripts/build_table6_dm.py [--regime {lagged,oracle}] [--eval-split {val,test}]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky_power import (
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
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
ALPHA = 0.05

CSV_COLUMNS = [
    "array", "horizon", "model_1", "model_2",
    "dbar", "dm_stat", "hln_stat", "p_raw", "p_holm", "n", "better_model",
]


def add_clearsky_power_to_test(train, test, nameplate_kw, gamma_pdc):
    """Apply the SAME train-only-fitted temperature climatology and gain
    add_clearsky_power_per_split already used for train/val to test too -
    never refitting anything on val or test. Only called when --eval-split
    test is passed; the val-only default path never touches this function
    or the test split at all.

    Deliberately re-derives temp_clim/gain from `train` rather than
    threading add_clearsky_power_per_split's internal fit results out of
    that function, mirroring scripts/run_final_test.py's own
    add_clearsky_power_all_splits (that script is write-once and is not
    imported from here - see its own docstring). fit_gain is
    deterministic given `train`, so re-deriving it is equivalent to
    reusing it, not a second, possibly-different fit.
    """
    temp_clim = fit_temperature_climatology(train)
    p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
    gain, _n_gain_hours, _gain_iqr = fit_gain(train, p_cs_raw_train)

    p_cs_raw_test = model_clearsky_power(test.index, nameplate_kw, gamma_pdc, temp_clim)
    return add_clearsky_power(test, p_cs_raw_test, gain, nameplate_kw)


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

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("lagged", "oracle"), default="lagged")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    return parser.parse_args()


def _free_gpu_memory():
    """Called after each model's predictions are collected in
    fit_all_models, and again at the end of build_cell. torch.cuda.
    empty_cache() only releases CUDA blocks its caching allocator can
    prove are unreferenced by any live Python object - deleting the
    model object first (at each call site, before this is called) is
    what makes that memory reclaimable in the first place; calling this
    alone without the preceding del would do nothing for a model still
    referenced by a local variable. See this file's 2026-08-09 CHANGE
    LOG entry for the OOM crash this fixes - a single build_cell call
    fits 7 models, 3 of which (the residual-corrected pair) internally
    fit their LSTM/CNN-LSTM base up to 3 times each, and none of that
    was being freed until the whole function returned.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fit_all_models(train, val, eval_df, horizon, seed, regime):
    """Fit all 5 benchmark models plus smart persistence and the convex
    reference at `seed`, on this (array, horizon)'s train/val, and return
    {model_name: predictions Series} evaluated on `eval_df`. Mirrors
    scripts/run_xgb_dev.py / run_lstm_dev.py / run_cnn_lstm_dev.py /
    run_residual_dev.py exactly, minus the per-script result-JSON writing
    (this script's own output is results/table6_dm_<regime>[_test].csv,
    not one run JSON per model).

    `val` is used ONLY for early stopping, the residual oof base refit,
    and the convex weight w - never for the predictions this function
    returns. `eval_df` is val itself when --eval-split val (the default,
    reproducing the exact previous behaviour), or test when --eval-split
    test - matching scripts/run_final_test.py's own convention that VAL
    is still used for early stopping and for fitting the convex weight w
    even when the final evaluation target is test.

    Residual models use residual_fit_split='oof' (the default and only
    scheme that should ever produce a reported result, CLAUDE.md rule 6).
    """
    set_all_seeds(seed)
    preds = {}

    xgb_model = XGBForecaster(seed=seed, regime=regime)
    xgb_model.fit(train, horizon, df_val=val)
    preds["xgboost"] = xgb_model.predict(eval_df, horizon)
    check_no_lookahead(eval_df, preds["xgboost"], horizon)
    del xgb_model
    _free_gpu_memory()

    lstm_model = LSTMForecaster(seed=seed, regime=regime)
    lstm_model.fit(train, horizon, df_val=val)
    preds["lstm"] = lstm_model.predict(eval_df, horizon)
    check_no_lookahead(eval_df, preds["lstm"], horizon)
    del lstm_model
    _free_gpu_memory()

    cnn_lstm_model = CNNLSTMForecaster(seed=seed, regime=regime)
    cnn_lstm_model.fit(train, horizon, df_val=val)
    preds["cnn_lstm"] = cnn_lstm_model.predict(eval_df, horizon)
    check_no_lookahead(eval_df, preds["cnn_lstm"], horizon)
    del cnn_lstm_model
    _free_gpu_memory()

    # residual_fit_split='oof' fits the base model up to 3 times inside
    # a single .fit() call (once per out-of-fold fold, once on the full
    # training period, CLAUDE.md rule 6) - the biggest single source of
    # the accumulation this cleanup targets.
    lstm_residual_model = ResidualCorrected(
        LSTMForecaster(seed=seed, regime=regime), seed=seed, residual_fit_split="oof"
    )
    lstm_residual_model.fit(train, horizon, val)
    preds["lstm_residual"] = lstm_residual_model.predict(eval_df, horizon)
    check_no_lookahead(eval_df, preds["lstm_residual"], horizon)
    del lstm_residual_model
    _free_gpu_memory()

    cnn_lstm_residual_model = ResidualCorrected(
        CNNLSTMForecaster(seed=seed, regime=regime), seed=seed, residual_fit_split="oof"
    )
    cnn_lstm_residual_model.fit(train, horizon, val)
    preds["cnn_lstm_residual"] = cnn_lstm_residual_model.predict(eval_df, horizon)
    check_no_lookahead(eval_df, preds["cnn_lstm_residual"], horizon)
    del cnn_lstm_residual_model
    _free_gpu_memory()

    # ALSO includes smart persistence and the convex reference as
    # comparators - not just as the two skill-score denominators
    # (CLAUDE.md rule 4), but as models in their own right in this test.
    # Neither holds GPU memory, but deleting + freeing here too keeps
    # every model in this function following the same pattern rather
    # than making the recurrent ones a special case to remember.
    sp_model = SmartPersistence()
    sp_model.fit(train, horizon)
    preds["smart_persistence"] = sp_model.predict(eval_df, horizon)
    check_no_lookahead(eval_df, preds["smart_persistence"], horizon)
    del sp_model
    _free_gpu_memory()

    convex_model = ConvexCombination()
    convex_model.fit(train, horizon, val)  # w fit on VAL only, never test
    preds["convex_reference"] = convex_model.predict(eval_df, horizon)
    check_no_lookahead(eval_df, preds["convex_reference"], horizon)
    del convex_model
    _free_gpu_memory()

    return preds


def build_cell(array, horizon, train, val, eval_df, nameplate_kw, regime):
    preds = fit_all_models(train, val, eval_df, horizon, SEED, regime)

    common_idx = preds["xgboost"].index
    for name, p in preds.items():
        if name == "xgboost":
            continue
        common_idx = common_idx.intersection(p.index)

    daylight_mask = eval_df.loc[common_idx, "is_daylight"].to_numpy()
    daylight_idx = common_idx[daylight_mask]

    outage_mask = exclusion_mask(array, daylight_idx)
    eval_idx = daylight_idx[~outage_mask.to_numpy()]
    n_excluded_outage = int(outage_mask.sum())

    print(
        f"array={array}  horizon={horizon}  n_intersection(7 models)={len(common_idx)}  "
        f"n_daylight={len(daylight_idx)}  n_excluded_outage={n_excluded_outage}  "
        f"n_eval={len(eval_idx)}"
    )

    y_true = eval_df.loc[eval_idx, "Active_Power"]
    errors = {name: (y_true - p.loc[eval_idx]) for name, p in preds.items()}

    # preds itself is done with once errors are computed - dm_matrix only
    # needs `errors` from here on. Same cleanup pattern as fit_all_models
    # (see _free_gpu_memory's docstring): del the last reference, then
    # free, so nothing from this cell's 7 models lingers into the next.
    del preds
    _free_gpu_memory()

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


def load_existing_cells(out_path):
    """Set of (array, horizon) cells already present in out_path, read
    from the CSV itself rather than tracked separately - the file IS the
    record of what completed. Empty set if out_path does not exist yet.

    Same purpose as scripts/run_seed_sweep.py's skip-if-exists check for
    run JSONs: a cell is only ever written here in one shot, all 21 of
    its pairwise rows at once, right after build_cell returns (see
    main() below) - so a crash mid-run can leave the file missing its
    LAST in-progress cell, never a partially-written one, and it is safe
    to treat "any row for (array, horizon) is present" as "this cell is
    done."
    """
    if not out_path.exists():
        return set()
    existing = pd.read_csv(out_path, usecols=["array", "horizon"])
    return set(zip(existing["array"], existing["horizon"].astype(int)))


def main():
    args = parse_args()
    regime = args.regime
    eval_split = args.eval_split
    split_label = "validation split, 2014" if eval_split == "val" else "TEST split, 2015"

    if not run_self_test():
        print("self-test FAILED - aborting before touching any data")
        sys.exit(1)

    if eval_split == "test":
        print(
            "\n--eval-split test: this run fits on train, uses val ONLY for "
            "early stopping / residual oof / the convex weight w, and "
            "evaluates the Diebold-Mariano error series on the TEST split "
            "(2015). See this script's module docstring, CHANGE LOG "
            "2026-08-09.\n"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    # val output path is UNCHANGED from before --eval-split existed; test
    # gets an explicit _test suffix - a distinct filename, not just a
    # distinct flag value, so it can never overwrite or shadow val's.
    suffix = "" if eval_split == "val" else "_test"
    out_path = RESULTS_DIR / f"table6_dm_{regime}{suffix}.csv"

    done_cells = load_existing_cells(out_path)
    if done_cells:
        print(
            f"resuming {out_path}: {len(done_cells)} (array, horizon) cell(s) "
            f"already present, will be skipped\n"
        )

    array11_matrices = {}
    n_computed = 0
    n_skipped = 0

    for array in sorted(ARRAYS):
        df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
        train, val, test = split_chronological(df)
        train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)

        if eval_split == "test":
            test = add_clearsky_power_to_test(train, test, nameplate_kw, gamma_pdc)
            eval_df = test
        else:
            eval_df = val

        for horizon in HORIZONS:
            if (array, horizon) in done_cells:
                print(f"skip array={array}  horizon={horizon}h  (already in {out_path})")
                n_skipped += 1
                continue

            print(f"\n{'=' * 90}")
            print(f"array={array}  horizon={horizon}h  seed={SEED}  regime={regime}  ({split_label})")
            print("=" * 90)

            start = time.perf_counter()
            pairs_df, hln_df, p_holm_df = build_cell(array, horizon, train, val, eval_df, nameplate_kw, regime)
            elapsed = time.perf_counter() - start
            print(f"  cell done in {elapsed:.1f}s")

            # Written immediately, one cell at a time, rather than held
            # in memory and written once at the end - a crash on a LATER
            # cell (e.g. the array12 h=6 CUDA OOM this fixes) now loses
            # only the in-progress cell, not every cell already computed
            # this run. write_header is re-evaluated per cell rather than
            # cached, since the first successful append is what makes
            # out_path start existing.
            write_header = not out_path.exists()
            pairs_df[CSV_COLUMNS].to_csv(out_path, mode="a", header=write_header, index=False)
            print(f"  appended {len(pairs_df)} rows to {out_path}")

            n_computed += 1
            if array == "array11":
                array11_matrices[horizon] = (hln_df, p_holm_df)

    print(
        f"\n{n_computed} cell(s) computed this run, {n_skipped} skipped "
        f"(already present) - {out_path}"
    )

    if array11_matrices:
        print(f"\n{'=' * 90}")
        print("array11 DM matrices (HLN statistic; row = model_1, column = model_2 in dm_test's sign")
        print("convention - hln_df.loc[a, b] < 0 means a has lower loss than b)")
        print("(cells skipped this run as already-done are not reprinted here)")
        print("=" * 90)
        for horizon, (hln_df, p_holm_df) in sorted(array11_matrices.items()):
            print(f"\n--- array11, h={horizon} : HLN statistics ---")
            print(hln_df.round(3).to_string())
            print(f"\n--- array11, h={horizon} : Holm-adjusted p-values ---")
            print(p_holm_df.round(4).to_string())


if __name__ == "__main__":
    main()
