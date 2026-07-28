"""Cheap diagnostic for the seed-sweep finding that lstm_residual scores
lower than plain lstm on skill_vs_convex in every one of 18 array x horizon
cells (results/seed_sweep_summary.csv, all deltas -0.024 to -0.046).

Two competing explanations:
  A. residual correction does not help on this problem
  B. the out-of-fold scheme (CLAUDE.md rule 6) is weak with only 2 folds
     on 3 TRAIN_YEARS (2011-2013) - the fold predicting 2013 saw only
     2011-2012, while the deployed base model saw all three years, so the
     residual XGBoost may be fitting fold-specific base-model errors that
     the final, stronger base model does not make.

This script does not distinguish A from B on its own (that needs the
5-year/4-fold rerun in scripts/rerun_residual_5yr.py - see that script's
docstring). What it CAN show cheaply, with no retraining sweep: whether
the residual corrector's predicted residual on VALIDATION (the split it
is evaluated on) has any positive relationship with the ACTUAL residual
there, versus on the out-of-fold TRAINING rows it was fit on. If the
out-of-fold correlation is high but the validation correlation is near
zero or negative, the corrector learned something that does not
transfer to the deployed base model's actual errors - consistent with
explanation B (or with A, but at minimum shows the oof-fitted signal is
non-transferable regardless of which explanation is right).

Fixed to array11, h=3, seed=0, TRAIN_YEARS=(2011,2012,2013) (the current
default, src/data/splits.py) - the smallest single cell that reproduces
the reported penalty (results/lstm_residual_array11_h3_lagged_seed0.json:
skill_vs_convex 0.2539 vs lstm's 0.2786, a -0.025 delta, in line with the
other 17 cells).

Refits the base LSTM and the two-fold out-of-fold residual scheme once
(ResidualCorrected.fit, exactly the production path - see
src/models/residual.py), then calls the private _oof_residuals a second
time on the same (train, val) to recover the pooled out-of-fold feature
matrix/residuals for the correlation below (fit() itself does not retain
them - only the fitted residual model and n_oof_residuals/oof_years). This
second call is deterministic given the seeds already set: same fold
splits, same fold clone hyperparameters, same base_model.seed. It costs
one extra pair of fold-model trainings ( a few seconds each), which is
why this script targets one cell rather than the full grid.

Usage:
    python scripts/diagnose_residual_signal.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import add_clearsky_power_per_split, load_and_prepare
from src.data.splits import split_chronological
from src.eval.runner import set_all_seeds
from src.models.lstm import LSTMForecaster
from src.models.residual import ResidualCorrected

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ARRAY = "array11"
HORIZON = 3
SEED = 0
REGIME = "lagged"


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print(f"array={ARRAY}  horizon={HORIZON}  seed={SEED}  regime={REGIME}  "
          f"(TRAIN_YEARS from src/data/splits.py, current default)\n")

    set_all_seeds(SEED)

    df, nameplate_kw, gamma_pdc = load_and_prepare(ARRAY, PROCESSED_DIR)
    train, val, test = split_chronological(df)
    train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)
    print(f"n_train={len(train)}  n_val={len(val)}\n")

    base_model = LSTMForecaster(seed=SEED, regime=REGIME)
    residual_model = ResidualCorrected(base_model, seed=SEED, residual_fit_split="oof")

    print("fitting base model + out-of-fold residual stage (production path)...")
    residual_model.fit(train, HORIZON, val)
    print(
        f"  n_oof_residuals={residual_model.n_oof_residuals}  "
        f"oof_years={residual_model.oof_years}\n"
    )

    # Recover the pooled out-of-fold feature matrix / actual residuals the
    # residual stage was fit on - fit() does not retain these itself. See
    # module docstring: deterministic given the seeds already set above, so
    # this reproduces exactly what fit() used, at the cost of retraining the
    # two fold clones a second time.
    print("recomputing out-of-fold residuals (same fold scheme, for the correlation below)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        X_oof, residual_oof_actual = residual_model._oof_residuals(train, HORIZON, val)
    residual_oof_pred = residual_model._residual_model.predict(X_oof[residual_model._feature_cols])

    # Actual residual on VALIDATION - the split the corrected model is
    # evaluated on. Uses the same private helper the leaked
    # residual_fit_split='val' scheme uses to CONSTRUCT its residuals - here
    # it is only being used to READ OFF the actual residual for comparison,
    # never to fit anything, so this does not reintroduce the rule-6 leak.
    X_val, residual_val_actual = residual_model._val_residuals(val, HORIZON)
    residual_val_pred = residual_model._residual_model.predict(X_val[residual_model._feature_cols])

    corr_oof = pearson(residual_oof_pred, residual_oof_actual)
    corr_val = pearson(residual_val_pred, residual_val_actual)

    mean_abs_pred_val = float(np.mean(np.abs(residual_val_pred)))
    mean_abs_actual_val = float(np.mean(np.abs(residual_val_actual)))

    print("=" * 72)
    print(f"corr(predicted residual, actual residual)  out-of-fold (train) = {corr_oof:+.4f}")
    print(f"corr(predicted residual, actual residual)  validation          = {corr_val:+.4f}")
    print()
    print(f"mean |predicted residual|  validation = {mean_abs_pred_val:.4f} kW")
    print(f"mean |actual residual|     validation = {mean_abs_actual_val:.4f} kW")
    print("=" * 72)

    if corr_oof > 0.15 and corr_val < 0.05:
        print(
            "\nout-of-fold correlation is real but validation correlation is "
            "near zero/negative: the corrector learned a pattern that does "
            "not transfer to the deployed base model's actual errors."
        )
    else:
        print(
            "\nno clean out-of-fold-strong / validation-weak split - "
            "inspect the numbers above directly rather than relying on "
            "this script's threshold."
        )


if __name__ == "__main__":
    main()
