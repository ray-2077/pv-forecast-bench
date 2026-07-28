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

The 5-year/4-fold ablation (scripts/rerun_residual_5yr.py, 36 runs) showed
the penalty SHRINKS under more folds but does not vanish, and shrinks more
at short horizons (h=1: 75-96% recovered) than long ones (h=6: 19-43%
recovered) - claim B at short horizons, claim A at long ones.

This script does not run that sweep. What it CAN show cheaply, with no
retraining sweep beyond one extra fold-refit pass: whether the residual
corrector's predicted residual on VALIDATION (the split it is evaluated
on) has any positive relationship with the ACTUAL residual there, versus
on the out-of-fold TRAINING rows it was fit on, and - the addition for
the fold-count question above - whether the corrector's predicted
correction is properly SIZED relative to that relationship.

The sizing question has a closed form. Adding a correction p to a base
model with residual r changes MSE by -2*rho*sigma_r*sigma_p + sigma_p^2
(rho = corr(p, r)). This is negative (helps) only when
sigma_p < 2*rho*sigma_r - the BREAK-EVEN ratio. If the out-of-fold
correlation is high but the validation correlation is low, and the
predicted-residual std is well above 2*rho_val*sigma_r_val, the corrector
is not merely "learning nothing" on validation - it is OVERCONFIDENT: it
applies corrections sized for a relationship stronger than the one that
actually transfers.

Runs the production fit path (ResidualCorrected.fit, residual_fit_split=
'oof') once per (array, horizon, train_years) cell, then calls the
private _oof_residuals a second time on the same (train, val) to recover
the pooled out-of-fold feature matrix/residuals for the correlation below
(fit() itself does not retain them - only the fitted residual model and
n_oof_residuals/oof_years). This second call is deterministic given the
seeds already set: same fold splits, same fold clone hyperparameters,
same base_model.seed. It costs one extra pass of fold-model trainings per
cell (2 folds for the 3-year window, 4 for the 5-year window), which is
why this script targets a handful of cells rather than a full grid.

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
SEED = 0
REGIME = "lagged"

TRAIN_YEARS_3YR = (2011, 2012, 2013)
TRAIN_YEARS_5YR = (2009, 2010, 2011, 2012, 2013)

# (horizon, train_years, label) - h1 is not in the requested set, only the
# two horizons asked for (h3, the smallest cell that reproduces the
# penalty, and h6, where the 5-year sweep showed the penalty survives).
CONFIGS = [
    (3, TRAIN_YEARS_3YR, "h3  3yr/2fold"),
    (3, TRAIN_YEARS_5YR, "h3  5yr/4fold"),
    (6, TRAIN_YEARS_3YR, "h6  3yr/2fold"),
    (6, TRAIN_YEARS_5YR, "h6  5yr/4fold"),
]


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1])


def diagnose_one(array, horizon, seed, train_years, label):
    print(f"--- {label}  (array={array}  horizon={horizon}  seed={seed}  "
          f"train_years={train_years}) ---")

    set_all_seeds(seed)

    df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
    train, val, test = split_chronological(df, train_years=train_years)
    train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)
    print(f"n_train={len(train)}  n_val={len(val)}")

    base_model = LSTMForecaster(seed=seed, regime=REGIME)
    residual_model = ResidualCorrected(base_model, seed=seed, residual_fit_split="oof")

    print("fitting base model + out-of-fold residual stage (production path)...")
    residual_model.fit(train, horizon, val)
    print(
        f"  n_oof_residuals={residual_model.n_oof_residuals}  "
        f"oof_years={residual_model.oof_years} "
        f"({len(residual_model.oof_years)} folds)"
    )

    # Recover the pooled out-of-fold feature matrix / actual residuals the
    # residual stage was fit on - fit() does not retain these itself. See
    # module docstring: deterministic given the seeds already set above, so
    # this reproduces exactly what fit() used, at the cost of retraining the
    # fold clones a second time.
    print("recomputing out-of-fold residuals (same fold scheme, for the correlation below)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        X_oof, residual_oof_actual = residual_model._oof_residuals(train, horizon, val)
    residual_oof_pred = residual_model._residual_model.predict(X_oof[residual_model._feature_cols])

    # Actual residual on VALIDATION - the split the corrected model is
    # evaluated on. Uses the same private helper the leaked
    # residual_fit_split='val' scheme uses to CONSTRUCT its residuals - here
    # it is only being used to READ OFF the actual residual for comparison,
    # never to fit anything, so this does not reintroduce the rule-6 leak.
    X_val, residual_val_actual = residual_model._val_residuals(val, horizon)
    residual_val_pred = residual_model._residual_model.predict(X_val[residual_model._feature_cols])

    corr_oof = pearson(residual_oof_pred, residual_oof_actual)
    corr_val = pearson(residual_val_pred, residual_val_actual)

    sigma_p_val = float(np.std(residual_val_pred, ddof=1))
    sigma_r_val = float(np.std(residual_val_actual, ddof=1))
    ratio = sigma_p_val / sigma_r_val
    breakeven = 2 * corr_val

    print()
    print(f"  corr(pred, actual)  out-of-fold = {corr_oof:+.4f}")
    print(f"  corr(pred, actual)  validation  = {corr_val:+.4f}")
    print(f"  sigma_p (pred residual std, val)   = {sigma_p_val:.4f} kW")
    print(f"  sigma_r (actual residual std, val) = {sigma_r_val:.4f} kW")
    print(f"  sigma_p / sigma_r                  = {ratio:.3f}")
    print(f"  break-even ratio (2 * rho_val)      = {breakeven:.3f}")
    if ratio > breakeven:
        print(f"  -> OVERCONFIDENT: correction is {ratio / breakeven if breakeven > 0 else float('inf'):.1f}x "
              f"the size the validation correlation justifies")
    else:
        print("  -> correction magnitude is within the break-even bound")
    print()

    return {
        "label": label,
        "corr_oof": corr_oof,
        "corr_val": corr_val,
        "sigma_p_val": sigma_p_val,
        "sigma_r_val": sigma_r_val,
        "ratio": ratio,
        "breakeven": breakeven,
    }


def main():
    print(f"array={ARRAY}  seed={SEED}  regime={REGIME}\n")

    results = []
    for horizon, train_years, label in CONFIGS:
        results.append(diagnose_one(ARRAY, horizon, SEED, train_years, label))

    print("=" * 100)
    print(f"{'config':16s} {'rho_oof':>9s} {'rho_val':>9s} {'sigma_p':>9s} "
          f"{'sigma_r':>9s} {'ratio':>8s} {'breakeven':>10s} {'status':>14s}")
    print("-" * 100)
    for r in results:
        status = "overconfident" if r["ratio"] > r["breakeven"] else "within bound"
        print(
            f"{r['label']:16s} {r['corr_oof']:+9.4f} {r['corr_val']:+9.4f} "
            f"{r['sigma_p_val']:9.4f} {r['sigma_r_val']:9.4f} {r['ratio']:8.3f} "
            f"{r['breakeven']:10.3f} {status:>14s}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
