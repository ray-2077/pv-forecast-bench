"""Compute per-row forecast residuals for plain LSTM and LSTM+residual,
array11, h=6, seed=0, lagged regime, validation split (2014) - the raw
data behind paper figure F4 (scripts/build_figures.py).

Neither the committed run JSONs (results/lstm_array11_h6_lagged_seed0.json,
results/lstm_residual_array11_h6_lagged_seed0.json) nor any other
committed CSV stores per-row predictions, only aggregate metrics - so F4
needs its own refit, exactly as PROJECT_CHECKPOINT.md Finding 10's own
+0.2110 -> +0.1768 numbers were produced: two SEPARATE model fits, each
its own set_all_seeds(0) call, mirroring how scripts/run_lstm_dev.py and
scripts/run_residual_dev.py were actually invoked as two independent
processes. This script reproduces that pairing in one place, deliberately
NOT sharing a single fitted base model between the "before" and "after"
predictions - see CLAUDE.md's determinism note (torch.nn.LSTM has no
deterministic CUDA backward pass) for why two separately-seeded fits are
not bit-for-bit identical, same as the two original committed runs.

Pipeline mirrors scripts/run_lstm_dev.py / scripts/run_residual_dev.py
exactly up through predict(): load array -> chronological split -> train-
only clear-sky power gain -> fit -> predict on val. Evaluation rows are
restricted to daylight (src.data.clearsky is_daylight) and documented-
outage hours are dropped (src.eval.exclusions), same as every reported
metric elsewhere in this project - this script does not compute a skill
score or intersect with persistence/climatology/convex, since neither is
needed for a residual-distribution plot, so eval_idx here is only
daylight & not-outage & both models produced a prediction.

residual = y_true - y_pred, matching src/models/residual.py's own sign
convention (_oof_residuals: residual_fold = y_true_fold - fold_pred).

Writes results/f4_residuals_array11_h6_lagged_seed0.csv: columns
timestamp, model (lstm | lstm_residual), y_true, y_pred, residual.

Evaluated on the VALIDATION split (2014) only, per CLAUDE.md's research-
integrity rules - test (2015) is never read here.

Usage:
    python scripts/compute_f4_residuals.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import add_clearsky_power_per_split, load_and_prepare
from src.data.splits import split_chronological
from src.eval.exclusions import exclusion_mask
from src.eval.runner import set_all_seeds
from src.models.base import check_no_lookahead
from src.models.lstm import LSTMForecaster
from src.models.residual import ResidualCorrected

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

ARRAY = "array11"
HORIZON = 6
SEED = 0
REGIME = "lagged"


def fit_and_predict(model_name, make_model, train, val, horizon):
    set_all_seeds(SEED)
    model = make_model()
    print(f"fitting {model_name}...")
    model.fit(train, horizon, df_val=val)
    preds = model.predict(val, horizon)
    check_no_lookahead(val, preds, horizon)
    return preds


def main():
    df, nameplate_kw, gamma_pdc = load_and_prepare(ARRAY, PROCESSED_DIR)
    train, val, test = split_chronological(df)
    train, val, _ = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)
    print(f"n_train={len(train)}  n_val={len(val)} (test not touched)")

    preds_lstm = fit_and_predict(
        "lstm", lambda: LSTMForecaster(seed=SEED, regime=REGIME), train, val, HORIZON
    )
    preds_lstm_residual = fit_and_predict(
        "lstm_residual",
        lambda: ResidualCorrected(
            LSTMForecaster(seed=SEED, regime=REGIME), seed=SEED, residual_fit_split="oof"
        ),
        train,
        val,
        HORIZON,
    )

    common_idx = preds_lstm.index.intersection(preds_lstm_residual.index)
    is_daylight = val.loc[common_idx, "is_daylight"]
    outage_mask = exclusion_mask(ARRAY, common_idx)
    eval_idx = common_idx[is_daylight & ~outage_mask]
    print(
        f"predictions: lstm={len(preds_lstm)}  lstm_residual={len(preds_lstm_residual)}  "
        f"common={len(common_idx)}  daylight+no-outage={len(eval_idx)}"
    )

    y_true = val.loc[eval_idx, "Active_Power"]

    rows = []
    for model_name, preds in [("lstm", preds_lstm), ("lstm_residual", preds_lstm_residual)]:
        y_pred = preds.loc[eval_idx]
        residual = y_true - y_pred
        for ts, yt, yp, r in zip(eval_idx, y_true, y_pred, residual):
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "model": model_name,
                    "y_true": yt,
                    "y_pred": yp,
                    "residual": r,
                }
            )
        print(
            f"{model_name:14s}  mean_residual={residual.mean():+.4f} kW  "
            f"std_residual={residual.std():.4f} kW"
        )

    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"f4_residuals_{ARRAY}_h{HORIZON}_{REGIME}_seed{SEED}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
