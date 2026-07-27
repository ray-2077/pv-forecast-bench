# CLAUDE.md - PV Forecasting Benchmark

## Project

Short-term PV power forecasting benchmark on DKASC (Alice Springs) data.

Models: smart persistence, XGBoost, LSTM, CNN-LSTM, CNN-LSTM+XGBoost residual hybrid.

The contribution is NOT the architecture. It is a leakage-controlled evaluation

protocol and a measurement of how much reported accuracy comes from evaluation

choices rather than model capacity. When a tradeoff arises between a better

number and a more defensible protocol, the protocol wins.

## Environment (verified 2026-07-27)

- Windows 11, RTX 3070 Ti Laptop (8 GB VRAM)

- Conda env `pvfc`, Python 3.12.13. Activate with: conda activate pvfc

- Project root: C:\\Users\\Sudhanshu\\projects\\pv-forecast-bench

- torch 2.11.0+cu128 (CUDA available), numpy 2.5.1, pandas 3.0.5,

&#x20; scikit-learn 1.9.0, xgboost 3.3.0, pvlib 0.15.2, statsmodels 0.14.6

These are RECENT major versions. Do not rely on remembered APIs.

pandas 3.x has copy-on-write by default and a different default string dtype.

numpy 2.x and xgboost 3.x both had breaking changes. If unsure about an API,

check the installed version before writing code, not after it fails.

## Non-negotiable methodology rules

1. Chronological splits only. Never shuffle a time series. No random splits.

2. Daylight filtering. Night hours excluded from evaluation, or reported

&#x20;  separately. Predicting zero at night is free accuracy.

3. Scalers and feature statistics fit on TRAINING data only. Never on

&#x20;  validation or test.

4. Skill score vs the CONVEX COMBINATION of climatology and smart persistence
   is the headline metric, not raw RMSE and not skill vs persistence alone.
   Weight w fitted on VALIDATION only, per Yang et al. 2020 Solar Energy
   210:20-37. Skill vs plain persistence is reported alongside, because the
   gap between them is a result: on array11 2014, XGBoost h=6 scores +0.659
   vs persistence but +0.215 vs the convex reference. Persistence collapses
   at long horizons (MBE to -2.15 kW at midday, h=6) because the issue time
   falls near dawn and its k_p is stale.

5. Lagged and oracle feature regimes must never be mixed. Any result using

&#x20;  future weather is labelled explicitly as a perfect-forecast upper bound.

6. XGBoost residual stage is fit on OUT-OF-FOLD TRAINING residuals: an
   expanding window over TRAIN_YEARS (fit on years[:k], predict year k,
   pool the out-of-fold residuals across folds - no fold for the first
   training year, since there is no prior year to expand from). The base
   model is then refit once on ALL of TRAIN_YEARS, with VALIDATION used
   ONLY for early stopping, never to fit the residual stage. See
   src/models/residual.py, ResidualCorrected.fit and _oof_residuals, for
   the exact scheme. Each fold's base model sees less data than the final
   one, so out-of-fold residuals run slightly pessimistic - the correct
   direction to err.

   CHANGE LOG (2026-07-27): this rule originally read "XGBoost residual
   stage is fit on VALIDATION-split residuals, not training" - the
   reasoning at the time was that TRAINING residuals are contaminated
   because the base model has already fitted them. That reasoning was
   correct, but the fix was wrong: fitting the residual stage on
   validation and then EVALUATING it on validation is in-sample
   performance for the residual stage, not merely optimistic - [L1.1] in
   Kapoor & Narayanan's leakage taxonomy. Evidence of the magnitude:
   array11 h6 seed0 went from skill_vs_convex +0.2104 (plain LSTM) to
   +0.5447 (val-fit residual stage), when every genuine architectural
   effect measured elsewhere in this project is 0.01-0.02. That run is
   preserved, not deleted, as
   results/leaked_lstm_residual_array11_h6_lagged_seed0.json (top-level
   key "INVALID_LEAKED": true) as evidence for a protocol-inflation
   table, not as a result. The leaked scheme is still reachable on
   purpose, for that table, via
   ResidualCorrected(residual_fit_split='val') - it raises a warning and
   records leaked_by_design=True in config when used; the default is
   residual_fit_split='oof', the corrected scheme above.

7. Every experiment writes results/<run_id>.json containing config, git commit

&#x20;  hash, random seed, all metrics, and timing.

If asked to write code that violates any of these, refuse and say which rule.

## Research integrity

- NEVER invent, estimate, or placeholder a numeric result. Every number in the

&#x20; paper comes from a real run that produced a real log file.

- If asked to fill in a plausible-looking metric, refuse and say to run the

&#x20; experiment instead.

- Do not tune on the test set. Test set is touched once, at the end.

- A negative result is publishable. Plain XGBoost beating the deep hybrid is an

&#x20; acceptable outcome. Do not massage numbers toward a nicer story.

## Scope control

Hard deadline: 4-6 weeks, 10-15 h/week, single laptop GPU.

If I propose new scope, say honestly whether it fits. If it does not, tell me to

write it in FUTURE_WORK.md and move on. Do not silently expand scope.

## Repo layout

src/data/       loading, resampling, cleaning, clear-sky, splits

src/features/   feature builders (lagged and oracle regimes)

src/models/     all models behind one fit()/predict() interface

src/eval/       metrics, skill score, sky classification, Diebold-Mariano

configs/        YAML experiment configs

results/        run JSONs, VERSION CONTROLLED, this is the audit trail

scripts/        entry points, aggregation into tables and figures

paper/          LaTeX

data/           raw and processed data, GITIGNORED, never commit

## Windows / PowerShell notes

- NEVER write text files with the > redirect. PowerShell writes UTF-16 and

&#x20; Python and git cannot read it. Use System.IO.File WriteAllLines or write

&#x20; from Python.

- Use forward slashes or pathlib.Path in Python. No hardcoded backslash paths.

- Smart App Control is OFF on this machine (it blocked unsigned .pyd files).

- Keep this file ASCII only. No em-dashes, no smart quotes.

## Working style

- I am new to ML research, PyTorch, and Claude Code. Explain before doing.

- One change at a time. Do not refactor unrelated code while fixing something.

- Commit after each working piece, with a message describing what changed.

- Prefer boring, readable code over clever code.

- Set random seeds explicitly everywhere. Log them.


## Data window (decided 2026-07-27 from results/data_audit.csv; revised 2026-07-27
after scripts/audit_dead_periods.py and scripts/diagnose_array17_events.py)
Arrays: 11 (poly-Si, 5.0kW), 12 (mono-Si, 5.1kW), 17 (HIT, 6.3kW). All fixed-mount,
DKASC Alice Springs. These are three co-located ARRAYS, not three sites. They
share one weather station. Their errors are correlated. Do not treat them as
independent samples.
array07 (CdTe, 7.0kW) EXCLUDED: results/dead_period_audit.csv shows 48.4 pct of
its 2014 daylight hours dead (exactly zero output) and a 48-day near-zero run
from 2015-11-14 to 2015-12-31 - both inside the evaluation years (2014
validation, 2015 test). This is a paper result, not a data gap: keep
data/raw/array07_CdTe.csv and its rows in results/dead_period_audit.csv and
results/data_audit.csv as the evidence, do not delete them. See
scripts/audit_dead_periods.py and FUTURE_WORK.md.
Years: train 2011-2013, validate 2014, test 2015. Whole calendar years only, so
no split is seasonally biased. Train starts at 2011, not 2009: array17 was not
installed until 11 March 2010, and all arrays share one training window so the
cross-array comparison is not confounded by differing training length.
Wind_Speed is 19.5 pct NaN in 2016 and 100 pct NaN from 2017. This is why the
window ends at 2015.
