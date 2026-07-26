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

4. Skill score vs smart persistence is the headline metric, not raw RMSE.

5. Lagged and oracle feature regimes must never be mixed. Any result using

&#x20;  future weather is labelled explicitly as a perfect-forecast upper bound.

6. XGBoost residual stage is fit on VALIDATION-split residuals, not training.

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

