# PROJECT CHECKPOINT

Written 2026-07-28, covering everything from environment setup to the
90-run seed sweep. Purpose: a self-contained record of what was built,
what was found, and WHY each decision was made, so the paper can be
written from this document plus the repo without reconstructing the
reasoning from scratch.

ASCII only. Numbers here are copied from real runs; none are estimated.
Where a number is stale (superseded by a later config change) it is
marked STALE and the reason given.

---

## 0. PROJECT IDENTITY

Working title:
"How Much of It Is the Model? A Protocol-Controlled Evaluation of Hybrid
Deep Learning for Short-Term PV Power Forecasting"

Core claim: the architecture is NOT the contribution. The contribution is
a leakage-controlled evaluation protocol, and a measurement of how much
reported accuracy in this literature comes from evaluation choices rather
than model capacity.

Repo: https://github.com/ray-2077/pv-forecast-bench (private; make public
at camera-ready, after any double-blind review is done)

Research questions:
- RQ1 Component attribution. Which parts of the hybrid contribute, and
  does the residual stage justify its complexity?
- RQ2 Protocol sensitivity (HEADLINE). How much does reported accuracy
  change under common evaluation choices?
- RQ3 Conditional performance. Clear skies vs variable cloud.
- RQ4 Cost-effectiveness. Accuracy gain per unit training compute.

---

## 1. ENVIRONMENT (verified working)

- Windows 11, RTX 3070 Ti Laptop 8 GB, driver 595.97 (CUDA 13.2)
- Miniconda, conda env `pvfc`, Python 3.12.13, conda-forge channel only
- torch 2.11.0+cu128 (CUDA available: True), numpy 2.5.1, pandas 3.0.5,
  scikit-learn 1.9.0, xgboost 3.3.0, pvlib 0.15.2, statsmodels 0.14.6,
  pyarrow 25.0.0
- Claude Code 2.1.220, native install at ~/.local/bin/claude
- Project root: C:\Users\Sudhanshu\projects\pv-forecast-bench

Environment gotchas encountered (worth one line in Experimental Setup):
- conda 26.x blocks Anaconda default channels behind a Terms of Service
  acceptance. Solved by using conda-forge exclusively with
  --override-channels. Also keeps the project inside its free/open-tools
  constraint.
- Windows Smart App Control had to be DISABLED. It blocks unsigned
  compiled Python extensions (.pyd), so pandas failed to import with
  "An Application Control policy has blocked this file". SAC has no
  exclusion list and cannot be re-enabled without reinstalling Windows.
- PowerShell `>` redirection writes UTF-16, which git treats as binary
  and Python cannot read. All file writing is done from Python with
  explicit encoding='utf-8', newline='\n'.
- torch.nn.LSTM has NO deterministic CUDA backward pass. Full determinism
  is only achievable on CPU. Runs record
  full_determinism_achieved=False honestly. THIS IS WHY THE 5-SEED DESIGN
  MATTERS and belongs in Experimental Setup.

---

## 2. DATA

### Source
DKASC (Desert Knowledge Australia Solar Centre), Alice Springs.
Site: latitude -23.767, longitude 133.867, altitude 558 m,
timezone Australia/Darwin (UTC+9:30, no daylight saving).

Downloaded per-array "Single Technologies (full data set with weather
data)" CSVs, ~300 MB each, native 5-minute resolution, spanning
2008-09-12 to 2025-08-23.

IMPORTANT: every DKASC export starts at SITE inception (2008-09-12)
regardless of when the individual array was installed. A 2008 start date
in the file does NOT mean the array existed in 2008. This caused a
misidentification during the project (see Section 8, error 5).

### Arrays used
All fixed-mount, tilt 20 deg, azimuth 0 deg (solar north), per DKASC's
own technology pages.

| Key      | Site | Manufacturer | Tech     | kW  | Slug             |
|----------|------|--------------|----------|-----|------------------|
| array11  | 11   | BP Solar     | poly-Si  | 5.0 | dka-m5-c-phase   |
| array12  | 12   | BP Solar     | mono-Si  | 5.1 | dka-m5-b-phase   |
| array17  | 17   | Sanyo        | HIT      | 6.3 | dka-m4-b-phase   |

Dropped: array07 (site 7, First Solar, CdTe, 7.0 kW, dka-m6-a-phase).
See Finding 6.

WORDING CAUTION FOR THE PAPER: these are three co-located ARRAYS, not
three sites. They share ONE weather station. Their forecast errors are
correlated, so they are not independent samples. Do not average across
them as if they were, and do not describe the study as multi-site.
Evidence: the data audit produced byte-identical coverage, gap and NaN
statistics for all arrays, because one logger records all of them.

### Window and splits
Processed data: hourly, 2009-01-01 to 2015-12-31, 61,344 rows per array
(7 years, only 2012 is a leap year: 6*8760 + 8784 = 61344).

Splits (chronological, never shuffled):
- train 2011, 2012, 2013
- validate 2014
- test 2015 (NEVER TOUCHED to date except for the dead-period data
  quality audit, which read availability only and computed no forecast
  metric; this is recorded deliberately)

Why training starts at 2011 and not 2009: array17 was installed
11 March 2010. All arrays must share an identical training window or the
cross-array comparison is confounded by training length.

TRAIN_YEARS remains configurable in src/data/splits.py specifically so a
training-length ablation can vary it on arrays 11 and 12, which have
clean data back to 2009. VAL_YEARS and TEST_YEARS must never be varied.

Whole calendar years only, so no split is seasonally biased. This matters:
a half-year test set would confound test error with season.

### Documented exclusions
src/eval/exclusions.py, KNOWN_OUTAGES:
- array17, 2015-06-05 to 2015-06-09: DKASC published maintenance note —
  sites 4, 5, 17, 20, 22, 34, 35 were switched off by unknown persons
  before a long weekend and not discovered until staff returned.

This exclusion is derived from EXTERNAL METADATA (DKASC's published
notes), not from model performance. It is applied identically to every
model and the count is recorded in every run JSON as n_excluded_outage.
It must be declared in the paper.

### Other documented array17 events (verified in the data)
- Install completed 11 March 2010. Jan 2009 - Feb 2010 shows ~0.04 kW
  mean power, performance ratio 0.0001 (noise floor). March 2010 jumps
  to 2.39 kW.
- Inverter replaced approximately July 2013 (SMA SMC 6000A, correcting an
  earlier published SMC 7000TL). Shows as GRADUAL decline in the 99th
  percentile of power, not a step: 2010 6.12 kW, 2011 5.95, 2012 5.98,
  2013 5.81, 2014 5.71, 2015 5.68. Judged to be real degradation /
  re-rating, roughly 1.4%/yr, not a level-shift artifact.

---

## 3. PIPELINE ARCHITECTURE

```
src/data/loader.py        load raw CSV, localise tz, physical range
                          filtering, 5-min -> hourly resample
src/data/clearsky.py      solar position, clear-sky GHI, daylight mask,
                          k_ghi
src/data/clearsky_power.py clear-sky POWER, temperature climatology, gain
src/data/splits.py        chronological split, TRAIN/VAL/TEST years
src/data/pipeline.py      shared setup used by all run scripts
src/features/build.py     tabular feature matrix, lagged/oracle regimes
src/features/sequences.py sequence tensors for recurrent models
src/features/scaling.py   Scaler, fit-once, training data only
src/models/base.py        BaseForecaster interface + check_no_lookahead
src/models/persistence.py SmartPersistence
src/models/climatology.py Climatology, ConvexCombination
src/models/xgb.py         XGBForecaster
src/models/lstm.py        LSTMForecaster
src/eval/metrics.py       mae, rmse, nrmse, mbe, skill_score, all_metrics
src/eval/runner.py        run JSON writer, environment capture, seeding
src/eval/exclusions.py    KNOWN_OUTAGES, exclusion_mask
```

### Key conventions (these ARE the protocol)

ALIGNMENT: feature matrices and predictions are indexed by TARGET TIME t.
A row at target time t for horizon h is the input to a forecast ISSUED at
t-h. Any observed quantity in that row must come from t-h or earlier.
Deterministic quantities (solar geometry, clear-sky power, calendar
encodings) are permitted at t itself because they are computable in
advance.

HOUR LABELLING: hour-beginning, closed='left', label='left'. The 12:00
row is the mean over 12:00-12:55.

SOLAR POSITION vs CLEAR-SKY IRRADIANCE ARE TREATED DIFFERENTLY ON
PURPOSE:
- solar position is evaluated at the hour MIDPOINT (12:30 for the 12:00
  row). It is a FEATURE describing the hour.
- clear-sky irradiance is computed at 5-minute resolution and AVERAGED to
  hourly, matching how the measurements were aggregated. It is being
  COMPARED against an hourly mean measurement.
Getting this wrong was Finding 1.

DROPPED COLUMNS (leakage traps, documented in loader.py):
- Active_Energy_Delivered_Received: cumulative counter derived from power
- Performance_Ratio: derived from power and irradiance, encodes target
- Current_Phase_Average: derived from power

### Feature regimes
- LAGGED: only information available at issue time t-h, plus
  deterministic quantities at t.
- ORACLE: adds measured weather AT TARGET TIME t, prefixed `oracle_` so
  it can never be mistaken for a legitimate feature in a
  feature-importance plot. Explicitly a perfect-forecast UPPER BOUND.
The two must never be mixed in one matrix.

### Lag naming (changed mid-project, see Finding 3)
Suffixes are stable across horizons:
- `_issue` (shift h), `_issue_m1` (shift h+1), `_issue_m2` (shift h+2)
- `_daily` (FIXED shift of 24, so it stays aligned to hour-of-day at
  every horizon; asserted h <= 24)
Old names (`_lag1` etc.) were horizon-relative and would have misled any
reader of the feature-importance figure.

### Rolling statistics: two different kinds, on purpose
- Active_Power: WALL-CLOCK windows (3h, 24h). Power is 0 at night, not
  NaN, so a wall-clock window is well defined at any issue time.
- k_p and k_ghi: LAST-N-VALID-OBSERVATIONS windows, named
  `k_p_last3obs_mean` etc. These are NaN at night by construction, so a
  wall-clock window ending pre-dawn contains zero valid samples.
  DO NOT "simplify" this into ffill-then-rolling-std: the std of repeated
  forward-filled values is exactly 0, which would tell the model
  "perfectly stable sky" when the truth is "no observation since
  yesterday evening". That is a fabricated feature value, worse than a
  dropped row.

---

## 4. PROTOCOL RULES (CLAUDE.md, current state)

1. Chronological splits only. Never shuffle a time series.
2. Daylight filtering. Night hours excluded from evaluation or reported
   separately. Predicting zero at night is free accuracy.
3. Scalers and feature statistics fit on TRAINING data only.
4. **[REVISED]** Skill score vs the CONVEX COMBINATION of climatology and
   smart persistence is the headline metric, with weight w fitted on
   VALIDATION only, per Yang et al. 2020. Skill vs plain persistence is
   reported alongside, because the GAP BETWEEN THEM IS A RESULT.
5. Lagged and oracle regimes never mixed. Oracle results explicitly
   labelled as an upper bound.
6. XGBoost residual stage fits on VALIDATION-split residuals.
7. Every experiment writes results/<run_id>.json with config, git commit,
   git_dirty flag, seed, metrics, timings.

Research integrity: no invented numbers, ever. Test set touched once, at
the end. A negative result is publishable.

Daylight threshold: solar_elevation > 10 deg (zenith < 80 deg). Note this
is STRICTER than the zenith < 85 deg that Yang et al. (2020) give as
typical practice. The threshold is itself an RQ2 knob.

---

## 5. FINDINGS (with reasoning chains)

### Finding 1: clear-sky sampling mismatch
TRIGGER: median k_ghi was 1.02 and clear days ~1.12, when a well-specified
clear-sky model should give ~1.0 on clear days.

FIRST HYPOTHESES, both wrong: (a) uniform Linke turbidity bias for a very
clean desert atmosphere; (b) a solar-position timing error.

EVIDENCE THAT DISCRIMINATED: hour 7 mean k_ghi was 0.920 (BELOW 1) while
hour 18 was 1.499 (ABOVE 1). A sign flip. Turbidity bias would depress or
raise both ends together. A timing bug was ruled out separately: the mean
hour of max measured GHI was 11.98 vs 12.00 for modelled, agreeing to
within a minute.

ACTUAL CAUSE: measured GHI is an hourly MEAN of twelve 5-minute values;
ghi_cs was a single INSTANTANEOUS value at the hour midpoint. Near the
horizon irradiance changes steeply within the hour, so the two are not
comparable. Morning: midpoint sun sits above the hour's average, model
too high, k < 1. Evening: reverse.

FIX: compute clear-sky at 5-minute resolution and average to hourly with
the same closed='left', label='left' convention as the measurements.

RESULT: morning-vs-afternoon difference at matched elevation collapsed
from +0.02..+0.06 to within 0.015 across the 20-80 degree bins, which is
where essentially all generation happens.

RESIDUAL, deliberately not fixed: hours 7, 18 and the 10-20 degree
elevation bin still differ. Two real physical causes the Ineichen model
cannot represent — twilight diffuse (pvlib drives clear-sky to zero at
the geometric horizon; the real sky keeps scattering) and the
pyranometer's ~2.7 W/m2 thermal offset, which is 7-10% of an hourly
ghi_cs of only 25-40 W/m2. Out of scope; goes in Limitations.

PAPER: Methodology 4.1, and this is a concrete instance of the RQ2 thesis
— an evaluation-construction error that inflates or deflates a headline
index without any model being involved. Yang et al. (2020) independently
document the same low-elevation instability and the zenith-filter
convention that exists because of it. WE FOUND IT EMPIRICALLY BEFORE
READING THAT PAPER, which is worth stating.

Script: scripts/diagnose_clearsky_bias.py (NOTE: retained unchanged for
provenance; run against the original 2012-2013 window).

### Finding 2: night-hour inclusion, with a closed form
OBSERVED (array11, XGBoost h=3, validation 2014):
- daylight nRMSE 8.89%, all-hours nRMSE 5.85%, ratio 0.658
- daylight skill 0.530, all-hours skill 0.532

DERIVATION: if night errors are ~0 (persistence predicts ~0, truth ~0),
then RMSE_all = RMSE_day * sqrt(N_day / N_all).
sqrt(3799/8760) = 0.6586. Observed 0.66.

INTERPRETATION: including night hours deflates reported nRMSE by ~34% at
this site, and the deflation factor is a property of LATITUDE AND SEASON,
not of the model. Skill score is essentially immune (changes by 0.002).

WHY THIS IS STRONG: it is closed-form and a reviewer can check it in one
line. It also demonstrates that the metric Yang et al. recommend is the
one that resists the most common inflating protocol choice.

PAPER: Table 4 (protocol inflation), and probably the Introduction.

CAVEAT / OPEN ITEM: the current all_hours metrics block in run JSONs
intersects XGBoost, persistence, climatology AND convex. Climatology has
no prediction at always-night cells, so the block shrank from 8694 to
4247 rows and NO LONGER SPANS A 24-HOUR CYCLE. It cannot answer the
night-inclusion question as written. NEEDS a separate
`all_hours_vs_persistence` metrics subset intersecting only
XGBoost + persistence. This is a known outstanding fix.

### Finding 3: feature coverage collapse at long horizons
TRIGGER: build_features drops rows with any NaN feature. k_p is NaN at
night by construction. At h=6 a midday target is issued near dawn.

MEASURED (array11, 2009-2014, daylight target retention):
- initial:                h1 75.6%, h3 56.5%, h6 28.0%
- after ffill of k_p/k_ghi lags (limit 24h):
                          h1 85.1%, h3 66.0%, h6 37.5%
- after last-N-valid-obs rolling stats:
                          h1 99.0%, h3 99.0%, h6 99.0%

THE DANGEROUS PART: at h=6 before the fix, targets at hours 8-13 were
dropped on essentially EVERY DAY (~2191 each, i.e. every day in the
period). Models would have been trained and evaluated on afternoon
targets only, while smart persistence forecast all of them. Every h=6
skill score would have compared two different populations and NOTHING IN
THE MODEL OUTPUT WOULD HAVE LOOKED WRONG.

FIXES: (a) forward-fill k_p/k_ghi with limit 24h before shifting, adding
`k_p_hours_stale` and `k_p_is_stale` so the model can discount stale
information the way the baseline implicitly does; (b) last-N-valid-obs
rolling windows as described in Section 3.

PAPER: Methodology 4.1, and a strong argument for reporting the number of
evaluated samples per model per horizon — which almost no paper does.

Scripts: scripts/check_feature_coverage.py

### Finding 4: smart persistence degrades structurally at long horizons
MEASURED fallback_fraction (proportion of daylight predictions using a
forward-filled rather than directly observed k_p), array11, 2014:
h=1 3.3%, h=3 22.6%, h=6 51%.

MEASURED MBE by hour, persistence, h=6, array11: swings from +0.01 kW to
-2.15 kW at midday on a 5 kW array.

CAUSE: at h=6 a midday target is issued near dawn, so the persisted k_p
is stale, often forward-filled from the previous evening.

WHAT WE TESTED AND REJECTED: stricter daylight thresholds. Raising the
elevation cutoff 10 -> 15 -> 20 degrees made h=3 and h=6 WORSE (0.95 ->
0.98 -> 1.01 kW and 1.55 -> 1.60 -> 1.66 kW), because it removes cheap
low-error edge hours and concentrates the average on expensive midday
ones. The dominant error is NOT low-sun clear-sky bias.

CONSEQUENCE: skill scores measured against a structurally broken baseline
are inflated. Models carry `k_p_is_stale` and can learn to discount it;
persistence cannot. A large share of apparent h=6 "hybrid gain" would
have been the model handling staleness better than a naive baseline, and
would have been attributed to architecture.

Scripts: scripts/diagnose_baseline_error.py

### Finding 5: the reference forecast changes the headline (HEADLINE RESULT)
Implemented Climatology (mean k_p per training month x hour) and
ConvexCombination (w * persistence + (1-w) * climatology, w fitted on
VALIDATION by grid search in 0.01 steps), per Yang et al. (2020), who
recommend exactly this reference.

MEASURED (validation 2014, daylight, XGBoost, TRAIN_YEARS = 2011-2013):

| array   | h | w    | skill vs persistence | skill vs convex |
|---------|---|------|----------------------|-----------------|
| array11 | 1 | 0.77 | +0.253               | +0.200          |
| array11 | 3 | 0.25 | +0.526               | +0.276          |
| array11 | 6 | 0.04 | +0.652               | +0.194          |
| array12 | 1 | 0.77 | +0.244               | +0.190          |
| array12 | 3 | 0.29 | +0.515               | +0.275          |
| array12 | 6 | 0.05 | +0.639               | +0.201          |
| array17 | 1 | 0.83 | +0.166               | +0.122          |
| array17 | 3 | 0.50 | +0.341               | +0.183          |
| array17 | 6 | 0.31 | +0.386               | +0.132          |

TWO SEPARATE RESULTS IN THIS TABLE:

(a) MAGNITUDE. At h=6, roughly two-thirds of apparent skill is an
artifact of a broken baseline: +0.652 becomes +0.194. Consistent across
three different module technologies.

(b) SHAPE — arguably the stronger finding. Against persistence, skill
rises MONOTONICALLY with horizon (0.253, 0.526, 0.652), which reads as
"our model's advantage grows at longer horizons" and is a claim made
throughout the hybrid literature. Against the proper reference, skill is
NON-MONOTONIC and peaks at h=3 (0.200, 0.276, 0.194). The apparent
horizon trend is entirely a baseline artifact. Same data, same model,
same run; only the reference changed.

At h=6, w = 0.04 means the convex reference is effectively pure
climatology — persistence is nearly worthless there.

NOTE ON array17: its w stays higher at every horizon (0.83/0.50/0.31).
This is NOT the array07 pathology. Normalised by nameplate, array17's
persistence is BETTER in relative terms (h=3 nRMSE 14.4% vs array11's
19.1%), so the optimiser leans on it because it works. Plausible reason:
HIT's better temperature coefficient (-0.30%/degC vs -0.40%) gives less
thermally-driven scatter in k_p, so the clear-sky index is more
persistent. Not diagnosed further; one sentence at most.

PAPER: this is Table 4 and probably a figure. It is the paper's single
strongest evidence.

Scripts: scripts/compare_references.py
Data: results/reference_comparison.csv, results/reference_mbe_by_hour.csv

### Finding 6: array07 was dead, and the audit could not see it
TRIGGER: XGBoost scored NEGATIVE skill on array07 at every horizon, and
its convex weight sat at 0.99/0.96/0.90 while arrays 11/12 fell to ~0.05.

REASONING: climatology and XGBoost both learn a level from training and
apply it to 2014; persistence uses a recent observation and self-corrects.
A weight near 1.0 means climatology is useless. That pattern points at a
level shift between training and validation.

EVIDENCE: performance ratio (mean Active_Power / mean GHI, daylight only)
by year for array07: 0.0060, 0.0058, 0.0057, 0.0056, 0.0051, then 0.0019
in 2014 — a 67% drop. Monthly breakdown showed EXACTLY ZERO for months
3-9 of 2014. Confirmed in the RAW CSV: 48-97% of clearly-daytime records
(GHI > 200) have Active_Power exactly 0 across those months. Not a
pipeline artifact.

Also found: a 48-day near-zero run from 2015-11-14 to 2015-12-31, i.e.
inside the TEST year, 14.89% of 2015 daylight hours.

THE IMPORTANT PART: results/data_audit.csv PASSED array07 2014 with
99.99% coverage and 0.00% NaN. The logger kept recording, and what it
recorded was zeros. ZERO IS NOT NaN. A completeness-based audit is
structurally blind to a healthy sensor reporting a dead array.

SECOND BLIND SPOT, found immediately after: the dead-period audit
initially tested for EXACTLY zero. array17 logs ~0.04 kW standby during
its pre-install period and the June 2015 outage, so neither triggered a
FAIL despite a confirmed 3-day outage inside the test year. Threshold
changed to "below 1% of nameplate", which then correctly flagged
array17 2015 with longest_below_1pct_run = 5 days [2015-06-05 ..
2015-06-09] — matching the documented outage to the day — and also
flipped array07 2013 and 2015 from PASS to FAIL.

DECISION: array07 excluded. Replaced with array17 (Sanyo HIT), which also
adds genuine technology diversity — arrays 11 and 12 are near-duplicates
(performance ratios 0.0040 and 0.0041; skill within 0.02 at every
horizon), so the original three arrays were really one silicon array
measured twice plus a broken CdTe one.

data/raw/array07_CdTe.csv and its audit rows are RETAINED as evidence.

PAPER: Data section, and strong Introduction motivation. DKASC is a
heavily used open dataset; any study taking array 7 through 2014 on the
strength of a completeness check inherited seven months of zeros.

Scripts: scripts/diagnose_array_level_shift.py (NOTE: provenance only,
run against the original 2009-2013 window),
scripts/audit_dead_periods.py, scripts/diagnose_array17_events.py
Data: results/dead_period_audit.csv

### Finding 7: training length barely matters at this site
The training window was narrowed from 2009-2013 (1826 days) to 2011-2013
(1096 days) for array17 compatibility. array11 h=3 XGBoost skill_vs_convex
moved from +0.276 to +0.276; skill_vs_persistence from +0.530 to +0.526.

This is evidence AGAINST the meta-analytic claim (Nguyen & Musgens) that
solar forecast skill improves with training length up to ~2000 days, at
least at this site and horizon. It also de-risks the window decision
entirely.

A proper training-length ablation is cheap and already possible: only
arrays 11 and 12 have clean data back to 2009, TRAIN_YEARS is a parameter.

### Finding 8: architecture matters only at long horizons (RQ1)
90-run seed sweep: 2 models x 3 arrays x 3 horizons x lagged x 5 seeds.
21 minutes wall clock, 0 failures. All 90 run JSONs committed.

skill_vs_convex, mean +/- std across 5 seeds, validation 2014, daylight:

| array   | h | XGBoost           | LSTM              | LSTM - XGB |
|---------|---|-------------------|-------------------|------------|
| array11 | 1 | +0.1967 +/-0.0023 | +0.1938 +/-0.0021 | -0.0029    |
| array11 | 3 | +0.2761 +/-0.0029 | +0.2722 +/-0.0042 | -0.0040    |
| array11 | 6 | +0.1947 +/-0.0019 | +0.2104 +/-0.0055 | **+0.0157**|
| array12 | 1 | +0.1890 +/-0.0023 | +0.1903 +/-0.0015 | +0.0013    |
| array12 | 3 | +0.2750 +/-0.0007 | +0.2751 +/-0.0046 | +0.0002    |
| array12 | 6 | +0.2002 +/-0.0027 | +0.2150 +/-0.0037 | **+0.0149**|
| array17 | 1 | +0.1190 +/-0.0022 | +0.1190 +/-0.0022 | -0.0000    |
| array17 | 3 | +0.1828 +/-0.0012 | +0.1806 +/-0.0046 | -0.0022    |
| array17 | 6 | +0.1302 +/-0.0027 | +0.1542 +/-0.0061 | **+0.0239**|

READING: at h=1 and h=3 the two are indistinguishable (differences
0.000-0.004 against seed std 0.001-0.005). At h=6 the LSTM wins on ALL
THREE arrays by 0.015-0.024, consistent in sign across three module
technologies.

INTERPRETATION: the sequence model earns nothing where point lags
suffice, and earns something only when the issue time is far enough from
the target that the SHAPE of the recent trajectory carries information a
handful of aggregated lags cannot.

SECOND RESULT IN THE SAME TABLE: LSTM seed variance is consistently
2-3x XGBoost's (e.g. array17 h6: 0.0061 vs 0.0027). Any paper reporting a
single LSTM run reports a number whose unstated error bar is larger than
most claimed improvements in this literature.

RQ4 PREVIEW: LSTM fit ~16 s vs XGBoost sub-second, roughly 30:1 compute
for zero measurable gain at h=1 and h=3.

STATISTICAL CAVEAT, IMPORTANT: the aggregator's current test compares the
difference in means against 2x the std of INDIVIDUAL RUNS. That is not a
correct two-sample test — the standard error of the difference is
sqrt(s1^2/5 + s2^2/5), which for array11 h6 is ~0.0026 against a
difference of 0.0157 (t ~ 6). Under a proper test the h=6 results are
much stronger and h=1/h=3 remain non-significant. Same conclusion, better
justified.

MORE IMPORTANT: seed spread is the WRONG uncertainty for the paper's
claim. It measures training stochasticity, not sampling uncertainty in
the evaluation period. The significance test must be DIEBOLD-MARIANO on
paired forecast error series across the ~3760 daylight hours, with a HAC
variance estimator and the Harvey-Leybourne-Newbold small-sample
correction. Report seed spread as a reproducibility statistic (Table 3)
and DM as the significance test (Table 6). Do not let the 2x std
heuristic stand in for either.

Scripts: scripts/run_seed_sweep.py, scripts/aggregate_seed_sweep.py
Data: results/seed_sweep_summary.csv, 90 run JSONs

---

## 6. WHAT IS BUILT vs WHAT REMAINS

BUILT AND VALIDATED:
- data layer (load, clean, resample, clear-sky irradiance, clear-sky
  power, splits, exclusions)
- feature layer (tabular both regimes, sequences, scaler) with real
  leakage assertions that are proven to catch injected leaks
- models: SmartPersistence, Climatology, ConvexCombination, XGBoost, LSTM
- metrics, run-record writer with environment capture
- 90 committed runs

REMAINING:
- CNN-LSTM (prompt already drafted; next step)
- CNN-LSTM + XGBoost residual stage (rule 6: fit on VALIDATION residuals)
- sky-condition classification for RQ3 — MUST use k_ghi and its
  variability, NOT k_p. Two reasons: sky condition is a property of the
  atmosphere and is shared across arrays, whereas k_p would give three
  different classifications of the same sky; and k_p is computed from
  measured power, i.e. the target, so stratifying errors by it means
  conditioning on the thing being predicted.
- Diebold-Mariano tests (HAC + HLN correction)
- oracle-regime runs across the grid
- full grid runner + config system
- aggregation into LaTeX tables, figures F1-F7
- ONE final test-set (2015) run, at the very end
- the paper

OUTSTANDING FIXES (known, not yet done):
1. all_hours metrics block no longer spans 24 hours (see Finding 2
   caveat). Needs a separate all_hours_vs_persistence subset.
2. Aggregator significance test is not a proper two-sample test.
3. Pipeline setup costs ~10 s per run and is recomputed identically 90
   times. For a 450-run grid that is over an hour of pure waste. Cache
   the prepared dataframe per array before the full grid.
4. Several older scripts still hardcode their own TRAIN_YEARS and are
   retained for provenance only; each has a NOTE in its docstring.

---

## 7. PAPER MAPPING

| Finding | Section | Artifact |
|---------|---------|----------|
| 1 clear-sky sampling | 4.1 Methodology; RQ2 | diagnose_clearsky_bias.py |
| 2 night inclusion + closed form | Intro; Table 4 | run JSONs |
| 3 coverage collapse | 4.1; report n per cell | check_feature_coverage.py |
| 4 persistence degradation | 4.1; Limitations | diagnose_baseline_error.py |
| 5 reference choice (HEADLINE) | Table 4; figure | reference_comparison.csv |
| 6 dead array / audit blind spots | Data; Intro motivation | dead_period_audit.csv |
| 7 training length | ablation | seed sweep |
| 8 architecture x horizon | Table 3, Table 5 | seed_sweep_summary.csv |

Structure (~8 pages, IEEE two-column): Introduction, Related Work, Data
and Preprocessing, Methodology (4.1 protocol FIRST, then 4.2
architectures — the ordering is deliberate), Experimental Setup, Results
organised BY RESEARCH QUESTION not by model, Limitations, Conclusion.

Key literature (notes in paper/literature_notes.md):
- Yang et al. 2020, Solar Energy 210:20-37 — the cornerstone. Recommends
  the RMSE skill score against the optimal convex combination of
  climatology and persistence. Also documents the low-elevation
  clear-sky-index instability and the zenith<85 filter convention.
- Kapoor & Narayanan 2023, Patterns 4:100804 — leakage taxonomy L1-L3
  (8 types), 294 papers across 17 fields, and the civil-war case where
  ML superiority vanished once leakage was fixed. PRESENT SECTION 4.1 AS
  A FILLED-IN MODEL INFO SHEET USING THEIR LABELS.
- Nguyen & Musgens, arXiv 2208.10536 — meta-analysis, 1447 screened /
  320 read. Horizon dominates all other factors, so report per horizon.
- Hewamalage, Ackermann & Bergmeir, arXiv 2203.10716 — forecast
  evaluation pitfalls from the general ML side.
- Three coded hybrid papers in results/literature_survey.csv; TEN MORE
  NEEDED. Code the six columns that matter: night hours excluded,
  baseline used, skill score reported, weather source, split type, seed
  variance reported.

REVIEWER RISK TO PRE-EMPT: Yang et al. recommend the convex combination
as reference. We now report both, which is stronger than either alone —
but say so explicitly rather than leaving it implicit.

---

## 8. RECORD OF WRONG PREDICTIONS (kept deliberately)

These are logged because the pattern matters: in every case the raw data
corrected the prediction, and the habit of checking output rather than
trusting a plausible explanation is what caught the real bugs.

1. Predicted the gain would rise for all three arrays when the training
   window was extended backwards. Only CdTe moved (0.835 -> 0.861);
   silicon barely changed (0.916 -> 0.917). Cause: CdTe degrades faster.
2. Predicted "95%+ daylight retention" after the ffill fix. Actual 37.5%
   at h=6. A second, different mechanism (rolling stats on raw NaN
   series) was responsible.
3. Predicted 61,368 processed rows for 2009-2015. Actual 61,344 — added a
   leap day that does not exist.
4. Predicted the persistence midday error was low-elevation clear-sky
   bias and that stricter daylight thresholds would shrink it. Thresholds
   made h=3 and h=6 WORSE. Actual cause: issue time near dawn.
5. Identified array17's download as "not array 17" because the file
   started in 2008. Wrong: every DKASC export starts at site inception
   regardless of array install date.
6. Predicted `git check-ignore -v` would print nothing for negated
   patterns. It prints the matching negation line.
7. Let the 2x-std significance heuristic stand initially instead of
   specifying a proper two-sample test and DM from the start.

---

## 9. HOW TO RESUME

```
conda activate pvfc
cd $env:USERPROFILE\projects\pv-forecast-bench
```

Read first: CLAUDE.md, then this file, then FUTURE_WORK.md.
Working style: Claude Code for doing, PowerShell for looking. Run
diagnostics in the terminal and read the raw numbers directly.

Next step: CNN-LSTM (src/models/cnn_lstm.py), then the residual stage.
