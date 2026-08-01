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
6. **[REVISED 2026-07-29]** The XGBoost residual stage is fit on
   OUT-OF-FOLD residuals generated within the TRAINING period, using an
   expanding window over TRAIN_YEARS (fit on 2011 -> predict 2012; fit on
   2011-2012 -> predict 2013). The validation split is used ONLY for
   early stopping of the base model, never for fitting the residual
   stage.

    CHANGE LOG. The rule previously read: "XGBoost residual stage is fit
    on validation-split residuals, not training-split residuals." That was
    WRONG and produced the largest leak in the project. Fitting on
    validation residuals and then evaluating on validation is in-sample
    performance for the correction stage ([L1.1], Kapoor & Narayanan).
    Evidence: array11 h6 seed0 gave skill_vs_convex +0.5447 under the old
    rule, +0.1768 under the corrected scheme, against a plain LSTM
    baseline of +0.2110. The old behaviour is retained as an explicit,
    warned opt-in (residual_fit_split='val', leaked_by_design=True) so it
    can be quantified in the protocol-inflation table.
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

### Finding 8: a suggestive, not established, architecture edge at long horizons (RQ1)
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
0.000-0.004 against seed std 0.001-0.005; DM confirms this, all six
cells p_holm=1.0). At h=6 the sign favours the LSTM on all three arrays
by 0.015-0.024, but under Diebold-Mariano (HAC variance, HLN
correction, Holm-Bonferroni within cell) only array17 is significant
(hln_stat 3.26, p_holm 0.0114); array11 (hln_stat 2.56, p_holm 0.073)
and array12 (hln_stat 2.64, p_holm 0.059) are not. Three co-located
arrays sharing one weather station are not three independent
confirmations, so "wins on all three" overstates what one significant
result and two non-significant ones support.

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
correct two-sample test. This section originally replaced it with a
seed-based two-sample calculation - standard error of the difference
sqrt(s1^2/5 + s2^2/5), giving ~0.0026 for array11 h6 against a
difference of 0.0157, "t ~ 6" - and called the h=6 result "much
stronger" under it. That calculation is deleted: it is wrong for the
same reason the 2x-std heuristic is wrong (seed spread is training
stochasticity, not sampling uncertainty over the evaluation period), and
it overstated the effect by roughly 2x. The real test is
Diebold-Mariano on paired forecast error series across the ~3760
daylight hours (HAC variance, HLN small-sample correction,
Holm-Bonferroni within cell): the actual array11 h6 statistic is
hln_stat 2.56, p_holm 0.073, not t~6, and it does not clear
significance. See Finding 12 Part A and results/table6_dm.csv. This
finding committed, within the same section that goes on to warn against
it, the exact error of letting seed spread substitute for DM - see
Record of Wrong Predictions #12.

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

### Finding 9: convolution shows a consistent directional cost, not a significant one

TRIGGER: the planned hybrid was CNN-LSTM + XGBoost residual. Before
building the residual stage on top of it, the CNN-LSTM was run through
the same 5-seed sweep as XGBoost and LSTM (135 runs total, 15.7 min,
0 failures).

FIRST EXPECTATION: convolution would either add a small amount at h=6
(where the LSTM had shown a real advantage) or be neutral. The hybrid
literature treats a Conv1d front-end as a straightforward improvement.

EVIDENCE — skill_vs_convex, mean +/- std over 5 seeds, validation 2014,
daylight:

| array   | h | XGBoost           | LSTM              | CNN-LSTM          | CNN-LSTM - LSTM |
|---------|---|-------------------|-------------------|-------------------|-----------------|
| array11 | 1 | +0.1967 +/-0.0023 | +0.1938 +/-0.0021 | +0.1945 +/-0.0058 | +0.0007         |
| array11 | 3 | +0.2761 +/-0.0029 | +0.2722 +/-0.0042 | +0.2646 +/-0.0032 | -0.0075         |
| array11 | 6 | +0.1947 +/-0.0019 | +0.2104 +/-0.0055 | +0.2017 +/-0.0047 | -0.0087         |
| array12 | 1 | +0.1890 +/-0.0023 | +0.1903 +/-0.0015 | +0.1893 +/-0.0035 | -0.0010         |
| array12 | 3 | +0.2750 +/-0.0007 | +0.2751 +/-0.0046 | +0.2671 +/-0.0061 | -0.0080         |
| array12 | 6 | +0.2002 +/-0.0027 | +0.2150 +/-0.0037 | +0.2014 +/-0.0063 | -0.0137         |
| array17 | 1 | +0.1190 +/-0.0022 | +0.1190 +/-0.0022 | +0.1168 +/-0.0049 | -0.0022         |
| array17 | 3 | +0.1828 +/-0.0012 | +0.1806 +/-0.0046 | +0.1763 +/-0.0052 | -0.0043         |
| array17 | 6 | +0.1302 +/-0.0027 | +0.1542 +/-0.0061 | +0.1491 +/-0.0087 | -0.0050         |

READING: CNN-LSTM is worse than or equal to the plain LSTM in 8 of 9
cells. The single positive cell (+0.0007) is trivially small. At h=6,
where the LSTM has a genuine advantage over XGBoost, the convolution
gives back roughly half of it on all three arrays.

STATISTICS, STATED CAREFULLY: most individual cells do not clear the
aggregator's (over-conservative) 2x-std threshold. This was originally
reported here via a two-sided sign test over the 9 array x horizon
cells (8 of 9 negative, p ~ 0.039) - that test is WRONG as stated,
because it treats the 9 cells as independent observations. They are
not: array11, array12 and array17 are co-located and share one weather
station, so their errors are correlated, and 3 of the 9 "observations"
are really 3 correlated views of the same 3 horizons. The sign test is
retracted; see Finding 12 Part A for the test that replaces it. Under
Diebold-Mariano (HAC variance, HLN correction, Holm-Bonferroni within
cell), lstm vs cnn_lstm is NOT significant in ANY of the 9 cells
(largest |hln_stat| 2.30, array12 h=6, p_holm=0.13). The evidence for
Finding 9 is therefore the CONSISTENCY OF SIGN and the two auxiliary
axes below, not a demonstrated accuracy loss.

THREE AXES:
1. Accuracy: worse in 8 of 9 cells, direction consistent (worse at h=3
   and h=6 in all three arrays, better at h=1 in all three), but not
   Holm-significant in any cell under DM (Finding 12 Part A).
2. Stability: CNN-LSTM has higher seed variance than XGBoost in all 9
   cells, and higher than the LSTM in 7 of 9 (the exceptions are
   array11 h3 and array11 h6). Largest gap: array17 h6, +/-0.0087 vs
   +/-0.0061 LSTM vs +/-0.0027 XGBoost.
3. Compute: fit times ~20-31 s vs ~14-27 s for the LSTM, roughly 20-30
   percent more.

INTERPRETATION: at h=6, where recurrent models hold a genuine advantage
over XGBoost, convolution gives back roughly half of it on all three
arrays - but this directional effect does not reach significance under
DM, so it should be reported as a consistent, non-significant cost, not
a demonstrated one. Combined with higher seed variance in most cells and
20-30 percent higher compute, a component that is standard in this
literature fails to earn its place on any of the three axes, without
the accuracy claim itself being statistically established.

CONSEQUENCE FOR THE STUDY DESIGN: this result forced the residual stage
to be built as a 2x2 - {LSTM, CNN-LSTM} x {no residual, residual} -
rather than only on CNN-LSTM as originally planned. Testing the residual
stage only on the weaker base would have confounded two questions.

PAPER: RQ1 component attribution, Table 5. Also RQ4, since the compute
cost is measured. Note explicitly that this is an ABLATION, which most of
the surveyed hybrid papers do not run - they report the full architecture
against its own components or against nothing.

Data: results/seed_sweep_summary.csv, 135 run JSONs,
results/_sweep_log_cnn.txt

### Finding 10: residual-stage leakage (the largest effect measured)

TRIGGER: the first residual-corrected run (array11, h=6, lagged, seed 0,
base LSTM) returned skill_vs_convex = +0.5447 against the plain LSTM's
+0.2110. RMSE halved, 0.516 -> 0.298.

WHAT MADE IT OBVIOUS: a jump of +0.334. Every genuine architectural
effect measured across 135 runs in this project is 0.01-0.02. This was
roughly TWENTY TIMES larger than anything real. The magnitude itself was
the diagnostic - the number was impossible before it was explained.

FIRST DESCRIPTION, AND WHY IT WAS INADEQUATE: the run was initially
flagged as "optimistic" because both stages had partly seen the
validation split. That is too weak. The residual XGBoost was FIT on
validation residuals and then EVALUATED on validation - the same rows.
That is in-sample performance for the correction stage: [L1.1] in Kapoor
& Narayanan's taxonomy, "no test set - training and testing on the same
data". Not optimism. Leakage.

ROOT CAUSE — A PROTOCOL RULE THAT WAS ITSELF WRONG: CLAUDE.md rule 6
originally said the residual stage fits on VALIDATION-split residuals.
The reasoning behind it was sound as far as it went: training-split
residuals are artificially small because the base model has already
fitted them. But the rule avoided that contamination by creating a worse
one. THIS IS THE MOST IMPORTANT METHODOLOGICAL LESSON OF THE PROJECT:
a written, deliberate, well-intentioned protocol rule was the source of
the largest leak found.

FIX — out-of-fold residuals within the TRAINING period only, the standard
stacking construction adapted to time series:
- expanding window over TRAIN_YEARS: fit base on 2011 -> predict 2012;
  fit on 2011-2012 -> predict 2013
- pool those out-of-sample predictions and their residuals
- fit the residual XGBoost on the pooled out-of-fold set
- then fit the final base model on all training years, using validation
  ONLY for early stopping
n_oof_residuals = 16551, contributed by years 2012 and 2013 (2011
contributes no fold, by construction).
Each fold's base model sees less training data than the final model, so
out-of-fold residuals are slightly PESSIMISTIC. That is the correct
direction to err.

RESULT AFTER THE FIX (array11, h=6, lagged, seed 0):

| variant                          | daylight skill_vs_convex |
|----------------------------------|--------------------------|
| plain LSTM                       | +0.2110                  |
| LSTM + residual, corrected (oof) | +0.1768                  |
| LSTM + residual, leaked (val)    | +0.5447                  |

TWO SEPARATE RESULTS:

(a) The residual stage HURTS by about -0.034 once leakage is removed -
    roughly six LSTM seed standard deviations. This is no longer awaiting
    confirmation: Diebold-Mariano (Finding 12 Part A,
    results/table6_dm.csv) independently confirms lstm vs lstm_residual
    at array11 h=6 is significant in the direction of plain LSTM
    (hln_stat -3.45, p_holm 0.0056).

(b) A single plausible-looking protocol choice turns a component that
    hurts by 0.034 into one that appears to help by 0.334. That is not
    just inflation - it is a SIGN FLIP plus an order of magnitude.
    "Fit the second stage on the validation split" reads as reasonable
    in a methods section. CAVEAT ON THIS COMPARISON: it rests on ONE
    seed, ONE array (array11), ONE horizon (h=6). The leaked variant was
    never run through the DM pipeline - table6_dm.csv has no leaked-model
    column - so the sign-flip/magnitude claim itself has no significance
    test behind it, only the single-run numbers above. That is adequate
    for demonstrating that the leak is possible and large, but it is not
    a general effect-size claim.

FEATURE IMPORTANCES ALSO REORDERED, which matters as much as the metric.
Leaked run's top features: k_p_hours_stale (2.461),
Active_Power_roll24_std (2.388), k_p_issue_m1 (2.267), k_p_issue_m2
(2.192), k_ghi_issue_m2 (1.883). Corrected run's top features: hour_cos,
k_ghi_issue_m2, Active_Power_roll24_std. Any RQ1 component-attribution
claim drawn from the leaked run would have been wrong about WHICH
features carried residual signal, not merely by how much.

EVIDENCE PRESERVED, NOT DELETED: the leaked run is kept as
results/leaked_lstm_residual_array11_h6_lagged_seed0.json with a
top-level "INVALID_LEAKED": true key and an explanatory note. The leaked
behaviour is also available as an EXPLICIT opt-in flag
residual_fit_split='val' on ResidualCorrected, which warns on
construction and sets leaked_by_design=True in the config. This makes it
a reproducible protocol configuration for Table 4 rather than an
accident.

PAPER: Table 4, alongside night inclusion and reference choice. This is
the single largest protocol effect in the paper, and arguably the most
persuasive, because the mistake is so easy to make and the consequence is
a sign flip rather than a magnitude change. It is also direct evidence
for the Introduction's claim.

Scripts: src/models/residual.py, scripts/run_residual_dev.py

---

### Finding 11: the residual penalty shrinks with more folds, but the short-recovers/long-persists split is not confirmed by DM

TRIGGER: the 225-run seed sweep (Finding 10 addendum) showed residual
correction negative in all 18 array x horizon cells (-0.024 to -0.046
skill_vs_convex).

FIRST READING, TOO STRONG: "residual correction does not help on this
problem". The 18-of-18 consistency was itself suspicious - real effects
in this project have been noisier.

CONFOUND IDENTIFIED: with TRAIN_YEARS = 2011-2013 the expanding window
gives only TWO folds (base on 2011 -> predict 2012; base on 2011-2012 ->
predict 2013), seeing 1/3 and 2/3 of the training data against the
deployed model's 3/3. The corrector may learn to fix errors a stronger
model does not make.

CONTROL: rerun with TRAIN_YEARS = 2009-2013 (five years, four folds),
arrays 11 and 12, 3 seeds, 36 runs. n_oof_residuals 16.5k -> 34k.

RESULT - the penalty shrinks but does not vanish, and the pattern is
horizon-structured:

| array   | h | 3yr/2fold | 5yr/4fold | recovery |
|---------|---|-----------|-----------|----------|
| array11 | 1 | -0.0271   | -0.0069   | 75%      |
| array11 | 3 | -0.0260   | -0.0105   | 60%      |
| array11 | 6 | -0.0309   | -0.0175   | 43%      |
| array12 | 1 | -0.0310   | -0.0013   | 96%      |
| array12 | 3 | -0.0400   | -0.0071   | 82%      |
| array12 | 6 | -0.0248   | -0.0200   | 19%      |

INTERPRETATION, AS ORIGINALLY WRITTEN: claim B (fold starvation) at
short horizons, claim A (genuine penalty) at long ones. At h=1 with four
folds the penalty is inside seed noise - with 3 seeds the standard error
on these differences is roughly 0.003-0.005, so -0.0013 and -0.0069 are
not distinguishable from zero. At h=6 a penalty of -0.0175 to -0.0200
survives at 3-4 standard errors.

CORRECTION AFTER DM: the clean short-recovers/long-persists split above
was never tested with Diebold-Mariano - the "3-4 standard errors" figure
is a 3-seed two-sample SE (the same category of statistic Finding 8 had
to retract, not an autocorrelation-aware test on the daylight-hour error
series), and it was computed only for the 5-year/4-fold sensitivity run,
which was never run through the DM pipeline at all: table6_dm.csv has no
5-year-window entries, only the standard 3-year/2-fold config. The
numbers in both tables above stand; the significance status does not.
On the 3-year/2-fold config that IS in table6_dm.csv, DM on lstm vs
lstm_residual gives: array11 h1 not significant (hln_stat -2.46, p_holm
0.069); array11 h3 significant (-3.45, p_holm 0.0040); array11 h6
significant (-3.45, p_holm 0.0056); array12 h1 significant (-2.89, p_holm
0.027); array12 h3 significant (-6.24, p_holm 4.9e-9); array12 h6 NOT
significant (-2.09, p_holm 0.185). This does not reproduce a clean h=1
recovers / h=6 persists split: array12 h6 is a long horizon with a
penalty similar in size to array11 h6's significant one (-0.0248 vs
-0.0309) yet is not significant, while array12 h1 is short and IS
significant. Report the fold-count mechanism as a real, quantified effect
on the OOF/validation correlation gap (that part does not depend on
seed-based significance), but do not claim DM confirms a clean
horizon-based split - it does not.

CNN-LSTM DOES NOT FIT THE MECHANISM EITHER: cnn_lstm vs
cnn_lstm_residual is Holm-significant in all 9 array x horizon cells,
including h=1 (p_holm 0.0002 to 0.035; see Finding 12 Part A). If fold
starvation explained the short-horizon recovery seen in the LSTM base,
CNN-LSTM's residual penalty should show the same pattern at h=1 and it
does not - it is significant everywhere. The fold-count control
(CONTROL, above) was only run for the LSTM base on arrays 11 and 12; the
mechanism has not been shown to generalise to CNN-LSTM or to array17.

MECHANISM (scripts/diagnose_residual_signal.py, array11, seed 0):
correlation between predicted and actual residual is +0.76 to +0.79
out-of-fold but only +0.04-0.13 on validation, under both the 3-year and
5-year windows. The corrector is not learning nothing - it is
OVERCONFIDENT: it sizes its correction to the out-of-fold relationship,
not the weaker one that transfers to validation. Sizing this precisely
with the closed form for how a correction p changes a base model's MSE
(-2*rho*sigma_r*sigma_p + sigma_p^2, so p helps only when
sigma_p < 2*rho_val*sigma_r) confirms it, and confirms the fold-count
story quantitatively rather than just directionally:

| config        | rho_oof | rho_val | sigma_p/sigma_r | break-even (2*rho_val) | overconfidence |
|---------------|---------|---------|-----------------|-------------------------|----------------|
| h3  3yr/2fold | +0.765  | +0.096  | 0.380           | 0.192                   | 2.0x           |
| h3  5yr/4fold | +0.651  | +0.131  | 0.325           | 0.261                   | 1.2x           |
| h6  3yr/2fold | +0.790  | +0.036  | 0.340           | 0.072                   | 4.7x           |
| h6  5yr/4fold | +0.666  | +0.059  | 0.335           | 0.117                   | 2.9x           |

The 5-year window moves h3 to within 1.2x of break-even (nearly
calibrated) but leaves h6 at 2.9x - the same short-horizon-recovers,
long-horizon-persists split as the skill-score table above, now visible
in the mechanism that produces it: more folds raise rho_val and lower
sigma_p together, and h3's validation correlation responds much more to
added folds (+0.10 -> +0.13) than h6's does (+0.036 -> +0.059).

WHY THIS IS A BETTER RESULT THAN THE FIRST READING: "component X hurts"
is an architecture verdict on one dataset. "Component X's measured value
changes by 75-96 percent depending on how the training residuals are
constructed, with no leakage in either version" is a protocol-sensitivity
finding, which is this paper's thesis. It belongs alongside the
reference-choice result, not in competition with it.

NEAR MISS WORTH RECORDING: two hardcoded 3-year assumptions
(clearsky_power.py's train-year guard, residual.py's _oof_residuals fold
derivation) would have silently kept using 2 folds even under the 5-year
window. That would have produced a "no change" result reading as
confirmation of claim A - a stronger claim than the data supports. A bug
that produces a plausible NULL is more dangerous than one that crashes.

PAPER: RQ1, and a row in Table 4. The 3-year sweep is the main table
(2011-2013 is the window all three arrays share); the 5-year run is the
sensitivity analysis that makes it credible. Report both.

Scripts: scripts/rerun_residual_5yr.py, scripts/diagnose_residual_signal.py

---

### Finding 12: DM retires the 2x-std heuristic; sky stratification finds partly-cloudy, not overcast, is the hard case

TRIGGER: two REQUIRED items from Section 6 closed two commits apart: (1)
replace the 2x-seed-std significance heuristic (Record of Wrong
Predictions #7) with a real hypothesis test before publication, (2)
build the sky-condition classification for RQ3 from k_ghi alone, never
k_p (src/eval/sky.py's module docstring works through why k_p would
leak array-specific, target-derived information into what is supposed
to be an atmosphere-only stratification).

PART A - DIEBOLD-MARIANO: src/eval/dm.py implements dm_test (HAC
long-run variance, Bartlett kernel, truncated at lag h-1, since h-step
forecast errors overlap and are correlated by construction even when
the underlying series is not; HLN small-sample correction against
t(n-1) rather than the normal DM originally used) and dm_matrix (all
C(7,2)=21 pairwise comparisons per array x horizon cell across the 5
forecasters plus smart_persistence and convex_reference, Holm-Bonferroni
corrected within each cell). scripts/build_table6_dm.py writes
results/table6_dm.csv: 189 pairs (21 x 3 arrays x 3 horizons), 138/189
significant at Holm-corrected p<0.05 overall - but only 39/90 once the
two baseline comparators are excluded and only the 5 core forecasters
are compared against each other. Most of the raw significance count
was never in question (every model beats smart_persistence and
convex_reference at high significance); the informative number is the
39/90 among architectures.

Two examples of what DM changes about existing findings:

- Finding 9 (convolution): lstm vs cnn_lstm is NOT Holm-significant in
  ANY of the 9 array x horizon cells (largest |hln_stat| 2.30, array12
  h=6, p_holm=0.13). Direction is consistent with the skill-score sweep
  - cnn_lstm is worse at h=3 and h=6 in all three arrays, better at h=1
  in all three - but "convolution costs accuracy" should be reported as
  a consistent, non-significant directional effect, not as 9
  independent losses.
- Finding 11 (residual penalty): cnn_lstm vs cnn_lstm_residual is
  Holm-significant in all 9 cells (p_holm 0.035 to 0.00007) - the
  convolutional base model's residual penalty is the more robust of the
  two under DM. lstm vs lstm_residual is significant in only 5 of 9
  (array11 h3/h6, array12 h1/h3, array17 h1) and NOT significant at
  array11 h1, array12 h6, array17 h3/h6. This is roughly consistent with
  Finding 11's short-horizon-recovers story but not a clean h<=1-vs-h=6
  split - array12 h6 fails to reach significance despite being a long
  horizon, array17 h1 reaches it despite being short. Report the DM
  result alongside Finding 11's mechanism rather than as a cleaner
  restatement of it.

PART B - SKY-CONDITION STRATIFICATION (RQ3): src/eval/sky.py's
classify_sky bins each daylight row into clear / partly_cloudy /
overcast from k_ghi_mean and k_ghi_std over the last 3 valid
observations (thresholds 0.75 / 0.10 / 0.40, conventional literature
values, NOT fitted on this site or any split). scripts/build_table_sky.py
writes results/table_sky.csv (xgboost, lstm, lstm_residual x 3 arrays x
3 horizons x 3 classes). Pooled daylight-hour counts: clear 60,048,
partly_cloudy 33,021, overcast 6,426 - overcast is the rarest condition
at this site, ~6.5% of daylight hours.

RESULT - mean skill_vs_convex by class: clear 0.332, overcast 0.235,
partly_cloudy 0.066. partly_cloudy is the WORST class, below overcast,
in every one of the 9 array x horizon cells with zero exceptions (e.g.
array11 h=3: clear 0.412, overcast 0.401, partly_cloudy 0.051; array17
h=6: clear 0.249, overcast 0.184, partly_cloudy 0.025).

INTERPRETATION: counterintuitive on first reading - overcast is dim and
low-output, so it looks like the harder condition. It is not: overcast
skies are temporally stable hour to hour, so the convex reference
(climatology + persistence) already tracks them well and the models add
little beyond that. Partly-cloudy is where fast-moving cloud edges
produce irradiance ramps that lagged features cannot anticipate - the
k_ghi_std component of the classifier is, by construction, selecting for
exactly the regime where short-horizon forecasting is hardest. This
matches the general solar-forecasting literature (ramp events under
broken cloud are the standard hard case) and is a genuine empirical
result, not another protocol-sensitivity finding like 1-11 - flag that
distinction when writing it up.

PAPER: RQ3 (sky stratification, new empirical result) and Table 6 (DM
replaces the 2x-std heuristic as the paper's significance test; seed
spread stays in Table 3 as a reproducibility statistic only, per
src/eval/dm.py's module docstring).

Scripts: src/eval/dm.py, scripts/build_table6_dm.py, src/eval/sky.py,
scripts/build_table_sky.py

---

## 6. WHAT IS BUILT vs WHAT REMAINS

BUILT AND VALIDATED:
- data layer (load, clean, resample, clear-sky irradiance, clear-sky
  power, splits, exclusions)
- feature layer (tabular both regimes, sequences, scaler) with leakage
  assertions proven to catch injected leaks
- models: SmartPersistence, Climatology, ConvexCombination, XGBoost,
  LSTM, CNN-LSTM, ResidualCorrected (wraps either recurrent base)
- src/models/recurrent_base.py: shared training loop, scalers, early
  stopping, prediction. Refactor verified behaviour-preserving by
  re-running LSTM array11/h6/seed0 and diffing bit-for-bit against the
  previously committed result.
- metrics, run-record writer with environment capture
- 225-run 5-seed sweep for lstm_residual and cnn_lstm_residual, complete,
  0 failures: residual correction is net negative in all 18 array x
  horizon cells under the default 3-year TRAIN_YEARS (Finding 10 addendum
  / results/seed_sweep_summary.csv)
- 5-year/4-fold training-length ablation for the residual stage
  (scripts/rerun_residual_5yr.py, 36 runs, results/train5yr/): the
  penalty shrinks (43-96% recovery depending on horizon) but does not
  vanish at h=6 - see Finding 11
- scripts/diagnose_residual_signal.py: out-of-fold vs validation
  correlation and predicted/actual residual-std diagnostic behind
  Finding 11's OVERCONFIDENCE mechanism, parameterised over horizon and
  train_years
- 262 committed runs (135-run architecture sweep + 225-run residual sweep
  + 36-run 5yr ablation + residual dev runs, some overlapping the sweeps)

REMAINING — REQUIRED:
- Diebold-Mariano tests with HAC variance and the Harvey-Leybourne-Newbold
  small-sample correction. MUST replace the 2x-std heuristic before
  publication.
- fix the all_hours metrics block (see Finding 2 caveat) - needs a
  separate all_hours_vs_persistence subset intersecting only the model
  and persistence, so it spans a true 24-hour cycle again
- sky-condition classification for RQ3, from k_ghi and its variability,
  NOT k_p (k_p is derived from the target; also sky condition is shared
  across arrays while k_p is not)
- oracle-regime runs
- aggregation into LaTeX tables; figures F1-F7
- ONE final test-set (2015) run, at the very end

REMAINING — OPTIONAL, and probably worth SKIPPING to protect writing time:
- extending the 5-year/4-fold ablation to array17 and to the 6-seed grid
  the main sweep uses (currently 3 seeds, arrays 11/12 only, per Finding
  11's CONTROL) - the horizon-structured pattern is already clear enough
  to report as a sensitivity analysis, not a second main result
- caching the prepared dataframe per array (only matters if a wider grid
  is run; ~10 s of redundant setup per run)

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
| 9 convolution ablation | RQ1, Table 5; RQ4 compute | seed_sweep_summary.csv |
| 10 residual leakage | Table 4 (largest effect); Intro | leaked_*.json + residual.py |
| 11 residual penalty fold-sensitivity | RQ1; Table 4 | rerun_residual_5yr.py, diagnose_residual_signal.py |

Table 4 (protocol inflation) now has at least five rows, in ascending
order of severity:
1. night-hour inclusion: nRMSE deflated ~34%, skill unchanged (+0.002)
2. reference choice: h=6 skill +0.65 -> +0.19, and the horizon trend
   changes from monotonic to non-monotonic
3. residual fit split: -0.034 -> +0.334, a SIGN FLIP
4. residual training-window length: penalty magnitude changes 19-96%
   depending on TRAIN_YEARS (2011-2013 vs 2009-2013), no sign flip but a
   large swing in a supposedly fixed architectural verdict
5. (planned) oracle vs lagged weather

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

8. Wrote protocol rule 6 (residual stage fits on validation residuals)
   into CLAUDE.md at the start of the project and left it unexamined
   through every subsequent review, until the +0.5447 result forced the
   issue. The rule was carried over from the original project spec
   without working through what it implied for evaluation. It then
   produced the largest leak in the project. The lesson generalises: a
   protocol rule being WRITTEN DOWN and DELIBERATE is not evidence that
   it is correct, and a documented rule is harder to question than an
   undocumented habit.

9. Described the leaked residual result as "optimistic" on first reading
   rather than as leakage. The correct classification ([L1.1], in-sample
   evaluation) only became clear from the magnitude.

10. Predicted CNN-LSTM would be neutral-to-slightly-positive at h=6.
    It was negative in 8 of 9 cells.

11. Read the 225-run sweep's 18-of-18 negative cells for residual
    correction as "the component does not help" (Finding 11). The
    consistency itself should have been the tell - no other effect in
    this project has been that uniform - before the 5-year/4-fold control
    showed most of the penalty at short horizons was fold starvation, not
    a genuine architectural verdict.

12. Wrote Finding 8's claim that the LSTM h=6 advantage was established
    across all three arrays, and supported it with a seed-based
    two-sample t of ~6, within the same finding that warns seed spread
    must not substitute for DM. The actual DM statistic is 2.56 and only
    one of three arrays is Holm-significant.

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
