# Audit: li2022eemdssalstm

Source: `data/papers/1-s2.0-S235248472201441X-main.pdf` (14 pages, 53,209 chars)

Li, Z., Xu, R., Luo, X., Cao, X., Du, S., Sun, H. (2022). "Short-term
photovoltaic power prediction based on modal reconstruction and hybrid deep
learning model." *Energy Reports* 8:9919-9932.

DOI: https://doi.org/10.1016/j.egyr.2022.07.176

## Coded fields

**year**: 2022 | "Energy Reports 8 (2022) 9919-9932"

**venue**: Energy Reports 8:9919-9932

**dataset**: 100 MW PV plant, Zhejiang Province, China, **April 2017 only**
(one calendar month), 15-min sampling, daytime 7:00-18:00 only | "This
paper selects the PV power and other meteorological factors data collected
in April 2017 from a PV power plant in Zhejiang Province, China, with a
total capacity of 100 MW for simulation analysis and research." (p.2); "PV
power generation data from 7:00 to 18:00 is used as the prediction object
in this paper, with a sampling period of 15 min." (p.6)

**night_hours_excluded = yes** | "PV power generation data from 7:00 to
18:00 is used as the prediction object in this paper" (p.6)

**baseline_used = own_components** | comparison table (p.10): LSTM,
PCC-LSTM, PCC-EEMD-LSTM, SSA-LSTM, PCC-SSA-LSTM, PCC-EEMD-SSA-LSTM - each
row adds one more of the paper's own preprocessing/optimization modules
(PCC feature selection, EEMD decomposition, SSA hyperparameter search) to
plain LSTM. No persistence, climatology, or convex reference anywhere.
EXHAUSTIVE CHECK: searched persistence/naive/benchmark/reference forecast/
climatology/reference model - zero matches in 53,209 characters.

**skill_score_reported = no** | EXHAUSTIVE CHECK: searched "skill score"/
"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence" - zero matches. Reported metrics are R2, MAE, RMSE only (Eqs.
26-28), each defined against ground truth, never against a reference
forecast.

**weather_source = measured** | inputs are GHI and temperature "collected
... from a PV power plant" (p.2) alongside PV power itself; no NWP/
forecast-weather language anywhere (exhaustive check: measured/NWP/
numerical weather/exogenous/meteorological/ground station/pyranometer/
satellite/reanalysis/ERA5/MERRA - only generic "meteorological factors
data" describing the plant's own logged data).

**split_type = not_stated** | "The training and test sets are split into
9:1 ratios." (p.6) - a ratio only, with no statement of whether the split
preserves chronological order or how the 9:1 partition is drawn from the
one-month series. Per the coding rule, a bare ratio without a stated
ordering method is not_stated, even though the underlying data is a single
contiguous month.

**n_seeds = not_stated**

**variance_reported = no** | Table 1 (p.9) reports MAE/RMSE/R2 for 5
different RANDOMLY CHOSEN LSTM hyperparameter combinations ("LSTM
parameters are randomly set", column header) as a hyperparameter-
sensitivity check, not repeated runs of the same final model/seed - this
is not run-to-run variance of the reported result and does not satisfy the
rule's definition. EXHAUSTIVE CHECK: no other seed/repeat/standard-
deviation-of-accuracy/error-bar language found; the one other "standard
deviation" hit (p.10) is an EEMD noise-parameter setting ("The standard
deviation of the noise is set to 0.2"), unrelated to model variance.

**code_available = not_stated** | "Data availability: Data will be made
available on request." (p.11) - addresses data, not code; nothing else in
the text mentions a code repository. Per the coding rule, "available on
request" is not_stated, not "yes".

**key_claim**: "On one month (April 2017) of 15-min, daytime-only
(7:00-18:00) data from a 100MW PV plant in Zhejiang, China, the full
PCC-EEMD-SSA-LSTM pipeline reaches R2=99.71%, MAE=1.398, RMSE=1.760 versus
plain LSTM's R2=88.93%, MAE=9.046, RMSE=10.824, on a 9:1 train/test split
of unstated ordering; forecast horizon is never stated (the model predicts
'PV power' at 15-min resolution with no stated lead time given between
input and target)."

## What this paper does WELL

- Explicit daytime-only restriction with stated clock hours (7:00-18:00),
  not just a vague "daylight" mention.
- Ablates its own pipeline stage-by-stage (LSTM -> +PCC -> +EEMD -> +SSA)
  rather than reporting only the full stack - a real component-attribution
  table, closer to this project's own RQ1 approach than most of the
  literature surveyed so far.

## MAJOR FLAG: decomposition stated to occur BEFORE the train/test split

"Divide the reconstructed components of PV power, temperature, and GHI
into the training and test sets, train the model, and use LSTM to predict
PV power." (p.7, Section 4, describing the modelling procedure)

This sentence's own stated ORDER OF OPERATIONS is: (1) EEMD-decompose and
SE-reconstruct the full PV power / temperature / GHI series, THEN (2)
divide the reconstructed components into training and test sets. This is
stronger than the "not stated" caution already on file for xu2025lstm-
xgboosteemdso in this survey (results/literature_survey.csv) - there the
paper never says where the split occurs relative to decomposition; HERE
the paper explicitly describes decomposing first and splitting second.
EEMD is a full-series, non-causal decomposition (each IMF component is
computed using information from the whole input series, past and future
of any given point). Reconstructing components from the full month before
holding out a test slice means the test-set IMF values were computed
using information from the training-set portion of the series AND
vice versa - a textbook [L1.2] leakage pattern (Kapoor & Narayanan:
"pre-processing on training and test set"), stated as the paper's own
procedure rather than inferred from silence. This is coded as a FLAG, not
folded into any of the closed-vocabulary judgement fields, because none of
those fields (night_hours_excluded / baseline_used / skill_score_reported /
weather_source / split_type / variance_reported / code_available) has a
slot for "leakage in the preprocessing pipeline" - recommend the survey
write-up treat this as its own category, parallel to this project's own
Finding 10 (residual-stage leakage from a well-intentioned but wrong
protocol rule).

## Other observations

- Extremely short dataset: one calendar month, no cross-season validation
  possible, no stated year-over-year or seasonal generalization test.
  Combined with the split_type=not_stated ratio-only description, and the
  EEMD-before-split ordering above, this is one of the weakest-evaluated
  papers in the batch so far by this survey's standards, despite very
  strong reported numbers (R2=99.71%) - directly illustrating this
  project's central thesis that reported accuracy tracks evaluation rigor,
  not model capacity.
- Same "own_components-only, no naive baseline" pattern as
  xu2025lstmxgboosteemdso (already in the CSV) and shares the EEMD
  decomposition method with it - worth a cross-reference in the survey
  writeup as two independent instances of the same undisclosed-leakage-risk
  pattern in the EEMD-hybrid sub-literature, not a shared-authorship
  overlap (different author groups/institutions).
- No shared authors/institution with any paper coded earlier in this batch.
