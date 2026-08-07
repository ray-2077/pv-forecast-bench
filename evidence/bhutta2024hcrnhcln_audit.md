# Audit: bhutta2024hcrnhcln

Source: `data/papers/s41598-024-68030-5.pdf` (25 pages, 113,554 chars)

Bhutta, M.S., Li, Y., Abubakar, M., Almasoudi, F.M., Alatawi, K.S.S.,
Altimania, M.R., Al-Barashi, M. (2024). "Optimizing solar power efficiency
in smart grids using hybrid machine learning models for accurate energy
generation prediction." *Scientific Reports* 14:17101.
https://doi.org/10.1038/s41598-024-68030-5

(Verified NOT a duplicate of the already-coded xu2025lstmxgboosteemdso -
different authors, different journal issue (14:17101 vs 15:30177),
different model architecture, despite both being Scientific Reports
papers using EEMD/hybrid-CNN naming conventions.)

DOI: https://doi.org/10.1038/s41598-024-68030-5

## Coded fields

**year**: 2024 | "Scientific Reports | (2024) 14:17101"

**venue**: Scientific Reports 14:17101

**dataset**: real-time solar plant data, three parameters only: power
production (MWh), plane of array (POA), and Performance Ratio (PR) | "A
real-time data collected from a solar plant, comprising three essential
parameters: power production (MWh), the plane of array (POA), and
Performance ratio (PR)." (p.4)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches
for night/daytime/daylight/sunrise/sunset/zenith/clear-sky/diurnal
anywhere in 113,554 characters.

**baseline_used = own_components** | "The study employs three hybrid
models: Hybrid Convolutional-Recurrence Net (HCRN), Hybrid Convolutional-
GRU Net (HCGRN), and Hybrid Convolutional-LSTM Net (HCLN). These models
are the modified improved versions of CNN-RNN, CNN-GRU, and CNN-LSTM
respectively" (p.4) - compared against "the basic state of the art
machine learning models" (p.3, section title: "A comparative analysis
between developed HCRN, HCGRN, HCLN models, and the basic state of the
art machine learning models") - i.e. their own hybrid versions vs. the
unmodified base architectures they started from. EXHAUSTIVE CHECK: zero
matches for persistence/naive/climatology/reference model/reference
forecast anywhere.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence".

**weather_source = measured** | POA (plane-of-array irradiance) is a
directly measured, on-site quantity; no NWP/forecast-weather language
anywhere in the text.

**split_type = not_stated** | "To develop the predictive models, 80% of
the collected data is utilized for training, while the remaining 20% is
employed for testing and validation process." (p.4) - ratio only, no
statement of chronological vs. random ordering (exhaustive check: zero
matches for chronological/random/shuffle/k-fold describing this split).

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: the only "standard
deviation" hit in the text (p.13) is a descriptive EDA statistic ("compute
statistical measures like mean, median, and standard deviation" of the raw
data), not run-to-run model variance. No seed/repeated-run/confidence-
interval language anywhere describing model accuracy.

**code_available = not_stated** | "Data availability: The datasets used
and/or analysed during the current study available from the corresponding
author on reasonable request." (p.24) - data only, not code, and gated
behind request per the coding rule's own worked example.

**key_claim**: "For solar plant energy generation prediction (three input
parameters only: power, POA, and Performance Ratio; 80/20 train/test
split, ordering not stated), the proposed hybrid models HCRN/HCGRN/HCLN
outperform their unmodified base architectures (CNN-RNN/CNN-GRU/CNN-LSTM)
and other 'basic state of the art' ML models; forecast horizon is never
stated anywhere in the extracted text." (Abstract, p.1, and Section
headings, p.3-4 - the abstract does not give a single consolidated
headline number the way most other papers in this batch do; results are
spread across multiple per-model tables in the body.)

## MAJOR FLAG: Performance Ratio used as a model INPUT feature

The model's feature set is deliberately minimal: power production, POA,
and Performance Ratio (PR) - three parameters, stated explicitly (p.4,
quoted above). Performance Ratio is conventionally defined as actual
energy yield divided by theoretical/expected yield under the measured
irradiance - i.e., PR is ALGEBRAICALLY DERIVED FROM the power output
being predicted (and from POA, the other input). This project's own
src/data/loader.py explicitly drops "Performance_Ratio: derived from
power and irradiance, encodes target" as one of exactly three named
leakage-trap columns (CLAUDE.md, "DROPPED COLUMNS"). This paper uses PR as
one of only three total input features to a model whose target is power
production - i.e., PR here is not a side feature but roughly a third of
the entire feature space, and it is constructed from the target variable
itself. The paper never discusses this relationship, never states whether
PR is lagged (computed only from past power/POA) or contemporaneous (using
the same-timestep power value the model is trying to predict), and never
flags it as a potential leakage source. This is coded as a FLAG rather
than folded into any closed-vocabulary field, because it is a feature-
legitimacy concern ([L2] in Kapoor & Narayanan's taxonomy - "model uses
features that are not legitimate"), not a night-hour/baseline/split/
variance/code issue.

## What this paper does WELL

- Reports a "computational cost of various models" table (Table 6, p.23) -
  an efficiency/compute axis alongside accuracy, relevant to this
  project's own RQ4.
- Structures its comparison explicitly as hybrid-vs-base-architecture
  ablation (HCRN vs CNN-RNN, etc.) rather than only reporting the full
  proposed stack - a real, if narrow, component-attribution exercise.

## Other observations

- No shared authors/institution with any other paper coded in this batch.
- Cannot check RMSE^2~mean^2+SD^2 or MAE<=RMSE from the extracted
  abstract/intro text alone - full per-model metrics are in body tables
  not fully captured in the sections read for this audit; recommend a
  closer pass on Tables 1-6 if this paper is used as primary survey
  evidence beyond the PR-leakage flag above.
