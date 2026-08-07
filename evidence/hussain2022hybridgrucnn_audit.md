# Audit: hussain2022hybridgrucnn

Source: `data/papers/Complexity - 2022 - Hussain - A Hybrid Deep Learning-Based Network for Photovoltaic Power Forecasting.pdf` (12 pages, 98,742 chars)

Hussain, A., Khan, Z.A., Hussain, T., Ullah, F.U.M., Rho, S., Baik, S.W.
(2022). "A Hybrid Deep Learning-Based Network for Photovoltaic Power
Forecasting." *Complexity* (Hindawi) 2022:7040601.

**SAME DATASET FAMILY AS THIS PROJECT, MOST DIRECTLY OF ANY PAPER CODED SO
FAR**: uses FOUR DKASC arrays across two sites: "DKASC-AS-1A, DKASC-AS-1B,
DKASC-AS-2Eco, and DKASC-Yulara-SITE-3A gathered in DKASC, Alice Springs
(AS), Australia" (p.6). AS-1A and AS-1B were both "completed on Thursday,
January 8, 2009" (p.6) - i.e. installed before this project's own 2011
training-window start. Array IDs (1A/1B/2Eco/Yulara-SITE-3A) do not match
this project's own array11/array12/array17 naming, so they may be
different physical arrays within the DKASC network - not verified further
here - but this is the fourth paper in this survey batch drawing on the
DKASC/DKA Solar Centre family, and the first to use multiple DKASC arrays
simultaneously the way this project does.

DOI: https://doi.org/10.1155/2022/7040601

## Coded fields

**year**: 2022 | "Volume 2022, Article ID 7040601"

**venue**: Complexity (Hindawi) 2022:7040601

**dataset**: DKASC-AS-1A (10.5kW), DKASC-AS-1B (23.4kW), DKASC-AS-2Eco,
DKASC-Yulara-SITE-3A - 5-minute resolution, power + meteorological
(wind speed, temperature, etc.) | "we use four publicly available
real-world PV power datasets such as DKASC-AS-1A, DKASC-AS-1B,
DKASC-AS-2Eco, and DKASC-Yulara-SITE-3A gathered in DKASC, Alice Springs
(AS), Australia... All these datasets are recorded from active solar power
generation plants at five-minute resolution with different power
generation capabilities. It consists of different attributes, for example,
power generation and meteorological elements such as wind speed, weather
temperature, etc." (p.6)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches for
night/daytime/daylight/sunrise/sunset/zenith/clear-sky/diurnal anywhere in
98,742 characters.

**baseline_used = other_ML** [see flag re: "Naive"] | Figure 3 legend
(p.7): "Actual, LSTM-CNN, Proposed, Naive (SVR)" - comparators are
LSTM-CNN and an SVR-based model. EXHAUSTIVE CHECK: the only "naive" hit in
the whole text is this Figure 3 legend entry "Naive (SVR)" and its caption,
"Figure 3: Visual representation to check the predictability of the
proposed model with naive and state-of-the-art model on two days ahead
forecasting." (p.7)

FLAG (uncertain field): "Naive (SVR)" most plausibly reads as "a naive
[i.e. simple/baseline] SVR-based model" - a trained SVR regressor used as
a simple comparator - not literal PERSISTENCE (predict-the-last-value) in
this survey's technical sense. No formula or description of what "naive"
means operationally is given anywhere in the text; the word could also be
loosely gesturing at a persistence-like baseline without formally defining
it, which would push the coding toward "persistence" instead. Coded
other_ML because SVR is explicitly named as the model type, but flagged
as genuinely uncertain and listed again in the final uncertain-fields
report.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Reported metrics are RMSE, MSE, MAE, MBE (Tables 4-5,
pp.8-9).

**weather_source = measured** | on-site DKASC meteorological sensors
(wind speed, temperature - see dataset quote above); no NWP/forecast-
weather language anywhere.

**split_type = not_stated** | "For training purposes, these datasets are
divided into 70% for training, 20% for validation, and 10% for testing."
(p.6) - ratio only, no statement of chronological vs. random ordering
anywhere in the text.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches for seed/
repeated runs/standard-deviation-of-accuracy/"+/-"/confidence interval
anywhere in the text (the one "standard deviation" hit, p.5, describes an
outlier-removal preprocessing method - "outliers... removed via the
min-max and standard deviation methods" - not model-performance variance).

**code_available = yes** | "Data Availability: The codes and related
materials can be downloaded from https://github.com/Altaf-hucn/
Hybrid-Deep-Learning-Network-for-Photovoltaic-Power-Forecasting." (p.10)
This is the FIRST genuine "yes" code_available coding in this survey
batch - every other paper coded so far is either not_stated or an
"available on request" statement that the coding rules explicitly treat
as not_stated.

**key_claim**: "Evaluated on four DKASC PV datasets (Alice Springs x3,
Yulara x1) for hour-ahead forecasting (70/20/10 train/validation/test
split, ordering not stated), the proposed hybrid GRU-CNN model (temporal
features first via GRU, then spatial features via CNN) achieves the
lowest RMSE/MSE/MAE/MBE of all compared methods across all four datasets,
including on DKASC-AS-1B where it is reported achieving MAE/RMSE improving
to 0.1727/0.0298 versus 0.221/0.621 for an LSTM-then-spatial-features
ablation and 0.294/0.693 for a plain-SVR-style comparator (p.10)."

## FLAG: forecast horizon stated inconsistently (hour-ahead vs. two-days-ahead)

Abstract/Introduction (p.1): "Our proposed model was evaluated on four
publicly available PV power generation datasets for an hour-ahead
forecasting."

Figure 3 caption (p.7): "Visual representation to check the predictability
of the proposed model with naive and state-of-the-art model on TWO DAYS
AHEAD forecasting."

These two statements describe different horizons (1 hour vs. 2 days) for
what appears to be the same evaluation exercise (same model set: proposed,
LSTM-CNN, naive/SVR). The text never reconciles this - it is possible
Figure 3 is a specific example window spanning two days of hourly
predictions (i.e. "two days" describing the PLOT'S TIME AXIS SPAN, not the
forecast lead time), which would not actually be a contradiction, but the
paper's own wording ("on two days ahead forecasting") reads as a lead-time
description, not a plot-duration description. Not resolved here - both
readings are plausible from the text alone.

## What this paper does WELL

- Genuine public code release with a working-looking GitHub URL, not an
  "available on request" gate - rare in this survey.
- Evaluates on FOUR independent datasets (across 2 sites) rather than one,
  and reports per-dataset results (Table 4/5) rather than only a pooled
  number.
- Uses multiple regression metrics (RMSE, MSE, MAE, MBE) including MBE
  (mean bias error), which most papers in this batch omit - relevant to
  this project's own emphasis on bias, not just magnitude, of error.

## Other observations

- Cannot fully verify RMSE^2~mean^2+SD^2 or MAE<=RMSE from the fragments
  extracted here; the isolated numbers quoted above (MAE=0.1727 vs
  RMSE=0.0298 for the proposed model on one dataset) show MAE > RMSE,
  which is the same impossible pattern flagged in molu2024bilstmaadc and
  elmousalami2025saispf elsewhere in this batch - a THIRD independent
  instance of this exact pattern across the survey. Given how consistently
  this specific error type recurs (RMSE < MAE) across three unrelated
  papers/journals/publishers, it may indicate a systematic mislabeling
  convention in parts of this literature (e.g. some papers computing
  RMSE on a different, smaller-magnitude normalized scale than MAE) rather
  than three independent typos - worth investigating as its own pattern in
  the survey writeup rather than three isolated flags.
- No shared authors/institution with any other paper coded in this batch
  (Sejong University / Chung-Ang University, South Korea).
