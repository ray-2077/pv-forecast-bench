# Audit: ye2026distributedcnnlstm

Source: `data/papers/ctaf131.pdf` (8 pages, 41,464 chars)

Ye, H., Xu, T. (2026). "Research on the distributed photovoltaic power
prediction method based on CNN-LSTM." *International Journal of
Low-Carbon Technologies* 21:1-8. https://doi.org/10.1093/ijlct/ctaf131

**SAME SITE AS THIS PROJECT**: "historical photovoltaic power and
meteorological data from Alice Springs Photovoltaic Research Center in
Australia from June to August 2017 were used" (p.5) - this is DKASC Alice
Springs, the exact site this project's own dataset comes from (this
project uses 2011-2015; this paper uses June-August 2017 - no temporal
overlap, but the same physical site/network). Fourth paper in this survey
batch drawing on the DKA Solar Centre family (after zhou2024, hou2024, and
implicitly related to this project's own data source), and the first to
use the identical site rather than a sister site.

DOI: https://doi.org/10.1093/ijlct/ctaf131

## Coded fields

**year**: 2026 | "International Journal of Low-Carbon Technologies, 2026,
21, 1-8"

**venue**: International Journal of Low-Carbon Technologies 21:1-8

**dataset**: DKASC Alice Springs, Australia, June-August 2017, 15-min
sampling (96 points/day), 8821 valid records | "historical photovoltaic
power and meteorological data from Alice Springs Photovoltaic Research
Center in Australia from June to August 2017 were used, Sampling intervals
of 15 minutes were set, yielding 96 data points per day and a total of
8821 valid datasets." (p.5)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches
for night/daytime/daylight/sunrise/sunset/zenith/clear-sky/diurnal
anywhere in the text.

**baseline_used = own_components** | Table 2 comparison (p.8): CNN,
CNN-LSTM, 2D-CNN-LSTM-SSA, 2D-CNN-LSTM-ISSA - each row adds one more of
the paper's own pipeline stages. EXHAUSTIVE CHECK: zero matches for
persistence/naive/benchmark/reference forecast/climatology/reference model
anywhere in the text.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Reported metrics are MAE and RMSE only (Table 2, p.8).

**weather_source = measured** | "historical photovoltaic power and
meteorological data from Alice Springs Photovoltaic Research Center" (p.5)
- on-site historical records, no NWP/forecast-weather language anywhere.

**split_type = not_stated** | "The experimental data were partitioned into
training and validation sets at an 8:2 [ratio]" (p.5) - ratio only, no
statement of chronological vs. random ordering anywhere in the text.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: no seed/repeated-run/
standard-deviation-of-accuracy/confidence-interval/error-bar language
describing model performance anywhere. The one "random" hit in the text
(p.7, "alpha in (0,1] is a random...") is a sparrow-search-algorithm (SSA)
optimizer parameter, not run-to-run model variance.

**code_available = not_stated** | No "Data availability" statement of any
kind found in the extracted text (unlike most other papers in this batch,
this one appears to have no formal data/code availability section at
all - checked the Author Contributions section (p.8) and surrounding text,
no data/code statement present).

**key_claim**: "For distributed PV power prediction at DKASC Alice Springs
(Jun-Aug 2017, 15-min data, 8:2 train/validation split, ordering not
stated), the proposed 2D-CNN-LSTM-ISSA model achieves the lowest MAE/RMSE
of all compared variants (CNN, CNN-LSTM, 2D-CNN-LSTM-SSA), with the
CNN-LSTM-SSA fusion showing a 32.55% MAE and 33.36% RMSE reduction over
the single CNN-LSTM model and a 905.45-second reduction in modeling time;
a companion experiment shows that reducing the training-set fraction
(increasing the test-set fraction) reduces prediction accuracy across all
models, though the specific alternate ratio used is not given in the
extracted text; forecast horizon is never stated."

## What this paper does WELL

- Runs an explicit training/test SIZE ablation ("first group" vs "second
  group" with different training-set fractions) and reports that shrinking
  the training set degrades accuracy across every model tested - directly
  relevant to (and consistent with) this project's own Finding 7 discussion
  of training-length sensitivity, though in the opposite direction (this
  paper finds training length DOES matter, where this project's own
  finding at a different site/window was closer to "barely matters").
- Reports modeling/training TIME alongside accuracy (Table 2), an
  efficiency axis relevant to RQ4.
- Ablates its own pipeline stage-by-stage (CNN -> CNN-LSTM -> +SSA -> +ISSA)
  rather than only the full stack.

## Other observations

- No MAPE used at all (only MAE/RMSE) - avoids the near-zero-denominator
  MAPE problem flagged in several other papers in this batch, though this
  may simply be a metric choice rather than a deliberate avoidance.
- Cannot check RMSE^2~mean^2+SD^2: no separate residual mean/SD reported
  alongside RMSE. MAE<=RMSE holds for all four models in Table 2 as
  extracted (e.g. proposed model MAE=0.7952 <= RMSE=1.4734) - no
  contradiction found, unlike molu2024bilstmaadc and
  elmousalami2025saispf elsewhere in this batch.
- Short paper (8 pages) with comparatively thin methodology reporting
  overall (no data-availability statement, no split-ordering statement) -
  consistent with the general pattern in this batch that shorter/faster-
  turnaround venues report less protocol detail.
- No shared authors/institution with any other paper coded in this batch.
