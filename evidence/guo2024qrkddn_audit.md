# Audit: guo2024qrkddn

Source: `data/papers/sensors-24-01593-v3.pdf` (24 pages, 94,951 chars)

Guo, W., Xu, L., Wang, T., Zhao, D., Tang, X. (2024). "Photovoltaic Power
Prediction Based on Hybrid Deep Learning Networks and Meteorological
Data." *Sensors* (MDPI) 24:1593.

**SAME SITE AS THIS PROJECT, WITH A GENUINE PUBLIC DATA LINK**: "the
Desert Knowledge Australia Solar Centre (DKASC) Hanwha Solar dataset was
selected as the research subject." (p.8) - sixth paper in this survey
batch to use DKASC-family data (after zhou2024, hou2024, ye2026,
hussain2022, and alharkan2023), and the first to cite the actual public
DKASC portal URL as its own data-availability statement (see
code_available below) - directly confirming that this is the same open
dataset this project's own CLAUDE.md describes.

DOI: https://doi.org/10.3390/s24051593

## Coded fields

**year**: 2024 | "Sensors 2024, 24, 1593"

**venue**: Sensors (MDPI) 24:1593

**dataset**: DKASC Hanwha Solar array, Alice Springs, full year 2020,
daytime 6:00-19:00 only, features: temperature, relative humidity,
radiation, rainfall, PV power | "The original data used for analysis
encompass the output power of the PV generation system and meteorological
data collected through an array of sensors from 1 January to 31 December
2020. The weather data comprise crucial meteorological variables,
including temperature, relative humidity, radiation data, and rainfall.
To ensure the accuracy of the results, only data collected between 6:00
and 19:00 [were used]." (p.8)

**night_hours_excluded = yes** | "To ensure the accuracy of the results,
only data collected between 6:00 and 19:00 [were used]." (p.8) - explicit
clock-hour daytime filter.

**baseline_used = own_components** | Ablation/comparison set: QR-GRU,
QR-BiGRU, QR-BiGRU-Attention, QR-CNN-BiGRU, QR-CNN-BiLSTM-Attention, all
compared against the proposed QRKDDN (p.28-29: "...is 32.51% lower than
that of the QR-GRU model, 26.20% lower than that of the QR-BiGRU model,
21.58% lower than that of the QR-BiGRU-Attention model, 13.23% lower than
that of the QR-CNN-BiGRU model..."). The paper calls QR-BiGRU-Attention
its "reference model" (p.28: "we compared the reference model,
QR-BiGRU-Attention, with the QRKDDN prediction results") - this is an
INTERNAL ablation reference (one component fewer than the full proposal),
not a persistence/climatology reference forecast in this survey's sense.
EXHAUSTIVE CHECK: zero matches for persistence/naive/climatology anywhere
in 94,951 characters; "benchmark models" (p.4, p.33) always refers to this
same QR-prefixed ablation family plus classical ML models mentioned
generically ("comparisons with classical machine learning models,"
Abstract, p.1) but never named as persistence/climatology specifically.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Point-prediction metrics are RMSE/MAE/R2-style (implied by
"higher R2" language, p.28); probabilistic metrics are CRPS (Continuous
Ranked Probability Score), PICP (Prediction Interval Coverage
Probability), and PINAW (Prediction Interval Normalized Average Width) -
genuine interval/probabilistic evaluation metrics, but none is a skill
score against a reference forecast.

**weather_source = measured** | on-site DKASC sensor array (temperature,
humidity, radiation, rainfall - see dataset quote above); no NWP/forecast-
weather language anywhere.

**split_type = rolling** | "Training/test set ratio 0.7/0.3" and
"Cross-validation method: Rolling cross validation" (Table, p.20) - this
is the only paper in this survey batch to explicitly name "rolling"
cross-validation as its split methodology, one of the allowed_values
categories that otherwise went completely unused across the rest of this
batch.

**n_seeds = not_stated**

**variance_reported = no** [see STRENGTH flag] | EXHAUSTIVE CHECK: zero
matches for seed/repeated runs/"+/-"/confidence interval/error bars/
multiple runs describing run-to-run training variance. However, the paper
reports genuine PROBABILISTIC uncertainty via PICP and PINAW (prediction
interval coverage and width) derived from quantile regression and kernel
density estimation - this is real uncertainty quantification, just not
the seed-variance axis this survey tracks (same category of exception as
nadour2026cnnbilstm's bootstrap CIs elsewhere in this batch), and is
recorded as a strength below.

**code_available = not_stated** | "Data Availability Statement: The
required datasets for the experiment can be obtained for free from
https://dkasolarcentre.com.au/ (accessed on 4 May 2023)." (p.23) - a
genuine, specific, freely-accessible public DATA URL (the strongest data-
availability statement of any paper in this batch), but says nothing
about code.

**key_claim**: "For PV power prediction at the DKASC Hanwha Solar array
(full year 2020, daytime 6:00-19:00 only, 70/30 rolling cross-validation
split), the proposed QRKDDN model (Pearson feature selection + GMM
day-clustering + CNN-BiGRU-attention with quantile regression and kernel
density estimation) outperforms QR-GRU, QR-BiGRU, QR-BiGRU-Attention,
QR-CNN-BiGRU, and QR-CNN-BiLSTM-Attention on deterministic (point), interval,
and probabilistic prediction metrics; on a rainy/highly-fluctuating-weather
case study specifically, adding the CNN layer reduces point-prediction RMSE
by 24.29% and improves R2 by 5.11% versus the QR-BiGRU-Attention reference
ablation, with PICP 6.51% higher and PINAW 21.17% lower." | pp.1, 28-29.
Forecast horizon is not explicitly stated as a lead time anywhere in the
passages read.

## What this paper does WELL

- Explicit, clean, stated daytime-only clock-hour filter (6:00-19:00).
- Explicitly names "Rolling cross validation" as its split methodology -
  the only paper in this survey to use this specific, time-series-
  appropriate term (most others give only a static ratio with no stated
  ordering).
- Reports genuine probabilistic/interval forecasting metrics (CRPS, PICP,
  PINAW) alongside point-prediction error - a materially richer evaluation
  than every other paper in this batch, none of which attempt interval or
  probabilistic prediction at all.
- Explicit case-study stratification by weather regime (rainy/highly-
  fluctuating vs. general), directly relevant to this project's own RQ3.
- Names the actual public DKASC portal URL as its data source, the
  clearest and most specific data-availability statement of any paper
  coded in this batch.

## Other observations

- No shared authors/institution with any other paper coded in this batch
  (Wuhan University of Technology, China).
- Given the DKASC-family cluster now spans 6 papers in this 22-paper
  batch (zhou2024, hou2024, ye2026, hussain2022, alharkan2023, guo2024),
  the survey writeup should probably discuss DKASC's outsized presence in
  this literature as its own observation - it is a genuinely popular
  benchmark dataset, which cuts both ways: results are more comparable
  across these papers in principle, but also means "the literature" on PV
  forecasting evaluation practice is, in this sample, disproportionately
  literature about ONE dataset family.
