# Audit: lim2022cnnlstmsunnycloudy

Source: `data/papers/energies-15-08233.pdf` (17 pages, 70,950 chars)

Lim, S.-C., Huh, J.-H., Hong, S.-H., Park, C.-Y., Kim, J.-C. (2022). "Solar
Power Forecasting Using CNN-LSTM Hybrid Model." *Energies* (MDPI) 15:8233.

DOI: https://doi.org/10.3390/en15218233

## Coded fields

**year**: 2022 | "Energies 2022, 15, 8233"

**venue**: Energies (MDPI) 15:8233

**dataset**: PV power plant, Busan, Korea, 10 September 2019 - 22 July
2021, classified sunny/cloudy via Korea Meteorological Administration
(KMA) weather-condition steps | "This study used the power generation
data collected from a PV power plant in Busan, Korea... The data were
collected from 10 September 2019 to 22 July 2021. The data defined as
corresponding to sunny and cloudy weather conditions by the Korea
Meteorological Administration were used. The Korea Meteorological
Administration defines weather conditions in 11 steps. In this study,
steps 1-5 were classified as sunny, while steps 6-11 were classified as
cloudy." (p.8)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches for
an explicit exclusion rule. "On a sunny day, the graph has a semi-circular
data distribution affected by sunrise and sunset" (p.9) describes the
SHAPE of the power curve (implying night/dawn/dusk data is present in the
plotted series), not a stated exclusion.

**baseline_used = own_components** | The only models compared are the
paper's own CNN (weather classifier) and LSTM (power forecaster) stages
and their combination; no external comparator model is named anywhere.
EXHAUSTIVE CHECK: zero matches for persistence/naive/benchmark/reference
forecast/climatology/reference model in the whole text.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Metrics are MAPE, RMSE, MAE, R2 (Table 2, p.11).

**weather_source = none** [exact match to the coding rules' own worked
exception] | "Each model only uses power generation data." (p.7, stated
twice, describing both the CNN classification stage and LSTM forecasting
stage in Section 4's step-by-step description). The KMA sunny/cloudy
"weather condition" categorization is used ONLY to route input to one of
two separately-trained LSTM sub-models (a routing/labeling function), not
as a continuous input feature to the power-forecasting network itself.
This is precisely the coding rule's own example: "Weather used ONLY to
label days for a classifier is NOT a weather input - code none and say
so." Coded none accordingly, rather than "measured."

**split_type = not_stated** | EXHAUSTIVE CHECK: zero matches anywhere in
70,950 characters for a training/test split ratio, percentage, or ordering
statement of any kind. This paper does not state its split ratio at all -
one of only two papers in this batch (with vennila2022solarensemble) to
omit a split ratio entirely.

**n_seeds = not_stated**

**variance_reported = no** [see STRENGTH flag] | "Figures 10 and 11 plot
the residuals between observed and forecasted values for the sunny and
cloudy day test data. The dotted line indicates the standard deviation
(SD) corresponding to the residuals." (p.11); "The number of data
deviating from the standard deviation section is less than the..." (p.13)
- per the coding rule, RESIDUAL standard deviation of a single model is
NOT run-to-run variance across seeds/repeats, so this is coded "no" - but
is recorded as a STRENGTH below: the paper visualizes residual dispersion
directly on its plots, which is more transparent than a bare point
estimate even though it is not the seed-variance axis this survey tracks.
EXHAUSTIVE CHECK: no seed/repeated-run/"+/-"/confidence-interval language
describing model-to-model or run-to-run accuracy spread anywhere.

**code_available = not_stated** | "Data Availability Statement: Not
applicable." (p.16) - the exact MDPI boilerplate the coding rules
explicitly instruct to code as not_stated, not "no."

**key_claim**: "For PV power forecasting at a Busan, Korea plant (Sep
2019-Jul 2021, split ratio never stated), a hybrid model that first
CNN-classifies each day as sunny/cloudy (via KMA weather-condition steps)
and then routes to one of two separately-trained LSTM power-forecasting
sub-models achieves MAPE=4.58/RMSE=43.87/MAE=34.00/R2=0.99 on sunny days
and MAPE=7.21/RMSE=9.09/MAE=6.97/R2=0.99 on cloudy days (Table 2, p.11);
forecast horizon is never explicitly stated in the passages read."

## FLAG: abstract's cloudy-day MAPE does not match its own results table

Abstract (p.1): "The proposed model achieved a mean absolute percentage
error of 4.58 on a sunny day and 7.06 on a cloudy day in the quantitative
evaluation."

Table 2 (p.11, "Quantitative validation results of the power generation
forecasting model"): "Sunny 4.58 ... Cloudy 7.21 ..." (MAPE column)

The sunny-day figure matches exactly (4.58 = 4.58), but the abstract's
cloudy-day MAPE (7.06) does not match the results table's cloudy-day MAPE
(7.21) - a small but exact, verbatim-checkable discrepancy, in the same
category as the abstract/table mismatches already flagged in
molu2024bilstmaadc and hussain2022hybridgrucnn elsewhere in this batch.
Not resolved here (could be a late-stage number revision that was updated
in the table but not the abstract, or a transcription slip in either
direction).

## What this paper does WELL

- Explicitly separates and reports performance by sky condition (sunny vs
  cloudy), directly relevant to this project's own RQ3 sky stratification,
  though via a different classifier (official meteorological-agency
  categorical steps rather than a continuous k_ghi-derived measure).
- Visualizes residual standard deviation directly on its result plots
  (Figs. 10-11) rather than reporting only a single point-estimate metric.
- Correctly avoids feeding a same-timestep-derived weather LABEL into the
  power-forecasting network as a continuous feature - the sunny/cloudy
  classification only selects WHICH sub-model to use, which is a cleaner
  design than several other papers in this batch that blend contemporaneous
  weather values directly into the predictor (cf. zhou2024
  cnnlstmattnbayes's oracle-style same-timestep weather input, flagged
  elsewhere in this survey).

## Other observations

- No shared authors/institution with any other paper coded in this batch
  (TEF Co./Korea Maritime and Ocean University/Sunchon National
  University, South Korea).
