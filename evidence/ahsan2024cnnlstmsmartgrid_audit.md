# Audit: ahsan2024cnnlstmsmartgrid

Source: `data/papers/Paper+5.pdf` (9 pages, 40,588 chars)

Ahsan, A., Zafar, A., Afzal, M.A., Javed, M., Zafar, D., Ali, M. (2024).
"Improved Solar Power Prediction Using CNN-LSTM Models for Optimized
Smart Grid Performance." *Journal of Engineering, Science and
Technological Trends (JESTT)* 1(2). https://doi.org/10.48112/jestt.v1i2c.5

SCOPE NOTE: this paper forecasts at DAILY resolution (~365 values/year:
one daily power total and one daily radiance total per day), not
hourly/sub-hourly like most other papers in this batch - this changes how
several fields should be read (see night_hours_excluded below).

DOI: https://doi.org/10.48112/jestt.v1i2c.5

## Coded fields

**year**: 2024 | "Vol. 1, Issue 2, August 2024"

**venue**: Journal of Engineering, Science and Technological Trends
(JESTT) 1(2), doi:10.48112/jestt.v1i2c.5

**dataset**: "a large-scale solar power facility," location/technology
never named, one year of DAILY values (~365 samples), two metrics only:
daily power generation and daily radiance | "The dataset comprised daily
power output data from a large-scale solar power facility collected over
one year." (p.3); "The methodology involves collecting one year's worth of
real-time data from a solar farm, focusing on two key metrics: daily power
generation and radiance. Each metric includes approximately 365 values,
representing a year of data." (p.3)

**night_hours_excluded = not_stated** | Not applicable in the usual sense:
the data is DAILY-AGGREGATED (one value per day), so there is no intraday
night/day distinction to exclude at this resolution. EXHAUSTIVE CHECK:
zero matches for night/daytime/daylight/sunrise/sunset/zenith/clear-sky/
diurnal anywhere in the text; the paper never discusses resolution-related
night handling because its resolution makes the question moot, not because
it addressed and dismissed it.

**baseline_used = other_ML** | "This study employed three machine learning
models: RF, LSTM, and CNN LSTM to analyze a static time series dataset...
This study found that adding a CNN to the LSTM model improved accuracy
and reduced both MAE and MSE compared to the LSTM and RF models." (p.2-3)
RF is an external ML paradigm, not a sub-component of CNN-LSTM; LSTM is
architecturally a component of CNN-LSTM but is also independently a
"reference model" the paper names explicitly (p.2: "a hybrid framework
outperforms the reference models (LSTM and RF)"). No persistence/
climatology own baseline. EXHAUSTIVE CHECK: the only "persistence" hit in
the whole text (p.14, reference [24]: "Data-driven day-ahead PV estimation
using autoencoder-LSTM and persistence model") is a CITED paper's title in
the bibliography, not this paper's own comparison.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence" describing this paper's own results. Metrics used are MAE and
MSE (Abstract, p.1) plus RMSE/MAPE mentioned generically in a related-work
sentence (p.2) describing a different cited study's metric choice.

**weather_source = measured** | the only weather-adjacent variable used by
THIS paper's own model (as distinct from the input-feature list of a
DIFFERENT cited paper, ref [31], discussed in the same paragraph) is daily
radiance: "focusing on two key metrics: daily power generation and
radiance" (p.3) - a measured historical value from the facility's own
records, not an NWP forecast.

FLAG: p.2 lists a much richer feature set ("panel surface temperature,
accumulated energy, solar radiation, humidity, irradiance, and past solar
energy") immediately after describing ref [31]'s LSTM-autoencoder model -
the sentence "These models utilize diverse input features such as..." most
plausibly refers back to ref [31]'s models, not this paper's own (which
Section 2 later confirms uses only "daily power generation and radiance").
Read carefully to avoid misattributing another paper's feature list to
this one; flagged in case this reading is wrong.

**split_type = not_stated** | "The data was split, with 80% (10 months)
used for training the models and 20% (two months) reserved for final
predictions" (p.4) - a ratio (and, unusually, a duration in months) is
given, but no word indicating chronological vs. random ordering (no
"previous," "first," "chronological," etc., unlike hou2024vmdwoalstm or
mayer2022physmlhybrid elsewhere in this batch, which used similar
month-based phrasing WITH an explicit ordering cue).

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches for seed/
repeated runs/standard deviation/"+/-"/confidence interval/error bars/
multiple runs anywhere in the text.

**code_available = not_stated** | "Data availability statement: The whole
data of this research is included in this article." (p.8) - states DATA
is included (a stronger positive than most "on request" statements seen
elsewhere in this batch), but says nothing about code at all.

**key_claim**: "For daily-resolution solar power and radiance forecasting
at an unnamed large-scale solar facility (one year of ~365 daily values,
80%/20% train/test split by month-count, ordering not stated), CNN-LSTM
outperforms both LSTM and Random Forest, achieving MAE=0.1335 and
MSE=0.0497 for daily power generation (Abstract, p.1); forecast horizon is
implicitly one day (next-day) given the daily aggregation, though never
explicitly stated as such."

## What this paper does WELL

- Reports results on two separate target variables (daily power
  generation AND daily radiance) rather than only one, with separate
  tables/figures for each (Table/Figs for "Daily Power Generation" and
  "Radiance" datasets, p.9-10 per the extracted section headers).
- States data is included directly in the article rather than gated
  behind a request - a stronger data-availability statement than most
  papers in this batch, even though it says nothing about code.

## Other observations

- Venue note: JESTT is a newer, less established journal (Vol. 1, Issue 2,
  first published August 2024) published by SCOPUA; this does not bear on
  the coding itself but is worth noting for context on how much
  editorial/review rigor to assume when interpreting the paper's claims.
- Possible further DKASC connection (unconfirmed, citation-only): ref [31]
  is described as using "a dataset from a 23.40 kW PV power plant in
  Australia" (p.2) - 23.40 kW matches EXACTLY the capacity this survey's
  own hussain2022hybridgrucnn entry gives for "DKASC-AS-1B" ("generated
  23.4 kW from 4 x 30 number of panels"). This is a citation to a THIRD
  paper (ref [31], not fully identified in the extracted text), not a
  direct dataset use by THIS paper, but it further underscores how
  concentrated this literature is around the same handful of public DKASC
  arrays - worth a combined note in the survey writeup alongside the
  zhou2024/hou2024/hussain2022/ye2026 DKASC entries already coded.
- No shared authors/institution with any other paper coded in this batch.
