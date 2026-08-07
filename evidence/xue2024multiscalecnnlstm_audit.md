# Audit: xue2024multiscalecnnlstm

Source: `data/papers/energies-17-03877.pdf` (13 pages, 44,325 chars)

Xue, H., Ma, J., Zhang, J., Jin, P., Wu, J., Du, F. (2024). "Power
Forecasting for Photovoltaic Microgrid Based on MultiScale CNN-LSTM
Network Models." *Energies* (MDPI) 17:3877.

**SAME RESEARCH GROUP AS ma2024cnnlstmspatiotemporal, ALSO CODED IN THIS
BATCH**: shared author (Junwei Ma appears on both papers), shared
institution (State Grid Shanxi Electric Power Company / State Grid Block
Chain Technology, Beijing - identical affiliations to ma2024's author
list), and a strikingly similar dataset: "a microgrid consisting of six
distributed PV power stations, denoted as A, B, C, D, E and F, located in
Shanxi Province, China" (p.9) versus ma2024's "six neighboring PV power
plants in Lingchuan County, Jincheng City, Shanxi Province... designated
as the target plant, A, and the reference plants, B, C, D, E, and F"
(same six-plant A-F naming scheme). The two papers use DIFFERENT date
ranges (this paper: 7 Nov 2022 - 9 Nov 2023; ma2024: 30 Jun 2021 - 30 Aug
2022), so they are not simply duplicate submissions of the same result,
but they are very likely companion papers built on the same underlying
6-station monitoring deployment, possibly with overlapping sensors/sites.
Recommend treating these two entries as NOT independent data points for
any "how many papers in the literature do X" count in the survey - the
same research group's practices (night-hour handling, split ratio, no
naive baseline) would otherwise be counted twice.

DOI: https://doi.org/10.3390/en17163877

## Coded fields

**year**: 2024 | "Energies 2024, 17, 3877"

**venue**: Energies (MDPI) 17:3877

**dataset**: 6 distributed PV stations (A-F), Shanxi Province, China, 8
meteorological parameters, 5-min native / 30-min aggregated, daytime
7am-7pm only, 7 Nov 2022 - 9 Nov 2023, 9000 time series per station | see
night_hours_excluded quote below for the full passage; "The PV
characterization data consist of eight meteorological parameters (e.g.,
solar irradiance, ambient temperature, wind speed, etc.)... The final
dataset consists of 9000 time series for each station" (p.9)

**night_hours_excluded = yes** | "Since PV power stations do not produce
power output at night, data from 7 a.m. to 7 p.m. during the period of 7
November 2022 to 9 November 2023 were selected for the experiments." (p.9)
- near-identical construction to this project's own CLAUDE.md rule 2
rationale, and to the coding rules' own worked example #4.

**baseline_used = own_components** | "comparative experiments were
conducted on the test set against baseline models: CNN, LSTM and CNN-LSTM
[21]. CNN-LSTM (multiscale) utilizes a multiscale architecture, while
CNN-LSTM (cascade) employs a two-stage cascade multiscale architecture."
(p.11) - CNN, LSTM, and CNN-LSTM are the exact component architectures
the proposed multiscale/cascade models build on. EXHAUSTIVE CHECK: zero
matches for persistence/naive/climatology/reference model/reference
forecast describing this paper's own comparison (the one "benchmarking"
hit, p.13, is a citation title in the reference list, ref [8],
Pombo et al., describing a different paper).

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Metrics are RMSE and MAE only (Section 3.4, p.9).

**weather_source = measured** | 8 named on-site meteorological parameters
(wind speed, humidity, pressure, rainfall, cloud density, solar
irradiation angle, temperature, solar irradiance - Table 1, p.10); no
NWP/forecast-weather language anywhere.

**split_type = not_stated** | "which were divided into a training set and
a test set at a ratio of 9:1" (p.10) - ratio only, no statement of
chronological vs. random ordering.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches for seed/
repeated runs/"+/-"/confidence interval/error bars/multiple runs. The two
"standard deviation" hits (p.9) are RMSE's own textbook definition ("RMSE
represents the sample standard deviation of the difference between the
predicted and true values") and a correlation-coefficient formula
component (p.6), neither a reported model-variance statistic.

**code_available = not_stated** | "Data Availability Statement: Please
contact the authors with regard to data requests." (p.12) - data only,
gated behind contact, and does not mention code at all.

**key_claim**: "For PV microgrid power forecasting across 6 spatially
correlated stations in Shanxi Province, China (daytime-only 7am-7pm data,
Nov 2022-Nov 2023, 9:1 train/test split, ordering not stated), multiscale
and cascade CNN-LSTM variants that fuse the target station's own features
with the most-correlated neighboring station's power and irradiance
outperform CNN, LSTM, and plain CNN-LSTM baselines on RMSE and MAE;
forecast horizon (implied by the half-hour aggregation) is not explicitly
stated as a lead time anywhere in the extracted text."

## What this paper does WELL

- Explicit, well-justified night-hour exclusion with exact clock hours -
  matches the coding rules' own worked positive example almost verbatim.
- Reports a full correlation table (Table 1, p.10) between each individual
  meteorological parameter and PV output before using it as a feature,
  rather than asserting relevance without evidence.
- Explicitly selects the spatial reference station by measured correlation
  strength rather than an arbitrary neighbor, with the correlation values
  shown (Figure 7).

## Other observations

- Given the shared-authorship/shared-site relationship with
  ma2024cnnlstmspatiotemporal noted above, the two papers' respective
  strengths and gaps (night exclusion: yes/yes; baseline: own_components/
  own_components; skill score: no/no; split: 9:1 ratio/9:1 ratio, ordering
  not stated in both) are essentially a single research group's practice
  pattern reported twice, not two independent literature data points -
  flagged for the survey writeup to avoid double-counting.
- No shared authors/institution with any OTHER paper coded in this batch
  besides ma2024cnnlstmspatiotemporal.
