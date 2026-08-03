# Audit: ma2024cnnlstmspatiotemporal

Source: `data/papers/Engineering Reports - 2024 - Ma - Integrated CNN-LSTM for Photovoltaic Power Prediction based on Spatio-Temporal Feature.pdf` (10 pages, 43,591 chars)

Ma, J., Huo, M., Han, J., Liu, Y., Lu, S., Yu, X. (2024/2025). "Integrated
CNN-LSTM for Photovoltaic Power Prediction based on Spatio-Temporal
Feature Fusion." *Engineering Reports* (Wiley). https://doi.org/10.1002/eng2.13088
(received 22 Feb 2024, published as a 2025 issue per the extracted Wiley
running header "2025, 1")

## Coded fields

**year**: 2024 (received date) / 2025 (issue header shows "2025, 1") -
recorded as 2024 per the paper's own "Received: 22 February 2024" date;
flag the 2024/2025 ambiguity if cross-referencing by year.

**venue**: Engineering Reports (Wiley), https://doi.org/10.1002/eng2.13088

**dataset**: 6 neighboring PV plants (1 target + 5 reference: A-F),
Lingchuan County, Jincheng City, Shanxi Province, China, 8 meteorological/
plant features, 5-min sampling, June 30 2021 - August 30 2022, daytime
8am-8pm only, 9,375 cleaned samples per plant | "The simulation data were
obtained from actual PV data from six neighboring PV power plants in
Lingchuan County, Jincheng City, Shanxi Province. The plants included in
the study are designated as the target plant, A, and the reference
plants, B, C, D, E, and F." (p.5); "after the anomaly analysis and nulling
procedures, 9,375 PV timing data were generated for each PV plant." (p.6)

**night_hours_excluded = yes** | "To exclude the effect of nighttime PV
power predictions, we select PV data from June 30, 2021, to August 30,
2022, from 8 a.m. to 8 p.m. for the simulation." (p.5) - explicit stated
rationale AND explicit clock-hour filter, one of the clearest and most
directly-stated night-exclusion quotes in this survey batch.

**baseline_used = own_components** | "an integrated network architecture
comprising three individual models, CNN, LSTM, and CNN-LSTM, is designed.
The SENet attention mechanism is utilized to add non-linear integration
weights to the outputs of the individual models." (Abstract, p.1) -
comparators (CNN, LSTM, CNN-LSTM) are exactly the individual sub-models
fused into the paper's own ensemble. EXHAUSTIVE CHECK: zero matches for
persistence/naive/climatology/reference model/reference forecast anywhere
in 43,591 characters.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Reported metrics are RMSE and MAE only (p.5), both against
ground truth.

**weather_source = measured** | 8 named features (humidity, pressure,
cloud thickness, rainfall, solar irradiance, temperature, wind direction,
wind speed) drawn from the 6 plants' own recorded data (see dataset
quote); no NWP/forecast-weather language anywhere.

**split_type = not_stated** | "These data sets were then divided into two
subsets: a training set and a test set, with a ratio of 9:1." (p.6) -
ratio only, no statement of chronological vs. random ordering.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches for seed/
repeated runs/"+/-"/confidence interval/error bars/multiple runs
describing model accuracy. The one "standard deviation" hit (p.5) is
RMSE's own textbook definition ("The RMSE represents the sample standard
deviation of the differences between predicted and observed values"), not
a reported dispersion statistic.

**code_available = not_stated** | "Data Availability Statement: If you
need access to the original data or code, please contact the author,
Shunfa Lu, for your requests." (p.9) - explicitly mentions code, but
gated behind "contact the author," which the coding rule's own worked
example treats as not_stated rather than "yes."

**key_claim**: "For PV power prediction at 6 neighboring plants in Shanxi
Province, China (daytime-only 8am-8pm data, Jun 2021-Aug 2022, 5-min
sampling, 9:1 train/test split, ordering not stated), a SENet-weighted
ensemble of CNN + LSTM + CNN-LSTM reduces RMSE by 13.5%, 6.9%, and 5.1%
versus the individual CNN, LSTM, and CNN-LSTM models respectively, at a
5-minute prediction interval; the paper also reports separate 5-minute and
10-minute-ahead comparisons (Section 4.3.4) and stratifies performance by
sunny/rainy/cloudy sky conditions (Table 4)." | Abstract (p.1); Section
4.3.4 (p.9).

## What this paper does WELL

- Explicit, clean, stated night-hour exclusion with both a rationale
  ("to exclude the effect of nighttime PV power predictions") and exact
  clock hours (8am-8pm) - one of the strongest examples of this in the
  batch.
- Explicitly states multiple forecast horizons (5 min and 10 min ahead,
  Section 4.3.4) rather than leaving lead time unstated, unlike several
  other papers in this batch.
- Stratifies results by sky condition (sunny/rainy/cloudy, Table 4) -
  directly analogous to this project's own RQ3 sky-condition
  stratification (src/eval/sky.py), though this paper's classes are
  weather-report categories rather than a k_ghi-derived classifier.
- Uses multi-plant spatial correlation (5 reference plants) as an
  explicit additional feature source, with the correlation values reported
  in a table (Table 2) rather than asserted without evidence.

## Other observations / consistency checks

- MAE < RMSE reported (13.5%/6.9%/5.1% RMSE reduction figures are internally
  consistent as percentages, not absolute values, so no direct MAE-vs-RMSE
  magnitude check is possible from the abstract numbers alone) - no
  contradiction found in what was extracted, unlike the MAE>RMSE pattern
  flagged in three other papers in this batch (molu2024bilstmaadc,
  elmousalami2025saispf, hussain2022hybridgrucnn).
- No shared authors/institution with any other paper coded in this batch
  (State Grid Shanxi Electric Power Company / State Grid Block Chain
  Technology, China).
