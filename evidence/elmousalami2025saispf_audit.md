# Audit: elmousalami2025saispf

Source: `data/papers/buildings-15-02785-v2.pdf` (33 pages, 98,742 chars)

Elmousalami, H., Hui, F.K.P., Alnaser, A.A. (2025). "Enhancing Smart and
Zero-Carbon Cities Through a Hybrid CNN-LSTM Algorithm for Sustainable
AI-Driven Solar Power Forecasting (SAI-SPF)." *Buildings* 15:2785 (MDPI).

DOI: https://doi.org/10.3390/buildings15152785

## Coded fields

**year**: 2025 | "Buildings 2025, 15, 2785", "Published: 6 August 2025"

**venue**: Buildings 15:2785 (MDPI)

**dataset**: Benban Solar Park, Aswan, Egypt, and Sakaka Solar Power Plant,
Saudi Arabia - two utility-scale solar installations, high-resolution
meteorological + operational data (irradiance, temperature, wind speed,
humidity, solar zenith angle, panel parameters) | "data collection from
two major utility-scale solar installations: Benban Solar Park in Egypt
and Sakaka Solar Power Plant in Saudi Arabia" (p.8)

**night_hours_excluded = not_stated** | The paper describes rather than
excludes night behavior: "Power generation begins at sunrise, gradually
increases to a peak around midday... and then steadily declines as the
sun sets... with maximum output... occurring between 10:00 and 14:00 h and
minimal output during nighttime hours." (p.12) - this describes the SHAPE
of a figure (implying night rows are present in the dataset with near-zero
values), not a stated exclusion rule. EXHAUSTIVE CHECK: no other daytime/
daylight/night-filter language anywhere in the text.

**baseline_used = other_ML** | "the model training stage involves fitting
several machine-learning and deep learning algorithms, including Random
Forest (RF), Support Vector Machine (SVM), Gradient Boosting Machine
(GBM), Long Short-Term Memory (LSTM) networks, Convolutional Neural
Networks (CNNs), a hybrid CNN-LSTM model, and the ARIMA statistical
method." (p.9) - external ML/statistical comparators, not persistence or
climatology, and not literal ablation components of CNN-LSTM.
EXHAUSTIVE CHECK: zero matches for persistence/naive/climatology/
reference model/reference forecast anywhere in the text.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Reported metrics are MAPE, RMSE, MAE, and adjusted R2
(p.9) - all against ground truth.

**weather_source = measured** | "high-resolution meteorological and
operational data, including... solar irradiance, ambient temperature,
wind speed, humidity, solar zenith angle" (p.8) collected from the two
plants' own monitoring; no NWP/forecast-weather language anywhere.

**split_type = not_stated** [see flag - genuinely ambiguous] | "The
refined dataset is then partitioned into training (70%), validation (15%),
and testing (15%) sets" (p.9) - no statement of chronological vs. random
ordering for this primary split. SEPARATELY, the paper also states: "To
ensure robustness and generalizability, K-fold cross-validation is applied
during model development, averaging results across multiple partitions of
the training set." (p.9)

FLAG (uncertain field): two different validation procedures are named -
a 70/15/15 static split (which appears to produce the actual reported
Table results) and K-fold cross-validation "during model development"
(role and relationship to the 70/15/15 split not fully specified - applied
within the training partition only, per the sentence, but whether it is a
time-respecting/blocked K-fold or an ordinary random K-fold on
sequential/autocorrelated time-series data is NEVER STATED). If it is an
ordinary random K-fold, this is the same [L3.1] temporal-leakage pattern
this project's own literature_notes.md already flags for a different
paper in this survey (energyeng2025cnnlstmcascade, "random k-fold on time
series data"). Coded split_type as not_stated for the primary 70/15/15
split since ordering is never stated for it either; the K-fold detail is
recorded here as an unresolved leakage-risk flag, not folded into the
coded value, and listed again in the final uncertain-fields report.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches for seed/
repeated runs/standard deviation/"+/-"/confidence interval/error bars/
multiple runs describing model-accuracy spread. K-fold "averaging results
across multiple partitions" (p.9) reports only that results are averaged,
never the spread across those partitions.

**code_available = not_stated** | "Data Availability Statement: Data will
be made available on request." (p.31) - data only, not code; no other
mention of a code repository.

**key_claim**: "For solar power forecasting at Benban (Egypt) and Sakaka
(Saudi Arabia) solar parks (70/15/15 train/validation/test split, ordering
not stated, Bayesian/TPE hyperparameter tuning against the validation
set), the hybrid CNN-LSTM model outperforms RF, SVM, GBM, LSTM, CNN, and
ARIMA, reaching MAPE=2.04%, RMSE=184, MAE=252, R2=0.99 on Benban and
MAPE=2.00%, RMSE=190, MAE=255, R2=0.98 on Sakaka; forecast horizon is
never explicitly stated anywhere in the extracted text." | Abstract (p.1).
NOTE: RMSE/MAE units (184, 190, 252, 255) are not given a physical unit
(kW/MW/W) in the abstract or in the immediate results discussion found in
the extracted text - given these are utility-scale parks (Benban alone is
one of the world's largest, multi-hundred-MW), these are plausibly in kW,
but the text itself does not state the unit at the point the numbers are
given.

## What this paper does WELL

- Reports carbon-footprint/compute cost (CO2 emissions per computational
  hour) alongside accuracy for every model - directly relevant to this
  project's own RQ4 (cost-effectiveness), and a genuinely unusual axis to
  report in this literature.
- Uses two independent utility-scale sites (Benban, Sakaka) and reports
  results for both, rather than a single site - closer to a real
  generalization test than most papers in this batch.
- Explicit hyperparameter-tuning protocol described (Bayesian/TPE
  optimization against a held-out validation set, not the test set).
- Compares against ARIMA, a genuine time-series statistical baseline, in
  addition to ML models - closer to a real reference than the ML-vs-ML-only
  comparisons common elsewhere in this batch, even though ARIMA is not
  persistence/climatology/convex.

## MAJOR FLAG: MAE > RMSE in 6 of 7 models per site table, both sites [confirmed 2026-08-07, re-extracted directly from clean, unscrambled table text]

Table 5 (Benban, p.15) and Table 6 (Sakaka, p.19), full rows, columns are
MAPE / RMSE / MAE / R2 as printed in the header ("Model Notation AI Model
MAPE RMSE MAE R*2"):

Table 5 (Benban): RF 4.47/332.8/516/0.95; SVM 8.42/416.4/789.6/0.88; GBM
6.88/724/502.4/0.87; LSTM 3.44/840/936/0.84; CNNs 4.26/336/528/0.85;
CNN-LSTM 2.04/184/252/0.99; ARIMA 11.77/300/464/0.91.

Table 6 (Sakaka): RF 4.5/335/520/0.94; SVM 8.4/420/790/0.87; GBM
6.9/730/505/0.86; LSTM 3.4/850/940/0.83; CNNs 4.3/340/530/0.84; CNN-LSTM
2/190/255/0.98; ARIMA 11.8/310/470/0.9.

Under the header's own column labels, MAE > RMSE for RF, SVM, LSTM, CNNs,
and CNN-LSTM in BOTH tables - only GBM has RMSE > MAE, in both tables.
That is 6 of 7 models violating RMSE >= MAE identically across two
independently-labeled site tables. HYPOTHESIS, not a correction: assuming
the RMSE and MAE column headers are swapped (i.e. the column printed
under "RMSE" is actually MAE and vice versa) makes 6 of the 7 rows in
each table internally consistent, and leaves GBM as the sole outlier
instead of the other six - the more parsimonious reading of the two, but
the paper's own header literally reads "MAPE RMSE MAE R*2" and this audit
does not resolve which is correct. Recorded as a hypothesis for your
judgment, not applied as a recode to key_claim or notes.

Two further findings from the same re-extraction, recorded rather than
resolved:
- Table 5 and Table 6 report near-identical values across two different
  plants in two different countries (Benban, Egypt vs. Sakaka, Saudi
  Arabia) - e.g. RF: 332.8/516 vs. 335/520; CNN-LSTM: 184/252 vs.
  190/255; every model's numbers differ by only 1-2%, site to site. This
  is unusual for independently-fit models on independent sites, but not
  necessarily wrong; noted, not asserted as an error.
- The Sakaka discussion paragraph (p.19, immediately following Table 6)
  reads "achieving the lowest MAPE (2.04%) and RMSE (184) among all
  models" - these are Table 5's Benban CNN-LSTM values, not Table 6's own
  Sakaka CNN-LSTM row (MAPE 2, RMSE 190). This looks like a copy-paste
  carryover from the Benban section's write-up into the Sakaka section.

## Other observations / consistency checks

- No shared authors/institution with any other paper coded in this batch
  (University of Melbourne / King Saud University).
