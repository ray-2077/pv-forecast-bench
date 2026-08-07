# Audit: nadour2026cnnbilstm

Source: `data/papers/1-s2.0-S2352484726000880-main.pdf` (20 pages, 103,183 chars)

Nadour, M., Rabehi, A., Hadroug, N., Guermoui, M., Tibermacine, I.E.,
Alanazi, A.K., Habib, M., Rabehi, A. (2026). "Deep hybrid CNN-biLSTM model
for accurate solar photovoltaic power forecasting: A comparative study with
classical and neural models." *Energy Reports* 15:109119.

This is the most methodologically rigorous paper coded in this batch so
far - flagged upfront because most of what follows is confirmation rather
than the usual gap-finding.

DOI: https://doi.org/10.1016/j.egyr.2026.109119

## Coded fields

**year**: 2026 | "Energy Reports 15 (2026) 109119"

**venue**: Energy Reports 15:109119

**dataset**: Ghardaia PV plant, Algeria (multi-technology: thin-film a-Si,
CdTe, polycrystalline Si, monocrystalline Si), 2018-2019, hourly, h=1
forecast horizon, 16,843 of 17,520 possible hourly samples retained (96.1%)
| "For this study, operational data from 2018 to 2019 were employed to
validate the forecasting models. The dataset comprises hourly recordings
of meteorological parameters and PV power output from January 2018 to
December 2019. This two-year period provided 17,520 potential hourly
measurements... resulting in a final cleaned dataset of 16,843 complete,
high-quality samples, representing a 96.1% data retention." (p.5); "the
prediction horizon h was set to 1" (p.6)

**night_hours_excluded = not_stated** | No statement anywhere in the
paper's own preprocessing description addresses night-hour handling one
way or the other. The only preprocessing exclusion criterion given is
"Samples with missing or invalid data were excluded... records with
missing values or physically implausible readings (e.g., negative
irradiance) were removed." (p.5) - night-time zero power/irradiance is
neither missing nor implausible, so this criterion does not obviously
remove it, but the paper never says so explicitly either way.
FLAG (not part of the coded value - circumstantial, not a quote): 17,520 =
365 x 2 x 24 exactly, i.e. the "potential measurements" baseline already
counts every hour of the two years including nights, and 96.1% retention
is a high fraction consistent with (but not proof of) night hours still
being present in the modelled set. Recorded as an observation only, not as
grounds for coding "no" - the ABSOLUTE RULE requires an explicit statement
either way, which is absent here.

**baseline_used = other_ML** | "The proposed CNN-biLSTM model is evaluated
against four benchmark models, including Multilayer Perceptron (MLP),
Support Vector Regression (SVR), Random Forest (RF), and a unidirectional
CNN-LSTM" (Abstract, p.1); "the MLP served as a baseline deep learning
model due to its simplicity and widespread use in regression tasks" (p.7).
No persistence/climatology/convex reference used for THIS paper's own
evaluation (see skill_score_reported exhaustive check below, same search
covers baseline terms - the only "persistence" hits are in the Related
Work comparison table describing OTHER cited papers' methods, e.g. "CNN
achieved 12% skill score over persistence" describing Zhang et al. 2018,
not this paper's own comparison).

**skill_score_reported = no** | EXHAUSTIVE CHECK: this paper's own
Results/metrics (Abstract p.1, Section 3 pp.9-15) report R2, RMSE, MAE,
MAPE, sMAPE, and normalised RMSE (nRMSErange) - never a skill score against
a reference forecast. "Skill score" appears only inside the Related Work
literature-comparison table (p.4, describing Zhang et al. 2018 and El
hendouzi et al.'s prior results, not this paper's own).

**weather_source = measured** | "hourly recordings of meteorological
parameters and PV power output" (p.5) from the Ghardaia plant's own
monitoring; features include "solar irradiance (W/m2), ambient
temperature (C), day, hour of year, and PV power output (kW)" (p.5,
reconstructed from a garbled two-column table-of-variables extraction) -
all measured on-site historical values, no NWP/forecast-weather language
anywhere in the text.

**split_type = chronological** | "To maintain temporal order and reflect
operational forecasting scenarios, the cleaned dataset underwent a strict
chronological split: the first 70% for training and the remaining 30% for
testing." (p.5); reiterated at p.9: "All models were evaluated using a
strict chronological 70/30 train-test..." This is the cleanest, most
explicit split statement of any paper coded in this survey so far.

**n_seeds = not_stated** | no mention of multiple training seeds or
repeated end-to-end training runs anywhere in the text (see
variance_reported below, same search).

**variance_reported = no** [see STRENGTH flag below] | EXHAUSTIVE CHECK:
searched seed/repeated runs/standard deviation/"+/-"/confidence interval/
std dev/error bars/multiple runs. The paper DOES report "Bootstrap 95%
confidence intervals for key performance metrics" (p.16) and presents
"Table 4... 95% confidence intervals for RMSE and MAE" (p.17) and "Fig. 14
... 95% bootstrap confidence intervals for RMSE and MAE" (p.19-20).
However, per the coding rule, variance_reported means RUN-TO-RUN spread
across independent seeds/repeats of TRAINING - a bootstrap CI is
resampling uncertainty over the FIXED prediction residuals of a single
trained model, not variance across independently retrained models. Coded
"no" per the strict definition, but this is exactly the kind of thing the
coding rules ask to be recorded as a STRENGTH regardless: this paper
quantifies prediction uncertainty more rigorously than any other paper in
this batch, just not via the specific seed-variance axis the schema
tracks.

**code_available = not_stated** | "To support reproducibility, the
processed dataset and preprocessing code used in this study are available
upon reasonable request from the corresponding author." (p.5) Explicitly
mentions code, but gated behind "upon reasonable request" - per the coding
rule's own worked example ("Contact the author" is not_stated), this is
not_stated rather than "yes", even though it is a much more specific and
good-faith statement than most "not_stated" papers in this survey (most
say nothing about code at all).

**key_claim**: "For one-hour-ahead (h=1) forecasting at the multi-
technology Ghardaia PV plant, Algeria (2018-2019 hourly data, strict
chronological 70/30 split, scaler fit on training data only), the proposed
CNN-biLSTM model achieves R2=0.99848, RMSE=0.5939 W, MAE=0.398 W, and
nRMSErange=1.18%, outperforming MLP, SVR, RF, and unidirectional CNN-LSTM
benchmarks." | Abstract (p.1). NOTE: RMSE/MAE units are given as "W" (watts)
in the abstract - given the plant is described elsewhere as part of a
network contributing "about 400 MW to the national electricity grid" (p.4),
these error magnitudes are almost certainly on a normalized/per-unit or
per-panel basis rather than raw plant-level watts, but the paper's stated
units are quoted as printed; the text does not reconcile the W scale
against the plant's nameplate capacity anywhere in the extracted pages.

## What this paper does WELL (this is the strongest showing in the batch)

- Explicit, strict, repeatedly-stated chronological 70/30 split with an
  explicit stated rationale ("to maintain temporal order and reflect
  operational forecasting scenarios").
- Scaler/normalization statistics fit on TRAINING DATA ONLY, with an
  EXPLICIT anti-leakage rationale stated in the text: "Z-score
  normalization was applied after the train-test split to prevent data
  leakage. Normalization parameters (mean and standard deviation) were
  calculated exclusively from the training set... This approach ensures
  that no information from the test set influences the training process,
  providing an unbiased evaluation of model performance." (p.5) - this is
  this project's own CLAUDE.md rule 3, stated near-verbatim as a deliberate
  methodological choice, not something coded silently.
- Explicit, single, clearly stated forecast horizon (h=1).
- Reports bootstrap 95% confidence intervals on RMSE and MAE - real
  uncertainty quantification, rare in this literature even if not the
  seed-variance axis this survey tracks.
- Explicitly justifies NOT using a separate validation split ("No separate
  validation set was created for hyperparameter tuning, as we employed
  fixed architectural parameters and regularization strategies to prevent
  overfitting, preserving the maximum amount of temporal data for
  training.") - a stated, defensible design decision rather than a silent
  gap.
- Names an explicit code/data reproducibility path (even if gated behind
  "reasonable request" rather than a public link).

## Other observations / consistency checks

- No naive reference forecast (persistence/climatology/convex) anywhere in
  this paper's own evaluation - despite its strong protocol hygiene
  otherwise, it shares the "no naive baseline at all" gap with every other
  paper coded in this batch so far. Worth stating explicitly in the survey
  writeup: split rigor and baseline choice are independent axes, and this
  paper is strong on the first and weak on the second.
- RMSE > MAE as required (0.5939 > 0.398 W) - internally consistent, no
  contradiction found.
- Night-hour handling is the one genuine gap in an otherwise unusually
  careful methods section - see the not_stated coding and flag above.
- No shared authors/institution with any paper coded earlier in this batch
  (Algeria/Sweden/Italy/Saudi Arabia consortium, distinct from the Hungary,
  China, and Australia-sited papers coded so far).
