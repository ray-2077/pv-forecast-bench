# Audit: zhou2024cnnlstmattnbayes

Source file: `data/papers/1-s2.0-S2096511724000860-main.pdf` (15 pages, 66,578 extracted chars)

Zhou, N., Shang, B., Xu, M., Peng, L., Feng, G. (2024). "Enhancing
photovoltaic power prediction using a CNN-LSTM-attention hybrid model with
Bayesian hyperparameter optimization." *Global Energy Interconnection*
7(5):667-681. https://doi.org/10.1016/j.gloei.2024.10.005

**SAME SOURCE DATASET AS THIS PROJECT**: this paper uses DKASC (Desert
Knowledge Australia Solar Centre) - "A subset of PV data from the DKASC
dataset [36], encompassing 2020-2021 with a 5 min sampling frequency, was
selected as the final original dataset Q for analysis." (p.8) DKASC is the
same open dataset this project uses (different years: this paper uses
2020-2021, this project uses 2011-2015; unclear if the same array/site
within DKASC - not stated in the extracted text which specific site number).

## Coded fields

**year**: 2024

**venue**: Global Energy Interconnection 7(5):667-681

**dataset**: DKASC subset, 2020-2021, 5-min sampling, 12 feature variables
(AE_Power, Current, Power, Wind_Speed, Temp, Humidity, GHI, DHI, Wind_dir,
Rainfall, RGT, RDT) | "A subset of PV data from the DKASC dataset [36],
encompassing 2020-2021 with a 5 min sampling frequency, was selected as the
final original dataset Q for analysis." (p.8); "The dataset includes twelve
feature variables: received active energy (AE_Power), current phase average
(Current), active power (Power), wind speed (Wind_Speed), temperature
(Temp), weather relative humidity (Humidity), global horizontal irradiance
(GHI), diffuse horizontal irradiance (DHI), wind direction (Wind_dir),
rainfall (Rainfall), global tilted radiation (RGT), and diffuse tilted
radiation (RDT)." (p.8) NOTE: Current_Phase_Average and
Active_Energy_Delivered_Received are two of the three columns this
project's own src/data/loader.py explicitly drops as target-derived leakage
traps (CLAUDE.md, "DROPPED COLUMNS"). This paper uses both as model input
features without discussion.

**night_hours_excluded = yes** | "a PV power system requires ample
sunshine and sufficient daylight. Therefore, data pertaining to night-time
power generation were excluded. Missing daytime values were addressed
using the k-nearest neighbors mean method [37]." (p.8)

**baseline_used = other_ML** [AMBIGUOUS, see flag] | "The experimental
results indicated that within acceptable model training times, the
CNN-LSTM-attention model outperformed the LSTM, GRU, CNN-LSTM, CNN-LSTM
with autoencoders, and parallel CNN-LSTM attention models." (Abstract, p.1)
No persistence, climatology, or convex-combination reference appears
anywhere in the text (exhaustive check below).

FLAG: comparators are LSTM, GRU (standard external architectures - clearly
"other_ML") plus CNN-LSTM, CNN-LSTM+autoencoder, and parallel
CNN-LSTM-attention (architecturally close variants/ablation steps toward
the paper's own final model - arguably "own_components"). The schema
forces one label for a mixed comparison set. Coded other_ML because LSTM
and GRU (well-established independent architectures, not derived from this
paper's design) are named first and are the most standard comparators; but
own_components is equally defensible for the other three. No naive
statistical baseline (persistence/climatology/convex) is used at all - that
absence is the more important finding regardless of which ML label is
picked. See exhaustive check.

EXHAUSTIVE CHECK for baseline_used=none possibility: searched full text for
persistence/naive/benchmark/reference forecast/climatology/reference model
- zero matches anywhere in 66,578 characters. Confirmed: no naive/
statistical reference forecast of any kind is used; every comparison is
against another deep-learning model.

**skill_score_reported = no** | EXHAUSTIVE CHECK: searched for "skill
score", "forecast skill", "relative to persistence", "SS", "improvement
over persistence" - zero matches. Reported metrics are the coefficient of
determination (R2, called "goodness of fit"), MAPE, MAE, RMSE, and relative
training time (Section 4.1-4.3, Tables 3-6). None is a skill score against
a reference forecast; R2 and percentage-improvement-over-another-ML-model
are both explicitly excluded from "yes" by the coding rules.

**weather_source = measured** | features are historical DKASC site-logged
GHI/DHI/temperature/humidity/wind (see dataset quote above), not NWP
forecasts - no NWP/forecast-weather language appears anywhere (exhaustive
check: searched measured/NWP/numerical weather/exogenous/meteorological/
ground station/pyranometer/satellite/reanalysis/ERA5/MERRA - only
"meteorological conditions" generic mentions, describing the DKASC
station's own logged data, no forecast-weather source named).

FLAG (important, distinct from weather_source itself): the model input
construction explicitly includes weather features FROM THE SAME TIMESTEP
AS THE PREDICTED POWER, not just lagged/historical values: "The input
vector for the model comprises historical feature data from d time points
of the previous day and partial feature data from the prediction time
step to numerically predict the PV power output at the prediction time
step." (p.4) and "the feature data at the current prediction time can be
represented as a 1xk vector, denoted as I'_t = [I'_t,1,...,I'_t,k], where k
is the number of features at the current prediction time. Merging this
with the historical features yields a complete feature vector..." (p.4)
This is functionally the same construction this project labels the
"oracle" regime and treats as an explicit perfect-forecast upper bound
(CLAUDE.md rule 5) - measured weather AT THE TARGET TIME used as a model
input. This paper does not name it as an upper bound, does not discuss
that such same-timestep weather would not be available operationally at
forecast issue time, and does not report a separate "lagged-only" result
for comparison. This is the single most consequential undisclosed
methodological choice found in this paper.

**split_type = chronological** [scope of test set left ambiguous, see key
claim] | "data from July 1, 2020, to December 31, 2020, were used as the
training set for model training, whereas data from July 1, 2021, to July
20, 2021, served as the validation set for model verification." (p.13) -
train (H2 2020) strictly precedes validation (Jul 2021), and "the ratio of
the training to validation sets was 9:1" (184 days : 20 days ~ 9.2:1,
consistent). No explicit date range is ever given for the "test set"
itself, only that it is a third, later, non-overlapping partition: "the
original dataset Q must be divided into three non-overlapping parts:
training, validation, and test sets... The test set was used for the model
prediction." (p.13)

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: searched for seed/repeated/
standard deviation/"+/-"/confidence interval/std dev/error bars/multiple
runs. The only "standard deviation" match (p.9, Table 2) is a descriptive
statistic of the raw feature data (dataset characterization), not run-to-
run variance of model performance. No mention of repeated training runs,
multiple seeds, or any dispersion statistic on model accuracy across runs.

**code_available = not_stated** | EXHAUSTIVE CHECK: no GitHub/code
repository/data availability/reproducibility/supplementary-material
statement anywhere in the text (search returned zero matches for any of
these terms).

**key_claim**: "For the DKASC PV dataset (2020-2021, 5-min data, trained on
Jul-Dec 2020 / validated on 3 weeks of Jul 2021), a CNN-LSTM-attention
model with Bayesian-optimized hyperparameters outperforms LSTM, GRU,
CNN-LSTM, CNN-LSTM+autoencoder, and parallel CNN-LSTM-attention on R2, MAE
and RMSE, evaluated separately on 'smooth' and 'fluctuating' data periods
(no forecast horizon is ever stated anywhere in the text)." AMBIGUITY, per
instructions - not resolved here: the paper never states (a) the test
set's date range or duration, (b) whether "smooth" and "fluctuating" are
the full test set split into two exhaustive segments or two short,
separately hand-picked example windows, or (c) the forecast horizon (how
far ahead "the prediction time step" is from the historical input). The
result-plot x-axes read "Number of sample points, 0 to 300" (Figs 6-8),
which at the paper's stated 5-min sampling is only ~25 hours per period -
consistent with two short illustrative windows rather than a full test-set
aggregate, but this is a plausible reading of a plot axis, not a stated
fact, so it is not coded as established.

## What this paper does WELL

- Explicit, stated night-hour exclusion rule.
- Explicit chronological train/validation date ranges.
- Reports training time as an explicit efficiency axis alongside accuracy
  (Section 4.1) - relevant to this project's own RQ4.
- Ablates its own architecture against multiple simpler variants (LSTM,
  GRU, CNN-LSTM, CNN-LSTM+AE) rather than only the full proposed model -
  closer to a real ablation than most of this literature.

## Other observations / consistency checks

- MAPE FLAG: reported MAPE values are extremely high - 69.356% (smooth
  period) and 57.723% (fluctuating period), Table 4 - with NO statement
  anywhere of how near-zero power denominators (dawn/dusk, or any residual
  night-adjacent low-output samples) are handled in the MAPE formula
  (Eq. 25, p.11: plain mean(|y-yhat|/y), no epsilon or floor term shown).
  A well-behaved model reporting >50% MAPE alongside R2 > 0.99 and MAE
  under 0.13 kW is the classic signature of a few near-zero actual-power
  samples dominating the ratio - exactly the pattern this project's own
  daylight-threshold discussion (Finding 4) warns about, though this
  paper's threshold/inclusion criterion for the MAPE calculation specifically
  is not stated separately from the general night-exclusion rule.
- Internal number check: MAE < RMSE in both reported periods (0.058 <
  0.076 smooth; 0.120 < 0.221 fluctuating), which is the required
  MAE <= RMSE relationship - no contradiction found. Cannot check the
  RMSE^2 ~ mean^2 + SD^2 identity: no per-sample residual mean/SD is
  reported separately from RMSE on a stated common sample.
- Counterintuitive result flagged BY THE PAPER ITSELF and left partly
  unresolved even in their own discussion: MAE is better (lower) in the
  smooth period but MAPE is worse (higher) in the smooth period than the
  fluctuating period - the paper offers a post-hoc explanation (relative
  vs absolute error sensitivity) but this is exactly the kind of scale-
  dependent metric disagreement this survey is built to surface.
- Shared dataset with another paper in this survey: none yet identified
  among the 3 previously-coded papers; DKASC usage should be checked
  against remaining papers in this batch as they are coded.
- No shared authors/institution with mayer2022physmlhybrid (Budapest) or
  the three papers already coded before this survey pass.
