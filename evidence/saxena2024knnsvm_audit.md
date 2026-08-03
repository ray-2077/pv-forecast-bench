# Audit: saxena2024knnsvm

Source: `data/papers/1-s2.0-S2667010024000040-main.pdf` (10 pages, 32,727 chars)

Saxena, N., Kumar, R., Rao, Y.K.S.S., Mondloe, D.S., Dhapekar, N.K.,
Sharma, A., Yadav, A.S. (2024). "Hybrid KNN-SVM machine learning approach
for solar power forecasting." *Environmental Challenges* 14:100838.

## Coded fields

**year**: 2024 | "Environmental Challenges 14 (2024) 100838"

**venue**: Environmental Challenges 14:100838

**dataset**: "Jodhpur real-time series dataset obtained from the data
centers of weather stations using Meteonorm... Hourly Average Temperature
(HAT), Hourly Total Sunlight Duration (HTSD), Hourly Total Global Solar
Radiation (HTGSR), and Hourly Total Photovoltaic Energy Generation
(HTPEG)." (Abstract, p.1)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches
anywhere for night/daytime/daylight/sunrise/sunset/zenith/clear-sky/
diurnal in the full 32,727-character text.

**baseline_used = other_ML** | "The conventional LSTM technique is also
implemented to compare the performance of the proposed hybrid technique."
(Abstract, p.1) - a single external comparator (LSTM), not persistence/
climatology/convex and not an ablation of the proposed KNN-SVM's own
sub-components (KNN and SVM are fused into one hybrid, not tested
separately against each other in the reported results as far as the
extracted text shows). EXHAUSTIVE CHECK: zero matches for persistence/
naive/climatology/reference model/reference forecast anywhere.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence".

**weather_source = not_stated** [AMBIGUOUS, see flag] | "tested on the
Jodhpur real-time series dataset obtained from the data centers of weather
stations using Meteonorm" (Abstract, p.1); "The datasets for the preferred
location have been collated from Meteonorm." (p.4)

FLAG: Meteonorm is a commercial software product best known for generating
SYNTHETIC typical-meteorological-year (TMY) data (statistically
representative, not observations of any specific real date), though it can
also interface with real station archives depending on product/mode. The
paper's own phrase "real-time series dataset obtained from... weather
stations USING Meteonorm" is internally ambiguous about which of these
Meteonorm was used for here, and never clarifies. Coding "measured" would
assert real ground-truth observations the paper does not clearly claim;
coding "NWP_forecast" would be wrong in a different way (Meteonorm is not
a numerical weather prediction/forecast product either). Coded not_stated
because neither allowed value is a confident match to what the paper
actually describes - flagged again in the final uncertain-fields list.

**split_type = not_stated** | "The collated data has been segmented into
training data, validation data, and testing data." (Abstract, p.1) - a
three-way split is named but no ratio and no statement of ordering
(chronological/random) is given anywhere in the text (exhaustive check:
zero matches for 70%/80%/20%/30%/chronological/random/shuffle/k-fold
describing this paper's own split).

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches for seed/
repeated runs/standard deviation/"+/-"/confidence interval/error bars/
multiple runs describing model accuracy. The only "RMSE" mentions in the
whole paper (p.9, "the mean squared error (RMSE) of the training has
reached to 0.001"; Figs. 18-19 "RMSE and loss during training progress
plot") describe TRAINING LOSS CONVERGENCE CURVES, not a final evaluation
metric with any reported spread.

**code_available = not_stated** | "Data availability: No data was used
for the research described in the article." (p.9) - see MAJOR FLAG below;
this statement does not mention code at all regardless of its own internal
contradiction.

**key_claim**: "On the Jodhpur (India) Meteonorm-derived dataset (split
into training/validation/testing, ratio and ordering not stated), the
proposed KNN-SVM hybrid outperforms a conventional LSTM by 7.1% accuracy,
6.62% sensitivity, and 7.25% specificity for HTPEG (PV energy generation);
overall the hybrid technique is reported as achieving '98% accuracy'
(Abstract)." AMBIGUITY, not resolved here: this is a continuous PV-power/
energy forecasting task, but the paper's ONLY reported evaluation metrics
are accuracy/sensitivity/specificity - metrics defined via true/false
positive/negative counts, i.e. CLASSIFICATION metrics. The paper never
states how a continuous power/energy forecast was converted into a
binary or categorical outcome for these metrics to be computed (no
threshold, no class definition, no confusion matrix shown in the extracted
text). No RMSE, MAE, MAPE, or R2 is reported for the final model at all.

## MAJOR FLAG: evaluation metrics do not match the stated task

The paper explicitly forecasts continuous quantities (HTGSR, HTPEG - hourly
solar radiation and PV energy generation, both continuous physical
measurements) but reports ONLY "accuracy, sensitivity, and specificity"
(Section 3.3, p.6: "The complete research work will be evaluating the
sensitivity, accuracy and specificity of the predictions and methods
implemented.") - these are defined in the paper's own Eq. 10-and-
surrounding text via True Positives/False Negatives (Sensitivity) and True
Negatives/False Positives (Specificity), which require a binary/categorical
outcome. No regression metric (RMSE, MAE, MAPE, R2) is reported for the
model's final test performance anywhere in the extracted text - the one
RMSE mention found is a training-loss convergence curve, not a test-set
accuracy figure. This is either (a) an undisclosed binarization step
(e.g. thresholding power output into "high/low" classes) that the paper
never describes, or (b) a fundamental mismatch between the stated
forecasting task and the metrics used to evaluate it. Not resolved here -
flagged as the single most consequential unstated methodological choice in
this paper, more severe than a missing baseline or missing split
description because it makes the headline "98% accuracy" claim
uninterpretable as a power-forecasting result without knowing what was
classified.

## Second flag: contradictory Data Availability statement

"Data availability: No data was used for the research described in the
article." (p.9) directly contradicts the paper's own Abstract and Section
2, which describe in detail "the Jodhpur real-time series dataset obtained
from the data centers of weather stations using Meteonorm" with four
named variables (HAT, HTSD, HTGSR, HTPEG). This reads as a boilerplate
Elsevier data-availability template left unedited/misfilled rather than a
genuine claim that no data existed - included here as a verbatim
observation, not an accusation, since the mechanism (template error vs.
genuine statement) cannot be determined from the text alone.

## What this paper does WELL

- States four named input variables explicitly (HAT, HTSD, HTGSR, HTPEG)
  with full names, not just abbreviations.
- Explicitly compares against an external, independently-established
  method (LSTM) rather than only ablating its own components.

## Other observations

- No shared authors/institution with any paper coded earlier in this
  batch (Indian engineering-college consortium, distinct from the
  Hungary/China/Australia/Algeria/Cameroon sites coded so far).
- Given the metrics mismatch above, no meaningful RMSE^2~mean^2+SD^2 or
  MAE<=RMSE consistency check is possible - no comparable regression
  residual statistics are reported anywhere in the paper.
