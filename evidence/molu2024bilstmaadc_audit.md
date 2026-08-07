# Audit: molu2024bilstmaadc

Source: `data/papers/1-s2.0-S2590123024007163-main.pdf` (16 pages, 77,593 chars)

Molu, R.J.J., Tripathi, B., Mbasso, W.F., Naoussi, S.R.D., Bajaj, M., Wira,
P., Blazek, V., Prokop, L., Misak, S. (2024). "Advancing short-term solar
irradiance forecasting accuracy through a hybrid deep learning approach
with Bayesian optimization." *Results in Engineering* 23:102461.

SCOPE NOTE: this paper forecasts solar IRRADIANCE (W/m2), not PV power
directly - included in this survey's dataset (data/papers/) presumably for
its evaluation-practice relevance (irradiance forecasting shares the same
night/baseline/split issues as PV power forecasting, and Mayer 2022 in this
same batch discusses irradiance-to-power conversion directly). Coded using
the same schema; "PV power" in the schema's intent is read as "solar
irradiance" throughout for this entry.

## Coded fields

**year**: 2024 | "Results in Engineering 23 (2024) 102461"

**venue**: Results in Engineering 23:102461

**dataset**: Solar irradiance probe (WS501-UMB weather sensor), Douala
Institute University of Technology, Cameroon | "The meteorological station
situated at the Douala Institute University of Technology (IUT) serves as
a vital experimental prototype for our research endeavours... The WS501-UMB
intelligent weather sensor... is a pivotal instrument at this facility."
(p.4-5)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: no night/
daytime/daylight/sunrise/sunset/zero-output language anywhere describing
this paper's own preprocessing. "Zenith angle" appears once (p.19,
supplementary-parameters table) as a derived INPUT FEATURE name, not as an
exclusion criterion.

**baseline_used = own_components** | "multiple deep learning models,
including Long Short-Term Memory, Bidirectional Long Short-Term Memory,
Artificial Neural Networks, Bidirectional Long Short-Term Memory with
Additive Attention Mechanism, and Bidirectional Long Short-Term Memory
with Additive Attention Mechanism and Dilated Convolutional layers, are
trained and evaluated" (Abstract, p.1) - ANN/LSTM/BiLSTM/BiLSTM-AA are all
progressive ablation stages toward the proposed BiLSTM-AADC. No
persistence/climatology/convex reference anywhere (exhaustive check: zero
matches for persistence/naive/reference forecast/climatology/reference
model in 77,593 characters; "benchmark models" always refers to
ANN/LSTM/BiLSTM/BiLSTM-AA).

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Metrics used throughout are RMSE, MAE, sMAPE, NRMSE - all
against ground truth, never a reference-forecast-relative skill score.

**weather_source = measured** | ground-station sensor data (WS501-UMB), no
NWP/forecast-weather language anywhere in the text.

**split_type = not_stated** | "The authors partitioned the dataset into
training and testing sets with an 80:20 ratio to ensure the models'
generalization and robustness." (p.18) - a ratio only; no statement of
chronological vs. random ordering anywhere in the text (exhaustive check:
searched chronological/random/shuffle/k-fold/cross-validation/hold-out/
walk-forward - only this one 80:20 ratio statement found, no ordering
qualifier).

**n_seeds = 20** | "The authors conducted 20 runs for each model,
implementing an early stopping mechanism limited to a maximum of 1000
iterations, and recorded the mean results." (p.18)

**variance_reported = no** [see flag] | Despite running each model 20
times, the paper states only that it "recorded the mean results" - no
standard deviation, confidence interval, min/max, or any other dispersion
statistic across those 20 runs is reported anywhere in Tables 3-4 or the
surrounding text (exhaustive check: no "+/-", "std", "confidence interval"
or similar found attached to any reported metric). This is a distinct
case worth flagging as a STRENGTH-ADJACENT gap: the paper does the right
thing procedurally (repeats each model 20 times rather than reporting one
lucky run) but then discards the spread and reports only a point estimate
- closer to good practice than the majority of this survey's papers
(single run, n_seeds not even stated), but still short of variance_reported.

**code_available = not_stated** | "Data availability: Data will be made
available on request." (p.30) - data only, not code; no other mention of
a code repository anywhere.

**key_claim**: "For next-day solar irradiance forecasting at a Douala,
Cameroon weather station (80:20 split, ordering not stated, mean of 20
runs per model), the proposed BiLSTM-AADC model achieves RMSE=17.0718,
MAE=57.5241, sMAPE=0.6564, NRMSE=0.2250, beating ANN/LSTM/BiLSTM/BiLSTM-AA
(Table 3, p.19); accuracy degrades gracefully as the horizon extends to 5,
7, and 15 days (Table 4, p.20), e.g. RMSE rises from 17.07 (next-day) to
22.94 (5-day) to 26.94 (7-day) to 40.64 (15-day) for the same proposed
model."

## MAJOR FLAG: abstract mixes metrics from two different forecast horizons

Abstract (p.1): "The results show a Symmetric Mean Absolute Percentage
Error of 0.6564, a Normalized Root Mean Square Error of 0.2250, and a Root
Mean Square Error of 22.9445, surpassing previous studies in the
literature."

Cross-checked against the paper's own tables:
- Table 3 (p.19, "next-day" forecasting), BiLSTM-AADC row: RMSE=17.0718,
  sMAPE=0.6564, NRMSE=0.2250.
- Table 4 (p.20, "5 Day" forecasting), BiLSTM-AADC row: RMSE=22.9445,
  sMAPE=0.8455, NRMSE=0.6506.

The abstract's sMAPE (0.6564) and NRMSE (0.2250) match the NEXT-DAY table
exactly, but its RMSE (22.9445) matches the 5-DAY table instead of the
next-day table's own RMSE of 17.0718 - the abstract silently combines three
metrics from two different forecast horizons into one sentence with no
indication that they are not all from the same evaluation. This is a
verbatim, cross-table-checkable inconsistency, not an inference - both
source numbers are quoted above. Given the abstract's RMSE is the LARGER
(worse) of the two candidates, this does not appear to be a favorable
cherry-pick, more likely a copy-paste/transcription error while assembling
the abstract from separate result tables - but the effect either way is
that the headline number a reader takes from the abstract alone does not
correspond to a single, coherent evaluation.

## MAJOR FLAG: MAE > RMSE for the proposed model at every forecast horizon [confirmed 2026-08-07, re-extracted directly from clean table text, not the earlier scrambled reading]

RMSE >= MAE is a mathematical identity for any sample (RMSE is an L2 norm,
MAE an L1 norm, of the same per-point errors), so MAE > RMSE is never
possible for two metrics computed on the same data. Re-extracting Tables
3-6 (next-day, 5-day, 7-day, 15-day) directly, cleanly, column-by-column
(ANN, LSTM, BiLSTM, BiLSTM-AA, BiLSTM-AADC in that order):

- Next-day: RMSE row 117.4586 / 204.5819 / 145.1278 / 119.0530 / 17.0718;
  MAE row 58.3540 / 158.7977 / 91.7301 / 64.1064 / 57.5241.
- 5-day: RMSE 140.0578 / 233.3878 / 166.7302 / 123.1981 / 22.9445; MAE
  70.3145 / 181.3889 / 106.0118 / 67.4150 / 65.5817.
- 7-day: RMSE 154.0636 / 252.5137 / 178.9897 / 146.6333 / 26.9407; MAE
  77.3459 / 196.9270 / 115.3587 / 83.3946 / 73.1371.
- 15-day: RMSE 195.8436 / 306.4404 / 213.6873 / 170.0822 / 40.6394; MAE
  98.3211 / 238.9826 / 138.1536 / 96.7307 / 110.3254.

At every one of the four horizons, MAE > RMSE holds for the BiLSTM-AADC
(proposed) column only - ANN, LSTM, BiLSTM, and BiLSTM-AA all show the
normal RMSE >= MAE relationship in every row. The proposed-model numbers
are stated identically in the running prose as well as the tables (e.g.
p.10: "the lowest RMSE of 17.0718, MAE of 57.5241"; p.14: "BiLSTM-AADC
model achieves an RMSE of 22.9445, MAE of 65.5817"), so this is not a
one-off scrambled-table extraction artifact - the same pairing recurs
verbatim in prose, independent of table layout, at all four horizons.
This is recorded as a documented inconsistency in the paper's own
reported numbers, isolated specifically to its own best-performing model.
No cause is speculated here (mismatched metric definition for that one
model, a transcription error carried consistently into the prose, or
something else) - only the pattern itself is recorded.

## Other observations / consistency checks

- Additional unresolved scale question (recorded as an ambiguity, not
  asserted as an error): the paper's own sMAPE formula (Eq. 18, p.17)
  includes an explicit "100/n" prefactor, i.e. is defined to already
  produce a percentage. The reported sMAPE values (0.66-4.86 across Tables
  3-4) and NRMSE values (0.23-14.07) are of a similar small magnitude to
  each other despite one metric nominally being a percentage (0-200 scale)
  and the other a dimensionless ratio - plausible either as very good
  percentage errors or as values the reader must additionally multiply,
  and the text does not clarify which. Not resolved here.
- Shared-institution check: Mohit Bajaj (Graphic Era, India) is a
  co-author here; no overlap with any other paper's author list coded so
  far in this batch.
