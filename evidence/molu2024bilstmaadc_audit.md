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
- RMSE > MAE holds in every table row checked (e.g. next-day BiLSTM-AADC:
  17.07 > not applicable comparison since MAE=57.52 > RMSE=17.07 here -
  see NEXT flag) -
  ACTUAL CHECK: Table 3 next-day BiLSTM-AADC row shows MAE=57.5241 >
  RMSE=17.0718, i.e. MAE EXCEEDS RMSE, which is mathematically impossible
  for the standard definitions of both (RMSE >= MAE always, since RMSE is
  an L2 norm and MAE an L1 norm of the same errors). This is a second,
  independent internal-consistency problem, verified directly from Table 3
  as extracted (p.10 in the raw extraction): "RMSE 117.4586 204.5819
  145.1278 119.0530 17.0718 / MAE 58.3540 158.7977 91.7301 64.1064
  57.5241" - for the proposed model (last column), RMSE=17.0718 while
  MAE=57.5241. The same MAE>RMSE pattern holds for the ANN column too
  (RMSE=117.46 > MAE=58.35 - that one is fine) but BREAKS specifically for
  the two best-performing columns (BiLSTM-AA: RMSE=119.05 > MAE=64.11 is
  fine; BiLSTM-AADC: RMSE=17.07 < MAE=57.52 is NOT fine). This suggests a
  possible column-alignment or transcription artifact specific to the
  proposed model's RMSE cell in Table 3 - flagged for the user's own
  judgment rather than resolved, since the raw PDF table extraction could
  itself be misaligned (text was extracted from a dense multi-column PDF
  table); this should ideally be checked against the original PDF table
  layout, not just this extraction.
- Shared-institution check: Mohit Bajaj (Graphic Era, India) is a
  co-author here; no overlap with any other paper's author list coded so
  far in this batch.
