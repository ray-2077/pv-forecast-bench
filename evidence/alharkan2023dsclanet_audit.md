# Audit: alharkan2023dsclanet

Source: `data/papers/sensors-23-00945.pdf` (12 pages, 40,683 chars)

Alharkan, H., Habib, S., Islam, M. (2023). "Solar Power Prediction Using
Dual Stream CNN-LSTM Architecture." *Sensors* (MDPI) 23:945.

**SAME SITE AS THIS PROJECT, AND CONFIRMED OVERLAP WITH ANOTHER PAPER IN
THIS SURVEY BATCH**: "we utilized DKASC Alice Spring DKASC-AS datasets...
Three datasets are selected from DKASC-AS, namely Trina 10.5 kW mono-Si
Dual 2009 (Trina 1A), Trina 23.4 kW mono-Si Dual 2009 (Trina 1B), and
eco-Kinetics 26.5 kW mono-Si Dual 2010 (Eco 2)." (p.6) "Trina 23.4 kW"
matches EXACTLY the "DKASC-AS-1B... generated 23.4 kW" capacity reported
by hussain2022hybridgrucnn elsewhere in this same survey batch - strong
confirmation that both papers use the same physical DKASC array (1B),
independently of each other (different author groups, different
architectures, no shared authorship). This is now the fifth paper in this
survey batch to use DKASC-family data, and one of the few to use multiple
arrays the way this project itself does (3 arrays here vs. this project's
3 arrays, though the specific arrays differ - Trina 1A/1B and Eco 2 here
vs. this project's array11/array12/array17).

## Coded fields

**year**: 2023 | "Sensors 2023, 23, 945"

**venue**: Sensors (MDPI) 23:945

**dataset**: DKASC Alice Springs, 3 arrays: Trina 1A (10.5kW mono-Si,
installed 2009), Trina 1B (23.4kW mono-Si, installed 2009), Eco 2
(eco-Kinetics, 26.5kW mono-Si, installed 2010); historical weather + power
| quoted above (p.6)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches
anywhere for night/daytime/daylight/sunrise/sunset/zenith/clear-sky/
diurnal in 40,683 characters.

**baseline_used = own_components** | "we conducted experiments on several
models based on deep learning. These models include LSTM, CNN, GRU,
CNNGRU, CNNLSTM, and DCNN-BRLSTM." (p.7) - every comparator is a deep
learning architecture, ablation-adjacent to the proposed dual-stream
CNN-LSTM-attention model (DSCLANet).

FLAG (verified NOT this paper's own baseline): "naive" appears once
(p.3), in a RELATED WORK SUMMARY TABLE listing what a DIFFERENT cited
paper (Sorkun et al. [37]) compared against ("LSTM, naive, GRU, RNN"),
not this paper's own comparison set. EXHAUSTIVE CHECK confirms zero
"persistence"/"climatology"/"reference forecast" hits anywhere, and the
Related Work table is the only "benchmark"/"naive" context in the text
outside this paper's own Table 3 (LSTM/CNN/GRU/CNNGRU/CNNLSTM/
DCNN-BRLSTM/DSCLANet).

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Metrics are MSE, MAE, RMSE (Table 3, p.7).

**weather_source = measured** | "historical weather and solar power
generation data" (p.6) from the DKASC site logger; no NWP/forecast-weather
language anywhere.

**split_type = not_stated** | "All the datasets are split into 70%, 20%,
and 10% training, testing, and validation data, respectively." (p.6-7) -
ratio only, no statement of chronological vs. random ordering.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches anywhere for
seed/standard deviation/"+/-"/confidence interval/error bars/multiple
runs/repeated trials.

**code_available = not_stated** | "Data Availability Statement: Not
applicable." (p.11) - the exact MDPI boilerplate the coding rules
explicitly instruct to code as not_stated, not "no" (same pattern as
lim2022cnnlstmsunnycloudy elsewhere in this batch).

**key_claim**: "For one-hour-ahead PV power prediction from two hours of
historical input, evaluated on three DKASC Alice Springs arrays (Trina
1A/10.5kW, Trina 1B/23.4kW, Eco 2/26.5kW; 70/20/10 train/test/validation
split, ordering not stated), the proposed dual-stream CNN-LSTM +
self-attention model (DSCLANet) achieves the lowest MSE/MAE/RMSE of all
compared deep learning models (LSTM, CNN, GRU, CNNGRU, CNNLSTM,
DCNN-BRLSTM) on every one of the three arrays, e.g. Trina 1A:
MSE=0.0167/MAE=0.0632/RMSE=0.1291 versus plain CNNLSTM's
MSE=0.0679/MAE=0.12/RMSE=0.2606." | Section 3.3, Table 3 (p.7).

## What this paper does WELL

- Evaluates on THREE separate DKASC arrays with different technologies/
  capacities and reports per-array results (Table 3) rather than a single
  pooled number - the closest design in this batch to this project's own
  multi-array evaluation structure, even though the specific arrays and
  years differ.
- Explicit, single, stated forecast horizon and input window (2h input ->
  1h ahead).
- Runs a genuine architecture ablation (LSTM, CNN, GRU, and their
  pairwise/triple combinations) rather than only comparing to unrelated
  external work.

## Other observations

- No RMSE<MAE or other internal-consistency violation found in the
  numbers quoted in Table 3 as extracted (RMSE consistently exceeds MAE
  across every model/dataset cell checked) - unlike the MAE>RMSE pattern
  flagged in three other papers in this batch.
- No shared authors/institution with any other paper coded in this batch
  (Qassim University / Onaizah Colleges, Saudi Arabia) - the DKASC overlap
  with hussain2022hybridgrucnn is a shared DATASET, not a shared author or
  institution.
