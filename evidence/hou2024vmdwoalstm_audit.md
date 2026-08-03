# Audit: hou2024vmdwoalstm

Source: `data/papers/1-s2.0-S2352484724002749-main.pdf` (14 pages, 78,097 chars)

Hou, Z., Zhang, Y., Liu, Q., Ye, X. (2024). "A hybrid machine learning
forecasting model for photovoltaic power." *Energy Reports* 11:5125-5138.

**SAME DATASET FAMILY AS THIS PROJECT**: "The dataset used in this study is
obtained from Yulara Solar System in Uluru (Ayers Rock), Australia. It
covers the period from January 1, 2018 to December 31, 2018 (DKA Solar
Centre, 2018)." (p.9) Yulara is a DKA Solar Centre site - the same network
this project's own DKASC data comes from (Alice Springs), and is explicitly
named in this project's own FUTURE_WORK.md as the candidate for
"Multi-location generalisation." This is the second paper in this survey
batch built on DKA Solar Centre data (after zhou2024cnnlstmattnbayes,
which used the main DKASC Alice Springs pool) - different site, same
network/data provider.

## Coded fields

**year**: 2024 | "Energy Reports 11 (2024) 5125-5138"

**venue**: Energy Reports 11:5125-5138

**dataset**: Yulara Solar System (DKA Solar Centre), 1.8 MW, Uluru,
Australia, full year 2018, hourly (resampled from source), daytime
7am-7pm only, 4745 samples | "The dataset used in this study is obtained
from Yulara Solar System in Uluru (Ayers Rock), Australia. It covers the
period from January 1, 2018 to December 31, 2018 (DKA Solar Centre,
2018)... weather temperature, relative humidity, global horizontal
radiation, diffuse horizontal radiation, and PV power data were selected
from the original dataset for the period between 7:00 am and 7:00 pm every
day. The data resolution was one hour, with a total of 4745 sampling
points." (p.9)

**night_hours_excluded = yes** | "As the PV array does not generate
electricity at night, weather temperature, relative humidity, global
horizontal radiation, diffuse horizontal radiation, and PV power data were
selected from the original dataset for the period between 7:00 am and 7:00
pm every day." (p.9)

**baseline_used = own_components** | "The effectiveness of the model is
tested by comparing it with other benchmark models, such as LSTM,
WOA-LSTM, VMD-LSTM, and WOA-VMD-LSTM." (p.3) - all four comparators are
ablation stages of the paper's own proposed pipeline (progressively adding
WOA optimization and VMD decomposition to a plain LSTM). No persistence or
climatology reference is used BY THIS PAPER for its own evaluation.
EXHAUSTIVE CHECK: the only "persistence" hit in the full text (p.4) is in
the Related Work section, describing a DIFFERENT paper's method ("... and
persistence models in forecasting solar power data from Flanders, Belgium
..." - a cited prior study, not this paper's own baseline) - not this
paper's own comparison set.

**skill_score_reported = no** | EXHAUSTIVE CHECK: searched "skill score"/
"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence" - zero matches describing this paper's own metrics. Reported
metrics throughout are MAE, RMSE, MAPE, R2 (Abstract, p.1; Results,
pp.13-15) - all against ground truth, not a reference forecast. The
"X% improvement/decrease compared to LSTM" figures reported (e.g.
"Compared to the LSTM, MAE, RMSE, and MAPE decreased by 84.942%,
86.746%, and 82.611%, respectively") are percentage improvement over
another ML model, which the coding rules explicitly exclude from
skill_score_reported=yes.

**weather_source = measured** | inputs are weather temperature, relative
humidity, global horizontal radiation, diffuse horizontal radiation - all
drawn directly from the DKA Solar Centre historical log (see dataset quote
above); no NWP/forecast-weather language anywhere in the text.

**split_type = chronological** | "The previous 70% of the data is used as
the training set, and the remaining 30% is used as the testing set." (p.9)
Coded chronological on the word "previous" (= temporally earlier), the
same class of judgment as this survey's mayer2022physmlhybrid entry, which
used explicit calendar years rather than the word "chronological" itself.
FLAG (minor): "previous" is suggestive but not as unambiguous as named
calendar boundaries - a stricter reading could call this not_stated, since
the paper never explicitly rules out e.g. a stratified-but-still-ordered
split. Recorded in the final uncertain-fields list.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: searched seed/repeated/
standard deviation/"+/-"/confidence interval/std dev/error bars/multiple
runs. One "standard deviation... k-fold cross-validation" hit (p.2) is
Related Work, describing a different cited paper's (Konstantinou et al.,
2021) results, not this paper's own. WOA's "repeated iterations" (p.11)
refers to the optimizer's internal search iterations for one hyperparameter
run, not repeated end-to-end model training across seeds. No run-to-run
variance of this paper's own reported accuracy is given anywhere.

**code_available = not_stated** | "Data Availability: Data will be made
available on request." (p.14) - data only, not code; no other mention of a
code repository anywhere in the text.

**key_claim**: "For one-hour-ahead forecasting on the 1.8MW Yulara (DKA
Solar Centre) system, full-year 2018 hourly daytime (7am-7pm) data, 70/30
train/test split, the proposed WOA-VMD-LSTM model reaches MAE=15.247kW,
RMSE=19.753kW, MAPE=4.405%, R2=0.997, versus plain LSTM decreased by
84.942% (MAE), 86.746% (RMSE), 82.611% (MAPE), with R2 increased 23.438%."
| Abstract (p.1); consistent with the Conclusion (3) restatement (p.14).

## What this paper does WELL

- Explicit, stated daytime-only clock-hour filter (7am-7pm), not a vague
  mention.
- Reports a literature comparison table (p.15, "Comparison of the proposed
  model with other methods in recent studies") citing other papers' MAPE
  values against their own on the SAME general problem, at least attempting
  cross-study context even though the underlying datasets differ (a form of
  transparency about how the result fits the wider literature, imperfect as
  cross-study nRMSE/MAPE comparison is per Yang et al. 2020's own warning).
- States forecast horizon explicitly and consistently (one-hour-ahead,
  5-timestep input window) - many papers surveyed so far never state a
  horizon at all.

## Other observations / consistency checks

- MAPE is 4.405% here, far more plausible than li2022eemdssalstm's 57-69%
  or zhou2024's similar issue in this same survey batch, and no evidence of
  a near-zero-denominator problem despite the same general daytime-only PV
  setup - worth noting as a case where MAPE reporting looks well-behaved,
  a genuine positive contrast within this survey.
- R2=0.997 alongside MAE=15.247kW on a 1.8MW system (MAE ~0.85% of
  nameplate) is internally plausible and consistent in scale - no
  contradiction found.
- Cannot verify RMSE^2 ~ mean^2 + SD^2: no separate mean-bias or residual-
  SD figure is reported alongside RMSE on a stated common sample.
- Data-provenance overlap: both this paper and zhou2024cnnlstmattnbayes
  (also in this survey batch) draw from the DKA Solar Centre network,
  independently of each other (different sites, different institutions,
  no shared authors) - worth a combined note in the survey writeup about
  how often this literature draws on the same small set of public PV
  datasets (DKASC network appears in at least 2 of the papers coded in
  this batch, plus this project's own primary dataset).
- No shared authors/institution with any paper coded earlier in this
  batch.
