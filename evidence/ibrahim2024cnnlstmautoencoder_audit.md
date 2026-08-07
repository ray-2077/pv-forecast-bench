# Audit: ibrahim2024cnnlstmautoencoder

Source: `data/papers/s00202-023-02220-8.pdf` (17 pages, 40,963 chars)

Ibrahim, M.S., Gharghory, S.M., Kamal, H.A. (2024). "A hybrid model of CNN
and LSTM autoencoder-based short-term PV power generation forecasting."
*Electrical Engineering* (Springer) 106:4239-4255.

DOI: https://doi.org/10.1007/s00202-023-02220-8

## Coded fields

**year**: 2024 | "Electrical Engineering (2024) 106:4239-4255"

**venue**: Electrical Engineering (Springer) 106:4239-4255

**dataset**: Southern UK solar farm, power + co-located weather data,
70/30 train/test split by sample count (27,840 train / 9,360 test),
sliding 2-day window, no seasonal stratification | "The introduced model
is tested on dataset of power generation from southern UK solar farm and
the weather data corresponding to same location and time intervals"
(Abstract, p.1); "A sliding window technique was used with a window of
two days and no classification of seasons were used. The datasets were
divided so that 70% of the data were used for training the models (27,840
sample points) and 30% of the dataset were used for testing the models
(9360 sample points)." (p.7)

**night_hours_excluded = no** | "it is clear that PV power generation is
approximately near to zero value at night, whereas the tendency of all the
prediction models is to attain satisfactory forecasting accuracy in the
other durations' day." (p.9) - this discusses model behavior AT NIGHT as
part of the reported results, which is only possible if night-time rows
remain in the evaluated dataset; this is affirmative evidence of
inclusion, not merely an absence of an exclusion statement.

**baseline_used = own_components** | "The used model is compared with
different models from the literature either of pure type of network such
as LSTM and gated recurrent unit (GRU) or hybrid combination of different
networks like CNN-LSTM and CNN-GRU." (Abstract, p.1) - LSTM, GRU,
CNN-LSTM, and CNN-GRU are the component/ablation set the proposed
CNN-LSTM-autoencoder builds from. EXHAUSTIVE CHECK: the only
"persistence" hit (p.16, reference list) is the SAME Zhang et al. 2020
"autoencoder-LSTM and persistence model" citation already seen in this
survey's ahsan2024cnnlstmsmartgrid entry - a bibliography title, not this
paper's own comparison.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Metrics are RMSE and MAE only (Abstract, p.1).

**weather_source = reanalysis** [RECODED 2026-08-06, was "measured" -
"reanalysis" did not exist as an allowed value when this paper was first
coded; the audit's original exhaustive check searched the vocabulary in
force at the time and could not have found this] | "a weather data is
used in this study as well, it includes temperature and irradiance data
that has been extracted for the same period from several sites
surrounding the solar farms using the MERRA-2 reanalysis data, and then
they had been averaged to get the weather condition at the solar farm."
(p.10, Section 3.1 Dataset - text extraction interleaves two PDF columns
around this passage, see raw extraction if re-verifying) - explicit named
reanalysis product (MERRA-2), spatially interpolated from surrounding
sites, not a station observation. The Abstract's "weather data
corresponding to same location and time intervals" (p.1) is a looser
paraphrase of this same MERRA-2-derived series, not a separate measured
source. NOTABLE: the paper explicitly runs its comparison "with and
without weather data" (p.9) as a stated ablation axis - i.e., it
separately reports a weather-free (power-only) variant alongside the
weather-including variant, for every compared model and horizon.

**split_type = not_stated** | Ratio (70/30) and exact sample counts given
(quoted above), but no statement of chronological vs. random ordering
anywhere in the text.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches anywhere for
seed/repeated runs/standard deviation (of model performance)/"+/-"/
confidence interval/error bars/multiple runs.

**code_available = not_stated** [DATA availability is a genuine positive,
see below] | "Data availability: The datasets are publicly available at
the Western Power Distribution Open Data Hub site
(https://www.westernpower.co.uk/innovation/pod, accessed on August 6,
2021) upon login." (p.15) - a real public DATA URL (login-gated but not
"on request" from an author), which per the coding rules should be
recorded separately as a positive; it says nothing about code, so
code_available remains not_stated.

**key_claim**: "For short-term (0.5h, 1h, 2h ahead) PV power forecasting
at a southern UK solar farm (70/30 split by sample count, ordering not
stated, tested both with and without co-located weather data), the
proposed CNN-LSTM-autoencoder model improves RMSE/MAE by an average of 5%
to 25% over LSTM, GRU, CNN-LSTM, and CNN-GRU comparators, while reducing
training time by almost 70%; accuracy is better at shorter horizons than
longer ones across all models tested." | Abstract (p.1); p.9 ("It is
obvious that the prediction of less horizon is better than that for
larger horizon.")

## What this paper does WELL

- Genuine public dataset with a working URL (Western Power Distribution
  Open Data Hub), not gated behind an author request - one of the
  strongest data-availability statements in this batch.
- Explicit multi-horizon evaluation (0.5h/1h/2h) with per-horizon results,
  and an explicit, stated finding that shorter horizons forecast better
  than longer ones - consistent with, and independent corroborating
  evidence for, Nguyen & Musgens' meta-analytic finding (cited in this
  project's own literature_notes.md) that horizon dominates other factors.
- Runs a genuine with/without-weather-data ablation for every compared
  model and horizon - a real, stated feature-regime sensitivity analysis,
  the closest thing in this batch to this project's own lagged-vs-oracle
  regime distinction (though this paper's "without weather data" variant
  is a power-only ablation, not a perfect-forecast upper bound the way
  this project's oracle regime is).
- Reports training-time reduction (70% less) alongside accuracy - relevant
  to RQ4.
- Discusses model behavior specifically at night as part of its own
  results narrative rather than silently pooling all hours together.

## Other observations

- No shared authors/institution with any other paper coded in this batch
  (Egyptian university consortium).
- Shares one bibliography reference (Zhang et al. 2020, autoencoder-LSTM
  and persistence) with ahsan2024cnnlstmsmartgrid elsewhere in this batch
  - both papers cite it, but there is no shared authorship or dataset
  between the two.
