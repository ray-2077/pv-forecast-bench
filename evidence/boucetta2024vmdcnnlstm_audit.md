# Audit: boucetta2024vmdcnnlstm

Source: `data/papers/energies-17-01781-v2.pdf` (21 pages, 82,989 chars after
removing 119 embedded null bytes from the raw pdfplumber extraction, which
had made the file appear binary to text search tools)

Boucetta, L.N., Amrane, Y., Chouder, A., Arezki, S., Kichou, S. (2024).
"Enhanced Forecasting Accuracy of a Grid-Connected Photovoltaic Power
Plant: A Novel Approach Using Hybrid Variational Mode Decomposition and a
CNN-LSTM Model." *Energies* (MDPI) 17:1781.

## Coded fields

**year**: 2024 | "Energies 2024, 17, 1781"

**venue**: Energies (MDPI) 17:1781

**dataset**: Boussada PV plant, Algeria, 10 MW, 1 Jan 2019 - 31 Dec 2020,
15-min intervals, 69,195 data points, features: panel temperature, tilt
radiation, total radiation, direct radiation, humidity, PV power | "The
database used in this study is sourced from a photovoltaic (PV) plant
located in Boussada, a central-eastern city in Algeria known for its
partly desert climate. The PV plant has a capacity of 10 megawatts. Data
collection spanned from 1 January 2019, to 31 December 2020, with
measurements taken at fifteen-minute intervals, counting 69,195 data
points. The dataset contains parameters such as PV panel temperature,
tilt radiation, total radiation, direct radiation, humidity, and PV
power." (p.6)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches
anywhere for night/daytime/daylight/sunrise/sunset/zenith/clear-sky/
diurnal describing this paper's OWN data handling (the phrase "day/night
cycles" appears once, p.5, describing the CHALLENGE that motivates a
DIFFERENT cited paper - Phan et al. [22] - not this paper's own
preprocessing).

**baseline_used = own_components** | "The performance of the developed
model is benchmarked against other deep learning models across various
time horizons (15, 30, and 60 min): variational mode decomposition-
convolutional neural network (VMD-CNN), variational mode decomposition-
long short-term memory (VMD-LSTM), and convolutional neural network-long
short-term memory (CNN-LSTM)" (Abstract, p.1) - all three comparators are
ablation stages of the proposed VMD-CNN-LSTM (remove VMD, remove CNN, or
remove LSTM in turn). EXHAUSTIVE CHECK: zero matches for persistence/
naive/climatology/reference model/reference forecast describing this
paper's own comparison (only "benchmarked against other deep learning
models" language, always referring to the VMD-CNN/VMD-LSTM/CNN-LSTM set).

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Metrics used are RMSE, MAE, NRMSE, and R2 (p.19) - all
against ground truth.

**weather_source = measured** | on-site plant-monitored parameters (panel
temperature, tilt/total/direct radiation, humidity - see dataset quote
above). NOTE: the text does contain an NWP mention ("leveraging two years
of numerical weather prediction (NWP) data from Taiwan's Central Weather
Bureau," p.5) but this describes a DIFFERENT cited paper (Phan et al.
[22]) in the Related Work section, not this paper's own Boussada dataset,
which uses only on-site measured sensor data per Section 3.1.

**split_type = not_stated** | EXHAUSTIVE CHECK: zero matches anywhere in
82,989 characters for a training/test split ratio, percentage, or ordering
statement of any kind. Like lim2022cnnlstmsunnycloudy and
vennila2022solarensemble elsewhere in this batch, this paper does not
report a split ratio at all.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches for seed/
repeated runs/standard deviation/"+/-"/confidence interval/error bars/
multiple runs anywhere in the cleaned text.

**code_available = not_stated** | "Data Availability Statement: Data are
contained within the article." (p.20) - a genuine positive DATA statement
(not gated behind request), but says nothing about code.

**key_claim**: "For grid-connected PV power forecasting at the Boussada
plant, Algeria (10MW, 2019-2020, 15-min data, 69,195 points, split ratio
never stated), the proposed VMD-CNN-LSTM hybrid outperforms VMD-CNN,
VMD-LSTM, and plain CNN-LSTM ablations across three forecast horizons (15,
30, and 60 minutes ahead) on RMSE, MAE, NRMSE, and R2." | Abstract (p.1);
Section 4 horizon description (p.16: "different forecasting horizons: one
step (15 min), two steps (30 min), and fourth steps ahead (60 min)").

## What this paper does WELL

- Reports results across THREE explicit forecast horizons (15/30/60 min)
  with per-horizon tables, rather than a single pooled number - directly
  consistent with this project's own emphasis (and Nguyen & Musgens'
  meta-analytic finding, cited in this project's own literature_notes.md)
  that horizon should be analyzed separately, never pooled.
- Reports model complexity (parameter counts) alongside accuracy
  (p.14: "various metrics were analyzed, including model complexity, as
  measured by the number of...") - relevant to this project's own RQ4.
- States a genuine positive data-availability statement rather than an
  "on request" gate.

## Other observations

- A minor technical note for this project's own pipeline: the raw
  pdfplumber-extracted text for this specific PDF contained 119 embedded
  null bytes partway through, which made grep/ripgrep treat the file as
  binary and silently skip pattern matches until the null bytes were
  stripped. Worth checking for in any future automated re-extraction of
  this batch (data/papers/energies-17-01781-v2.pdf specifically), since a
  tool that doesn't handle this could silently under-search this one file.
- No shared authors/institution with any other paper coded in this batch
  (USTHB Algeria / University of M'Sila / Czech Technical University in
  Prague).
