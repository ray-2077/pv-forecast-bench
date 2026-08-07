# Audit: elizabethmichael2022cnnstackedlstm

Source: `data/papers/energies-15-02150.pdf` (20 pages, 70,679 chars)

Elizabeth Michael, N., Mishra, M., Hasan, S., Al-Durra, A. (2022).
"Short-Term Solar Power Predicting Model Based on Multi-Step CNN Stacked
LSTM Technique." *Energies* (MDPI) 15:2150.

SCOPE NOTE: forecasts solar IRRADIANCE (GHI) and Plane-of-Array (POA)
irradiance, not PV power output directly - same scope caveat as
molu2024bilstmaadc elsewhere in this batch.

DOI: https://doi.org/10.3390/en15062150

## Coded fields

**year**: 2022 | "Energies 2022, 15, 2150"

**venue**: Energies (MDPI) 15:2150

**dataset**: Sweihan Photovoltaic Independent Power Project, Abu Dhabi,
UAE; GHI (0-8.12 kWh/m2) and POA irradiance (0-1114 W/m2) | "The real
solar data from Sweihan Photovoltaic Independent Power Project in Abu
Dhabi, UAE is preprocessed" (Abstract, p.1); "irradiance varied from 0 to
8.12 (kWh/m2) and POA varied from 0 to 1114 (W/m2)" (p.5)

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches
for night/daytime/daylight/sunrise/sunset/zenith/clear-sky/diurnal
anywhere in the text. The stated irradiance range's minimum of 0
kWh/m2 is circumstantial (consistent with night/dawn/dusk values being
present) but not a stated fact about inclusion or exclusion either way.

**baseline_used = own_components** | "two different kinds of DL
techniques (such as CNN and LSTM) and a proposed hybrid multi-step
CNN-LSTM are utilized for short-term solar energy prediction. Afterward,
these models are evaluated and compared" (p.5) - CNN and LSTM are the
component architectures fused into the proposed hybrid. The paper's only
"persistence" mention is a literature-review TAXONOMY sentence, not its
own baseline: "the PV power forecasting approaches are again classified
into persistence model, statistical model, and Machine Learning (ML)
model-based methods" (p.2) - general background, not a comparator this
paper itself runs (EXHAUSTIVE CHECK: no other persistence/naive/
climatology/reference model hits anywhere, and the Results section names
only CNN, LSTM, and the proposed hybrid as compared models).

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence". Metrics are RMSE, MAPE, MAE (labelled "MAE" but formula
context suggests MSE is meant in places - not resolved here), and R2
(Abstract, p.1).

**weather_source = measured** | GHI and POA are directly measured at the
Sweihan plant; no NWP/forecast-weather language anywhere in the text.

**split_type = not_stated** | "the total collected data is divided into a
7:3 ratio for the training and testing of the proposed model... the data
division evaluated in this research work covers 70-30%." (p.5, stated
twice in near-identical wording at two points in the extracted text,
likely a two-column PDF extraction artifact rather than genuine
repetition) - ratio only, no statement of chronological vs. random
ordering anywhere (also repeated once more at p.14: "split into training
(70%) and testing (30%) datasets").

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: the only "standard
deviation" hits (p.5, x2) describe Z-SCORE DATA STANDARDIZATION
preprocessing ("Rescaling the data set so that the mean of the observed
values is 0 and the standard deviation is 1 is called data
standardization"), not model-performance variance. No seed/repeated-run/
"+/-"/confidence-interval language describing accuracy anywhere.

**code_available = not_stated** | No "Data Availability" statement of any
kind found in the extracted text - the paper has Author Contributions,
Funding, and Conflicts of Interest sections (p.19) but no data/code
availability statement at all, similar to ye2026distributedcnnlstm
elsewhere in this batch.

**key_claim**: "For short-term GHI and POA irradiance prediction at the
Sweihan PV plant, Abu Dhabi (70/30 train/test split, ordering not stated),
the proposed multi-step CNN-stacked-LSTM model achieves RMSE=0.36,
R2=0.98 for solar irradiance (GHI) prediction and RMSE=61.24, R2=0.96 for
POA prediction, outperforming plain CNN and plain LSTM as well as
'published works in the literature' (cross-study comparison, methodology
for that comparison not detailed in the extracted text); forecast horizon
is described only as 'short-term' / 'multi-step' without a specific lead
time stated in the passages read." | Abstract (p.1).

## What this paper does WELL

- Predicts and reports two separate, explicitly named target variables
  (GHI and POA) with separate metrics for each, rather than conflating
  them into one number.
- Explicitly describes its data standardization procedure with the exact
  formula used (Eq. 2, p.5) - fit-on-what-data is not stated, but the
  procedure itself is transparent.

## Other observations

- Claims comparison against "published works in the literature" (Abstract)
  for cross-study context, in the same spirit as hou2024vmdwoalstm
  elsewhere in this batch - but per Yang et al. (2020)'s own warning
  (paper/literature_notes.md, section 1), cross-publication nRMSE/RMSE
  comparison is not generally valid without matching site/period/
  normalization, and this paper does not describe how it normalized for
  that comparison.
- No shared authors/institution with any other paper coded in this batch
  (BITS Pilani Dubai / Siksha O Anusandhan / Khalifa University).
