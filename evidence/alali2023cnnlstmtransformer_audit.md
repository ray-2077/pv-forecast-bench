# Audit: alali2023cnnlstmtransformer

Source: `data/papers/mathematics-11-00676-v2.pdf` (19 pages, 72,463 chars)

Al-Ali, E.M., Hajji, Y., Said, Y., Hleili, M., Alanzi, A.M., Laatar, A.H.,
Atri, M. (2023). "Solar Energy Production Forecasting Based on a Hybrid
CNN-LSTM-Transformer Model." *Mathematics* (MDPI) 11:676.

## Coded fields

**year**: 2023 | "Mathematics 2023, 11, 676"

**venue**: Mathematics (MDPI) 11:676

**dataset**: Fingrid open dataset (Finland's transmission system operator),
hourly solar power plant data, Finland | "The Fingrid open dataset [25]
was used to obtain the training data. The gathered data were updated at
an hourly rate. The data were collected from a solar power plant
established in Finland, which provided historical time series of weather
conditions." (p.13)

**night_hours_excluded = yes** [LOW EXTRACTION CONFIDENCE, see flag] |
Page 13 of the PDF suffers severe two-column text-merge corruption in the
raw extraction (the same passage appears twice, character-interleaved).
Reconstructing the legible fragments, the intended sentence reads: "By
excluding power levels throughout the night and on overcast days, the
data were then filtered." Coded "yes" on night exclusion because the
phrase "excluding power levels throughout the night" is legible and
consistent across both interleaved copies of the sentence.

MAJOR FLAG, separate from the night-hours field: the SAME sentence also
states overcast days were excluded ("...and on overcast days, the data
were then filtered"), which is a much less common and more consequential
protocol choice - it removes an entire weather-condition category from
evaluation, not just non-generating night hours. If accurate, this
directly inflates reported accuracy by construction, dropping exactly the
harder-to-predict cloudy regime rather than merely removing hours with no
signal to predict. This project's own Finding 12 Part B found overcast is
NOT the hardest condition at its site (partly-cloudy is), so the direction
of the inflation this would cause is not guaranteed to be large - but the
methodological move itself (silently dropping a whole condition class
before evaluation) is a clean, quotable example of exactly this project's
central thesis. Given the extraction quality issue, this reading should
be verified against the original PDF page 13 before being used as a firm
claim in the survey - recorded here as a flag, not a certainty, and listed
again in the final uncertain-fields report.

**baseline_used = own_components** | "Compared to existing models and
other combinations, such as LSTM-CNN, the proposed CNN-LSTM-Transformer
model achieved the highest accuracy." (Abstract, p.1) - comparators are
architectural variants/subsets of the proposed hybrid (an ablation study
is also explicitly mentioned: "the impact of the proposed components,"
p.13). EXHAUSTIVE CHECK: zero matches for persistence/naive/climatology/
reference model/reference forecast anywhere in 72,463 characters.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence" anywhere.

**weather_source = measured** | "historical time series of weather
conditions" (p.13) from the Fingrid-linked solar plant's own records; no
NWP/forecast-weather language anywhere.

**split_type = not_stated** | "After splitting the data into three
subsets, 60% were utilized for training, 10% were utilized for
validation, and 30% were utilized for testing." (p.13) - ratios given
(60/10/30), no statement of chronological vs. random ordering.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches anywhere for
seed/repeated runs/standard deviation (of model performance)/"+/-"/
confidence interval/error bars/multiple runs.

**code_available = not_stated** | "Data Availability Statement: Data will
be made available by the corresponding author upon request." (p.19) -
data only, gated behind request, no mention of code.

**key_claim**: "For solar power forecasting on the Fingrid (Finland)
open dataset (hourly data, night hours AND overcast days excluded per the
flagged passage above, 60/10/30 train/validation/test split, ordering not
stated), the proposed CNN-LSTM-Transformer hybrid outperforms LSTM-CNN and
other component combinations; the paper additionally presents a
sunny/cloudy/rainy conditional example (Figure 7) without a full
per-condition metrics table in the passages read; forecast horizon is
never explicitly stated as a lead time anywhere in the extracted text."

## What this paper does WELL

- Runs an explicit component-ablation study ("the impact of the proposed
  components... on the Transformer model," p.13) rather than only
  reporting the full stack.
- Uses a genuinely open, named, citable dataset (Fingrid) rather than an
  unnamed private facility - one of relatively few papers in this batch to
  do so, even though the data itself is only available "upon request"
  through the paper's own channel rather than a direct public link.
- Shows a qualitative sunny/cloudy/rainy breakdown (Figure 7), a further
  RQ3-relevant data point in this batch alongside
  ma2024cnnlstmspatiotemporal, vennila2022solarensemble, and
  lim2022cnnlstmsunnycloudy.

## Other observations

- PDF extraction quality issue: page 13 (dataset description) is severely
  corrupted in the raw pdfplumber text (two columns merged character-by-
  character), which is why the night/overcast-exclusion quote above is
  flagged as lower-confidence than other quotes in this survey - it should
  be re-verified against the original PDF rendering before being used as a
  firm citation in the paper's own writeup.
- No shared authors/institution with any other paper coded in this batch
  (Tabuk University / University of Tunis El Manar / Northern Border
  University / King Khalid University - Saudi Arabia/Tunisia consortium).
