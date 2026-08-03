# Audit: vennila2022solarensemble

Source: `data/papers/International Journal of Photoenergy - 2022 - Vennila - Forecasting Solar Energy Production Using Machine Learning.pdf` (7 pages, 26,957 chars)

Vennila, C., Titus, A., Sri Sudha, T., Sreenivasulu, U., Pandu Ranga
Reddy, N., Jamal, K., Lakshmaiah, D., Jagadeesh, P., Belay, A. (2022).
"Forecasting Solar Energy Production Using Machine Learning."
*International Journal of Photoenergy* (Hindawi) 2022:7797488.

This is the thinnest paper on methodology reporting coded in this batch:
almost none of the survey's judgement fields have any supporting text at
all, positive or negative.

## Coded fields

**year**: 2022 | "Volume 2022, Article ID 7797488"

**venue**: International Journal of Photoenergy (Hindawi) 2022:7797488

**dataset**: 10 MW capacity, polycrystalline (Poly-Si) and thin-film
(TFSC) solar panels, 5-minute resolution; SITE NEVER NAMED | "a total of
10 MW of capacity was achieved through the use of polycrystalline solar
panels (Poly-SI) and thin-film solar cells (TFSC)... we used data with a
five-minute resolution." (p.6)

FLAG (observation, not a coded field): the Poly-Si + thin-film technology
mix and 5-minute native resolution closely resemble this project's own
DKASC dataset composition (array11 is poly-Si, array07/dropped is CdTe
thin-film), and the paper's own reference list cites "Machine learning
based PV power generation forecasting in Alice Springs" (IEEE Access 9,
2021) in the same discussion paragraph about how "the location and
construction of the power plant may have an impact" on generalization
(p.6). This is suggestive but NOT a stated fact - the paper never names
its own dataset's location or source, and the Alice Springs mention is
explicitly a citation to a DIFFERENT paper, not a description of this
paper's own data. Not coded as DKASC; recorded as an unresolved
resemblance only.

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: zero matches
anywhere for night/daytime/daylight/sunrise/sunset/zenith/clear-sky/
diurnal in the full 26,957-character text.

**baseline_used = none** | EXHAUSTIVE CHECK: zero matches anywhere for
persistence/naive/benchmark/reference forecast/climatology/reference
model. The paper compares an "ensemble model that integrates all of the
combination strategies" against "standard individual models" (Abstract,
p.1) and "conventional individual models" (p.6), but no specific model
names, architectures, or baseline descriptions are given anywhere in the
extracted text - not even which individual ML algorithms make up the
ensemble. This is coded "none" rather than "other_ML"/"own_components"
because the comparators are never named or described at all, not because
a naive reference was affirmatively absent from an otherwise detailed
comparison - flagged as a borderline none/not_stated case in the final
uncertain-fields report, since the paper clearly ran SOME comparison
(it reports the ensemble "outperformed" something) but never says what.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches for "skill
score"/"forecast skill"/"relative to persistence"/"SS"/"improvement over
persistence" anywhere. The only quantitative metric named anywhere in the
extracted text is RMSE, mentioned once in passing ("The RMSE of the power
forecast for a gloomy day is slightly higher than that of the power
forecast for a clear day," p.6) with no table of values found.

**weather_source = not_stated** | no NWP/measured/exogenous-weather
language of any kind describing model inputs; the paper discusses "clear,"
"overcast," and "partly cloudy" conditions (see RQ3-relevant note below)
but never states what weather variables, if any, feed the model, or their
source.

**split_type = not_stated** | EXHAUSTIVE CHECK: zero matches anywhere for
a training/test ratio, percentage, or ordering statement of any kind -
this paper does not report a split ratio at all, unlike every other paper
coded in this batch (all of which give at least a ratio even when ordering
is unstated).

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches for seed/
repeated runs/standard deviation/"+/-"/confidence interval/error bars/
multiple runs anywhere in the text.

**code_available = not_stated** | "Data Availability: The data used to
support the findings of this study are included within the article.
Further data or information is available from the corresponding author
upon request." (p.6) - data only, not code, and gated behind request.

**key_claim**: "An ensemble machine learning model (architecture and
individual member models never named) outperforms unnamed 'conventional
individual models' for solar energy forecasting on a 10MW Poly-Si/thin-
film dataset (5-min resolution, site never named); performance is better
under clear-sky conditions than overcast or partly-cloudy conditions, with
partly-cloudy days showing particularly poor agreement between actual and
forecasted power (p.6); forecast horizon, train/test split, and the
specific models compared are never stated anywhere in the paper."

## RQ3-RELEVANT FINDING (worth cross-referencing in this project's own Related Work)

"According to the example studies depicted in Figure 5, clear weather
conditions outperformed overcast and partly cloudy conditions for each ML
model... It is the actual and forecasted [power] for the partly cloudy day
that are particularly bad. The error in a forecast is influenced by the
variety of the forecast. It is mostly owing to the greater unpredictability
associated with partially cloudy conditions that the abovementioned
discrepancies in forecast errors exist." (p.6)

This is a qualitative, non-quantified (no numbers given) version of this
project's own Finding 12 Part B: that partly-cloudy conditions are harder
to forecast than overcast ones, not just harder than clear skies. Unlike
this project's finding, this paper does not rank overcast vs. partly-cloudy
against EACH OTHER explicitly, and gives no k_ghi-style classifier
definition, sample counts, or per-class skill scores - it is a descriptive
observation, not a controlled stratified result. Still worth citing as
independent, if weak, corroborating evidence for the "partly-cloudy is the
hard case" claim in this project's own RQ3 writeup.

## What this paper does WELL

- The clear/overcast/partly-cloudy stratification observation above,
  however thinly evidenced, is a genuine attempt to look at conditional
  performance rather than only a single pooled accuracy number.

## Other observations

- This is the shortest and least methodologically detailed paper in the
  batch (7 pages; no split ratio, no named baseline, no named site, no
  horizon, no specific model list found anywhere in the extracted text) -
  worth using as the batch's low-water-mark example if the survey writeup
  wants one.
- No shared authors/institution with any other paper coded in this batch
  (multi-institution Indian/Ethiopian consortium).
