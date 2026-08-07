# Audit: abumohsen2024cnnlstmrf

Source: `data/papers/1-s2.0-S277267112400216X-main (1).pdf` (21 pages, ~99,600
chars). NOTE: this exact filename is a duplicate copy identified during the
2026-08-06 survey batch; content matches the paper originally coded as
abumohsen2024cnnlstmrf before this project's evidence-extraction tool
existed. This audit file was written retroactively, from this duplicate
PDF's extracted text, specifically to upgrade this row from
evidence_level=summary_only to evidence_level=quoted - it is a real
independent re-verification against the source PDF, not a formality.

Abumohsen, M., Owda, A.Y., Owda, M., Abumihsan, A. (2024). "Hybrid machine
learning model combining of CNN-LSTM-RF for time series forecasting of
Solar Power Generation." *e-Prime - Advances in Electrical Engineering,
Electronics and Energy* 9:100636.

DOI: https://doi.org/10.1016/j.prime.2024.100636

## Coded fields (verified against source; existing CSV values checked one by one)

**year = 2024** | "e-Prime - Advances in Electrical Engineering, Electronics
and Energy 9 (2024) 100636" (p.1). MATCHES CSV.

**venue = e-Prime 9:100636** | as above. MATCHES CSV.

**dataset**: proprietary Tubas Electricity Company (Palestine) SCADA log,
5045 records, ~14 months, Jun 3 2022 - Jul 31 2023 | "This data contains
5045 records, dataset utilized in our study encompasses a period of
approximately 14 months, spanning from the date Jun 3, 2022, to the date
of July 31, 2023." (p.4, Section 3.1 Data collection); "The data used in
this paper were obtained from Tubas Electricity Company - Palestine. All
loads are stored through the SCADA program in a database." (p.4).
MATCHES CSV ("proprietary 5045 records ~14 months Jun2022-Jul2023" -
"proprietary" is an editorial label for a private utility's SCADA log,
not a verbatim word, consistent with how the dataset column is used
elsewhere in this survey as a free-text descriptor, not a restricted
vocabulary field).

**night_hours_excluded = not_stated** | EXHAUSTIVE CHECK: matches for
night/daylight/sunrise/sunset/zenith/clear-sky/diurnal are all either
metaphorical ("SPG reaches its zenith in the summer months" (p.5) means
seasonal peak, not solar zenith angle) or descriptive EDA about observed
generation timing, not a stated evaluation-protocol exclusion: "production
significantly decreases before sunrise and after sunset... in the early
morning hours between 5 a.m. and 8 a.m., the average power generated is
only about 200 kW" (p.5) - this states early-morning power is LOW, not
ZERO, and nowhere says these hours are dropped from the 80/20 train/test
split or from the RMSE/MAE/R2 computation. "variations in daylight hours
throughout the year" (p.6) is a speculative aside about a possible
confound, not a filtering statement. MATCHES CSV VALUE, but see AMBIGUITY
note below - whether night hours are excluded from evaluation is
genuinely unclear either way, not just "unmentioned."

**baseline_used**: CSV currently has `none (persistence mentioned in intro
only)` - format issue and content flag, see "DOES NOT CLEANLY SURVIVE"
below.

**skill_score_reported = no** | EXHAUSTIVE CHECK: zero matches anywhere for
skill score/forecast skill/relative to persistence in ~99,600 characters.
All metrics are RMSE, MAE, R2 (Tables 2-5). MATCHES CSV.

**weather_source**: CSV currently has `measured` - content flag, see
"DOES NOT CLEANLY SURVIVE" below.

**split_type**: CSV currently has `80/20 chronology not stated` - not a
member of this project's allowed split_type vocabulary
(chronological/random/k-fold/rolling/not_stated). Content is verified
correct - "The dataset was split into a training set, accounting for 80%
of the data, and a test set, representing the remaining 20%." (p.12) -
ratio given, ordering (chronological vs. random) never stated anywhere.
FORMAT ISSUE, see below.

**n_seeds = not_stated** | EXHAUSTIVE CHECK: zero matches for seed anywhere
in the text. MATCHES CSV.

**variance_reported = no** | EXHAUSTIVE CHECK: zero matches for seed/
standard deviation/confidence interval/error bars/repeated run(s) or
trial(s) anywhere in the text. MATCHES CSV.

**code_available**: CSV currently has `not stated` (missing underscore,
format issue only) | "Data availability: Data will be made available on
request." (p.18) - this is a DATA availability statement, not a code
statement; EXHAUSTIVE CHECK: zero matches for github/code available/
open-source/zenodo/repository anywhere. Content correctly not_stated.

**key_claim**: "the hybrid model combining CNN-LSTM-RF demonstrated
superior accuracy with R-squared of 92%, a Root Mean Square Error (RMSE)
of 0.07 kW, and a Mean Absolute Error (MAE) of 0.05 kW" (Abstract, p.1),
confirmed again in Table 5 (p.17): RF 0.09/0.05/0.89, Bi-LSTM
0.10/0.06/0.9027, CNN-LSTM-RF 0.07/0.05/0.92 (RMSE/MAE/R2). MATCHES CSV
exactly.

## DOES NOT CLEANLY SURVIVE THE CHECK

1. **baseline_used content, not just format**: the paper's own comparison
   set is RF, SVR (classical ML, Table 2), LSTM, Bi-LSTM, RNN, GRU (deep
   learning, Table 3), and LSTM-RF, CNN-LSTM-RF (hybrid, Table 4) - CNN,
   LSTM, and RF are literally the three components of the proposed hybrid
   model. Elsewhere in this same 27-row survey, a comparator set built
   from the proposed model's own building blocks is consistently coded
   `own_components` (e.g. li2022eemdssalstm vs. plain LSTM,
   alharkan2023dsclanet vs. LSTM/CNN/GRU/CNNGRU/CNNLSTM), not `none`. The
   current `none` value looks like it is tracking "no persistence/
   climatology reference forecast" (which is true - see below), but that
   is a different question from "was there a baseline at all," and is
   inconsistent with how `none` vs. `own_components` was drawn everywhere
   else in this batch. I have NOT recoded this - it is a category-boundary
   judgment call, not a factual quote correction like the ibrahim2024 case,
   and you said resolve nothing except what I was explicitly told to fix.
   Flagging for your call: either recode to `own_components`, or treat
   this as confirmation that `none` was always meant to mean "no
   persistence/climatology reference," in which case several of the other
   24 rows may be using `own_components`/`other_ML` too loosely relative
   to a stricter reading. The one thing I can confirm with a quote: the
   ONLY persistence mention in the whole paper is in the literature
   review, describing the general field, not this paper's own experiment:
   "the persistence (or smart persistence) model being one of the most
   basic yet essential methods... It acts as a standard for comparing the
   effectiveness of various forecasting techniques" (p.2, Section 2.1) -
   this paper never runs persistence itself.

2. **weather_source content, not just format**: this is NOT uniformly
   "measured." Power, date/hour, and temperature come from the utility's
   own SCADA log (genuinely measured, on-site). But wind speed, humidity,
   and atmospheric pressure do not: "To examine the significance and
   influence of the pressure factor, wind speed, and humidity on solar
   energy generation, the data obtained from the Tubas Electricity Company
   lacked information on these variables. Consequently, data for these
   factors was sourced from NASA (nasa.gov) [56]." (p.4), where ref [56]
   resolves to "https://power.larc.nasa.gov/data-access-viewer/" (p.19) -
   i.e. NASA POWER. I have NOT recoded this to `reanalysis`, unlike
   ibrahim2024cnnlstmautoencoder's MERRA-2 case: this paper never
   characterizes the NASA data as reanalysis, satellite, or model-derived
   - it just says "sourced from NASA." Recoding to `reanalysis` would be
   inferring the technical nature of NASA POWER's data pipeline from
   outside knowledge, which is exactly the kind of inference this
   project's coding rules forbid. `measured` is also not a clean fit,
   since 3 of the paper's weather variables are not from the site's own
   sensors. This is a genuine unresolved ambiguity, listed below.

3. **Format nonconformance (content is correct, only the string doesn't
   match ALLOWED_VALUES)**: `night_hours_excluded` = "not stated" should
   be "not_stated"; `split_type` = "80/20 chronology not stated" is not in
   {chronological, random, k-fold, rolling, not_stated} at all (the ratio
   belongs in dataset/notes, the value itself should just be
   "not_stated"); `code_available` = "not stated" should be "not_stated".
   These three rows (this one plus xu2025lstmxgboosteemdso and
   energyeng2025cnnlstmcascade) all predate `code_literature_survey.py`'s
   strict-vocabulary enforcement and were entered as free text. I have not
   reformatted them - that touches more than the row you asked me to
   verify, and one of the three (baseline_used) is now also a content
   question, not just formatting, per point 1. Flag only.

## What this paper does WELL

- States an exact record count, exact date range, and exact split ratio
  (5045 records, Jun 3 2022 - Jul 31 2023, 80/20) - more precise than most
  of this survey's other papers, which typically give only a ratio or only
  a vague date range.
- Transparent about a real data limitation and its fix: explicitly says
  the utility's own SCADA log lacked wind/humidity/pressure and names
  exactly where the substitute came from (NASA POWER), rather than
  silently blending sources without disclosure.
- Reports training time and memory usage per model (Tables 2-4) - a
  genuine reproducibility-relevant detail essentially no other paper in
  this batch includes.
- Table 6 (p.18) is an unusually extensive literature comparison table (24
  cited studies with their own reported metrics and locations side by
  side), more thorough related-work quantification than typical in this
  batch.

## Other observations

- No shared authors/institution with any DKASC-family paper in this batch
  (Arab American University, Palestine; site is Tubas, Palestine, not
  DKASC).
- Self-citation pattern: references [11], [12], [14] are the same
  author group's own prior electrical-load-forecasting papers - not a
  cross-paper survey concern, just noted for completeness.
- No MAE>RMSE inconsistency anywhere in Tables 2-5 (RMSE consistently
  meets or exceeds MAE in every reported cell) - unlike the pattern
  flagged in several other papers in this batch.
