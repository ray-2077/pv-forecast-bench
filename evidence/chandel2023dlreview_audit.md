# Audit: chandel2023dlreview

Source: `data/papers/1-s2.0-S2772940023000292-main.pdf` (12 pages, 80,328 chars)

Chandel, S.S., Gupta, A., Chandel, R., Tajjour, S. (2023). "Review of deep
learning techniques for power generation prediction of industrial solar
photovoltaic plants." *Solar Compass* 8:100061.

SCOPE FLAG, READ BEFORE THE CODED FIELDS: this is a REVIEW/SURVEY paper,
not an original empirical study - "a comprehensive updated review of
standalone and hybrid machine learning techniques for PV power
forecasting is presented" (Abstract, p.1). It synthesizes findings from
roughly 60+ other papers (reference list runs to at least [62] in the
extracted text) and, in its discussion, PROPOSES (but does not appear to
implement or empirically test) "a novel architecture of the Deep Learning
Network Model (DLNM)... considering factors influencing industrial solar
power generation" (Abstract, p.1; restated p.15: "A novel architecture of
DLNM for PV systems is proposed whereas all factors which influence PV
system outputs, are taken into [account]"). No dataset, no train/test
split, no reported RMSE/MAE/skill-score for this proposed DLNM was found
anywhere in the extracted text - it is presented as a conceptual design,
not a tested model. Every judgement field below is therefore coded as
"this paper has no empirical study of its own to report", NOT as "this
paper ran an experiment and omitted these details" - the distinction
matters for how the survey should weight this entry, and it is
recommended this paper be tagged/handled separately from the empirical
papers in the survey's summary statistics rather than counted as another
instance of under-reporting.

DOI: https://doi.org/10.1016/j.solcom.2023.100061

## Coded fields

**year**: 2023 | "Solar Compass 8 (2023) 100061"

**venue**: Solar Compass 8:100061

**dataset**: not_stated | this is a literature review; no dataset of its
own is collected or used. (Individual reviewed papers' datasets are
discussed throughout the body text, but those belong to the ~60 papers
being reviewed, not to this paper.)

**night_hours_excluded = not_stated** | no own preprocessing/data-handling
description exists for this paper to state a night-hour policy on.

**baseline_used = none** | EXHAUSTIVE CHECK: this paper implements and
tests no model of its own against any baseline - it has no "Results"
section with its own comparison table. Every persistence/baseline/skill-
score mention found in the text (e.g. "Skill score-MAE... Skill Score-RMSE
were achieved for 10 min ahead of PV forecasting i.e. 12.1% and 7.7%,
respectively" p.7) is the review summarizing a DIFFERENT paper's own
reported result, attributed to that paper's citation, not this paper's own
evaluation.

**skill_score_reported = no** | EXHAUSTIVE CHECK: same as above - "skill
score" appears multiple times in the text but always describing OTHER
papers' results within the review's synthesis (e.g. p.7, p.9), never a
skill score this paper computed itself, because it computed nothing
itself.

**weather_source = not_stated** | no own model, so no own weather input to
characterize.

**split_type = not_stated** | no own train/test split exists.

**n_seeds = not_stated**

**variance_reported = no** | EXHAUSTIVE CHECK: no repeated-run or seed
language describing an experiment this paper itself ran; the "training and
testing MSE of 7.524" figure at p.5 ("proposed architecture achieved
training and testing MSE of 7.524...") is, in context, summarizing a
REVIEWED paper's architecture and result (the surrounding paragraph
discusses a specific cited DNN/CNN study), not the DLNM this paper itself
proposes.

**code_available = not_stated**

**key_claim**: "This review of standalone and hybrid deep-learning PV
power forecasting techniques concludes that LSTM outperforms other deep
learning architectures across all examined time horizons, that GRU is
preferable to LSTM for small datasets given fewer required training
iterations, and that grouping datasets by input-feature similarity improves
reported accuracy across the reviewed literature; it further proposes -
without implementing or empirically testing - a novel 'Deep Learning
Network Model' (DLNM) architecture intended to incorporate all factors
influencing industrial PV system output." | Abstract (p.1); "A novel
architecture of DLNM for PV systems is proposed" (p.15).

## What this paper does WELL (as a review, on a different axis)

- Explicitly synthesizes evaluation-practice details ACROSS many papers
  (forecast horizon, dataset size, metric choice) as part of its narrative,
  which is closer to this project's own literature-survey exercise than
  any other paper coded in this batch - worth reading in full for
  candidate papers to add to this project's own survey, separate from
  coding it as a data point itself.
- Explicitly identifies "grouping datasets based on input feature
  similarity" as an accuracy-relevant methodological choice (p.1) - a
  protocol-sensitivity observation in the same spirit as this project's
  own RQ2, even though it is not empirically demonstrated here.

## Recommendation for the survey writeup

Treat this entry as a REVIEW, reported separately from the ~21 empirical
papers coded in this batch (and the 3 coded before it) rather than folded
into the same summary statistics (e.g. "X of 25 papers exclude night
hours") - counting a review paper's not_stated/none codings alongside
empirical papers' genuine omissions would conflate two different kinds of
absence and weaken the survey's central claim rather than support it.

## Other observations

- No shared authors/institution with any other paper coded in this batch
  (Shoolini University, India / University of Madeira, Portugal).
- Not counted toward the "no naive baseline" or "no skill score" totals
  for the same reason given in the scope flag above - recommend a
  separate review-paper count instead.
