# Literature Notes

Notes extracted 2026-07-28 from eight papers. Organised by role in the paper,
not by topic. Page-level quotes kept short; everything else is paraphrase.

ASCII only, per CLAUDE.md.

---

## 1. Yang et al. (2020), "Verification of deterministic solar forecasts"
Solar Energy 210:20-37. ~33 co-authors (Kleissl, Coimbra, Gueymard, Hong,
Perez, Lauret, Zhang...). Functions as community consensus, not one group's
opinion. THE cornerstone citation for this paper.

### What it recommends
- Murphy-Winkler distribution-oriented verification as standard practice
  (examines reliability, resolution, association, discrimination via the
  joint distribution of forecasts and observations).
- Universally report the RMSE skill score based on the OPTIMAL CONVEX
  COMBINATION of climatology and persistence as the reference.
- Rationale: skill score reflects the inherent difficulty of different
  forecasting situations, allowing comparison by relative improvement rather
  than absolute error size.

### CRITICAL for our RQ2 - the zenith angle filter
Direct paraphrase of their argument: during sunrise and sunset, at small solar
elevation angles, both the clear-sky index and the clearness index can become
very large, because of measurement uncertainty and clear-sky model
inaccuracy. Forecast errors are therefore large at those times. To exclude
these undesirable forecasts, which may severely distort error metrics, solar
forecasters usually apply a zenith angle filter (they give zenith < 85 deg as
the example) before computing errors.

>>> This is EXACTLY the residual we found at hours 7 and 18 and in the
>>> 10-20 degree elevation bin (see scripts/diagnose_clearsky_bias.py). We
>>> rediscovered a documented phenomenon empirically. Cite this.

Implications for us:
- Our threshold is solar_elevation > 10 deg, i.e. zenith < 80 deg. That is
  STRICTER than their example of 85 deg. State this explicitly and cite.
- The threshold itself becomes a legitimate RQ2 protocol knob:
  zenith < 85 vs < 80 vs no filter. Cheap to run, directly citable.
- They note multiplicative seasonality (which our smart persistence uses,
  k_p * P_cs) is what causes the problem; an ADDITIVE alternative exists:
  r_{t+h} = x_t - c_t + c_{t+h}. Worth one sentence in Methodology
  explaining why we chose multiplicative.

### On nRMSE - affects our metric choice
They criticise Blaga et al. (2019) for comparing nRMSE across publications.
Their point: nRMSE normalised by the mean of observations gives a false sense
of cross-scenario comparability and cannot be used to argue one forecaster
has more skill than another, because different data have different
predictability.

>>> We normalise by NAMEPLATE CAPACITY, not by mean observed power, which
>>> avoids the specific failure they name. But the broader warning still
>>> applies: nRMSE is not cross-scenario comparable. Say so, and lean on the
>>> skill score as the headline (CLAUDE.md rule 4).

### On reproducibility
They state that source code and data should be made available wherever
possible, and that without reproducibility it is cumbersome if not impossible
to verify reported forecast performance. Direct support for contribution #1.

### Honest note they make about themselves
They say the original intent was to propose a specific suite of metrics, but
it became clear no consensus was reachable - e.g. on MAE vs RMSE, or whether
normalised metrics should be used at all. Useful for our Introduction: even
the field's consensus paper concedes that verification practice is
unstandardised.

---

## 2. Nguyen & Musgens, "A Meta-Analysis of Solar Forecasting Based on Skill Score"
arXiv 2208.10536. First comprehensive meta-analysis of deterministic solar
forecasting by skill score.

### Scale
Screened 1,447 papers from Google Scholar, reviewed 320 full texts, built a
database of 4,687 data points, analysed with MARS, partial dependence plots
and linear regression.

### Findings that matter to us
1. **Forecast horizon dominates all other factors.** They conclude the
   analysis of solar forecasts should be done SEPARATELY FOR EACH HORIZON.
   >>> Validates our design (1h / 3h / 6h reported separately, never pooled).

2. **Training data length: more is better, up to a point.** They observe
   over-fitting when training data exceeds ~2000 days, and that around 2000
   days achieves the highest forecast accuracy.
   >>> OUR TRAINING SET IS 730 DAYS (2012+2013). Well under the optimum.
   >>> See ACTION ITEM below.

3. **Climate zone correlates significantly with skill score.** Arid (CZB in
   Koppen-Geiger) is a category in their model. Alice Springs is BWh (hot
   desert). Cite when discussing external validity.

4. **Ensemble-hybrid models achieve the best accuracy at all horizons**, and
   hybrids are strongest at intra-hour. BUT they also recommend, on
   accuracy-complexity grounds, pragmatic use of simple methods such as time
   series models - potentially with pre- and post-processing - before moving
   to complex ones.
   >>> This is our RQ4 (cost-effectiveness) stated by a meta-analysis. Cite
   >>> it as motivation, and note our result either supports or contradicts
   >>> the "ensemble-hybrid is best" conclusion under a controlled protocol.

5. They use "CP" = a convex combination of smart persistence and climatology
   as the reference concept - same as Yang et al. (2020).

6. Lower resolutions and shorter horizons yield higher skill scores.

---

## 3. Kapoor & Narayanan (2023), "Leakage and the reproducibility crisis in
machine-learning-based science", Patterns 4:100804.

### The headline
Data leakage affects at least 294 papers across 17 scientific fields. In their
civil war prediction case study, complex ML models were believed to vastly
outperform logistic regression; once leakage was corrected, they did not
perform substantively better than decades-old LR.

>>> This is the structural precedent for our expected result. If plain
>>> XGBoost matches the deep hybrid under a controlled protocol, that is not
>>> a null result - it is the same finding as theirs, in a new domain.

### The taxonomy of eight types (use these labels in our paper)
- **[L1] Lack of clean separation of training and test data**
  - [L1.1] No test set - training and testing on the same data.
  - [L1.2] Pre-processing on training and test set - using the entire dataset
    for imputation, scaling, over/under sampling.
  - [L1.3] Feature selection on training and test set - selecting features
    using information about test-set performance.
  - [L1.4] Duplicates in datasets - same record in both splits.
- **[L2] Model uses features that are not legitimate** - features that should
  not be available in the modelling exercise (proxies for the target).
- **[L3] Test set is not drawn from the distribution of scientific interest**
  - [L3.1] Temporal leakage - test set contains data from before the training
    set; model built using data "from the future".
  - [L3.2] Nonindependence between training and test samples.
  - [L3.3] Sampling bias in test distribution.

They propose "model info sheets" as the mitigation - a checklist filled in per
type.

### Direct mapping to our protocol rules
| Their label | Our CLAUDE.md rule | How we address it |
|---|---|---|
| L1.2 | rule 3 | scalers and feature stats fit on train only |
| L1.3 | rule 3 | feature selection must be train-only |
| L2   | loader.py | dropped Active_Energy_Delivered_Received, Performance_Ratio, Current_Phase_Average as target-derived; oracle regime labelled as upper bound |
| L3.1 | rule 1 | chronological splits, never shuffled |
| L3.2 | data window note | three co-located arrays share ONE weather station; errors correlated; not independent samples |
| L3.3 | rule 2 | daylight filtering, and sky-condition stratification |

>>> Strong move for the paper: present our protocol AS a filled-in model info
>>> sheet, citing Kapoor & Narayanan. That is a concrete, citable framework
>>> rather than an ad hoc list of rules, and it makes Section 4.1 defensible.

---

## 4. Hewamalage, Ackermann & Bergmeir, "Forecast Evaluation for Data
Scientists: Common Pitfalls and Best Practices", arXiv 2203.10716.

- Random cross-validation splitting of a time series does not preserve
  temporal order and is described as problematic and often avoided. Reasons
  given: it makes serial correlation hard to capture, and non-stationarities
  cause problems depending on how the partition falls.
- Rolling-origin evaluation (also called time series cross-validation or
  prequential evaluation) is naturally susceptible to leakage, because data
  passes from the test set of one step into the training set of the next.
- They state they regularly encounter papers in top AI/ML conferences and
  journals - including best-paper winners - that use inadequate and
  misleading benchmark methods for comparison, and misuse error measures such
  as MAPE where it is clearly inappropriate.

>>> Useful for Related Work: this is a general-ML source saying the same
>>> thing we are saying about solar specifically. It broadens the claim
>>> beyond one subfield.

---

## 5-7. The hybrid literature we position against

These were read for EVALUATION PRACTICE, not architecture. Coded rows are in
results/literature_survey.csv.

### 5. Abumohsen, Owda, Owda & Abumihsan (2024), CNN-LSTM-RF
e-Prime 9:100636.
- Data: 5,045 records, approx 14 months, Jun 2022 to Jul 2023.
- Split: 80/20 train/test. Chronological ordering NOT stated.
- Reported: R-squared 92%, RMSE 0.07 kW, MAE 0.05 kW.
- No skill score. No persistence baseline used, although smart persistence is
  described in their introduction as one of the most basic yet essential
  methods.
- Night-hour handling not stated. Seed variance not reported.

### 6. Xu, Ji & Zhu (2025), LSTM-XGBoost-EEMD-SO
Scientific Reports 15:30177.
- Pipeline as described in the abstract: Thompson-Tau-Newton interpolation for
  missing data -> Pearson correlation feature selection -> EEMD decomposition
  of the power sequence into subsequences, reconstructed into low- and
  high-frequency components by sample entropy -> parallel XGBoost (low
  frequency) and LSTM (high frequency) -> Snake Optimization weight
  allocation.
- Reported: RMSE reduced 66.08% vs XGBoost and 31.33% vs LSTM; MAE reduced
  64.69% and 61.70% respectively.
- Baseline is its own components. No persistence baseline, no skill score.
- CAUTION, stated carefully: the described pipeline applies EEMD and Pearson
  feature selection to the power sequence with no train/test split described
  beforehand. The paper does not state that decomposition and feature
  selection are confined to training data. IF they are computed on the full
  series, that is [L1.2] and [L1.3] in Kapoor & Narayanan's taxonomy, and
  decomposition-before-split is a known leakage pattern in this literature.
  We should NOT assert they leaked - we should note that the paper does not
  report where the split occurs, which is itself the reporting problem our
  paper is about.

### 7. Improved CNN-LSTM with cascade learning (2025)
Energy Engineering 122(5):1975-1999.
- Data: one year from a PV station in Shanghai. Forecasts made for 05:00-19:00
  only, i.e. they DO apply a daylight-type restriction (one of the few).
- Split: 8:2 train/test.
- Feature selection by Pearson correlation coefficient and mutual information,
  described in their pipeline BEFORE the split step.
- Hyperparameter tuning by GridSearchCV using k-fold cross-validation, which
  they describe as randomly splitting the dataset into k mutually exclusive
  subsets.
  >>> Random k-fold on time-series data is [L3.1] temporal leakage in the
  >>> hyperparameter selection stage. This is a clean, citable example.
- Cascade expansion decided on an R-squared criterion. No skill score, no
  persistence baseline.

---

## 8. Rzadkowska (2025), "Quantifying artificial intelligence impacts on the
photovoltaics value chain"

Techno-economic / policy framework paper, not a forecasting-methods paper.
Maps technical KPIs to cash-flow items, propagates uncertainty with Monte
Carlo, reports delta-NPV and delta-LCOE.

Relevance to us is narrow but real: it insists on defensible counterfactual
baselines matched to the same physical and market context. That is the
economic-framing version of our argument about reference forecasts. Usable for
one sentence in the Introduction on why forecast accuracy claims need honest
baselines to translate into value. Low priority otherwise.

---

## ACTION ITEMS ARISING

1. **Extend the training window.** Meta-analysis evidence says accuracy
   improves with training length up to roughly 2000 days. We currently train
   on 730 days. The audit showed 2009-2016 is clean, and arrays 7, 11 and 12
   were all installed in 2008. Proposed: train 2009-2013 (1826 days),
   validate 2014, test 2015. Cost is one config change and one re-run of
   build_processed.py; gain and temperature climatology refit automatically.
   MUST be decided before any model is trained.

2. **Cite Yang et al. (2020) for the daylight threshold** and state that our
   10 deg elevation cutoff (80 deg zenith) is stricter than the 85 deg zenith
   they give as typical. Add the threshold as an RQ2 protocol knob.

3. **Reframe Section 4.1 as a model info sheet** per Kapoor & Narayanan,
   with the L1-L3 labels mapped to our seven rules.

4. **Justify smart persistence over the recommended convex combination** of
   climatology and persistence, or implement the convex combination as a
   second reference. A reviewer who knows Yang et al. (2020) will ask.

5. **Report skill score separately per horizon**, never pooled - the
   meta-analysis is explicit that horizon dominates every other factor.
