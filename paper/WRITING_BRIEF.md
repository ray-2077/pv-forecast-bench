# WRITING BRIEF

Written 2026-08-07. Purpose: a single, self-contained document from which
the paper can be written by someone with ONLY this file, the results
CSVs under results/, and the run JSONs under results/. It does not
assume access to this conversation, to PROJECT_CHECKPOINT.md, or to
CLAUDE.md, though all three exist in the repo and agree with what is
written here as of this date.

Method: every number below was re-derived directly from a file in this
repo during the writing of this brief (CSV query, JSON read, or file
count), not copied from memory or from PROJECT_CHECKPOINT.md's prose
without re-checking. Where PROJECT_CHECKPOINT.md's prose could not be
reproduced from the committed data, that is stated explicitly in Section
5 or Section 9 (GAPS), not silently corrected or silently repeated.

ASCII only, per CLAUDE.md.

---

## 1. THE ARGUMENT IN FIVE SENTENCES

1. Short-term PV power forecasting is a large and active literature, but
   the survey conducted for this paper (27 papers coded, verbatim-quoted
   evidence for 25 of them) found only 1 that reports a proper reference
   forecast (skill score against an optimal convex combination of
   climatology and persistence, per Yang et al. 2020) and 0 that report
   run-to-run seed variance.
2. This project built a chronologically-split, leakage-audited evaluation
   protocol for the same problem (three co-located DKASC arrays, 2011-2013
   train / 2014 validate) and used it to measure how much of a hybrid
   deep-learning model's apparent advantage survives when the evaluation
   protocol is held fixed and only the architecture varies.
3. Under that fixed protocol, architecture differences are mostly small
   and mostly not statistically significant (Diebold-Mariano,
   Holm-corrected): XGBoost and a plain LSTM are indistinguishable at
   short horizons, convolution does not earn a significant accuracy gain
   anywhere, and a residual-correction stage that looks like a strong win
   under one plausible-but-wrong protocol choice (fitting the correction
   on the validation split) turns into a small, mostly-significant loss
   once that leak is corrected.
4. Meanwhile, protocol choices that are common in the surveyed literature
   change the headline number far more than architecture does: including
   night hours deflates nRMSE by roughly a third with almost no effect on
   skill; switching the reference forecast from plain persistence to the
   convex combination Yang et al. recommend cuts the apparent long-horizon
   "hybrid gain" at h=6 from +0.65 to +0.19 on one array and flips the
   reported horizon trend from monotonically increasing to non-monotonic;
   and the single largest effect measured in the whole project is a sign
   flip (-0.03 to +0.33) produced by one plausible-sounding but wrong
   sentence in a methods section (fit the residual stage on the
   validation split).
5. The paper's claim is therefore not "our architecture is better" but
   "most of what this literature reports as an architecture effect is at
   least as attributable to which of several defensible-looking evaluation
   choices was made" - and the paper documents, with a protocol-inflation
   table and a survey of the field's own reporting practice, how large
   that gap can be.

---

## 2. CLAIMS TABLE

Confidence levels used throughout this brief:
- **established** = Diebold-Mariano Holm-significant (p_holm < 0.05), or a
  closed-form/exact identity that was checked against data.
- **suggestive** = directional and consistent, but not Holm-significant,
  or resting on a single seed/array/horizon cell.
- **descriptive** = a count, a structural fact, or a reporting-practice
  statistic; not a statistical claim about a population.

| # | Claim | Source (exact) | Finding # | Section / Table-Fig | Confidence |
|---|---|---|---|---|---|
| C1 | XGBoost and LSTM are statistically indistinguishable at h=1 and h=3 (all 6 array x horizon cells, p_holm=1.0) | `results/table6_dm_lagged.csv`, rows where model_1/model_2={xgboost,lstm}, horizon in {1,3} | 8, 12A | Results RQ2; T5 | established |
| C2 | LSTM shows higher mean skill_vs_convex than XGBoost at h=6 on all 3 arrays (+0.0157/+0.0149/+0.0239), but is Holm-significant only on array17 | `results/table6_dm_lagged.csv`, xgboost-vs-lstm rows, horizon=6; means from `results/seed_sweep_summary_lagged.csv` | 8, 12A | Results RQ2; T3, T5 | suggestive (array17 leg only: established) |
| C3 | LSTM's 5-seed skill_vs_convex std is 2-3x XGBoost's at the same cell (e.g. array17 h6: 0.0061 vs 0.0027) | `results/seed_sweep_summary_lagged.csv`, std_skill_vs_convex column | 8 | Results RQ2/RQ4; T3 | descriptive |
| C4 | CNN-LSTM's mean skill_vs_convex is lower than plain LSTM's in 8 of 9 array x horizon cells | `results/seed_sweep_summary_lagged.csv`, mean_skill_vs_convex, model in {lstm,cnn_lstm} | 9 | Results RQ2; T3, T5 | descriptive (direction) |
| C5 | lstm vs cnn_lstm is NOT Holm-significant in any of the 9 cells; largest \|hln_stat\| is 2.30 (array12 h6, p_holm=0.13) | `results/table6_dm_lagged.csv`, model_1/model_2={lstm,cnn_lstm} | 9, 12A | Results RQ2; T5, T6 | established (non-effect) |
| C6 | CNN-LSTM has higher seed variance than XGBoost in all 9 cells, and higher than LSTM in 7 of 9 | `results/seed_sweep_summary_lagged.csv`, std_skill_vs_convex, three-way compare | 9 | Results RQ2/RQ4; T3 | descriptive |
| C7 | Residual correction (out-of-fold, corrected scheme) on the LSTM base at array11 h6 seed0: skill_vs_convex +0.2110 (plain LSTM) -> +0.1768 (corrected residual), a -0.034 penalty | `results/lstm_array11_h6_lagged_seed0.json` vs `results/lstm_residual_array11_h6_lagged_seed0.json`, metrics.daylight.skill_vs_convex | 10 | Results RQ2; T4 (row 3) | descriptive (single seed) |
| C8 | lstm vs lstm_residual at array11 h6 is Holm-significant in the direction of plain LSTM (hln_stat -3.45, p_holm 0.0056) | `results/table6_dm_lagged.csv`, array=array11, horizon=6, model_1/model_2={lstm,lstm_residual} | 10, 12A | Results RQ2; T6 | established |
| C9 | Residual correction (default 3yr/2fold, lagged) is net negative in all 18 array x horizon cells (5-model sweep) | `results/seed_sweep_summary_lagged.csv`, mean_skill_vs_convex, model={lstm_residual,cnn_lstm_residual} vs base | 10, 11 | Results RQ2; T3 | descriptive (direction) |
| C10 | Under DM, lstm vs lstm_residual is Holm-significant in 5 of 9 cells (array11 h3/h6, array12 h1/h3, array17 h1), NOT in the other 4 (array11 h1, array12 h6, array17 h3/h6) - no clean h-based split | `results/table6_dm_lagged.csv`, model_1/model_2={lstm,lstm_residual}, all 9 rows | 11, 12A | Results RQ2; T6 | established (per-cell); the "clean split" reading is explicitly NOT supported |
| C11 | cnn_lstm vs cnn_lstm_residual is Holm-significant in ALL 9 array x horizon cells (p_holm 0.000147 to 0.0350) | `results/table6_dm_lagged.csv`, model_1/model_2={cnn_lstm,cnn_lstm_residual} | 12A | Results RQ2; T6 | established |
| C12 | 5-year/4-fold training window (arrays 11+12, 3 seeds) recovers 19-96% of the residual penalty vs the 3-year/2-fold default, but does not eliminate it at h=6 (recovery 43% array11, 19% array12) | `results/train5yr/` run JSONs, 36 files; recovery arithmetic against `results/seed_sweep_summary_lagged.csv`'s 3yr numbers | 11 | Results RQ2; T4 (row 4) | suggestive (3 seeds, never DM-tested - see Section 3) |
| C13 | Mechanism: out-of-fold predicted/actual residual correlation is +0.76 to +0.79, but the same correlation on the validation split is only +0.04-0.13 (3yr window) - the corrector is overconfident by 2.0-4.7x the break-even ratio | `scripts/diagnose_residual_signal.py` output (rho_oof, rho_val, sigma_p/sigma_r columns) | 11 | Results RQ2 / Methodology; text only, no table assigned | suggestive |
| C14 | Night-hour inclusion: nRMSE ratio (all-24h / daylight-only) is 0.658, matching the closed-form prediction sqrt(N_day/N_all)=0.658, at array11 h=3 | `results/table4_protocol_lagged.csv`, array11/h3, config C5 (nrmse, n=3762) vs C6 (nrmse, n=8694) | 2 | Intro; Methodology 4.1; T4 (row 1) | established (closed-form, verified) |
| C15 | Skill vs smart persistence is essentially unchanged by night-hour inclusion (0.5265 daylight-only vs 0.5279 all-24h, array11 h3) | `results/table4_protocol_lagged.csv`, array11/h3, config C2 (skill) vs C4 (skill) | 2 | Intro; T4 (row 1) | established |
| C16 | Reference-forecast choice: at array11 h=6, skill vs plain persistence is +0.652, skill vs the convex reference is +0.194 - roughly two-thirds of apparent skill is a baseline artifact | `results/reference_comparison.csv`, array11/h=6, xgb_skill_vs_persistence vs xgb_skill_vs_convex | 5 | Intro; T4 (row 2); F1 | established (single seed=0, all 3 arrays consistent - see Section 3) |
| C17 | The horizon trend flips sign of curvature depending on reference: monotonically increasing vs persistence (0.253/0.526/0.652 at h1/3/6, array11), non-monotonic and peaking at h=3 vs convex (0.200/0.276/0.194) | `results/reference_comparison.csv`, array11, all 3 horizons | 5 | Intro; T4 (row 2); F1 | established |
| C18 | Convex weight w is not uniform across arrays at h=6: array11 w=0.04, array12 w=0.05, array17 w=0.31 - persistence remains informative on array17 when it is nearly worthless on the other two | `results/reference_comparison.csv`, convex_weight column | 5 | Methodology 4.1; T4 | descriptive - see Section 5 for the exact-value trap |
| C19 | Residual fit-split leakage: fitting the residual stage on validation residuals (leaked_by_design) instead of out-of-fold training residuals turns a -0.034 penalty into an apparent +0.334 gain - a sign flip and an order of magnitude, on one seed/array/horizon | `results/leaked_lstm_residual_array11_h6_lagged_seed0.json` (INVALID_LEAKED=true) vs the corrected pair in C7 | 10 | Intro; T4 (row 3) | descriptive (single cell, explicitly not a general effect-size claim - see Section 3) |
| C20 | Oracle regime beats lagged regime on skill_vs_convex in all 45 model x array x horizon cells, gap +0.51 to +0.72, growing with horizon | `results/seed_sweep_summary_oracle.csv` vs `results/seed_sweep_summary_lagged.csv`, mean_skill_vs_convex, joined on model/array/horizon | (post-checkpoint; not in PROJECT_CHECKPOINT.md - see Section 9) | T4 (row 5); F2 (F7 was reassigned 2026-08-08 - see Section 7) | established (oracle is an upper bound, not itself a forecast result - see Section 4) |
| C21 | Oracle-regime protocol machinery is regime-agnostic: the closed-form night-inclusion check (C14's identity) also holds under oracle, within 0.001-0.028 across all 9 cells | `results/table4_protocol_oracle.csv`, same C5/C6 comparison as C14 | (post-checkpoint) | Methodology 4.1 (robustness check) | established |
| C22 | Sky-condition stratification: mean skill_vs_convex by class (one array x horizon cell, e.g. array11 h1) is NOT monotonic in visible cloudiness - partly_cloudy scores lowest (0.066 pooled mean), below overcast (0.235), below clear (0.332), with zero exceptions across the 9 cells checked | `results/table_sky.csv`, grouped by sky_class, skill_vs_convex column | 12B | Results RQ3; T7; F5 | established (direction; see Section 5 for the "pooled count" trap in the N reported) |
| C23 | Overcast is the rarest sky condition at this site: at h=6 (any array, all three identical), n=236 daylight hours out of ~3756 | `results/table_sky.csv`, sky_class=overcast, horizon=6 | 12B | Results RQ3; T7 | descriptive - see Section 3, n=236 caveat |
| C24 | RQ4 compute: mean fit time is 0.55s (xgboost), 10.11s (lstm), 12.23s (cnn_lstm), 22.73s (lstm_residual), 25.02s (cnn_lstm_residual), across all 45 lagged runs per model | Computed directly from `results/<model>_array*_h*_lagged_seed*.json`, timings.fit_seconds, this session - NOT copied from PROJECT_CHECKPOINT.md prose (see Section 5) | 8, 9 (numbers superseded - see Section 5) | Results RQ4; T3 | descriptive |
| C25 | LSTM costs ~18x XGBoost's fit time (10.11s vs 0.55s mean) for a gain that is not significant at h=1/h=3 and significant on only 1 of 3 arrays at h=6 | Same as C24, joined with C1/C2 | 8 | Results RQ4 | descriptive + established (the "not significant" part) |
| C26 | CNN-LSTM costs ~21% more fit time than plain LSTM (12.23s vs 10.11s mean) for a non-significant accuracy change (C5) | Same as C24, joined with C5 | 9 | Results RQ4 | descriptive + established (non-effect) |
| C27 | Literature survey: 27 papers coded, 25 with a verbatim-quoted audit file (`evidence_level=quoted`), 2 coded from pre-existing summary notes only (`evidence_level=summary_only`, no locatable source PDF) | `results/literature_survey.csv`, evidence_level column, 27 rows | (not in PROJECT_CHECKPOINT.md - see Section 9) | Related Work; T2 | descriptive |
| C28 | 26 of 27 surveyed papers report NO skill score against any reference forecast; the one exception (mayer2022) uses the exact Yang et al. (2020) convex-combination reference | `results/literature_survey.csv`, skill_score_reported column (no=26, yes=1) and baseline_used column (convex=1) | (existence proof for the paper's whole framing) | Intro; Related Work; T2 | descriptive |
| C29 | 27 of 27 surveyed papers report ZERO run-to-run (seed) variance | `results/literature_survey.csv`, variance_reported column (no=27) | | Intro; Related Work; T2 | descriptive |
| C30 | Only 1 of 27 surveyed papers (hussain2022) states code is available; 26 leave code_available not_stated | `results/literature_survey.csv`, code_available column | | Related Work; T2 | descriptive |
| C31 | 6 of 27 surveyed papers confirmed to use DKASC-family data (zhou2024, hou2024, ye2026, hussain2022, alharkan2023, guo2024); 1 more (vennila2022) resembles the DKASC technology mix but the site is never named and is explicitly coded unconfirmed | `results/literature_survey.csv`, dataset column, filtered for "DKASC"/"DKA " and manually checked against each row's notes | | Related Work; Data section (external validity) | descriptive - do not round to 7 |
| C32 | 3 of 27 surveyed papers have a documented leakage pattern, confirmed by a direct quote from the paper's own text: bhutta2024 (Performance Ratio, a target-derived quantity, used as 1 of 3 input features), li2022 (EEMD decomposition stated to run before the train/test split), zhou2024 (two DKASC target-derived columns used as input features, same columns this project's own loader.py drops) | `results/literature_survey.csv`, leakage_flag=documented (3 rows); `evidence/bhutta2024hcrnhcln_audit.md`, `evidence/li2022eemdssalstm_audit.md`, `evidence/zhou2024cnnlstmattnbayes_audit.md` | | Related Work; Intro motivation | descriptive |
| C33 | split_type is not_stated in 22 of 27 surveyed papers; only 3 explicitly state "chronological" | `results/literature_survey.csv`, split_type column | | Related Work; T2 | descriptive |
| C34 | The three DKASC arrays used in this study (11, 12, 17) share one weather station; data-audit coverage/gap/NaN statistics are byte-identical across arrays, confirming they are not independent samples | `results/data_audit.csv`, compare rows across array column for the same year | Data section (Section 2) | Data; Limitations | descriptive |
| C35 | array07 (CdTe) was excluded: 48-97% of clearly-daytime (GHI>200) records show exactly zero power across March-September 2014, and a completeness-based audit (coverage/NaN only) passed it at 99.99% coverage, 0.00% NaN | `results/data_audit.csv` (array07_CdTe, 2014 row: coverage/NaN) vs `results/dead_period_audit.csv` (array07, 2014 row: pct_zero_power_daylight, status) | 6 | Data section; Intro motivation | descriptive |
| C36 | array11's processed hourly dataset has exactly 61,344 rows, 2009-01-01 to 2015-12-31 (7 years, only 2012 a leap year) | `data/processed/array11_polySi_hourly.parquet`, row count and index min/max, read directly | Section 2 | Data section | descriptive (exact) |

---

## 3. WHAT WE MUST NOT CLAIM

Every item below cites the finding it comes from and the exact reason the
stronger claim is not supported. These are not stylistic cautions - each
one has already been drafted as an overclaim once in this project's own
history (see PROJECT_CHECKPOINT.md Section 8, "Record of Wrong
Predictions") and corrected.

1. **"Convolution is significantly worse than a plain recurrent model."**
   Not supported. Direction is consistent (worse in 8/9 cells) but under
   DM, lstm vs cnn_lstm is Holm-significant in NONE of the 9 cells; the
   largest test statistic is p_holm=0.13 (array12 h6). Report this as a
   consistent, non-significant directional cost - see Finding 9's own
   correction of its first draft, and Finding 12A. (C5 in Section 2.)

2. **"The LSTM's h=6 advantage over XGBoost is established."** Not
   supported as a general claim. It is Holm-significant on array17 only
   (hln_stat 3.26, p_holm 0.0114). array11 (hln_stat 2.56, p_holm 0.073)
   and array12 (hln_stat 2.64, p_holm 0.059) are NOT significant at
   alpha=0.05. This exact overclaim was made once already and is
   recorded as Record of Wrong Predictions #12 in PROJECT_CHECKPOINT.md -
   a seed-based two-sample statistic ("t~6") was substituted for DM and
   overstated the effect by roughly 2x. Do not use seed spread as a
   significance test anywhere in the paper; it measures training
   stochasticity, not sampling uncertainty over the evaluation period.
   (C2 in Section 2.)

3. **"Three arrays confirm X."** The three arrays (11, 12, 17) share one
   weather station (Section 2 of PROJECT_CHECKPOINT.md; confirmed via
   byte-identical data-audit statistics, `results/data_audit.csv`). They
   are not independent samples. A result that holds on all three arrays
   is one (correlated) piece of evidence, not three. This directly
   affects how claim #2 above should be worded: "significant on one of
   three co-located, non-independent arrays" is accurate; "confirmed
   across three sites" is not. (See also Section 4, wording constraints.)

4. **"The residual stage is architecturally harmful."** Overstated.
   Finding 11 shows the penalty measured under the default 3-year/2-fold
   training window shrinks by 19-96% (depending on array/horizon) when
   the window is extended to 5 years/4 folds, and the mechanism
   (out-of-fold vs validation residual correlation, `scripts/
   diagnose_residual_signal.py`) shows the corrector is overconfident by
   a factor that itself shrinks with more folds (4.7x -> 2.9x at h=6,
   2.0x -> 1.2x at h=3). Some of the measured penalty is a fold-count
   artifact of the expanding-window construction, not a pure
   architecture verdict. The correct claim is: "under the default 3-year
   window, the correction stage measurably hurts; part but not all of
   that penalty is attributable to having only 2 out-of-fold training
   periods rather than a genuine architectural limitation." Also: the
   5-year/4-fold sensitivity run (36 JSONs, 3 seeds, arrays 11/12 only)
   was NEVER run through the Diebold-Mariano pipeline -
   `results/table6_dm_lagged.csv` has no 5-year-window rows at all. Its
   "43%/19% recovery, still significant at h=6" reading rests on a
   3-seed standard-error calculation, the same category of statistic
   Finding 8 had to retract for the architecture claim above - not an
   autocorrelation-aware test. Do not describe the 5-year result as
   DM-confirmed. (C10, C12 in Section 2.)

5. **"The overcast-vs-partly-cloudy skill gap is a robust site-wide
   result."** The direction (partly_cloudy worse than overcast in all 9
   cells) is real and consistent, but overcast is the rarest sky
   condition at this site: n=236 daylight hours at h=6 (any array - all
   three are numerically identical because sky classification depends
   only on the shared weather-station GHI reading, verified directly:
   array11/array12/array17 all report n=239 at h=1 and n=236 at h=6 for
   the overcast class, `results/table_sky.csv`). A mean skill score over
   236 points is noisier than one over the ~2200 clear-sky points at the
   same cell, and no significance test (DM or otherwise) has been run
   ON THE SKY-STRATIFIED SUBSETS - `results/table6_dm_lagged.csv` is
   computed over the full daylight set, not per sky class. State the
   result as a consistent, striking pattern across 9 cells with zero
   exceptions, backed by a mechanism (temporal stability of overcast
   skies favoring the convex reference; irradiance ramps under broken
   cloud defeating lagged features) - not as a statistically tested
   difference. (C22, C23 in Section 2.)

6. **"The 5-year fold ablation confirms a short-horizon-recovers /
   long-horizon-persists split."** Duplicated from item 4 above because
   it is a distinct and separately tempting overclaim: Finding 11 itself
   originally read the recovery-percentage table (75%/60%/43% for
   array11 h1/h3/h6; 96%/82%/19% for array12) as confirming a clean
   split, then retracted that reading after checking DM on the config
   that IS in `results/table6_dm_lagged.csv` (the 3-year/2-fold default):
   lstm vs lstm_residual is significant at array12 h1 (short horizon,
   should have "recovered" per the pattern) and NOT significant at
   array12 h6 (long horizon, should have "persisted"). The mechanism
   (fold-count -> overconfidence) is real and quantified; the clean
   horizon-based split it was expected to produce is not confirmed by
   any significance test that has actually been run. (C10 in Section 2.)

**Also flag, not explicitly requested but directly adjacent:**

7. Table 4 and Table 6 (the protocol-inflation and DM-significance
   tables) are computed at a SINGLE seed (`SEED = 0`, hardcoded in both
   `scripts/build_table4_protocol.py` and `scripts/build_table6_dm.py`).
   This is a deliberate design choice - DM's significance test is over
   the ~3700+ paired daylight-hour forecast errors within one model
   instance, not across seeds - but it means Table 4's specific
   percentage figures (e.g. the +0.652 -> +0.194 reference-choice swing)
   are single-run numbers, not 5-seed means like Table 3/5. Do not
   describe Table 4's numbers as "averaged over seeds" anywhere.

8. The residual-fit-split leakage sign flip (C19: -0.034 -> +0.334) is
   demonstrated on exactly one seed, one array (array11), one horizon
   (h=6). PROJECT_CHECKPOINT.md's own Finding 10 text says this
   explicitly: "adequate for demonstrating that the leak is possible and
   large, but it is not a general effect-size claim." The leaked variant
   was never run through the DM pipeline at all -
   `results/table6_dm_lagged.csv` has no leaked-model column. Report the
   number as a demonstration case, not a generalizable effect size.

---

## 4. WORDING CONSTRAINTS

Fixed phrasings. Deviating from these has already produced an incorrect
draft claim once each in this project (see Section 3 above and
PROJECT_CHECKPOINT.md Section 8).

- **"arrays," never "sites."** All three evaluated PV systems (11, 12,
  17) are co-located at one DKASC site (Alice Springs) and share one
  weather station. "Site" implies independence they do not have.
- **State explicitly, every time correlated-array results are reported,
  that the three arrays share one weather station and are therefore not
  independent samples.** Do not let "on all three arrays" stand alone as
  if it were three confirmations.
- **The oracle regime is a perfect-forecast UPPER BOUND, never an
  achievable result.** It uses measured weather AT TARGET TIME
  (`oracle_`-prefixed features, CLAUDE.md rule 5). Every mention of an
  oracle-regime number must be adjacent to this qualifier - do not let a
  reader encounter an oracle skill number without the upper-bound label
  in the same sentence or table caption.
- **Skill vs the convex reference is the headline metric.** Skill vs
  plain persistence is reported alongside, because the GAP between the
  two is itself a result (C16/C17). Never report skill vs persistence
  alone as if it were the paper's primary claim.
- **"Documented leakage" means a direct quote from the paper's own text
  states the leaky procedure** (e.g. li2022's stated pre-split EEMD
  decomposition; bhutta2024's stated 3-feature input list including
  Performance Ratio). It does NOT mean inferred from silence about
  ordering or from a plausible-but-unconfirmed reading. The literature
  survey's `leakage_flag=suspected` value exists for the inferred case
  and must not be conflated with `documented` in prose.
- **DKASC-family paper counts in the survey: state "6 confirmed, 1
  unconfirmed resemblance" (C31), never round to 7.** vennila2022's
  dataset field is explicitly coded "SITE NEVER NAMED (resembles DKASC
  tech mix, not confirmed)" - it must not be silently added to the
  confirmed count.
- **Table 4 and Table 6 report single-seed (seed=0) results; Table 3 and
  Table 5 report 5-seed means with std.** State which kind of number is
  being quoted whenever a specific figure from either pair is cited (see
  Section 3, item 7).
- **Report convex weight w per array, never as a single pooled range.**
  At h=6, w is 0.04 (array11), 0.05 (array12), but 0.31 (array17) - a
  materially different regime, not noise (see Section 5).
- **"Protocol inflation," not "our method is more honest."** The paper's
  claim is about the SIZE of the gap between defensible-looking choices,
  not that any other paper's authors were careless. Several of the
  surveyed papers are transparent about their own limitations even while
  not reporting a skill score (e.g. energyeng2025's stated feature
  engineering pipeline, or ma2024's explicit sky-condition
  stratification, per `paper/literature_notes.md` sections 5-7 and
  `evidence/`). Keep the framing on the protocol, not the authors.

---

## 5. NUMBERS THAT ARE EASY TO GET WRONG

Each of these has already been miscounted, misattributed, or stated
imprecisely once in this project's own history, or was found to be
imprecise while building this brief. Correct value and the trap, both
stated.

1. **61,344, not 61,368.** Processed row count per array, 2009-2015
   hourly. The trap: 7 years x 365 days x 24 hours = 61,320; adding one
   leap day gives 61,344 (only 2012 is a leap year in this window,
   +24 hours). PROJECT_CHECKPOINT.md's own "Record of Wrong Predictions"
   #3 records predicting 61,368 - one leap day too many - before
   checking. Verified directly against `data/processed/
   array11_polySi_hourly.parquet` (61,344 rows exactly) for this brief.

2. **225 vs 90, not "the 225-run sweep is the residual-stage evidence."**
   One execution of `scripts/run_seed_sweep.py` (lagged regime) produces
   225 run JSONs: 5 models x 3 arrays x 3 horizons x 5 seeds. The
   residual-stage finding (Finding 10/11) uses only the
   lstm_residual/cnn_lstm_residual PORTION of that grid: 2 models x 3
   arrays x 3 horizons x 5 seeds = 90 runs, across 18 array x horizon
   cells. The architecture-comparison finding (8/9) uses the
   xgboost/lstm/cnn_lstm portion: 135 runs. PROJECT_CHECKPOINT.md Section
   6 itself records this exact conflation as a found-and-fixed staleness
   item ("STALE ITEMS FOUND" #1) - the original text called 225 "the
   sweep for lstm_residual and cnn_lstm_residual," which is the size of
   the whole grid, not the residual-only portion. Verified by file count:
   exactly 45 JSONs per model name under `results/*_lagged_seed*.json`,
   225 total.

3. **0.658, traced to one specific cell, not a single site-wide
   constant.** The closed-form night-inclusion ratio
   nRMSE(24h)/nRMSE(daylight) = sqrt(N_day/N_all) evaluates to 0.6581
   (observed) vs 0.6578 (analytic) specifically at array11, h=3, in
   `results/table4_protocol_lagged.csv` (config C5 n=3762, config C6
   n=8694). The value is close across all 9 array x horizon cells
   (0.6573 to 0.6604 observed, verified directly for this brief) but is
   not IDENTICAL across them - N_day and N_all both vary slightly by
   horizon (feature-coverage effects, Finding 3) and very slightly by
   array. Do not present "0.658" as if it were an exact universal
   constant; present it as the array11/h=3 instance of a relationship
   that holds within about 0.5% across all 9 cells, and give the
   sqrt(N_day/N_all) formula so a reviewer can recompute any cell.

4. **Convex weight w at h=6: 0.04 and 0.05, but 0.31 on array17 - not a
   single "0.01-0.05" range.** Verified directly against `results/
   reference_comparison.csv`: array11 w=0.04, array12 w=0.05, array17
   w=0.31 at h=6. array17's persistence remains informative at h=6 when
   it is nearly worthless on the other two arrays - Finding 5 attributes
   this (tentatively, "not diagnosed further") to HIT's better
   temperature coefficient reducing thermally-driven scatter in k_p.
   Stating a single low range for all three arrays would silently drop
   array17's genuinely different behavior, which PROJECT_CHECKPOINT.md's
   Finding 5 explicitly calls out as NOT the array07 pathology and worth
   exactly one sentence, not zero.

5. **C3 in Table 4 is "convex-covered hours," not a 24-hour protocol
   variant.** `scripts/build_table4_protocol.py`'s own docstring and
   in-code comment state this directly: C3 restricts to whatever hours
   the Climatology/convex-reference model actually produced a prediction
   for, which excludes always-night (month, hour) cells the training
   data never saw as daylight - it is NOT a night-inclusive config. It is
   kept in the table only to document that coverage gap via n_samples
   (its skill is within 0.0002 of C1's at h=3 and h=6), not to be read as
   "skill with night included." The only config in Table 4 that genuinely
   spans a 24-hour cycle is C4 (all 24 hours, skill vs smart persistence,
   n=8694 or 8682 depending on horizon) - because SmartPersistence
   forward-fills through the night and therefore has a prediction at
   every hour, unlike Climatology.

6. **The literature-survey filename and row counts have moved since
   PROJECT_CHECKPOINT.md's Section 6 was last written (2026-08-03).**
   Section 6 there states "3 coded rows... ~27 papers remain to code."
   The actual current state, verified directly against `results/
   literature_survey.csv` for this brief: 27 rows total, 25 with a
   verbatim-quoted audit file. PROJECT_CHECKPOINT.md is stale on this
   specific point as of this brief's writing - see Section 9 (GAPS).

7. **RQ4 compute figures in PROJECT_CHECKPOINT.md's prose (Findings 8/9)
   do not match any aggregation this brief could reproduce from the
   committed run JSONs - use the recomputed figures in this brief's
   Section 2 (C24) instead.** Finding 8 states "LSTM fit ~16 s vs
   XGBoost sub-second, roughly 30:1." Finding 9 states "fit times ~20-31
   s vs ~14-27 s for the LSTM" (cnn_lstm vs lstm). Recomputing directly
   from all 45 `results/lstm_array*_h*_lagged_seed*.json` and the
   equivalent xgboost/cnn_lstm files for this brief gives: xgboost mean
   0.55s (range 0.43-0.92s), lstm mean 10.11s (range 5.86-18.30s),
   cnn_lstm mean 12.23s (range 6.74-20.88s) - a ratio of about 18:1
   (lstm/xgboost), not 30:1, and ranges that do not match "14-27s" or
   "20-31s" under any per-horizon breakdown checked (overall, or split
   by h=1/h=3/h=6). This is not a large error in kind - LSTM is still an
   order of magnitude slower than XGBoost, and CNN-LSTM is still
   modestly slower than LSTM - but the specific figures in the prose
   should not be copied into the paper without using the recomputed
   numbers above, which are directly reproducible from the committed
   JSONs.

8. **The sky-stratification "pooled daylight-hour counts" in Finding 12B
   (60,048 / 33,021 / 6,426) triple- and 9x-count the same clock-hours.**
   Verified directly against `results/table_sky.csv` for this brief: the
   n column is IDENTICAL across all 3 models (xgboost/lstm/lstm_residual)
   for the same array+horizon+sky_class cell (e.g. array11 h1 clear:
   2229, 2229, 2229), and also IDENTICAL across all 3 arrays for the same
   horizon+sky_class (h1 clear: array11/12/17 all 2229) - because sky
   classification depends only on the shared weather-station GHI signal,
   not on per-array data. Summing the raw n column over all 81 rows (3
   models x 3 arrays x 3 horizons) therefore counts each distinct set of
   clock-hours roughly 9 times (60,048 / 3 models / ~3 = ~2229x9, modulo
   small per-horizon coverage differences). The minimally-duplicated
   citable number is ONE array's ONE horizon's three class counts, e.g.
   array11 h=1: clear 2229, partly_cloudy 1223, overcast 239. No
   correctly-deduplicated multi-cell "pooled" total has been computed
   anywhere in the repo as of this brief - see Section 9 (GAPS). Do not
   copy 60,048/33,021/6,426 into the paper as a real headcount.

---

## 6. SECTION-BY-SECTION OUTLINE

8 sections, IEEE two-column, target ~8 pages total (per
PROJECT_CHECKPOINT.md Section 7). Column-length estimates are rough
budgets, not measured.

### 1. Introduction (~0.75 column)
Argues: this literature under-reports evaluation protocol (C27-C33); a
protocol-controlled study is needed to separate architecture effects from
protocol effects; this paper is that study, on DKASC data, with a
specific, closed-form demonstration (night-hour inclusion, C14) that a
reviewer can check independently. Claims used: C14, C16, C19 (as a
teaser, full development in Results), C27-C29 (existence-proof framing;
cite mayer2022 explicitly per Section 8). No tables/figures of its own;
may forward-reference F1 and T4.

### 2. Related Work (~1 column)
Argues: (a) the general-ML leakage/evaluation-pitfalls literature (Kapoor
& Narayanan; Hewamalage, Ackermann & Bergmeir) already documents these
failure modes outside solar; (b) the solar-forecasting verification
literature (Yang et al. 2020) already recommends the fix and documents
the low-elevation instability this project independently rediscovered;
(c) the surveyed hybrid-PV literature (27 papers) overwhelmingly does not
follow either. Claims: C27-C33, all of Section 8's citation plan. Table:
T2 (survey summary). No figures.

### 3. Data and Preprocessing (~1 column)
Argues: DKASC array selection and exclusion (array07 dropped, C35),
co-located-arrays-not-sites framing (C34), chronological split
rationale, daylight/oracle regime definitions. Claims: C34, C35, C36.
Table: T1 (data summary - array technology, capacity, years, row counts).
No figures (or a small site/array map if available - not currently
generated, see Section 9).

### 4. Methodology (~1.5 columns, two subsections, 4.1 BEFORE 4.2 -
deliberate ordering per PROJECT_CHECKPOINT.md Section 7)

**4.1 Evaluation protocol.** Present as a filled-in "model info sheet"
per Kapoor & Narayanan's L1-L3 taxonomy (cite their labels explicitly;
see the mapping table already in PROJECT_CHECKPOINT.md Section 5,
Finding 3's L1.2/L1.3 mapping and Finding 10's L1.1 mapping). Covers:
chronological splits, daylight filtering and the 10-degree elevation
threshold (cite Yang et al.'s 85-degree convention and state this
project's threshold is stricter), scaler/statistics fit-on-train-only,
the convex-reference skill score (cite Yang et al. 2020 directly, Eq.
form), lagged/oracle regime definition (oracle = upper bound, wording
constraint above), the out-of-fold residual-stage construction (Finding
10's fix), and the single-test-touch rule. This is where Finding 1
(clear-sky sampling mismatch) belongs as a worked example of a
protocol-construction bug.

**4.2 Model architectures.** SmartPersistence, Climatology,
ConvexCombination, XGBoost, LSTM, CNN-LSTM, residual-corrected variants.
Brief - the paper's stance is that architecture is not the contribution.
Claims used in 4.1: C14, C15, C18 (w reporting convention), C7-C11
(residual construction, forward reference to Results). No new
tables/figures; T4 and T6 are introduced conceptually here, populated in
Results.

### 5. Experimental Setup (~0.5 column)
Environment, seeds (5 per cell, why - torch's nondeterministic CUDA LSTM
backward pass, PROJECT_CHECKPOINT.md Section 1), the 225-combo lagged
grid and its 45-combo oracle counterpart, DM test configuration (HAC
variance, Bartlett kernel truncated at lag h-1, HLN small-sample
correction, Holm-Bonferroni within cell), sky-classification thresholds
(0.75/0.10/0.40, literature-conventional, not fitted). No claims table
rows of its own; sets up Results.

### 6. Results, organized BY RESEARCH QUESTION, not by model (~2.5
columns - the paper's core)

**6.1 RQ1 Protocol sensitivity (HEADLINE).** RESOLVED 2026-08-08 (was an
open ordering question as of this brief's original writing - see gap 9
below): PROJECT_CHECKPOINT.md Section 0 was renumbered so RQ1 IS
protocol sensitivity, matching both "present protocol first" and literal
RQ numeric order - Results now follows RQ1/RQ2/RQ3/RQ4 straight through
with no ordering exception to explain to a reader. Table 4, all 5 rows
(C14, C16-C19, C20). Figure F1 (reference-choice shape, C16/C17) and
Figure F2 (skill by model and feature regime, lagged vs oracle, C20) -
both built 2026-08-08, see paper/figures/CAPTIONS.md for captions.
NOTE: the originally-proposed F2 (a waterfall/bar of all 5 Table 4
effect sizes) was superseded by this pairing during actual figure
construction and was never built - Table 4 itself already tabulates
all 5 effect sizes, so a combined waterfall is a nice-to-have, not a
gap; if still wanted it needs a new figure number, F1-F7 are now all
assigned to other content (see Section 7). This is where the sign-flip
(C19) and the reference-choice shape change (C17) get full prose
treatment; both are the paper's strongest single results.

**6.2 RQ2 Component attribution.** C1-C11 (architecture and residual
findings), Table 3 (seed reproducibility), Table 5 (component-attribution
summary), Table 6 (DM matrix, or a condensed version - 189 rows is too
large to print in full; likely a per-model-pair summary with the full
matrix as supplementary data). Figure F4 (residual distribution before
and after the residual stage, array11 h=6, built 2026-08-08 - visualizes
Finding 11's overconfidence mechanism directly, see paper/figures/
CAPTIONS.md). NOTE: the originally-proposed F3 (skill by model x
horizon, boxed over 5 seeds) and F4 (DM significance heatmap) were both
superseded and never built - Table 3 and the condensed Table 6 already
carry that information; if a dedicated visual is still wanted either
needs a new figure number. Full "must not claim" cautions from Section
3 apply throughout this subsection especially.

**6.3 RQ3 Conditional performance.** C22, C23. Table 7. Figure F3
(error and skill by sky condition, array11 h=3 only, built 2026-08-08 -
state that single-cell restriction explicitly, see paper/figures/
CAPTIONS.md) covers this for one cell. NOTE: the originally-proposed F5
(skill by sky class, grouped by horizon, all 9 cells) was superseded -
F5 was reassigned to feature importance instead - and was never built;
if the full multi-cell version is wanted it needs a new figure number
and Table 7's own condensation script (see Section 7, T7 row). State
the n=236 caveat (Section 3, item 5) directly in this subsection, not
just in Limitations.

**6.4 RQ4 Cost-effectiveness.** C24-C26. Reuses Table 3/5's timing
columns if added, or a small dedicated table (not yet in the T1-T7 plan -
see Section 7). RESOLVED 2026-08-08 (was "no dedicated figure planned"
as of this brief's original writing): Figure F6 (skill vs mean training
compute, log scale, array11 h=6, one marker per model) is exactly the
compute-vs-skill scatter this subsection flagged as a candidate - built,
see paper/figures/CAPTIONS.md.

### 7. Limitations (~0.5-0.75 column)
Must include, per Section 3 and the consistency-principle note from
Mayer (2022, entry 9 in `paper/literature_notes.md`): the three-arrays-
one-weather-station non-independence (C34); the residual-penalty
fold-construction confound (item 4, Section 3); the un-DM-tested 5-year
ablation and sky-stratified subsets (items 4/5/6, Section 3); the
low-elevation clear-sky residual bias Finding 1 deliberately did not fix
(twilight diffuse, pyranometer thermal offset); the MSE-vs-MAE
consistency-principle axis this project did not vary (train on MSE loss,
report RMSE-based skill - internally consistent but a protocol axis left
unexplored, per Mayer 2022 Section 2.6); the test set (2015) has not yet
been touched for a forecast metric at all (see Section 9, GAPS) - if this
remains true at submission time, Limitations must say so explicitly
rather than the paper silently reporting only validation-split results
throughout.

### 8. Conclusion (~0.25 column)
Restate the five-sentence argument (Section 1). No new claims - every
sentence here should trace to a claim already in Section 2's table.

---

## 7. TABLES AND FIGURES

Only Tables 3, 4, 5, and 6 have an attested number anywhere in the repo
(named directly in `scripts/build_table4_protocol.py`'s and
`scripts/build_table6_dm.py`'s own docstrings, and in
`scripts/aggregate_seed_sweep.py`'s docstring for Table 3; Table 5 is
named only in `PROJECT_CHECKPOINT.md` prose, no script docstring uses the
number). T1, T2, T5, and T7 have NO defined content or generation
script anywhere in the repo as of this brief - the assignments for
those four are still PROPOSED, not sourced. This is flagged again in
Section 9 (GAPS).

FIGURES UPDATED 2026-08-08: F1-F6 are now built (`scripts/
build_figures.py`, captions in `paper/figures/CAPTIONS.md`) and F7 is
written as a TikZ source file (`paper/figures/F7_pipeline.tex`, not yet
compile-tested - no LaTeX toolchain in this dev environment). Their
ACTUAL content was decided during construction, not from this table's
original proposals, and diverged from several of them - F1 matches its
original proposal; F2 through F7 do not. Four originally-proposed ideas
were superseded and never built: the protocol-inflation waterfall
(original F2), the DM significance heatmap (original F4), the boxed
skill-by-model-x-horizon reproducibility plot (original F3), and the
leaked-vs-corrected feature-importance comparison (original F6). None
of the four is a blocking gap - Table 4 already tabulates the 5 effect
sizes the waterfall would show, Table 6 already carries DM significance
in tabular form, Table 3 already carries the seed-reproducibility
numbers, and F4 (residual distribution) makes a closely related
mechanism point to the leaked-vs-corrected comparison. If any is still
wanted for the paper, it needs a new figure number - F1-F7 are now all
assigned to other content, see the table below and paper/figures/
CAPTIONS.md for the actual caption of each.

| ID | What it shows | Generating file | Exists? | Still needed |
|---|---|---|---|---|
| T1 (proposed) | Data summary: array technology/capacity/years, row counts, exclusions | none | NO | A small script or manual table from `results/data_audit.csv` + Section 2 facts (C34-C36) |
| T2 (proposed) | Literature survey summary: per-column counts (C27-C33) | none | NO | A small aggregation script over `results/literature_survey.csv`, or a manually built table - the counts themselves are all in Section 2 (C27-C33) and directly queryable |
| T3 (attested: "Table 3" in code docstrings) | Seed-variance reproducibility: mean/std skill_vs_convex, skill_vs_persistence, RMSE per model x array x horizon | `scripts/aggregate_seed_sweep.py` | YES | `results/seed_sweep_summary_lagged.csv` (45 rows) and `results/seed_sweep_summary_oracle.csv` (45 rows) both exist and are ready to typeset; needs a LaTeX table generator (none exists yet) |
| T4 (attested: "Table 4" in code docstring + prose) | Protocol inflation, 6 configs x 3 arrays x 3 horizons, both regimes | `scripts/build_table4_protocol.py` | YES | `results/table4_protocol_lagged.csv` (54 rows) and `results/table4_protocol_oracle.csv` (54 rows) both exist. Needs: a LaTeX generator, AND the 5-row "effect summary" framing (C14-C20) is currently prose in this brief and PROJECT_CHECKPOINT.md, not a generated sub-table |
| T5 (attested: "Table 5" in prose only) | RQ2 component-attribution summary (condensed view of the architecture comparison) | none dedicated - would be built from `results/seed_sweep_summary_lagged.csv` filtered to the 3 base models plus a DM-significance annotation column | NO | A script joining seed_sweep_summary and table6_dm_lagged into one compact table; does not exist |
| T6 (attested: "Table 6" in code docstring + prose) | DM significance matrix, all 21 pairs x 3 arrays x 3 horizons | `scripts/build_table6_dm.py` | YES (both regimes) | `results/table6_dm_lagged.csv` (189 rows) and `results/table6_dm_oracle.csv` (189 rows) both exist as of 2026-08-08 - see PROJECT_CHECKPOINT.md's 2026-08-08 sync. 189 rows is too large to print in full in an 8-page paper; needs a condensation script (e.g. per-model-pair summary across cells, full matrix as supplementary CSV) |
| T7 (proposed) | RQ3 sky-condition stratification: skill_vs_convex by class x model x array x horizon | `scripts/build_table_sky.py` | YES (data), NO (as a formatted table) | `results/table_sky.csv` (81 rows) exists and is ready to aggregate; needs a condensation script (likely mean skill by class x horizon, pooled correctly across the 3 arrays given they are identical - see Section 5 item 8's duplication warning) and a LaTeX generator. F3 (below) previews one cell of this same data. |
| F1 (built 2026-08-08) | Skill vs horizon, two lines (vs persistence, vs convex), XGBoost only, one panel per array - visualizes C16/C17's monotonic-vs-non-monotonic shape result | `scripts/build_figures.py` (`build_f1`) | YES | Matches its original proposal. `paper/figures/F1_skill_vs_horizon.pdf`; caption in `paper/figures/CAPTIONS.md` |
| F2 (built 2026-08-08) | Skill vs convex reference by model and by feature regime (lagged vs oracle), array11, three horizons - visualizes C20 (the lagged-to-oracle gap dwarfs any model-to-model difference) | `scripts/build_figures.py` (`build_f2`) | YES | Does NOT match the original proposal (a protocol-inflation waterfall of 5 Table 4 effect sizes) - that idea was superseded and never built, see the note above this table. `paper/figures/F2_skill_by_model_regime.pdf` |
| F3 (built 2026-08-08) | nRMSE and skill_vs_convex by sky class, three models (XGBoost/LSTM/LSTM+residual), array11 h=3 ONLY - visualizes C22 (nRMSE and skill rank sky classes differently) | `scripts/build_figures.py` (`build_f3`) | YES | Does NOT match the original proposal (skill_vs_convex by model x horizon, boxed over 5 seeds, all arrays) - that idea was superseded and never built. Also distinct from the originally-proposed F5 (sky class grouped by horizon, all 9 cells) - F3 previews one cell of that. `paper/figures/F3_error_by_sky_condition.pdf` |
| F4 (built 2026-08-08) | Residual distribution (histogram) before and after the residual-correction stage, LSTM base, array11 h=6 seed 0 - visualizes Finding 11's overconfidence mechanism directly | `scripts/build_figures.py` (`build_f4`), data from `scripts/compute_f4_residuals.py` (a real refit - results/f4_residuals_array11_h6_lagged_seed0.csv) | YES | Does NOT match the original proposal (a DM significance heatmap) - that idea was superseded and never built. `paper/figures/F4_residual_distribution.pdf` |
| F5 (built 2026-08-08) | XGBoost top-10 feature importance (gain), lagged vs oracle, array11 h=3 seed 0 - visualizes that one oracle feature dominates while no lagged feature does | `scripts/build_figures.py` (`build_f5`) | YES | Does NOT match the original proposal (skill by sky class grouped by horizon) - that idea was superseded and never built, see F3's row above. `paper/figures/F5_feature_importance.pdf` |
| F6 (built 2026-08-08) | Skill_vs_convex vs mean training compute (fit_seconds, log scale), one marker per model, array11 h=6 - visualizes RQ4 (XGBoost reaches comparable skill at ~1/18th LSTM's compute) | `scripts/build_figures.py` (`build_f6`) | YES | Does NOT match the original proposal (leaked-vs-corrected feature importance) - that idea was superseded and never built. This IS the "compute-vs-skill scatter" the Section 6 outline flagged as a candidate for F6/F7 - that note is now resolved. `paper/figures/F6_skill_vs_compute.pdf` |
| F7 (redefined 2026-08-08) | The evaluation pipeline: raw DKASC data through to skill scores, with protocol decision points marked (daylight filter, chronological split boundaries, lagged/oracle regime fork, clear-sky reference chain, the three reference forecasts) | `paper/figures/F7_pipeline.tex` (TikZ, hand-written, NOT compile-tested - no LaTeX toolchain in this dev environment) | PARTIAL (TikZ source exists, not verified to compile) | The originally-proposed F7 (oracle vs lagged skill gap by horizon) was dropped as redundant with the built F2 above and reassigned to this pipeline/protocol diagram, which nothing else in the figure set covers and which the paper needs since protocol is its actual contribution. Plain-text ASCII sketch of the same structure in `paper/figures/CAPTIONS.md`, reviewable without a LaTeX build. Needs: a real compile-and-check pass (node spacing is estimated by hand, not visually verified) before camera-ready. |

---

## 8. CITATION PLAN

Only papers present in `paper/literature_notes.md` (10 numbered entries)
or `results/literature_survey.csv` (27 coded rows). No other citation is
invented for this brief.

### Tier 1 (methodology/framework papers, `literature_notes.md` entries 1-4, 9, 10)

| Paper | Cited where | For what specific claim |
|---|---|---|
| Yang et al. (2020), Solar Energy 210:20-37 | Methodology 4.1 (skill score definition, Eq. form); Methodology 4.1 (zenith/elevation filter, cite their 85-degree convention against this project's stricter 80-degree/10-degree cutoff); Intro (Finding 1's independent rediscovery of the low-elevation clear-sky instability, state explicitly this was found empirically BEFORE reading the paper); Limitations (nRMSE cross-scenario-comparability warning, and note this project normalizes by nameplate capacity, not mean observed power, avoiding their named Blaga et al. 2019 failure specifically, but the broader warning still applies) | Defines the headline metric (convex-combination skill score); documents the elevation-filter convention; motivates reporting skill as primary over nRMSE |
| Kapoor & Narayanan (2023), Patterns 4:100804 | Methodology 4.1 (present the entire protocol as a filled-in "model info sheet" using their L1-L3 labels, per the mapping table in PROJECT_CHECKPOINT.md Section 5); Related Work (294-paper, 17-field leakage prevalence; the civil-war case study as structural precedent - "if plain XGBoost matches the deep hybrid under a controlled protocol, that is not a null result, it is the same finding as theirs in a new domain") | Provides the leakage taxonomy this paper's protocol is built against, and the framing device for 4.1 |
| Hewamalage, Ackermann & Bergmeir, arXiv 2203.10716 | Related Work only (one paragraph) | Broadens the evaluation-pitfalls claim beyond solar specifically - general-ML source making the same argument about random CV splitting and misused error measures |
| Nguyen & Musgens, arXiv 2208.10536 (`literature_notes.md` entry 2 - DO NOT CONFUSE with entry 10, different paper, same first two authors) | Methodology (Section 2, horizon-separate reporting - "the analysis of solar forecasts should be done separately for each horizon"); Data section (Koppen-Geiger arid-zone/BWh external-validity note); Discussion (their "ensemble-hybrid is best" conclusion, RQ4 framing - state whether this project's results support or contradict it under a controlled protocol); Limitations or a footnote (Finding 7 is evidence AGAINST their training-length-up-to-2000-days claim at this site/horizon - cite and contrast) | Meta-analytic support for per-horizon reporting; contrast case for the training-length finding; motivates RQ4's framing |
| Mayer (2022), Renewable and Sustainable Energy Reviews 168:112772 (`literature_notes.md` entry 9) | Introduction and Related Work (the ONE paper in the 27-paper survey meeting the evaluation standard - existence proof, prevents the survey reading as a strawman); Methodology 4.1 (his exact convex-combination reference and zenith<90-degree daylight filter, cite as the one instance of this literature using the same construction this project uses); Limitations (his Section 2.6 consistency-principle result - different error metrics minimized by different functionals, changes conclusions when the whole study is rerun under a different loss - cite as a protocol axis this project did NOT vary, since it trains on MSE and reports RMSE-based skill); Discussion (his modest, honestly-reported 5.2%/10.4% MAE reductions against a real baseline, contrasted with the 30-60% improvements the surveyed hybrid papers report against their own components) | The paper's single positive existence-proof citation; also the source of the NWP-forecast "middle regime" framing (his weather inputs are neither this project's lagged nor oracle regime - state this explicitly, wording constraint in Section 4) |
| Nguyen & Musgens (2021), "What drives the accuracy of PV output forecasts?", arXiv 2111.02092 (`literature_notes.md` entry 10 - NOT the same paper as entry 2 above) | Introduction (their "cherry picking" finding - test-set length negatively correlated with reported accuracy - and the one-year test-set justification, this project's test set is exactly one calendar year); Related Work (scale of the literature - 180 papers meta-analyzed, 13 prior narrative surveys with no statistical analysis; and the notable absence that "skill score" does not appear anywhere in a 180-paper PV-forecast-error meta-analysis, stated neutrally, one sentence); Methodology (test-set-length justification, cite their explicit "has not been addressed in any previous work" statement) | Independent literature-level support for both the one-year test-set choice and the paper's central "evaluation practice varies enough to swamp real differences" argument; their headline ("hybrid models... will most likely be the future") should be contrasted carefully against the fact that it aggregates reported errors from the same 27-paper-type literature this project's own survey found reports no reference forecast in 26/27 cases |

### Tier 2 (the 27 surveyed hybrid/architecture papers, `results/literature_survey.csv` + `evidence/*.md`)

Not cited individually by name in most of the paper - they are the
DENOMINATOR for the survey counts (C27-C33), cited collectively in
Related Work with the aggregate statistics, and the CSV/evidence files
as the supporting artifact. Individual call-outs, where a specific paper
illustrates a specific point (all already have `paper/literature_notes.md`
entries 5-7 with citable detail, or an `evidence/*.md` audit file with
verbatim quotes for the rest):

- **abumohsen2024cnnlstmrf** (`literature_notes.md` #5): no skill score,
  no persistence baseline despite mentioning persistence as "basic yet
  essential" in its own introduction - good one-sentence example of the
  gap between describing best practice and following it.
- **xu2025lstmxgboosteemdso** (`literature_notes.md` #6): EEMD +
  Pearson feature selection with no stated train/test boundary - example
  of the "reporting problem" framing (do not assert leakage, note the
  paper does not state where the split occurs - it is `evidence_level=
  summary_only` in the survey, no verbatim-quote audit exists).
- **li2022eemdssalstm** (`literature_notes.md` #7, `evidence/
  li2022eemdssalstm_audit.md`): the one paper with `leakage_flag=
  documented` via a direct quote that decomposition runs before the
  split - the strongest citable leakage example in the survey.
- **bhutta2024hcrnhcln**, **zhou2024cnnlstmattnbayes**
  (`evidence/*.md`): the other two `leakage_flag=documented` cases
  (target-derived input features) - use for the [L2] taxonomy example in
  4.1.
- **mayer2022physmlhybrid**: see Tier 1 above.

Do not cite any paper's specific numeric result (e.g. a reported R2 or
RMSE) as if it were comparable to this project's own numbers - the whole
point of the survey is that these numbers are not evaluated under a
common protocol.

---

## 9. GAPS

Exhaustive list of everything the paper needs that does not exist yet,
or that this brief could not verify. Organized by how blocking each one
is.

### Blocking (paper cannot be submitted without these)

1. **The test-set (2015) run has never been executed.** Confirmed for
   this brief: every script that writes a run JSON (`run_xgb_dev.py`,
   `run_lstm_dev.py`, `run_cnn_lstm_dev.py`, `run_residual_dev.py`,
   `run_seed_sweep.py`, `rerun_residual_5yr.py`, `build_table4_protocol.py`,
   `build_table6_dm.py`) hardcodes `eval_split="val"` or an equivalent.
   2015 has been read only for the dead-period data-quality audit
   (availability only, no forecast metric). Every claim in this brief and
   in PROJECT_CHECKPOINT.md is a VALIDATION-split result. Per CLAUDE.md's
   research-integrity rule ("test set touched once, at the end"), this is
   correct procedure so far, but the paper needs a final test-set run
   before submission, and every table in this brief will need a
   test-split counterpart or an explicit statement that only validation
   results are reported.

2. **No LaTeX file exists anywhere in the repo.** `paper/` contains only
   `PROJECT_CHECKPOINT.md`, `literature_notes.md`, `WRITING_BRIEF.md`
   (this file), and `.gitkeep`. The entire paper needs to be written from
   scratch.

3. **RESOLVED 2026-08-08.** F1-F6 are built (`scripts/build_figures.py`,
   PDF+PNG under `paper/figures/`, captions in `paper/figures/
   CAPTIONS.md`) and F7 (the evaluation-pipeline/protocol diagram) is
   written as TikZ source (`paper/figures/F7_pipeline.tex`) but not yet
   compile-tested - no LaTeX toolchain in this dev environment, and no
   .tex file for the paper itself exists yet (gap 2 above still stands:
   these figures have nowhere to be `\input` into until the paper is
   drafted). See Section 7's 2026-08-08 note for which figures matched
   their original proposal and which were superseded by more useful
   content decided during construction.

4. **T1, T2, T5, T7 have no generating script.** Only T3, T4, T6 can be
   produced today by running an existing script. The other four need new
   aggregation code before they can be typeset (see Section 7 for exactly
   what each needs).

### Not blocking, but must be resolved before the affected section is drafted

5. **`results/table6_dm_oracle.csv` does not exist yet as of this
   brief's writing.** `python scripts/build_table6_dm.py --regime
   oracle` was launched in the background during this session and had
   not produced output at the time this brief was finalized. Table 6's
   oracle-regime counterpart, and any claim about oracle-regime
   significance, is currently ungrounded. Check for this file before
   writing anything about oracle-regime significance.

6. **PROJECT_CHECKPOINT.md Section 6 ("WHAT IS BUILT vs WHAT REMAINS")
   is now stale relative to the repo's current state**, and was already
   marked as having been stale once before (rewritten 2026-08-03, then
   drifted again). Specific staleness found while building this brief:
   - Says "3 coded rows" for the literature survey; actual is 27 (C27).
   - Says oracle-regime runs do not exist ("zero oracle result JSONs
     exist anywhere"); actual is 225/225 complete, plus
     `results/seed_sweep_summary_oracle.csv` and
     `results/table4_protocol_oracle.csv` both exist.
   - References `results/seed_sweep_summary.csv`,
     `results/table4_protocol.csv`, `results/table6_dm.csv` by their old,
     un-suffixed names; all three have been renamed to `_lagged` variants
     with `_oracle` counterparts added alongside.
   - Section 7's "Key literature" list has 4 entries (Yang, Kapoor &
     Narayanan, Nguyen & Musgens 2208.10536, Hewamalage et al.); it has
     not been updated to include Mayer (2022) or Nguyen & Musgens
     (2111.02092), both added to `literature_notes.md` as entries 9-10
     after Section 7 was last written.
   - Section 7 also still says "TEN MORE NEEDED" for the literature
     survey against a "~10-13 target" - both the count and the target
     framing are stale now that 27 have been coded.
   This file (PROJECT_CHECKPOINT.md) needs another synchronization pass;
   this brief was built by reading the underlying repo state directly
   wherever PROJECT_CHECKPOINT.md's Section 6 claims could not be
   confirmed, but the prose elsewhere in PROJECT_CHECKPOINT.md (Findings
   1-12) was independently spot-checked against source CSVs/JSONs during
   this brief's construction and found accurate except where Section 5
   above says otherwise (RQ4 timing figures, sky-stratification pooled
   counts).

7. **RESOLVED 2026-08-08.** The correctly-deduplicated sky-stratification
   "pooled" figure now exists: PROJECT_CHECKPOINT.md Finding 12 Part B,
   dated 2026-08-08 addendum. Aggregation implemented is "sum over the 3
   horizons within ONE array" (arrays are redundant, horizons are not),
   giving clear 6672 (60.4%), partly_cloudy 3669 (33.2%), overcast 714
   (6.5%), total 11055, computed from array11/xgboost and representative
   of any array per the model/array-invariance already verified in
   Section 5 item 8. Table 7 / Figure F5 should cite this figure, named
   explicitly as one array's pooled-over-horizon count in the caption.

8. **T5's exact content (RQ2 component-attribution summary table) is not
   specified anywhere** beyond "Table 5" being named in prose. This brief
   proposes joining `seed_sweep_summary_lagged.csv` with a
   DM-significance annotation column, but the exact columns/format need
   deciding before `scripts/build_table5_*.py` (does not exist) can be
   written.

9. **RESOLVED 2026-08-08.** PROJECT_CHECKPOINT.md Section 0 has been
   renumbered: RQ1 is now protocol sensitivity (HEADLINE), RQ2 is
   component attribution. This brief's Section 6 outline already put
   protocol first (6.1) and architecture second (6.2), so no reordering
   of Results was needed - only the RQ labels attached to each
   subsection changed (6.1 is now literally RQ1, 6.2 is literally RQ2,
   see the updated headers above). Results follows RQ1/RQ2/RQ3/RQ4
   numeric order straight through with no exception to flag for a
   reader. Every RQ1/RQ2 cross-reference in this brief's Claims Table
   (Section 2) and elsewhere was updated in the same pass to match the
   new numbering - see PROJECT_CHECKPOINT.md Section 0's own note on the
   rename for the full list of files touched.

10. **The recurrent_base.py "bit-for-bit" determinism claim is
    unverified.** PROJECT_CHECKPOINT.md Section 6 itself flags this: "a
    bit-for-bit-diff verification against a prior committed result is
    carried over from the 2026-07-28 checkpoint text and not
    independently re-checked" - this brief also did not re-run it (would
    require re-executing a model fit, not reading a file). If this claim
    appears in Experimental Setup, it should be re-verified first or
    softened to "was verified once, 2026-07-28, not re-checked since."

11. **No site/array map or diagram exists.** Data section (Section 6
    outline, item 3) may want one; not currently generated by anything in
    `scripts/`.

12. **F6's source data (`extra.top_features_gain`) exists only in two
    specific run JSONs** (the array11/h6/seed0 leaked-vs-corrected pair).
    Whether the paper wants this figure for exactly this one cell, or a
    broader feature-importance comparison across cells, is undecided -
    the broader version would need new aggregation code and does not
    currently exist for any other cell.

### Explicitly out of scope (from FUTURE_WORK.md, listed here only so
they are not mistaken for gaps in THIS paper)

13. Multi-location generalization (adding Yulara as a genuinely separate
    site) - deferred, different schema, second pipeline.
14. Product/tool direction (dashboard, pip package, live inference) -
    deferred to post-submission; the "cheap subset kept in scope" is a
    reproduction README and YAML configs, not yet built either, but not
    required for the paper text itself.
