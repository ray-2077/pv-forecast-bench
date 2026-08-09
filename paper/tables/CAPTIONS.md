# Table captions

Working captions for each built table, in the same format as
paper/figures/CAPTIONS.md. Kept in sync with each table's own
generating script (comment header + `\caption{}` in the .tex output) so
the caption travels with the table rather than being reinvented at
writing time. See paper/WRITING_BRIEF.md Section 7 for the full T1-T7
plan - T3, T4, and T6 already had generating scripts before this pass
(`scripts/aggregate_seed_sweep.py`, `scripts/build_table4_protocol.py`,
`scripts/build_table6_dm.py`); T1, T2, T5, and T7 are new here.

NOT COMPILE-TESTED: this development environment has no LaTeX toolchain
(no pdflatex/latexmk/xelatex found) - every .tex fragment below has
been checked by hand for booktabs/tabular syntax but never actually
compiled. Same caveat as paper/figures/F7_pipeline.tex. Verify all four
render correctly, and that none overflows its column/page width at
actual print size, before camera-ready.

ASCII only, per CLAUDE.md.

---

## T1 - dataset summary

Files: paper/tables/T1_dataset.csv, paper/tables/T1_dataset.tex
Generating script: scripts/build_table1_dataset.py
Sources: results/data_audit.csv, results/dead_period_audit.csv (array07
exclusion reason only), scripts/build_processed.py (nameplate kW only -
manufacturer/technology/site number are not in any script, see the
script's own docstring for where they are sourced from instead),
src/data/clearsky_power.py (tilt/azimuth); row counts, geometric
daylight-hour counts, and outage-adjusted evaluable counts all computed
live via src.data.pipeline.load_and_prepare + src.data.splits.
split_chronological + src.eval.exclusions.exclusion_mask on each
processed parquet - not read from a side CSV.

CORRECTED 2026-08-09 (three issues caught before this table could be
cited): (1) the original daylight-hour columns were purely geometric
(solar_elevation > 10 deg, identical across every array by
construction, since it depends only on the shared weather station) and
were named plainly "n_daylight_*", which could be misread as the count
that matters for evaluation. Renamed to n_daylight_geometric_* and
added n_evaluable_val/n_evaluable_test, which subtract documented
equipment outages (src.eval.exclusions.exclusion_mask) - array17's
n_evaluable_test (3757) is now visibly smaller than its
n_daylight_geometric_test (3802) because of the 2015-06-05..09 outage,
while array11/array12 are unaffected (3802 for both). (2) array07's
row/daylight/evaluable columns previously showed real-looking numbers
computed from a leftover, no-longer-pipeline-registered parquet file -
identical to the other three arrays' numbers by construction (shared
calendar), which implied array07 was measured on the same footing.
Replaced with "n/a (excluded)" throughout. (3) array07's tilt/azimuth
were "not recorded" out of excess caution; DKASC's fixed-mount
convention documents ALL arrays at this site, including array07, at
tilt 20 deg / azimuth 0 deg, same source already cited for
array11/12/17 - filled in, with the source noted in the script's
module docstring.

Caption:
Table 1. Dataset summary: the three co-located DKASC arrays used in
evaluation (array11 poly-Si, array12 mono-Si, array17 HIT - one shared
weather station, not three independent sites) plus array07 (CdTe),
excluded. Row counts and geometric daylight-hour counts (solar
elevation $>10^\circ$, identical across arrays by construction) by
chronological split (train 2011-2013, validation 2014, test 2015 -
touched once, at the end), plus outage-adjusted evaluable hour counts
for validation and test: array17's documented 2015-06-05 to 2015-06-09
outage reduces its evaluable test count to 3757 daylight hours, versus
3802 for array11 and array12, which are unaffected. array07 is
retained as an excluded row, not deleted, with its row/daylight/
evaluable columns reported as not applicable rather than measured (no
run in this project fits or scores a model against array07's data): a
completeness audit (results/data_audit.csv) passed its 2014 data at
99.99% coverage and 0.00% NaN in Active_Power, while a dead-period
audit (results/dead_period_audit.csv) found 48.41% of that same year's
daylight hours at exactly zero power (status FAIL), plus a 48-day
near-zero run inside the test year - a completeness-only audit is
structurally blind to a healthy sensor reporting a dead array
(Finding 6).

Must convey: array11/array12/array17 are co-located, sharing one
weather station, not three independent sites (CLAUDE.md wording
constraint); array17's smaller evaluable test sample must be visible as
a real, numbered consequence of the documented outage, not folded into
an undifferentiated "daylight hours" count that looks identical across
arrays; and array07's exclusion must read as a real, numbered audit
finding with columns that honestly say "not applicable," never as
plausible-looking numbers that imply it was evaluated.

---

## T2 - feature list by regime

Files: paper/tables/T2_features.csv, paper/tables/T2_features.tex
Generating script: scripts/build_table2_features.py
Source: src/features/build.py's own feature_names() and internal
constants (DETERMINISTIC_NAMES, LAG_BASE_COLS, WALLCLOCK_ROLLING_SPECS,
OBS_ROLLING_SPECS, WEATHER_COLS) - imported directly, not
hand-transcribed, and the generated row set is asserted to match
feature_names('lagged'|'oracle', horizon) exactly before the table is
written, so a future change to build.py that isn't reflected here fails
the script loudly instead of shipping a stale table.

Caption:
Table 2. Feature list by regime: all 37 lagged-regime features, plus
the 5 additional oracle-only features (42 total), grouped into four
categories (deterministic at target time, lagged observations, rolling
statistics, oracle weather), with the shift applied to each relative
to target time $t$ and issue time $t-h$. Every lagged-regime feature's
shift is at or after the issue time ($\geq h$); oracle features are
prefixed oracle\_ and use measured weather AT TARGET TIME $t$ (shift
0) - a perfect-forecast upper bound, never achievable, and never mixed
with the lagged regime in one feature matrix (CLAUDE.md rule 5).

Must convey: completeness and explicit shifts. This is the table a
reviewer checks the leakage claim against - every category-B/C feature
must show a shift that is either $\geq h$ or explicitly marked as the
oracle regime's target-time upper bound, with no feature omitted.

---

## T5 - component attribution (RQ2)

Files: paper/tables/T5_component_attribution.csv,
paper/tables/T5_component_attribution.tex
Generating script: scripts/build_table5_component_attribution.py
Sources: results/seed_sweep_summary_lagged.csv (mean/std
skill_vs_convex, 5 seeds), results/table6_dm_lagged.csv (pairwise
Diebold-Mariano significance, seed=0) - joined and reformatted, no new
computation

Caption:
Table 5. Component attribution (RQ2): skill\_vs\_convex by model
(XGBoost, LSTM, CNN-LSTM, LSTM+residual, CNN-LSTM+residual), array, and
horizon - mean $\pm$ 1 seed standard deviation across 5 seeds - with
Diebold-Mariano significance (HAC variance, HLN small-sample
correction, Holm-Bonferroni within cell, single seed=0) for the three
comparisons RQ2's Results turns on: LSTM vs. XGBoost, CNN-LSTM vs.
LSTM, and LSTM+residual vs. LSTM. At $h=1$ and $h=3$, XGBoost and LSTM
are statistically indistinguishable on every array ($p_{holm}=1.0$
throughout); at $h=6$, LSTM's edge over XGBoost is significant on only
one of three co-located arrays (array17). The residual stage is
significant in the direction of the plain base model in 5 of 9
lstm\_residual cells, with no clean horizon-based split - array12
$h=1$ is significant, array12 $h=6$ is not, despite being the longer
horizon.

Must convey: seed spread (the mean $\pm$ std columns) must never be
read as a significance test on its own - the DM annotation is what
actually establishes or fails to establish a claim
(PROJECT_CHECKPOINT.md Finding 8's own retraction of exactly that
substitution). A reader should be able to see, cell by cell, which
apparent gaps are established and which are merely suggestive.

---

## T7 - sky-condition results (RQ3)

Files: paper/tables/T7_sky_condition.csv,
paper/tables/T7_sky_condition.tex
Generating script: scripts/build_table7_sky.py
Source: results/table_sky.csv, filtered to array11 h=3 (same single
cell Figure F3 visualizes - see paper/figures/CAPTIONS.md)

Caption:
Table 7. Sky-condition results (RQ3), array11, $h=3$: nRMSE and
skill\_vs\_convex by sky class (clear / partly cloudy / overcast) for
XGBoost, LSTM, and LSTM+residual, with class counts ($n$). nRMSE and
skill rank the classes differently: overcast has the worst nRMSE of
the three classes (15.4-16.2\%) but the second-highest skill (+0.38 to
+0.42, comparable to clear sky); partly cloudy has middling nRMSE
(11.7-11.8\%) but the lowest skill by a wide margin (+0.04 to +0.06).
array11, $h=3$ only; the same ranking holds in all 9 array x horizon
cells with zero exceptions (PROJECT_CHECKPOINT.md Finding 12 Part B).

Must convey: the disagreement between the nRMSE ranking and the skill
ranking of sky conditions - a reader comparing only the nRMSE column
would conclude overcast is the hardest case; the skill column shows
the opposite. This is the tabular form of Figure F3's single point,
not a separate result.

NOTE: this is deliberately the same one-cell scope as Figure F3, not
the full 9-cell grid results/table_sky.csv actually contains. A
correctly-pooled multi-cell version (summing daylight hours over the
3 horizons within one array, per Finding 12 Part B's 2026-08-08
addendum - arrays are redundant, horizons are not) is still an unbuilt
need, tracked in paper/WRITING_BRIEF.md Section 7's T7 row.

---

## T8 - literature survey evaluation practice (Section 2, Related Work)

Files: paper/tables/T8_survey.csv, paper/tables/T8_survey.tex
Generating script: scripts/build_table8_survey.py
Source: results/literature_survey.csv (27 rows, one per surveyed
paper). Added 2026-08-09: the survey coding previously had no assigned
table number (paper/WRITING_BRIEF.md's original T2 proposal - a
literature-survey summary - was reassigned to the feature-list table
during construction, see T2's own entry above), despite being the
evidence base for four claims in Section 2 (C22, C27-C33 in
paper/WRITING_BRIEF.md Section 2's claims table).

SCOPE, DELIBERATE: a SUMMARY, not a per-paper listing. One row per
coded value per dimension (count and percentage of n=27), for the
eight dimensions the paper's prose claims are built on:
night_hours_excluded, baseline_used, skill_score_reported,
weather_source, split_type, variance_reported, code_available,
leakage_flag. A full 27-row-by-17-column transcription would consume
most of a column and is not what any claim in the paper needs. The
script asserts every dimension's counts sum to 27 and raises if not.

Caption:
Table 8. Literature survey evaluation practice, $n=27$ papers on
hybrid convolutional-recurrent PV power forecasting, coded on eight
dimensions of evaluation protocol (Related Work, Section 2-B).
Twenty-six of 27 report no skill score against any reference forecast
and all 27 report no run-to-run variance across seeds; 22 of 27 do not
state whether their train-test split preserves temporal order; 17 of
27 do not state whether night hours are excluded from evaluation; and
3 of 27 describe, in their own text, a procedure that constitutes
documented leakage under the taxonomy of Kapoor and Narayanan (2023).
Coding was conservative: a field was recorded as not stated unless the
paper explicitly stated it. The full per-paper coding, with a
verbatim-quoted supporting audit file for 25 of the 27 papers
(evidence\_level=quoted; the remaining 2 are evidence\_level=
summary\_only, coded from pre-existing notes with no locatable source
PDF), is in results/literature\_survey.csv and evidence/*.md, not
reproduced here.

Must convey: this is a summary of coded evaluation PRACTICE, not an
architecture comparison and not a per-paper leaderboard - no numeric
result from any surveyed paper appears in this table, deliberately,
per paper/WRITING_BRIEF.md Section 9's citation-plan note not to treat
those numbers as comparable across studies. The "not stated" values
are themselves a coded, conservative judgement (silence, not inferred
absence), not missing data.
