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
Sources: results/data_audit.csv, results/dead_period_audit.csv,
scripts/build_processed.py (nameplate kW only - manufacturer/
technology/site number are not in any script, see the script's own
docstring for where they are sourced from instead), src/data/
clearsky_power.py (tilt/azimuth), row/daylight counts computed directly
via src.data.splits.split_chronological on each processed parquet

Caption:
Table 1. Dataset summary: the three co-located DKASC arrays used in
evaluation (array11 poly-Si, array12 mono-Si, array17 HIT - one shared
weather station, not three independent sites) plus array07 (CdTe),
excluded. Row and daylight-hour counts by chronological split (train
2011-2013, validation 2014, test 2015 - touched once, at the end).
array07 is retained as an excluded row, not deleted: a completeness
audit (results/data_audit.csv) passed its 2014 data at 99.99% coverage
and 0.00% NaN in Active_Power, while a dead-period audit (results/
dead_period_audit.csv) found 48.41% of that same year's daylight hours
at exactly zero power (status FAIL), plus a 48-day near-zero run
inside the test year - a completeness-only audit is structurally blind
to a healthy sensor reporting a dead array (Finding 6).

Must convey: array11/array12/array17 are co-located, sharing one
weather station, not three independent sites (CLAUDE.md wording
constraint) - and array07's exclusion must be visible as a real,
numbered audit finding in this same table, not an assertion made only
in prose elsewhere.

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
