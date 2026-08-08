# Figure captions

Working captions for each built figure, kept in sync with the comment
blocks in scripts/build_figures.py so the caption travels with the
figure and does not drift or get reinvented at writing time. Appended
to as each figure is built - see paper/WRITING_BRIEF.md Section 7 for
the full F1-F7 plan and which of the remaining four are not yet built.

ASCII only, per CLAUDE.md.

---

## F1 - skill vs forecast horizon, by reference forecast

File: paper/figures/F1_skill_vs_horizon.pdf (+ .png preview)
Source: results/seed_sweep_summary_lagged.csv

Caption:
Fig. 1. Forecast skill against two reference forecasts, XGBoost, lagged
regime, validation year 2014. Skill against smart persistence (solid)
rises monotonically with horizon; skill against the optimal convex
combination of climatology and persistence (dashed) is non-monotonic
and peaks at 3 h. The apparent horizon trend is a property of the
reference, not the model. Error bars show one standard deviation across
five seeds and are smaller than the markers. Only XGBoost is shown; the
other four models follow the same pattern (Table 3).

Must visually demonstrate: skill vs persistence rises monotonically
with horizon while skill vs the convex reference is non-monotonic and
peaks at h=3 - same model, same data, only the reference forecast
changed.

---

## F2 - skill by model and by feature regime

File: paper/figures/F2_skill_by_model_regime.pdf (+ .png preview)
Source: results/seed_sweep_summary_lagged.csv, results/seed_sweep_summary_oracle.csv

Caption:
Fig. 2. Skill against the convex reference, by model and by feature
regime (lagged: information available at issue time; oracle: measured
weather AT TARGET TIME, a perfect-forecast UPPER BOUND, never an
achievable result), array11, three horizons. Within either regime,
skill varies by at most 0.03-0.05 across the five models at a given
horizon (Table 3). Switching from lagged to oracle features moves
skill by 0.51-0.63 for every model and horizon shown here, and by
0.51-0.72 across all three arrays and 45 model x array x horizon cells
(Table 4 row 5). Perfect weather knowledge is worth roughly an order of
magnitude more than any architecture choice tested here. array11 only;
the same lagged-vs-oracle gap holds on array12 and array17 (Table 3).

Must visually demonstrate: the lagged-to-oracle gap dwarfs every
model-to-model difference within a regime - perfect weather knowledge
is worth far more than any architecture choice tested. (Scale choice:
plain linear axis, deliberately not broken/log - see the SCALE CHOICE
note in build_f2()'s docstring for why the within-regime differences
remain legible without one.)

---

## F3 - error and skill by sky condition

File: paper/figures/F3_error_by_sky_condition.pdf (+ .png preview)
Source: results/table_sky.csv

Caption:
Fig. 3. Forecast error and skill by sky condition (XGBoost, LSTM, LSTM
+ residual; array11, h=3, validation year 2014). Left: nRMSE (percent
of nameplate capacity). Right: skill against the optimal convex
combination of climatology and persistence. nRMSE and skill rank the
sky classes differently: overcast has the worst nRMSE of the three
classes but retains high skill, while partly-cloudy skies have middling
nRMSE but near-zero skill. Overcast skies are temporally stable hour to
hour, so the convex reference already tracks them well; partly-cloudy
skies produce irradiance ramps that lagged features cannot anticipate.
array11, h=3 only; the same ranking holds in all 9 array x horizon
cells with zero exceptions (Table 7).

Must visually demonstrate: nRMSE and skill rank the sky classes
differently - overcast has the worst nRMSE but retains high skill;
partly-cloudy has middling nRMSE but near-zero skill.
