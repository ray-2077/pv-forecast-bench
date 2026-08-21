captions = {
"T4": "Protocol configurations, site 11, lagged regime, 2014: sample count, "
      "normalised RMSE, skill, and reference for six configurations at each "
      "horizon (C1/C2: daylight hours, convex/persistence reference; C3: "
      "hours where the convex reference is defined; C4: all 24 hours, "
      "persistence; C5/C6: raw nRMSE, daylight/24-hour, no reference). C1 vs "
      "C2 isolates the reference effect; C5 vs C6 isolates night-hour "
      "deflation. Site 11 only; sites 12 and 17 match (Results text).",

"T5": "Component attribution (RQ2): skill\\_vs\\_convex by model, array, and "
      "horizon, mean $\\pm$ 1 seed standard deviation (5 seeds), with "
      "Diebold-Mariano significance markers (Holm-Bonferroni, "
      "$\\alpha=0.05$, seed 0) for four comparisons: vs. XGBoost, vs. LSTM "
      "(CNN-LSTM base), +residual vs. LSTM, +residual vs. CNN-LSTM. "
      "Asterisk marks significance; full HLN statistics and p-values are "
      "in the repository.",

"T8": "Literature survey of evaluation practice, $n=27$ papers on hybrid "
      "convolutional-recurrent PV power forecasting, coded on eight "
      "dimensions of evaluation protocol (Section II); counts and "
      "percentages are per coded value within each dimension. Full "
      "per-paper coding with verbatim supporting evidence is available at "
      "an anonymised repository, link withheld for review.",

"F1": "Forecast skill against two reference forecasts (smart persistence, "
      "solid; convex climatology-persistence combination, dashed), "
      "XGBoost, lagged regime, validation year 2014, all three arrays. "
      "Error bars: one standard deviation across five seeds. Only XGBoost "
      "shown; other models follow the same pattern (Table \\ref{tab:T5}).",

"F2": "Skill against the convex reference, by model and feature regime "
      "(lagged vs. oracle, a perfect-forecast upper bound), array11, "
      "three horizons. Within either regime, skill varies by at most "
      "0.03-0.05 across models; switching regimes moves skill by "
      "0.51-0.63 for every model/horizon shown (Table \\ref{tab:T5}), "
      "matching array12 and array17.",
}

current = {"T4": 147, "T5": 247, "T8": 148, "F1": 81, "F2": 117}

total_before = 0
total_after = 0
for k, v in captions.items():
    n = len(v.split())
    total_before += current[k]
    total_after += n
    print(f"{k}: {current[k]} -> {n} words (delta {current[k]-n})")

print("TOTAL:", total_before, "->", total_after, "(delta", total_before-total_after, ")")
