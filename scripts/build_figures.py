"""Build paper figures from committed results/ CSVs.

Currently implements F1 (skill vs forecast horizon, by reference
forecast), F2 (skill by model and feature regime), F3 (error and skill
by sky condition), F4 (residual distribution before/after the residual
stage), F5 (feature importance, lagged vs oracle), and F6 (skill vs
training compute) - see paper/WRITING_BRIEF.md Section 7 for the full
F1-F7 plan and paper/PROJECT_CHECKPOINT.md Section 9 for what remains.
F7 is not implemented; do not add a stub function for it until it is
actually built (CLAUDE.md: no half-finished implementations).

WIDTH CONVENTION: multi-panel figures (F1, F3, F5) use the full IEEE
column width (7.16in); single-panel figures (F2 uses 3 panels so it is
full-width too, F4 and F6 are genuinely one panel each) use single-
column width (3.5in) - stretching one panel across the full page looks
sparse and is not requested by any figure's own content.

F1: one panel per array (array11 poly-Si, array12 mono-Si, array17
HIT), shared y-axis. x = forecast horizon, plotted at its true numeric
spacing (1, 3, 6), not evenly-spaced categorical ticks - the peak at
h=3 (skill vs convex) and the fact that h=3-to-h=6 covers twice the gap
of h=1-to-h=3 are both easier to read that way.

Two lines per panel, XGBoost only:
  skill_vs_persistence (solid line, circle markers)
  skill_vs_convex      (dashed line, square markers)
Both drawn in black, distinguished only by marker and linestyle, so the
figure is unambiguous in a greyscale-printed copy.

MODEL CHOICE - CAPTION NOTE FOR WHOEVER WRITES THE LATEX CAPTION:
XGBoost only, not all five models. Plotting all five would show the
same reference-choice effect five times, under five different noisy
curves, which buries the point the figure exists to make
(PROJECT_CHECKPOINT.md Finding 5 / WRITING_BRIEF.md claims C16-C17:
same model, same data, only the reference forecast changes). State this
restriction explicitly in the caption - do not let a reader assume the
other four models look different without saying so.

What the figure shows (Finding 5): skill vs persistence rises
monotonically with horizon - an artifact of smart persistence
collapsing at long horizons (Finding 4: the issue time falls near dawn
at h=6, so the persisted k_p is stale) - while skill vs the convex
reference is non-monotonic and peaks at h=3. Same model, same data,
only the reference changed.

Error bars: seed standard deviation from the 5-seed sweep
(results/seed_sweep_summary_lagged.csv's std_skill_vs_* columns), a
reproducibility statistic, not a confidence interval - do not caption
these as confidence intervals (see PROJECT_CHECKPOINT.md Finding 8's
own caution against treating seed spread as a significance measure).

Source: results/seed_sweep_summary_lagged.csv (lagged regime only -
CLAUDE.md rule 5, lagged and oracle must never be mixed in one figure).
Validation split (2014), single seed sweep already committed - this
script only reads and plots, it fits nothing and touches no split.

Usage:
    python scripts/build_figures.py [--figure F1]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"

ARRAYS = [
    ("array11", "poly-Si"),
    ("array12", "mono-Si"),
    ("array17", "HIT"),
]
HORIZONS = [1, 3, 6]

# IEEE two-column page: 3.5in single column, 7.16in full width.
FULL_WIDTH_IN = 7.16
SINGLE_WIDTH_IN = 3.5
FONT_SIZE = 8

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "font.family": "serif",
    }
)


# CAPTION (drafted 2026-08-08, travels with the figure so it is not
# reinvented at writing time - copy into the LaTeX \caption{} verbatim
# or edit in place if the figure changes):
#
# Fig. 1. Forecast skill against two reference forecasts, XGBoost,
# lagged regime, validation year 2014. Skill against smart persistence
# (solid) rises monotonically with horizon; skill against the optimal
# convex combination of climatology and persistence (dashed) is
# non-monotonic and peaks at 3 h. The apparent horizon trend is a
# property of the reference, not the model. Error bars show one standard
# deviation across five seeds and are smaller than the markers. Only
# XGBoost is shown; the other four models follow the same pattern
# (Table 3).
def build_f1():
    """skill_vs_persistence and skill_vs_convex vs horizon, XGBoost
    only, one panel per array. Writes F1_skill_vs_horizon.{pdf,png}
    under paper/figures/.
    """
    df = pd.read_csv(RESULTS_DIR / "seed_sweep_summary_lagged.csv")
    df = df[df["model"] == "xgboost"]

    missing_horizons = set(HORIZONS) - set(df["horizon"].unique())
    if missing_horizons:
        raise ValueError(
            f"missing horizons in seed_sweep_summary_lagged.csv: {missing_horizons}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.6), sharey=True)

    for ax, (array_key, tech) in zip(axes, ARRAYS):
        sub = df[df["array"] == array_key].sort_values("horizon")
        if len(sub) != len(HORIZONS):
            raise ValueError(
                f"{array_key}: expected {len(HORIZONS)} horizon rows, got {len(sub)}"
            )

        ax.errorbar(
            sub["horizon"],
            sub["mean_skill_vs_persistence"],
            yerr=sub["std_skill_vs_persistence"],
            color="black",
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=1.0,
            capsize=2,
            elinewidth=0.7,
            label="skill vs persistence",
        )
        ax.errorbar(
            sub["horizon"],
            sub["mean_skill_vs_convex"],
            yerr=sub["std_skill_vs_convex"],
            color="black",
            linestyle="--",
            marker="s",
            markersize=4,
            linewidth=1.0,
            capsize=2,
            elinewidth=0.7,
            label="skill vs convex",
        )

        # Anchors the eye: every value plotted here is positive skill,
        # and a reader should be able to see that at a glance.
        ax.axhline(0, color="grey", linewidth=0.5, zorder=0)

        ax.set_title(f"{array_key} ({tech})")
        ax.set_xlabel("forecast horizon h")
        ax.set_xticks(HORIZONS)
        ax.set_xlim(0.5, 6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("skill score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "F1_skill_vs_horizon.pdf"
    png_path = FIGURES_DIR / "F1_skill_vs_horizon.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


# CAPTION (drafted 2026-08-08, travels with the figure so it is not
# reinvented at writing time - copy into the LaTeX \caption{} verbatim
# or edit in place if the figure changes):
#
# Fig. 2. Skill against the convex reference, by model and by feature
# regime (lagged: information available at issue time; oracle: measured
# weather AT TARGET TIME, a perfect-forecast UPPER BOUND, never an
# achievable result), array11, three horizons. Within either regime,
# skill varies by at most 0.03-0.05 across the five models at a given
# horizon (Table 3). Switching from lagged to oracle features moves
# skill by 0.51-0.63 for every model and horizon shown here, and by
# 0.51-0.72 across all three arrays and 45 model x array x horizon
# cells (Table 4 row 5). Perfect weather knowledge is worth roughly an
# order of magnitude more than any architecture choice tested here.
# array11 only; the same lagged-vs-oracle gap holds on array12 and
# array17 (Table 3).
def build_f2():
    """skill_vs_convex by model, lagged vs oracle regime, array11 only,
    one panel per horizon. Writes F2_skill_by_model_regime.{pdf,png}
    under paper/figures/.

    SCALE CHOICE, DECIDED 2026-08-08: plain linear axis, checked and
    kept deliberately rather than a broken axis or log scale. The
    within-regime model spread (0.03-0.05 lagged, 0.015-0.033 oracle)
    turns out to be legible as-is: the residual penalty (Findings
    10-11) is visible as bars 4-5 sitting visibly shorter than bars 1-3
    within the lagged cluster, at all three horizons - it does not need
    rescuing. A broken axis would have bought a small legibility gain on
    that secondary point at the cost of visually shrinking the figure's
    actual headline (the lagged-to-oracle gap), which is the wrong
    trade when the plain version already shows both. Revisit only if a
    reader genuinely cannot make out the within-regime bars at print
    size.
    """
    ARRAY = "array11"
    MODELS = [
        ("xgboost", "XGBoost"),
        ("lstm", "LSTM"),
        ("cnn_lstm", "CNN-LSTM"),
        ("lstm_residual", "LSTM+resid"),
        ("cnn_lstm_residual", "CNN-LSTM+resid"),
    ]
    REGIMES = [
        ("lagged", "lagged", "white", None),
        ("oracle", "oracle (upper bound)", "0.4", "//"),
    ]

    lagged = pd.read_csv(RESULTS_DIR / "seed_sweep_summary_lagged.csv")
    oracle = pd.read_csv(RESULTS_DIR / "seed_sweep_summary_oracle.csv")
    lagged["regime"] = "lagged"
    oracle["regime"] = "oracle"
    df = pd.concat([lagged, oracle], ignore_index=True)
    df = df[df["array"] == ARRAY]

    expected_rows = len(MODELS) * len(REGIMES) * len(HORIZONS)
    if len(df) != expected_rows:
        raise ValueError(
            f"{ARRAY}: expected {expected_rows} rows across both regime "
            f"CSVs, got {len(df)}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.8), sharey=True)

    bar_width = 0.35
    offsets = [-bar_width / 2, bar_width / 2]
    group_positions = range(len(MODELS))

    for ax, horizon in zip(axes, HORIZONS):
        sub_h = df[df["horizon"] == horizon]

        for (regime_key, regime_label, facecolor, hatch), offset in zip(
            REGIMES, offsets
        ):
            sub = sub_h[sub_h["regime"] == regime_key].set_index("model")
            sub = sub.loc[[key for key, _ in MODELS]]
            x = [pos + offset for pos in group_positions]

            ax.bar(
                x,
                sub["mean_skill_vs_convex"],
                yerr=sub["std_skill_vs_convex"],
                width=bar_width,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.6,
                hatch=hatch,
                capsize=2,
                error_kw={"elinewidth": 0.7},
                label=regime_label,
            )

        ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
        ax.set_title(f"h = {horizon}")
        ax.set_xticks(list(group_positions))
        ax.set_xticklabels(
            [label for _, label in MODELS], rotation=30, ha="right"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("skill vs convex")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
    )

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "F2_skill_by_model_regime.pdf"
    png_path = FIGURES_DIR / "F2_skill_by_model_regime.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


# CAPTION (drafted 2026-08-08, travels with the figure so it is not
# reinvented at writing time - copy into the LaTeX \caption{} verbatim
# or edit in place if the figure changes):
#
# Fig. 3. Forecast error and skill by sky condition (XGBoost, LSTM,
# LSTM + residual; array11, h=3, validation year 2014). Left: nRMSE
# (percent of nameplate capacity). Right: skill against the optimal
# convex combination of climatology and persistence. nRMSE and skill
# rank the sky classes differently: overcast has the worst nRMSE of the
# three classes but retains high skill, while partly-cloudy skies have
# middling nRMSE but near-zero skill. Overcast skies are temporally
# stable hour to hour, so the convex reference already tracks them
# well; partly-cloudy skies produce irradiance ramps that lagged
# features cannot anticipate. array11, h=3 only; the same ranking holds
# in all 9 array x horizon cells with zero exceptions (Table 7).
def build_f3():
    """nRMSE and skill_vs_convex by sky class, three models, array11
    h=3 only. Writes F3_error_by_sky_condition.{pdf,png} under
    paper/figures/.
    """
    ARRAY = "array11"
    HORIZON = 3
    MODELS = [
        ("xgboost", "XGBoost", "white", None),
        ("lstm", "LSTM", "0.7", "//"),
        ("lstm_residual", "LSTM + residual", "0.35", "xx"),
    ]
    SKY_CLASSES = [
        ("clear", "clear"),
        ("partly_cloudy", "partly\ncloudy"),
        ("overcast", "overcast"),
    ]

    df = pd.read_csv(RESULTS_DIR / "table_sky.csv")
    df = df[(df["array"] == ARRAY) & (df["horizon"] == HORIZON)]

    expected_rows = len(MODELS) * len(SKY_CLASSES)
    if len(df) != expected_rows:
        raise ValueError(
            f"{ARRAY} h={HORIZON}: expected {expected_rows} rows in "
            f"table_sky.csv, got {len(df)}"
        )

    fig, (ax_nrmse, ax_skill) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.8))

    group_positions = range(len(SKY_CLASSES))
    bar_width = 0.25
    offsets = [-bar_width, 0.0, bar_width]

    for (model_key, model_label, facecolor, hatch), offset in zip(MODELS, offsets):
        sub = df[df["model"] == model_key].set_index("sky_class")
        sub = sub.loc[[key for key, _ in SKY_CLASSES]]
        x = [pos + offset for pos in group_positions]

        ax_nrmse.bar(
            x,
            sub["nrmse"],
            width=bar_width,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
            label=model_label,
        )
        ax_skill.bar(
            x,
            sub["skill_vs_convex"],
            width=bar_width,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
            label=model_label,
        )

    for ax, ylabel in ((ax_nrmse, "nRMSE (%)"), (ax_skill, "skill vs convex")):
        ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
        ax.set_xticks(list(group_positions))
        ax.set_xticklabels([label for _, label in SKY_CLASSES])
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = ax_nrmse.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "F3_error_by_sky_condition.pdf"
    png_path = FIGURES_DIR / "F3_error_by_sky_condition.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


# CAPTION (drafted 2026-08-08, travels with the figure so it is not
# reinvented at writing time - copy into the LaTeX \caption{} verbatim
# or edit in place if the figure changes):
#
# Fig. 4. Forecast residual (y_true - y_pred) distribution before and
# after the residual-correction stage, LSTM base, array11, h=6, seed 0,
# lagged regime, validation year 2014 (n=3679 daylight, non-outage
# hours). The residual stage does not tighten the error distribution:
# mean residual moves from -0.008 kW (plain LSTM) to +0.014 kW
# (corrected), and its standard deviation widens from 0.516 kW to 0.538
# kW. This matches the overconfidence mechanism identified in Finding
# 11: at h=6, the residual corrector's predicted-to-actual correlation
# on validation is only rho_val=+0.036 (versus +0.79 out-of-fold, the
# split it was fit on), and its correction is sized sigma_p/sigma_r =
# 0.34, roughly 4.7x the break-even ratio of 0.072 a correction needs
# to help. It is not a well-calibrated correction lightly attenuated -
# it is a correction sized for a relationship that does not transfer to
# validation.
def build_f4():
    """Overlaid residual distributions for plain LSTM and LSTM+residual,
    array11, h=6, seed 0, lagged regime. Writes
    F4_residual_distribution.{pdf,png} under paper/figures/.

    Reads results/f4_residuals_array11_h6_lagged_seed0.csv, generated by
    scripts/compute_f4_residuals.py - a real refit, since no committed
    run JSON stores per-row predictions (see that script's docstring).
    Run it first if the CSV is missing.
    """
    csv_path = RESULTS_DIR / "f4_residuals_array11_h6_lagged_seed0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found - run scripts/compute_f4_residuals.py first"
        )
    df = pd.read_csv(csv_path)

    MODELS = [
        ("lstm", "LSTM", "-"),
        ("lstm_residual", "LSTM + residual", "--"),
    ]

    bin_edges = np.linspace(df["residual"].min(), df["residual"].max(), 41)

    fig, ax = plt.subplots(figsize=(SINGLE_WIDTH_IN, 2.8))

    for model_key, label, linestyle in MODELS:
        sub = df.loc[df["model"] == model_key, "residual"]
        ax.hist(
            sub,
            bins=bin_edges,
            density=True,
            histtype="step",
            color="black",
            linestyle=linestyle,
            linewidth=1.1,
            label=f"{label} (mean={sub.mean():+.3f}, std={sub.std():.3f})",
        )

    ax.axvline(0, color="grey", linewidth=0.5, zorder=0)
    ax.set_xlabel("residual, y_true - y_pred (kW)")
    ax.set_ylabel("density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # frameon=True + opaque white facecolor: the zero-line and both
    # histogram outlines pass directly under this legend's usual corner
    # spots, and a frameless legend let the zero-line bleed through the
    # text (caught on first render).
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fontsize=FONT_SIZE - 1,
        loc="upper left",
    )

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "F4_residual_distribution.pdf"
    png_path = FIGURES_DIR / "F4_residual_distribution.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def _top_features(regime, array="array11", horizon=3, seed=0, n=10):
    """Top-n (feature, gain) pairs for one XGBoost run, sorted by gain
    descending.

    src.eval.runner.write_run dumps every run JSON with
    json.dump(..., sort_keys=True), which alphabetises every dict
    including extra.top_features_gain - the on-disk key order is NOT
    gain order. Re-sort here rather than trusting it.
    """
    path = RESULTS_DIR / f"xgboost_{array}_h{horizon}_{regime}_seed{seed}.json"
    with open(path) as fh:
        record = json.load(fh)
    feats = record["extra"]["top_features_gain"]
    ranked = sorted(feats.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n]


# CAPTION (drafted 2026-08-08, travels with the figure so it is not
# reinvented at writing time - copy into the LaTeX \caption{} verbatim
# or edit in place if the figure changes):
#
# Fig. 5. XGBoost feature importance (gain), top 10, array11, h=3, seed
# 0, validation year 2014, lagged vs oracle regime. Under oracle
# features (measured weather AT TARGET TIME, a perfect-forecast UPPER
# BOUND, prefixed oracle_ so it cannot be mistaken for a legitimate
# feature), oracle_Global_Horizontal_Radiation dominates at 145.6 gain,
# more than 3x the next feature (p_cs, 45.5). Under the lagged regime -
# the only features available at real issue time - no single feature
# dominates: the top four (solar_elevation 502.1, hour_cos 422.6, p_cs
# 370.9, solar_zenith 110.3) are all deterministic solar-geometry
# features, not measured weather, and the top two differ by less than
# 20%. array11, h=3, seed 0 only.
def build_f5():
    """Top-10 XGBoost feature importances (gain), lagged vs oracle,
    array11 h=3 seed 0. Writes F5_feature_importance.{pdf,png} under
    paper/figures/.
    """
    N_TOP = 10
    REGIMES = [
        ("lagged", "lagged", "white", None),
        ("oracle", "oracle (upper bound)", "0.6", "//"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.2))

    for ax, (regime_key, title, facecolor, hatch) in zip(axes, REGIMES):
        ranked = _top_features(regime_key, n=N_TOP)
        names = [name for name, _ in ranked]
        gains = [gain for _, gain in ranked]
        y_pos = list(range(len(names)))

        ax.barh(
            y_pos,
            gains,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=FONT_SIZE - 1)
        ax.invert_yaxis()  # rank 1 (highest gain) at top
        ax.set_xlabel("gain")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "F5_feature_importance.pdf"
    png_path = FIGURES_DIR / "F5_feature_importance.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


# CAPTION (drafted 2026-08-08, travels with the figure so it is not
# reinvented at writing time - copy into the LaTeX \caption{} verbatim
# or edit in place if the figure changes):
#
# Fig. 6. Skill against the convex reference vs. mean training compute,
# array11, h=6, lagged regime, validation year 2014. x-axis: mean
# fit_seconds across all 45 lagged runs per model (5 models x 3 arrays
# x 3 horizons x 5 seeds), log scale. y-axis: mean skill_vs_convex at
# this array/horizon cell, 5 seeds. XGBoost reaches skill 0.195 in
# 0.55 s; LSTM reaches a higher 0.210 in 10.11 s, roughly 18x the
# compute for a gain of +0.015 that is not Diebold-Mariano significant
# on this array (Table 6: array11 h=6, p_holm=0.073). The two residual
# variants cost more compute still (22.7 s, 25.0 s) for lower skill
# than their own base models. array11, h=6, lagged only. Mean
# fit_seconds is a global average, so the compute gap is the same at
# every array and horizon; the skill gap it buys is largest here (h=6)
# and smaller still at h=1 and h=3, where XGBoost and LSTM are
# statistically indistinguishable everywhere (Table 6).
def build_f6():
    """skill_vs_convex (array11, h=6, lagged) vs mean training compute
    (fit_seconds, log scale), one marker per model. Writes
    F6_skill_vs_compute.{pdf,png} under paper/figures/.

    Mean fit_seconds is computed here directly from all 45 lagged run
    JSONs per model - the same computation behind
    PROJECT_CHECKPOINT.md's 2026-08-07 correction (WRITING_BRIEF.md
    claim C24), not copied from that file's prose, which the
    correction itself flags as having been wrong once (~16s/~30:1
    instead of the correct 10.11s/~18:1 for lstm/xgboost).
    """
    MODELS = [
        ("xgboost", "XGBoost", "o", (6, 4)),
        ("lstm", "LSTM", "s", (6, 4)),
        ("cnn_lstm", "CNN-LSTM", "^", (6, -10)),
        ("lstm_residual", "LSTM+resid", "D", (-10, 8)),
        ("cnn_lstm_residual", "CNN-LSTM+resid", "x", (6, -10)),
    ]

    skill_df = pd.read_csv(RESULTS_DIR / "seed_sweep_summary_lagged.csv")
    skill_df = skill_df[(skill_df["array"] == "array11") & (skill_df["horizon"] == 6)]

    fig, ax = plt.subplots(figsize=(SINGLE_WIDTH_IN, 3.0))

    for model_key, label, marker, (dx, dy) in MODELS:
        run_files = [
            fp for fp in RESULTS_DIR.glob(f"{model_key}_array*_h*_lagged_seed*.json")
            if "leaked" not in fp.name
        ]
        if not run_files:
            raise FileNotFoundError(f"no lagged run JSONs found for model={model_key}")
        fit_seconds = []
        for fp in run_files:
            with open(fp) as fh:
                fit_seconds.append(json.load(fh)["timings"]["fit_seconds"])
        mean_fit = sum(fit_seconds) / len(fit_seconds)

        skill = skill_df.loc[skill_df["model"] == model_key, "mean_skill_vs_convex"].iloc[0]

        ax.scatter([mean_fit], [skill], marker=marker, color="black", s=28)
        ax.annotate(
            label,
            (mean_fit, skill),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=FONT_SIZE - 1,
        )

    ax.set_xscale("log")
    ax.set_xlabel("mean fit time (s, log scale)")
    ax.set_ylabel("skill vs convex (array11, h=6)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "F6_skill_vs_compute.pdf"
    png_path = FIGURES_DIR / "F6_skill_vs_compute.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


FIGURES = {
    "F1": build_f1,
    "F2": build_f2,
    "F3": build_f3,
    "F4": build_f4,
    "F5": build_f5,
    "F6": build_f6,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=sorted(FIGURES),
        default="F1",
        help="which figure to build (F1-F6 are implemented so far, not F7)",
    )
    args = parser.parse_args()
    FIGURES[args.figure]()


if __name__ == "__main__":
    main()
