"""Build the single-column, vertically-stacked F1 figure for
paper/overleaf_conf/ (the six-page UPCON conference paper).

This is NOT a modification of scripts/build_figures.py's build_f1(),
which produces the three-panel-horizontal, full-width (7.16in) version
used in paper/overleaf/ and paper/overleaf_short/. Those are finished
and must not change. This script reads the same committed CSV and
produces a different physical layout: three panels stacked vertically
in one column at IEEE single-column width (3.5in), for the conference
paper's two-float budget.

Source: results/seed_sweep_summary_lagged.csv (lagged regime, XGBoost
only, validation split 2014) - same source, same model, same split as
the journal-version F1. This script only reads and plots.

Usage:
    python scripts/build_figure_conf_f1.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "paper" / "overleaf_conf" / "figures"

ARRAYS = [
    ("array11", "poly-Si"),
    ("array12", "mono-Si"),
    ("array17", "HIT"),
]
HORIZONS = [1, 3, 6]

SINGLE_WIDTH_IN = 3.5
FONT_SIZE = 7

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


def build():
    df = pd.read_csv(RESULTS_DIR / "seed_sweep_summary_lagged.csv")
    df = df[df["model"] == "xgboost"]

    missing_horizons = set(HORIZONS) - set(df["horizon"].unique())
    if missing_horizons:
        raise ValueError(
            f"missing horizons in seed_sweep_summary_lagged.csv: {missing_horizons}"
        )

    fig, axes = plt.subplots(
        3, 1, figsize=(SINGLE_WIDTH_IN, 4.6), sharex=True, sharey=True
    )

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
            markersize=3,
            linewidth=0.9,
            capsize=2,
            elinewidth=0.6,
            label="skill vs persistence",
        )
        ax.errorbar(
            sub["horizon"],
            sub["mean_skill_vs_convex"],
            yerr=sub["std_skill_vs_convex"],
            color="black",
            linestyle="--",
            marker="s",
            markersize=3,
            linewidth=0.9,
            capsize=2,
            elinewidth=0.6,
            label="skill vs convex",
        )

        ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
        ax.set_ylabel(f"{array_key}\n({tech})", fontsize=FONT_SIZE)
        ax.set_xticks(HORIZONS)
        ax.set_xlim(0.5, 6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("forecast horizon h")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(0.5, -0.08),
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


if __name__ == "__main__":
    build()
