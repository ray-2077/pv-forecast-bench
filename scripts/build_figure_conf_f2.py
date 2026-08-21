"""Build the protocol-vs-architecture effect-size summary figure for
paper/overleaf_conf/ (the six-page UPCON conference paper).

Three effects, two different units (skill score vs percent nRMSE), so
this is two stacked panels on separate axes rather than one bar chart
with a misleading shared length scale:

  Top panel (skill score, x in [0, ~0.5]), three bars:
    - largest difference among the three base architectures (xgboost,
      lstm, cnn_lstm), any array/horizon: 0.024
      (results/seed_sweep_summary_lagged.csv, array17 h=6,
      lstm-xgboost = 0.0239)
    - largest architectural effect of any kind, including the
      residual-corrected variants: 0.053
      (results/seed_sweep_summary_lagged.csv, array11 h=6,
      cnn_lstm_residual-lstm = 0.0528)
    - reference-forecast effect at h=6, array11: 0.652 - 0.194 = 0.458
      (results/reference_comparison.csv, xgb_skill_vs_persistence -
      xgb_skill_vs_convex, array11, horizon 6)

  Bottom panel (percent nRMSE, x in [0, ~40]):
    - night-hour inclusion deflates normalised RMSE by ~34 percent
      (results/table4_protocol_lagged.csv, array11, C5 vs C6, mean
      across the three horizons)

Single-column width (3.5in). Values are annotated directly on each bar
so the figure does not depend on precise gridline reading.

Usage:
    python scripts/build_figure_conf_f2.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "paper" / "overleaf_conf" / "figures"

SINGLE_WIDTH_IN = 3.5
FONT_SIZE = 7

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "font.family": "serif",
    }
)

# Verified against source, see module docstring.
ARCH_DIFF_BASE = 0.024
ARCH_DIFF_ALL = 0.053
REF_EFFECT = 0.458
NIGHT_HOUR_PCT = 34


def build():
    fig, (ax_skill, ax_pct) = plt.subplots(
        2, 1, figsize=(SINGLE_WIDTH_IN, 3.0),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    skill_labels = [
        "Architecture\n(base only)",
        "Architecture\n(incl. residual)",
        "Reference forecast\n($h=6$, array 11)",
    ]
    skill_vals = [ARCH_DIFF_BASE, ARCH_DIFF_ALL, REF_EFFECT]
    y_skill = [0, 1, 2]
    bars = ax_skill.barh(y_skill, skill_vals, height=0.55, color="black")
    ax_skill.set_yticks(y_skill)
    ax_skill.set_yticklabels(skill_labels)
    ax_skill.set_xlim(0, 0.55)
    ax_skill.set_xlabel(r"protocol / architecture effect ($\Delta$ skill)")
    ax_skill.spines["top"].set_visible(False)
    ax_skill.spines["right"].set_visible(False)
    for b, v in zip(bars, skill_vals):
        ax_skill.text(
            v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
            va="center", ha="left", fontsize=FONT_SIZE,
        )

    y_pct = [0]
    bar_pct = ax_pct.barh(y_pct, [NIGHT_HOUR_PCT], height=0.55, color="black")
    ax_pct.set_yticks(y_pct)
    ax_pct.set_yticklabels(["Night-hour\ninclusion"])
    ax_pct.set_xlim(0, 45)
    ax_pct.set_xlabel("nRMSE deflation (%), separate axis")
    ax_pct.spines["top"].set_visible(False)
    ax_pct.spines["right"].set_visible(False)
    for b, v in zip(bar_pct, [NIGHT_HOUR_PCT]):
        ax_pct.text(
            v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%",
            va="center", ha="left", fontsize=FONT_SIZE,
        )

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "F2_effect_summary.pdf"
    png_path = FIGURES_DIR / "F2_effect_summary.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    build()
