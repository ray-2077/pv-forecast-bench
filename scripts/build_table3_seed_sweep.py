"""Build the paper's Table 3 (seed-variance reproducibility): one row per
array x horizon, mean skill_vs_convex +/- 1 seed standard deviation (5
seeds) for each of the five models.

Distinct from Table 5 (scripts/build_table5_component_attribution.py),
which joins this same skill data with Diebold-Mariano significance
annotations for three specific comparisons. This table is the plain
seed-spread reproducibility statistic on its own - CLAUDE.md's own
wording caution applies here more than anywhere else: the mean +/- std
columns in this table must never be read as a significance test (see
PROJECT_CHECKPOINT.md Finding 8's retraction of exactly that
substitution, and paper/WRITING_BRIEF.md Section 3 item 2).

Source: results/seed_sweep_summary_lagged.csv (45 rows: 5 models x 3
arrays x 3 horizons, 5 seeds each) - a single-source-of-truth artifact
already in the repo; this script only pivots and reformats it, no new
computation.

Writes:
  paper/tables/T3_seed_sweep.csv
  paper/tables/T3_seed_sweep.tex (booktabs fragment)

Usage:
    python scripts/build_table3_seed_sweep.py
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

ARRAYS = ["array11", "array12", "array17"]
HORIZONS = [1, 3, 6]

MODELS = ["xgboost", "lstm", "cnn_lstm", "lstm_residual", "cnn_lstm_residual"]
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "cnn_lstm": "CNN-LSTM",
    "lstm_residual": "LSTM+res",
    "cnn_lstm_residual": "CNN-LSTM+res",
}


def load_skill_summary():
    """{(array, horizon, model): (mean, std, n_seeds)}"""
    out = {}
    with open(RESULTS_DIR / "seed_sweep_summary_lagged.csv") as fh:
        for row in csv.DictReader(fh):
            key = (row["array"], int(row["horizon"]), row["model"])
            out[key] = (
                float(row["mean_skill_vs_convex"]),
                float(row["std_skill_vs_convex"]),
                int(row["n_seeds"]),
            )
    return out


def build_rows():
    skill = load_skill_summary()
    rows = []
    for array in ARRAYS:
        for horizon in HORIZONS:
            row = {"array": array, "horizon": horizon}
            for model in MODELS:
                mean, std, n_seeds = skill[(array, horizon, model)]
                row[f"{model}_mean_skill_vs_convex"] = mean
                row[f"{model}_std_skill_vs_convex"] = std
                row[f"{model}_n_seeds"] = n_seeds
            rows.append(row)
    return rows


def write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows, path):
    lines = []
    lines.append("% T3_seed_sweep.tex - seed-variance reproducibility, booktabs fragment.")
    lines.append("% NOT compile-tested (no LaTeX toolchain in this dev")
    lines.append("% environment) - same caveat as the other paper/tables/*.tex files.")
    lines.append("% Needs \\usepackage{booktabs}.")
    lines.append("% WORDING CAUTION: the mean +/- std columns here are a")
    lines.append("% reproducibility statistic, NOT a significance test - see this")
    lines.append("% script's module docstring and paper/WRITING_BRIEF.md Section 3")
    lines.append("% item 2. Significance is Table 6, not this table.")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(
        "  \\caption{Seed-variance reproducibility: mean skill\\_vs\\_convex "
        "$\\pm$ 1 standard deviation across 5 seeds, lagged regime, "
        "validation year 2014, for all five models by array and horizon. "
        "This is a reproducibility statistic, not a significance test - "
        "seed spread measures whether a result recurs on retraining, not "
        "whether one model forecasts better than another on this "
        "evaluation sample (Table \\ref{tab:T6}). The largest seed spread observed is "
        "0.017 (CNN-LSTM+res, array17, $h=6$), of the same order as the "
        "largest architecture-to-architecture difference measured "
        "anywhere in this study (0.024, LSTM vs. XGBoost, array17, "
        "$h=6$, Table \\ref{tab:T6}).}"
    )
    lines.append("  \\label{tab:T3}")
    lines.append("  \\begin{tabular}{ll rrrrr}")
    lines.append("    \\toprule")
    lines.append(
        "    Array & $h$ & XGBoost & LSTM & CNN-LSTM & LSTM+res & CNN-LSTM+res \\\\"
    )
    lines.append(
        "    & & \\multicolumn{5}{c}{skill\\_vs\\_convex, mean $\\pm$ std (5 seeds)} \\\\"
    )
    lines.append("    \\midrule")

    for row in rows:
        cells = [row["array"], str(row["horizon"])]
        for model in MODELS:
            mean = row[f"{model}_mean_skill_vs_convex"]
            std = row[f"{model}_std_skill_vs_convex"]
            cells.append(f"{mean:+.3f}$\\pm${std:.3f}")
        lines.append("    " + " & ".join(cells) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    rows = build_rows()
    for row in rows:
        print(f"{row['array']} h={row['horizon']}")
        for model in MODELS:
            mean = row[f"{model}_mean_skill_vs_convex"]
            std = row[f"{model}_std_skill_vs_convex"]
            print(f"  {MODEL_LABELS[model]:14s} {mean:+.4f} +/- {std:.4f}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "T3_seed_sweep.csv"
    tex_path = TABLES_DIR / "T3_seed_sweep.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
