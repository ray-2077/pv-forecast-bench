"""Build the paper's Table 7 (RQ3 sky-condition results): one row per
sky class per model, nRMSE and skill_vs_convex, array11 h=3 only, with
the class counts - the tabular form of Figure F3
(paper/figures/F3_error_by_sky_condition.pdf), same cell, same source.

SCOPE, DELIBERATE: array11 h=3 only, not the full 9 array x horizon
grid results/table_sky.csv actually has (81 rows: 3 models x 3 arrays x
3 horizons x 3 classes). A full condensation across all 9 cells is a
separate, not-yet-built need - see paper/WRITING_BRIEF.md Section 7's
T7 row and Section 5 item 8's pooling-duplication warning (summing n
across arrays double/triple-counts the same clock-hours, since sky
classification depends only on the shared weather-station GHI signal,
identical across arrays and models - see PROJECT_CHECKPOINT.md Finding
12 Part B's 2026-08-08 addendum for the correct multi-horizon pooling
if that full table is built later). This script deliberately does not
attempt that; it reads exactly the one cell F3 already visualizes.

Source: results/table_sky.csv, filtered to array=array11, horizon=3.

Writes:
  paper/tables/T7_sky_condition.csv
  paper/tables/T7_sky_condition.tex (booktabs fragment)

Usage:
    python scripts/build_table7_sky.py
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

ARRAY = "array11"
HORIZON = 3

MODELS = ["xgboost", "lstm", "lstm_residual"]
MODEL_LABELS = {"xgboost": "XGBoost", "lstm": "LSTM", "lstm_residual": "LSTM+res"}

SKY_CLASSES = ["clear", "partly_cloudy", "overcast"]
SKY_LABELS = {"clear": "clear", "partly_cloudy": "partly cloudy", "overcast": "overcast"}


def load_rows():
    with open(RESULTS_DIR / "table_sky.csv") as fh:
        all_rows = list(csv.DictReader(fh))

    filtered = [
        r for r in all_rows
        if r["array"] == ARRAY and int(r["horizon"]) == HORIZON
    ]
    expected = len(MODELS) * len(SKY_CLASSES)
    if len(filtered) != expected:
        raise ValueError(
            f"{ARRAY} h={HORIZON}: expected {expected} rows in "
            f"table_sky.csv, got {len(filtered)}"
        )
    return filtered


def build_rows():
    raw = {(r["model"], r["sky_class"]): r for r in load_rows()}

    rows = []
    for sky_class in SKY_CLASSES:
        for model in MODELS:
            r = raw[(model, sky_class)]
            rows.append(
                {
                    "sky_class": sky_class,
                    "model": model,
                    "n": int(r["n"]),
                    "nrmse_pct": float(r["nrmse"]),
                    "skill_vs_convex": float(r["skill_vs_convex"]),
                }
            )
    return rows


def write_csv(rows, path):
    fieldnames = ["sky_class", "model", "n", "nrmse_pct", "skill_vs_convex"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows, path):
    lines = []
    lines.append("% T7_sky_condition.tex - RQ3 sky-condition results, booktabs fragment.")
    lines.append("% NOT compile-tested (no LaTeX toolchain in this dev")
    lines.append("% environment) - same caveat as the other paper/tables/*.tex")
    lines.append("% files and paper/figures/F7_pipeline.tex. Needs")
    lines.append("% \\usepackage{booktabs}.")
    lines.append(f"% array11, h={HORIZON} only - see this script's module docstring")
    lines.append("% for why the full 9-cell version is a separate, unbuilt need.")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(
        "  \\caption{Sky-condition results (RQ3), array11, $h=3$. See "
        "paper/tables/CAPTIONS.md for the full caption.}"
    )
    lines.append("  \\label{tab:sky-condition}")
    lines.append("  \\begin{tabular}{llrrr}")
    lines.append("    \\toprule")
    lines.append("    Sky class & Model & $n$ & nRMSE (\\%) & skill vs convex \\\\")
    lines.append("    \\midrule")

    for i, sky_class in enumerate(SKY_CLASSES):
        class_rows = [r for r in rows if r["sky_class"] == sky_class]
        for j, r in enumerate(class_rows):
            label = SKY_LABELS[sky_class] if j == 0 else ""
            n_cell = str(r["n"]) if j == 0 else ""
            lines.append(
                "    "
                + " & ".join(
                    [
                        label,
                        MODEL_LABELS[r["model"]],
                        n_cell,
                        f"{r['nrmse_pct']:.2f}",
                        f"{r['skill_vs_convex']:+.3f}",
                    ]
                )
                + " \\\\"
            )
        if i != len(SKY_CLASSES) - 1:
            lines.append("    \\addlinespace")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    rows = build_rows()
    for r in rows:
        print(
            f"{r['sky_class']:14s} {MODEL_LABELS[r['model']]:10s} n={r['n']:4d}  "
            f"nRMSE={r['nrmse_pct']:6.2f}%  skill_vs_convex={r['skill_vs_convex']:+.4f}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "T7_sky_condition.csv"
    tex_path = TABLES_DIR / "T7_sky_condition.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
