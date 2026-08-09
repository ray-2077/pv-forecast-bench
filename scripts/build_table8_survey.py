"""Build the paper's Table 8 (literature survey evaluation practice): one
row per coded value per dimension, across the 27 papers coded in
results/literature_survey.csv, with count and percentage - the evidence
base for the survey's headline claims (C22, C27-C33 in
paper/WRITING_BRIEF.md Section 2).

SCOPE, DELIBERATE: this is a SUMMARY table, not a per-paper listing. A
full 27-row-by-17-column transcription would consume most of a column;
the eight dimensions requested here are exactly the ones the paper's
prose claims are built on. The full per-paper coding, including a
verbatim-quoted supporting audit file for 25 of the 27 papers
(evidence_level=quoted; 2 are evidence_level=summary_only, coded from
pre-existing notes with no locatable source PDF), remains in
results/literature_survey.csv and evidence/*.md - cite the CSV, do not
inline it.

Dimensions summarised, in the order requested:
  night_hours_excluded, baseline_used, skill_score_reported,
  weather_source, split_type, variance_reported, code_available,
  leakage_flag

Each dimension's value counts sum to 27 by construction (every paper is
coded on every dimension - "not_stated" is itself a coded value, not a
missing row). This script asserts that sum for every dimension and
raises if it does not hold, so a future edit to the source CSV that
drops a row silently cannot ship a table that no longer adds to 27.

Source: results/literature_survey.csv (27 rows, one per surveyed paper).

Writes:
  paper/tables/T8_survey.csv
  paper/tables/T8_survey.tex (booktabs fragment)

Usage:
    python scripts/build_table8_survey.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

SURVEY_CSV = RESULTS_DIR / "literature_survey.csv"

DIMENSIONS = [
    "night_hours_excluded",
    "baseline_used",
    "skill_score_reported",
    "weather_source",
    "split_type",
    "variance_reported",
    "code_available",
    "leakage_flag",
]

DIMENSION_LABELS = {
    "night_hours_excluded": "Night hours excluded",
    "baseline_used": "Baseline used",
    "skill_score_reported": "Skill score reported",
    "weather_source": "Weather source",
    "split_type": "Split type",
    "variance_reported": "Run-to-run variance reported",
    "code_available": "Code available",
    "leakage_flag": "Documented leakage",
}

VALUE_LABELS = {
    "not_stated": "not stated",
    "yes": "yes",
    "no": "no",
    "partial": "partial",
    "own_components": "own components only",
    "other_ML": "other ML models",
    "none": "none",
    "convex": "convex combination",
    "measured": "measured",
    "NWP_forecast": "NWP forecast",
    "reanalysis": "reanalysis",
    "chronological": "chronological",
    "k-fold": "k-fold",
    "rolling": "rolling-origin",
    "documented": "documented",
}


def load_rows():
    with open(SURVEY_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 27:
        raise ValueError(
            f"expected 27 rows in {SURVEY_CSV}, got {len(rows)} - "
            "this table's n=27 caption depends on that count"
        )
    return rows


def build_rows():
    rows = load_rows()
    n_total = len(rows)

    out = []
    for dim in DIMENSIONS:
        counts = Counter(r[dim] for r in rows)
        dim_sum = sum(counts.values())
        if dim_sum != n_total:
            raise ValueError(
                f"dimension {dim!r} counts sum to {dim_sum}, expected {n_total}"
            )
        for value, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append(
                {
                    "dimension": dim,
                    "value": value,
                    "n": n,
                    "pct": round(100 * n / n_total, 1),
                }
            )
    return out


def write_csv(rows, path):
    fieldnames = ["dimension", "value", "n", "pct"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows, path):
    lines = []
    lines.append("% T8_survey.tex - literature survey evaluation practice, booktabs fragment.")
    lines.append("% NOT compile-tested (no LaTeX toolchain in this dev")
    lines.append("% environment) - same caveat as the other paper/tables/*.tex")
    lines.append("% files and paper/figures/F7_pipeline.tex. Needs")
    lines.append("% \\usepackage{booktabs}.")
    lines.append("% SUMMARY table: one row per coded value per dimension, n=27 papers.")
    lines.append("% Full per-paper coding is in results/literature_survey.csv, not here.")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(
        "  \\caption{Literature survey evaluation practice ($n=27$). See "
        "paper/tables/CAPTIONS.md for the full caption.}"
    )
    lines.append("  \\label{tab:survey}")
    lines.append("  \\begin{tabular}{llrr}")
    lines.append("    \\toprule")
    lines.append("    Dimension & Value & $n$ & \\% \\\\")
    lines.append("    \\midrule")

    for i, dim in enumerate(DIMENSIONS):
        dim_rows = [r for r in rows if r["dimension"] == dim]
        for j, r in enumerate(dim_rows):
            label = DIMENSION_LABELS[dim] if j == 0 else ""
            value_label = VALUE_LABELS.get(r["value"], r["value"])
            lines.append(
                "    "
                + " & ".join(
                    [
                        label,
                        value_label,
                        str(r["n"]),
                        f"{r['pct']:.1f}",
                    ]
                )
                + " \\\\"
            )
        if i != len(DIMENSIONS) - 1:
            lines.append("    \\addlinespace")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    rows = build_rows()
    for r in rows:
        label = DIMENSION_LABELS[r["dimension"]]
        value_label = VALUE_LABELS.get(r["value"], r["value"])
        print(f"{label:30s} {value_label:22s} n={r['n']:2d}  {r['pct']:5.1f}%")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "T8_survey.csv"
    tex_path = TABLES_DIR / "T8_survey.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
