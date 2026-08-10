"""Build the paper's Table 6 (Diebold-Mariano significance, architecture
comparisons): one row per array x horizon x comparison, restricted to
the four comparisons the architecture-attribution results (RQ2) turn
on - lstm vs xgboost, cnn_lstm vs lstm, lstm_residual vs lstm,
cnn_lstm_residual vs cnn_lstm - with the HLN statistic and
Holm-adjusted p-value.

NOT to be confused with scripts/build_table6_dm.py, which computes the
underlying results/table6_dm_lagged.csv by independently refitting all
7 comparators (5 models + 2 references) and running the full pairwise
Diebold-Mariano test (189 rows: 9 cells x 21 pairs). This script does no
new computation - it reads that CSV and restricts/reformats 4 of the 21
pairs, the same relationship build_table5_component_attribution.py has
to the same source file (Table 5 embeds 3 of these same 4 comparisons as
compact annotation text inside a wider table; this table is the
complementary full numeric version, all 9 cells, clean HLN/p_holm
columns, plus the 4th comparison Table 5 does not carry).

(model_1, model_2) order below matches table6_dm_lagged.csv's own fixed
model ordering (xgboost, lstm, cnn_lstm, lstm_residual,
cnn_lstm_residual, smart_persistence, convex_reference) - see
build_table5_component_attribution.py's module docstring for the same
convention. hln_stat > 0 means model_1 has lower loss (is better); < 0
means model_2 is better (src/eval/dm.py's sign convention).

Source: results/table6_dm_lagged.csv (189 rows: 9 cells x 21 pairs).

Writes:
  paper/tables/T6_dm_architecture.csv
  paper/tables/T6_dm_architecture.tex (booktabs fragment)

Usage:
    python scripts/build_table6_dm_summary.py
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

ALPHA = 0.05

# (label, model_1, model_2) - order matches table6_dm_lagged.csv's fixed
# model ordering, not the "vs" reading order a human would say aloud.
COMPARISONS = [
    ("LSTM vs XGBoost", "xgboost", "lstm"),
    ("CNN-LSTM vs LSTM", "lstm", "cnn_lstm"),
    ("LSTM+res vs LSTM", "lstm", "lstm_residual"),
    ("CNN-LSTM+res vs CNN-LSTM", "cnn_lstm", "cnn_lstm_residual"),
]

# Display labels for the "Better" column - the raw CSV values (xgboost,
# cnn_lstm, lstm_residual, cnn_lstm_residual) contain underscores that
# break LaTeX compilation if inserted unescaped. Every model key that can
# appear in better_model must be mapped here, not just the ones currently
# observed in results/table6_dm_lagged.csv, since which model "wins" a
# given cell is data-dependent.
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "cnn_lstm": "CNN-LSTM",
    "lstm_residual": "LSTM+residual",
    "cnn_lstm_residual": "CNN-LSTM+residual",
}


def load_dm_table():
    """{(array, horizon, model_1, model_2): row_dict}"""
    out = {}
    with open(RESULTS_DIR / "table6_dm_lagged.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["array"], int(row["horizon"]), row["model_1"], row["model_2"])
            out[key] = row
    return out


def build_rows():
    dm = load_dm_table()
    rows = []
    for label, m1, m2 in COMPARISONS:
        for array in ARRAYS:
            for horizon in HORIZONS:
                r = dm[(array, horizon, m1, m2)]
                p_holm = float(r["p_holm"])
                rows.append(
                    {
                        "comparison": label,
                        "array": array,
                        "horizon": horizon,
                        "hln_stat": float(r["hln_stat"]),
                        "p_holm": p_holm,
                        "significant": p_holm < ALPHA,
                        "better_model": r["better_model"],
                    }
                )
    return rows


def write_csv(rows, path):
    fieldnames = ["comparison", "array", "horizon", "hln_stat", "p_holm", "significant", "better_model"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows, path):
    lines = []
    lines.append("% T6_dm_architecture.tex - DM significance, architecture comparisons, booktabs fragment.")
    lines.append("% NOT compile-tested (no LaTeX toolchain in this dev")
    lines.append("% environment) - same caveat as the other paper/tables/*.tex files.")
    lines.append("% Needs \\usepackage{booktabs}.")
    lines.append("% SCOPE: 4 of the 21 pairwise comparisons in the full")
    lines.append("% table6_dm_lagged.csv - the ones RQ2's architecture results turn on.")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(
        "  \\caption{Diebold-Mariano significance (HAC variance, HLN "
        "small-sample correction, Holm-Bonferroni within cell), lagged "
        "regime, validation year 2014, seed 0, restricted to the four "
        "architecture comparisons RQ2's results turn on. LSTM vs. "
        "XGBoost is significant in 1 of 9 array x horizon cells "
        "(array17, $h=6$); CNN-LSTM vs. LSTM is significant in none; "
        "LSTM+residual vs. LSTM is significant in 5 of 9, with no clean "
        "horizon-based split; CNN-LSTM+residual vs. CNN-LSTM is "
        "significant in all 9, in the direction of the plain CNN-LSTM "
        "base. HLN $>0$ favours model\\_1 in each comparison's fixed "
        "(model\\_1, model\\_2) ordering (XGBoost, LSTM, LSTM, CNN-LSTM "
        "respectively); the Better column names the favoured model "
        "directly so the sign need not be tracked separately.}"
    )
    lines.append("  \\label{tab:T6}")
    lines.append("  \\begin{tabular}{ll rrl}")
    lines.append("    \\toprule")
    lines.append("    Array & $h$ & HLN & $p_{holm}$ & Better \\\\")
    lines.append("    \\midrule")

    for i, (label, m1, m2) in enumerate(COMPARISONS):
        lines.append(f"    \\multicolumn{{5}}{{l}}{{\\textit{{{label}}}}} \\\\")
        comp_rows = [r for r in rows if r["comparison"] == label]
        for r in comp_rows:
            sig = "sig" if r["significant"] else "ns"
            cells = [
                r["array"],
                str(r["horizon"]),
                f"{r['hln_stat']:+.2f}",
                f"{r['p_holm']:.4f} ({sig})",
                MODEL_LABELS.get(r["better_model"], r["better_model"]),
            ]
            lines.append("    " + " & ".join(cells) + " \\\\")
        if i != len(COMPARISONS) - 1:
            lines.append("    \\addlinespace")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    rows = build_rows()
    for r in rows:
        sig = "SIG" if r["significant"] else "ns"
        print(
            f"{r['comparison']:26s} {r['array']:8s} h={r['horizon']}  "
            f"hln={r['hln_stat']:+.3f}  p_holm={r['p_holm']:.4f} ({sig})  "
            f"better={r['better_model']}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "T6_dm_architecture.csv"
    tex_path = TABLES_DIR / "T6_dm_architecture.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
