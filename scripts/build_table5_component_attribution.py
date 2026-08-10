"""Build the paper's Table 5 (RQ2 component-attribution summary): one
row per array x horizon, all five models' mean skill_vs_convex (seed
std in the same cell), plus a Diebold-Mariano significance annotation
for the three comparisons RQ2's Results actually turns on -
lstm vs xgboost, cnn_lstm vs lstm, lstm_residual vs lstm (Findings 8-11).

Not every pairwise DM comparison is included here on purpose - Table 6
already has the full 21-pair matrix; this table is the condensed
component-attribution VIEW the Section 6 outline (6.2 RQ2) asks for,
built from the same two committed CSVs rather than a new computation:

  results/seed_sweep_summary_lagged.csv   mean/std skill_vs_convex, per
                                           model x array x horizon (5
                                           seeds)
  results/table6_dm_lagged.csv            pairwise DM significance
                                           (HAC variance, HLN small-
                                           sample correction, Holm-
                                           Bonferroni within cell),
                                           single seed=0

(both filenames above are the --eval-split val default; see CHANGE LOG
below for the test-split filenames)

Both CSVs are lagged-regime only (CLAUDE.md rule 5) and both are
single-source-of-truth artifacts already in the repo - this script only
joins and reformats them, it fits nothing and reads no run JSON
directly.

CHANGE LOG (2026-08-09): added --eval-split {val,test}, default val,
same strict-argparse shape as scripts/aggregate_seed_sweep.py and
scripts/build_table6_dm.py. val reads results/seed_sweep_summary_
<regime>.csv and results/table6_dm_<regime>.csv and writes paper/tables/
T5_component_attribution.csv/.tex - all UNCHANGED from before this flag
existed. test reads the _test-suffixed counterparts (results/
seed_sweep_summary_<regime>_test.csv, results/table6_dm_<regime>_test.csv)
and writes DISTINCT, _test-suffixed output files (paper/tables/
T5_component_attribution_test.csv/.tex) - never the same path as the val
output. Running with --eval-split test requires those two input CSVs to
already exist: results/seed_sweep_summary_<regime>_test.csv comes from
`aggregate_seed_sweep.py --eval-split test --regime <regime>`; results/
table6_dm_<regime>_test.csv comes from `build_table6_dm.py --eval-split
test --regime <regime>`, which refits every model against the test split
and is not run automatically by this script.

table6_dm_lagged.csv orders each pair (model_1, model_2) by a fixed
model order (xgboost, lstm, cnn_lstm, lstm_residual, cnn_lstm_residual,
smart_persistence, convex_reference), never the reverse - the three
comparisons this table needs are looked up as (xgboost, lstm),
(lstm, cnn_lstm), (lstm, lstm_residual) accordingly. `better_model`
in that CSV already resolves the sign for us (whichever model the DM
test favours, independent of significance), so the annotation here
reports better_model + hln_stat + p_holm rather than re-deriving a
sign from dbar/dm_stat.

WORDING CAUTION (CLAUDE.md / paper/WRITING_BRIEF.md Section 3): do not
read a "not significant" cell here as "no difference" - it means the
DM test did not clear p_holm<0.05 on this one array x horizon cell,
computed at a single seed (seed=0) over ~3700 paired daylight-hour
forecast errors. Seed spread (Table 3) is a reproducibility statistic,
not a substitute significance test - see PROJECT_CHECKPOINT.md
Finding 8's own retraction of exactly that substitution.

Writes:
  paper/tables/T5_component_attribution.csv
  paper/tables/T5_component_attribution.tex (booktabs fragment)
(or the _test-suffixed counterparts - see CHANGE LOG above)

Usage:
    python scripts/build_table5_component_attribution.py [--eval-split {val,test}]
"""

import argparse
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

# (label, model_1, model_2) - the (model_1, model_2) order must match
# table6_dm_lagged.csv's own fixed model ordering (see module docstring),
# not the "vs" reading order a human would say out loud.
DM_COMPARISONS = [
    ("lstm vs xgboost", "xgboost", "lstm"),
    ("cnn_lstm vs lstm", "lstm", "cnn_lstm"),
    ("lstm_residual vs lstm", "lstm", "lstm_residual"),
]

ALPHA = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    # parse_args(), not parse_known_args() - an unrecognised flag must be a
    # hard error, same reasoning as scripts/aggregate_seed_sweep.py and
    # scripts/build_table6_dm.py's --regime/--eval-split flags.
    return parser.parse_args()


def load_skill_summary(suffix):
    """{(array, horizon, model): (mean, std)}"""
    out = {}
    with open(RESULTS_DIR / f"seed_sweep_summary_lagged{suffix}.csv") as fh:
        for row in csv.DictReader(fh):
            key = (row["array"], int(row["horizon"]), row["model"])
            out[key] = (
                float(row["mean_skill_vs_convex"]),
                float(row["std_skill_vs_convex"]),
            )
    return out


def load_dm_table(suffix):
    """{(array, horizon, model_1, model_2): row_dict}"""
    out = {}
    with open(RESULTS_DIR / f"table6_dm_lagged{suffix}.csv") as fh:
        for row in csv.DictReader(fh):
            key = (row["array"], int(row["horizon"]), row["model_1"], row["model_2"])
            out[key] = row
    return out


def format_dm_cell(dm_row):
    if dm_row is None:
        return "n/a"
    p_holm = float(dm_row["p_holm"])
    hln_stat = float(dm_row["hln_stat"])
    better = MODEL_LABELS.get(dm_row["better_model"], dm_row["better_model"])
    sig = "sig" if p_holm < ALPHA else "ns"
    return f"{better} better, hln={hln_stat:+.2f}, p_holm={p_holm:.4f} ({sig})"


# Short tokens for the LaTeX table's DM columns - the full "X better,
# hln=+1.23, p_holm=0.0456 (sig)" string used in the console printout and
# CSV is too wide for a two-column IEEE page across 3 columns x 9 rows.
# The full HLN/p_holm numbers are not lost - Table 6 carries all of them
# (plus a 4th comparison this table omits) - so the compact cell here is
# just the winner with a significance marker, not a second source of the
# same numbers.
SHORT_LABELS = {
    "xgboost": "XGB",
    "lstm": "LSTM",
    "cnn_lstm": "CNN",
    "lstm_residual": "L+res",
    "cnn_lstm_residual": "CL+res",
}


def format_dm_cell_compact(dm_row):
    if dm_row is None:
        return "n/a"
    p_holm = float(dm_row["p_holm"])
    better = SHORT_LABELS.get(dm_row["better_model"], dm_row["better_model"])
    return f"{better}*" if p_holm < ALPHA else better


def build_rows(suffix):
    skill = load_skill_summary(suffix)
    dm = load_dm_table(suffix)

    rows = []
    for array in ARRAYS:
        for horizon in HORIZONS:
            row = {"array": array, "horizon": horizon}
            for model in MODELS:
                mean, std = skill[(array, horizon, model)]
                row[f"{model}_mean_skill_vs_convex"] = mean
                row[f"{model}_std_skill_vs_convex"] = std

            for label, m1, m2 in DM_COMPARISONS:
                dm_row = dm.get((array, horizon, m1, m2))
                key = label.replace(" ", "_")
                if dm_row is None:
                    row[f"dm_{key}_better"] = ""
                    row[f"dm_{key}_hln_stat"] = ""
                    row[f"dm_{key}_p_holm"] = ""
                    row[f"dm_{key}_significant"] = ""
                else:
                    row[f"dm_{key}_better"] = dm_row["better_model"]
                    row[f"dm_{key}_hln_stat"] = float(dm_row["hln_stat"])
                    row[f"dm_{key}_p_holm"] = float(dm_row["p_holm"])
                    row[f"dm_{key}_significant"] = float(dm_row["p_holm"]) < ALPHA
            rows.append(row)
    return rows


def write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%")


def write_latex(rows, dm, path):
    lines = []
    lines.append("% T5_component_attribution.tex - RQ2 summary, booktabs fragment.")
    lines.append("% NOT compile-tested (no LaTeX toolchain in this dev")
    lines.append("% environment) - same caveat as paper/tables/T1_dataset.tex,")
    lines.append("% T2_features.tex, and paper/figures/F7_pipeline.tex. Needs")
    lines.append("% \\usepackage{booktabs}.")
    lines.append("% Wide (5 models x mean+-std, plus 3 DM columns): written as")
    lines.append("% table* (spans both IEEE columns), \\scriptsize. DM columns")
    lines.append("% carry only a winner + significance marker (full HLN/p_holm")
    lines.append("% is Table 6) - compacted from the full annotation string,")
    lines.append("% which was this table's page-width overflow source.")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\scriptsize")
    lines.append(
        "  \\caption{Component attribution (RQ2): skill\\_vs\\_convex by "
        "model (XGBoost, LSTM, CNN-LSTM, LSTM+residual, "
        "CNN-LSTM+residual), array, and horizon - mean $\\pm$ 1 seed "
        "standard deviation across 5 seeds - with a Diebold-Mariano "
        "significance marker for the three comparisons RQ2's Results "
        "turns on: LSTM vs. XGBoost, CNN-LSTM vs. LSTM, and "
        "LSTM+residual vs. LSTM. Each DM column names the winning model; "
        "an asterisk marks Holm-Bonferroni significance at $\\alpha=0.05$ "
        "(HAC variance, HLN small-sample correction, single seed=0). "
        "Full HLN statistics and Holm-adjusted p-values for these three "
        "comparisons, plus a fourth (CNN-LSTM+residual vs. CNN-LSTM) "
        "this table omits, are in Table \\ref{tab:T6}. At $h=1$ and "
        "$h=3$, XGBoost and LSTM are statistically indistinguishable on "
        "every array ($p_{holm}=1.0$ throughout); at $h=6$, LSTM's edge "
        "over XGBoost is significant on only one of three co-located "
        "arrays (array17). The residual stage is significant in the "
        "direction of the plain base model in 5 of 9 lstm\\_residual "
        "cells, with no clean horizon-based split - array12 $h=1$ is "
        "significant, array12 $h=6$ is not, despite being the longer "
        "horizon.}"
    )
    lines.append("  \\label{tab:T5}")
    lines.append("  \\begin{tabular}{ll rrrrr lll}")
    lines.append("    \\toprule")
    lines.append(
        "    & & \\multicolumn{5}{c}{skill\\_vs\\_convex, mean $\\pm$ std (5 seeds)} & "
        "\\multicolumn{3}{c}{Better (DM sig., Table \\ref{tab:T6})} \\\\"
    )
    lines.append("    \\cmidrule(lr){3-7} \\cmidrule(lr){8-10}")
    lines.append(
        "    Array & $h$ & XGBoost & LSTM & CNN-LSTM & LSTM+res & CNN-LSTM+res & "
        "vs XGB & vs LSTM & +res vs LSTM \\\\"
    )
    lines.append("    \\midrule")

    for row in rows:
        cells = [tex_escape(row["array"]), str(row["horizon"])]
        for model in MODELS:
            mean = row[f"{model}_mean_skill_vs_convex"]
            std = row[f"{model}_std_skill_vs_convex"]
            cells.append(f"{mean:+.3f}$\\pm${std:.3f}")
        for label, m1, m2 in DM_COMPARISONS:
            dm_row = dm.get((row["array"], row["horizon"], m1, m2))
            cells.append(tex_escape(format_dm_cell_compact(dm_row)))
        lines.append("    " + " & ".join(cells) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    eval_split = parse_args().eval_split
    # val output paths are UNCHANGED from before --eval-split existed
    # (suffix ""); test gets an explicit _test suffix on every input and
    # output filename, a distinct path rather than just a distinct flag
    # value, so it can never overwrite or shadow the val artifacts.
    suffix = "" if eval_split == "val" else "_test"

    rows = build_rows(suffix)
    dm = load_dm_table(suffix)

    print(f"eval_split={eval_split!r}\n")
    for row in rows:
        print(f"{row['array']} h={row['horizon']}")
        for model in MODELS:
            mean = row[f"{model}_mean_skill_vs_convex"]
            std = row[f"{model}_std_skill_vs_convex"]
            print(f"  {MODEL_LABELS[model]:14s} {mean:+.4f} +/- {std:.4f}")
        for label, m1, m2 in DM_COMPARISONS:
            dm_row = dm.get((row["array"], row["horizon"], m1, m2))
            print(f"  DM {label:24s} {format_dm_cell(dm_row)}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / f"T5_component_attribution{suffix}.csv"
    tex_path = TABLES_DIR / f"T5_component_attribution{suffix}.tex"
    write_csv(rows, csv_path)
    write_latex(rows, dm, tex_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
