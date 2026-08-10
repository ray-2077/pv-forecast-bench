"""Build the paper's Table 4 (protocol-inflation configurations): one
row per protocol configuration C1-C6, for site 11 (array11), all three
horizons - n_samples, nRMSE, skill, and the reference used.

NOT to be confused with scripts/build_table4_protocol.py, which computes
the underlying results/table4_protocol_lagged.csv by re-running the
protocol-sensitivity comparison directly against processed data (54
rows: 3 arrays x 3 horizons x 6 configs). This script does no new
computation - it reads that CSV and formats a subset of it as a LaTeX
table, the same relationship build_table5_component_attribution.py and
build_table7_sky.py already have to their own source CSVs.

SCOPE, DELIBERATE: array11 only, not all three arrays (54 rows would not
fit one float). This is the paper's headline protocol-inflation table -
the four effects in CLAUDE.md rule 4 / the RQ1 results section - and
array11 is the array used as the running example throughout the
Results text (Section VI-A) for exactly this reason. array12 and
array17 show the same pattern (see the Results prose, which reports the
convex-vs-persistence numbers for all three); restricting the table
itself to one array is stated explicitly in the caption below, per the
brief for this table.

CONFIG LEGEND (config_id -> config_description, from
results/table4_protocol_lagged.csv, reproduced here only as a print aid
- the authoritative text is the CSV's own config_description column,
asserted to match on load):
  C1: daylight only, skill vs convex reference (correct protocol)
  C2: daylight only, skill vs smart persistence
  C3: convex-covered hours only, skill vs convex reference
  C4: all 24 hours, skill vs smart persistence
  C5: daylight only, raw nRMSE, no skill score
  C6: all 24 hours, raw nRMSE, no skill score

Source: results/table4_protocol_lagged.csv (54 rows: 3 arrays x 3
horizons x 6 configs).

Writes:
  paper/tables/T4_protocol.csv
  paper/tables/T4_protocol.tex (booktabs fragment)

Usage:
    python scripts/build_table4_config_summary.py
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

ARRAY = "array11"
HORIZONS = [1, 3, 6]
CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6"]

REFERENCE_LABELS = {
    "convex_reference": "convex",
    "smart_persistence": "persistence",
    "none": "--",
}

# Asserted against the CSV's own config_description column at load time
# (see load_rows) so this table fails loudly if build_table4_protocol.py
# ever changes what a config ID means, rather than silently mislabeling.
EXPECTED_DESCRIPTIONS = {
    "C1": "daylight only, skill vs convex reference",
    "C2": "daylight only, skill vs smart persistence",
    "C3": "convex-covered hours only, skill vs convex reference",
    "C4": "all 24 hours, skill vs smart persistence",
    "C5": "daylight only, raw nRMSE, no skill score",
    "C6": "all 24 hours, raw nRMSE, no skill score",
}


def load_rows():
    with open(RESULTS_DIR / "table4_protocol_lagged.csv", newline="") as fh:
        all_rows = list(csv.DictReader(fh))

    filtered = [r for r in all_rows if r["array"] == ARRAY]
    expected = len(HORIZONS) * len(CONFIGS)
    if len(filtered) != expected:
        raise ValueError(f"{ARRAY}: expected {expected} rows, got {len(filtered)}")

    for r in filtered:
        expected_prefix = EXPECTED_DESCRIPTIONS[r["config_id"]]
        if not r["config_description"].startswith(expected_prefix):
            raise ValueError(
                f"{r['config_id']}: config_description changed upstream - "
                f"expected prefix {expected_prefix!r}, got {r['config_description']!r}"
            )
    return filtered


def build_rows():
    raw = {(r["horizon"], r["config_id"]): r for r in load_rows()}
    rows = []
    for horizon in HORIZONS:
        for config_id in CONFIGS:
            r = raw[(str(horizon), config_id)]
            skill = r["skill"]
            rows.append(
                {
                    "horizon": horizon,
                    "config_id": config_id,
                    "n_samples": int(r["n_samples"]),
                    "nrmse_pct": float(r["nrmse"]),
                    "skill": float(skill) if skill else None,
                    "reference_used": r["reference_used"],
                }
            )
    return rows


def write_csv(rows, path):
    fieldnames = ["horizon", "config_id", "n_samples", "nrmse_pct", "skill", "reference_used"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows, path):
    lines = []
    lines.append("% T4_protocol.tex - protocol-inflation configurations (array11), booktabs fragment.")
    lines.append("% NOT compile-tested (no LaTeX toolchain in this dev")
    lines.append("% environment) - same caveat as the other paper/tables/*.tex files.")
    lines.append("% Needs \\usepackage{booktabs}.")
    lines.append("% SCOPE: array11 only - see this script's module docstring for why.")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(
        "  \\caption{Protocol configurations, site 11 (array11), lagged "
        "regime, validation year 2014: sample count, normalised RMSE, "
        "skill score, and reference forecast for six evaluation-protocol "
        "configurations (C1-C6) at each of the three horizons. C1: "
        "daylight hours only, skill against the convex reference (the "
        "protocol used throughout this paper). C2: daylight hours only, "
        "skill against smart persistence. C3: hours where the convex "
        "reference has a prediction (not a full 24-hour cycle - included "
        "only to document this coverage restriction). C4: all 24 hours, "
        "skill against smart persistence. C5 and C6: raw nRMSE at "
        "daylight-only and all-24-hours, no reference forecast. Comparing "
        "C5 against C6 shows the closed-form night-hour deflation; "
        "comparing C1 against C2 shows the reference-forecast effect - at "
        "$h=6$, skill falls from $+0.652$ against persistence to "
        "$+0.194$ against the convex reference, using the identical "
        "forecasts. Restricted to array11; sites 12 and 17 show the same "
        "pattern (reported in the Results text).}"
    )
    lines.append("  \\label{tab:T4}")
    lines.append("  \\begin{tabular}{cl rrrl}")
    lines.append("    \\toprule")
    lines.append(
        "    $h$ & Config & $n$ & nRMSE (\\%) & skill & Reference \\\\"
    )
    lines.append("    \\midrule")

    for i, horizon in enumerate(HORIZONS):
        h_rows = [r for r in rows if r["horizon"] == horizon]
        for j, r in enumerate(h_rows):
            h_cell = str(r["horizon"]) if j == 0 else ""
            skill_cell = f"{r['skill']:+.3f}" if r["skill"] is not None else "--"
            ref_cell = REFERENCE_LABELS.get(r["reference_used"], r["reference_used"])
            cells = [
                h_cell,
                r["config_id"],
                str(r["n_samples"]),
                f"{r['nrmse_pct']:.2f}",
                skill_cell,
                ref_cell,
            ]
            lines.append("    " + " & ".join(cells) + " \\\\")
        if i != len(HORIZONS) - 1:
            lines.append("    \\addlinespace")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    rows = build_rows()
    for r in rows:
        skill_str = f"{r['skill']:+.4f}" if r["skill"] is not None else "n/a"
        print(
            f"h={r['horizon']} {r['config_id']}  n={r['n_samples']:5d}  "
            f"nrmse={r['nrmse_pct']:6.2f}%  skill={skill_str}  ref={r['reference_used']}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "T4_protocol.csv"
    tex_path = TABLES_DIR / "T4_protocol.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
