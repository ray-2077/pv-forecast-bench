"""Build the paper's Table 1 (dataset summary): one row per array, plus
an excluded-array footnote row for array07.

SOURCES, per array-metadata column:
  - nameplate kW: scripts/build_processed.py's ARRAYS dict.
  - tilt/azimuth: src/data/clearsky_power.py's SURFACE_TILT/SURFACE_AZIMUTH
    constants (build_processed.py does NOT have these - the user-given
    source list for this table named build_processed.py, but tilt/azimuth
    live in clearsky_power.py; noting the discrepancy here rather than
    silently citing the wrong file).
  - DKASC site number, manufacturer, technology, DKASC slug: NOT in any
    script or committed CSV. Hardcoded below from
    PROJECT_CHECKPOINT.md Section 2's array table, itself sourced "per
    DKASC's own technology pages". If DKASC's public pages are ever
    re-checked, update ARRAY_METADATA below and this comment together.
  - install date: only array17's is documented anywhere in this project
    (11 March 2010 - CLAUDE.md "Data window", PROJECT_CHECKPOINT.md
    Section 2). array11/array12's install dates are NOT recorded in
    this project (their data simply starts within the processed window
    with no install-event marker) - reported as "not recorded" rather
    than guessed.

SOURCES, per data-volume column:
  - n_rows_total/train/val/test: computed directly here via
    src.data.splits.split_chronological on each array's processed
    hourly parquet (data/processed/*.parquet) - not from a committed
    CSV, since none has exactly this number. Cheap (no model fitting,
    no clear-sky columns needed for a row count) and this is the same
    61,344/26,304/8,760/8,760 pattern already stated in CLAUDE.md's
    "Data window" section and PROJECT_CHECKPOINT.md claim C36 - this
    script verifies it directly rather than re-typing it.
  - n_daylight_{train,val,test}: summed from results/dead_period_audit.csv's
    n_daylight_hours column over the relevant split years (TRAIN
    2011-2013, VAL 2014, TEST 2015) - a committed audit artifact a
    reader can check without rerunning anything. Cross-checked once
    during development against a live split_chronological +
    is_daylight computation for all three used arrays: exact match
    (11414/3799/3802 for every array, since daylight hours depend only
    on the shared weather-station solar geometry, not per-array data -
    see CLAUDE.md's "co-located arrays, not sites" framing).

ARRAY07 FOOTNOTE ROW: array07 was dropped from the current pipeline (not
in build_processed.py's ARRAYS dict, no longer in src/data/pipeline.py's
ARRAYS), but data/processed/array07_CdTe_hourly.parquet still exists on
disk (kept as evidence, per CLAUDE.md) so its row/daylight counts are
computed and reported the same way as the three used arrays - this
table is the exclusion's evidence, not just an assertion of it. The
reason column pairs results/data_audit.csv's array07/2014 row (99.99%
coverage, 0.00% NaN - a completeness audit finds nothing wrong) against
results/dead_period_audit.csv's array07/2014 row (48.41% of daylight
hours exactly zero power, status FAIL) - Finding 6's exact point: a
completeness-based audit is structurally blind to a healthy sensor
reporting a dead array.

Writes:
  paper/tables/T1_dataset.csv
  paper/tables/T1_dataset.tex (booktabs fragment)

Usage:
    python scripts/build_table1_dataset.py
"""

import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import split_chronological

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

# (parquet filename, dkasc_site, manufacturer, technology, nameplate_kw,
#  tilt_deg, azimuth_deg, install_date, dkasc_slug, used_in_evaluation)
# tilt/azimuth from src/data/clearsky_power.py (SURFACE_TILT=20.0,
# SURFACE_AZIMUTH=0.0) - stated there as shared across all fixed-mount
# arrays; not independently confirmed for array07 specifically (it was
# dropped before that generalisation was written), so array07's
# tilt/azimuth are left as "not recorded" rather than assumed equal.
ARRAY_METADATA = {
    "array11": {
        "parquet": "array11_polySi_hourly.parquet",
        "site": "11",
        "manufacturer": "BP Solar",
        "technology": "poly-Si",
        "nameplate_kw": 5.0,
        "tilt_deg": 20.0,
        "azimuth_deg": 0.0,
        "install_date": "not recorded",
        "slug": "dka-m5-c-phase",
        "used": True,
    },
    "array12": {
        "parquet": "array12_monoSi_hourly.parquet",
        "site": "12",
        "manufacturer": "BP Solar",
        "technology": "mono-Si",
        "nameplate_kw": 5.1,
        "tilt_deg": 20.0,
        "azimuth_deg": 0.0,
        "install_date": "not recorded",
        "slug": "dka-m5-b-phase",
        "used": True,
    },
    "array17": {
        "parquet": "array17_HIT_hourly.parquet",
        "site": "17",
        "manufacturer": "Sanyo",
        "technology": "HIT",
        "nameplate_kw": 6.3,
        "tilt_deg": 20.0,
        "azimuth_deg": 0.0,
        "install_date": "2010-03-11",
        "slug": "dka-m4-b-phase",
        "used": True,
    },
    "array07": {
        "parquet": "array07_CdTe_hourly.parquet",
        "site": "07",
        "manufacturer": "First Solar",
        "technology": "CdTe",
        "nameplate_kw": 7.0,
        "tilt_deg": None,
        "azimuth_deg": None,
        "install_date": "not recorded",
        "slug": "dka-m6-a-phase",
        "used": False,
    },
}

SPLIT_YEAR_GROUPS = {
    "train": (2011, 2012, 2013),
    "val": (2014,),
    "test": (2015,),
}


def row_counts(parquet_filename):
    df = pd.read_parquet(PROCESSED_DIR / parquet_filename)
    train, val, test = split_chronological(df)
    return len(df), len(train), len(val), len(test)


def daylight_hour_sums():
    """n_daylight_hours per (array, split) from results/dead_period_audit.csv,
    summed over each split's years. Returns {array: {split: n}}.
    """
    with open(RESULTS_DIR / "dead_period_audit.csv") as fh:
        rows = list(csv.DictReader(fh))

    by_array_year = {}
    for row in rows:
        by_array_year[(row["array"], int(row["year"]))] = int(row["n_daylight_hours"])

    out = {}
    for array in ARRAY_METADATA:
        out[array] = {}
        for split, years in SPLIT_YEAR_GROUPS.items():
            out[array][split] = sum(by_array_year[(array, y)] for y in years)
    return out


def array07_exclusion_reason():
    """Pairs the completeness-audit numbers against the dead-period
    numbers for array07/2014, per Finding 6 - see module docstring.
    """
    with open(RESULTS_DIR / "data_audit.csv") as fh:
        audit_rows = {
            (r["array"], int(r["year"])): r for r in csv.DictReader(fh)
        }
    with open(RESULTS_DIR / "dead_period_audit.csv") as fh:
        dead_rows = {
            (r["array"], int(r["year"])): r for r in csv.DictReader(fh)
        }

    coverage = audit_rows[("array07_CdTe", 2014)]
    dead = dead_rows[("array07", 2014)]
    return (
        f"Completeness audit (results/data_audit.csv) passed array07 2014 at "
        f"{float(coverage['coverage_pct']):.2f}% coverage, "
        f"{float(coverage['nan_pct_Active_Power']):.2f}% NaN in Active_Power. "
        f"Dead-period audit (results/dead_period_audit.csv) found "
        f"{float(dead['pct_zero_power_daylight']):.2f}% of 2014 daylight hours "
        f"at exactly zero power (status={dead['status']}), plus a 48-day "
        f"near-zero run in Nov-Dec 2015 (inside the test year). A "
        f"completeness-only audit is structurally blind to a healthy sensor "
        f"reporting a dead array (Finding 6)."
    )


def build_rows():
    daylight = daylight_hour_sums()
    rows = []
    for array_key, meta in ARRAY_METADATA.items():
        n_total, n_train, n_val, n_test = row_counts(meta["parquet"])
        d = daylight[array_key]
        rows.append(
            {
                "array": array_key,
                "dkasc_site": meta["site"],
                "dkasc_slug": meta["slug"],
                "manufacturer": meta["manufacturer"],
                "technology": meta["technology"],
                "nameplate_kw": meta["nameplate_kw"],
                "tilt_deg": meta["tilt_deg"],
                "azimuth_deg": meta["azimuth_deg"],
                "install_date": meta["install_date"],
                "n_rows_total": n_total,
                "n_rows_train": n_train,
                "n_rows_val": n_val,
                "n_rows_test": n_test,
                "n_daylight_train": d["train"],
                "n_daylight_val": d["val"],
                "n_daylight_test": d["test"],
                "used_in_evaluation": meta["used"],
                "exclusion_reason": (
                    "" if meta["used"] else array07_exclusion_reason()
                ),
            }
        )
    return rows


def write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value):
    s = str(value)
    return s.replace("_", "\\_").replace("%", "\\%")


def write_latex(rows, path):
    used_rows = [r for r in rows if r["used_in_evaluation"]]
    excluded_rows = [r for r in rows if not r["used_in_evaluation"]]

    lines = []
    lines.append("% T1_dataset.tex - dataset summary, booktabs fragment.")
    lines.append("% NOT compile-tested (no LaTeX toolchain in this dev")
    lines.append("% environment) - verify before camera-ready, same caveat")
    lines.append("% as paper/figures/F7_pipeline.tex. Needs \\usepackage{booktabs}.")
    lines.append("% Wide (13 data columns): written as table* to span both")
    lines.append("% IEEE columns; \\small or \\scriptsize will likely still")
    lines.append("% be needed to fit - not tuned here without a real render.")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\caption{Dataset summary. See paper/tables/CAPTIONS.md for the full caption.}")
    lines.append("  \\label{tab:dataset}")
    lines.append("  \\begin{tabular}{lllrrrl rrrr rrr}")
    lines.append("    \\toprule")
    lines.append(
        "    & \\multicolumn{6}{c}{Array identity} & "
        "\\multicolumn{4}{c}{Rows (total / train / val / test)} & "
        "\\multicolumn{3}{c}{Daylight hours (train / val / test)} \\\\"
    )
    lines.append("    \\cmidrule(lr){2-7} \\cmidrule(lr){8-11} \\cmidrule(lr){12-14}")
    lines.append(
        "    Array & Site & Manuf. & Tech. & kW & Tilt/Azim. & Installed & "
        "Total & Train & Val & Test & Train & Val & Test \\\\"
    )
    lines.append("    \\midrule")
    for r in used_rows:
        lines.append(
            "    "
            + " & ".join(
                [
                    tex_escape(r["array"]),
                    tex_escape(r["dkasc_site"]),
                    tex_escape(r["manufacturer"]),
                    tex_escape(r["technology"]),
                    f"{r['nameplate_kw']:.1f}",
                    f"{r['tilt_deg']:.0f}$^\\circ$/{r['azimuth_deg']:.0f}$^\\circ$",
                    tex_escape(r["install_date"]),
                    f"{r['n_rows_total']:,}",
                    f"{r['n_rows_train']:,}",
                    f"{r['n_rows_val']:,}",
                    f"{r['n_rows_test']:,}",
                    f"{r['n_daylight_train']:,}",
                    f"{r['n_daylight_val']:,}",
                    f"{r['n_daylight_test']:,}",
                ]
            )
            + " \\\\"
        )
    lines.append("    \\midrule")
    for r in excluded_rows:
        lines.append(
            "    "
            + " & ".join(
                [
                    tex_escape(r["array"]) + "$^{\\dagger}$",
                    tex_escape(r["dkasc_site"]),
                    tex_escape(r["manufacturer"]),
                    tex_escape(r["technology"]),
                    f"{r['nameplate_kw']:.1f}",
                    "not recorded",
                    tex_escape(r["install_date"]),
                    f"{r['n_rows_total']:,}",
                    f"{r['n_rows_train']:,}",
                    f"{r['n_rows_val']:,}",
                    f"{r['n_rows_test']:,}",
                    f"{r['n_daylight_train']:,}",
                    f"{r['n_daylight_val']:,}",
                    f"{r['n_daylight_test']:,}",
                ]
            )
            + " \\\\"
        )
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("")
    for r in excluded_rows:
        lines.append(
            f"  \\vspace{{2pt}}\\par\\noindent\\footnotesize"
            f"$^{{\\dagger}}$ {tex_escape(r['array'])} EXCLUDED from evaluation. "
            f"{tex_escape(r['exclusion_reason'])}"
        )
    lines.append("\\end{table*}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    rows = build_rows()
    for r in rows:
        status = "used" if r["used_in_evaluation"] else "EXCLUDED"
        print(
            f"{r['array']:10s} site={r['dkasc_site']:>3s} {r['manufacturer']:12s} "
            f"{r['technology']:8s} {r['nameplate_kw']:.1f}kW  "
            f"rows total/train/val/test="
            f"{r['n_rows_total']}/{r['n_rows_train']}/{r['n_rows_val']}/{r['n_rows_test']}  "
            f"daylight train/val/test="
            f"{r['n_daylight_train']}/{r['n_daylight_val']}/{r['n_daylight_test']}  [{status}]"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "T1_dataset.csv"
    tex_path = TABLES_DIR / "T1_dataset.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
