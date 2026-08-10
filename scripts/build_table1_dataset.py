"""Build the paper's Table 1 (dataset summary): one row per array, plus
an excluded-array footnote row for array07.

SOURCES, per array-metadata column:
  - nameplate kW: scripts/build_processed.py's ARRAYS dict.
  - tilt/azimuth: src/data/clearsky_power.py's SURFACE_TILT/SURFACE_AZIMUTH
    constants (build_processed.py does NOT have these - the user-given
    source list for this table named build_processed.py, but tilt/azimuth
    live in clearsky_power.py; noting the discrepancy here rather than
    silently citing the wrong file). array07's tilt/azimuth are the same
    20 deg / 0 deg - DKASC documents ALL fixed-mount arrays at this site,
    including array07, at this tilt/azimuth (CORRECTED 2026-08-09: this
    file previously left them as "not recorded" out of excess caution
    about an array-specific citation that does not exist; the site-wide
    fixed-mount convention DOES apply to array07, per the same DKASC
    technology-page source already cited for array11/12/17).
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

SOURCES, per data-volume column (array11/array12/array17 ONLY - see
ARRAY07 below for why its row/daylight/evaluable columns are not
measured the same way):
  - n_rows_total/train/val/test: computed directly here via
    src.data.pipeline.load_and_prepare + src.data.splits.split_chronological
    on each array's processed hourly parquet - not from a committed CSV,
    since none has exactly this number. This is the same
    61,344/26,304/8,760/8,760 pattern already stated in CLAUDE.md's
    "Data window" section and PROJECT_CHECKPOINT.md claim C36 - this
    script verifies it directly rather than re-typing it.
  - n_daylight_geometric_{train,val,test}: is_daylight (src.data.clearsky.
    add_daylight_mask, solar_elevation > 10 deg) summed per split,
    computed live from the same load_and_prepare call above, not read
    from results/dead_period_audit.csv - CORRECTED 2026-08-09: this file
    previously named "n_daylight_train/val/test" and summed
    dead_period_audit.csv's n_daylight_hours column instead. Same
    number, but the OLD name implied it was already the count that
    matters for evaluation. It is not: it depends only on shared
    solar geometry (identical across all three arrays, and even
    array07, which shares the same weather station - CLAUDE.md
    "co-located arrays, not sites") and says nothing about documented
    equipment outages. Renamed "geometric" to make that explicit, and
    see n_evaluable_val/test below for the number that actually differs
    per array.
  - n_evaluable_val/n_evaluable_test: n_daylight_geometric minus hours
    excluded by src.eval.exclusions.exclusion_mask (documented
    equipment outages) for that array - ADDED 2026-08-09. This is the
    column that actually varies per array: array17's 2015-06-05 to
    2015-06-09 outage falls inside daylight hours of the TEST split, so
    array17's n_evaluable_test (3757) is measurably smaller than its
    n_daylight_geometric_test (3802) and smaller than array11/array12's
    n_evaluable_test (3802, unaffected - no documented outage touches
    either array in this window). n_evaluable_val is included for
    symmetry even though no array in this project has a documented
    outage inside VAL (2014), so n_evaluable_val currently equals
    n_daylight_geometric_val for all three arrays - kept as a real,
    computed column rather than assumed equal, so a future outage
    addition to KNOWN_OUTAGES would be reflected here automatically.
    train is deliberately not given an n_evaluable_train: models are
    FIT on train, not evaluated against it, so an outage-adjusted count
    is not a meaningful quantity there the way it is for val/test.

ARRAY07: NOT measured, NOT inferred, reported as "n/a (excluded)" for
every row/daylight/evaluable column - CORRECTED 2026-08-09. This file
previously computed real numbers for array07 by reading the leftover
data/processed/array07_CdTe_hourly.parquet directly (bypassing
src.data.pipeline.load_and_prepare, since array07 is not in that
module's ARRAYS registry - array07 is not part of the CURRENT
pipeline). Every number that came out of that (rows, daylight hours)
was identical to array11/array12/array17's, because it depends only on
the shared calendar index every array shares, structurally, regardless
of whether that array's own data was ever touched - it was not
measuring anything about array07 specifically, and presenting it as a
table cell alongside three arrays that WERE genuinely evaluated implied
array07 was audited on the same footing. It was not: no run in this
project ever fits or scores a model against array07. The exclusion
reason (still populated, see array07_exclusion_reason()) is unaffected
by this correction - it was already sourced from results/data_audit.csv
and results/dead_period_audit.csv, not from the processed parquet.

Writes:
  paper/tables/T1_dataset.csv
  paper/tables/T1_dataset.tex (booktabs fragment)

Usage:
    python scripts/build_table1_dataset.py
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import load_and_prepare
from src.data.splits import split_chronological
from src.eval.exclusions import exclusion_mask

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

NA = "n/a (excluded)"

# tilt/azimuth from src/data/clearsky_power.py (SURFACE_TILT=20.0,
# SURFACE_AZIMUTH=0.0) - DKASC documents ALL fixed-mount arrays at this
# site at this tilt/azimuth, including array07 (see module docstring).
ARRAY_METADATA = {
    "array11": {
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
        "site": "07",
        "manufacturer": "First Solar",
        "technology": "CdTe",
        "nameplate_kw": 7.0,
        # DKASC's site-wide fixed-mount convention applies to array07 too
        # (see module docstring) - this is a documented site fact, not an
        # array-specific measurement, so it is filled rather than left
        # "not recorded" the way install_date (a genuinely per-array,
        # genuinely undocumented-in-this-project fact) is below.
        "tilt_deg": 20.0,
        "azimuth_deg": 0.0,
        "install_date": "not recorded",
        "slug": "dka-m6-a-phase",
        "used": False,
    },
}


def measured_counts(array_key):
    """Row counts, geometric daylight-hour counts, and outage-adjusted
    evaluable counts for one array, computed live from the pipeline -
    see module docstring for exactly what each means and why. Only
    called for array11/array12/array17 - array07 is not in
    src.data.pipeline.ARRAYS (this call would KeyError for it), which
    is itself evidence for why array07's columns are "n/a (excluded)"
    rather than measured.
    """
    df, nameplate_kw, gamma_pdc = load_and_prepare(array_key, PROCESSED_DIR)
    train, val, test = split_chronological(df)

    n_daylight_train = int(train["is_daylight"].sum())
    n_daylight_val = int(val["is_daylight"].sum())
    n_daylight_test = int(test["is_daylight"].sum())

    daylight_val_idx = val.index[val["is_daylight"]]
    daylight_test_idx = test.index[test["is_daylight"]]
    n_evaluable_val = int((~exclusion_mask(array_key, daylight_val_idx)).sum())
    n_evaluable_test = int((~exclusion_mask(array_key, daylight_test_idx)).sum())

    return {
        "n_rows_total": len(df),
        "n_rows_train": len(train),
        "n_rows_val": len(val),
        "n_rows_test": len(test),
        "n_daylight_geometric_train": n_daylight_train,
        "n_daylight_geometric_val": n_daylight_val,
        "n_daylight_geometric_test": n_daylight_test,
        "n_evaluable_val": n_evaluable_val,
        "n_evaluable_test": n_evaluable_test,
    }


NA_COUNTS = {
    "n_rows_total": NA,
    "n_rows_train": NA,
    "n_rows_val": NA,
    "n_rows_test": NA,
    "n_daylight_geometric_train": NA,
    "n_daylight_geometric_val": NA,
    "n_daylight_geometric_test": NA,
    "n_evaluable_val": NA,
    "n_evaluable_test": NA,
}


def array07_exclusion_reason():
    """Pairs the completeness-audit numbers against the dead-period
    numbers for array07/2014, per Finding 6 - see module docstring.
    Unaffected by the array07 n/a correction: always sourced from
    results/data_audit.csv and results/dead_period_audit.csv, never
    from the processed parquet.
    """
    with open(RESULTS_DIR / "data_audit.csv") as fh:
        audit_rows = {(r["array"], int(r["year"])): r for r in csv.DictReader(fh)}
    with open(RESULTS_DIR / "dead_period_audit.csv") as fh:
        dead_rows = {(r["array"], int(r["year"])): r for r in csv.DictReader(fh)}

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
    rows = []
    for array_key, meta in ARRAY_METADATA.items():
        counts = measured_counts(array_key) if meta["used"] else dict(NA_COUNTS)
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
                **counts,
                "used_in_evaluation": meta["used"],
                "exclusion_reason": "" if meta["used"] else array07_exclusion_reason(),
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


def fmt_count(value):
    return f"{value:,}" if isinstance(value, int) else tex_escape(value)


def write_latex(rows, path):
    used_rows = [r for r in rows if r["used_in_evaluation"]]
    excluded_rows = [r for r in rows if not r["used_in_evaluation"]]

    lines = []
    lines.append("% T1_dataset.tex - dataset summary, booktabs fragment.")
    lines.append("% NOT compile-tested (no LaTeX toolchain in this dev")
    lines.append("% environment) - verify before camera-ready, same caveat")
    lines.append("% as paper/figures/F7_pipeline.tex. Needs \\usepackage{booktabs}.")
    lines.append("% Columns reduced from 16 to 10 to fit IEEE two-column width")
    lines.append("% (paper/overleaf compile-fix pass): dropped Site (identical to")
    lines.append("% the Array column - array11/site 11 etc.), Tilt/Azim and")
    lines.append("% Installed (both constant or near-constant across arrays,")
    lines.append("% moved to the caption), and the three Daylight-geometric")
    lines.append("% columns (identical across all arrays by construction, since")
    lines.append("% they depend only on shared solar geometry - moved to the")
    lines.append("% caption as fixed numbers). The Evaluable columns are kept in")
    lines.append("% full: they are the one count that varies by array")
    lines.append("% (array17's documented outage).")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\scriptsize")
    lines.append(
        "  \\caption{Dataset summary: the three co-located DKASC arrays used "
        "in evaluation (array11 poly-Si, array12 mono-Si, array17 HIT - one "
        "shared weather station, not three independent sites) plus array07 "
        "(CdTe), excluded. All arrays are fixed-mount at 20$^\\circ$ tilt / "
        "0$^\\circ$ azimuth; only array17's install date is documented "
        "(2010-03-11), which is why the training window in this paper "
        "starts at 2011. Row counts by chronological split (train "
        "2011-2013, validation 2014, test 2015 - touched once, at the "
        "end), plus outage-adjusted evaluable hour counts for validation "
        "and test: array17's documented 2015-06-05 to 2015-06-09 outage "
        "reduces its evaluable test count to 3757 daylight hours, versus "
        "3802 for array11 and array12, which are unaffected. Geometric "
        "daylight-hour counts (solar elevation $>10^\\circ$) are identical "
        "across all arrays by construction - 11,414 (train), 3,799 (val), "
        "3,802 (test) - and are omitted from the table on that basis. "
        "array07 is retained as an excluded row, not deleted, with its "
        "row/evaluable columns reported as not applicable rather than "
        "measured (no run in this project fits or scores a model against "
        "array07's data): a completeness audit "
        "(results/data\\_audit.csv) passed its 2014 data at 99.99\\% "
        "coverage and 0.00\\% NaN in Active\\_Power, while a dead-period "
        "audit (results/dead\\_period\\_audit.csv) found 48.41\\% of that "
        "same year's daylight hours at exactly zero power (status FAIL), "
        "plus a 48-day near-zero run inside the test year - a "
        "completeness-only audit is structurally blind to a healthy "
        "sensor reporting a dead array (Finding 6).}"
    )
    lines.append("  \\label{tab:T1}")
    lines.append("  \\begin{tabular}{llll rrrr rr}")
    lines.append("    \\toprule")
    lines.append(
        "    & & & & \\multicolumn{4}{c}{Rows (total / train / val / test)} & "
        "\\multicolumn{2}{c}{Evaluable (val / test)} \\\\"
    )
    lines.append("    \\cmidrule(lr){5-8} \\cmidrule(lr){9-10}")
    lines.append(
        "    Array & Manuf. & Tech. & kW & "
        "Total & Train & Val & Test & Val & Test \\\\"
    )
    lines.append("    \\midrule")

    def row_cells(r, dagger):
        return [
            tex_escape(r["array"]) + dagger,
            tex_escape(r["manufacturer"]),
            tex_escape(r["technology"]),
            f"{r['nameplate_kw']:.1f}",
            fmt_count(r["n_rows_total"]),
            fmt_count(r["n_rows_train"]),
            fmt_count(r["n_rows_val"]),
            fmt_count(r["n_rows_test"]),
            fmt_count(r["n_evaluable_val"]),
            fmt_count(r["n_evaluable_test"]),
        ]

    for r in used_rows:
        lines.append("    " + " & ".join(row_cells(r, "")) + " \\\\")
    lines.append("    \\midrule")
    for r in excluded_rows:
        lines.append("    " + " & ".join(row_cells(r, "$^{\\dagger}$")) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
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
            f"tilt/azim={r['tilt_deg']}/{r['azimuth_deg']}  "
            f"rows total/train/val/test="
            f"{r['n_rows_total']}/{r['n_rows_train']}/{r['n_rows_val']}/{r['n_rows_test']}  "
            f"daylight(geom) train/val/test="
            f"{r['n_daylight_geometric_train']}/{r['n_daylight_geometric_val']}/{r['n_daylight_geometric_test']}  "
            f"evaluable val/test={r['n_evaluable_val']}/{r['n_evaluable_test']}  [{status}]"
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
