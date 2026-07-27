"""Audit for dead periods (zero-output stretches) per array per year.

MOTIVATION: scripts/diagnose_array_level_shift.py found array07's monthly
performance ratio (mean Active_Power / mean GHI) is exactly 0.0000 for
months 3-9 of 2014 - the array produced no power for seven months.
results/data_audit.csv did not catch this: coverage was 99.99 percent and
NaN 0.00 percent, because the logger kept recording and recorded zeros.
A completeness audit (rows present, values not NaN) cannot see a dead
array that is faithfully logging zero. This script adds the missing check:
is the array actually producing power during daylight, not just logging.

PART 1 reads the RAW CSV directly (no loader, no cleaning) to confirm the
zeros are in the source data and not introduced by our pipeline.

PART 2 audits every array and every year 2009-2015, INCLUDING 2015 (test),
using the processed hourly parquet.

Justification for looking at 2015 here: this is a DATA QUALITY check on
availability only. It reads no model output and computes no forecast
metric on 2015 - it only asks whether the recorded Active_Power values are
physically plausible (nonzero when the sun is up). Verifying the test data
is physically valid before using it is not tuning on the test set;
discovering after the final run that the test year was dead would be far
worse. This check, and what it looked at (Active_Power daylight-hour
availability only, no metrics, no model output), is recorded here so the
scope is auditable: run once, read-only, availability only.

PART 3 prints PASS/FAIL per array-year. FAIL means more than 5 percent of
daylight hours are below 1 percent of nameplate capacity, or the longest
such run exceeds 3 days.

SECOND BLIND SPOT (found diagnosing array17, see
scripts/diagnose_array17_events.py): this audit originally judged FAIL on
EXACTLY zero Active_Power. That missed array17's pre-install period and
its documented 5-9 June 2015 outage entirely, because array17 logs a small
nonzero standby value (~0.04 kW, not 0.0 kW) while dead. The first blind
spot (above) was a completeness audit missing an array that faithfully
logs real zeros (array07, 2014). This second one is the mirror case: an
exact-zero audit missing an array that faithfully logs near-zero standby
instead of true zeros. Same lesson twice - "is this array producing
power" cannot be answered by any single fixed check; below-1pct-of-
nameplate is a wider net than exact-zero and catches both. The exact-zero
columns are kept in the output below for comparison, but no longer drive
PASS/FAIL.

No fixes, no plotting, no changes to any existing file.

Usage:
    python scripts/audit_dead_periods.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky import add_daylight_mask, add_solar_position

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

# Same arrays, filenames, and nameplate values as scripts/run_xgb_dev.py.
# array07 excluded - see CLAUDE.md "Data window" and the PASS/FAIL result
# for array07 below, which is the evidence for that exclusion. It is
# audited separately in PART 1 (audit_raw_array07_2014) from the raw CSV
# directly, since it is no longer part of the processed-parquet pipeline.
ARRAYS = {
    "array11": ("array11_polySi.csv", "array11_polySi_hourly.parquet", 5.0),
    "array12": ("array12_monoSi.csv", "array12_monoSi_hourly.parquet", 5.1),
    "array17": ("array17_HIT.csv", "array17_HIT_hourly.parquet", 6.3),
}

# array07's raw CSV, referenced only by PART 1 (audit_raw_array07_2014) -
# kept as the evidence-generating check for the exclusion decision, not
# because array07 is still processed anywhere else.
ARRAY07_RAW_CSV = RAW_DIR / "array07_CdTe.csv"

AUDIT_YEARS = (2009, 2010, 2011, 2012, 2013, 2014, 2015)

RAW_DAYTIME_GHI_THRESHOLD = 200.0  # W/m2, "clearly daytime" per motivation
# FAIL thresholds, applied to the below-1pct-nameplate mask, not exact zero
# - see the SECOND BLIND SPOT note in the module docstring.
BELOW_1PCT_FAIL_THRESHOLD = 5.0  # percent
LONGEST_RUN_FAIL_DAYS = 3


# --------------------------------------------------------------------------
# PART 1 - confirm the zeros are in the raw source, not our pipeline.
# --------------------------------------------------------------------------

def audit_raw_array07_2014():
    print("=" * 78)
    print("PART 1: array07 raw CSV, 2014, no loader, no cleaning")
    print("=" * 78)

    usecols = ["timestamp", "Global_Horizontal_Radiation", "Active_Power"]
    df = pd.read_csv(ARRAY07_RAW_CSV, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"].dt.year == 2014]

    is_daytime = df["Global_Horizontal_Radiation"] > RAW_DAYTIME_GHI_THRESHOLD
    is_zero_power = df["Active_Power"] == 0

    daytime = df[is_daytime]
    daytime_zero = df[is_daytime & is_zero_power]

    monthly_daytime = daytime.groupby(daytime["timestamp"].dt.month).size()
    monthly_zero = daytime_zero.groupby(daytime_zero["timestamp"].dt.month).size()

    table = pd.DataFrame(
        {
            "n_daytime_5min_records": monthly_daytime,
            "n_zero_power_daytime": monthly_zero,
        }
    ).fillna(0).astype(int)
    table.index.name = "month"
    table["pct_zero_power_daytime"] = (
        100.0 * table["n_zero_power_daytime"] / table["n_daytime_5min_records"]
    ).round(2)

    print(
        f"\narray07 raw 2014, daytime = Global_Horizontal_Radiation > "
        f"{RAW_DAYTIME_GHI_THRESHOLD:.0f} W/m2:\n"
    )
    print(table.to_string())
    print(
        "\nIf months show high pct_zero_power_daytime, the zeros are present "
        "in the raw file and the pipeline is not the cause.\n"
    )


# --------------------------------------------------------------------------
# PART 2 - every array, every year 2009-2015, processed hourly data.
# --------------------------------------------------------------------------

def longest_true_run(daylight_df, mask):
    """Longest consecutive run of daylight hours where `mask` is True.

    'Consecutive' means consecutive within the daylight-hours-only series
    (night hours are skipped over freely, since they are excluded from
    daylight_df entirely) - i.e. a run is broken only by a daylight hour
    where mask is False, not by the sun going down. Returns
    (run_length_days, first_date, last_date) for the longest such run, or
    (0, None, None) if mask is never True.

    Generic over `mask` so it can measure either the exact-zero run or the
    below-1pct-nameplate run with the same logic - see SECOND BLIND SPOT
    in the module docstring for why both are tracked.
    """
    if not mask.any():
        return 0, None, None

    group_id = (~mask).cumsum()
    true_rows = daylight_df.loc[mask]
    group_sizes = group_id[mask].groupby(group_id[mask]).size()
    longest_group_id = group_sizes.idxmax()

    run_index = true_rows.index[group_id[mask] == longest_group_id]
    first_date = run_index.min().date()
    last_date = run_index.max().date()
    run_length_days = (last_date - first_date).days + 1
    return run_length_days, first_date, last_date


def audit_one_array_year(array_key, daylight_df, year, nameplate_kw):
    year_df = daylight_df[daylight_df.index.year == year]
    n_daylight = len(year_df)

    if n_daylight == 0:
        return None

    # Exact-zero: kept and reported for comparison only, no longer drives
    # PASS/FAIL - see SECOND BLIND SPOT in the module docstring.
    is_zero = year_df["Active_Power"] == 0
    n_zero = int(is_zero.sum())
    pct_zero = 100.0 * n_zero / n_daylight
    zero_run_days, zero_run_start, zero_run_end = longest_true_run(year_df, is_zero)

    # Below-1pct-nameplate: this is what drives PASS/FAIL now, because it
    # also catches near-zero standby output (array17), which exact-zero
    # missed entirely.
    below_1pct_threshold = 0.01 * nameplate_kw
    is_below_1pct = year_df["Active_Power"] < below_1pct_threshold
    n_below_1pct = int(is_below_1pct.sum())
    pct_below_1pct = 100.0 * n_below_1pct / n_daylight
    below_run_days, below_run_start, below_run_end = longest_true_run(
        year_df, is_below_1pct
    )

    fail = (
        (pct_below_1pct > BELOW_1PCT_FAIL_THRESHOLD)
        or (below_run_days > LONGEST_RUN_FAIL_DAYS)
    )
    status = "FAIL" if fail else "PASS"

    return {
        "array": array_key,
        "year": year,
        "n_daylight_hours": n_daylight,
        "n_zero_power_daylight": n_zero,
        "pct_zero_power_daylight": round(pct_zero, 2),
        "longest_zero_run_days": zero_run_days,
        "longest_zero_run_start": zero_run_start,
        "longest_zero_run_end": zero_run_end,
        "n_below_1pct_nameplate": n_below_1pct,
        "pct_below_1pct_nameplate": round(pct_below_1pct, 2),
        "longest_below_1pct_run_days": below_run_days,
        "longest_below_1pct_run_start": below_run_start,
        "longest_below_1pct_run_end": below_run_end,
        "status": status,
    }


def audit_processed_all_arrays():
    print("=" * 78)
    print("PART 2: all arrays, 2009-2015, processed hourly data (daylight hours only)")
    print("=" * 78)
    print(
        "\nScope of this check: Active_Power daylight-hour availability only. "
        "No model output, no forecast metric, no 2015 model evaluation - "
        "availability only, read-only.\n"
    )

    rows = []
    for array_key in sorted(ARRAYS):
        _, parquet_name, nameplate_kw = ARRAYS[array_key]
        df = pd.read_parquet(PROCESSED_DIR / parquet_name)
        df = add_solar_position(df)
        df = add_daylight_mask(df)
        daylight_df = df[df["is_daylight"]]

        for year in AUDIT_YEARS:
            row = audit_one_array_year(array_key, daylight_df, year, nameplate_kw)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def print_part2_table(audit_df):
    with pd.option_context(
        "display.max_rows", None, "display.width", 200, "display.max_columns", None
    ):
        print(audit_df.to_string(index=False))


def print_pass_fail_summary(audit_df):
    print("\n" + "=" * 78)
    print("PART 3: PASS/FAIL summary")
    print(
        f"FAIL if pct_below_1pct_nameplate > {BELOW_1PCT_FAIL_THRESHOLD}% "
        f"or longest_below_1pct_run_days > {LONGEST_RUN_FAIL_DAYS} "
        "(exact-zero columns shown for comparison only, see SECOND BLIND "
        "SPOT in the module docstring)"
    )
    print("=" * 78 + "\n")

    for _, row in audit_df.iterrows():
        marker = row["status"]
        print(
            f"{row['array']} {row['year']}: {marker}  "
            f"(below_1pct={row['pct_below_1pct_nameplate']}%, "
            f"longest_below_1pct_run={row['longest_below_1pct_run_days']}d "
            f"[{row['longest_below_1pct_run_start']} .. {row['longest_below_1pct_run_end']}], "
            f"zero={row['pct_zero_power_daylight']}%, "
            f"longest_zero_run={row['longest_zero_run_days']}d)"
        )

    n_fail = int((audit_df["status"] == "FAIL").sum())
    n_total = len(audit_df)
    print(f"\n{n_fail}/{n_total} array-years FAILED.")


def main():
    audit_raw_array07_2014()

    audit_df = audit_processed_all_arrays()
    print_part2_table(audit_df)
    print_pass_fail_summary(audit_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "dead_period_audit.csv"
    audit_df.to_csv(out_path, index=False)
    print(f"\nSaved full table to {out_path}")


if __name__ == "__main__":
    main()
