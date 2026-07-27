"""Diagnose array17 (Sanyo HIT, 6.3 kW) against the three documented events
from the DKASC site page, before it is added to the training pipeline for
real:

- Installation completed 11 March 2010. The site's CSV export starts at
  2008-09-12 for every array, so rows before March 2010 predate array17's
  existence and should show near-zero output.
- Inverter replaced approximately July 2013 (SMA SMC 6000A, correcting an
  earlier published SMC 7000TL). Possible output level shift mid-training.
- Outage 5-9 June 2015: array17 among seven arrays switched off. This falls
  in the TEST year.

This script only reads the processed hourly parquet (already restricted to
2009-2015 by build_processed.py) and prints descriptive tables. No fixes, no
plotting, no changes to src/data/loader.py or src/data/clearsky.py.

Usage:
    python scripts/diagnose_array17_events.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky import add_daylight_mask, add_solar_position

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

ARRAY17_PARQUET = "array17_HIT_hourly.parquet"
DIAGNOSIS_YEARS = (2009, 2010, 2011, 2012, 2013, 2014, 2015)

# Same FAIL thresholds as scripts/audit_dead_periods.py, for the summary
# printed at the end of this script. FAIL is driven by below-1pct-nameplate,
# not exact-zero - see that script's SECOND BLIND SPOT docstring note,
# which this array17 diagnosis is what surfaced.
BELOW_1PCT_FAIL_THRESHOLD = 5.0  # percent
LONGEST_RUN_FAIL_DAYS = 3


def load_array17():
    df = pd.read_parquet(PROCESSED_DIR / ARRAY17_PARQUET)
    df = df[df.index.year.isin(DIAGNOSIS_YEARS)]
    df = add_solar_position(df)
    df = add_daylight_mask(df)
    return df


def monthly_table(daylight_df):
    """Monthly mean daylight Active_Power and performance ratio
    (mean Active_Power / mean GHI), one row per (year, month), 2009-2015.
    """
    grouped = daylight_df.groupby([daylight_df.index.year, daylight_df.index.month])
    table = pd.DataFrame(
        {
            "mean_active_power_kw": grouped["Active_Power"].mean(),
            "mean_ghi_wm2": grouped["Global_Horizontal_Radiation"].mean(),
            "n_daylight_hours": grouped.size(),
        }
    )
    table["performance_ratio"] = (
        table["mean_active_power_kw"] / table["mean_ghi_wm2"]
    )
    table.index.names = ["year", "month"]
    return table[
        ["mean_active_power_kw", "mean_ghi_wm2", "performance_ratio", "n_daylight_hours"]
    ]


def june_2015_daily(daylight_df):
    """Daily mean daylight Active_Power, 1-15 June 2015."""
    window = daylight_df[
        (daylight_df.index >= pd.Timestamp("2015-06-01", tz=daylight_df.index.tz))
        & (daylight_df.index < pd.Timestamp("2015-06-16", tz=daylight_df.index.tz))
    ]
    grouped = window.groupby(window.index.date)
    table = pd.DataFrame(
        {
            "mean_active_power_kw": grouped["Active_Power"].mean(),
            "n_daylight_hours": grouped.size(),
        }
    )
    table.index.name = "date"
    return table


def yearly_p99(daylight_df):
    """99th percentile of Active_Power per year, daylight hours only -
    restricted to daylight so the percentile reflects daytime output
    levels (an inverter clipping ceiling), not diluted by the night-hour
    population that make up more than half of a full 8760-hour year.
    """
    grouped = daylight_df.groupby(daylight_df.index.year)
    table = pd.DataFrame(
        {
            "p99_active_power_kw": grouped["Active_Power"].quantile(0.99),
            "max_active_power_kw": grouped["Active_Power"].max(),
        }
    )
    table.index.name = "year"
    return table


def print_dead_period_summary():
    """Re-print the PASS/FAIL summary from results/dead_period_audit.csv
    (written by scripts/audit_dead_periods.py) for all four arrays, so this
    diagnosis and that audit are read together without re-running it.
    """
    print("\n" + "=" * 78)
    print("Dead-period audit PASS/FAIL summary (results/dead_period_audit.csv)")
    print(
        f"FAIL if pct_below_1pct_nameplate > {BELOW_1PCT_FAIL_THRESHOLD}% "
        f"or longest_below_1pct_run_days > {LONGEST_RUN_FAIL_DAYS}"
    )
    print("=" * 78 + "\n")

    csv_path = RESULTS_DIR / "dead_period_audit.csv"
    audit_df = pd.read_csv(csv_path)

    for _, row in audit_df.iterrows():
        print(
            f"{row['array']} {row['year']}: {row['status']}  "
            f"(below_1pct={row['pct_below_1pct_nameplate']}%, "
            f"longest_below_1pct_run={row['longest_below_1pct_run_days']}d "
            f"[{row['longest_below_1pct_run_start']} .. {row['longest_below_1pct_run_end']}], "
            f"zero={row['pct_zero_power_daylight']}%)"
        )

    n_fail = int((audit_df["status"] == "FAIL").sum())
    n_total = len(audit_df)
    print(f"\n{n_fail}/{n_total} array-years FAILED.")


def main():
    daylight_df = load_array17()[lambda d: d["is_daylight"]]

    print("=" * 78)
    print("array17 monthly mean daylight Active_Power and performance ratio, 2009-2015")
    print("(watch for: near-zero before March 2010; a step around July 2013;")
    print(" a dip in June 2015)")
    print("=" * 78 + "\n")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(monthly_table(daylight_df).round(4).to_string())

    print("\n" + "=" * 78)
    print("array17 daily mean daylight Active_Power, 1-15 June 2015")
    print("=" * 78 + "\n")
    print(june_2015_daily(daylight_df).round(4).to_string())

    print("\n" + "=" * 78)
    print("array17 99th percentile and max of daylight Active_Power, per year")
    print("(watch for: a change in the ceiling between 2012, 2013, and 2014)")
    print("=" * 78 + "\n")
    print(yearly_p99(daylight_df).round(4).to_string())

    print_dead_period_summary()


if __name__ == "__main__":
    main()
