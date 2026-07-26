"""Audit raw DKASC CSVs for coverage and completeness, per array per year.

Reads only the columns needed. Does not modify data/raw/. Writes the full
per-array per-year table to results/data_audit.csv and prints a readable
summary to the console. Feeds Table 1 of the paper.

Usage:
    python scripts/audit_raw_data.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results"

RAW_FILES = {
    "array07_CdTe": RAW_DIR / "array07_CdTe.csv",
    "array11_polySi": RAW_DIR / "array11_polySi.csv",
    "array12_monoSi": RAW_DIR / "array12_monoSi.csv",
}

TIMESTAMP_COL = "timestamp"
NAN_CHECK_COLS = [
    "Active_Power",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Wind_Speed",
]
USE_COLS = [TIMESTAMP_COL] + NAN_CHECK_COLS

SLOTS_PER_DAY = 24 * 60 // 5  # 288 five-minute slots per day


def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def expected_rows_for_year(year: int) -> int:
    days = 366 if is_leap_year(year) else 365
    return days * SLOTS_PER_DAY


def largest_gap_hours(sorted_unique_timestamps: pd.Series) -> float:
    if len(sorted_unique_timestamps) < 2:
        return 0.0
    diffs = sorted_unique_timestamps.diff().dropna()
    if len(diffs) == 0:
        return 0.0
    return diffs.max().total_seconds() / 3600.0


def audit_one_file(array_name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=USE_COLS)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    dup_mask = df[TIMESTAMP_COL].duplicated(keep=False)
    df["year"] = df[TIMESTAMP_COL].dt.year

    rows = []
    for year, group in df.groupby("year"):
        group = group.sort_values(TIMESTAMP_COL)
        unique_ts = group[TIMESTAMP_COL].drop_duplicates()

        actual_rows = len(group)
        expected_rows = expected_rows_for_year(int(year))
        coverage_pct = 100.0 * actual_rows / expected_rows

        nan_pcts = {
            col: 100.0 * group[col].isna().mean() for col in NAN_CHECK_COLS
        }

        n_duplicates = int(dup_mask.loc[group.index].sum())
        gap_hours = largest_gap_hours(unique_ts)

        rows.append(
            {
                "array": array_name,
                "year": int(year),
                "first_timestamp": group[TIMESTAMP_COL].min(),
                "last_timestamp": group[TIMESTAMP_COL].max(),
                "actual_rows": actual_rows,
                "expected_rows": expected_rows,
                "coverage_pct": round(coverage_pct, 2),
                **{
                    f"nan_pct_{col}": round(pct, 2)
                    for col, pct in nan_pcts.items()
                },
                "n_duplicate_timestamps": n_duplicates,
                "largest_gap_hours": round(gap_hours, 2),
            }
        )

    return pd.DataFrame(rows)


def print_summary(audit_df: pd.DataFrame) -> None:
    display_cols = [
        "array",
        "year",
        "first_timestamp",
        "last_timestamp",
        "actual_rows",
        "expected_rows",
        "coverage_pct",
        "n_duplicate_timestamps",
        "largest_gap_hours",
    ]
    with pd.option_context(
        "display.max_rows", None,
        "display.width", 200,
        "display.max_columns", None,
    ):
        print(audit_df[display_cols].to_string(index=False))

    print()
    nan_cols = ["array", "year"] + [f"nan_pct_{c}" for c in NAN_CHECK_COLS]
    with pd.option_context(
        "display.max_rows", None,
        "display.width", 200,
        "display.max_columns", None,
    ):
        print(audit_df[nan_cols].to_string(index=False))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_audits = []
    for array_name, path in RAW_FILES.items():
        print(f"Auditing {array_name} ({path.name}) ...")
        all_audits.append(audit_one_file(array_name, path))

    audit_df = pd.concat(all_audits, ignore_index=True)
    audit_df = audit_df.sort_values(["array", "year"]).reset_index(drop=True)

    print()
    print_summary(audit_df)

    out_path = RESULTS_DIR / "data_audit.csv"
    audit_df.to_csv(out_path, index=False)
    print(f"\nSaved full table to {out_path}")


if __name__ == "__main__":
    main()
