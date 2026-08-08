"""Build the paper's Table 2 (feature list by regime): every feature in
both the lagged and oracle regimes, grouped by category, with the shift
applied and a one-line description.

This is the table a reviewer checks the leakage claim against, so it is
built by IMPORTING src.features.build directly (DETERMINISTIC_NAMES,
LAG_BASE_COLS, WALLCLOCK_ROLLING_SPECS, OBS_ROLLING_SPECS, WEATHER_COLS,
and feature_names() itself) rather than hand-transcribing the column
list - a hand-typed list could silently drift from the real code the
first time build.py changes. main() asserts the generated row set
matches feature_names('oracle', horizon) exactly (both directions) and
raises if not, so a future change to build.py that isn't reflected here
fails loudly instead of shipping a stale table.

CATEGORIES (matching src/features/build.py's own A/B/C docstring
categories, split into 4 for readability - B is split into "lagged
observations" and "rolling statistics"):
  A. deterministic at target time - solar geometry, clear-sky power,
     calendar encodings. Known in advance, computable for any future
     time, never a measurement.
  B1. lagged observations - Active_Power/k_p/k_ghi at three issue-
     relative shifts plus a fixed daily shift, staleness tracking, and
     weather at the issue time.
  B2. rolling statistics - wall-clock windows (Active_Power, defined at
     night) and last-N-valid-observation windows (k_p/k_ghi, undefined
     at night - see build.py's OBS_ROLLING_SPECS comment for why these
     cannot use a wall-clock window).
  C. oracle weather - measured weather AT TARGET TIME, oracle_-prefixed,
     lagged regime only omits this category entirely.

Writes:
  paper/tables/T2_features.csv
  paper/tables/T2_features.tex (booktabs fragment)

Usage:
    python scripts/build_table2_features.py
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.build import (
    DETERMINISTIC_NAMES,
    K_P_HOURS_STALE_COL,
    K_P_IS_STALE_COL,
    LAG_BASE_COLS,
    LAG_DAILY_SUFFIX,
    LAG_ISSUE_SUFFIXES,
    OBS_ROLLING_SPECS,
    ORACLE_PREFIX,
    WALLCLOCK_ROLLING_SPECS,
    WEATHER_COLS,
    feature_names,
)

TABLES_DIR = REPO_ROOT / "paper" / "tables"

# A horizon large enough to see every lagged-regime column (all suffix
# shifts exist regardless of horizon - see build.py's own comment that
# horizon does not change which columns exist) but <= MAX_HORIZON.
PROBE_HORIZON = 6

DETERMINISTIC_DESCRIPTIONS = {
    "p_cs": "Clear-sky power model (temperature climatology + gain, TRAIN-only fit)",
    "ghi_cs": "Clear-sky GHI (Ineichen model, 5-min resolution averaged to hourly - Finding 1)",
    "solar_zenith": "Solar zenith angle at the hour midpoint",
    "solar_azimuth": "Solar azimuth angle at the hour midpoint",
    "solar_elevation": "Solar elevation angle at the hour midpoint (90 minus zenith)",
    "hour_sin": "Sine encoding of hour-of-day",
    "hour_cos": "Cosine encoding of hour-of-day",
    "doy_sin": "Sine encoding of day-of-year",
    "doy_cos": "Cosine encoding of day-of-year",
}

LAG_BASE_DESCRIPTIONS = {
    "Active_Power": "Active power",
    "k_p": "Clear-sky power index k_p = Active_Power / p_cs "
    "(forward-filled up to 24h before shifting)",
    "k_ghi": "Clear-sky index of GHI "
    "(forward-filled up to 24h before shifting)",
}

WEATHER_DESCRIPTIONS = {
    "Weather_Temperature_Celsius": "Air temperature",
    "Weather_Relative_Humidity": "Relative humidity",
    "Wind_Speed": "Wind speed",
    "Global_Horizontal_Radiation": "Global horizontal irradiance (GHI)",
    "Diffuse_Horizontal_Radiation": "Diffuse horizontal irradiance (DHI)",
}

# Lowercase-leading-word variants for mid-sentence use (the oracle-weather
# description below) - WEATHER_DESCRIPTIONS.lower() would also lowercase
# the GHI/DHI acronyms, which reads wrong ("(ghi)").
WEATHER_DESCRIPTIONS_MIDSENTENCE = {
    "Weather_Temperature_Celsius": "air temperature",
    "Weather_Relative_Humidity": "relative humidity",
    "Wind_Speed": "wind speed",
    "Global_Horizontal_Radiation": "global horizontal irradiance (GHI)",
    "Diffuse_Horizontal_Radiation": "diffuse horizontal irradiance (DHI)",
}

# (suffix, shift text, description-suffix) for the three issue-relative
# lags plus the fixed daily lag - mirrors LAG_ISSUE_SUFFIXES + the daily
# shift in src/features/build.py exactly.
LAG_SHIFT_TEXT = {
    "_issue": ("h", "at issue time t-h"),
    "_issue_m1": ("h+1", "at t-h-1"),
    "_issue_m2": ("h+2", "at t-h-2"),
    LAG_DAILY_SUFFIX: ("24 (fixed)", "24h before target time (same hour, previous day)"),
}


def build_rows():
    rows = []

    # --- A: deterministic at target time ---
    for name in DETERMINISTIC_NAMES:
        rows.append(
            {
                "feature": name,
                "category": "deterministic at target time",
                "regime": "lagged + oracle",
                "shift": "0 (at t; deterministic, computable in advance)",
                "description": DETERMINISTIC_DESCRIPTIONS[name],
            }
        )

    # --- B1: lagged observations (issue-relative + daily shifts) ---
    for base in LAG_BASE_COLS:
        for suffix, _offset in LAG_ISSUE_SUFFIXES:
            shift_text, desc_suffix = LAG_SHIFT_TEXT[suffix]
            rows.append(
                {
                    "feature": f"{base}{suffix}",
                    "category": "lagged observations",
                    "regime": "lagged + oracle",
                    "shift": shift_text,
                    "description": f"{LAG_BASE_DESCRIPTIONS[base]} {desc_suffix}",
                }
            )
        shift_text, desc_suffix = LAG_SHIFT_TEXT[LAG_DAILY_SUFFIX]
        rows.append(
            {
                "feature": f"{base}{LAG_DAILY_SUFFIX}",
                "category": "lagged observations",
                "regime": "lagged + oracle",
                "shift": shift_text,
                "description": f"{LAG_BASE_DESCRIPTIONS[base]} {desc_suffix}",
            }
        )

    rows.append(
        {
            "feature": K_P_HOURS_STALE_COL,
            "category": "lagged observations",
            "regime": "lagged + oracle",
            "shift": "h",
            "description": "Hours since the last genuinely observed k_p, "
            "as of issue time (0 if observed exactly at issue time)",
        }
    )
    rows.append(
        {
            "feature": K_P_IS_STALE_COL,
            "category": "lagged observations",
            "regime": "lagged + oracle",
            "shift": "h",
            "description": f"1 if {K_P_HOURS_STALE_COL} > 0 "
            "(forward-filled, not directly observed), else 0",
        }
    )

    for col in WEATHER_COLS:
        rows.append(
            {
                "feature": f"{col}_issue",
                "category": "lagged observations",
                "regime": "lagged + oracle",
                "shift": "h",
                "description": f"{WEATHER_DESCRIPTIONS[col]} at issue time t-h",
            }
        )

    # --- B2: rolling statistics ---
    for base, window, stat in WALLCLOCK_ROLLING_SPECS:
        rows.append(
            {
                "feature": f"{base}_roll{window}_{stat}",
                "category": "rolling statistics",
                "regime": "lagged + oracle",
                "shift": "h (window ends at issue time t-h)",
                "description": f"{stat.capitalize()} of {base} over the "
                f"trailing {window} wall-clock hours ending at issue time",
            }
        )
    for base, window, stat in OBS_ROLLING_SPECS:
        rows.append(
            {
                "feature": f"{base}_last{window}obs_{stat}",
                "category": "rolling statistics",
                "regime": "lagged + oracle",
                "shift": "h (last N valid obs at or before issue time)",
                "description": (
                    f"{stat.capitalize()} of {base} over the last {window} "
                    f"VALID (non-NaN) observations at or before issue time "
                    f"- NOT a wall-clock window, since {base} is undefined "
                    "at night (see build.py's OBS_ROLLING_SPECS comment)"
                ),
            }
        )

    # --- C: oracle weather ---
    for col in WEATHER_COLS:
        rows.append(
            {
                "feature": f"{ORACLE_PREFIX}{col}",
                "category": "oracle weather",
                "regime": "oracle only",
                "shift": "0 (AT TARGET TIME t - upper bound only)",
                "description": f"Measured {WEATHER_DESCRIPTIONS_MIDSENTENCE[col]} "
                "AT TARGET TIME t - perfect-forecast UPPER BOUND, never achievable",
            }
        )

    return rows


def verify_against_source(rows):
    """Cross-check the generated row set against feature_names() itself,
    both directions - see module docstring. Raises AssertionError on any
    mismatch rather than silently shipping a stale table.
    """
    generated = {r["feature"] for r in rows}
    lagged_actual = set(feature_names("lagged", PROBE_HORIZON))
    oracle_actual = set(feature_names("oracle", PROBE_HORIZON))

    lagged_generated = {
        r["feature"] for r in rows if r["category"] != "oracle weather"
    }
    if lagged_generated != lagged_actual:
        missing = lagged_actual - lagged_generated
        extra = lagged_generated - lagged_actual
        raise AssertionError(
            f"lagged-regime mismatch vs feature_names('lagged', {PROBE_HORIZON}): "
            f"missing={missing} extra={extra}"
        )
    if generated != oracle_actual:
        missing = oracle_actual - generated
        extra = generated - oracle_actual
        raise AssertionError(
            f"oracle-regime mismatch vs feature_names('oracle', {PROBE_HORIZON}): "
            f"missing={missing} extra={extra}"
        )


def write_csv(rows, path):
    fieldnames = ["feature", "category", "regime", "shift", "description"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%")


CATEGORY_ORDER = [
    "deterministic at target time",
    "lagged observations",
    "rolling statistics",
    "oracle weather",
]


def write_latex(rows, path):
    lines = []
    lines.append("% T2_features.tex - feature list by regime, booktabs fragment.")
    lines.append("% NOT compile-tested (no LaTeX toolchain in this dev")
    lines.append("% environment) - same caveat as paper/figures/F7_pipeline.tex")
    lines.append("% and paper/tables/T1_dataset.tex. Needs \\usepackage{booktabs}.")
    lines.append("% Long (42 rows): written as a single longtable-style table*")
    lines.append("% assuming \\usepackage{longtable} or that the paper accepts")
    lines.append("% it spanning onto a following page - a plain tabular in a")
    lines.append("% table* environment will NOT paginate on its own. Swap the")
    lines.append("% table*/tabular pair below for longtable if the compiled")
    lines.append("% length overflows a page - not decided here without a real")
    lines.append("% render.")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\scriptsize")
    lines.append("  \\caption{Feature list by regime. See paper/tables/CAPTIONS.md for the full caption.}")
    lines.append("  \\label{tab:features}")
    lines.append("  \\begin{tabular}{llll p{6.5cm}}")
    lines.append("    \\toprule")
    lines.append("    Feature & Category & Regime & Shift & Description \\\\")
    lines.append("    \\midrule")

    by_category = {cat: [] for cat in CATEGORY_ORDER}
    for r in rows:
        by_category[r["category"]].append(r)

    for cat_idx, cat in enumerate(CATEGORY_ORDER):
        cat_rows = by_category[cat]
        lines.append(
            f"    \\multicolumn{{5}}{{l}}{{\\textit{{{tex_escape(cat)} "
            f"({len(cat_rows)} features)}}}} \\\\"
        )
        for r in cat_rows:
            lines.append(
                "    "
                + " & ".join(
                    [
                        f"\\texttt{{{tex_escape(r['feature'])}}}",
                        "",  # category already shown as a group header above
                        tex_escape(r["regime"]),
                        tex_escape(r["shift"]),
                        tex_escape(r["description"]),
                    ]
                )
                + " \\\\"
            )
        if cat_idx != len(CATEGORY_ORDER) - 1:
            lines.append("    \\addlinespace")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    rows = build_rows()
    verify_against_source(rows)

    n_lagged = sum(1 for r in rows if r["category"] != "oracle weather")
    n_oracle_only = sum(1 for r in rows if r["category"] == "oracle weather")
    print(f"lagged regime: {n_lagged} features")
    print(f"oracle regime: {n_lagged + n_oracle_only} features ({n_oracle_only} oracle-only)")
    print("verified against src.features.build.feature_names() - OK")

    for cat in CATEGORY_ORDER:
        n = sum(1 for r in rows if r["category"] == cat)
        print(f"  {cat}: {n}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "T2_features.csv"
    tex_path = TABLES_DIR / "T2_features.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
