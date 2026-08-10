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

# (suffix, shift code, description-suffix) for the three issue-relative
# lags plus the fixed daily lag - mirrors LAG_ISSUE_SUFFIXES + the daily
# shift in src/features/build.py exactly. Shift codes are the compact
# form explained once in the table caption (h/h+1/h+2/24/h(w)/h(o)/0),
# not the verbose per-row text this table used before the compile-fix
# pass that shortened it to fit IEEE page width.
LAG_SHIFT_TEXT = {
    "_issue": ("h", "at issue time t-h"),
    "_issue_m1": ("h+1", "at t-h-1"),
    "_issue_m2": ("h+2", "at t-h-2"),
    LAG_DAILY_SUFFIX: ("24", "24h before target (same hour, previous day)"),
}

# Short prose labels for the lag-base descriptions, used only in the
# per-row Description cell - the formal k_p/k_ghi ratio definitions and
# the forward-fill convention are stated once in the table caption
# instead of being repeated in every one of the ~11 rows that use them.
LAG_BASE_SHORT_LABELS = {
    "Active_Power": "Active power",
    "k_p": "k_p",
    "k_ghi": "k_ghi",
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
                "shift": "0",
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
                    "description": f"{LAG_BASE_SHORT_LABELS[base]} {desc_suffix}",
                }
            )
        shift_text, desc_suffix = LAG_SHIFT_TEXT[LAG_DAILY_SUFFIX]
        rows.append(
            {
                "feature": f"{base}{LAG_DAILY_SUFFIX}",
                "category": "lagged observations",
                "regime": "lagged + oracle",
                "shift": shift_text,
                "description": f"{LAG_BASE_SHORT_LABELS[base]} {desc_suffix}",
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
                "shift": "h(w)",
                "description": f"{stat.capitalize()} of {base}, trailing {window}h",
            }
        )
    for base, window, stat in OBS_ROLLING_SPECS:
        rows.append(
            {
                "feature": f"{base}_last{window}obs_{stat}",
                "category": "rolling statistics",
                "regime": "lagged + oracle",
                "shift": "h(o)",
                "description": f"{stat.capitalize()} of {base}, last {window} valid obs",
            }
        )

    # --- C: oracle weather ---
    for col in WEATHER_COLS:
        rows.append(
            {
                "feature": f"{ORACLE_PREFIX}{col}",
                "category": "oracle weather",
                "regime": "oracle only",
                "shift": "0",
                "description": f"Measured {WEATHER_DESCRIPTIONS_MIDSENTENCE[col]} at target time t",
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
    lines.append("% Columns reduced from 5 to 3 to fit IEEE two-column width")
    lines.append("% and height (paper/overleaf compile-fix pass): dropped")
    lines.append("% Category (was always emitted as an empty string - the")
    lines.append("% category is shown by the multicolumn group header instead,")
    lines.append("% this column had never carried content) and Regime (lagged")
    lines.append("% + oracle for every row except the oracle-weather group,")
    lines.append("% which is oracle-only - stated once in the caption instead")
    lines.append("% of repeated 42 times). Shift codes are compacted")
    lines.append("% (0/h/h+1/h+2/24/h(w)/h(o)) with a legend in the caption,")
    lines.append("% and the k_p/k_ghi ratio + forward-fill definitions and the")
    lines.append("% wall-clock-vs-valid-observation rolling-window distinction")
    lines.append("% are stated once in the caption instead of per row.")
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\scriptsize")
    lines.append(
        "  \\caption{Feature list by regime: all 37 lagged-regime features, "
        "plus the 5 additional oracle-only features (42 total), grouped "
        "into four categories (deterministic at target time, lagged "
        "observations, rolling statistics, oracle weather). All features "
        "in the first three groups belong to both regimes; the oracle "
        "weather group belongs to the oracle regime only, is measured AT "
        "TARGET TIME $t$, and is a perfect-forecast upper bound, never "
        "achievable and never mixed with the lagged regime in one feature "
        "matrix (CLAUDE.md rule 5). Shift, relative to issue time $t-h$ "
        "or target time $t$: 0 = deterministic at $t$ (first group) or "
        "measured at $t$ (oracle group); h/h+1/h+2 = issue time and the "
        "two preceding hours; 24 = fixed 24-hour lag (same hour, previous "
        "day); h(w) = wall-clock window ending at issue time; h(o) = "
        "window of the last $N$ valid (non-NaN) observations at or before "
        "issue time, used instead of a wall-clock window for $k_p$ and "
        "$k_{ghi}$-based statistics because both are undefined at night. "
        "$k_p$ = Active\\_Power / p\\_cs and $k_{ghi}$ = the clear-sky "
        "index of GHI, both forward-filled up to 24h before shifting.}"
    )
    lines.append("  \\label{tab:T2}")
    lines.append("  \\begin{tabular}{ll p{4.5cm}}")
    lines.append("    \\toprule")
    lines.append("    Feature & Shift & Description \\\\")
    lines.append("    \\midrule")

    by_category = {cat: [] for cat in CATEGORY_ORDER}
    for r in rows:
        by_category[r["category"]].append(r)

    for cat_idx, cat in enumerate(CATEGORY_ORDER):
        cat_rows = by_category[cat]
        lines.append(
            f"    \\multicolumn{{3}}{{l}}{{\\textit{{{tex_escape(cat)} "
            f"({len(cat_rows)} features)}}}} \\\\"
        )
        for r in cat_rows:
            lines.append(
                "    "
                + " & ".join(
                    [
                        f"\\texttt{{{tex_escape(r['feature'])}}}",
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
