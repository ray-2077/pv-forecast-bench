"""Diagnose the array17 val-to-test skill_vs_convex shift before it goes
in the paper.

MOTIVATING EVIDENCE (results/seed_sweep_summary_lagged.csv vs
results/seed_sweep_summary_lagged_test.csv, mean_skill_vs_convex, all 5
models x 3 horizons): array17's test skill is lower than its val skill
in all 15 cells, by 0.011 to 0.076. array11 and array12 both move the
OTHER way (test higher than val) in all 18 of their cells. The direction
is consistent across every model and horizon at array17, so this is an
array-level effect, not an architecture one - no model choice explains
it, and none is proposed as a fix here.

Two candidate causes, both diagnosed below, plus a check on the
reference forecast itself:

1. THE JUNE 2015 OUTAGE. src/eval/exclusions.py's KNOWN_OUTAGES excludes
   array17 readings for 2015-06-05..09 (DKASC-documented, 7 arrays
   switched off, discovered 9 June). If real, degraded readings persist
   in the UNEXCLUDED days immediately before or after that window, the
   exclusion is too narrow and contaminated hours are still entering the
   reported test metrics. Checked via three daily tables, 2015-06-01 to
   2015-06-15: array17 (the case), array11 over the same dates (weather
   control - if array11 dips on the same days, it's weather, not the
   outage), and array17 in 2014 over the same calendar dates (seasonal
   control - rules out an ordinary early-winter dip that just happens to
   coincide with the outage window). Both raw Active_Power (was output
   depressed) and k_p (was output depressed RELATIVE TO WHAT THE SKY
   ALLOWED that day) are printed, since they answer different questions.

2. DEGRADATION / CLEAR-SKY-INDEX DRIFT. The clear-sky power gain is fit
   once on TRAIN (2011-2013) and applied UNCHANGED to val (2014) and
   test (2015) - CLAUDE.md's own rule (scalers/statistics fit on
   training data only). If array17's real-world output has drifted since
   2013 (soiling, degradation, an uncorrected calibration change) while
   array11/array12 have not, k_p - and everything downstream of it:
   features, smart persistence, the convex reference - would be
   systematically biased for array17 on the later years in a way the
   other two arrays would not share. Checked via mean daylight k_p and
   mean daylight performance ratio (mean Active_Power / mean GHI, which
   normalises out inter-annual weather so a trend here is the array
   itself, not the weather) for all three arrays, every year 2011-2015,
   printed side by side.

REFERENCE-FORECAST CHECK. The convex weight w is fit by grid search
against RMSE on val ONLY in every real run (CLAUDE.md rule 4;
src.models.climatology.ConvexCombination.fit's own docstring: "there is
no path here that could touch a test split"). This script additionally
fits w a SECOND time per horizon, diagnostically, AS IF test were the
fitting split - i.e. what would the optimal persistence/climatology
blend have been had test been fit like val is - purely to compare
against the real, val-fit w. This second fit is NEVER used to produce
a reported result; it exists only to answer whether the reference
forecast's own optimal blend genuinely differs between the two years
for array17. If it does, part of the val-to-test shift is the reference
moving under the models' feet, not the models moving.

Uses the same pipeline as scripts/build_table6_dm.py --eval-split test:
load_and_prepare -> split_chronological -> temperature climatology and
gain fit on TRAIN ONLY -> clear-sky power applied unchanged to val
(add_clearsky_power_per_split) and to test (add_clearsky_power_to_test
below, a local duplicate of build_table6_dm.py's own helper of the same
name - not imported, since scripts/run_final_test.py, the other place
this logic exists, is write-once and must not be imported from or
touched).

No fixes, no plotting, no changes to src/. Diagnosis only.

Usage:
    python scripts/diagnose_array17_test_shift.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky_power import (
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
from src.data.pipeline import add_clearsky_power_per_split, load_and_prepare
from src.data.splits import split_chronological
from src.eval.exclusions import TZ, exclusion_mask
from src.models.climatology import ConvexCombination

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ARRAYS = ["array11", "array12", "array17"]
HORIZONS = (1, 3, 6)


def add_clearsky_power_to_test(train, test, nameplate_kw, gamma_pdc):
    """Same train-only-fitted temperature climatology and gain
    add_clearsky_power_per_split already used for train/val, applied to
    test too - never refitting anything on val or test. Duplicated from
    scripts/build_table6_dm.py's own function of the same name rather
    than imported (that script is not a shared library module); fit_gain
    is deterministic given `train`, so re-deriving it here is equivalent
    to reusing it, not a second, possibly-different fit.
    """
    temp_clim = fit_temperature_climatology(train)
    p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
    gain, _n_gain_hours, _gain_iqr = fit_gain(train, p_cs_raw_train)

    p_cs_raw_test = model_clearsky_power(test.index, nameplate_kw, gamma_pdc, temp_clim)
    return add_clearsky_power(test, p_cs_raw_test, gain, nameplate_kw)


def load_all_splits(array):
    """train, val, test, combined(=train+val+test, 2011-2015) - all with
    p_cs/k_p added, test via add_clearsky_power_to_test above.
    """
    df, nameplate_kw, gamma_pdc = load_and_prepare(array, PROCESSED_DIR)
    train, val, test = split_chronological(df)
    train, val, gain_info = add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc)
    test = add_clearsky_power_to_test(train, test, nameplate_kw, gamma_pdc)
    combined = pd.concat([train, val, test]).sort_index()
    return train, val, test, combined, nameplate_kw, gain_info


def daily_window_table(combined, array, start_date, end_date_inclusive):
    """Daily mean daylight Active_Power, daily mean daylight k_p,
    daylight-hour count, and an outage-window flag (from the real
    src.eval.exclusions.exclusion_mask, not a re-typed date literal), one
    row per calendar date in [start_date, end_date_inclusive].
    """
    daylight = combined[combined["is_daylight"]]
    start_ts = pd.Timestamp(start_date, tz=TZ)
    end_ts = pd.Timestamp(end_date_inclusive, tz=TZ) + pd.Timedelta(days=1)
    window = daylight[(daylight.index >= start_ts) & (daylight.index < end_ts)]

    outage = exclusion_mask(array, window.index)
    grouped = window.groupby(window.index.date)
    table = pd.DataFrame(
        {
            "mean_active_power_kw": grouped["Active_Power"].mean(),
            "mean_k_p": grouped["k_p"].mean(),
            "n_daylight_hours": grouped.size(),
            "in_excluded_outage_window": pd.Series(outage.to_numpy(), index=window.index)
            .groupby(window.index.date)
            .any(),
        }
    )
    table.index.name = "date"
    return table


def yearly_stats(combined):
    """Per-year (2011-2015, from `combined`), daylight-only: mean
    Active_Power, mean GHI, the weather-normalised performance ratio
    mean(Active_Power)/mean(GHI), and mean k_p.
    """
    daylight = combined[combined["is_daylight"]]
    grouped = daylight.groupby(daylight.index.year)
    stats = pd.DataFrame(
        {
            "mean_active_power_kw": grouped["Active_Power"].mean(),
            "mean_ghi_wm2": grouped["Global_Horizontal_Radiation"].mean(),
            "mean_k_p": grouped["k_p"].mean(),
            "n_daylight_hours": grouped.size(),
        }
    )
    stats["performance_ratio"] = stats["mean_active_power_kw"] / stats["mean_ghi_wm2"]
    stats.index.name = "year"
    return stats


def convex_weight_val_vs_test(train, val, test):
    """Real, val-fit w (as every actual run computes it) vs a DIAGNOSTIC-
    ONLY w fit the same way but against test instead of val - never used
    to produce a reported result, see module docstring.
    """
    rows = []
    for h in HORIZONS:
        w_val = ConvexCombination().fit(train, h, val).w
        w_test_diagnostic = ConvexCombination().fit(train, h, test).w
        rows.append(
            {
                "horizon": h,
                "w_val (real)": w_val,
                "w_test (diagnostic only)": w_test_diagnostic,
                "diff": w_test_diagnostic - w_val,
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def main():
    print("=" * 90)
    print("loading train/val/test for array11, array12, array17 "
          "(TRAIN_YEARS fit; test via local add_clearsky_power_to_test)")
    print("=" * 90)

    splits = {}
    for array in ARRAYS:
        train, val, test, combined, nameplate_kw, gain_info = load_all_splits(array)
        splits[array] = {"train": train, "val": val, "test": test, "combined": combined}
        print(
            f"  {array}: gain={gain_info['gain']:.4f} "
            f"(n_hours={gain_info['gain_n_hours']}, IQR={gain_info['gain_iqr']:.4f})"
        )

    # ------------------------------------------------------------------
    # Candidate 1: the June 2015 outage
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CANDIDATE 1: June 2015 outage - daily daylight Active_Power and k_p, 2015-06-01..15")
    print("Outage window (src.eval.exclusions.KNOWN_OUTAGES): 2015-06-05..09 inclusive, array17 only")
    print("=" * 90)

    print("\n--- array17, 2015 (the case) ---")
    t_array17_2015 = daily_window_table(splits["array17"]["combined"], "array17", "2015-06-01", "2015-06-15")
    print(t_array17_2015.round(4).to_string())

    print("\n--- array11, 2015 (weather control - same dates, no outage at this array) ---")
    t_array11_2015 = daily_window_table(splits["array11"]["combined"], "array11", "2015-06-01", "2015-06-15")
    print(t_array11_2015.round(4).to_string())

    print("\n--- array17, 2014 (seasonal control - same calendar dates, one year earlier) ---")
    t_array17_2014 = daily_window_table(splits["array17"]["combined"], "array17", "2014-06-01", "2014-06-15")
    print(t_array17_2014.round(4).to_string())

    # ------------------------------------------------------------------
    # Candidate 2: degradation / clear-sky-index drift
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CANDIDATE 2: degradation drift - yearly mean k_p and performance_ratio, 2011-2015, all three arrays")
    print("performance_ratio = mean(Active_Power)/mean(GHI), daylight only - normalises out inter-annual weather")
    print("watch for: array17's k_p/performance_ratio declining year on year while array11/array12 stay flat")
    print("=" * 90)

    yearly_by_array = {array: yearly_stats(splits[array]["combined"]) for array in ARRAYS}
    combined_yearly = pd.concat(
        {array: yearly_by_array[array][["performance_ratio", "mean_k_p"]] for array in ARRAYS},
        axis=1,
    )
    print("\n" + combined_yearly.round(4).to_string())

    print("\n--- full per-array yearly detail (mean_active_power_kw, mean_ghi_wm2, n_daylight_hours) ---")
    for array in ARRAYS:
        print(f"\n{array}:")
        print(yearly_by_array[array].round(4).to_string())

    # ------------------------------------------------------------------
    # Reference-forecast check: convex weight w, val vs test, array17
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("REFERENCE CHECK: array17 convex weight w, real (val-fit) vs diagnostic-only (test-fit)")
    print("w_test (diagnostic only) is NEVER used to produce a reported result - see module docstring")
    print("=" * 90 + "\n")

    w_table = convex_weight_val_vs_test(splits["array17"]["train"], splits["array17"]["val"], splits["array17"]["test"])
    print(w_table.round(4).to_string())


if __name__ == "__main__":
    main()
