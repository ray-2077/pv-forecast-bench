"""Chronological train/validation/test split, per CLAUDE.md rule 1: never
shuffle a time series.

Splits are whole calendar years in Australia/Darwin local time:
train = 2011-2013, val = 2014, test = 2015.

CRITICAL - do NOT filter daylight here. Keep all 24 hours in every split.
Daylight filtering is an EVALUATION choice, not a data choice: RQ2 measures
how much reported accuracy changes when night hours are included versus
excluded. If the daylight filter were baked into the splits, that
comparison would be impossible to run later - there would be no night
hours left to add back in. Filtering happens downstream, at evaluation
time, not here.
"""

import pandas as pd

TZ = "Australia/Darwin"

# TRAIN_YEARS starts at 2011, not 2009, because array17 (added to the
# pipeline for cross-array comparison - see scripts/diagnose_array17_events.py)
# was installed 11 March 2010: 2009 and early 2010 predate it entirely. All
# arrays sharing one training window - rather than each getting whatever
# years its own history happens to cover - is required so the cross-array
# comparison measures the arrays, not a confound from differing training
# length. TRAIN_YEARS is still deliberately kept variable (not folded into
# a single hardcoded split): a training-length ablation on array11 and
# array12, which do have real data back to 2009, is valid future work and
# should vary this constant rather than touch VAL_YEARS/TEST_YEARS.
TRAIN_YEARS = (2011, 2012, 2013)
VAL_YEARS = (2014,)
TEST_YEARS = (2015,)
# VAL_YEARS and TEST_YEARS must NOT be varied by any experiment - the test
# set is touched once, at the end (CLAUDE.md research integrity rule).
# Only TRAIN_YEARS may be varied, e.g. for a training-length ablation.


def split_chronological(df, train_years=TRAIN_YEARS, val_years=VAL_YEARS,
                          test_years=TEST_YEARS):
    """Split df into (train, val, test) by calendar year, in local time.

    Defaults to train = 2011-2013, val = 2014, test = 2015. Year boundaries
    are evaluated in Australia/Darwin local time; df.index must already be
    tz-aware in that zone (as produced by src.data.loader.load_array).

    df may legitimately contain rows outside [min(train_years)-01-01,
    max(test_years)-12-31] - e.g. the processed parquet covers 2009-2015
    while TRAIN_YEARS starts at 2011 (array17 wasn't installed until March
    2010, see the TRAIN_YEARS comment above). Those out-of-window rows are
    silently excluded from all three splits; the count is printed so that
    is visible rather than silent. Any row INSIDE the window landing in no
    split (or more than one) is still a bug and still raises.

    Raises AssertionError if train_years/val_years/test_years are not
    pairwise disjoint, if they are not strictly chronological
    (max(train_years) < min(val_years) < min(test_years)), if any pair of
    resulting splits shares a timestamp, if the splits are not strictly
    chronological at the timestamp level, or if the rows of df inside the
    split window are not each covered by exactly one split.
    """
    train_set, val_set, test_set = set(train_years), set(val_years), set(test_years)
    assert train_set.isdisjoint(val_set), "train_years and val_years overlap"
    assert train_set.isdisjoint(test_set), "train_years and test_years overlap"
    assert val_set.isdisjoint(test_set), "val_years and test_years overlap"
    assert max(train_years) < min(val_years) < min(test_years), (
        "train_years, val_years, test_years are not strictly chronological"
    )

    train_start = pd.Timestamp(f"{min(train_years)}-01-01", tz=TZ)
    val_start = pd.Timestamp(f"{min(val_years)}-01-01", tz=TZ)
    test_start = pd.Timestamp(f"{min(test_years)}-01-01", tz=TZ)
    test_end = pd.Timestamp(f"{max(test_years) + 1}-01-01", tz=TZ)

    train = df.loc[(df.index >= train_start) & (df.index < val_start)]
    val = df.loc[(df.index >= val_start) & (df.index < test_start)]
    test = df.loc[(df.index >= test_start) & (df.index < test_end)]

    n_out_of_window = _validate_split(df, train, val, test, train_start, test_end)
    if n_out_of_window:
        print(
            f"split_chronological: {n_out_of_window} row(s) fall outside "
            f"[{train_start.date()}, {(test_end - pd.Timedelta(days=1)).date()}] "
            "and were excluded from every split (expected when the input "
            "covers more years than train_years/val_years/test_years)"
        )

    return train, val, test


def _validate_split(df, train, val, test, window_start, window_end):
    """Validate disjointness, chronological order, and full coverage of the
    split WINDOW only (window_end is exclusive).

    The original version of this check asserted
    len(train)+len(val)+len(test) == len(df), which is only valid when the
    input df is already trimmed to exactly the split years. That stopped
    being true once TRAIN_YEARS was narrowed to 2011-2013 while the
    processed parquet still covers 2009-2015 (array17's March 2010 install
    date - see TRAIN_YEARS above): the 2009-2010 rows are supposed to
    belong to no split, and the strict equality would fail on every call
    even though nothing is wrong. The replacement checks full coverage
    only within [window_start, window_end) - the range this call's
    train_years/test_years actually claim - so an accidentally dropped row
    inside that range still raises, while deliberately-out-of-range rows
    do not. Returns the count of out-of-window rows so the caller can log
    it.
    """
    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)

    assert train_idx.isdisjoint(val_idx), "train and val share timestamps"
    assert train_idx.isdisjoint(test_idx), "train and test share timestamps"
    assert val_idx.isdisjoint(test_idx), "val and test share timestamps"

    if len(train) and len(val):
        assert train.index.max() < val.index.min(), (
            "train is not entirely earlier than val"
        )
    if len(val) and len(test):
        assert val.index.max() < test.index.min(), (
            "val is not entirely earlier than test"
        )

    in_window = df.loc[(df.index >= window_start) & (df.index < window_end)]
    total_in_splits = len(train) + len(val) + len(test)
    assert total_in_splits == len(in_window), (
        f"rows inside the split window [{window_start.date()}, "
        f"{(window_end - pd.Timedelta(days=1)).date()}] are not each covered by "
        f"exactly one split: {len(train)} + {len(val)} + {len(test)} = "
        f"{total_in_splits}, expected {len(in_window)}"
    )

    return len(df) - len(in_window)
