"""Chronological train/validation/test split, per CLAUDE.md rule 1: never
shuffle a time series.

Splits are whole calendar years in Australia/Darwin local time:
train = 2009-2013, val = 2014, test = 2015.

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

TRAIN_YEARS = (2009, 2010, 2011, 2012, 2013)
VAL_YEARS = (2014,)
TEST_YEARS = (2015,)
# VAL_YEARS and TEST_YEARS must NOT be varied by any experiment - the test
# set is touched once, at the end (CLAUDE.md research integrity rule).
# Only TRAIN_YEARS may be varied, e.g. for a training-length ablation.


def split_chronological(df, train_years=TRAIN_YEARS, val_years=VAL_YEARS,
                          test_years=TEST_YEARS):
    """Split df into (train, val, test) by calendar year, in local time.

    Defaults to train = 2009-2013, val = 2014, test = 2015. Year boundaries
    are evaluated in Australia/Darwin local time; df.index must already be
    tz-aware in that zone (as produced by src.data.loader.load_array).

    Raises AssertionError if train_years/val_years/test_years are not
    pairwise disjoint, if they are not strictly chronological
    (max(train_years) < min(val_years) < min(test_years)), if any pair of
    resulting splits shares a timestamp, if the splits are not strictly
    chronological at the timestamp level, or if the three split lengths do
    not sum to len(df).
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

    _validate_split(df, train, val, test)

    return train, val, test


def _validate_split(df, train, val, test):
    total = len(train) + len(val) + len(test)
    assert total == len(df), (
        f"split lengths {len(train)} + {len(val)} + {len(test)} = {total} "
        f"do not sum to input length {len(df)}"
    )

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
