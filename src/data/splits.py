"""Chronological train/validation/test split, per CLAUDE.md rule 1: never
shuffle a time series.

Splits are whole calendar years in Australia/Darwin local time:
train = 2012-2013, val = 2014, test = 2015.

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

TRAIN_YEARS = (2012, 2013)
VAL_YEAR = 2014
TEST_YEAR = 2015


def split_chronological(df):
    """Split df into (train, val, test) by calendar year, in local time.

    train = 2012 and 2013, val = 2014, test = 2015. Year boundaries are
    evaluated in Australia/Darwin local time; df.index must already be
    tz-aware in that zone (as produced by src.data.loader.load_array).

    Raises AssertionError if any pair of splits shares a timestamp, if the
    splits are not strictly chronological (every train timestamp earlier
    than every val timestamp, every val timestamp earlier than every test
    timestamp), or if the three split lengths do not sum to len(df).
    """
    train_start = pd.Timestamp(f"{TRAIN_YEARS[0]}-01-01", tz=TZ)
    val_start = pd.Timestamp(f"{VAL_YEAR}-01-01", tz=TZ)
    test_start = pd.Timestamp(f"{TEST_YEAR}-01-01", tz=TZ)
    test_end = pd.Timestamp(f"{TEST_YEAR + 1}-01-01", tz=TZ)

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
