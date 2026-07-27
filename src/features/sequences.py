"""Build (sequence, static, y) tensors for recurrent models (LSTM,
CNN-LSTM). This module reuses the alignment convention, the lag/ffill
logic, and the regime split from src/features/build.py - it does NOT
reimplement any of that; it imports build.py's private helpers directly
so a bug fixed there is not silently left unfixed here.

ALIGNMENT (same convention as build.py): a row at target time t, horizon
h, is the input to a forecast issued at t-h. The SEQUENCE input for that
row is the window [t-h-seq_len+1 ... t-h] of per-hour channels - every
element at or before the issue time t-h. The STATIC input is build.py's
Category A deterministic set (evaluated unshifted at t, since it is
computable in advance - solar geometry, clear-sky power, calendar) plus
the k_p staleness pair, plus oracle_ weather at t for the oracle regime
only.

Requires df to already carry the same upstream columns as build.py (see
build.py's REQUIRED_COLS / module docstring): add_solar_position,
add_clearsky, add_clearsky_index_ghi, and add_clearsky_power must have
already run, with any fitted parameters (temperature climatology, gain)
coming from TRAINING data only.

CONSTRAINTS: numpy/pandas only, no torch import (testable without a GPU).
Deterministic. Does not modify src/features/build.py. Does not scale.
"""

import numpy as np
import pandas as pd

from src.features.build import (
    K_P_HOURS_STALE_COL,
    K_P_IS_STALE_COL,
    LAG_FFILL_LIMIT,
    DETERMINISTIC_NAMES,
    _check_regime,
    _check_required_columns,
    _deterministic_features,
    _hours_since_valid,
    _is_stale,
    _lag_source,
    _oracle_feature_names,
    _oracle_features,
)

# The per-hour channels inside the sequence window, in this fixed order -
# order matters, since it defines the last axis of X_seq.
SEQUENCE_CHANNELS = [
    "Active_Power",
    "k_p",
    "k_ghi",
    "Global_Horizontal_Radiation",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Wind_Speed",
    "solar_elevation",
    "p_cs",
]

# k_p and k_ghi are undefined at night (see clearsky_power.py); forward-
# fill them (limit LAG_FFILL_LIMIT, via build.py's _lag_source) before
# windowing, exactly as build.py does for its B-category lag features -
# see that module's docstring for why this cannot leak (ffill only ever
# copies a PAST value forward in time).
FFILL_SEQUENCE_CHANNELS = ("k_p", "k_ghi")


def _sequence_channel_series(df, channel):
    if channel in FFILL_SEQUENCE_CHANNELS:
        return _lag_source(df, channel)
    return df[channel]


def _build_channel_window(series, horizon, seq_len):
    """(n_timestamps, seq_len) array: column j is the value at
    t - horizon - (seq_len - 1 - j) hours, i.e. row t's window
    [t-h-seq_len+1 ... t-h] in ascending time order - column seq_len-1
    (the last timestep) is always the value at t-h, never t.
    """
    columns = []
    for j in range(seq_len):
        shift_amount = horizon + seq_len - 1 - j
        columns.append(series.shift(shift_amount).to_numpy())
    return np.column_stack(columns)


def static_feature_names(regime):
    """Column names/order of the STATIC block build_sequences(...)
    returns for `regime`, without building it.
    """
    _check_regime(regime)
    names = list(DETERMINISTIC_NAMES) + [K_P_HOURS_STALE_COL, K_P_IS_STALE_COL]
    if regime == "oracle":
        names += _oracle_feature_names()
    return names


def build_sequences(df, horizon, regime, seq_len=24):
    """Build (X_seq, X_static, y, index) for one horizon and regime.

    X_seq: float ndarray, shape (n_rows, seq_len, len(SEQUENCE_CHANNELS)).
    X_static: float ndarray, shape (n_rows, len(static_feature_names(regime))),
        columns in that order.
    y: float ndarray, shape (n_rows,) - Active_Power at target time t.
    index: DatetimeIndex of target times t, one per row.

    Rows with any NaN (in the sequence window, the static block, or y) are
    dropped. Does not scale, normalise, or impute.
    """
    _check_regime(regime)
    _check_required_columns(df)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")

    channel_windows = [
        _build_channel_window(_sequence_channel_series(df, channel), horizon, seq_len)
        for channel in SEQUENCE_CHANNELS
    ]
    X_seq_full = np.stack(channel_windows, axis=-1)  # (n, seq_len, n_channels)

    static_full = _deterministic_features(df)
    hours_stale = _hours_since_valid(df["k_p"], LAG_FFILL_LIMIT).shift(horizon)
    static_full[K_P_HOURS_STALE_COL] = hours_stale
    static_full[K_P_IS_STALE_COL] = _is_stale(hours_stale)
    if regime == "oracle":
        static_full = pd.concat([static_full, _oracle_features(df)], axis=1)
    static_full = static_full[static_feature_names(regime)]

    y_full = df["Active_Power"].copy()

    seq_has_nan = np.isnan(X_seq_full).any(axis=(1, 2))
    static_has_nan = static_full.isna().any(axis=1).to_numpy()
    y_has_nan = y_full.isna().to_numpy()
    valid = ~seq_has_nan & ~static_has_nan & ~y_has_nan

    index = df.index[valid]
    X_seq = X_seq_full[valid]
    X_static = static_full.to_numpy(dtype=float)[valid]
    y = y_full.to_numpy(dtype=float)[valid]

    return X_seq, X_static, y, index


def assert_no_leakage_sequences(df, X_seq, index, horizon, seq_len):
    """Real leakage check: verify by INDEPENDENT reconstruction that the
    LAST timestep of every sequence (X_seq[:, -1, :]) corresponds to the
    issue time t-h, not the target time t. This is the single most likely
    bug (an off-by-one in the window's shift arithmetic), so every channel
    is checked, not just one.

    For each channel, rebuilds the expected last-timestep value directly
    from df by inlining the same ffill (for k_p/k_ghi) and a plain
    timestamp lookup at t - horizon hours - NOT by calling
    _build_channel_window/_lag_source - so a bug shared with those
    functions would not be silently repeated here. Raises AssertionError
    naming the first offending channel and row count if any mismatch.
    """
    _check_required_columns(df)
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")

    issue_times = index - pd.Timedelta(hours=horizon)

    for ch_pos, channel in enumerate(SEQUENCE_CHANNELS):
        if channel in FFILL_SEQUENCE_CHANNELS:
            rebuilt_source = df[channel].ffill(limit=LAG_FFILL_LIMIT)
        else:
            rebuilt_source = df[channel]

        expected_last = rebuilt_source.reindex(issue_times).to_numpy(dtype=float)
        actual_last = X_seq[:, seq_len - 1, ch_pos].astype(float)

        both_nan = np.isnan(expected_last) & np.isnan(actual_last)
        close = np.isclose(expected_last, actual_last, atol=1e-9, rtol=0.0)
        ok = both_nan | close

        if not bool(np.all(ok)):
            n_bad = int((~ok).sum())
            raise AssertionError(
                f"assert_no_leakage_sequences: channel {channel!r}'s last "
                f"timestep does not match the issue time t-{horizon}h at "
                f"{n_bad} row(s) - the sequence window is misaligned "
                "(off-by-one) or leaking data from the target time"
            )
