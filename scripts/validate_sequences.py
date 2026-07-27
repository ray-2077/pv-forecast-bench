"""Validate src/features/sequences.py and src/features/scaling.py on
array11 for horizons 1, 3, 6.

Uses train+val only, built the same way as scripts/validate_features.py:
clear-sky power's temperature climatology and gain are fit on TRAINING
years only, then applied to train+val's own index. The 2015 test split is
never loaded here - this is feature-building validation, not a modelling
result, and per CLAUDE.md the test set is touched once, at the end.

Prints, per horizon:
1. sequence shape, static shape, row count, per regime
2. assert_no_leakage_sequences PASS/FAIL
3. two deliberate corruptions of a correct X_seq, and confirmation that
   assert_no_leakage_sequences catches both:
   - np.roll along the time axis (shifts every timestep)
   - substituting the value AT TARGET TIME t into the last timestep only
     (the actual bug shape this check exists for: a window ending at t
     instead of t-h)
4. fits a Scaler on the training rows of the static features, prints the
   mean/std of three columns before and after transform, asserts the
   transformed training data has mean ~ 0 and std ~ 1, and asserts that
   transforming the validation rows does NOT give mean ~ 0 (proof the
   scaler was not refit on validation)

Usage:
    python scripts/validate_sequences.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky import (
    add_clearsky,
    add_clearsky_index_ghi,
    add_daylight_mask,
    add_solar_position,
)
from src.data.clearsky_power import (
    GAMMA_PDC_SILICON,
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
from src.data.splits import TRAIN_YEARS, VAL_YEARS, split_chronological
from src.features.scaling import Scaler
from src.features.sequences import (
    FFILL_SEQUENCE_CHANNELS,
    SEQUENCE_CHANNELS,
    assert_no_leakage_sequences,
    build_sequences,
    static_feature_names,
)
from src.features.build import LAG_FFILL_LIMIT

PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "array11_polySi_hourly.parquet"
NAMEPLATE_KW = 5.0
GAMMA_PDC = GAMMA_PDC_SILICON

HORIZONS = [1, 3, 6]
REGIMES = ["lagged", "oracle"]
SEQ_LEN = 24


def load_array11_train_val():
    """Same construction as scripts/validate_features.py's
    load_array11_train_val - duplicated here rather than imported, per
    this repo's convention of self-contained validate scripts.
    """
    df = pd.read_parquet(PROCESSED_PATH)
    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)

    train, val, _test = split_chronological(df)  # 2015 test never touched

    temp_clim = fit_temperature_climatology(train)
    p_cs_raw_train = model_clearsky_power(train.index, NAMEPLATE_KW, GAMMA_PDC, temp_clim)
    gain, n_gain_hours, gain_iqr = fit_gain(train, p_cs_raw_train)
    print(f"gain (fit on training years only): {gain:.4f}  "
          f"(hours used: {n_gain_hours}, IQR: {gain_iqr:.4f})")

    train_val = pd.concat([train, val]).sort_index()
    p_cs_raw = model_clearsky_power(train_val.index, NAMEPLATE_KW, GAMMA_PDC, temp_clim)
    train_val = add_clearsky_power(train_val, p_cs_raw, gain, NAMEPLATE_KW)

    return train_val


def channel_values_at(df, times):
    """(len(times), len(SEQUENCE_CHANNELS)) array of each channel's value
    AT the given timestamps (k_p/k_ghi forward-filled first, same as the
    sequence window itself) - used to build the "window ending at t"
    corruption.
    """
    columns = []
    for channel in SEQUENCE_CHANNELS:
        if channel in FFILL_SEQUENCE_CHANNELS:
            source = df[channel].ffill(limit=LAG_FFILL_LIMIT)
        else:
            source = df[channel]
        columns.append(source.reindex(times).to_numpy(dtype=float))
    return np.column_stack(columns)


def check_leakage_corruptions(df, X_seq, index, horizon):
    """Corrupt a known-good X_seq two ways and confirm
    assert_no_leakage_sequences catches both."""

    rolled = np.roll(X_seq, shift=1, axis=1)
    try:
        assert_no_leakage_sequences(df, rolled, index, horizon, SEQ_LEN)
        print("  [corruption: np.roll]           FAIL - was NOT caught")
    except AssertionError:
        print("  [corruption: np.roll]           PASS - caught")

    substituted = X_seq.copy()
    substituted[:, -1, :] = channel_values_at(df, index)
    try:
        assert_no_leakage_sequences(df, substituted, index, horizon, SEQ_LEN)
        print("  [corruption: window ends at t]  FAIL - was NOT caught")
    except AssertionError:
        print("  [corruption: window ends at t]  PASS - caught")


def split_rows_by_year(index, X_static, years):
    mask = np.isin(index.year, years)
    return X_static[mask]


def check_scaler(X_static, index, regime):
    names = static_feature_names(regime)
    preview_cols = names[:3]
    preview_idx = [names.index(c) for c in preview_cols]

    train_static = split_rows_by_year(index, X_static, TRAIN_YEARS)
    val_static = split_rows_by_year(index, X_static, VAL_YEARS)

    print(f"  [Scaler] fit on train (n={len(train_static)}), "
          f"transform train (n={len(train_static)}) and val (n={len(val_static)})")
    print(f"  columns previewed: {preview_cols}")
    print(f"  before: mean={train_static[:, preview_idx].mean(axis=0)}  "
          f"std={train_static[:, preview_idx].std(axis=0)}")

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_static)
    val_scaled = scaler.transform(val_static)

    print(f"  after:  mean={train_scaled[:, preview_idx].mean(axis=0)}  "
          f"std={train_scaled[:, preview_idx].std(axis=0)}")

    train_mean = train_scaled.mean(axis=0)
    train_std = train_scaled.std(axis=0)
    assert np.allclose(train_mean, 0.0, atol=1e-6), (
        f"scaled training mean not ~0: {train_mean}"
    )
    assert np.allclose(train_std, 1.0, atol=1e-6), (
        f"scaled training std not ~1: {train_std}"
    )
    print("  [Scaler] train mean ~ 0, std ~ 1: PASS")

    val_mean = val_scaled.mean(axis=0)
    assert not np.allclose(val_mean, 0.0, atol=1e-3), (
        "scaled validation mean is ~0 - the scaler appears to have been "
        "refit on validation data"
    )
    print(f"  [Scaler] validation transformed mean (should NOT be ~0): "
          f"{val_mean[preview_idx]}")
    print("  [Scaler] validation mean != 0: PASS (scaler was fit on train only)")


def main():
    df = load_array11_train_val()

    for horizon in HORIZONS:
        print(f"\n{'=' * 78}")
        print(f"array11_polySi, horizon = {horizon}h, seq_len = {SEQ_LEN}")
        print("=" * 78)

        built = {}
        for regime in REGIMES:
            X_seq, X_static, y, index = build_sequences(df, horizon, regime, SEQ_LEN)
            built[regime] = (X_seq, X_static, y, index)

            print(f"\n[{regime}] X_seq shape={X_seq.shape}  "
                  f"X_static shape={X_static.shape}  n_rows={len(index)}")

            try:
                assert_no_leakage_sequences(df, X_seq, index, horizon, SEQ_LEN)
                print(f"[{regime}] assert_no_leakage_sequences: PASS")
            except AssertionError as e:
                print(f"[{regime}] assert_no_leakage_sequences: FAIL - {e}")

        X_seq_lagged, X_static_lagged, _y_lagged, index_lagged = built["lagged"]

        print(f"\n[lagged] injected off-by-one corruption checks:")
        check_leakage_corruptions(df, X_seq_lagged, index_lagged, horizon)

        print(f"\n[lagged] Scaler check:")
        check_scaler(X_static_lagged, index_lagged, "lagged")


if __name__ == "__main__":
    main()
