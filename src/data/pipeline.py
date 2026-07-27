"""Shared per-array data preparation, used by every model dev script
(scripts/run_xgb_dev.py, scripts/run_lstm_dev.py, scripts/run_seed_sweep.py).

Moved out of run_xgb_dev.py/run_lstm_dev.py where it was duplicated
identically in both files - see scripts/run_seed_sweep.py's docstring.

ARRAYS excludes array07 - see CLAUDE.md "Data window" and
results/dead_period_audit.csv.
"""

from src.data.clearsky_power import (
    GAMMA_PDC_HIT,
    GAMMA_PDC_SILICON,
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)
from src.data.clearsky import (
    add_clearsky,
    add_clearsky_index_ghi,
    add_daylight_mask,
    add_solar_position,
)

ARRAYS = {
    "array11": ("array11_polySi_hourly.parquet", 5.0, GAMMA_PDC_SILICON),
    "array12": ("array12_monoSi_hourly.parquet", 5.1, GAMMA_PDC_SILICON),
    "array17": ("array17_HIT_hourly.parquet", 6.3, GAMMA_PDC_HIT),
}


def load_and_prepare(array_key, processed_dir):
    """Load one array's processed parquet and add every column
    build_features/build_sequences and SmartPersistence require EXCEPT
    clear-sky power (p_cs, k_p), which depends on train-only fitted
    parameters and is added per-split by add_clearsky_power_per_split.

    Not restricted to the split years here: the processed parquet covers
    2009-2015 while TRAIN_YEARS starts at 2011, and split_chronological
    tolerates (and logs) rows outside the split window on its own - see
    src/data/splits.py.
    """
    import pandas as pd

    filename, nameplate_kw, gamma_pdc = ARRAYS[array_key]
    df = pd.read_parquet(processed_dir / filename)
    df = add_solar_position(df)
    df = add_clearsky(df)
    df = add_daylight_mask(df)
    df = add_clearsky_index_ghi(df)
    return df, nameplate_kw, gamma_pdc


def add_clearsky_power_per_split(train, val, test, nameplate_kw, gamma_pdc):
    """Fit the temperature climatology and gain on TRAIN ONLY, then apply
    them to produce p_cs/k_p on train and val. test is left untouched -
    callers must not read the test split at all in a dev script.
    """
    temp_clim = fit_temperature_climatology(train)

    p_cs_raw_train = model_clearsky_power(train.index, nameplate_kw, gamma_pdc, temp_clim)
    gain, n_gain_hours, gain_iqr = fit_gain(train, p_cs_raw_train)
    train = add_clearsky_power(train, p_cs_raw_train, gain, nameplate_kw)

    p_cs_raw_val = model_clearsky_power(val.index, nameplate_kw, gamma_pdc, temp_clim)
    val = add_clearsky_power(val, p_cs_raw_val, gain, nameplate_kw)

    return train, val, {"gain": gain, "gain_n_hours": n_gain_hours, "gain_iqr": gain_iqr}
