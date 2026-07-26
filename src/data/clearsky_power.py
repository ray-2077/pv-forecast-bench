"""Model clear-sky AC power per array, to compute the clear-sky index of
POWER (as opposed to clear-sky index of irradiance, which is clearsky.py).

This is the reference that smart persistence persists against, so it must
contain NO information unavailable at forecast time.

CRITICAL - no future weather: smart persistence predicts
P(t+h) = k(t) * P_cs(t+h). If P_cs used measured weather at t+h, the lagged
regime would leak future weather into a "lagged" forecast. So P_cs may
depend ONLY on solar geometry, clear-sky irradiance, and a temperature
CLIMATOLOGY fitted on training years - never on measured temperature, wind,
or humidity at the forecast time. Wind speed is fixed at 1.0 m/s for the
same reason: there is no climatology for it here, and a constant carries no
information about any particular hour.
"""

import pandas as pd
from pvlib import irradiance, pvsystem, temperature

from src.data.clearsky import get_location

SURFACE_TILT = 20.0
# pvlib convention: 0 = north. Fixed arrays face solar north (DKASC is in
# the southern hemisphere).
SURFACE_AZIMUTH = 0.0
ALBEDO = 0.2
FIXED_WIND_SPEED = 1.0  # m/s - see module docstring

TEMP_MODEL_PARAMS = temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
    "open_rack_glass_polymer"
]

# Datasheet-typical power temperature coefficients (%/C), NOT measured for
# these specific arrays.
GAMMA_PDC_SILICON = -0.0040
GAMMA_PDC_CDTE = -0.0025

TRAIN_YEARS = (2012, 2013)


def fit_temperature_climatology(df_train):
    """From TRAINING data only, return a month-by-hour table (index=month
    1-12, columns=hour 0-23) of mean Weather_Temperature_Celsius.

    Raises ValueError if df_train contains any year outside TRAIN_YEARS -
    this table must never see 2014 or 2015.
    """
    years = set(df_train.index.year.unique())
    if not years.issubset(TRAIN_YEARS):
        raise ValueError(
            f"fit_temperature_climatology got years {sorted(years)}, "
            f"expected only {TRAIN_YEARS}"
        )

    temp = df_train["Weather_Temperature_Celsius"]
    table = temp.groupby([df_train.index.month, df_train.index.hour]).mean()
    table = table.unstack()
    table.index.name = "month"
    table.columns.name = "hour"
    return table


def _lookup_climatology(temp_clim, times):
    """Map each timestamp in `times` to temp_clim[month, hour]. All 5-minute
    slots within one hour share the same (month, hour) key, so this never
    splits one hourly bin across two climatology cells.
    """
    lookup = temp_clim.stack()
    keys = pd.MultiIndex.from_arrays([times.month, times.hour])
    return lookup.reindex(keys).to_numpy()


def model_clearsky_power(index, nameplate_kw, gamma_pdc, temp_clim):
    """Physical clear-sky AC power chain, computed at 5-minute resolution
    then averaged to hourly with closed='left', label='left' - same
    convention as add_clearsky in clearsky.py, because the result must line
    up with hour-beginning-labelled measurements.

    Steps: solar position and clear-sky ghi/dni/dhi at 5-min -> POA
    irradiance via Hay-Davies transposition (albedo 0.2) -> cell temperature
    from POA and the climatological ambient temp for that month/hour, fixed
    wind speed 1.0 m/s, open-rack glass-polymer SAPM parameters -> DC power
    via PVWatts. Returns hourly mean power in kW, uncalibrated (no gain
    applied - see fit_gain).
    """
    location = get_location()

    five_min_times = pd.date_range(
        start=index.min(),
        end=index.max() + pd.Timedelta(minutes=55),
        freq="5min",
    )

    sp = location.get_solarposition(five_min_times)
    cs = location.get_clearsky(five_min_times, model="ineichen", solar_position=sp)
    dni_extra = irradiance.get_extra_radiation(five_min_times)

    poa = irradiance.get_total_irradiance(
        surface_tilt=SURFACE_TILT,
        surface_azimuth=SURFACE_AZIMUTH,
        solar_zenith=sp["apparent_zenith"],
        solar_azimuth=sp["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        dni_extra=dni_extra,
        model="haydavies",
        albedo=ALBEDO,
    )

    temp_air = _lookup_climatology(temp_clim, five_min_times)

    cell_temp = temperature.sapm_cell(
        poa_global=poa["poa_global"],
        temp_air=temp_air,
        wind_speed=FIXED_WIND_SPEED,
        a=TEMP_MODEL_PARAMS["a"],
        b=TEMP_MODEL_PARAMS["b"],
        deltaT=TEMP_MODEL_PARAMS["deltaT"],
    )

    dc_power = pvsystem.pvwatts_dc(
        effective_irradiance=poa["poa_global"],
        temp_cell=cell_temp,
        pdc0=nameplate_kw,
        gamma_pdc=gamma_pdc,
    )

    hourly = dc_power.resample("1h", closed="left", label="left").mean()
    hourly = hourly.reindex(index)
    hourly.name = "p_cs_raw"
    return hourly


def fit_gain(df_train, p_cs_raw_train):
    """Fit one scalar gain per array on TRAINING DATA ONLY, correcting for
    soiling, degradation, inverter efficiency, and true module area, none
    of which the physical model above accounts for.

    Requires df_train to already carry is_daylight, solar_elevation, and
    k_ghi (i.e. clearsky.py's add_solar_position, add_clearsky,
    add_daylight_mask, add_clearsky_index_ghi have already run) plus
    Active_Power.

    Clear training hours = solar_elevation > 20 AND k_ghi in the top decile
    of training daylight hours' k_ghi. gain = median(Active_Power /
    p_cs_raw) over those hours.

    Returns (gain, n_hours_used, iqr_of_ratio).
    """
    required = ["is_daylight", "solar_elevation", "k_ghi", "Active_Power"]
    missing = [c for c in required if c not in df_train.columns]
    if missing:
        raise KeyError(
            f"fit_gain requires columns {missing}; call add_solar_position, "
            "add_clearsky, add_daylight_mask, add_clearsky_index_ghi first"
        )

    p_cs_raw_train = p_cs_raw_train.reindex(df_train.index)

    daylight = df_train.loc[df_train["is_daylight"]]
    k_ghi_p90 = daylight["k_ghi"].quantile(0.9)

    clear_mask = (df_train["solar_elevation"] > 20) & (df_train["k_ghi"] >= k_ghi_p90)
    valid = (
        clear_mask
        & p_cs_raw_train.notna()
        & (p_cs_raw_train > 0)
        & df_train["Active_Power"].notna()
    )

    ratio = df_train.loc[valid, "Active_Power"] / p_cs_raw_train.loc[valid]

    gain = float(ratio.median())
    n_hours = int(valid.sum())
    iqr = float(ratio.quantile(0.75) - ratio.quantile(0.25))

    return gain, n_hours, iqr


def add_clearsky_power(df, p_cs_raw, gain, nameplate_kw):
    """Add p_cs = gain * p_cs_raw (kW) and k_p = Active_Power / p_cs.

    k_p is set to NaN where p_cs < 2 percent of nameplate_kw, to avoid
    dividing by near-zero. Not clipped otherwise.
    """
    df = df.copy()
    p_cs_raw = p_cs_raw.reindex(df.index)

    p_cs = gain * p_cs_raw
    k_p = df["Active_Power"] / p_cs
    k_p = k_p.where(p_cs >= 0.02 * nameplate_kw)

    df["p_cs"] = p_cs
    df["k_p"] = k_p
    return df
