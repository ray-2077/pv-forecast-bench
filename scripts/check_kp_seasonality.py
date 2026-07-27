"""Diagnose whether the monthly variation in k_p (clear-sky power index) is
real weather (cloud/turbidity seasonality) or a seasonal bias baked into the
clear-sky power model itself - e.g. the temperature climatology or cell
temperature model systematically over/under-correcting power in certain
months.

TRAINING YEARS ONLY (2012, 2013). 2014 (validation) and 2015 (test) are
never loaded here - same reasoning as scripts/diagnose_clearsky_bias.py:
this is a diagnosis that could influence how the clear-sky power model gets
calibrated, so it must be trained-eyes-only.

Uses the same clear-hour definition as fit_gain in src/data/clearsky_power.py:
solar_elevation > 20 AND k_ghi in the top decile of training daylight hours'
k_ghi. If k_p is flat by month over clear hours but varies by month over all
daylight hours, the variation is real cloud/weather signal, not a modelling
artifact. If k_p still drifts by month even restricted to clear hours, and
that drift tracks modelled cell temperature, the gamma_pdc temperature
correction (or the temperature climatology feeding it) is the suspect.

No plotting, no files written, no fixes.

Usage:
    python scripts/check_kp_seasonality.py
"""

import sys
from pathlib import Path

import pandas as pd
from pvlib import irradiance, temperature

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clearsky import (
    add_clearsky,
    add_clearsky_index_ghi,
    add_daylight_mask,
    add_solar_position,
    get_location,
)
from src.data.clearsky_power import (
    ALBEDO,
    FIXED_WIND_SPEED,
    GAMMA_PDC_CDTE,
    GAMMA_PDC_SILICON,
    SURFACE_AZIMUTH,
    SURFACE_TILT,
    TEMP_MODEL_PARAMS,
    TRAIN_YEARS,
    add_clearsky_power,
    fit_gain,
    fit_temperature_climatology,
    model_clearsky_power,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# name, parquet filename, nameplate kW, gamma_pdc - same as validate_clearsky_power.py
ARRAYS = [
    ("array07_CdTe", "array07_CdTe_hourly.parquet", 7.0, GAMMA_PDC_CDTE),
    ("array11_polySi", "array11_polySi_hourly.parquet", 5.0, GAMMA_PDC_SILICON),
    ("array12_monoSi", "array12_monoSi_hourly.parquet", 5.1, GAMMA_PDC_SILICON),
]


def _lookup_climatology(temp_clim, times):
    """Map each timestamp to temp_clim[month, hour]. Local copy of the
    private helper in src/data/clearsky_power.py - duplicated rather than
    imported since it is not part of that module's public interface.
    """
    lookup = temp_clim.stack()
    keys = pd.MultiIndex.from_arrays([times.month, times.hour])
    return lookup.reindex(keys).to_numpy()


def model_clearsky_cell_temp(index, temp_clim):
    """Modelled cell temperature (deg C), hourly mean. Same physical chain
    as model_clearsky_power (5-min solar position -> clear-sky irradiance ->
    Hay-Davies POA -> SAPM cell temperature -> hourly mean, closed='left',
    label='left'), stopped one step short of PVWatts DC power. Duplicated
    here because model_clearsky_power does not expose the intermediate
    cell temperature.
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

    hourly = cell_temp.resample("1h", closed="left", label="left").mean()
    hourly = hourly.reindex(index)
    hourly.name = "cell_temp"
    return hourly


def clear_hour_mask(df):
    """Same definition as fit_gain: solar_elevation > 20 AND k_ghi in the
    top decile of training daylight hours' k_ghi.
    """
    daylight = df.loc[df["is_daylight"]]
    k_ghi_p90 = daylight["k_ghi"].quantile(0.9)
    return (df["solar_elevation"] > 20) & (df["k_ghi"] >= k_ghi_p90)


def main() -> None:
    for name, filename, nameplate_kw, gamma_pdc in ARRAYS:
        print(f"\n{'=' * 60}")
        print(f"{name} (nameplate {nameplate_kw} kW, gamma_pdc {gamma_pdc})")
        print("=" * 60)

        df = pd.read_parquet(PROCESSED_DIR / filename)
        # Training years only - 2014/2015 are never read past this line.
        train_df = df[df.index.year.isin(TRAIN_YEARS)]

        train_df = add_solar_position(train_df)
        train_df = add_clearsky(train_df)
        train_df = add_daylight_mask(train_df)
        train_df = add_clearsky_index_ghi(train_df)

        temp_clim = fit_temperature_climatology(train_df)
        p_cs_raw_train = model_clearsky_power(
            train_df.index, nameplate_kw, gamma_pdc, temp_clim
        )
        gain, n_hours, iqr = fit_gain(train_df, p_cs_raw_train)
        train_df = add_clearsky_power(train_df, p_cs_raw_train, gain, nameplate_kw)

        train_df["cell_temp"] = model_clearsky_cell_temp(train_df.index, temp_clim)
        train_df["temp_air_clim"] = _lookup_climatology(temp_clim, train_df.index)

        clear = train_df.loc[clear_hour_mask(train_df)]
        daylight = train_df.loc[train_df["is_daylight"]]

        print(
            f"\nGain used: {gain:.4f} (fit_gain, {n_hours} clear training "
            f"hours, IQR {iqr:.4f})"
        )

        # 1. mean k_p by month, clear hours only, with count
        print("\n1. Mean k_p by month, CLEAR hours only:")
        clear_by_month = clear.groupby(clear.index.month)["k_p"].agg(
            ["mean", "count"]
        )
        clear_by_month.index.name = "month"
        print(clear_by_month.round(3).to_string())

        # 2. mean k_p by month, all daylight hours, for comparison
        print("\n2. Mean k_p by month, ALL daylight hours (comparison):")
        daylight_by_month = daylight.groupby(daylight.index.month)["k_p"].agg(
            ["mean", "count"]
        )
        daylight_by_month.index.name = "month"
        print(daylight_by_month.round(3).to_string())

        # 3. climatological ambient temp and modelled cell temp by month,
        # clear hours only
        print(
            "\n3. Mean climatological ambient temp and modelled cell temp "
            "by month, CLEAR hours:"
        )
        temp_by_month = clear.groupby(clear.index.month)[
            ["temp_air_clim", "cell_temp"]
        ].mean()
        temp_by_month.index.name = "month"
        print(temp_by_month.round(2).to_string())

        # 4. correlation between monthly mean clear-hour k_p and monthly
        # mean modelled cell temperature
        corr = clear_by_month["mean"].corr(temp_by_month["cell_temp"])
        print(
            "\n4. Correlation(monthly mean clear-hour k_p, monthly mean "
            f"modelled cell temp): {corr:.4f}"
        )


if __name__ == "__main__":
    main()
