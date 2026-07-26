"""Solar position, clear-sky irradiance, and daylight masking for DKASC
Alice Springs. This is step 1 of 2 - clear-sky POWER modelling is a
separate module and does not belong here.
"""

import pandas as pd
from pvlib.location import Location

LATITUDE = -23.767
LONGITUDE = 133.867
ALTITUDE_M = 558
TZ = "Australia/Darwin"


def get_location():
    """Return the pvlib Location for the DKASC Alice Springs site."""
    return Location(LATITUDE, LONGITUDE, tz=TZ, altitude=ALTITUDE_M)


def _solar_position_at_midpoint(df, location):
    """Solar position evaluated at hour midpoints.

    Processed rows are hour-beginning means (the 12:00 row is the mean over
    12:00-12:55), so solar position must be evaluated at 12:30, not at the
    12:00 label - using the label would bias every zenith/azimuth/elevation
    by up to 30 minutes. Returned frame is indexed by the midpoint
    timestamps, not by df's hour-beginning index.
    """
    midpoints = df.index + pd.Timedelta(minutes=30)
    return location.get_solarposition(midpoints)


def add_solar_position(df):
    """Add apparent solar_zenith, solar_azimuth, solar_elevation, evaluated
    at hour midpoints. Index is unchanged (hour-beginning labels).

    Deliberately still a single midpoint value, unlike add_clearsky below:
    these are model FEATURES describing where the sun is during the hour,
    not a quantity being averaged against an hourly measurement. A midpoint
    snapshot is the right feature; it just isn't the right way to build
    ghi_cs, which has to be compared against an hourly-mean measurement.
    """
    location = get_location()
    sp = _solar_position_at_midpoint(df, location)

    df = df.copy()
    df["solar_zenith"] = sp["apparent_zenith"].to_numpy()
    df["solar_azimuth"] = sp["azimuth"].to_numpy()
    df["solar_elevation"] = sp["apparent_elevation"].to_numpy()
    return df


def add_clearsky(df):
    """Add clear-sky ghi_cs, dni_cs, dhi_cs using the Ineichen model with
    the Linke turbidity climatology.

    Measured irradiance is an hourly MEAN of twelve 5-minute samples
    (resample_hourly in loader.py). A clear-sky value evaluated once at
    the hour midpoint is instantaneous, not a mean, and near sunrise/
    sunset irradiance changes steeply within the hour - so the two are not
    comparable. This was visible in scripts/diagnose_clearsky_bias.py: with
    a midpoint ghi_cs, mean k_ghi was 0.92 at hour 7 (sun still rising
    through the hour) and 1.50 at hour 18 (sun setting through the hour),
    while the sun's actual position was already confirmed correct (peak
    measured GHI and peak ghi_cs matched to within a minute). So instead,
    clear-sky is computed at 5-minute resolution and aggregated exactly
    the way the measurements were: same resample frequency, closed side,
    and label convention as resample_hourly.
    """
    location = get_location()

    five_min_times = pd.date_range(
        start=df.index.min(),
        end=df.index.max() + pd.Timedelta(minutes=55),
        freq="5min",
    )
    sp = location.get_solarposition(five_min_times)
    # pvlib returns 0 below the horizon, so hours where sunrise/sunset
    # falls inside the hour average correctly with no special handling.
    cs_5min = location.get_clearsky(five_min_times, model="ineichen", solar_position=sp)

    cs_hourly = cs_5min.resample("1h", closed="left", label="left").mean()
    cs_hourly = cs_hourly.reindex(df.index)

    df = df.copy()
    df["ghi_cs"] = cs_hourly["ghi"].to_numpy()
    df["dni_cs"] = cs_hourly["dni"].to_numpy()
    df["dhi_cs"] = cs_hourly["dhi"].to_numpy()
    return df


def add_daylight_mask(df, min_elevation=10.0):
    """Add boolean is_daylight, True where solar_elevation exceeds
    min_elevation. Requires add_solar_position to have been called first.

    10 degrees is chosen to exclude hours where clear-sky irradiance is
    negligible (near sunrise/sunset), while staying inside DKASC's
    guaranteed shading-free window of 0830-1630 local time - below that
    elevation, shading from surrounding terrain/structures is possible and
    not accounted for here.
    """
    if "solar_elevation" not in df.columns:
        raise KeyError(
            "add_daylight_mask requires solar_elevation; call "
            "add_solar_position first"
        )

    df = df.copy()
    df["is_daylight"] = df["solar_elevation"] > min_elevation
    return df


def add_clearsky_index_ghi(df):
    """Add k_ghi = Global_Horizontal_Radiation / ghi_cs. Requires
    add_clearsky to have been called first.

    k_ghi is set to NaN where ghi_cs < 20 W/m2 to avoid dividing by
    near-zero. Not clipped - cloud-enhancement values above 1 are kept
    visible in the raw distribution.
    """
    if "ghi_cs" not in df.columns:
        raise KeyError(
            "add_clearsky_index_ghi requires ghi_cs; call add_clearsky first"
        )

    df = df.copy()
    k_ghi = df["Global_Horizontal_Radiation"] / df["ghi_cs"]
    df["k_ghi"] = k_ghi.where(df["ghi_cs"] >= 20)
    return df
