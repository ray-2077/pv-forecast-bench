"""Build the model input matrix, in two switchable regimes: 'lagged' and
'oracle'. This is the module where leakage would happen, so the alignment
convention is stated here first and enforced by assertions below.

ALIGNMENT CONVENTION:
The feature matrix is INDEXED BY TARGET TIME t. A row at target time t, for
horizon h, is the input to a forecast ISSUED at time t-h. Therefore any
observed quantity in that row must come from time t-h or EARLIER. Features
describing time t itself are permitted ONLY if they are deterministic and
computable in advance (solar geometry, clear-sky power, calendar
encodings) - never a measurement.

THREE CATEGORIES OF FEATURE:

A. Deterministic, known for any future time - allowed unshifted, at t:
   p_cs, ghi_cs, solar_zenith, solar_azimuth, solar_elevation, hour-of-day
   sin/cos, day-of-year sin/cos. These depend only on the calendar and the
   site, never on a measurement.

B. Lagged observations, taken at t-h and earlier. For Active_Power, k_p,
   k_ghi: shifts h, h+1, h+2 (suffixes _issue, _issue_m1, _issue_m2 -
   stable meanings across horizons, so a feature-importance plot cannot be
   misread as "lag 1" when the actual shift is horizon-dependent), plus a
   FIXED shift of 24 (suffix _daily) for the same-time-yesterday reading -
   fixed rather than h+23 so the daily-cycle feature stays aligned to
   hour-of-day at every horizon. A shift of 24 is still at or before the
   issue time t-h for any h <= 24 (asserted).

   k_p and k_ghi are undefined at night by construction (see
   clearsky_power.py), so an exact-shift lag landing on a night issue time
   would otherwise be NaN and the whole row would be dropped - starving
   the lagged regime of daylight targets that smart persistence can still
   forecast (it forward-fills k_p up to 24h). To match that baseline, the
   k_p and k_ghi source series are forward-filled (limit 24h) BEFORE any
   of the shifts above are applied. Forward-fill only ever propagates a
   PAST value forward in time, so this cannot introduce a leak - see
   _lag_source. Active_Power is not forward-filled: it is measured (not
   derived) and is not NaN at night.

   Two more B features track that fill: k_p_hours_stale (hours since the
   last genuinely observed k_p, as of the issue time; 0 if observed
   exactly at the issue time) and k_p_is_stale (k_p_hours_stale > 0),
   mirroring SmartPersistence.fallback_fraction in src/models/persistence.py
   so a model can discount stale information the same way the baseline's
   accuracy already implicitly does.

   Rolling mean/std of Active_Power over 3h and 24h (WALLCLOCK_ROLLING_SPECS):
   computed on the RAW series shifted by h first, so the window ENDS at
   the issue time. Active_Power is 0 (not NaN) at night, so a wall-clock
   window is well defined for it at any issue time.

   Rolling mean/std of k_p (3 and 24) and rolling std of k_ghi (3)
   (OBS_ROLLING_SPECS): NOT a wall-clock window - a window of the last N
   VALID (non-NaN) observations, however far back in time they fall. k_p
   and k_ghi are undefined at night, so a wall-clock window ending at a
   pre-dawn issue time can contain zero valid samples and stay NaN even
   with min_periods=1, dropping the row. Forward-filling the series and
   THEN taking a wall-clock rolling std is deliberately not done either:
   the std of repeated identical (filled) numbers is exactly 0, which
   would tell a model "perfectly stable sky" when the truth is "no
   observation since yesterday evening" - a fabricated value, worse than
   a dropped row. See _last_n_valid_obs_stat. Named k_p_lastNobs_<stat> /
   k_ghi_lastNobs_<stat> (N = 3 or 24) so the feature-importance figure
   cannot mistake these for the wall-clock Active_Power ones.

   Plus weather (temperature, humidity, wind, GHI, DHI) shifted by h only
   - the issue-time reading, nothing more recent.

C. Oracle only - measured weather AT TARGET TIME t, no shift. Named with an
   explicit oracle_ prefix so they can never be mistaken for a legitimate
   feature in a feature-importance plot.

build_features(df, horizon, 'lagged') returns A + B.
build_features(df, horizon, 'oracle') returns A + B + C.
The two regimes must never be mixed - a single call returns one or the
other, never a blend.

Requires df to already carry p_cs, ghi_cs, solar_zenith, solar_azimuth,
solar_elevation, k_p, k_ghi - i.e. src.data.clearsky.add_solar_position,
add_clearsky, add_clearsky_index_ghi and src.data.clearsky_power.
add_clearsky_power have already run on it upstream, with any fitted
parameters (temperature climatology, gain) coming from TRAINING data only.

No scalers, no imputation, no model code, no plotting.
"""

import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "Active_Power",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Wind_Speed",
    "p_cs",
    "ghi_cs",
    "solar_zenith",
    "solar_azimuth",
    "solar_elevation",
    "k_p",
    "k_ghi",
]

DETERMINISTIC_NAMES = [
    "p_cs",
    "ghi_cs",
    "solar_zenith",
    "solar_azimuth",
    "solar_elevation",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

# (column suffix, offset added to horizon h) for the three issue-relative
# lags. Suffix meaning is stable across horizons - see module docstring.
LAG_ISSUE_SUFFIXES = [("_issue", 0), ("_issue_m1", 1), ("_issue_m2", 2)]

LAG_DAILY_SUFFIX = "_daily"
LAG_DAILY_SHIFT = 24  # fixed, not horizon-relative - see module docstring

LAG_BASE_COLS = ["Active_Power", "k_p", "k_ghi"]

# k_p and k_ghi are forward-filled before shifting; Active_Power is not
# (it is measured, not NaN at night). Limit matches
# SmartPersistence.FFILL_LIMIT_HOURS in src/models/persistence.py.
FFILL_LAG_BASE_COLS = ("k_p", "k_ghi")
LAG_FFILL_LIMIT = 24

K_P_HOURS_STALE_COL = "k_p_hours_stale"
K_P_IS_STALE_COL = "k_p_is_stale"

# The daily lag's shift (LAG_DAILY_SHIFT, fixed) must stay at or before
# the issue time t-h, i.e. LAG_DAILY_SHIFT >= h, for every horizon this
# module is used with.
MAX_HORIZON = LAG_DAILY_SHIFT

# (base column, window in WALL-CLOCK HOURS, stat) - computed on the base
# column shifted by h first. Active_Power ONLY: it is 0 (not NaN) at
# night, so an hour-window is well defined for it at any issue time. See
# OBS_ROLLING_SPECS below for k_p/k_ghi, which cannot use a wall-clock
# window.
WALLCLOCK_ROLLING_SPECS = [
    ("Active_Power", 3, "mean"),
    ("Active_Power", 3, "std"),
    ("Active_Power", 24, "mean"),
    ("Active_Power", 24, "std"),
]

# (base column, window in VALID OBSERVATION COUNT, stat) for k_p and
# k_ghi, which are NaN at night. A wall-clock window ending at a pre-dawn
# issue time can contain zero valid samples and stay NaN even with
# min_periods=1 - see check_feature_coverage.py's h=6 evidence (hours
# 8-13 dropped on essentially every day). The window here instead counts
# the last N OBSERVED (non-NaN) readings, however far back in wall-clock
# time they fall - see _last_n_valid_obs_stat.
OBS_ROLLING_SPECS = [
    ("k_p", 3, "mean"),
    ("k_p", 3, "std"),
    ("k_p", 24, "mean"),
    ("k_p", 24, "std"),
    ("k_ghi", 3, "std"),
]

WEATHER_COLS = [
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Wind_Speed",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
]

ORACLE_PREFIX = "oracle_"

REGIMES = ("lagged", "oracle")

DAYS_PER_YEAR = 365.25


def _check_regime(regime):
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")


def _check_horizon(horizon):
    assert horizon <= MAX_HORIZON, (
        f"horizon must be <= {MAX_HORIZON}: the daily lag uses a fixed "
        f"shift of {LAG_DAILY_SHIFT}, which must be at or before the "
        f"issue time t-h; got horizon={horizon}"
    )


def _check_required_columns(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"build_features requires columns {missing}; run "
            "src.data.clearsky.add_solar_position, add_clearsky, "
            "add_clearsky_index_ghi, and src.data.clearsky_power."
            "add_clearsky_power on df first"
        )


def _rolling_column_name(base, window, stat):
    return f"{base}_roll{window}_{stat}"


def _obs_rolling_column_name(base, window, stat):
    return f"{base}_last{window}obs_{stat}"


def _weather_issue_column_name(col):
    return f"{col}_issue"


def _oracle_column_name(col):
    return f"{ORACLE_PREFIX}{col}"


def _deterministic_feature_names():
    return list(DETERMINISTIC_NAMES)


def _lagged_feature_names():
    names = []
    for base in LAG_BASE_COLS:
        for suffix, _offset in LAG_ISSUE_SUFFIXES:
            names.append(f"{base}{suffix}")
        names.append(f"{base}{LAG_DAILY_SUFFIX}")
    names.append(K_P_HOURS_STALE_COL)
    names.append(K_P_IS_STALE_COL)
    for base, window, stat in WALLCLOCK_ROLLING_SPECS:
        names.append(_rolling_column_name(base, window, stat))
    for base, window, stat in OBS_ROLLING_SPECS:
        names.append(_obs_rolling_column_name(base, window, stat))
    for col in WEATHER_COLS:
        names.append(_weather_issue_column_name(col))
    return names


def _oracle_feature_names():
    return [_oracle_column_name(col) for col in WEATHER_COLS]


def feature_names(regime, horizon):
    """Return the column list build_features(df, horizon, regime) would
    produce, without building the matrix.

    horizon does not change which columns exist (suffixes are relative to
    the issue time, not absolute shift amounts, except _daily which is a
    fixed shift), but must still satisfy horizon <= MAX_HORIZON, checked
    here for symmetry with build_features.
    """
    _check_regime(regime)
    _check_horizon(horizon)
    names = _deterministic_feature_names() + _lagged_feature_names()
    if regime == "oracle":
        names += _oracle_feature_names()
    return names


def _deterministic_features(df):
    hour = df.index.hour.to_numpy(dtype=float)
    doy = df.index.dayofyear.to_numpy(dtype=float)

    out = pd.DataFrame(index=df.index)
    out["p_cs"] = df["p_cs"].to_numpy()
    out["ghi_cs"] = df["ghi_cs"].to_numpy()
    out["solar_zenith"] = df["solar_zenith"].to_numpy()
    out["solar_azimuth"] = df["solar_azimuth"].to_numpy()
    out["solar_elevation"] = df["solar_elevation"].to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / DAYS_PER_YEAR)
    out["doy_cos"] = np.cos(2 * np.pi * doy / DAYS_PER_YEAR)
    return out


def _lag_source(df, base):
    """The series lag columns for `base` are shifted from.

    k_p and k_ghi are forward-filled first (limit LAG_FFILL_LIMIT hours) -
    ffill only ever copies a PAST value forward in time, so a value at
    position i is still derived only from positions <= i. This cannot
    leak information from after the timestamp it fills. Active_Power is
    returned unchanged: it is measured, not NaN at night, so there is
    nothing to fill.
    """
    if base in FFILL_LAG_BASE_COLS:
        return df[base].ffill(limit=LAG_FFILL_LIMIT)
    return df[base]


def _hours_since_valid(series, limit):
    """Hours since the last non-NaN value at or before each position: 0 at
    a valid position itself, NaN if no valid value exists within `limit`
    hours - which is exactly where series.ffill(limit=limit) would still
    be NaN.

    Assumes a complete, regularly-spaced hourly index (true for this
    project's processed data - see src/data/loader.py resample_hourly),
    so that integer position differences equal elapsed hours.
    """
    valid = series.notna().to_numpy()
    position = np.arange(len(series))
    last_valid_position = np.maximum.accumulate(np.where(valid, position, -1))

    hours_since = (position - last_valid_position).astype(float)
    hours_since[last_valid_position < 0] = np.nan
    hours_since[hours_since > limit] = np.nan

    return pd.Series(hours_since, index=series.index)


def _is_stale(hours_stale):
    # Float 0.0/1.0, not bool, so every feature column stays numeric.
    # NaN (unknown - could not be filled within the limit either) is kept
    # as NaN rather than silently collapsing to "not stale".
    is_stale = np.where(
        hours_stale.isna(), np.nan, (hours_stale.to_numpy() > 0).astype(float)
    )
    return pd.Series(is_stale, index=hours_stale.index)


def _lagged_features(df, horizon):
    _check_horizon(horizon)
    out = pd.DataFrame(index=df.index)

    for base in LAG_BASE_COLS:
        source = _lag_source(df, base)
        for suffix, offset in LAG_ISSUE_SUFFIXES:
            out[f"{base}{suffix}"] = source.shift(horizon + offset)
        out[f"{base}{LAG_DAILY_SUFFIX}"] = source.shift(LAG_DAILY_SHIFT)

    hours_stale = _hours_since_valid(df["k_p"], LAG_FFILL_LIMIT).shift(horizon)
    out[K_P_HOURS_STALE_COL] = hours_stale
    out[K_P_IS_STALE_COL] = _is_stale(hours_stale)

    for base, window, stat in WALLCLOCK_ROLLING_SPECS:
        shifted = df[base].shift(horizon)
        rolled = _rolling_stat(shifted, window, stat)
        out[_rolling_column_name(base, window, stat)] = rolled

    for base, window, stat in OBS_ROLLING_SPECS:
        stat_series = _last_n_valid_obs_stat(df[base], window, stat)
        out[_obs_rolling_column_name(base, window, stat)] = stat_series.shift(horizon)

    for col in WEATHER_COLS:
        out[_weather_issue_column_name(col)] = df[col].shift(horizon)

    return out


def _rolling_stat(series, window, stat):
    # min_periods=1, not the pandas default of window. Used only for
    # Active_Power (WALLCLOCK_ROLLING_SPECS), which is 0 (not NaN) at
    # night, so this just protects the first `window` rows of history.
    # See _last_n_valid_obs_stat for k_p/k_ghi, which cannot use a
    # wall-clock window at all.
    rolling = series.rolling(window, min_periods=1)
    return rolling.mean() if stat == "mean" else rolling.std()


def _last_n_valid_obs_stat(series, window, stat):
    """Statistic over the last `window` VALID (non-NaN) observations of
    `series`, not the last `window` wall-clock hours. For k_p/k_ghi only.

    k_p and k_ghi are undefined at night, so a wall-clock rolling window
    ending at a pre-dawn issue time can contain zero valid samples and
    stay NaN even with min_periods=1 - this was the h=6 midday-target
    drop measured in scripts/check_feature_coverage.py. The tempting fix,
    forward-filling the series and THEN taking a wall-clock rolling std,
    is deliberately NOT done: the std of a run of repeated identical
    (forward-filled) numbers is exactly 0, which would tell a model
    "perfectly stable sky" when the truth is "no observation since
    yesterday evening". That is a fabricated feature value, worse than a
    dropped row. Do not "simplify" this back into a wall-clock rolling
    stat on a forward-filled series.

    Instead: drop the NaNs (keep only genuinely observed hours), compute
    the rolling stat on that compressed, valid-only series, then forward-
    fill the completed value back onto the full hourly index so every
    timestamp carries the most recently completed window's value. This
    does not leak: forward-fill only ever carries a PAST value forward in
    time, and the caller applies the horizon shift AFTER this function
    returns, so the issue-time cutoff still holds.
    """
    valid = series.dropna()
    min_periods = 1 if stat == "mean" else 2
    rolling = valid.rolling(window, min_periods=min_periods)
    stat_on_valid = rolling.mean() if stat == "mean" else rolling.std()
    return stat_on_valid.reindex(series.index).ffill()


def _oracle_features(df):
    out = pd.DataFrame(index=df.index)
    for col in WEATHER_COLS:
        out[_oracle_column_name(col)] = df[col].to_numpy()
    return out


def build_features(df, horizon, regime):
    """Build (X, y) for one horizon and regime.

    regime is 'lagged' (A + B) or 'oracle' (A + B + C). Raises ValueError on
    any other value, and fails an assertion if horizon > MAX_HORIZON. y is
    Active_Power at target time t. Rows where y is NaN or any feature is
    undefined (not enough history yet, or a gap wider than the forward-fill
    limit) are dropped. Does not scale, normalise, or impute.
    """
    _check_regime(regime)
    _check_required_columns(df)

    parts = [_deterministic_features(df), _lagged_features(df, horizon)]
    if regime == "oracle":
        parts.append(_oracle_features(df))

    X = pd.concat(parts, axis=1)
    X = X[feature_names(regime, horizon)]

    y = df["Active_Power"].copy()
    y.name = "y"

    combined = X.copy()
    combined["__y__"] = y
    combined = combined.dropna()

    y_clean = combined.pop("__y__")
    y_clean.name = "y"
    X_clean = combined

    return X_clean, y_clean


def _assert_series_equal(rebuilt, actual, label, atol=1e-9):
    a = rebuilt.to_numpy(dtype=float)
    b = actual.to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, atol=atol, rtol=0.0)
    ok = both_nan | close
    if not bool(np.all(ok)):
        n_bad = int((~ok).sum())
        raise AssertionError(
            f"assert_no_leakage: column {label!r} does not match an "
            f"independently rebuilt version at {n_bad} row(s) - possible "
            "leakage or an incorrect shift"
        )


def assert_no_leakage(df, X, y, horizon, regime):
    """Real leakage check, not a comment.

    For BOTH regimes: every column in the B category (lag, staleness,
    rolling, weather-at-issue) is rebuilt directly from df - independently
    of _lag_source/_hours_since_valid, by inlining the same ffill/shift
    logic here, so a bug in the production helpers would not be silently
    repeated in the check - using shift = horizon + offset (offset >= 0,
    for _issue/_issue_m1/_issue_m2) or the fixed LAG_DAILY_SHIFT (for
    _daily, asserted >= horizon), so the rebuilt shift is structurally
    always >= horizon. k_p and k_ghi are forward-filled BEFORE the shift,
    exactly as required by the module docstring: ffill only ever copies a
    PAST value forward, so the rebuilt columns still derive only from data
    at or before the issue time. The check asserts X's actual column
    equals that independent rebuild exactly.

    For the 'oracle' regime additionally: every oracle_-prefixed column is
    rebuilt as the UNSHIFTED value of its base column at target time t,
    and must equal X's column exactly. Combined with the B-category check
    above (whose columns are never unshifted, since horizon >= 1), this
    establishes that oracle_ columns are the only unshifted measured
    columns in X.

    Also asserts X's column set is exactly feature_names(regime, horizon)
    - no undeclared column can be hiding a leak.
    """
    _check_regime(regime)
    _check_horizon(horizon)

    expected_cols = set(feature_names(regime, horizon))
    actual_cols = set(X.columns)
    if actual_cols != expected_cols:
        raise AssertionError(
            f"assert_no_leakage: X columns {actual_cols} do not match "
            f"feature_names(regime, horizon) {expected_cols}"
        )

    idx = X.index

    for base in LAG_BASE_COLS:
        if base in FFILL_LAG_BASE_COLS:
            rebuilt_source = df[base].ffill(limit=LAG_FFILL_LIMIT)
        else:
            rebuilt_source = df[base]

        for suffix, offset in LAG_ISSUE_SUFFIXES:
            shift_amt = horizon + offset
            assert shift_amt >= horizon, "lag shift must be >= horizon"
            colname = f"{base}{suffix}"
            rebuilt = rebuilt_source.shift(shift_amt).reindex(idx)
            _assert_series_equal(rebuilt, X[colname], colname)

        assert LAG_DAILY_SHIFT >= horizon, "daily lag shift must be >= horizon"
        colname = f"{base}{LAG_DAILY_SUFFIX}"
        rebuilt = rebuilt_source.shift(LAG_DAILY_SHIFT).reindex(idx)
        _assert_series_equal(rebuilt, X[colname], colname)

    kp_valid = df["k_p"].notna().to_numpy()
    position = np.arange(len(df))
    last_valid_position = np.maximum.accumulate(np.where(kp_valid, position, -1))
    hours_since = (position - last_valid_position).astype(float)
    hours_since[last_valid_position < 0] = np.nan
    hours_since[hours_since > LAG_FFILL_LIMIT] = np.nan
    rebuilt_hours_stale = pd.Series(hours_since, index=df.index).shift(horizon).reindex(idx)
    _assert_series_equal(rebuilt_hours_stale, X[K_P_HOURS_STALE_COL], K_P_HOURS_STALE_COL)

    rebuilt_is_stale = pd.Series(
        np.where(
            rebuilt_hours_stale.isna(),
            np.nan,
            (rebuilt_hours_stale.to_numpy() > 0).astype(float),
        ),
        index=idx,
    )
    _assert_series_equal(rebuilt_is_stale, X[K_P_IS_STALE_COL], K_P_IS_STALE_COL)

    for base, window, stat in WALLCLOCK_ROLLING_SPECS:
        colname = _rolling_column_name(base, window, stat)
        shifted = df[base].shift(horizon)
        rebuilt = _rolling_stat(shifted, window, stat).reindex(idx)
        _assert_series_equal(rebuilt, X[colname], colname)

    # Independent rebuild of the last-N-valid-observations stats: dropna,
    # roll on the compressed series, ffill back onto the full index (never
    # bfill - forward-fill only ever carries a PAST value forward), then
    # shift by horizon. Inlined rather than calling
    # _last_n_valid_obs_stat, so a bug there would not be silently
    # repeated here.
    for base, window, stat in OBS_ROLLING_SPECS:
        colname = _obs_rolling_column_name(base, window, stat)
        valid = df[base].dropna()
        min_periods = 1 if stat == "mean" else 2
        rolling = valid.rolling(window, min_periods=min_periods)
        stat_on_valid = rolling.mean() if stat == "mean" else rolling.std()
        rebuilt = stat_on_valid.reindex(df.index).ffill().shift(horizon).reindex(idx)
        _assert_series_equal(rebuilt, X[colname], colname)

    for col in WEATHER_COLS:
        colname = _weather_issue_column_name(col)
        rebuilt = df[col].shift(horizon).reindex(idx)
        _assert_series_equal(rebuilt, X[colname], colname)

    if regime == "oracle":
        for col in WEATHER_COLS:
            colname = _oracle_column_name(col)
            rebuilt = df[col].reindex(idx)
            _assert_series_equal(rebuilt, X[colname], colname)

    y_expected = df["Active_Power"].reindex(idx)
    _assert_series_equal(y_expected, y.reindex(idx), "y")
