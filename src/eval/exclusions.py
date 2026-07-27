"""Evaluation-time exclusions for documented equipment outages.

These exclusions come from DKASC's published maintenance notes - external
metadata about a known, dated outage - NOT from anything about how a model
scored. Never add an entry here because a model performed badly during
some window; only add one when there is a documented, external reason the
array's readings during that window are known-invalid. Any exclusion
applied to a reported result must be declared in the paper.

This is deliberately separate from src/eval/metrics.py: metrics.py is a
pure function library over already-selected (y_true, y_pred, y_ref) rows.
Which rows are eligible to be evaluated at all is a data-selection
decision, made once per array and applied identically to every model
before any metric is computed - keeping it here means every model sees
the same evaluation set by construction, rather than each caller having to
remember to apply the same filter.
"""

import pandas as pd

TZ = "Australia/Darwin"

# (array, start_date, end_date) -> reason. start_date/end_date are
# inclusive calendar dates in Australia/Darwin local time.
KNOWN_OUTAGES = {
    ("array17", "2015-06-05", "2015-06-09"): (
        "DKASC documented outage: sites 4, 5, 17, 20, 22, 34, 35 switched "
        "off, discovered 9 June 2015"
    ),
}


def exclusion_mask(array, index):
    """Boolean Series aligned with `index`, True for timestamps that must
    be EXCLUDED from evaluation for `array` because of a documented outage
    in KNOWN_OUTAGES.

    Covers the full calendar days [start, end] inclusive, in
    Australia/Darwin local time, regardless of the daylight filter - a
    night hour inside an outage window is still an outage hour. `index`
    must be tz-aware in Australia/Darwin, as produced by
    src.data.loader.load_array.
    """
    mask = pd.Series(False, index=index)
    for (outage_array, start, end), _reason in KNOWN_OUTAGES.items():
        if outage_array != array:
            continue
        start_ts = pd.Timestamp(start, tz=TZ)
        end_ts = pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)
        mask |= (index >= start_ts) & (index < end_ts)
    return mask
