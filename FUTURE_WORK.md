# Future Work

Ideas deliberately deferred to protect the paper deadline.

## Product / tool direction (raised 2026-07-28)
Turn the harness into a usable forecasting tool: dashboard, config-driven CLI,
pip-installable package, live inference on new DKASC data.
Deferred: no room before the paper deadline. Revisit after submission.
Cheap subset kept in scope: README documenting the protocol, YAML configs, and
a single command that reproduces every table and figure.

## Multi-location generalisation
Add Yulara as a genuinely separate site. Different schema, second pipeline.

## Low-elevation clear-sky bias
Twilight diffuse and pyranometer thermal offset inflate k_ghi below 20 deg
elevation. Candidate protocol knob for RQ2: does the daylight threshold change
reported accuracy?

## Smart persistence degrades at long horizons (found 2026-07-28)
fallback_fraction on 2014 daylight hours: 3.3 pct at h=1, 22.6 pct at h=3,
51 pct at h=6. At h=6 most forecasts are issued at night, so k_p is
forward-filled from the previous afternoon. Baseline is therefore weaker at
long horizons, which INFLATES every model skill score at h=6.
DECISION: keep forward-fill, record fallback_fraction in every results JSON,
and add daylight-issued-only as a protocol configuration for RQ2 Table 4.

## Night-inclusion inflation is analytic
nRMSE_all / nRMSE_daylight = sqrt(N_daylight / N_all) when night errors are
near zero. Predicted 0.659 for 2014, observed 0.66 across all arrays and
horizons. Report this closed form in the paper.
