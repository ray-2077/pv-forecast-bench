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
