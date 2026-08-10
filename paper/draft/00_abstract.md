# ABSTRACT

Hybrid deep learning architectures dominate recent work on short-term
photovoltaic power forecasting, and reported accuracy improvements are
substantial. We coded 27 such papers on eight dimensions of evaluation
protocol: 26 report no skill score against any reference forecast, 22 do
not state whether their train-test split preserves temporal order, and
all 27 report point estimates with no measure of run-to-run variance.
This paper asks how much of reported accuracy in this literature is
attributable to evaluation choices rather than to model capacity.

We fix a leakage-controlled protocol and vary individual evaluation
choices while holding the model, data and training procedure constant,
across five architectures, three module technologies, three forecast
horizons, two feature regimes and five random seeds, on 900 recorded
runs from an open dataset. The choice of reference forecast accounts for
70 percent of reported skill at a six-hour horizon and inverts the
apparent relationship between skill and horizon. Including night hours
deflates normalised error by approximately 34 percent, by a factor that
is closed-form in the daylight fraction and independent of the model.
Fitting a residual correction stage on the split it is evaluated on
turns a component that costs 0.034 in skill into one that appears to
contribute 0.334.

Architectural differences are one to two orders of magnitude smaller.
Gradient boosting and a recurrent network are statistically
indistinguishable at one- and three-hour horizons under paired
Diebold-Mariano testing; a convolutional front end shows no detectable
benefit; and a gradient-boosted residual correction stage reduces skill
in all eighteen configurations tested. The largest difference between
any two architectures is 0.024 in skill score, against a gap of 0.51 to
0.72 between forecasting with available observations and with perfect
knowledge of future weather. All findings hold in direction on a
held-out year evaluated once. The harness, all run records, and the
coded survey are released.

[VERIFY: IEEE abstracts are typically 150-250 words. This is
approximately 290. Trim at assembly if the venue requires it, but do not
trim the survey counts or the three protocol effects - those are the
paper.]

# INDEX TERMS

Photovoltaic power forecasting, forecast verification, skill score,
data leakage, reproducibility, benchmark evaluation, deep learning.
