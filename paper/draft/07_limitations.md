# VII. LIMITATIONS AND THREATS TO VALIDITY

The protocol effects reported here are large, and the temptation is to
read them as general. Several constraints bound that reading, and we
state them before the conclusions rather than after.

## A. Scope of the Data

All results come from three arrays at a single site. The arrays are
co-located and share one weather station, so their forecast errors are
correlated and consistency across them is consistency across module
technology under common meteorology, not geographic generalisation.
Every array-by-horizon table in this paper therefore contains fewer
independent observations than its cell count suggests, and no
significance test in this paper treats the nine cells as nine
independent trials.

The site is a hot desert climate with approximately 60 percent of
daylight hours classified as clear. Results at a site with a different
cloud regime may differ, particularly the sky-stratified analysis in
Section [VERIFY], where the partly cloudy class is the one in which
models add least.

One evaluation year is used. The horizon range is 1 to 6 hours ahead,
and forecasts are deterministic; no probabilistic evaluation is
performed. [CITE Yang 2020] argue that probabilistic verification should
be standard practice, and its absence here is a limitation rather than a
design choice.

The overcast sky class comprises 714 of 11,055 daylight hours,
approximately 6.5 percent. Skill estimates for that class rest on
roughly one tenth the sample of the clear class, and should be read
accordingly. In particular, no significance test was performed within
sky-condition subsets, so the apparent ordering of models under overcast
conditions is not established.

## B. Bounds on the Architecture Results

Two architecture results in this paper are directional rather than
established, and are reported as such.

The convolutional front end is worse than the plain recurrent model in
eight of nine array-horizon cells, but no cell reaches significance
under a Holm-corrected Diebold-Mariano test in the lagged regime; the
largest statistic is 2.30, at p = 0.13. Because the three arrays are not
independent, a sign test across cells is also inadmissible. The claim
supported by the data is that the convolutional layer shows no
detectable benefit and a consistent tendency toward harm, not that it
demonstrably degrades accuracy.

The recurrent model's advantage over gradient boosting at the six-hour
horizon is significant on one array of three (site 17, p = 0.011);
sites 11 and 12 give p = 0.073 and p = 0.059 respectively, both above
threshold after correction. This is reported as suggestive.

## C. The Residual Stage

The residual correction stage is measured as net negative throughout,
but the magnitude is sensitive to how the out-of-fold residuals are
constructed, and that sensitivity is itself the more robust finding.

With three training years the expanding-window scheme yields only two
folds, whose base models see one third and two thirds of the training
data against the deployed model's full three years. Extending training
to five years, giving four folds, recovers 19 to 96 percent of the
penalty depending on array and horizon. The measured value of this
component therefore depends substantially on a construction detail that
is rarely reported.

Two further bounds apply. The fold-starvation mechanism was diagnosed
only for the recurrent base on sites 11 and 12; the convolutional base
shows a significant penalty in all nine cells including the one-hour
horizon, which that mechanism does not explain. And the five-year,
four-fold sensitivity run was never subjected to a Diebold-Mariano test:
its reported standard errors are two-sample errors across three seeds,
which measure training stochasticity rather than sampling uncertainty in
the evaluation period.

## D. Threats to Validity

*Optimisation directive.* All models here are trained under a squared
error loss and evaluated by root-mean-square error and skill scores
derived from it. This is internally consistent, but [CITE Mayer 2022]
demonstrates that the choice of directive is itself a protocol axis:
different error metrics are minimised by different functionals, and
running the same study under a mean-absolute-error directive can reverse
conclusions about whether a component helps. We did not vary the
directive, and results here should not be assumed to transfer to a
mean-absolute-error objective.

*Weather regime.* Two feature regimes are evaluated: a lagged regime
using only observations available at issue time, and an oracle regime
using measured weather at the target time. The oracle regime is a
perfect-forecast upper bound and is never presented as an achievable
result. Operational forecasting sits between the two, using numerical
weather prediction whose own error is not represented in either regime.
The gap reported here between the two therefore bounds, rather than
estimates, the value of weather information in practice.

*Reproducibility of recurrent results.* PyTorch provides no
deterministic CUDA backward pass for its recurrent layers, and run
records report that full determinism was not achieved. Recurrent results
are consequently reported with seed variance across five seeds rather
than as point values.

*Hyperparameters.* No hyperparameter tuning was performed, on any split,
for any model. This removes one class of leakage and makes the
architecture comparison a comparison of default configurations at
comparable scale. It also means no model is presented at its best
achievable accuracy, and a differently tuned configuration might reorder
the architecture results. The protocol effects reported in Section
[VERIFY] are an order of magnitude larger than the architecture
differences and are unlikely to be affected.

*Deferred work.* A second location and a probabilistic evaluation
were both scoped and deferred; the low-elevation clear-sky bias
documented in Section [VERIFY] is a candidate protocol axis not varied
here. These are recorded in the repository rather than claimed as future
work in the abstract.
