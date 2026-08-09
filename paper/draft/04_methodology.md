# IV. METHODOLOGY

This section describes the evaluation protocol before the models. That
ordering is deliberate: the protocol is the contribution, and the models
are instruments for measuring what the protocol choices are worth.

## A. Forecast Task and Alignment

The task is deterministic point forecasting of hourly mean AC power at
horizons of 1, 3 and 6 hours ahead. Each horizon is treated as a
separate problem and reported separately; horizon dominates every other
factor in aggregate analyses of this literature (Nguyen and Musgens,
arXiv:2208.10536), so pooling across horizons would obscure the effects we set out to
measure.

The feature matrix is indexed by target time t. A row at target time t
for horizon h is the input to a forecast issued at time t-h. Every
observed quantity in that row must therefore originate at t-h or
earlier. Quantities describing t itself are admissible only when they
are deterministic and computable in advance: solar geometry, clear-sky
irradiance and power, and calendar encodings. This convention is
enforced for the tabular feature matrix by an assertion that
independently reconstructs each lagged column from the source series and
verifies the applied shift, together with a correlation probe that
flags any feature correlating implausibly with the target. For the
sequence inputs to the recurrent models, the assertion is additionally
validated by deliberate corruption: it is confirmed to fire both when
the window is shifted by one step and when the target-time value is
substituted into the window's final timestep, which is the specific
failure the convention exists to prevent.

## B. Evaluation Protocol

We state the protocol as a set of explicit rules, in the spirit of the
model information sheets proposed by Kapoor and Narayanan (2023), and
map each to the leakage type it addresses in their taxonomy.

1. Splits are chronological and never shuffled. Training is 2011-2013,
   validation 2014, test 2015, each a whole calendar year. This
   addresses temporal leakage [L3.1].

2. Evaluation is restricted to daylight hours, defined as solar
   elevation above 10 degrees. Night hours contain near-zero power that
   every model predicts correctly, and their inclusion is treated as a
   protocol variable rather than a default (Section [VERIFY]).

3. All scalers and feature statistics are fitted on training data only.
   The scaler implementation permits a single fit per instance and
   raises on a second call, so refitting on validation or test is not
   reachable by accident. This addresses preprocessing leakage [L1.2].

4. The headline metric is the root-mean-square-error skill score against
   the optimal convex combination of climatology and persistence,
   following the recommendation of Yang et al. (2020). The skill score
   against plain persistence is reported alongside, because the
   difference between the two is a result rather than a robustness
   check.

5. The lagged and oracle feature regimes are never mixed within a
   single model. Oracle features carry an explicit prefix, so that they
   cannot be mistaken for admissible predictors in a feature-importance
   analysis, and every oracle result is labelled a perfect-forecast
   upper bound.

6. The residual correction stage is fitted on out-of-fold residuals
   generated within the training period, described in Section IV-F.
   This addresses evaluating a stage on the data it was fitted to
   [L1.1].

7. Every experiment writes a machine-readable record containing its
   configuration, all metrics, timings, sample counts, random seed, the
   git commit hash, and a flag indicating whether the working tree was
   clean at execution time.

No hyperparameter tuning was performed, on any split, for any model.
This removes a class of leakage analogous to [L1.3], which Kapoor and
Narayanan define for feature selection informed by test-set performance;
the mechanism - using held-out performance to make a modelling choice -
is the same. The cost of this decision is noted in Section [VERIFY].

Two properties of this dataset are worth naming against the same
taxonomy. The three arrays are co-located and share a weather station,
so their errors are not independent across arrays. This is not [L3.2]
in Kapoor and Narayanan's sense, which concerns nonindependence between
training and test samples; it is a separate dependence, across the
units over which results are aggregated, and it constrains how many
independent observations an array-by-horizon table contains. And three
recorded channels
- cumulative delivered energy, performance ratio, and average phase
current - are deterministic functions of the target and are excluded as
illegitimate features [L2]; we note in Section [VERIFY] that published
work on this dataset has used them as inputs.

## C. Feature Regimes

Two feature regimes are constructed, and the distinction between them is
the second protocol axis we measure.

The lagged regime contains 37 features in three groups. Nine are
deterministic at target time: clear-sky power and irradiance, solar
zenith, azimuth and elevation evaluated at the hour midpoint, and cyclic
encodings of hour-of-day and day-of-year. Nineteen are observations
taken at or before the issue time: measured power, the clear-sky indices
of power and of irradiance at the issue time and the two preceding
hours, the same three quantities at a fixed 24-hour lag, five weather
channels at the issue time, and two indicators recording how stale the
clear-sky index is. Nine are rolling statistics whose windows terminate
at the issue time.

The oracle regime adds five measured weather channels at the target
time, giving 42 features. It is not a forecasting method. It bounds what
perfect weather knowledge would be worth, which is the quantity against
which architectural gains should be judged.

Two construction details are stated because they are consequential and
rarely reported. First, the daily lag is a fixed 24 hours at every
horizon rather than a horizon-relative offset, so that it remains
aligned to hour-of-day; a horizon-relative lag would point at 06:00 when
forecasting 12:00 at a six-hour horizon, losing the diurnal alignment
that makes the feature useful.

Second, the clear-sky index is undefined at night, and at a six-hour
horizon a midday target is issued near dawn. Constructing lag features
naively therefore discards most midday targets at long horizons: daylight
target retention was 75.6, 56.5 and 28.0 percent at horizons of 1, 3 and
6 hours, with targets in hours 8 to 13 dropped on essentially every day
of the training period. Models and reference forecasts would then have
been evaluated on different populations, with nothing in the metrics to
indicate it. We forward-fill the clear-sky indices with a 24-hour limit
before shifting - propagating past values forward in time only, which
cannot leak - and add explicit staleness features so a model can
discount an old observation. Rolling statistics of the clear-sky indices
are computed over the last N valid observations rather than N wall-clock
hours, since a wall-clock window ending before dawn contains no valid
samples. Forward-filling the input to a rolling standard deviation would
instead report zero variability where the truth is no recent
observation. With these changes retention is 99.0 percent at all three
horizons.

## D. Reference Forecasts

Three reference forecasts are used.

Smart persistence forecasts P(t) = k_p(t-h) x P_cs(t), persisting the
clear-sky index from the issue time. Where no valid index exists at the
issue time it is forward-filled with a 24-hour limit, matching the
feature layer.

Climatology forecasts P(t) = k_p_bar(month, hour) x P_cs(t), where the
mean clear-sky index per calendar month and hour of day is estimated on
training data only. It uses no recent observation, so the horizon does
not affect it.

The convex reference is w x persistence + (1-w) x climatology, with w
selected by grid search in steps of 0.01 to minimise root-mean-square
error on the validation split, fitted independently for each array,
horizon, seed and regime. Following Yang et al. (2020), this is the
reference for the headline skill score.

The third reference is necessary because the first degrades structurally
with horizon. At a six-hour horizon a midday target is issued near dawn,
so the persisted clear-sky index is stale: the proportion of daylight
forecasts relying on a forward-filled rather than directly observed
index rises from 3.3 percent at one hour to 22.6 percent at three hours
and 51 percent at six. The resulting bias reaches -2.15 kW at midday on
a 5 kW array. A skill score measured against a reference that fails in
this way rewards a model for handling staleness rather than for
forecasting, and the models can do so - they receive explicit staleness
features - while the reference cannot.

## E. Metrics

We report mean bias error, mean absolute error, root-mean-square error,
root-mean-square error normalised by nameplate capacity, and the skill
score

  s = 1 - RMSE_forecast / RMSE_reference

Normalisation is by nameplate capacity rather than by mean observed
power, so that values are comparable across arrays of different sizes.
We note that normalised error is not comparable across studies with
different data regardless of the normaliser (Yang et al., 2020), and make
no cross-study accuracy comparisons on that basis.

All metrics for a given cell are computed on the intersection of
timestamps for which every compared forecast produced a prediction, and
that sample count is recorded with every result. Documented outage hours
are excluded before this intersection is formed.

## F. Models

Five models are evaluated, chosen to decompose a hybrid architecture
rather than to survey the field.

Gradient boosting over the tabular feature matrix serves as the
non-recurrent reference point. No scaling is applied, since trees are
invariant to monotone transforms of individual features; there is
consequently nothing to fit on training data and nothing to leak.

The recurrent model applies a single LSTM layer over a 24-step sequence
of nine per-hour channels ending at the issue time, concatenates the
final hidden state with the deterministic target-time features, and maps
the result through a two-layer head. The deterministic features are
necessary: without them the network has no representation of what time
of day it is forecasting for.

The convolutional-recurrent model prepends a one-dimensional
convolution over the time axis of the same sequence. Same-length padding
is used, which does not leak, because the entire sequence window lies at
or before the issue time by construction.

The two residual-corrected models wrap either recurrent base with a
gradient-boosted stage fitted to the base model's residuals. The
construction of those residuals is the third protocol axis we measure.
Residuals are generated out of fold within the training period by an
expanding window: a base model is fitted on the first training year and
predicts the second, then fitted on the first two and predicts the
third, and the out-of-sample residuals are pooled. The base model is
then refitted once on the full training period, with validation used
only for early stopping. Each fold's base model sees less data than the
deployed one, so the pooled residuals are slightly pessimistic, which is
the correct direction to err.

The alternative - fitting the residual stage on validation residuals -
is reachable behind an explicit flag that emits a warning and records
its use in the run configuration. It is included because measuring what
it produces is one of this paper's results, not because it is a method
we endorse.

## G. Significance Testing

Model comparisons use the Diebold-Mariano test on paired squared-error
series (Diebold and Mariano, 1995). Because h-step-ahead forecast errors are
autocorrelated by construction, the long-run variance is estimated with
a Bartlett kernel truncated at lag h-1. The small-sample correction of
Harvey, Leybourne and Newbold (1997) is applied and the statistic
compared against Student's t with n-1 degrees of freedom. Within each
array-horizon cell, twenty-one pairwise comparisons are made across the
five models and two of the reference forecasts, and p-values are
adjusted by the Holm-Bonferroni procedure.

Seed variance across five seeds is reported separately and is not used
as a significance test. The two quantities answer different questions:
seed spread measures whether a result would recur on retraining, while
the Diebold-Mariano test measures whether one forecast is better than
another on this evaluation sample. An earlier version of this analysis
used a two-standard-deviation seed heuristic in place of a paired test,
and doing so changed the reported significance of one architecture
comparison; the heuristic is reported in Section [VERIFY] as one of the
protocol variables under study rather than used as a method.
