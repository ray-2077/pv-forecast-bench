# VI. RESULTS

Results are organised by research question, beginning with protocol
sensitivity. All figures are reported for the validation year (2014)
unless stated otherwise; Section VI-E reports the held-out test year
(2015), which was evaluated once after every modelling and evaluation
choice was frozen.

## A. RQ1: Protocol Sensitivity

We vary four evaluation choices, holding the model, the data and the
training procedure constant. In each case only the protocol changes.

### 1) Night-hour inclusion

Restricting evaluation to daylight hours or retaining all 24 hours
changes reported error substantially and reported skill almost not at
all. For gradient boosting on site 11, normalised root-mean-square error
falls from 6.36 to 4.20 percent at a one-hour horizon, from 8.97 to 5.90
at three hours, and from 10.71 to 7.04 at six hours. The skill score
against smart persistence over the same rows moves from 0.252 to 0.261,
0.527 to 0.528, and 0.652 to 0.652.

The magnitude of the error reduction is not an empirical accident. If
night-time errors are approximately zero, then

  RMSE_all = RMSE_daylight x sqrt(N_daylight / N_all)

At six hours, N_daylight = 3756 and N_all = 8694, giving a predicted
ratio of 0.657 against an observed 0.658. Agreement is within 0.3
percent at every array and horizon.

Two consequences follow. Reported normalised error at this site is
deflated by approximately 34 percent by a protocol choice that is
unreported in 17 of the 27 papers we surveyed, and the deflation factor
is a property of latitude and season rather than of the forecasting
method. A skill score, by contrast, is nearly invariant to the choice,
because the reference forecast is deflated by the same factor.

### 2) Reference forecast

The choice of reference changes both the magnitude and the shape of
reported skill. For gradient boosting on site 11 over daylight hours,
skill against smart persistence is 0.252, 0.527 and 0.652 at one, three
and six hours. Against the convex combination of climatology and
persistence, over the same rows and the same forecasts, it is 0.200,
0.276 and 0.194.

At six hours, 70 percent of the apparent skill is attributable to the
reference. The pattern holds on all three arrays: 0.244 to 0.190, 0.510
to 0.275 and 0.639 to 0.201 on site 12, and 0.166 to 0.122, 0.341 to
0.183 and 0.386 to 0.132 on site 17.

The change in shape is the more consequential result. Measured against
persistence, skill increases monotonically with horizon, a pattern
frequently reported as evidence that a method's advantage grows at
longer lead times. Measured against the convex reference, skill is
non-monotonic and peaks at three hours, falling at six to approximately
its one-hour value. The monotonic trend is a property of the reference,
not of the model. Fig. [VERIFY F1] shows this for all three arrays.

The mechanism is documented in Section IV-D: smart persistence degrades
structurally with horizon at this site because a midday target at a
six-hour horizon is issued near dawn, when no valid clear-sky index
exists. The fitted convex weight reflects this directly, falling from
0.77 at one hour to 0.25 at three and 0.04 at six on site 11 - at long
horizons the optimal reference is almost entirely climatology.

Site 17 is an exception worth stating: its convex weight remains higher
at every horizon (0.83, 0.50, 0.31), because its persistence forecast is
stronger in relative terms. This matters in Section VI-E.

### 3) Residual-stage fitting

The residual correction stage can be fitted on out-of-fold training
residuals, as specified in Section IV-F, or on validation residuals and
then evaluated on validation. The second is in-sample evaluation of the
correction stage. Its effect is the largest single protocol effect we
measure.

On site 11 at a six-hour horizon with a recurrent base, the corrected
scheme gives a skill score of 0.177 against the convex reference. The
plain recurrent model without any correction gives 0.211. The in-sample
scheme gives 0.545.

A single choice in how residuals are constructed therefore turns a
component that costs 0.034 in skill into one that appears to contribute
0.334 - a reversal of sign and an order of magnitude in size. The
feature importances of the correction stage also reorder entirely, so
any component-attribution conclusion drawn from the in-sample variant
would be wrong about which features carry residual signal, not merely
about how much.

### 4) Weather information regime

Substituting measured weather at the target time for observations
available at issue time raises skill against the convex reference from
0.276 to 0.783 on site 11 at three hours. Across all arrays and
horizons the gap ranges from 0.51 to 0.72.

This is not an achievable result and is not reported as one; it bounds
what perfect weather knowledge would be worth. The bound is
approximately twenty-five times the largest difference we measure
between any two architectures under identical conditions.

## B. RQ2: Component Attribution

We decompose the hybrid architecture by measuring each component against
the same protocol, with five seeds per configuration.

### 1) Recurrence

At one- and three-hour horizons, the recurrent model and gradient
boosting are indistinguishable: differences in mean skill against the
convex reference are between 0.000 and 0.004 across the three arrays,
against seed standard deviations of 0.001 to 0.005.

At six hours the recurrent model leads on all three arrays, by 0.016,
0.015 and 0.024. Under a Holm-corrected Diebold-Mariano test this
difference is significant on one array of three (site 17, p = 0.011);
sites 11 and 12 give p = 0.073 and p = 0.059. We report the six-hour
advantage as suggestive rather than established, and note in Section
VI-E that it weakens further on the test year.

### 2) Convolution

The convolutional front end is worse than the plain recurrent model in
eight of nine array-horizon cells on validation, by up to 0.014, and it
has the highest seed variance of the three base architectures in seven
of nine cells. No cell reaches significance under a Holm-corrected
Diebold-Mariano test in the lagged regime; the largest statistic is
2.30, at p = 0.13. Because the three arrays share a weather station and
are not independent, a sign test across cells is not admissible either.

The claim supported is that the convolutional layer shows no detectable
benefit and a consistent tendency toward harm, at approximately 21
percent additional mean training time, though the per-cell ratio varies
widely, from 24 percent faster to 50 percent slower across the nine
array-horizon cells.

### 3) Residual correction

Under the corrected out-of-fold scheme, residual correction reduces
skill in all eighteen array-horizon-base cells, by 0.024 to 0.046. With
a convolutional base the penalty is Holm-significant in all nine cells;
with a recurrent base it is significant in five of nine.

The magnitude, however, depends on a construction detail. With three
training years the expanding window yields only two folds, whose base
models see one third and two thirds of the training data against the
deployed model's full three years. Extending training to five years,
giving four folds, recovers between 19 and 96 percent of the penalty
depending on array and horizon, with the recovery largest at one hour
and smallest at six.

A diagnostic on the correction stage itself explains the residual
penalty at long horizons. The correlation between the predicted and
actual residual is 0.76 out of fold but 0.10 on validation at three
hours, and 0.79 against 0.04 at six. The corrector is not learning
nothing; it is miscalibrated. Adding a correction p to a base with
residual r changes mean squared error by -2 rho sigma_r sigma_p +
sigma_p^2, so it helps only when sigma_p < 2 rho sigma_r. The observed
ratio sigma_p / sigma_r is 0.33 to 0.38 across configurations against a
break-even of 0.07 to 0.26 - the correction is applied at between 1.2
and 4.7 times the magnitude its out-of-sample correlation justifies. The
overconfidence is worst at six hours, which is where the penalty
survives fold correction.

### 4) The full architecture

The complete proposed hybrid - convolutional front end, recurrent layer,
and gradient-boosted residual correction - is worse than gradient
boosting alone in all nine array-horizon cells on validation, by 0.027
to 0.044 in skill against the convex reference, at approximately
forty-six times the training cost.

## C. RQ3: Conditional Performance

Stratifying by sky condition, classified from the clear-sky index of
irradiance and its variability over the last three valid observations,
gives class proportions of 60.4 percent clear, 33.2 percent partly
cloudy and 6.5 percent overcast across the daylight hours of the
validation year, identical across arrays as expected from a shared
weather station.

Error and skill rank the classes differently, and the disagreement is
the result. On site 11 at a three-hour horizon, overcast conditions give
the highest normalised error of any class, 16.2 percent against 11.7 for
partly cloudy and 5.4 for clear - but the highest skill after clear
conditions, 0.388 against 0.056. At six hours the pattern is stronger:
overcast error reaches 24.3 percent with skill 0.267, while partly
cloudy error is 11.7 percent with skill 0.035.

Overcast conditions are the hardest to forecast in absolute terms, and
the reference forecast fails there too, so a model retains skill. Partly
cloudy conditions produce middling absolute error but almost no skill,
because the model and the reference perform comparably. Reported by
error alone, overcast would be identified as the problem regime;
reported by skill, partly cloudy is the regime in which modelling adds
nothing.

Weighted by frequency, most of the aggregate skill at this site
originates in the 60 percent of hours that are clear - the regime in
which persistence is already strong. An aggregate skill score at a
desert site is largely a clear-sky skill score.

Residual correction is worse than the plain recurrent model in eight of
nine sky-class cells, independent confirmation of Section VI-B on a
different partition of the same data. We note that the overcast class
comprises 714 of 11,055 daylight hours and no significance test was
performed within sky-condition subsets.

## D. RQ4: Accuracy per Unit Compute

Mean fitting time per configuration, measured across all 45 lagged
validation runs per model, is 0.55 s for gradient boosting, 10.11 s for
the recurrent model, 12.23 s for the convolutional-recurrent model, 22.7
s for the recurrent model with residual correction and 25.0 s for the
convolutional-recurrent model with residual correction.

Gradient boosting therefore achieves skill statistically
indistinguishable from the recurrent model at one and three hours, at
approximately one eighteenth of the training cost, and skill higher than
either residual-corrected variant at one fortieth to one forty-fifth of
the cost. Fig. [VERIFY F6] plots skill against fitting time.

The comparison is bounded by the scale of this study: fitting times are
for a single laptop GPU on approximately 13,000 training samples, and
inference cost, which dominates in operational deployment, is not
measured.

## E. Test-Split Confirmation

The test split (2015) was evaluated once, after all modelling and
evaluation choices were frozen, over 450 runs comprising five models,
two regimes, three arrays, three horizons and five seeds.

Every direction of finding reported above holds on the held-out year.
All models beat the convex reference in all cells; the skill-versus-
horizon shape remains non-monotonic against the convex reference (0.210,
0.316, 0.232 for gradient boosting on site 11) and monotonic against
persistence (0.266, 0.574, 0.701); residual correction reduces skill in
all eighteen cells; the convolutional front end is worse than the plain
recurrent model in all nine; and the oracle-to-lagged gap remains
between 0.52 and 0.75.

Two results change in magnitude and are reported as such.

First, the six-hour advantage of the recurrent model over gradient
boosting weakens. Validation gives 0.016, 0.015 and 0.024 on the three
arrays; the test year gives 0.007, 0.009 and -0.000. On site 17 - the
one array where the difference reached significance on validation - it
disappears entirely. Combined with the marginal significance on the
other two arrays, we do not consider a six-hour architectural advantage
established by this study.

Paired Diebold-Mariano tests on the test split support this. Across all
nine array-horizon cells, no comparison between gradient boosting and
either recurrent model reaches significance after Holm correction (18 of
18 non-significant); at site 11 and six hours the statistic marginally
favours the recurrent model (HLN = 2.24, p = 0.127). The convolutional
front end is likewise never significantly different from the plain
recurrent model in any cell. Two results hold throughout: every model
beats smart persistence in every cell, and beats the convex reference in
88 of 90 comparisons, the two exceptions being the residual-corrected
variants at site 17 and one hour, which remain directionally better but
do not reach significance. The convex reference beats smart persistence
in all nine cells, with a loss differential reaching 2.30 kW-squared at
site 11 and six hours (HLN = 22.58). Residual correction is
significantly worse than its base model in 12 of the 18 base-versus-
residual comparisons: 4 of 9 with a recurrent base and 8 of 9 with a
convolutional-recurrent base.

Second, absolute skill shifts by array in opposite directions: upward by
0.013 to 0.039 on sites 11 and 12, and downward by 0.011 to 0.076 in 14
of the 15 model-horizon cells on site 17. The single exception is
gradient boosting at six hours, which rises by 0.013. Because the
direction is shared by four of the five architectures, including both
recurrent bases and both residual variants, an architectural explanation
is implausible, though not excluded by this evidence alone.

We traced it. It is not the documented June 2015 outage, whose exclusion
window is correctly sized - the one adjacent depressed day is shared
weather, confirmed against site 11, which had no outage. It is not
degradation drift: the clear-sky index of power is flat from 2013 to
2015 on all three arrays. It is not target volatility, which moved in
the opposite direction, site 17 becoming less volatile on the test year
while sites 11 and 12 became more volatile at the longer horizons.

It is a denominator effect. Decomposing skill into numerator and
denominator, the models' own errors improved modestly on site 17 while
its reference improved far more: the convex reference root-mean-square
error fell by 0.042 to 0.062 on site 17 against 0.010 to 0.022 on the
other two arrays. Tracing further, climatology improved comparably for
all three arrays, while persistence improved substantially for site 17
at every horizon and worsened for sites 11 and 12 at the longer
horizons. Site 17's convex reference weights persistence far more
heavily (0.31 to 0.83, against 0.04 to 0.29), so it is structurally more
exposed to persistence's year-to-year variation.

Reported skill therefore depends on how the reference behaves in the
evaluation year, and two co-located arrays sharing weather and models
can move in opposite directions between years because their optimal
reference weights differ. Why persistence's accuracy diverged by array
is not established here and is left open.
