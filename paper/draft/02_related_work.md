# II. RELATED WORK

## A. Hybrid Deep Learning for PV Power Forecasting

Hybrid architectures combining convolutional and recurrent layers, often
with a further gradient-boosted or decomposition stage, dominate recent
work on short-term photovoltaic power forecasting. Aggregate analyses of
the field support their prominence: a statistical meta-analysis of 180
studies published since 2007 reports that hybrid models consistently
outperform alternatives and concludes they are likely to be the future of
PV output forecasting [CITE Nguyen Musgens].

That conclusion aggregates reported errors. What follows examines what
those reported errors are measured against.

## B. A Survey of Evaluation Practice

To characterise evaluation practice rather than architecture, we coded 27
papers on PV power forecasting, drawn from searches for hybrid
convolutional-recurrent methods published between 2022 and 2026. Each
paper was coded on eight dimensions of evaluation protocol. Coding was
conservative: a field was recorded as not stated unless the paper
explicitly stated it, and a plotting range, a figure axis, or a
contextual implication was not accepted in place of a statement. Every
coded judgement is supported by a verbatim quotation from the source,
recorded in the accompanying repository; 25 of the 27 papers were coded
this way, and the two coded from summary notes without a source document
are marked as such and excluded from no count. Table [VERIFY] reports the
full coding.

Four results characterise the sample.

*No reference forecast.* Twenty-six of 27 papers report no skill score
against any reference forecast. Seventeen compare only against their own
architecture's components - a hybrid against the convolutional and
recurrent models it is built from - and a further six compare only
against other machine-learning models. Three report no comparison model
at all. Reported improvements in this literature are therefore
improvements over strictly weaker versions of the same approach, and
carry no information about whether the proposed model exceeds a naive
forecast.

*No run-to-run variance.* All 27 papers report point estimates without
any measure of variance across random seeds or repeated runs. Recurrent
models are stochastic in training, and the seed spread we measure in
Section [VERIFY] reaches 0.009 in skill score - larger than many of the
improvements claimed in this literature. Every reported figure in the
surveyed sample therefore carries an unstated error bar of unknown size.

*No stated split protocol.* Twenty-two of 27 papers do not state whether
their train-test split preserves temporal order. Three state a
chronological split, one uses rolling-origin evaluation, and one uses
k-fold cross-validation - which, applied to a time series, is temporal
leakage in the hyperparameter selection stage [CITE Kapoor Narayanan],
[CITE Hewamalage]. A stated ratio such as 80/20 is not a statement of
ordering, and we did not treat it as one.

*No night-hour statement.* Seventeen of 27 papers do not state whether
night hours are excluded from evaluation. Eight state a daylight
restriction, one applies a partial restriction, and one explicitly
retains all hours. Since night power is near zero and trivially
predicted, this choice materially affects any error metric normalised
over all hours, as we quantify in Section [VERIFY].

Three further observations. Only one paper in the sample makes code
available. Three papers describe procedures in their own text that
constitute leakage under the taxonomy of Kapoor and Narayanan [CITE]:
two apply signal decomposition or feature selection to the full series
before splitting [L1.2], and one uses features that are deterministic
functions of the target as model inputs [L2]. We identify these from the
papers' own descriptions and do not infer leakage where a paper is
silent; the twenty-two papers that do not state their split protocol may
or may not be affected.

## C. Rigorous Practice Exists

This is not a claim that solar forecasting lacks evaluation standards.
The verification of deterministic solar forecasts has been treated
extensively, and the recommendations are explicit: Yang et al. [CITE],
writing with more than thirty co-authors, recommend universally
reporting the root-mean-square-error skill score against the optimal
convex combination of climatology and persistence, and treat
distribution-oriented verification as the standard against which
measure-based evaluation is a supplement. The same work documents the
instability of the clear-sky index at low solar elevation and the zenith
filtering convention that exists because of it - practices that are
routine in the irradiance forecasting literature.

One paper in our sample of 27 meets that standard. Mayer [CITE] evaluates
day-ahead forecasts for fourteen plants using the skill score against the
convex combination of persistence and climatology, states its daylight
filter as a protocol, holds out a full year that is used in no part of
model selection, reports six metrics including bias and variance ratio,
and checks the operational feasibility of its forecast timing against the
grid's submission deadline. It also reports which of its own methods
perform worst. Its headline hybrid improvement is 5.2 percent in mean
absolute error and 1.0 percent in root-mean-square error over an
optimised physical model chain - an order of magnitude smaller than
improvements commonly reported in the sample surveyed above, and measured
against a considerably stronger baseline.

The gap this paper addresses is therefore not the absence of standards
but their non-adoption in a specific and productive subfield. The
standards are published in the same journals.

## D. Leakage and Evaluation Beyond Solar Forecasting

The pattern is not unique to this domain. Kapoor and Narayanan [CITE]
document leakage in at least 294 papers across 17 scientific fields and
provide a taxonomy of eight types, which we adopt in Section [VERIFY].
Their civil war prediction case study is directly analogous to the
finding we report: complex machine-learning models believed to
substantially outperform logistic regression were found, once leakage was
corrected, not to outperform a decades-old method at all.

From the forecasting side, Hewamalage et al. [CITE] observe that random
cross-validation does not preserve temporal order and that rolling-origin
evaluation is itself susceptible to leakage, and note encountering
inadequate benchmark comparisons in top-tier venues. Nguyen and Musgens
[CITE], in the meta-analysis cited above, find test-set length negatively
correlated with reported accuracy and attribute this in part to selective
reporting on favourable subsets, recommending test sets of at least one
year - a recommendation our design follows.

## E. Positioning

Against this background, this paper does not propose an architecture.
It fixes a leakage-controlled protocol, holds it constant, and measures
how much reported accuracy moves when individual evaluation choices are
varied - the choices that the survey above shows to be commonly
unreported. The models serve as instruments for that measurement rather
than as contributions in themselves.
