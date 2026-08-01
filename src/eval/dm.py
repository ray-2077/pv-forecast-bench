"""Diebold-Mariano tests for pairwise forecast comparison.

Diebold, F.X. and Mariano, R.S. (1995). "Comparing Predictive Accuracy."
Journal of Business & Economic Statistics, 13(3), 253-263.

Harvey, D., Leybourne, S., and Newbold, P. (1997). "Testing the Equality
of Prediction Mean Squared Errors." International Journal of Forecasting,
13(2), 281-291. Small-sample correction (HLN) applied below.

This REPLACES the 2x-seed-std heuristic in scripts/aggregate_seed_sweep.py
as the paper's significance test. That heuristic (does a mean difference
in skill_vs_convex exceed 2 standard deviations of either model's own
5-seed spread) answers a different question - whether five arbitrary
seeds happen to separate two models - and was never a hypothesis test
with a p-value. Seed spread stays in the paper as a reproducibility
statistic (Table 3); DM is the significance test (Table 6).

Only pure functions here, in the style of src/eval/metrics.py: no model
code, no file I/O, no plotting.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

_LOSS_FUNCS = {
    "squared": lambda e: e ** 2,
    "absolute": lambda e: np.abs(e),
}

_ALTERNATIVES = ("two-sided", "less", "greater")


def _loss_differential(e1, e2, loss):
    if loss not in _LOSS_FUNCS:
        raise ValueError(f"loss must be one of {list(_LOSS_FUNCS)}, got {loss!r}")
    L = _LOSS_FUNCS[loss]
    return L(e1) - L(e2)


def _bartlett_hac_variance(d, h):
    """Long-run variance of dbar = mean(d), via a HAC (heteroskedasticity
    and autocorrelation consistent) estimator with a Bartlett kernel,
    truncated at lag h - 1.

    WHY a HAC estimator rather than the plain sample variance: d_t is the
    loss differential for an h-step-ahead forecast. Even if the underlying
    series had no serial correlation of its own, h-step-ahead forecast
    errors overlap - the forecast issued for target time t and the one
    issued for t+1 share up to h-1 hours of the same unobserved future
    shocks - so d_t and d_{t-k} are correlated by construction for any
    k < h. Var(dbar) is then NOT sample_var(d) / n: ignoring the
    autocovariance terms (as a naive t-test does) understates Var(dbar)
    whenever that autocorrelation is positive, which overstates the DM
    statistic and understates the p-value. This is exactly the mistake
    Diebold & Mariano (1995) built this long-run-variance construction to
    avoid.

    gamma_0 is the sample variance (denominator n, not n - 1, matching the
    original DM derivation); gamma_k is the sample autocovariance at lag
    k. The Bartlett weight at lag k is (1 - k / h): it downweights linearly
    and would reach zero exactly at k = h, but lags k = h, h+1, ... are not
    included in the sum at all - "truncation lag h - 1" means only lags
    1 .. h-1 are summed, which is also why h = 1 (a one-step forecast, no
    induced overlap) reduces this exactly to the plain sample variance.
    """
    n = len(d)
    dbar = np.mean(d)
    centered = d - dbar

    var = np.mean(centered ** 2)  # gamma_0
    max_lag = min(h - 1, n - 1)
    for k in range(1, max_lag + 1):
        gamma_k = np.mean(centered[k:] * centered[:-k])
        weight = 1.0 - k / h
        var += 2.0 * weight * gamma_k
    return float(var)


def dm_test(e1, e2, h, loss="squared", alternative="two-sided"):
    """Diebold-Mariano test comparing two aligned forecast ERROR series.

    e1, e2: pandas Series of y_true - y_pred for two models forecasting
    the same target times at the same horizon `h`. Must be pandas Series
    with IDENTICAL indices - raises ValueError if the indices differ.
    Alignment is never done implicitly here: two error series that look
    the same length but correspond to different timestamps would silently
    produce a meaningless comparison, so the caller must align them (e.g.
    to a common prediction intersection) before calling this function.

    h: forecast horizon in hours. Used only to set the HAC truncation lag
    (h - 1) in _bartlett_hac_variance - see that function's docstring for
    why h-step errors need this instead of a plain t-test.

    loss: 'squared' (default, i.e. compares based on squared error) or
    'absolute'.

    alternative: 'two-sided' (default), 'less' (tests H1: model 1's
    expected loss is less than model 2's), or 'greater'.

    SIGN CONVENTION - read this before reporting a result:
        d_t = L(e1_t) - L(e2_t)
    dbar < 0 means MODEL 1 has the LOWER average loss, i.e. MODEL 1 IS
    BETTER. This is the single most common source of misreported DM
    results: the sign is relative to whichever series is passed as e1,
    not some absolute notion of "positive means better".

    Steps:
      1. d_t = L(e1_t) - L(e2_t)
      2. dbar = mean(d)
      3. HAC long-run variance of dbar (Bartlett kernel, truncation h - 1)
      4. DM = dbar / sqrt(hac_var / n)
      5. Harvey-Leybourne-Newbold small-sample correction:
         HLN = DM * sqrt((n + 1 - 2h + h(h-1)/n) / n)
         compared against a Student t distribution with n - 1 df, NOT the
         standard normal DM originally used - HLN (1997) showed the normal
         approximation rejects too often in samples of the size typical of
         forecast evaluation.

    Returns a dict: dm_stat, hln_stat, p_value (from the HLN statistic and
    the t(n-1) distribution), n, dbar, mean_loss_1, mean_loss_2,
    better_model ('model_1', 'model_2', or 'tie' if dbar is exactly 0).
    """
    if not isinstance(e1, pd.Series) or not isinstance(e2, pd.Series):
        raise TypeError(
            "dm_test requires e1 and e2 to be pandas Series indexed by "
            "timestamp, so index alignment can actually be checked"
        )
    if not e1.index.equals(e2.index):
        raise ValueError(
            "dm_test: e1 and e2 have different indices - align them "
            "explicitly (e.g. to a common prediction intersection) before "
            "calling; this function never silently aligns two series that "
            "might not correspond to the same timestamps"
        )
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")

    n = len(e1)
    if n < 2:
        raise ValueError(f"dm_test needs at least 2 aligned observations, got {n}")

    e1v = e1.to_numpy(dtype=float)
    e2v = e2.to_numpy(dtype=float)

    d = _loss_differential(e1v, e2v, loss)
    dbar = float(np.mean(d))

    hac_var = _bartlett_hac_variance(d, h)
    if hac_var <= 0:
        # Only possible if d is (numerically) constant - e.g. e1 == e2
        # exactly, so dbar is also exactly 0. Treat as "no difference"
        # rather than dividing by zero.
        dm_stat = 0.0
    else:
        dm_stat = dbar / np.sqrt(hac_var / n)

    hln_factor_sq = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if hln_factor_sq < 0:
        raise ValueError(
            f"HLN correction requires n large relative to h (got n={n}, "
            f"h={h}); (n + 1 - 2h + h(h-1)/n)/n = {hln_factor_sq:.4f} is "
            "negative, so its square root is undefined"
        )
    hln_stat = dm_stat * np.sqrt(hln_factor_sq)

    df = n - 1
    if alternative == "two-sided":
        p_value = 2.0 * stats.t.sf(abs(hln_stat), df=df)
    elif alternative == "less":
        p_value = stats.t.cdf(hln_stat, df=df)
    else:  # 'greater'
        p_value = stats.t.sf(hln_stat, df=df)
    p_value = float(min(p_value, 1.0))

    mean_loss_1 = float(np.mean(_LOSS_FUNCS[loss](e1v)))
    mean_loss_2 = float(np.mean(_LOSS_FUNCS[loss](e2v)))

    if dbar < 0:
        better_model = "model_1"
    elif dbar > 0:
        better_model = "model_2"
    else:
        better_model = "tie"

    return {
        "dm_stat": float(dm_stat),
        "hln_stat": float(hln_stat),
        "p_value": p_value,
        "n": int(n),
        "dbar": dbar,
        "mean_loss_1": mean_loss_1,
        "mean_loss_2": mean_loss_2,
        "better_model": better_model,
    }


def dm_matrix(errors_by_model, h, loss="squared", alternative="two-sided"):
    """Pairwise Diebold-Mariano tests across a dict of model name -> error
    series, all already aligned to an IDENTICAL index (see dm_test) -
    intended to be called once per array x horizon cell, over whatever
    intersection of timestamps every model in the dict actually produced a
    prediction for.

    With m models there are C(m, 2) pairwise comparisons - 10 for the 5
    models in this project's grid, or 21 once smart_persistence and
    convex_reference are added as comparators (scripts/build_table6_dm.py
    uses 7 models). A Holm-Bonferroni correction is applied across exactly
    those comparisons (one correction per array x horizon cell, not
    globally across cells): under the null of no true difference anywhere
    in a cell, the chance that at least one of 10 independent p < 0.05
    tests fires by chance alone is 1 - 0.95**10 ~ 40%, not 5% - reporting
    raw p-values as if each pair were the only test run would overstate
    how many "significant" differences the cell actually contains. Holm's
    step-down procedure controls the family-wise error rate without the
    strict (and here unnecessary) independence assumption plain Bonferroni
    needs, and is uniformly more powerful than plain Bonferroni.

    Returns (hln_df, p_holm_df, pairs_df):
      hln_df, p_holm_df: square DataFrames indexed and columned by model
        name. hln_df is antisymmetric (hln_df.loc[a, b] == -hln_df.loc[b, a],
        from calling dm_test(errors[a], errors[b], ...) with a as model 1);
        the diagonal is 0. p_holm_df is symmetric (a p-value does not
        depend on comparison direction) with the diagonal set to 1.0.
      pairs_df: one row per unordered pair (model_1, model_2), columns
        model_1, model_2, dbar, dm_stat, hln_stat, p_raw, p_holm, n,
        better_model - the raw, un-pivoted results, including BOTH raw and
        Holm-adjusted p-values (p_holm_df above only carries the adjusted
        ones). This is what scripts/build_table6_dm.py writes to
        results/table6_dm.csv after adding its own array/horizon columns.
    """
    model_names = list(errors_by_model.keys())
    m = len(model_names)
    if m < 2:
        raise ValueError(f"dm_matrix needs at least 2 models, got {m}")

    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            a, b = model_names[i], model_names[j]
            result = dm_test(errors_by_model[a], errors_by_model[b], h, loss=loss, alternative=alternative)
            pairs.append({"model_1": a, "model_2": b, **result})

    raw_p = [p["p_value"] for p in pairs]
    _, p_holm, _, _ = multipletests(raw_p, method="holm")

    pairs_df = pd.DataFrame([
        {
            "model_1": p["model_1"],
            "model_2": p["model_2"],
            "dbar": p["dbar"],
            "dm_stat": p["dm_stat"],
            "hln_stat": p["hln_stat"],
            "p_raw": p["p_value"],
            "p_holm": float(p_holm_i),
            "n": p["n"],
            "better_model": p["model_1"] if p["better_model"] == "model_1"
            else p["model_2"] if p["better_model"] == "model_2" else "tie",
        }
        for p, p_holm_i in zip(pairs, p_holm)
    ])

    hln_df = pd.DataFrame(0.0, index=model_names, columns=model_names)
    p_holm_df = pd.DataFrame(1.0, index=model_names, columns=model_names)
    for row in pairs_df.itertuples():
        hln_df.loc[row.model_1, row.model_2] = row.hln_stat
        hln_df.loc[row.model_2, row.model_1] = -row.hln_stat
        p_holm_df.loc[row.model_1, row.model_2] = row.p_holm
        p_holm_df.loc[row.model_2, row.model_1] = row.p_holm

    return hln_df, p_holm_df, pairs_df
