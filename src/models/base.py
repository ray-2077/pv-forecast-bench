"""Abstract interface every forecasting model implements.

ALIGNMENT CONVENTION (the single most important thing in this harness):
predict(df, horizon) returns a pd.Series INDEXED BY TARGET TIME. The
prediction stored at target time t must use ONLY information available at
time t - horizon hours. A model that peeks at df.loc[t] (or later) to
produce its prediction for t has leaked the future into the forecast and
the result is not a valid horizon-h forecast, no matter how good the
number looks. Every subclass's predict() must honour this, and
check_no_lookahead below is the reusable check that a given (df, preds,
horizon) triple is at least consistent with it.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Common fit/predict interface for all models in this benchmark.

    Subclasses set `name` (a class attribute) to a short identifier used in
    run IDs and results/<run_id>.json - see CLAUDE.md rule 7.
    """

    name: str

    @abstractmethod
    def fit(self, df_train: pd.DataFrame, horizon: int) -> "BaseForecaster":
        """Learn from training data for one specific forecast horizon.

        Returns self, so calls can be chained as model = Model().fit(...).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """Forecast `horizon` hours ahead.

        Returns predictions indexed by TARGET time t. Per the module
        docstring, the value at t must be derivable from information
        available at t - horizon only.
        """
        raise NotImplementedError


def check_no_lookahead(df: pd.DataFrame, preds: pd.Series, horizon: int) -> None:
    """Reusable sanity check for the alignment convention above.

    Verifies:
    1. Every timestamp in preds.index also appears in df.index - a model
       cannot predict a target time it was never given.
    2. None of the first `horizon` timestamps of df.index appear in
       preds.index - there is not yet horizon hours of history behind
       those rows, so nothing can be forecast for them.

    Raises AssertionError if either check fails. Does not check anything
    about what happens strictly between horizon and len(df) - individual
    models may legitimately produce fewer predictions than that (e.g. due
    to their own NaN handling).
    """
    assert preds.index.isin(df.index).all(), (
        "preds index is not a subset of df index"
    )

    forbidden = df.index[:horizon]
    assert not preds.index.isin(forbidden).any(), (
        f"predictions found within the first {horizon} hours of df.index; "
        "nothing can be forecast before enough history exists"
    )
