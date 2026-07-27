"""XGBoost forecaster: gradient-boosted trees on the feature matrix built by
src.features.build, in either regime.

See src/models/base.py for the alignment convention that predict() must
honour - it is enforced by build_features (src/features/build.py), which
this model relies on entirely for lag/shift correctness. This module does
no leakage-relevant work of its own beyond choosing which rows to keep.
"""

import xgboost as xgb
import pandas as pd

from src.features.build import build_features
from src.models.base import BaseForecaster


class XGBForecaster(BaseForecaster):
    """Gradient-boosted tree forecaster for one horizon and regime.

    Hyperparameters are stored as given, never tuned inside this class -
    per the task, tuning (if any) happens later, against the VALIDATION
    split only.
    """

    name = "xgboost"

    def __init__(
        self,
        seed,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        regime="lagged",
    ):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.regime = regime

        self._model = None
        self._feature_cols = None
        self.best_iteration = None

    def fit(self, df_train: pd.DataFrame, horizon: int, df_val: pd.DataFrame = None) -> "XGBForecaster":
        X_train, y_train = build_features(df_train, horizon, self.regime)
        self._feature_cols = list(X_train.columns)

        model_kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.seed,
            # device="cpu", not "cuda": ~50k rows x ~40 features is far too
            # small for the RTX 3070 Ti to pay off - PCIe transfer and
            # kernel-launch overhead dominate at this scale - and CPU
            # (hist) tree construction is bit-for-bit deterministic given a
            # fixed seed, whereas GPU histogram building is not
            # guaranteed to be. Reproducibility beats a speedup that
            # would not materialise anyway.
            device="cpu",
        )

        eval_set = None
        if df_val is not None:
            X_val, y_val = build_features(df_val, horizon, self.regime)
            eval_set = [(X_val, y_val)]
            # Only set when an eval_set exists: xgboost requires eval_set
            # to be present whenever early_stopping_rounds is configured.
            model_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self._model = xgb.XGBRegressor(**model_kwargs)

        # CRITICAL: no scaling here, and none should ever be added. Trees
        # split on raw feature thresholds and are invariant to any
        # monotone transform (standardising, log, min-max, ...) of a
        # feature - such a transform changes the threshold values a split
        # picks but never which rows end up on which side of it. So there
        # is nothing to fit on training data, and therefore nothing that
        # could leak from validation/test if it were fit there by mistake.
        # If you're adding a scaler here, stop - it does nothing for this
        # model and the CLAUDE.md rule about fitting only on TRAIN is not
        # a reason to add one, it's a reason not to.
        self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # best_iteration only exists on the fitted model when early
        # stopping actually ran (df_val given); otherwise all
        # n_estimators trees were built and there is no "best" subset.
        self.best_iteration = getattr(self._model, "best_iteration", None)

        return self

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")

        X, _y = build_features(df, horizon, self.regime)
        # Reorder/validate against the columns seen at fit time. Rows
        # dropped by build_features (not enough history, a gap wider than
        # the forward-fill limit, ...) simply have no entry in X and so no
        # prediction - that is correct, per the base class convention;
        # they are not filled in here.
        X = X[self._feature_cols]

        preds = pd.Series(self._model.predict(X), index=X.index, name="p_hat")
        # Clip at 0 below only - negative power is unphysical, but there
        # is no equivalent physical ceiling to clip at above (nameplate
        # can be exceeded briefly under real irradiance conditions).
        return preds.clip(lower=0)

    def feature_importance(self) -> dict:
        """Gain importance for every feature the model was fit with,
        sorted descending. Features never selected for a split (possible
        with colsample_bytree < 1) get 0.0 rather than being omitted, so
        the dict always covers the full feature set.
        """
        if self._model is None:
            raise RuntimeError("call fit() before feature_importance()")

        gains = self._model.get_booster().get_score(importance_type="gain")
        full = {col: float(gains.get(col, 0.0)) for col in self._feature_cols}
        return dict(sorted(full.items(), key=lambda kv: kv[1], reverse=True))
