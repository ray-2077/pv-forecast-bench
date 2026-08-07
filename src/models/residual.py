"""XGBoost residual-correction wrapper around a recurrent base model
(LSTMForecaster or CNNLSTMForecaster, src/models/lstm.py and
src/models/cnn_lstm.py).

The original plan was CNN-LSTM + XGBoost residual specifically. The seed
sweep (scripts/run_seed_sweep.py, results/seed_sweep_summary.csv) showed
CNN-LSTM <= LSTM in 8 of 9 array x horizon cells and the highest
seed-to-seed variance of the three base models - building the hybrid only
on CNN-LSTM would have confounded "does residual correction help" with
"CNN-LSTM happens to be the weaker base". This class instead wraps
whichever already-constructed recurrent forecaster it is given, so both
{lstm_residual, cnn_lstm_residual} can be run and compared - see
scripts/run_residual_dev.py and scripts/run_seed_sweep.py.

RESIDUAL FITTING SCHEME (CLAUDE.md rule 6) - read this before touching
either _oof_residuals or _val_residuals below:

residual_fit_split='oof' (the default, and the only scheme that should
ever produce a reported result): the residual XGBoost is fit on
OUT-OF-FOLD residuals from TRAIN_YEARS only, via an expanding window -
fit a fresh base model on years[:k], predict year k, pool the resulting
out-of-fold residuals across every k. VALIDATION is used only for early
stopping (of both the fold models and the final base model), never to fit
the residual stage. See _oof_residuals for the exact fold loop.

residual_fit_split='val': fits the residual XGBoost directly on the base
model's VALIDATION-split residuals - see _val_residuals. THIS IS LEAKED
and MUST NOT be used for a reported result: evaluating a model on the
same rows its correction stage was fit on is in-sample performance for
that stage, not merely optimistic - [L1.1] in Kapoor & Narayanan's
leakage taxonomy. It exists only so the inflation it causes can be
measured on purpose (a protocol-inflation table) instead of discovered by
accident. Constructing with residual_fit_split='val' raises a UserWarning
and sets self.leaked_by_design=True, which callers (scripts/run_residual_dev.py)
must record in the run's config.

CHANGE LOG (2026-07-27): this module originally only implemented what is
now residual_fit_split='val', because the original CLAUDE.md rule 6 said
to fit the residual stage on validation residuals (the reasoning being
that TRAINING residuals are contaminated - the base model has already
fitted them). That reasoning about training residuals was correct, but
fitting on validation and then evaluating on validation is not "slightly
optimistic", it is in-sample for the residual stage. Evidence:
results/leaked_lstm_residual_array11_h6_lagged_seed0.json - array11 h6
seed0 skill_vs_convex went from +0.2104 (plain LSTM) to +0.5447 (val-fit
residual stage), against a background where every genuine architectural
effect measured elsewhere in this project is 0.01-0.02. See CLAUDE.md
rule 6 for the full account.

See src/models/base.py for the alignment convention that predict() must
honour. This module does no leakage-relevant work beyond what is
described above and in fit()'s own docstring: the base model's own
fit()/predict() (src/models/recurrent_base.py) and build_features
(src/features/build.py) already enforce the issue-time cutoff for their
respective inputs.

torch is imported transitively through base_model, not directly here. No
plotting, no file writing.
"""

import warnings

import xgboost as xgb
import pandas as pd

from src.features.build import build_features
from src.models.base import BaseForecaster

RESIDUAL_FIT_SPLITS = ("oof", "val")

_LEAKED_WARNING = (
    "ResidualCorrected(residual_fit_split='val') fits the residual stage "
    "on the SAME validation rows used for early stopping and (later) "
    "evaluation - this is the LEAKED configuration (CLAUDE.md rule 6 "
    "CHANGE LOG): array11 h6 seed0 went from skill_vs_convex +0.2104 "
    "(plain LSTM) to +0.5447 (val-fit residual stage). Use this only to "
    "measure that inflation on purpose (a protocol-inflation table); "
    "never for a reported result. Use residual_fit_split='oof' (the "
    "default) instead."
)


def _clone_base_model(base_model):
    """Fresh, identically-configured instance of base_model's class - used
    to fit a fold's base model during out-of-fold residual construction
    (_oof_residuals) without mutating or reusing base_model itself. The
    caller's base_model is fit separately, exactly once, on all of
    TRAIN_YEARS (see ResidualCorrected.fit) - this clone exists purely to
    generate one fold's out-of-sample predictions.
    """
    cls = type(base_model)
    kwargs = dict(
        seed=base_model.seed,
        hidden_size=base_model.hidden_size,
        num_layers=base_model.num_layers,
        dropout=base_model.dropout,
        seq_len=base_model.seq_len,
        batch_size=base_model.batch_size,
        max_epochs=base_model.max_epochs,
        patience=base_model.patience,
        learning_rate=base_model.learning_rate,
        regime=base_model.regime,
        # base_model.device is already resolved to a torch.device; .type
        # gives back the 'cuda'/'cpu' string the constructor expects, and
        # resolves to the same device again (see RecurrentForecaster.__init__).
        device=base_model.device.type,
    )
    if hasattr(base_model, "n_filters"):
        kwargs["n_filters"] = base_model.n_filters
        kwargs["kernel_size"] = base_model.kernel_size
    return cls(**kwargs)


class ResidualCorrected(BaseForecaster):
    """Base recurrent forecaster + an XGBoost stage that predicts its
    residuals from the tabular feature set.

    base_model must already be constructed (an LSTMForecaster or
    CNNLSTMForecaster instance) with its own seed/regime/hyperparameters
    set - this class does not construct or configure it. `name` is
    derived from the base model's own name at construction time, so run
    ids come out as lstm_residual_* and cnn_lstm_residual_* for the
    default (correct) residual_fit_split='oof', and
    lstm_residual_leaked_* / cnn_lstm_residual_leaked_* for the
    deliberately-leaked residual_fit_split='val' - see module docstring.
    """

    def __init__(
        self,
        base_model,
        seed,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        residual_fit_split="oof",
    ):
        if residual_fit_split not in RESIDUAL_FIT_SPLITS:
            raise ValueError(
                f"residual_fit_split must be one of {RESIDUAL_FIT_SPLITS}, "
                f"got {residual_fit_split!r}"
            )

        self.base_model = base_model
        self.residual_fit_split = residual_fit_split
        self.leaked_by_design = residual_fit_split == "val"

        suffix = "" if residual_fit_split == "oof" else "_leaked"
        self.name = f"{base_model.name}_residual{suffix}"
        self.seed = seed
        # Smaller than XGBForecaster's standalone defaults (500/6 - see
        # src/models/xgb.py): this stage fits residuals of an
        # already-fitted recurrent model, which are lower-signal than the
        # raw target - most of the predictable structure has already been
        # removed by the base model, so a smaller tree budget and
        # shallower trees are less likely to fit noise in what is left.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self._residual_model = None
        self._feature_cols = None
        # Populated by fit(): the oof-mode pair when residual_fit_split ==
        # 'oof' (the default), the val-mode pair when == 'val'. The
        # inactive pair is left None so a run's config makes it obvious
        # which scheme actually produced a given result.
        self.n_oof_residuals = None
        self.oof_years = None
        self.n_val_residuals = None
        self.frac_val_usable = None

    def fit(self, df_train: pd.DataFrame, horizon: int, df_val: pd.DataFrame) -> "ResidualCorrected":
        """Fit the base model on ALL of df_train (val for early stopping,
        exactly as it is fitted standalone), then fit XGBoost to the base
        model's residuals using whichever scheme residual_fit_split
        selects.

        CLAUDE.md RULE 6 (corrected 2026-07-27): the residual stage is fit
        on OUT-OF-FOLD TRAINING residuals (residual_fit_split='oof', the
        default) via _oof_residuals - an expanding window over
        TRAIN_YEARS, never on validation-split residuals. VALIDATION is
        used ONLY for early stopping, both of the final base model here
        and of each fold's base model inside _oof_residuals - early
        stopping only selects which epoch's weights to keep, it does not
        feed validation's targets into anything the residual stage
        learns from, so this is not the leak rule 6 is about.

        KNOWN, ACCEPTED LIMITATION of the oof scheme: each fold's base
        model is fit on fewer years than the final base model fit here
        (e.g. the fold predicting 2013 was trained on 2011-2012 only,
        while the final base model trains on 2011-2013). A model trained
        on less data generally errs more, so out-of-fold residuals run
        slightly LARGER, and the residual XGBoost's training target is
        therefore slightly more pessimistic than the final base model's
        true error. That is the correct direction to err - it cannot
        inflate the reported skill score, only slightly understate the
        residual stage's potential.

        residual_fit_split='val' (see _val_residuals) reproduces the
        original, LEAKED scheme: fits directly on validation residuals,
        which are then evaluated on the same validation split. Raises a
        UserWarning when used and MUST NOT be used for a reported result -
        see module docstring and CLAUDE.md rule 6's CHANGE LOG for why,
        and for the array11 h6 seed0 evidence (+0.2104 -> +0.5447).
        """
        self.base_model.fit(df_train, horizon, df_val)

        if self.residual_fit_split == "val":
            warnings.warn(_LEAKED_WARNING, UserWarning)
            X_fit, residual = self._val_residuals(df_val, horizon)
        else:
            X_fit, residual = self._oof_residuals(df_train, horizon, df_val)

        self._feature_cols = list(X_fit.columns)

        model_kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.seed,
            # device="cpu" - same reproducibility rationale as
            # XGBForecaster (src/models/xgb.py): this data scale gains
            # nothing from the GPU, and CPU (hist) tree construction is
            # deterministic given a fixed seed while GPU histogram
            # building is not.
            device="cpu",
        )
        self._residual_model = xgb.XGBRegressor(**model_kwargs)
        self._residual_model.fit(X_fit, residual)

        return self

    def _oof_residuals(self, df_train, horizon, df_val):
        """Expanding-window out-of-fold residuals within the years actually
        present in df_train - the corrected CLAUDE.md rule 6 scheme.

        fold_years is read directly off df_train's own index, NOT
        re-filtered against src.data.splits.TRAIN_YEARS: df_train is
        already exactly the training window the caller chose (ordinarily
        TRAIN_YEARS, but deliberately variable for a training-length
        ablation - see src/data/splits.py's TRAIN_YEARS comment). Filtering
        against the default constant here would silently discard any extra
        years such an ablation adds, making the fold count invariant to
        the very thing being ablated.

        Fold k (k = 1 .. len(fold_years)-1): fit a FRESH base model
        (_clone_base_model - never self.base_model, which is fit
        separately on the full training set) on fold_years[:k], then
        predict on fold_years[:k+1] (the fit years PLUS the held-out
        year k) so the held-out year's sequence windows can see genuine
        PAST history from the prior fold years - exactly what a deployed
        model would have available at issue time - but only the held-out
        year's predictions are kept for the pooled out-of-fold set; the
        fit years' own predictions on themselves would be in-sample, not
        out-of-fold, and are discarded.

        There is no fold for fold_years[0] (2011 for the default
        TRAIN_YEARS): expanding from an empty prior window is undefined,
        so the first training year never contributes an out-of-fold
        residual. With TRAIN_YEARS = (2011, 2012, 2013) this yields
        exactly 2 folds (out-of-fold predictions for 2012, then 2013); a
        5-year window (2009-2013) yields 4 folds.

        VALIDATION (df_val) is passed to each fold's fit() for early
        stopping ONLY, identically to how the final base model uses it -
        see fit()'s docstring for why that is not the rule-6 leak.

        Each fold's base model sees LESS training data than the final
        model that ends up in self.base_model (which trains on all of
        TRAIN_YEARS at once), so these out-of-fold residuals run slightly
        larger than the final base model's true error - a conservative,
        accepted bias (see fit()'s docstring).

        Returns (X_oof, residual_oof): the pooled feature matrix and
        pooled residual target across all folds. Also sets
        self.n_oof_residuals (len(X_oof)) and self.oof_years (the list of
        held-out years that actually contributed, in order).
        """
        idx_years = df_train.index.year
        fold_years = sorted({int(y) for y in idx_years})

        X_parts = []
        residual_parts = []
        years_contributed = []

        for k in range(1, len(fold_years)):
            fit_years = fold_years[:k]
            predict_year = fold_years[k]
            context_years = fold_years[: k + 1]

            fold_train = df_train[idx_years.isin(fit_years)]
            fold_context = df_train[idx_years.isin(context_years)]

            fold_model = _clone_base_model(self.base_model)
            fold_model.fit(fold_train, horizon, df_val)
            fold_pred = fold_model.predict(fold_context, horizon)

            # Keep only the held-out year's predictions - fold_context
            # also contains fit_years, which fold_model has already seen
            # in training, and predictions there would be in-sample, not
            # out-of-fold.
            fold_pred = fold_pred[fold_pred.index.year == predict_year]

            X_fold, _y_fold = build_features(fold_context, horizon, self.base_model.regime)
            common_idx = fold_pred.index.intersection(X_fold.index)

            y_true_fold = fold_context.loc[common_idx, "Active_Power"]
            residual_fold = y_true_fold - fold_pred.loc[common_idx]

            X_parts.append(X_fold.loc[common_idx])
            residual_parts.append(residual_fold)
            years_contributed.append(predict_year)

        X_oof = pd.concat(X_parts, axis=0)
        residual_oof = pd.concat(residual_parts, axis=0)

        self.n_oof_residuals = len(X_oof)
        self.oof_years = years_contributed

        return X_oof, residual_oof

    def _val_residuals(self, df_val, horizon):
        """LEAKED (see module docstring / CLAUDE.md rule 6 CHANGE LOG):
        residuals of the ALREADY-FITTED self.base_model, computed directly
        on df_val, with no held-out separation from the rows this model
        will later be evaluated on. Only reachable via
        residual_fit_split='val', which warns on construction. Sets
        self.n_val_residuals and self.frac_val_usable (that count as a
        fraction of len(df_val)).
        """
        base_pred_val = self.base_model.predict(df_val, horizon)

        X_val, _y_val = build_features(df_val, horizon, self.base_model.regime)

        common_idx = base_pred_val.index.intersection(X_val.index)
        self.n_val_residuals = len(common_idx)
        self.frac_val_usable = len(common_idx) / len(df_val)

        y_true_val = df_val.loc[common_idx, "Active_Power"]
        residual = y_true_val - base_pred_val.loc[common_idx]

        return X_val.loc[common_idx], residual

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """Base prediction + residual correction, on the intersection of
        timestamps where both the base model and the tabular feature
        matrix produced output. Clipped at 0 below only, same rationale as
        XGBForecaster.predict and RecurrentForecaster.predict: negative
        power is unphysical, no equivalent ceiling exists.
        """
        if self._residual_model is None:
            raise RuntimeError("call fit() before predict()")

        base_pred = self.base_model.predict(df, horizon)

        X, _y = build_features(df, horizon, self.base_model.regime)
        X = X[self._feature_cols]

        common_idx = base_pred.index.intersection(X.index)
        residual_pred = self._residual_model.predict(X.loc[common_idx])

        combined = base_pred.loc[common_idx].to_numpy() + residual_pred
        preds = pd.Series(combined, index=common_idx, name="p_hat")
        return preds.clip(lower=0)

    def residual_importance(self) -> dict:
        """Gain importance for every feature the residual XGBoost was fit
        with, sorted descending. Features never selected for a split get
        0.0 rather than being omitted, so the dict always covers the full
        feature set - same convention as XGBForecaster.feature_importance
        (src/models/xgb.py). This answers RQ2 directly: which features
        carry signal the recurrent base model missed.
        """
        if self._residual_model is None:
            raise RuntimeError("call fit() before residual_importance()")

        gains = self._residual_model.get_booster().get_score(importance_type="gain")
        full = {col: float(gains.get(col, 0.0)) for col in self._feature_cols}
        return dict(sorted(full.items(), key=lambda kv: kv[1], reverse=True))
