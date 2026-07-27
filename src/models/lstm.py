"""LSTM forecaster: a recurrent model over the sequence channels built by
src/features/sequences.py, with the deterministic static block (solar
geometry, clear-sky power, calendar - Category A in src/features/build.py)
concatenated in at the head.

See src/models/base.py for the alignment convention that predict() must
honour. This model does no leakage-relevant work of its own beyond scaling
(fit on TRAINING sequences/statics/target only, per CLAUDE.md rule 3) -
src/features/sequences.py.build_sequences already enforces the
issue-time cutoff and is checked independently by
assert_no_leakage_sequences.

torch/numpy/pandas only. No plotting, no file writing.
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.scaling import Scaler
from src.features.sequences import build_sequences, static_feature_names
from src.models.base import BaseForecaster


class _LSTMNet(nn.Module):
    """LSTM over the sequence input -> final hidden state -> concatenate
    static features -> two-layer MLP head -> scalar.

    dropout is only meaningful between LSTM layers, so it is passed to
    nn.LSTM as 0.0 whenever num_layers == 1 - see LSTMForecaster.__init__
    for why (PyTorch itself would otherwise warn and silently ignore it).
    """

    def __init__(self, n_channels, n_static, hidden_size, num_layers, dropout):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_static, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_seq, x_static):
        _out, (h_n, _c_n) = self.lstm(x_seq)
        final_hidden = h_n[-1]  # last layer's hidden state at the last timestep
        combined = torch.cat([final_hidden, x_static], dim=1)
        return self.head(combined).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    """Small recurrent forecaster for one horizon and regime.

    Deliberately small (default hidden_size=64, num_layers=1): after
    daylight filtering there are roughly 4,400 usable training rows per
    array-year (see CLAUDE.md's data window - three train years), which is
    little enough data that anything bigger overfits before early stopping
    can even establish a val-loss trend. Hyperparameters are stored as
    given, never tuned inside this class.
    """

    name = "lstm"

    def __init__(
        self,
        seed,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        seq_len=24,
        batch_size=256,
        max_epochs=100,
        patience=10,
        learning_rate=1e-3,
        regime="lagged",
        device="cuda",
    ):
        self.seed = seed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.regime = regime

        # Fall back to CPU cleanly if CUDA was requested but is not
        # available on this machine.
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

        self._model = None
        self._seq_scaler = None
        self._static_scaler = None
        self._target_scaler = None
        self._static_cols = static_feature_names(regime)

        self.history = {"train_loss": [], "val_rmse": []}
        self.best_epoch = None
        self.epochs_run = 0
        self.full_determinism_achieved = None

    def _configure_determinism(self):
        """Set torch seeds and enable deterministic algorithms where
        achievable. torch.nn.LSTM has no deterministic CUDA
        implementation for its backward pass (see PyTorch's
        reproducibility docs), so full determinism is only achievable on
        CPU: there, use_deterministic_algorithms(True) is set strictly and
        will raise if any non-deterministic op is hit. On CUDA, cudnn is
        set to its most-reproducible mode (deterministic=True,
        benchmark=False) as a best effort, and
        use_deterministic_algorithms is set with warn_only=True so
        training does not crash on the known-nondeterministic LSTM
        backward - self.full_determinism_achieved records this honestly
        rather than claiming a guarantee CUDA cannot make.
        """
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            self.full_determinism_achieved = False
        else:
            try:
                torch.use_deterministic_algorithms(True)
                self.full_determinism_achieved = True
            except RuntimeError:
                # Some op in this model has no deterministic CPU
                # implementation either (unexpected but possible in a
                # torch version this project hasn't hit yet) - fall back
                # to warn_only rather than crash the fit, and record
                # honestly that full determinism was not achieved.
                torch.use_deterministic_algorithms(True, warn_only=True)
                self.full_determinism_achieved = False

    def _scale_sequences(self, X_seq, scaler, fit):
        n, seq_len, n_channels = X_seq.shape
        flat = X_seq.reshape(-1, n_channels)
        flat = scaler.fit_transform(flat) if fit else scaler.transform(flat)
        return flat.reshape(n, seq_len, n_channels)

    def _to_tensors(self, X_seq, X_static, y=None):
        seq_t = torch.as_tensor(X_seq, dtype=torch.float32)
        static_t = torch.as_tensor(X_static, dtype=torch.float32)
        if y is None:
            return seq_t, static_t
        y_t = torch.as_tensor(y, dtype=torch.float32)
        return seq_t, static_t, y_t

    def fit(self, df_train: pd.DataFrame, horizon: int, df_val: pd.DataFrame) -> "LSTMForecaster":
        self._configure_determinism()

        X_seq_train, X_static_train, y_train, _idx_train = build_sequences(
            df_train, horizon, self.regime, self.seq_len
        )
        X_seq_val, X_static_val, y_val, _idx_val = build_sequences(
            df_val, horizon, self.regime, self.seq_len
        )

        # Scalers and target scaling: fit on TRAINING only (CLAUDE.md rule
        # 3), never refit here on val. Raw kW targets range roughly 0-7,
        # which is not itself a problem, but scaling the target too keeps
        # its magnitude comparable to the (also scaled) inputs early in
        # training, before the network has learned any sensible output
        # scale of its own - this stabilises the first few epochs rather
        # than changing what the model can eventually represent.
        self._seq_scaler = Scaler()
        self._static_scaler = Scaler()
        self._target_scaler = Scaler()

        X_seq_train = self._scale_sequences(X_seq_train, self._seq_scaler, fit=True)
        X_static_train = self._static_scaler.fit_transform(X_static_train)
        y_train_scaled = self._target_scaler.fit_transform(y_train)

        X_seq_val = self._scale_sequences(X_seq_val, self._seq_scaler, fit=False)
        X_static_val = self._static_scaler.transform(X_static_val)

        seq_train_t, static_train_t, y_train_t = self._to_tensors(
            X_seq_train, X_static_train, y_train_scaled
        )
        seq_val_t, static_val_t = self._to_tensors(X_seq_val, X_static_val)

        train_dataset = TensorDataset(seq_train_t, static_train_t, y_train_t)
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(self.seed)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, generator=shuffle_gen
        )

        n_channels = X_seq_train.shape[-1]
        n_static = X_static_train.shape[-1]
        self._model = _LSTMNet(
            n_channels, n_static, self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        seq_val_t = seq_val_t.to(self.device)
        static_val_t = static_val_t.to(self.device)

        self.history = {"train_loss": [], "val_rmse": []}
        best_val_rmse = float("inf")
        best_epoch = 0
        best_state = None
        epochs_since_improvement = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            epoch_losses = []
            for seq_batch, static_batch, y_batch in train_loader:
                seq_batch = seq_batch.to(self.device)
                static_batch = static_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self._model(seq_batch, static_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses))

            self._model.eval()
            with torch.no_grad():
                val_pred_scaled = self._model(seq_val_t, static_val_t).cpu().numpy()
            val_pred = self._target_scaler.inverse_transform(val_pred_scaled)
            val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))

            self.history["train_loss"].append(train_loss)
            self.history["val_rmse"].append(val_rmse)
            self.epochs_run = epoch + 1

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(self._model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= self.patience:
                    break

        self._model.load_state_dict(best_state)
        self.best_epoch = best_epoch

        return self

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")

        X_seq, X_static, _y, index = build_sequences(df, horizon, self.regime, self.seq_len)

        X_seq = self._scale_sequences(X_seq, self._seq_scaler, fit=False)
        X_static = self._static_scaler.transform(X_static)

        seq_t, static_t = self._to_tensors(X_seq, X_static)
        seq_t = seq_t.to(self.device)
        static_t = static_t.to(self.device)

        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(seq_t, static_t).cpu().numpy()
        pred = self._target_scaler.inverse_transform(pred_scaled)

        preds = pd.Series(pred, index=index, name="p_hat")
        # Clip at 0 below only - same rationale as XGBForecaster.predict:
        # negative power is unphysical, no equivalent ceiling exists.
        return preds.clip(lower=0)
