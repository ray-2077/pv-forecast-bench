"""CNN-LSTM forecaster: a 1D convolution over the time axis of the
sequence input (src/features/sequences.py), feeding an LSTM whose final
hidden state is concatenated with the deterministic static block (solar
geometry, clear-sky power, calendar - Category A in src/features/build.py)
at the head. Intended to let the network learn short local patterns (e.g.
a few-hour ramp shape) before the LSTM aggregates them over the full
window, which a bare LSTM must do timestep-by-timestep.

Everything except the network architecture (determinism config, scaling,
the training loop, early stopping, best-weight restore, and predict()) is
shared with src/models/lstm.py via src/models/recurrent_base.py - see
that module's docstring. This file defines only the CNN-LSTM network
itself and the hyperparameters specific to it (n_filters, kernel_size).

torch/numpy/pandas only. No plotting, no file writing.
"""

import torch
import torch.nn as nn

from src.models.recurrent_base import RecurrentForecaster


class _CNNLSTMNet(nn.Module):
    """Conv1d over the TIME axis -> ReLU -> LSTM over the convolved
    sequence -> final hidden state -> concatenate static features ->
    two-layer MLP head -> scalar.

    padding='same' is safe here and does NOT leak, because the entire
    sequence window lies at or before the issue time t-h by construction
    (see src/features/sequences.py). There is no future information
    inside the window for a non-causal kernel to reach.

    dropout is only meaningful between LSTM layers, so it is passed to
    nn.LSTM as 0.0 whenever num_layers == 1, same rationale as
    src/models/lstm.py's _LSTMNet.
    """

    def __init__(self, n_channels, n_static, n_filters, kernel_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding="same",
        )
        self.relu = nn.ReLU()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_filters,
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
        # Conv1d expects (batch, channels, time); the sequence builder
        # hands us (batch, time, channels), so transpose there and back.
        x = x_seq.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)

        _out, (h_n, _c_n) = self.lstm(x)
        final_hidden = h_n[-1]  # last layer's hidden state at the last timestep
        combined = torch.cat([final_hidden, x_static], dim=1)
        return self.head(combined).squeeze(-1)


class CNNLSTMForecaster(RecurrentForecaster):
    """CNN-LSTM forecaster for one horizon and regime. Same interface and
    training pipeline as LSTMForecaster (see RecurrentForecaster in
    src/models/recurrent_base.py) - the only difference is the network
    architecture built by _build_network.
    """

    name = "cnn_lstm"

    def __init__(
        self,
        seed,
        n_filters=32,
        kernel_size=3,
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
        super().__init__(
            seed=seed,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            regime=regime,
            device=device,
        )
        self.n_filters = n_filters
        self.kernel_size = kernel_size

    def _build_network(self, n_channels, n_static):
        return _CNNLSTMNet(
            n_channels,
            n_static,
            self.n_filters,
            self.kernel_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
        )
