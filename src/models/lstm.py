"""LSTM forecaster: a recurrent model over the sequence channels built by
src/features/sequences.py, with the deterministic static block (solar
geometry, clear-sky power, calendar - Category A in src/features/build.py)
concatenated in at the head.

Everything except the network architecture (determinism config, scaling,
the training loop, early stopping, best-weight restore, and predict()) is
shared with src/models/cnn_lstm.py via src/models/recurrent_base.py - see
that module's docstring. This file defines only the LSTM network itself
and the hyperparameters specific to it.

torch/numpy/pandas only. No plotting, no file writing.
"""

import torch
import torch.nn as nn

from src.models.recurrent_base import RecurrentForecaster


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


class LSTMForecaster(RecurrentForecaster):
    """Small recurrent forecaster for one horizon and regime.

    See RecurrentForecaster (src/models/recurrent_base.py) for fit(),
    predict(), scaling, and early-stopping - identical to before this was
    pulled out into a shared base for src/models/cnn_lstm.py to reuse.
    """

    name = "lstm"

    def _build_network(self, n_channels, n_static):
        return _LSTMNet(n_channels, n_static, self.hidden_size, self.num_layers, self.dropout)
