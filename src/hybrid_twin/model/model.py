import torch
import torch.nn as nn
from pathlib import Path
from hybrid_twin.config.loader import load_config

cfg = load_config(Path(__file__).resolve().parent.parent.parent.parent / "config.yaml")
_m  = cfg["model"]


class BiasLearner(nn.Module):
    """
    Sequence-to-sequence LSTM bias learner.

    Input  : x  of shape (batch_size, seq_len, input_size)
    Output : delta_hat of shape (batch_size, seq_len, 1)

    The FC head is applied to EVERY LSTM hidden state (time-distributed),
    so the model produces a correction delta_hat(t) at every timestep,
    not only at the final one:

        sigma_hybrid(t) = sigma_sim(t) + delta_hat(t)     for all t in [0, T-1]
    """

    def __init__(
        self,
        input_size  = _m["INPUT_SIZE"],
        hidden_size = _m["HIDDEN_SIZE"],
        num_layers  = _m["NUM_LAYERS"],
        dropout     = _m["DROPOUT"],
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)        # (B, T, H)
        delta_pred  = self.fc(lstm_out)   # (B, T, 1)
        return delta_pred