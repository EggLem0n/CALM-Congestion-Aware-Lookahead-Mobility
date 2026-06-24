"""ConvLSTM congestion forecaster (PyTorch).

An encoder-forecaster ConvLSTM that predicts the *future* congestion cost map from
an observed window of scenario frames. This matches what the classical planner
queries (``solver.load_ai_congestion_cost`` / ``get_congestion_cost`` index future
timesteps) and the research plan (``docs/ConvLSTM_구현계획.md``).

Why forecasting, not same-step: the label ``y[t]`` is a deterministic Manhattan-kernel
blur of the occupancy channel (``metrics.build_additive_congestion_label_sequence``),
so a same-step ``x[t] -> y[t]`` map is reproducible by a single fixed convolution and
the recurrence would be pointless. Forecasting instead learns how congestion evolves.

Shapes
------
enc_in : (B, T_in, 5, H, W)   observed 5-channel frames
dec_in : (B, T_out, 5, H, W)  known exogenous future frames (occupancy channel zeroed)
output : (B, T_out, 1, H, W)  non-negative congestion cost maps
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn

__all__ = ["ConvLSTMCell", "CongestionConvLSTM"]

State = Tuple[torch.Tensor, torch.Tensor]  # (hidden, cell)


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell: the four LSTM gates are computed by one convolution
    over the concatenation of the input frame and the previous hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # "same" spatial size
            bias=bias,
        )

    def forward(self, x: torch.Tensor, state: State) -> State:
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch: int, height: int, width: int, device, dtype) -> State:
        zeros = torch.zeros(batch, self.hidden_dim, height, width, device=device, dtype=dtype)
        return zeros, zeros.clone()


class CongestionConvLSTM(nn.Module):
    """Encoder-forecaster ConvLSTM. A single stack of :class:`ConvLSTMCell` layers is
    rolled over the observation window (encode) and then over the future window
    (forecast); the top hidden state at each forecast step is projected to a
    one-channel congestion map through a 1x1 conv.

    The raw (signed) conv output is regressed directly against the congestion target.
    A terminal ReLU is intentionally NOT applied: the label is sparse (mostly zero),
    so a final ReLU is quickly driven negative everywhere, its gradient vanishes, and
    the whole network collapses to predicting all-zeros (the classic "dead ReLU").
    Non-negativity of the cost field is instead enforced at inference by clamping
    (see ``predict.py``).

    Defaults: observe t_in=60 frames (60 s at 1 Hz), forecast t_out=10 frames ahead."""

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dims: Sequence[int] = (64, 64),
        kernel_size: int = 3,
        t_in: int = 60,
        t_out: int = 10,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.kernel_size = int(kernel_size)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.num_layers = len(self.hidden_dims)

        cells: List[ConvLSTMCell] = []
        prev = self.in_channels
        for hidden in self.hidden_dims:
            cells.append(ConvLSTMCell(prev, hidden, self.kernel_size))
            prev = hidden
        self.cells = nn.ModuleList(cells)
        self.head = nn.Conv2d(self.hidden_dims[-1], 1, kernel_size=1)

    def _step(self, frame: torch.Tensor, states: List[State]) -> Tuple[torch.Tensor, List[State]]:
        """Advance every layer one timestep. Returns the top-layer hidden + new states."""
        x = frame
        new_states: List[State] = []
        for layer, cell in enumerate(self.cells):
            h, c = cell(x, states[layer])
            new_states.append((h, c))
            x = h  # hidden of this layer feeds the next
        return x, new_states

    def forward(self, enc_in: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        if enc_in.dim() != 5 or dec_in.dim() != 5:
            raise ValueError("enc_in/dec_in must be (B, T, C, H, W) tensors.")
        b, _, _, h, w = enc_in.shape
        states = [cell.init_state(b, h, w, enc_in.device, enc_in.dtype) for cell in self.cells]

        # Encode the observation window.
        for t in range(enc_in.shape[1]):
            _, states = self._step(enc_in[:, t], states)

        # Forecast: one congestion map per future step.
        outputs: List[torch.Tensor] = []
        for t in range(dec_in.shape[1]):
            top, states = self._step(dec_in[:, t], states)
            outputs.append(self.head(top))  # (B, 1, H, W), raw; clamped >=0 at inference
        return torch.stack(outputs, dim=1)  # (B, T_out, 1, H, W)
