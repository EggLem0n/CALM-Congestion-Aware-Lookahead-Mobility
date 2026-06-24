"""Live ConvLSTM inference wrapper for the online planner.

Loads the trained checkpoint once and exposes `predict(enc, dec)` returning a
`(t_out, H, W)` non-negative congestion-cost field. The transform mirrors
`macpf.convjam.predict` (model output * label_norm, clipped at 0) so the online
loop produces the same cost field the offline pipeline writes to disk -- only
recomputed every second from the live state instead of once up front.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class CongestionPredictor:
    """Trained CongestionConvLSTM held in memory for repeated 1 Hz inference."""

    def __init__(self, model_path: Path, device: str = "auto"):
        import torch

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Train one with "
                "`python -m macpf.convjam.train` or `python -m macpf.convjam.train_simvp`."
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(model_path, map_location="cpu")
        cfg = checkpoint["model_cfg"]
        model_type = str(checkpoint.get("model_type", "convlstm")).lower()
        self.t_in = int(cfg["t_in"])
        self.t_out = int(cfg["t_out"])
        self.label_norm = float(checkpoint.get("label_norm", 1.0)) or 1.0

        if model_type == "simvp":
            from macpf.convjam.simvp import CongestionSimVP

            model = CongestionSimVP(**cfg).to(device)
        elif model_type in {"simvp_openstl", "openstl_simvp"}:
            from macpf.convjam.simvp import OpenSTLCongestionSimVP

            model = OpenSTLCongestionSimVP(**cfg).to(device)
        else:
            from macpf.convjam.convlstm import CongestionConvLSTM

            model = CongestionConvLSTM(**cfg).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        self._torch = torch
        self.device = device
        self.model_type = model_type
        self.model = model

    def predict(self, enc_in: np.ndarray, dec_in: np.ndarray) -> np.ndarray:
        """Forecast future congestion.

        enc_in : (t_in, 5, H, W) float32   observed window
        dec_in : (t_out, 5, H, W) float32  known exogenous future (occupancy zeroed)
        returns: (t_out, H, W) float32     non-negative congestion cost per future step
        """
        torch = self._torch
        enc_t = torch.from_numpy(np.ascontiguousarray(enc_in[None])).to(self.device)
        dec_t = torch.from_numpy(np.ascontiguousarray(dec_in[None])).to(self.device)
        with torch.no_grad():
            pred = self.model(enc_t, dec_t)  # (1, t_out, 1, H, W)
        pred = pred.squeeze(0).squeeze(1).cpu().numpy() * self.label_norm  # (t_out, H, W)
        return np.clip(pred, 0.0, None).astype(np.float32)
