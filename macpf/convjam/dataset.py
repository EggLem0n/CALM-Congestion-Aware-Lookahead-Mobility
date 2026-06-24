"""Windowed forecasting dataset for the ConvLSTM congestion model.

Turns the per-episode shards from ``macpf.generate_heatmap`` (x: (T,5,H,W),
y: (T,1,H,W)) into sliding ``(encoder window, decoder exogenous window, future
label)`` samples. See ``docs/ConvLSTM_구현계획.md``.

For each window starting at ``i``:
    enc_in = x[i : i+t_in]                         observed 5-channel frames
    dec_in = x[i+t_in : i+t_in+t_out]  (occ -> 0)  known future exogenous channels
    target = y[i+t_in : i+t_in+t_out] / label_norm future congestion to predict

The occupancy channel is zeroed in ``dec_in`` because future occupancy is exactly
what congestion encodes — it is unknown at planning time; obstacle / start / task
markers, by contrast, are known ahead and are kept.
"""
from __future__ import annotations

import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from numpy.lib import format as npformat
from torch.utils.data import Dataset

from macpf.features import load_episode

__all__ = ["CongestionWindowDataset", "compute_label_norm"]

OCCUPANCY_CHANNEL = 0  # amr_occupancy — unknown in the future, zeroed for the decoder


def _episode_length(path: Path) -> int:
    """Read T from the shard's ``x`` array header without loading the whole array."""
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("x.npy") as member:
                major, _ = npformat.read_magic(member)
                if major == 1:
                    shape, _, _ = npformat.read_array_header_1_0(member)
                else:
                    shape, _, _ = npformat.read_array_header_2_0(member)
        return int(shape[0])
    except Exception:  # noqa: BLE001 - fall back to a full load if the header trick fails
        x, _ = load_episode(path)
        return int(x.shape[0])


class _EpisodeCache:
    """Tiny LRU cache of loaded (x, y) arrays so workers don't reload ~290MB shards
    for every window. Windows are grouped by episode, so a small cache is enough."""

    def __init__(self, maxsize: int = 2):
        self.maxsize = max(1, maxsize)
        self._cache: "OrderedDict[Path, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    def get(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        # Keep x as uint8 in the cache (4x smaller than float32) so many episodes
        # stay resident in RAM; __getitem__ casts only the small window slice.
        with np.load(path) as data:
            arrays = (np.asarray(data["x"]), np.asarray(data["y"], dtype=np.float32))
        self._cache[path] = arrays
        self._cache.move_to_end(path)
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return arrays


class CongestionWindowDataset(Dataset):
    def __init__(
        self,
        episode_paths: Sequence[Path],
        t_in: int = 60,
        t_out: int = 10,
        stride: int = 8,
        label_norm: float = 1.0,
        cache_size: int = 16,
    ):
        self.episode_paths = [Path(p) for p in episode_paths]
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.stride = max(1, int(stride))
        self.label_norm = float(label_norm) if label_norm else 1.0
        self._cache = _EpisodeCache(cache_size)

        window = self.t_in + self.t_out
        self.index: List[Tuple[int, int]] = []
        for ep_idx, path in enumerate(self.episode_paths):
            length = _episode_length(path)
            if length < window:
                continue
            n_windows = (length - window) // self.stride + 1
            for w in range(n_windows):
                self.index.append((ep_idx, w * self.stride))

        if not self.index:
            raise ValueError(
                f"No windows of length t_in+t_out={window} fit in any episode under "
                f"{[str(p) for p in self.episode_paths][:3]}... "
                "Reduce --t-in/--t-out or generate longer episodes (--seconds)."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ep_idx, start = self.index[idx]
        x, y = self._cache.get(self.episode_paths[ep_idx])

        enc_end = start + self.t_in
        dec_end = enc_end + self.t_out

        enc_in = x[start:enc_end].astype(np.float32)            # (t_in, 5, H, W)
        dec_in = x[enc_end:dec_end].astype(np.float32)          # (t_out, 5, H, W)
        dec_in[:, OCCUPANCY_CHANNEL] = 0.0                      # future occupancy is unknown
        target = (y[enc_end:dec_end] / self.label_norm).astype(np.float32)  # (t_out,1,H,W)

        return (
            torch.from_numpy(np.ascontiguousarray(enc_in)),
            torch.from_numpy(np.ascontiguousarray(dec_in)),
            torch.from_numpy(np.ascontiguousarray(target)),
        )


def compute_label_norm(episode_paths: Sequence[Path], max_episodes: int = 8) -> float:
    """Estimate a normalization constant (max label value) from a sample of episodes.

    Run ``python -m macpf.features --normalize`` for the exact dataset max; this
    samples a few shards so training can self-configure when ``--label-norm`` is 0.
    """
    paths = [Path(p) for p in episode_paths][:max_episodes]
    label_max = 0.0
    for path in paths:
        _, y = load_episode(path)
        label_max = max(label_max, float(y.max()))
    return label_max if label_max > 0 else 1.0
