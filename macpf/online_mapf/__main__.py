"""CLI entry point: python -m macpf.online_mapf"""
from __future__ import annotations

import os

# Import torch before the rest of online_mapf pulls in numerical libraries.
# On some Windows conda setups, loading OpenMP-linked libraries first can make
# PyTorch fail while resolving torch/lib/shm.dll.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401

from .runner import main

if __name__ == "__main__":
    main()
