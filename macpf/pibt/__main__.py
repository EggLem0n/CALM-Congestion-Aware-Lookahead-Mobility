"""CLI entry point: python -m macpf.pibt"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401  # Load torch before numpy-dependent runner imports on Windows.

from .runner import main

if __name__ == "__main__":
    main()
