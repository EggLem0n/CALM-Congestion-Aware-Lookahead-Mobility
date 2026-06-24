"""macpf.convjam: ConvLSTM congestion model (definition, training, inference, viz).

Tolerate duplicate OpenMP runtimes before NumPy / PyTorch / matplotlib load.
On Windows the MKL build of NumPy ships Intel's ``libiomp5md.dll`` while PyTorch
ships LLVM's ``libomp.dll``; loading both otherwise aborts the process with
"OMP: Error #15". This must be set before those libraries are imported, so it
lives in this package __init__ (imported before any ``macpf.convjam.*`` module
pulls in NumPy/torch).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
