# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

AVAILABLE_SIMULATORS = []

try:
    import cupy
    if cupy.cuda.is_available():
        AVAILABLE_SIMULATORS += ['cupy']
except (ImportError, ModuleNotFoundError):
    pass

try:
    import numba.cuda
    if numba.cuda.is_available():
        AVAILABLE_SIMULATORS += ['cuda']
except (ImportError, ModuleNotFoundError):
    pass

try:
    import torch
    if torch.cuda.is_available():
        AVAILABLE_SIMULATORS += ['torch']
except( ImportError, ModuleNotFoundError):
    pass


try:
    import numba, cupy, torch

    print("NUMBA detects CUDA:", numba.cuda.is_available())
    print("CUPY detects CUDA:", cupy.cuda.is_available())
    print("PyTorch detects CUDA:", torch.cuda.is_available())

    if not numba.cuda.is_available() or not cupy.cuda.is_available():
        print("Warning: CUDA is not available. Some features may not work as expected.")
except(Exception) as e:
    print("Error importing numba or cupy or pytorch:", e)

print("\nAvailable simulators:", AVAILABLE_SIMULATORS)

