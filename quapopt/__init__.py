# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


AVAILABLE_SIMULATORS = []

try:
    import cupy

    AVAILABLE_SIMULATORS += ["cupy"]
except (ImportError, ModuleNotFoundError):
    pass

try:
    import cupy.cuda

    if cupy.cuda.is_available():
        AVAILABLE_SIMULATORS += ["cuda"]
except (ImportError, ModuleNotFoundError):
    pass

try:
    import torch

    if torch.cuda.is_available():
        AVAILABLE_SIMULATORS += ["torch"]

except (ImportError, ModuleNotFoundError):
    pass


import pandas as pd

pd.set_option("display.max_columns", None)


if __name__ == "__main__":
    try:
        import cupy
        import numba

        print("NUMBA detects CUDA:", numba.cuda.is_available())
        print("CUPY detects CUDA:", cupy.cuda.is_available())

        if not numba.cuda.is_available() or not cupy.cuda.is_available():
            print(
                "Warning: CUDA is not available. Some features may not work as expected."
            )
    except Exception as e:
        print("Error importing numba or cupy:", e)

    try:
        import torch

        print("Torch detects CUDA:", torch.cuda.is_available())
    except Exception as e:
        pass

    print("\nAvailable simulators:", AVAILABLE_SIMULATORS)
