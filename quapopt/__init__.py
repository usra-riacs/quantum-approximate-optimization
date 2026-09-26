# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import pandas as pd
pd.set_option('display.max_columns', None)

AVAILABLE_SIMULATORS = []

try:
    import cupy.cuda
    if cupy.cuda.is_available():
        AVAILABLE_SIMULATORS += ['cupy']
except Exception:
    pass

try:
    import numba.cuda
    if numba.cuda.is_available():
        AVAILABLE_SIMULATORS += ['cuda']
except (ImportError, ModuleNotFoundError):
    pass

try:
    import torch.cuda
    if torch.cuda.is_available():
        AVAILABLE_SIMULATORS += ['torch']

except(ImportError, ModuleNotFoundError):
    pass


try:
    from cuquantum.tensornet.experimental import NetworkState  # noqa: F401
    AVAILABLE_SIMULATORS += ['cuquantum']
except (ImportError, ModuleNotFoundError):
    pass



try:
    from qiskit_aer.backends.aer_simulator import AerSimulator
    _available_devices_qiskit = [x.lower() for x in AerSimulator().available_devices()]
    if 'cpu' in _available_devices_qiskit:
        AVAILABLE_SIMULATORS += ['aer-cpu']
    if 'gpu' in _available_devices_qiskit:
        AVAILABLE_SIMULATORS += ['aer-gpu']

except(ImportError, ModuleNotFoundError):
    pass

from quapopt.precision import Precision, get_default_precision, set_default_precision, resolve_precision  # noqa: E402
