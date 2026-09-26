# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
from quapopt.ancillary_functions.presets.general import *

try:
    from quapopt.ancillary_functions.presets.qiskit import *
except(ModuleNotFoundError,ImportError):
    pass