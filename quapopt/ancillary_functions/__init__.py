from quapopt.additional_packages.ancillary_functions_usra.ancillary_functions import *
from quapopt.additional_packages.ancillary_functions_usra.efficient_math import *

try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
    import numpy as cp
import numpy as np

def convert_cupy_numpy_array(array:np.ndarray|cp.ndarray,
                             output_backend:str):
    from quapopt import AVAILABLE_SIMULATORS
    if 'cupy' not in AVAILABLE_SIMULATORS:
        return array
    if output_backend == 'numpy':
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        else:
            return array
    elif output_backend == 'cupy':
        if isinstance(array, np.ndarray):
            return cp.asarray(array)
        else:
            return array
    else:
        raise ValueError(f'output_backend should be either "numpy" or "cupy", not {output_backend}')

