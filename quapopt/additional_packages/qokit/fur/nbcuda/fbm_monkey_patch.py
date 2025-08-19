# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 
# import all types from typing

#this is set to 1 by default.
#it causes numba to spit out underutilization warnings
_GLOBAL_GRID_SIZE = 1
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)