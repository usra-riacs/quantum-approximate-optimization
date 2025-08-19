# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

from enum import Enum

class OptimizerType(Enum):
    optuna = 'OPTUNA'
    scipy = 'SCIPY'
    custom = 'CUSTOM'


class ParametersBoundType(Enum):
    SET = "SET"
    RANGE = "RANGE"
    CONSTANT = "CONSTANT"

