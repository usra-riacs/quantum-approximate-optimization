# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import copy
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from quapopt.optimization import OptimizationResult
from quapopt.optimization.parameter_setting import ParametersBoundType
from quapopt.optimization.parameter_setting import OptimizerType
from quapopt.optimization.parameter_setting.CustomOptimizer import CustomOptimizer


class AdaptiveOptimizer(CustomOptimizer):

    def __init__(self,
                 argument_names: List[str] = None,
                 parameter_bounds: List[Tuple[ParametersBoundType, Tuple[Any, ...]]] = None,
                 optimizer_name: str = "UnnamedAdaptiveOptimizer"):


        super().__init__(argument_names=argument_names,
                         parameter_bounds=parameter_bounds,
                         optimizer_name=optimizer_name
                         )



    @property
    def optimizer_name(self):
        return self._optimizer_name
    @optimizer_name.setter
    def optimizer_name(self, value: str):
        self._optimizer_name = value

    @property
    def optimizer_type(self):
        return self._optimizer_type

    @property
    def search_space(self):
        return self._search_space
    @search_space.setter
    def search_space(self, value):
        self._search_space = value

    @property
    def argument_names(self):
        return self._argument_names

    @property
    def parameter_bounds(self):
        return self._parameter_bounds

    def copy(self):
        return copy.deepcopy(self)

