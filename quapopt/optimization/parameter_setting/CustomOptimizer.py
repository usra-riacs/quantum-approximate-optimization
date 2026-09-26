# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import copy
from typing import List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from quapopt.optimization import OptimizationResult
from quapopt.optimization.parameter_setting import ParametersBoundType
from quapopt.optimization.parameter_setting import OptimizerType


class CustomOptimizer:

    def __init__(self,
                 argument_names: List[str] = None,
                 parameter_bounds: List[Tuple[ParametersBoundType, Tuple[Any, ...]]] = None,
                 optimizer_name: str = "UnnamedCustomOptimizer",
                 specific_points_to_include: Optional[Tuple[float, ...] | float] = None,
                 ):

        if argument_names is None and parameter_bounds is not None:
            argument_names = [f'ARG-{i}' for i in parameter_bounds]

        self._argument_names = argument_names
        self._parameter_bounds = parameter_bounds
        self._optimizer_name = optimizer_name
        self._optimizer_type = OptimizerType.custom
        self._specific_points_to_include = specific_points_to_include



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
    def argument_names(self):
        return self._argument_names


    def copy(self):
        return copy.deepcopy(self)

    def _run_optimization(self,
                          objective_function: callable,
                          number_of_function_calls: int = None,
                          verbosity: int = 0,
                          show_progress_bar: bool = False,
                          **kwargs
                          )->OptimizationResult:

        raise NotImplementedError("This method should be implemented in the child class.")
