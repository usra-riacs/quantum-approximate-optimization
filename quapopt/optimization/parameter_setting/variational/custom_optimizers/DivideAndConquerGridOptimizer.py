# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import plotly

from quapopt.optimization.parameter_setting import ParametersBoundType

import itertools
from typing import List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from quapopt.optimization.parameter_setting.non_adaptive_optimization.NonAdaptiveOptimizer import NonAdaptiveOptimizer
from quapopt.optimization.parameter_setting.non_adaptive_optimization.SimpleGridOptimizer import SimpleGridOptimizer
from quapopt.optimization.parameter_setting.variational.custom_optimizers.AdaptiveOptimizer import AdaptiveOptimizer
from quapopt.optimization import OptimizationResult


from quapopt.optimization.parameter_setting import ParametersBoundType
from quapopt.optimization.parameter_setting.variational.custom_optimizers.ConsecutiveOptimizersRunner import ConsecutiveOptimizersRunner
from quapopt.optimization.parameter_setting import OptimizerType

class DivideAndConquerGridOptimizer(ConsecutiveOptimizersRunner):
    def __init__(self,
                 parameter_bounds: List[Tuple[ParametersBoundType, Tuple[Any, ...]]],
                 calls_split:Optional[List[float]]=None,
                 shrink_factors:Optional[List[float]]=None,
                 argument_names: List[str] = None,
                 specific_points_to_include:Optional[Tuple[float,...]|float]=None,
                 ):

        super().__init__(global_parameters_bounds=parameter_bounds,
                         list_of_optimizer_builders=None,
                         argument_names=argument_names,
                         calls_split=calls_split,
                         shrink_factors=shrink_factors,
                         optimizer_name="DivideAndConquerGridOptimizer")

        precisions = self._calls_split

        local_optimizer_builders = []
        for precision_level, precision_split in enumerate(precisions):
            def _local_builder(parameter_bounds,
                               max_trials,
                               search_space):


                return SimpleGridOptimizer(parameter_bounds=parameter_bounds,
                                             argument_names=self._argument_names,
                                             max_trials=max_trials,
                                             specific_points_to_include=specific_points_to_include if precision_level == 0 else None,
                                             search_space=search_space)

            local_optimizer_builders.append((OptimizerType.custom,_local_builder))

        self.set_optimizer_builders(list_of_optimizer_builders=local_optimizer_builders)
