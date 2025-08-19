# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

import itertools
from typing import List, Tuple, Any

import numpy as np

from quapopt.optimization.parameter_setting.non_adaptive_optimization.NonAdaptiveOptimizer import NonAdaptiveOptimizer
from quapopt.optimization.parameter_setting import ParametersBoundType


class SimpleGridOptimizer(NonAdaptiveOptimizer):
    def __init__(self,
                 parameter_bounds: List[Tuple[ParametersBoundType, Tuple[Any, ...]]],
                 max_trials: int,
                 argument_names: List[str] = None
                 ):

        number_of_parameters = len(parameter_bounds)
        non_fixed_parameters = [x for x in parameter_bounds if x[0] != ParametersBoundType.CONSTANT]
        categorical_parameters = [x[1] for x in non_fixed_parameters if x[0] == ParametersBoundType.SET]
        total_mult_factor_categorical = np.prod([len(x) for x in categorical_parameters])

        assert total_mult_factor_categorical <= max_trials, "Number of trials is too small for minimal grid size."

        continuous_parameters = [x for x in non_fixed_parameters if x[0] == ParametersBoundType.RANGE]
        number_of_continuous_parameters = len(continuous_parameters)

        trials_left_for_continuous = max_trials // total_mult_factor_categorical

        if number_of_continuous_parameters != 0:
            size_each_continuous_grid = int(trials_left_for_continuous ** (1 / number_of_continuous_parameters))
        else:
            size_each_continuous_grid = 0

        local_search_spaces = []
        for i in range(number_of_parameters):
            bound_type, bound_specs = parameter_bounds[i]

            if bound_type == ParametersBoundType.RANGE:
                min_value, max_value = bound_specs
                local_search_spaces.append(np.linspace(start=min_value,
                                                       stop=max_value,
                                                       num=size_each_continuous_grid))
            elif bound_type == ParametersBoundType.SET:
                local_search_spaces.append(bound_specs)
            elif bound_type == ParametersBoundType.CONSTANT:
                if isinstance(bound_specs, list) or isinstance(bound_specs, tuple):
                    assert len(bound_specs) == 1, "Fixed parameter should have only one value."
                    bound_specs = bound_specs[0]
                local_search_spaces.append([bound_specs])
            else:
                raise ValueError('Unknown bound type.')

        search_space = list(itertools.product(*local_search_spaces))

        assert len(search_space) <= max_trials, "Number of trials is too small for minimal grid size."

        super().__init__(search_space=search_space,
                         argument_names=argument_names,
                         optimizer_name="GridOptimizer")

    def run_optimization(self,
                         objective_function: callable,
                         number_of_function_calls: int = None,
                         verbosity: int = 0,
                         show_progress_bar: bool = False,
                         #dummy variable to satisfy interface
                         optimizer_seed: int = None
                         ):


        return super()._run_optimization(objective_function=objective_function,
                                         number_of_function_calls=number_of_function_calls,
                                         verbosity=verbosity,
                                         show_progress_bar=show_progress_bar)
