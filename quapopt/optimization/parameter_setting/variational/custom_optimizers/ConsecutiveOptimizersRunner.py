# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import plotly

from quapopt.optimization.parameter_setting import ParametersBoundType

import itertools
from typing import List, Tuple, Any, Optional, Callable

import numpy as np
import pandas as pd
from quapopt.optimization.parameter_setting.non_adaptive_optimization.NonAdaptiveOptimizer import NonAdaptiveOptimizer
from quapopt.optimization.parameter_setting.non_adaptive_optimization.SimpleGridOptimizer import SimpleGridOptimizer
from quapopt.optimization.parameter_setting.variational.custom_optimizers.AdaptiveOptimizer import AdaptiveOptimizer
from quapopt.optimization import OptimizationResult
from quapopt.optimization.parameter_setting import OptimizerType


from quapopt.optimization.parameter_setting import ParametersBoundType


class ConsecutiveOptimizersRunner(AdaptiveOptimizer):
    def __init__(self,
                 global_parameters_bounds: List[Tuple[ParametersBoundType, Tuple[Any, ...]]],
                 list_of_optimizer_builders: Optional[List[Tuple[OptimizerType, callable]]]=None,
                 calls_split:Optional[List[float]]=None,
                 shrink_factors:Optional[List[float]]=None,
                 argument_names: List[str] = None,
                 optimizer_name:Optional[str]=None
                 ):

        assert np.all(np.array(calls_split)>=0), 'All precision splits must be positive.'
        assert np.all(np.array(shrink_factors)>0), 'All shrink factors must be positive.'
        assert abs(sum(calls_split)-1.0)<1e-8, 'Precision splits must sum to 1.'

        if optimizer_name is None:
            optimizer_name = 'ConsecutiveOptimizersRunner'

        super().__init__(parameter_bounds=global_parameters_bounds,
                         argument_names=argument_names,
                         optimizer_name=optimizer_name)


        if calls_split is None:
            calls_split = [0.5, 0.5]

        self._calls_split = calls_split

        if shrink_factors is None:
            shrink_factors = [0.1]

        self._shrink_factors = shrink_factors



        assert len(shrink_factors)==len(calls_split)-1, 'The number of shrink factors must be one less than the number of precision splits.'

        self._optimizer_builders = None

        if list_of_optimizer_builders is not None:
            self.set_optimizer_builders(list_of_optimizer_builders=list_of_optimizer_builders)


    def set_optimizer_builders(self, list_of_optimizer_builders: List[Tuple[OptimizerType, callable]]):
        self._optimizer_builders = list_of_optimizer_builders


    def run_optimization(self,
                         objective_function: callable,
                         number_of_function_calls: int,
                         verbosity: int = 0,
                         show_progress_bar: bool = False,
                         #dummy variable to satisfy interface
                         optimizer_seed: int = None,
                         ):


        precisions = self._calls_split

        global_parameter_bounds = self._parameter_bounds
        number_of_params = len(global_parameter_bounds)

        parameter_bounds_l = global_parameter_bounds
        best_arguments_l = None
        search_space_l = None

        _calls_so_far = 0


        all_dataframes = []
        for precision_level, precision_split in enumerate(precisions):

            if precision_split == 0.0:
                continue

            number_of_function_calls_l = int(number_of_function_calls*precision_split)

            _optimizer_type, _optimizer_builder = self._optimizer_builders[precision_level]


            if _optimizer_type  == OptimizerType.custom:
                #we assume that the custom optimizer is grid optimizer here. #TODO(FBM): extend this to other optimizers
                optimizer_res = _optimizer_builder(parameter_bounds=parameter_bounds_l,
                                                   max_trials=number_of_function_calls_l,
                                                   search_space=search_space_l
                                                   )
                _arg_show_progress_bar = {'show_progress_bar':show_progress_bar}

            elif _optimizer_type == OptimizerType.scipy:
                optimizer_res = _optimizer_builder(parameter_bounds=parameter_bounds_l,
                                                   starting_point=best_arguments_l,
                                                   )
                _arg_show_progress_bar = {}


            else:
                raise ValueError(f'Unsupported optimizer type: {_optimizer_type}')

            res_l = optimizer_res.run_optimization(objective_function=objective_function,
                                                   number_of_function_calls=number_of_function_calls_l,
                                                   verbosity=verbosity,
                                                   **_arg_show_progress_bar)

            _calls_so_far+=number_of_function_calls_l

            # all_dataframes.append(df_trials_l)

            if precision_level==len(precisions)-1:
                continue

            best_arguments_l = res_l.best_arguments
            shrink_factor_l2 = self._shrink_factors[precision_level]

            if precisions[precision_level+1] == 0.0:
                continue

            number_of_function_calls_l2 = number_of_function_calls * precisions[precision_level+1]

            side_size_l2 = int(number_of_function_calls_l2**(1/number_of_params))

            _optimizer_type_l2 = self._optimizer_builders[precision_level+1][0]

            new_bounds, local_search_spaces_l2 = [], []
            for index_a, (best_a, (_type, (_min, _max))) in enumerate(zip(best_arguments_l,parameter_bounds_l)):

                original_size = _max-_min
                current_size = original_size*shrink_factor_l2
                radius = current_size/2.0

                search_space_here = SimpleGridOptimizer.create_search_space_around_point(bound_type=_type,
                                                                                          center_point=best_a,
                                                                                          number_of_points=side_size_l2,
                                                                                          radius=radius,
                                                                                         bound_range_global=global_parameter_bounds[index_a][1])
                _min_local = min(search_space_here)
                _max_local = max(search_space_here)

                if _optimizer_type_l2 == OptimizerType.custom:
                    new_bounds.append((_type, (_min_local, _max_local)))
                    local_search_spaces_l2.append(search_space_here)
                elif _optimizer_type_l2 == OptimizerType.scipy:
                    new_bounds.append((_min_local, _max_local))


            parameter_bounds_l = new_bounds

            if _optimizer_type_l2 == OptimizerType.custom:
                search_space_l = list(itertools.product(*local_search_spaces_l2))


        best_funval = res_l.best_value
        best_arguments = res_l.best_arguments
        # df_trials = pd.concat(all_dataframes)



        return OptimizationResult(best_value=best_funval,
                                  best_arguments=best_arguments,
                                  # trials_dataframe=df_trials

                                  )
