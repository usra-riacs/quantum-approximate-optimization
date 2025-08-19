# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.
import copy
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from quapopt.optimization import OptimizationResult
from quapopt.optimization.parameter_setting import ParametersBoundType
from quapopt.optimization.parameter_setting import OptimizerType


class NonAdaptiveOptimizer:

    def __init__(self,
                 search_space: List[Any] = None,
                 argument_names: List[str] = None,
                 parameter_bounds: List[Tuple[ParametersBoundType, Tuple[Any, ...]]] = None,
                 optimizer_name: str = "UnnamedOptimizer"):
        self._search_space = search_space

        if argument_names is None:
            argument_names = [f'ARG-{i}' for i in range(len(search_space[0]))]

        self._argument_names = argument_names
        self._parameter_bounds = parameter_bounds
        self._optimizer_name = optimizer_name
        self._optimizer_type = OptimizerType.custom



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

    def _run_optimization(self,
                          objective_function: callable,
                          number_of_function_calls: int = None,
                          verbosity: int = 0,
                          show_progress_bar: bool = False,
                          search_space=None
                          ):

        if search_space is None:
            assert self.search_space is not None, "Search space is not defined."
        else:
            self._search_space = search_space

        if number_of_function_calls is None:
            number_of_function_calls = len(self.search_space)

        best_funval, best_arguments = np.inf, None

        real_number_of_trials = min(number_of_function_calls, len(self.search_space))

        function_values_list = []
        for trial_index, arguments in tqdm(list(dict(enumerate(self.search_space[0:real_number_of_trials])).items()),
                                           colour='yellow',
                                           position=0,
                                           disable=not show_progress_bar):

            funval = objective_function(*arguments)

            if abs(funval) >= 10 ** 10:
                raise ValueError(f'Function value is too large: {funval} for arguments: {arguments}')

            function_values_list.append(funval)

            if funval < best_funval:
                if verbosity > 0:
                    print(f'New minimum found at trial: {trial_index} with function value: {best_funval} -> {funval}')

                best_funval = funval
                best_arguments = arguments

        data = {'TrialIndex': list(range(real_number_of_trials)),
                'FunctionValue': function_values_list
                }
        for i in range(len(self.argument_names)):
            data[self.argument_names[i]] = [x[i] for x in self.search_space[0:real_number_of_trials]]

        df = pd.DataFrame(data)

        # print(function_values_list)

        return OptimizationResult(best_value=best_funval,
                                  best_arguments=best_arguments,
                                  trials_dataframe=df)

    # def run_optimization(self,
    #                      objective_function: callable,
    #                      number_of_trials: int=None,
    #                      verbosity: int=0,
    #                      show_progress_bar: bool=False,
    #                      search_space=None
    #                      ):
    #     return self._run_optimization(objective_function=objective_function,
    #                                   number_of_trials=number_of_trials,
    #                                   verbosity=verbosity,
    #                                   show_progress_bar=show_progress_bar,
    #                                   search_space=search_space)
