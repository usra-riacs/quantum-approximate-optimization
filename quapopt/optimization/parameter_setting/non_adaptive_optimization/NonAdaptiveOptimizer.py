# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import copy
import time
from typing import List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from quapopt.optimization import OptimizationResult
from quapopt.optimization.parameter_setting import ParametersBoundType
from quapopt.optimization.parameter_setting import OptimizerType
from quapopt.optimization.parameter_setting.CustomOptimizer import CustomOptimizer

class NonAdaptiveOptimizer(CustomOptimizer):

    def __init__(self,
                 search_space: Optional[List[Any]] = None,
                 argument_names: Optional[List[str]] = None,
                 parameter_bounds: Optional[List[Tuple[ParametersBoundType, Tuple[Any, ...]]]] = None,
                 specific_points_to_include: Optional[Tuple[float, ...] | float] = None,
                 local_search_spaces_sizes: Optional[List[int]] = None,

                 optimizer_name: str = "UnnamedNonAdaptiveOptimizer"):



        super().__init__(argument_names=argument_names,
                         parameter_bounds=parameter_bounds,
                         optimizer_name=optimizer_name,
                         specific_points_to_include=specific_points_to_include
                         )

        self._search_space = search_space
        self._local_search_spaces_sizes = local_search_spaces_sizes





    @property
    def search_space(self):
        return self._search_space
    @search_space.setter
    def search_space(self, value):
        self._search_space = value

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

        t_start = time.perf_counter()

        dt_funval, dt_add = 0.0, 0.0

        function_values_list = []

        _pbar = None
        if show_progress_bar:
            _pbar = tqdm(list(dict(enumerate(self.search_space[0:real_number_of_trials])).items()),
                                           colour='yellow',
                                           position=0,)

        for trial_index, arguments in dict(enumerate(self.search_space[0:real_number_of_trials])).items():

            t0 = time.perf_counter()
            funval = objective_function(*arguments)
            t1=time.perf_counter()
            if abs(funval) >= 10 ** 10:
                raise ValueError(f'Function value is too large: {funval} for arguments: {arguments}')

            function_values_list.append(funval)

            t2 = time.perf_counter()

            if show_progress_bar:
                _pbar.update(1)

            if funval < best_funval:
                if verbosity > 1:
                    print(f'New minimum found at trial: {trial_index} with function value: {best_funval} -> {funval}')

                best_funval = funval
                best_arguments = arguments

                if show_progress_bar:
                    _pbar.set_postfix(BestCost=f"{funval:.5f} (ARGS = {arguments})")

            t3 = time.perf_counter()


            # print('funval took:', t1-t0, 'iters per second:', 1/(t1-t0), '')
            # print('additions took:',t2-t1, 'iters per second:', 1/(t2-t1), '')
            # print("additions 2 took:", t3 - t2, 'iters per second:', 1 / (t3 - t2), '')




        t_end = time.perf_counter()

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
