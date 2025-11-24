# def get_wrapped_run_function_optuna(hamiltonian_representations:List[Union[CLH,CQH]]):
# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import itertools
from typing import Any, List, Tuple, Union

import numpy as np
import optuna


class SimplifiedGridSampler(optuna.samplers.BaseSampler):
    def __init__(
        self,
        parameter_names: List[str],
        parameter_bounds: List[Tuple[Union[float, int, Any], ...]],
        max_trials=10**4,
    ):

        number_of_parameters = len(parameter_names)

        # TODO FBM: Implement this
        fixed_params_names = []
        # if fixed_parameters_dict is not None:
        #     for key in fixed_parameters_dict.keys():
        #
        #

        # TODO(FBM): add support for grid of categorical parameters
        single_grid_size = int(max_trials ** (1 / number_of_parameters))
        all_grids = [
            tuple(
                np.linspace(
                    start=parameter_bounds[i][0],
                    stop=parameter_bounds[i][1],
                    num=single_grid_size,
                )
            )
            for i in range(number_of_parameters)
            if parameter_names[i] not in fixed_params_names
        ]
        all_combinations = list(itertools.product(*all_grids))

        assert (
            len(all_combinations) <= max_trials
        ), "Number of trials is too small for minimal grid size."

        fixed_search_space = {}
        for i in range(number_of_parameters):
            param_name = parameter_names[i]
            param_bounds = parameter_bounds[i]
            param_type = type(param_bounds[1])
            if param_type == int:
                fixed_search_space[param_name] = optuna.distributions.IntDistribution(
                    low=param_bounds[0], high=param_bounds[1]
                )
            elif param_type == float:
                fixed_search_space[param_name] = optuna.distributions.FloatDistribution(
                    low=param_bounds[0], high=param_bounds[1]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        self._fixed_search_space = fixed_search_space
        self._parameter_names = parameter_names
        self._number_of_parameters = number_of_parameters
        self._all_combinations = all_combinations
        self._modulo = len(all_combinations)

    def sample_relative(
        self, study: optuna.Study, trial: optuna.Trial, search_space
    ) -> dict:

        return {
            self._parameter_names[i]: self._all_combinations[
                int(trial._trial_id % self._modulo)
            ][i]
            for i in range(self._number_of_parameters)
        }

    def infer_relative_search_space(self, study, trial):
        return self._fixed_search_space

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )


class SimplifiedFixedArgumentsSampler(optuna.samplers.BaseSampler):
    def __init__(
        self,
        parameter_names: List[str],
        parameter_values: List[Tuple[Union[int, float]]],
    ):

        number_of_parameters = len(parameter_names)
        fixed_search_space = {}

        assert (
            len(set([len(param_values) for param_values in parameter_values])) == 1
        ), "All parameters must have the same number of values."

        for i in range(number_of_parameters):
            param_name = parameter_names[i]
            param_values = parameter_values[i]
            min_value = min(param_values)
            max_value = max(param_values)
            param_type = type(min_value)

            if param_type == int:
                fixed_search_space[param_name] = optuna.distributions.IntDistribution(
                    low=min_value, high=max_value
                )
            elif param_type == float:
                fixed_search_space[param_name] = optuna.distributions.FloatDistribution(
                    low=min_value, high=max_value
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        self._parameter_names = parameter_names
        self._number_of_parameters = number_of_parameters
        self._fixed_search_space = fixed_search_space
        self._all_fixed_values = parameter_values
        self._modulo = len(parameter_values[0])

    def sample_relative(
        self, study: optuna.Study, trial: optuna.Trial, search_space
    ) -> dict:
        return {
            self._parameter_names[i]: self._all_fixed_values[i][
                trial._trial_id % self._modulo
            ]
            for i in range(self._number_of_parameters)
        }

    def infer_relative_search_space(self, study, trial):
        return self._fixed_search_space

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
