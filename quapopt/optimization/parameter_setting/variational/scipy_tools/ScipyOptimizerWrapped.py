# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

from typing import List, Tuple, Dict, Any, Union

import numpy as np
from scipy import optimize as scopt

from quapopt.optimization import OptimizationResult
from quapopt.optimization.parameter_setting import OptimizerType




__SCIPY_SOLVERS_LOCAL_TYPES = {'no_grad': {'bounds': ['nelder-mead', 'powell', 'cobyqa'],
                                           'no_bounds': ['cobyla']},
                               'grad': {'bounds': ['l-bfgs-b', 'tnc', 'slsqp'],
                                        'no_bounds': ['cg', 'bfgs']},
                               'hessian': {'bounds': ['trust-constr'],
                                           'no_bounds': ['trust-ncg', 'trust-krylov', 'trust-exact']}}

SCIPY_SOLVERS_LOCAL = {
    ######### LOCAL SOLVERS THAT ARE NOT BASED ON GRADIENT ESTIMATION #########
    # LOCAL SOLVERS THAT RESPECT BOUNDS BY DEFAULT
    'nelder-mead': ('no_grad', 'bounds'),
    'powell': ('no_grad', 'bounds'),
    'cobyqa': ('no_grad', 'bounds'),
    # LOCAL SOLVERS THAT DO NOTRESPECT BOUNDS BY DEFAULT
    'cobyla': ('no_grad', 'no_bounds'),
    ######### LOCAL SOLVERS THAT ARE BASED ON GRADIENT ESTIMATION #########
    # LOCAL SOLVERS THAT RESPECT BOUNDS BY DEFAULT
    'l-bfgs-b': ('grad', 'bounds'),
    'tnc': ('grad', 'bounds'),
    'slsqp': ('grad', 'bounds'),
    # LOCAL SOLVERS THAT DO NOT RESPECT BOUNDS BY DEFAULT
    'cg': ('grad', 'no_bounds'),
    'bfgs': ('grad', 'no_bounds'),
    ######### LOCAL SOLVERS THAT ARE BASED ON HESSIAN ESTIMATION #########
    # LOCAL SOLVERS THAT RESPECT BOUNDS BY DEFAULT
    'trust-constr': ('hessian', 'bounds'),
    # LOCAL SOLVERS THAT DO NOT RESPECT BOUNDS BY DEFAULT
    'trust-ncg': ('hessian', 'no_bounds'),
    'trust-krylov': ('hessian', 'no_bounds'),
    'trust-exact': ('hessian', 'no_bounds'),

}


class ScipyOptimizerWrapped:
    def __init__(self,
                 parameters_bounds: List[Tuple[float, float]],
                 argument_names: List[str] = None,
                 optimizer_name=None,
                 optimizer_kwargs: Dict[str, Any] = None,
                 starting_point: Union[List[float], np.ndarray] = None,
                 basinhopping=False,
                 basinhopping_kwargs: Dict[str, Any] = None,
                 ):

        if starting_point is None:
            starting_point = [np.mean([bound[0], bound[1]]) + 0.01 for bound in parameters_bounds]

        if optimizer_name is None:
            optimizer_name = 'COBYQA'

        if optimizer_kwargs is None:
            optimizer_kwargs = {'options': {}}
            if optimizer_name.lower() == 'cobyla':
                optimizer_kwargs = {'options': {'disp': False,
                                                'maxiter': 100,
                                                'catol': 1e-2,

                                                }, }
            elif optimizer_name.lower() == 'powell':
                optimizer_kwargs = {'options': {'disp': False,
                                                'maxfev': 100,
                                                'xtol': 1e-2,
                                                'ftol': 1e-2,
                                                }, }

        optimizer_kwargs['method'] = optimizer_name
        optimizer_kwargs['bounds'] = parameters_bounds

        if basinhopping:
            if basinhopping_kwargs is None:
                basinhopping_kwargs = {'niter': 10,
                                       'T': 2.0,
                                       'disp': False,
                                       'stepsize': 0.1,
                                       'seed': None}
            basinhopping_kwargs['x0'] = starting_point
        else:
            optimizer_kwargs['x0'] = starting_point

        starting_point = np.array(starting_point)
        self._optimizer_name = optimizer_name
        self._optimizer_kwargs = optimizer_kwargs
        self._starting_point = starting_point
        self._parameters_bounds = parameters_bounds
        self._basinhopping = basinhopping
        self._basinhopping_kwargs = basinhopping_kwargs

        if argument_names is None:
            argument_names = [f'ARG-{i}' for i in range(len(parameters_bounds))]
        self._argument_names = argument_names

        self._optimizer_type = OptimizerType.scipy

    @property
    def parameters_bounds(self):
        return self._parameters_bounds

    @property
    def argument_names(self):
        return self._argument_names

    @property
    def optimizer_name(self):
        return self._optimizer_name

    @property
    def optimizer_type(self):
        return self._optimizer_type

    def run_optimization(self,
                         objective_function: callable,
                         number_of_function_calls: int = None,
                         verbosity: int = 0,
                         optimizer_seed=None,
                         ) -> OptimizationResult:
        # TODO FBM: extend this

        if self._basinhopping:
            basinhopping_kwargs_run = self._basinhopping_kwargs.copy()
            number_of_iterations = basinhopping_kwargs_run['niter']
            # This is number of function calls per basinhopping iterations
            number_of_function_calls = int(number_of_function_calls // number_of_iterations)
            basinhopping_kwargs_run['disp'] = verbosity > 0
            basinhopping_kwargs_run['seed'] = optimizer_seed

        optimizer_kwargs_run = self._optimizer_kwargs.copy()
        if self._optimizer_name.lower() in ['cobyla']:
            optimizer_kwargs_run['options']['maxiter'] = number_of_function_calls
            optimizer_kwargs_run['options']['disp'] = verbosity
        elif self._optimizer_name.lower() in ['nelder-mead', 'powell', 'cobyqa']:
            optimizer_kwargs_run['options']['maxfev'] = number_of_function_calls
            optimizer_kwargs_run['options']['disp'] = verbosity > 0
        else:
            raise ValueError("Optimizer not supported:", self._optimizer_name)

        if self._basinhopping:
            res = scopt.basinhopping(func=objective_function,
                                     **basinhopping_kwargs_run,
                                     minimizer_kwargs=optimizer_kwargs_run)
        else:
            # print(optimizer_kwargs_run)
            res = scopt.minimize(fun=objective_function,
                                 **optimizer_kwargs_run,

                                 )

        best_arguments = res.x
        best_funval = res.fun

        return OptimizationResult(best_value=best_funval,
                                  best_arguments=best_arguments,
                                  trials_dataframe=None)
