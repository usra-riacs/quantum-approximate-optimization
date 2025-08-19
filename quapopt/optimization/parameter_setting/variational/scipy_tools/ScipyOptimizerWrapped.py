# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

 
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import scipy as sc
from scipy.optimize import basinhopping
from tqdm.notebook import tqdm
from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from scipy import optimize as scopt
from quapopt.optimization.parameter_setting import OptimizerType, ParametersBoundType
from quapopt.optimization import OptimizationResult, BestResultsContainer
from quapopt.optimization.parameter_setting import OptimizerType


class ScipyOptimizerWrapped:
    def __init__(self,
                 parameters_bounds : List[Tuple[float, float]],
                 argument_names: List[str] = None,
                 optimizer_name = None,
                 optimizer_kwargs: Dict[str, Any] = None,
                 starting_point: Union[List[float],np.ndarray] = None,
                 basinhopping = False,
                 basinhopping_kwargs: Dict[str, Any] = None,
                 ):


        if starting_point is None:
            starting_point = [np.mean([bound[0], bound[1]])+0.01 for bound in parameters_bounds]


        if optimizer_name is None:
            optimizer_name = 'COBYLA'

        if optimizer_kwargs is None:
            optimizer_kwargs = {'options':{}}
            if optimizer_name == 'COBYLA':
                optimizer_kwargs = {'options': {'disp': False,
                                                'maxiter': 100,
                                                'catol': 1e-2,
                                                'rhobeg': 0.1
                                                }, }
        optimizer_kwargs['method'] = optimizer_name
        optimizer_kwargs['bounds'] = parameters_bounds

        if basinhopping:
            if basinhopping_kwargs is None:
                basinhopping_kwargs = {'niter':10,
                                       'T':2.0,
                                       'disp':False,
                                       'stepsize':0.1,
                                       'seed':None}
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
                         show_progress_bar: bool = False,
                         optimizer_seed=None,
                         )->OptimizationResult:
        #TODO FBM: extend this




        if self._basinhopping:
            basinhopping_kwargs_run = self._basinhopping_kwargs.copy()
            number_of_iterations = basinhopping_kwargs_run['niter']
            #This is number of function calls per basinhopping iterations
            number_of_function_calls = int(number_of_function_calls//number_of_iterations)
            basinhopping_kwargs_run['disp'] = verbosity > 0
            basinhopping_kwargs_run['seed'] = optimizer_seed

        optimizer_kwargs_run = self._optimizer_kwargs.copy()
        if self._optimizer_name in ['COBYLA']:
            optimizer_kwargs_run['options']['maxiter'] = number_of_function_calls
            optimizer_kwargs_run['options']['disp'] = verbosity > 0


        elif self._optimizer_name in ['nelder-mead', 'powell']:
            optimizer_kwargs_run['options']['maxfev'] = number_of_function_calls
            optimizer_kwargs_run['options']['disp'] = verbosity > 0
        else:
            raise ValueError("Optimizer not supported:", self._optimizer_name)

        if show_progress_bar:
            pbar = tqdm(total=number_of_function_calls,
                        desc=self.optimizer_name,
                        colour='yellow',
                        position=0,
                        )

        _history_holder = []
        if self._optimizer_name in ['COBYLA','SLSQP','TNC']:
            #TODO(FBM): those optimizers support only trivial callback
            #_callback_function = None
            pass

        else:
            def _callback_function(intermediate_result: scopt.OptimizeResult):
                data = {
                    'FunctionValue': [intermediate_result.fun]
                }
                for i in range(len(self.argument_names)):
                    data[self.argument_names[i]] = [intermediate_result.x[i]]

                _history_holder.append(pd.DataFrame(data=data))

                if show_progress_bar:
                    pbar.update(1)
                return False
            optimizer_kwargs_run['callback'] = _callback_function



        if self._basinhopping:
            # print("minimizer_kwargs:", optimizer_kwargs_run)
            # print("basinhopping_kwargs:", basinhopping_kwargs_run)


            res = scopt.basinhopping(func=objective_function,
                                     **basinhopping_kwargs_run,
                                     minimizer_kwargs=optimizer_kwargs_run)
        else:
            #print(optimizer_kwargs_run)
            res = scopt.minimize(fun=objective_function,
                                 **optimizer_kwargs_run,

                                 )
        if len(_history_holder)==0:
            df_trials = None
        else:
            df_trials = pd.concat(_history_holder, axis=0,ignore_index=True)
            df_trials['TrialIndex'] = list(range(len(_history_holder)))

        best_arguments = res.x
        best_funval = res.fun

        return OptimizationResult(best_value=best_funval,
                                  best_arguments=best_arguments,
                                  trials_dataframe=df_trials)
