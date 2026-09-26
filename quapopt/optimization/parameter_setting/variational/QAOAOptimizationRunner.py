# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import time
import warnings
from typing import Union, List, Tuple

import numpy as np
import optuna
import pandas as pd
#TODO(FBM): wrap optuna samplers
from optuna.samplers import BaseSampler as BaseSamplerOptuna
from tqdm.notebook import tqdm

from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import ClassicalMeasurementNoiseSampler
from quapopt.data_analysis.data_handling import STANDARD_NAMES_DATA_TYPES as SNDT
from quapopt.meta_algorithms.QRR import QRR_functions as qrr_fun
from quapopt.optimization import OptimizationResult, BestResultsContainer
from quapopt.optimization.QAOA import QAOAFunctionInputFormat as FIFormat
from quapopt.optimization.QAOA.QAOARunnerBase import QAOARunnerBase
from quapopt.optimization.QAOA.simulation.QAOARunnerExpValuesPauliBackprop import QAOARunnerExpValuesPauliBackprop
from quapopt.optimization.parameter_setting import OptimizerType
from quapopt.optimization.parameter_setting.CustomOptimizer import CustomOptimizer
from quapopt.optimization.parameter_setting.non_adaptive_optimization.SimpleGridOptimizer import SimpleGridOptimizer

from quapopt.optimization.parameter_setting.variational.scipy_tools.ScipyOptimizerWrapped import ScipyOptimizerWrapped
from quapopt.optimization.QAOA import QAOAResult
from quapopt import ancillary_functions as anf

_DEFAULT_HYPERPARAMETERS_TPE = {"consider_prior": True,
                                "prior_weight": 1.0,
                                "consider_magic_clip": True,
                                "consider_endpoints": False,
                                "n_startup_trials": 10,
                                "n_ei_candidates": 24,
                                # "gamma": default_gamma,
                                # "weights": default_weights,
                                # "seed": Optional[int] = None,
                                "multivariate": False,
                                "group": False,
                                "constant_liar": False}



def make_no_improvement_callback(patience: int):
    best_value = {"val": None}
    no_improvement_count = {"n": 0}

    def callback(study: optuna.Study,
                 trial: optuna.trial.FrozenTrial):
        current_best = study.best_value

        if best_value["val"] is None:
            best_value["val"] = current_best
            no_improvement_count["n"] = 0
            return

        # if best_value improved: reset counter
        if current_best < best_value["val"]:
            best_value["val"] = current_best
            no_improvement_count["n"] = 0
        else:
            no_improvement_count["n"] += 1

        if no_improvement_count["n"] >= patience:
            #print(f"Stopping: no improvement in {patience} trials.")
            study.stop()

    return callback


class QAOAOptimizationRunner:
    def __init__(self,
                 qaoa_runner: QAOARunnerBase):

        self._qaoa_runner = qaoa_runner

    @property
    def qaoa_runner(self):
        return self._qaoa_runner

    # @property
    # def best_results_container(self):
    #     return self._qaoa_runner._best_results_container
    @staticmethod
    def _transform_trials_to_df(trials_list: List[Tuple[int, float, float, np.ndarray]|list]) -> pd.DataFrame:
        
        
        
        #
        #
        # df_overview = pd.DataFrame(data={
        #     'TrialIndex': trials_list[:][0],
        #     'FunctionValue': trials_list[:][1],
        #     'WallClockTime': trials_list[:][2],
        #     "Arguments": trials_list[:][3],
        # })



        df_overview = pd.DataFrame(data={
            'TrialIndex': [trials_list[i][0] for i in range(len(trials_list))],
            'FunctionValue': [trials_list[i][1] for i in range(len(trials_list))],
            'WallClockTime': [trials_list[i][2] for i in range(len(trials_list))],
            "Arguments": [trials_list[i][3] for i in range(len(trials_list))],
        })



        return df_overview
    def run_optimization(self,
                         qaoa_depth: int,
                         number_of_function_calls: int,
                         classical_optimizer=Union[CustomOptimizer, ScipyOptimizerWrapped, BaseSamplerOptuna],
                         optimizer_seed=None,
                         run_id=None,
                         numpy_rng_sampling=None,
                         number_of_samples: int = None,
                         measurement_noise: ClassicalMeasurementNoiseSampler = None,
                         show_progress_bar=False,
                         verbosity=0,
                         optuna_pruner: optuna.pruners.BasePruner = None,
                         optuna_study_storage=None,
                         number_of_threads_optuna=1,
                         cvar_fraction=1.0,
                         max_unimproving_iterations_optuna=None,
                         additional_optimizer_kwargs=None,
                         save_afterwards:bool=False,
                         **additional_kwargs):

        if isinstance(classical_optimizer, CustomOptimizer):
            classical_optimizer_type = OptimizerType.custom
            _opt_string = classical_optimizer.optimizer_name
        elif isinstance(classical_optimizer, ScipyOptimizerWrapped):
            classical_optimizer_type = OptimizerType.scipy
            _opt_string = classical_optimizer.optimizer_name
        elif isinstance(classical_optimizer, BaseSamplerOptuna):
            classical_optimizer_type = OptimizerType.optuna
            _opt_string = classical_optimizer.__str__()
        else:
            raise ValueError(f"Unknown optimizer type for object: {classical_optimizer}. Please provide a valid optimizer.")

        if additional_optimizer_kwargs is None:
            additional_optimizer_kwargs = {}

        # TODO FBM: add logging optimization metadata
        self._qaoa_runner.clear_optimization_history()

        self.trial_index = 0
        self.trials = []  # list of dicts
        self.pbar = None

        self._global_minimum = float('inf')


        if show_progress_bar:


            self.pbar =  tqdm(total=number_of_function_calls,
                            desc=_opt_string,
                            colour='yellow',
                            position=0,
                              #close after finishing
                             leave=True,
                            )



        def __arguments_processing_function(*args):
            if isinstance(args, optuna.Trial):
                args_formatted = list(args.params)
            else:
                args_formatted = np.array(args).tolist()

            if classical_optimizer_type in [OptimizerType.custom]:
                if len(args_formatted) == 1:
                    if not isinstance(args_formatted[0], float):
                        args_formatted = args_formatted[0]
            elif classical_optimizer_type in [OptimizerType.scipy]:
                if len(args_formatted) == 1:
                    args_formatted = args_formatted[0]
                else:
                    raise ValueError("The arguments should be passed as a single tuple.")

                #if len(args_formatted) != 2 * qaoa_depth:
                   # raise ValueError("The number of arguments should be equal to 2*p, where p is the QAOA depth.")

            return args_formatted


        def __create_time_tracking_objective(func):
            def __objective_function(*args):
                t0 = time.perf_counter()
                funval = func(*args)
                t1 = time.perf_counter()


                args_formatted = __arguments_processing_function(*args)

                t2 = time.perf_counter()
                df_overview_simplified = (self.trial_index, funval, t1 - t0, args_formatted)


                self.trials.append(df_overview_simplified)

                if self.qaoa_runner.results_logger is not None and not save_afterwards:
                    if self.qaoa_runner.logging_level.value >= 1:
                        df_overview = pd.DataFrame(data={
                            'TrialIndex': [self.trial_index],
                            'FunctionValue': [funval],
                            'WallClockTime': [t1 - t0],
                            "Arguments":[args_formatted],
                        })



                        self.qaoa_runner.write_results(dataframe=df_overview,
                                                      data_type=SNDT.OptimizationOverviewAbstract)
                t3 = time.perf_counter()

                if self.pbar is not None:
                    self.pbar.update(1)
                t4 = time.perf_counter()

                if funval < self._global_minimum:
                    self._global_minimum = funval

                    #if verbosity>=2:
                    if self.pbar is not None:
                        self.pbar.set_postfix(BestCost=f"{funval:.5f}; t={self.trial_index}; "
                                                   f"args={[float(np.round(x,5)) for x in args_formatted]}")

                t5 = time.perf_counter()
                self.trial_index += 1

                # print("RUnning funval took:",t2-t1,"s", "iters per second:",1/(t2-t1))
                # print("Time tracking took:",t3-t2,"s", "iters per second:",1/(t3-t2))
                # print("Progress bar update took:",t4-t3,"s", "iters per second:",1/(t4-t3))
                # print("Best cost update took:",t5-t4,"s", "iters per second:",1/(t5-t4))
                #
                # raise KeyboardInterrupt

                return funval
            return __objective_function



        if classical_optimizer_type == OptimizerType.optuna:
            def __objective_function(trial: optuna.Trial):
                energy_mean = self._qaoa_runner.run_qaoa_wrapped(trial,
                                                                 qaoa_depth=qaoa_depth,
                                                                 number_of_samples=number_of_samples,
                                                                 measurement_noise=measurement_noise,
                                                                 numpy_rng_sampling=numpy_rng_sampling,
                                                                 input_format=FIFormat.optuna,
                                                                 cvar_fraction=cvar_fraction,
                                                                 **additional_kwargs)

                return energy_mean

            if classical_optimizer is None:
                classical_optimizer = optuna.samplers.TPESampler(seed=optimizer_seed,
                                                                 **_DEFAULT_HYPERPARAMETERS_TPE)
            if optuna_pruner is None:
                optuna_pruner = optuna.pruners.MedianPruner()

            callbacks_optuna = None
            if max_unimproving_iterations_optuna is not None:
                callbacks_optuna =[make_no_improvement_callback(patience=max_unimproving_iterations_optuna)]

            optuna_study = optuna.create_study(sampler=classical_optimizer,
                                               pruner=optuna_pruner,
                                               direction='minimize',
                                               storage=optuna_study_storage,
                                               study_name=run_id,
                                               load_if_exists=False,
                                               )
            optuna.logging.set_verbosity(verbosity)
            optuna_study.optimize(__create_time_tracking_objective(__objective_function),
                                  n_trials=number_of_function_calls,
                                  n_jobs=number_of_threads_optuna,
                                  show_progress_bar=False,
                                  callbacks=callbacks_optuna,
                                  **additional_optimizer_kwargs)

            df_optuna = optuna_study.trials_dataframe()
            names_map = {'value': 'FunctionValue'}
            arg_counter = 0
            for key in df_optuna.columns:
                key_split = key.split('_')
                if key_split[0] == 'params':
                    names_map[key] = f'ARG-{arg_counter}'
                    arg_counter += 1

            df_optuna = df_optuna.rename(columns=names_map)
            optimizer_res = OptimizationResult(best_value=optuna_study.best_value,
                                               best_arguments=optuna_study.best_params,
                                               trials_dataframe=df_optuna)
            # TODO FBM: add adding the best QAOAResult object


        elif classical_optimizer_type in [OptimizerType.custom]:


            if isinstance(classical_optimizer, SimpleGridOptimizer):
                classical_optimizer = classical_optimizer.rebuild_instance_with_the_same_search_space(max_trials=number_of_function_calls)


            global trial_index
            trial_index = 0
            def __objective_function(*args):

                t0 = time.perf_counter()
                args_formatted = __arguments_processing_function(*args)
                t1 = time.perf_counter()
                energy_mean = self._qaoa_runner.run_qaoa_wrapped(*args_formatted,
                                                                 qaoa_depth=qaoa_depth,
                                                                 number_of_samples=number_of_samples,
                                                                 measurement_noise=measurement_noise,
                                                                 numpy_rng_sampling=numpy_rng_sampling,
                                                                 input_format=None,
                                                                 cvar_fraction=cvar_fraction,
                                                                 **additional_kwargs)
                t2 = time.perf_counter()

                #print("qaoa took:",t2-t1,"s", "iters per second:",1/(t2-t1))


                return energy_mean

            optimizer_res = classical_optimizer.run_optimization(objective_function=__create_time_tracking_objective(__objective_function),
                                                                 number_of_function_calls=number_of_function_calls,
                                                                 verbosity=0,
                                                                 show_progress_bar=False,
                                                                 **additional_optimizer_kwargs

                                                                 )


        elif classical_optimizer_type == OptimizerType.scipy:
            _arguments_list = []
            _cost_values_list = []
            _trials_list = []
            _trials_counter = 0

            if classical_optimizer is None:
                classical_optimizer = ScipyOptimizerWrapped(parameters_bounds=[(-np.pi, np.pi)] * qaoa_depth + [(-np.pi/2,np.pi/2)]*qaoa_depth,
                                                            optimizer_name='COBYQA',
                                                            optimizer_kwargs={'options': {'disp': False,
                                                                                          'maxiter': number_of_function_calls,
                                                                                          'maxfev':number_of_function_calls,

                                                                                          }, },
                                                            starting_point=np.array([0.01] * (qaoa_depth * 2)),
                                                            basinhopping=False,
                                                            basinhopping_kwargs=None

                                                            )

            def __objective_function(*args):

                args_formatted = __arguments_processing_function(*args)


                energy_mean = self._qaoa_runner.run_qaoa_wrapped(*args_formatted,
                                                                 qaoa_depth=qaoa_depth,
                                                                 number_of_samples=number_of_samples,
                                                                 measurement_noise=measurement_noise,
                                                                 numpy_rng_sampling=numpy_rng_sampling,
                                                                 input_format=None,
                                                                 cvar_fraction=cvar_fraction,
                                                                 **additional_kwargs)
                energy_mean = float(energy_mean)

                return energy_mean

            optimizer_res = classical_optimizer.run_optimization(objective_function=__create_time_tracking_objective(__objective_function),
                                                                 number_of_function_calls=number_of_function_calls,
                                                                 verbosity=0,
                                                                 optimizer_seed=optimizer_seed,
                                                                 **additional_optimizer_kwargs)


            if show_progress_bar:
                self.pbar.close()
                self.pbar.clear()



        else:
            # TODO(FBM): add pytorch_implementation and tensorflow optimizers
            raise ValueError("Unknown optimizer type.")


        t0 = time.perf_counter()
        # df_overview = pd.concat(self.trials, axis=0, ignore_index=True)
        df_overview = self._transform_trials_to_df(trials_list=self.trials)

        optimizer_res._trials_dataframe = df_overview

        t1 = time.perf_counter()

        #print("putting df together took:",t1-t0)
        if save_afterwards and self.qaoa_runner.results_logger is not None:
            if self.qaoa_runner.logging_level.value >= 1:
                self.qaoa_runner.write_results(dataframe=df_overview,
                                               data_type=SNDT.OptimizationOverviewAbstract)


        return self._qaoa_runner.get_best_results(), optimizer_res




    def apply_QRR_to_optimization_results(self,
                                          return_full_history=False,
                                          show_progress_bar=False,
                                          store_n_best_solutions=1,
                                          solver=None):



        from quapopt import AVAILABLE_SIMULATORS
        #TODO(FBM): refactor type checking between cupy and numpy
        if solver is None:
            solver = self.qaoa_runner.hamiltonian_representations_cost[0].default_backend

        if solver == 'cupy':
            if 'cupy' not in AVAILABLE_SIMULATORS:
                raise ValueError("Cupy is not available. Please install cupy or use a different solver.")
            import cupy as bck
        else:
            import numpy as bck


        #print(solver)
        qaoa_sampler = self.qaoa_runner

        optimization_history = qaoa_sampler.optimization_history_full

        best_results_container = None
        if store_n_best_solutions != 1:
            best_results_container = BestResultsContainer(number_of_best_results=store_n_best_solutions, )

        if isinstance(qaoa_sampler, QAOARunnerExpValuesPauliBackprop):

            if len(optimization_history) == 0:
                print(
                    "Correlators are not stored in optimization history. Please rerun the optimization with store_correlators=True.")
                return None

            coupling_inverses_dict = {}
            for ind in qaoa_sampler.hamiltonian_representations_cost.keys():
                couplings_i = qaoa_sampler.couplings_cost_dict[ind]['numpy']
                couplings_i = bck.asarray(couplings_i)
                couplings_i_nonzero_mask = couplings_i != 0
                couplings_i_inverse = bck.zeros_like(couplings_i)
                couplings_i_inverse[couplings_i_nonzero_mask] = 1 / couplings_i[couplings_i_nonzero_mask]
                coupling_inverses_dict[ind] = couplings_i_inverse

            best_energy = bck.inf
            best_solution = None
            # TODO(FBM): consider what set of bitstrings to return (it could correspond to the best energy or the best mean energy)
            best_bitstrings = None
            best_result = None
            best_arguments = None

            full_history = []
            data_list = []
            for ind_j, qaoa_result_j in tqdm(enumerate(optimization_history),
                                             total=len(optimization_history),
                                             position=0,
                                             disable=not show_progress_bar,
                                             colour='blue'):

                qaoa_result_j:QAOAResult = qaoa_result_j

                hamiltonian_rep_index_j = qaoa_result_j.hamiltonian_representation_index
                correlators_j = qaoa_result_j.correlators
                if correlators_j is None:
                    print(
                        "Correlators are not stored in optimization history. Please rerun the optimization with store_correlators=True.")
                    return None

                #<H> = \sum_{i,j} J_{i,j} <ZiZj>
                correlators_weighted_j = bck.asarray(qaoa_result_j.correlators)
                #here we want JUST <ZiZj>
                #<ZiZj> = (p(00)+p(11))-(p(01)-p(10)) = p(SYM) - p(ASYM) = p(SYM) - (1-p(SYM)) = 2p(SYM)-1
                #(1+<ZiZj>)/2
                correlators_unweighted_j = correlators_weighted_j * coupling_inverses_dict[hamiltonian_rep_index_j]




                # TODO(FBM): normal QRR is supposed to round eigenvectors. Here I return them and round them manually to allow for future extensions to vanilla QRR
                candidate_eigenvectors_j, eigvals_j = qrr_fun.find_candidate_solutions_QRR(
                                                            correlations_matrix=correlators_unweighted_j,
                                                            return_eigenvectors=True,
                                                            solver=solver)

                candidate_eigenvectors_j[candidate_eigenvectors_j <= 0] = 0
                candidate_eigenvectors_j[candidate_eigenvectors_j > 0] = 1

                cost_hamiltonian_j = qaoa_sampler.hamiltonian_representations_cost[hamiltonian_rep_index_j]

                energies_j = cost_hamiltonian_j.evaluate_energy(bitstrings_array=candidate_eigenvectors_j,
                                                                backend_computation=solver
                                                                )

                best_energy_index_j = bck.argmin(energies_j)
                best_energy_j = float(energies_j[best_energy_index_j])

                if solver == 'cupy':
                    best_solution_j = bck.asnumpy(candidate_eigenvectors_j[best_energy_index_j])
                else:
                    best_solution_j = candidate_eigenvectors_j[best_energy_index_j]
                best_solution_j = tuple([int(x) for x in best_solution_j])

                # best_solution_j = tuple(best_solution_j)

                if best_energy_j < best_energy:
                    best_energy = best_energy_j
                    best_solution = best_solution_j
                    best_bitstrings = candidate_eigenvectors_j
                    best_result = qaoa_result_j
                    best_arguments = (qaoa_result_j.angles,
                                      qaoa_result_j.hamiltonian_representation_index)

                data_j = {'TrialIndex': [ind_j],
                          'Energy': [best_energy_j],
                          'Bitstring': [best_solution_j],
                          'Angles': [qaoa_result_j.angles],
                          'HamiltonianRepresentationIndex': [hamiltonian_rep_index_j], }

                data_list.append(pd.DataFrame(data_j))

                if return_full_history:
                    full_history.append((best_energy_j, best_solution_j, candidate_eigenvectors_j, qaoa_result_j))

                if store_n_best_solutions != 1:
                    add_args = (tuple(qaoa_result_j.angles),
                                qaoa_result_j.hamiltonian_representation_index)
                    best_results_container.add_multiple_bitstrings_results(bitstrings_array=candidate_eigenvectors_j,
                                                                           energies_array=energies_j,
                                                                           additional_global_specifiers_tuple=add_args,
                                                                           truncation=store_n_best_solutions * 3)


            optimizer_res = OptimizationResult(best_value=best_energy,
                                               best_arguments=best_arguments,
                                               trials_dataframe=pd.concat(data_list))

            if store_n_best_solutions > 1:
                best_res = best_results_container.get_best_results()
            else:
                best_res = [(best_energy, (best_solution, best_arguments))]

            return best_res, (optimizer_res, best_res, full_history)


        else:
            # TODO(FBM): implement this for other QAOA runners
            raise NotImplementedError("This method is only implemented for QAOARunnerExpValues as of now.")


    def estimate_gradient_finite_differences(self,
                                            angles: Union[List[float], np.ndarray],
                                            qaoa_depth: int,
                                            epsilon: float = 1e-5,
                                            number_of_samples: int = None,
                                            measurement_noise: ClassicalMeasurementNoiseSampler = None,
                                            numpy_rng_sampling: np.random.Generator = None,
                                            cvar_fraction: float = 1.0,
                                            method: str = 'central',
                                            **additional_kwargs) -> np.ndarray:
        """
        Estimate gradient using finite differences method.

        This method computes numerical gradients by perturbing each parameter
        independently and evaluating the cost function. Useful for gradient-based
        optimization with noisy/sampled objective functions.

        :param angles: Current QAOA angles (gammas + betas), shape (2*qaoa_depth,)
        :type angles: Union[List[float], np.ndarray]
        :param qaoa_depth: QAOA circuit depth
        :type qaoa_depth: int
        :param epsilon: Perturbation size for finite differences
        :type epsilon: float
        :param number_of_samples: Number of measurement samples (None for exact)
        :type number_of_samples: int, optional
        :param measurement_noise: Classical measurement noise model
        :type measurement_noise: ClassicalMeasurementNoiseSampler, optional
        :param numpy_rng_sampling: Random number generator for sampling
        :type numpy_rng_sampling: np.random.Generator, optional
        :param cvar_fraction: CVaR fraction for risk-aware optimization
        :type cvar_fraction: float
        :param method: 'forward', 'central', or 'backward' finite differences
        :type method: str
        :param additional_kwargs: Additional keyword arguments passed to run_qaoa_wrapped
        :return: Gradient estimate, shape (2*qaoa_depth,)
        :rtype: np.ndarray

        Example:
            runner = QAOAOptimizationRunner(qaoa_runner)
            angles = np.array([0.5, 1.0, 0.3, 0.8])  # depth=2
            gradient = runner.estimate_gradient_finite_differences(
                angles=angles,
                qaoa_depth=2,
                epsilon=1e-4,
                number_of_samples=1000
            )

        .. note::
            - Forward: g[i] ≈ (f(x+ε·eᵢ) - f(x)) / ε  [1 eval per parameter]
            - Central: g[i] ≈ (f(x+ε·eᵢ) - f(x-ε·eᵢ)) / (2ε)  [2 evals per parameter]
            - Backward: g[i] ≈ (f(x) - f(x-ε·eᵢ)) / ε  [1 eval per parameter]

            Central differences are more accurate but require 2x evaluations.
            - On a runner that computes in single precision (`computes_in_single_precision`:
              precision='single' for the statevector runners, precision_float=np.float32 for the
              expectation-values runners, always for the RDM runner), an exact gradient
              (number_of_samples=None) with a step below 1e-3 gives errors of about 1e-3 to
              9e-3 relative, or 1e-4 to 1e-3 with renormalize_probabilities=True, and can flip
              the sign of small components; the method warns. Use epsilon >= 1e-3, or a double
              precision runner. With shots, shot noise dominates and the method does not warn.
        """
        exact = number_of_samples in (None, np.inf)
        if exact and epsilon < 1e-3 and getattr(self._qaoa_runner, 'computes_in_single_precision', False):
            warnings.warn(f"estimate_gradient_finite_differences: epsilon = {epsilon:g} on a runner that computes "
                          f"in single precision. Its rounding gives gradient errors of about 1e-3 to "
                          f"9e-3 relative at steps below 1e-3, or 1e-4 to 1e-3 with renormalize_probabilities=True, "
                          f"and can flip the sign of small components. Use "
                          f"epsilon >= 1e-3, or build the runner with precision='double' "
                          f"(precision_float=np.float64 for the expectation-values runners).",
                          stacklevel=2)
        angles = np.array(angles)
        n_params = len(angles)
        gradient = np.zeros(n_params)

        if method not in ['forward', 'central', 'backward']:
            raise ValueError(f"Invalid method '{method}'. Choose from: 'forward', 'central', 'backward'")

        # Evaluate at current point (needed for forward/backward)
        if method in ['forward', 'backward']:
            f_current = self._qaoa_runner.run_qaoa_wrapped(
                *angles,
                qaoa_depth=qaoa_depth,
                number_of_samples=number_of_samples,
                measurement_noise=measurement_noise,
                numpy_rng_sampling=numpy_rng_sampling,
                input_format=None,
                cvar_fraction=cvar_fraction,
                **additional_kwargs
            )

        # Compute gradient for each parameter
        for i in range(n_params):
            angles_plus = angles.copy()
            angles_plus[i] += epsilon

            f_plus = self._qaoa_runner.run_qaoa_wrapped(
                *angles_plus,
                qaoa_depth=qaoa_depth,
                number_of_samples=number_of_samples,
                measurement_noise=measurement_noise,
                numpy_rng_sampling=numpy_rng_sampling,
                input_format=None,
                cvar_fraction=cvar_fraction,
                **additional_kwargs
            )

            if method == 'forward':
                gradient[i] = (f_plus - f_current) / epsilon

            elif method == 'central':
                angles_minus = angles.copy()
                angles_minus[i] -= epsilon

                f_minus = self._qaoa_runner.run_qaoa_wrapped(
                    *angles_minus,
                    qaoa_depth=qaoa_depth,
                    number_of_samples=number_of_samples,
                    measurement_noise=measurement_noise,
                    numpy_rng_sampling=numpy_rng_sampling,
                    input_format=None,
                    cvar_fraction=cvar_fraction,
                    **additional_kwargs
                )

                gradient[i] = (f_plus - f_minus) / (2 * epsilon)

            elif method == 'backward':
                gradient[i] = (f_current - f_plus) / epsilon

        return gradient