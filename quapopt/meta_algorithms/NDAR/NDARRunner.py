# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import copy
import time
from typing import List, Optional, Dict, Any, Callable, Tuple
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from quapopt import ancillary_functions as anf
from quapopt.data_analysis.data_handling import ResultsLogger
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.meta_algorithms.NDAR import (AttractorStateType,
                                          AttractorModel,
                                          ConvergenceCriterionNames,
                                          ConvergenceCriterion,
                                          NDARIterationResult,
                                          BestResultSignature,
                                          BitstringTypeSignature)

# BitstringTypeSignature = Tuple[int, ...] | np.ndarray | List[int]
# BestResultSignature = Tuple[Tuple[float, BitstringTypeSignature, int], Any]
LocalSamplerSignature = Callable[[List[ClassicalHamiltonian], Any], BestResultSignature]
LocalSamplerKwargsUpdaterSignature = Optional[Callable[[int, BestResultSignature, Optional[Any]], Dict[str, Any] ]]
LoggingCallableSignature = Callable[[NDARIterationResult, ResultsLogger], None]
from quapopt.meta_algorithms.NDAR import NDARIterationResult, handle_bitstring_format_inefficient


class NDARRunner:
    def __init__(self,
                 input_hamiltonian_representation: ClassicalHamiltonian,
                 attractor_model: Optional[AttractorModel] = None,
                 convergence_criterion: Optional[ConvergenceCriterion] = None,
                 numpy_rng_boltzmann: Optional[np.random.Generator] = None
                 ) -> None:
        """
        This class implements extended version of Noise-Directed Adaptive Remapping.
        A version of this was implemented in paper [1].
        Main differences from the vanilla version are:
        - it can handle multiple Hamiltonian representations
        - it can handle different attractor models

        Refs:
        [1]

        Args:
            input_hamiltonian_representations:
            sampler:
            attractor_model:
        """



        self._number_of_qubits = input_hamiltonian_representation.number_of_qubits
        self._input_hamiltonian = input_hamiltonian_representation

        if attractor_model is None:
            attractor_model = AttractorModel(attractor_state_type=AttractorStateType.zero,
                                             number_of_qubits=self._number_of_qubits)

        self._attractor_model = attractor_model
        if convergence_criterion is None:
            convergence_criterion = ConvergenceCriterion(
                convergence_criterion_name=ConvergenceCriterionNames.MaxUnsuccessfulTrials,
                convergence_value=3)
        self._convergence_criterion = convergence_criterion
        self._optimization_history:Dict[int,Tuple[float, BitstringTypeSignature, int]] = {}

        self._ndar_history = []
        self._ndar_iteration = 0
        self._best_energy_so_far = np.inf

        if numpy_rng_boltzmann is None:
            numpy_rng_boltzmann = np.random.default_rng(0)
        self._numpy_rng_boltzmann = numpy_rng_boltzmann

        self._pbar = None

    @property
    def number_of_qubits(self) -> int:
        return self._number_of_qubits

    @property
    def optimization_history(self) -> dict:
        return self._optimization_history

    @property
    def input_hamiltonian(self) -> ClassicalHamiltonian:
        return self._input_hamiltonian

    @property
    def ndar_history(self) -> List[NDARIterationResult]:
        return self._ndar_history

    @property
    def ndar_iteration(self) -> int:
        return self._ndar_iteration

    def increment_ndar_iteration(self):
        self._ndar_iteration += 1

    @property
    def best_energy_so_far(self) -> float:
        return self._best_energy_so_far

    def update_best_energy_so_far(self,
                                  candidate_energy: float):
        if self._best_energy_so_far is None:
            self._best_energy_so_far = candidate_energy
            return True
        else:
            if candidate_energy < self.best_energy_so_far:
                self._best_energy_so_far = candidate_energy
                return True
            return False

    @property
    def attractor_model(self) -> AttractorModel:
        return self._attractor_model

    @property
    def convergence_criterion(self) -> ConvergenceCriterion:
        return self._convergence_criterion

    def check_convergence(self):
        # TODO(FBM): extend this for other criteria

        current_iteration = copy.deepcopy(self.ndar_iteration)
        if current_iteration in [0, 1]:
            # We don't converge at the first iteration
            return False

        # because we check at the BEGINNING of the iteration
        current_iteration += -1
        previous_iteration = current_iteration - 1

        # print('hejka',current_iteration, previous_iteration)

        previous_optimization_results = self._optimization_history[previous_iteration]
        current_optimization_results = self._optimization_history[current_iteration]

        previous_value = previous_optimization_results[0]
        current_value = current_optimization_results[0]

        iteration_index = None
        if self.convergence_criterion.ConvergenceCriterion == ConvergenceCriterionNames.MaxIterations:
            iteration_index = current_iteration

        elif self.convergence_criterion.ConvergenceCriterion == ConvergenceCriterionNames.MaxUnsuccessfulTrials:
            # we need to count how many times we have failed since last improvement
            iteration_index = 0

            best_so_far = np.inf
            for i in range(len(self._optimization_history)):
                E_i = self._optimization_history[i][0]
                if E_i >= best_so_far:
                    iteration_index += 1
                else:
                    best_so_far = E_i
                    iteration_index = 0

                if iteration_index >= self.convergence_criterion.ConvergenceValue:
                    break

        elif self.convergence_criterion.ConvergenceCriterion == ConvergenceCriterionNames.BestEnergyChange:
            pass

        else:
            raise NotImplementedError(
                f"Convergence criterion: {self.convergence_criterion.ConvergenceCriterion} is not implemented")

        converged = self.convergence_criterion.check_convergence(previous_score=previous_value,
                                                                 current_score=current_value,
                                                                 iteration_index=iteration_index)

        # if converged:
        #     print(f"Converged at iteration {current_iteration}; unsuccessful trials: {iteration_index}")

        return converged

    def _metropolis_check(self,
                          energy_current: float,
                          temperature: Optional[float],
                          energy_previous: Optional[float]=None,

                          ):

        if temperature == 0.0:
            return True

        if energy_previous is None:
            energy_previous = self._best_energy_so_far
        if energy_previous is None:
            return True

        if energy_current <= energy_previous:
            return True
        elif temperature is None:
            return False


        dE = energy_current - energy_previous
        return np.log(self._numpy_rng_boltzmann.random(size=1)) >= dE / temperature

    def _handle_single_sampler(self,
                               hamiltonian_representations: List[ClassicalHamiltonian],
                               best_result_previous: BestResultSignature,
                               local_sampler_callable: LocalSamplerSignature,
                               local_sampler_kwargs_updater: LocalSamplerKwargsUpdaterSignature,
                               logging_callable: LoggingCallableSignature,
                               temperature_NDAR: Optional[float],
                               local_sampler_name: Optional[str] = None,
                               additional_data = None
                               #results_logger:Optional[ResultsLogger]=None
                               ):

        #TODO(FBM): I think this inference is fine. We pass more than 1 representation if we wish to optimize over more than 1 gauge.
        optimize_over_r_gauges = len(hamiltonian_representations)

        t0 = time.perf_counter()
        local_sampler_kwargs_i = local_sampler_kwargs_updater(self.ndar_iteration,
                                                              best_result_previous,
                                                              additional_data)

        best_results_i = local_sampler_callable(hamiltonian_representations,
                                                **local_sampler_kwargs_i)
        t1 = time.perf_counter()

        results_logger = local_sampler_kwargs_i.get('results_logger', None)


        # dt_optimization += t1 - t0

        if len(best_results_i) < optimize_over_r_gauges:
            diff = optimize_over_r_gauges - len(best_results_i)
            # Let's make copy and attach
            best_results_i = list(best_results_i)
            for i in range(diff):
                best_results_i.append(copy.deepcopy(best_results_i[0]))
            print("WARNING:",
                  'Not enough results returned by the optimizer. Filling with copies of the best result.')

        best_results_i = sorted(best_results_i, key=lambda x: x[0][0])
        best_energy_current_iteration, best_solution_current_iteration, _ = best_results_i[0][0]
        hamiltonian_representations_to_optimize = []

        last_best_energy = self._best_energy_so_far

        local_results_i = []
        for best_res_i_j in best_results_i:
            (energy_best_i_j, bitstring_best_i_j, representation_index_best_i_j), additional_data_i_j = best_res_i_j
            ham_rep_i_j: ClassicalHamiltonian = hamiltonian_representations[representation_index_best_i_j].copy()

            if self._metropolis_check(energy_current=energy_best_i_j,
                                      energy_previous=last_best_energy,
                                      temperature=temperature_NDAR,
                                      ):
                bitflip_transformation_i = self.attractor_model.return_bitflip_transformation(
                    bitstring=bitstring_best_i_j)
                hamiltonian_transformed_i = ham_rep_i_j.apply_bitflip(bitflip_transformation_i)

            else:
                hamiltonian_transformed_i = ham_rep_i_j
                bitflip_transformation_i = tuple([0] * ham_rep_i_j.number_of_qubits)

            ndar_results_i_j = NDARIterationResult(iteration_index=self.ndar_iteration,
                                                   best_result=best_res_i_j,
                                                   bitflip_transform=bitflip_transformation_i,
                                                   attractor_model=self.attractor_model,
                                                   local_sampler_specific_data=additional_data_i_j,
                                                   convergence_criterion=self.convergence_criterion,
                                                   local_sampler_name=local_sampler_name
                                                   )

            if logging_callable is not None:
                logging_callable(ndar_results_i_j,
                                 results_logger)

            hamiltonian_representations_to_optimize.append(hamiltonian_transformed_i)
            local_results_i.append(ndar_results_i_j)

        return local_results_i, hamiltonian_representations_to_optimize

    def run_NDAR(self,

                 # local_sampler_callables: List[LocalSamplerSignature]|Dict[int,LocalSamplerSignature]|LocalSamplerSignature,
                 # local_sampler_kwargs_updaters: List[LocalSamplerKwargsUpdaterSignature]|Dict[int,LocalSamplerKwargsUpdaterSignature]|LocalSamplerKwargsUpdaterSignature = None,
                 # logging_callables: List[LoggingCallableSignature]|Dict[int,LoggingCallableSignature]|LoggingCallableSignature = None,
                 # local_sampler_names:Optional[List[str]]|Dict[int,str]|str = None,
                 local_sampler_functions = List[Tuple[LocalSamplerSignature, LocalSamplerKwargsUpdaterSignature, LoggingCallableSignature, str]],
                 optimize_over_r_gauges:int=1,
                 show_progress_bar_ndar=True,
                 temperature_NDAR:Optional[float]=None,
                 verbosity=1,
                 initial_bitstrings:Optional[np.ndarray]=None,
                 max_runtime:Optional[float]=None,
                 break_after_finding_ground_state:bool=False



                 ):
        """

        :param local_sampler_functions: LIST of 4-tuples with relevant data for each local sampler.
        The samplers are called sequentially in the order they are provided. This happens at each NDAR iteration.

        Each tuple has a form (sampler_callable, kwargs_updater_callable, logging_callable, local_sampler_name)

        1. sampler_callable: callable with 1 + any number of arguments
        first argument must be:
            - List of ClassicalHamiltonian representations to optimize over
        the rest of arguments are handled via local_sampler_kwargs_updater
        it must return:
            - List of tuples of the form: (energy_best, bitstring_best, representation_index_best), additional_data_i_j

        #Note -- even if the internal optimizer optimizes a different cost function than energy_best, this is what we expect to be returned by the local_sampler_callable
        The list can be length one. Multiple entries mean the best "r" solutions are returned (should match "optimize_over_r_gauges" below)

        2.  kwargs_updater_callable: callable with two arguments
            - first argument must be the NDAR iteration index. This is for possible updating of seeds or some other local_sampler_kwargs
            - second argument must be the best results from previous iteration. This is for updating local sampler based on best results from previous iteration.
        it must return:
            - dictionary with local sampler kwargs to be passed to local_sampler_callable as local_sampler_callable(**kwargs)

        3. logging_callable: callable with 1 argument that must be NDARIterationResult

        4. local_sampler_name: string that will be used to label the local sampler in the NDAR history.



        :param optimize_over_r_gauges:
        Whether to optimize over more than single representation of the Hamiltonian each time.
        It should be matched with the number of results returned by the local_sampler_callable.
        If it is not, we are filling the list with copies of the best result.

        :param numpy_rng_boltzmann:
        This is rng for metropolis checks
        :param temperature_NDAR:
        This is temperature for metropolis checks. If it is 0.0 (default), all proposals are accepted.
        :param initial_solution:
        Initial solutions to apply to the input Hamiltonian representations before starting NDAR.

        :param show_progress_bar_ndar:
        :param verbosity:

        :return:
        """

        self.clean_optimization_history()

        t0_total = time.perf_counter()





        hamiltonian_representations_to_optimize = [self.input_hamiltonian.copy() for _ in range(optimize_over_r_gauges)]


        if initial_bitstrings is None and optimize_over_r_gauges>1:
            numpy_rng = self._numpy_rng_boltzmann
            random_gauges = numpy_rng.binomial(n=1, p=0.5, size=(optimize_over_r_gauges - 1, self.number_of_qubits))
            initial_bitstrings = np.array([0] * self.number_of_qubits).reshape(1, -1)
            initial_bitstrings = np.concatenate([initial_bitstrings, random_gauges],
                                                axis=0)

        if initial_bitstrings is not None:
            if isinstance(initial_bitstrings, (list,tuple)):
                initial_bitstrings = np.array(initial_bitstrings)

            if isinstance(initial_bitstrings, np.ndarray):
                if len(initial_bitstrings.shape)==1:
                    initial_bitstrings = initial_bitstrings.reshape(1,-1)
                    initial_bitstrings = np.repeat(initial_bitstrings,
                                                   optimize_over_r_gauges,
                                                   axis=0)
                else:
                    assert initial_bitstrings.shape[0]==optimize_over_r_gauges, ("initial_bitstrings must have shape "
                                                                                 "(N, optimize_over_r_gauges)"
                                                                                 " if it is not a 1D array."
                                                                                 f"The detected shape is: {initial_bitstrings.shape}"
                                                                                 )

            best_results_curr = []
            for i, hamiltonian_representation_i in enumerate(hamiltonian_representations_to_optimize):
                hamiltonian_representations_to_optimize[i] = hamiltonian_representation_i.copy().apply_bitflip(initial_bitstrings[i].tolist())
                en_i = hamiltonian_representations_to_optimize[i].compute_zero_energy()

                best_res_i_init:Tuple[Tuple[float, BitstringTypeSignature, int], Optional[Any]] = ((en_i, initial_bitstrings[i], i), None)

                # ndar_results_i_init = NDARIterationResult(iteration_index=self.ndar_iteration,
                #                                        best_result=best_res_i_init,
                #                                        bitflip_transform=initial_bitstrings[i],
                #                                        attractor_model=self.attractor_model,
                #                                        local_sampler_specific_data=None,
                #                                        convergence_criterion=self.convergence_criterion,
                #                                        local_sampler_name="InitialBitstrings"
                #                                        )


                self.update_best_energy_so_far(candidate_energy=en_i)

                best_results_curr.append(best_res_i_init)



        else:
            best_results_curr: List[BestResultSignature] = [((np.inf, tuple([0] * self.number_of_qubits), 0),
                                                             None)] * optimize_over_r_gauges

        if show_progress_bar_ndar:
            if self.convergence_criterion.ConvergenceCriterion == ConvergenceCriterionNames.MaxIterations:
                max_iterations = self.convergence_criterion.ConvergenceValue
            else:
                max_iterations = 10 ** 3
            self._pbar = tqdm(total=max_iterations, colour='blue', position=0)

        ground_state_energy = hamiltonian_representations_to_optimize[0].lowest_energy


        best_energy_curr, best_solution_curr, best_rep_index_curr = best_results_curr[0][0]


        if verbosity > 0:
            anf.cool_print("Starting NDAR with the following samplers:", [x[3] for x in local_sampler_functions], 'blue')

        t_start_ndar = time.perf_counter()
        dt_optimization = 0.0
        while not self.check_convergence():
            hamiltonian_representations_to_optimize: List[ClassicalHamiltonian] = [x.copy() for x in hamiltonian_representations_to_optimize]

            local_results_ndar_i = []
            for (local_sampler, local_updater, logging_handler, local_sampler_name) in local_sampler_functions:
                local_results_i, hamiltonian_representations_to_optimize = self._handle_single_sampler(
                                                                                hamiltonian_representations=hamiltonian_representations_to_optimize,
                                                                                best_result_previous=best_results_curr,
                                                                                local_sampler_callable=local_sampler,
                                                                                local_sampler_kwargs_updater=local_updater,
                                                                                logging_callable=logging_handler,
                                                                                local_sampler_name=local_sampler_name,
                                                                                temperature_NDAR=temperature_NDAR)

                best_results_i: NDARIterationResult = local_results_i[0]
                local_results_ndar_i.append(local_results_i)


                if best_results_i.best_energy < best_energy_curr:
                    best_energy_curr = best_results_i.best_energy
                    best_solution_curr = best_results_i.best_bitstring
                    best_rep_index_curr = best_results_i.best_hamiltonian_representation_index
                    best_results_curr = best_results_i.best_result

                    #TODO(FBM): should make a heap to include multiple best results checks
                    # best_results_curr = [x.best_result for x in local_results_i]


                did_update_energy = self.update_best_energy_so_far(candidate_energy=best_energy_curr)

                if self._pbar is not None and did_update_energy:
                    _postfix = f"{self.best_energy_so_far:.4f}"
                    ar = self.input_hamiltonian.calculate_approximation_ratio(energy=self.best_energy_so_far)
                    if ar is not None:
                        _postfix += f" (AR={ar:.5f})"
                    _postfix += f"; it = {self.ndar_iteration}"
                    _postfix += f"; ({local_sampler_name})"

                    self._pbar.set_postfix(BestCost=_postfix)

                if max_runtime is not None and time.perf_counter()-t_start_ndar > max_runtime:
                    anf.cool_print("Breaking NDAR loop because:",  "max_runtime was reached.", 'red',)
                    break

            self._ndar_history.append(local_results_ndar_i)
            self._optimization_history[self._ndar_iteration] = (best_energy_curr,
                                                                best_solution_curr,
                                                                best_rep_index_curr)

            if self._pbar is not None:
                self._pbar.update(1)

            self._ndar_iteration += 1
            #print('hejka', ground_state_energy, self.best_energy_so_far)

            if ground_state_energy is not None:
                #TODO(FBM): in the past, we'd break the optimization here, but not anymore.
                if abs(ground_state_energy - self.best_energy_so_far) < 1e-4:

                    if break_after_finding_ground_state:
                        anf.cool_print("FOUND ~GROUND STATE!", 'Breaking!', 'cyan')

                        break
                    anf.cool_print("FOUND ~GROUND STATE!", 'cool :-)', 'cyan')

                    # anf.cool_print("FOUND GROUND STATE!", 'breaking NDAR loop', 'cyan')
                    # break

            if max_runtime is not None and time.perf_counter() - t_start_ndar > max_runtime:
                break

        if self._pbar is not None:
            self._pbar.close()

        t1_total = time.perf_counter()

        dt_total = t1_total - t0_total

        if verbosity > 0:
            anf.cool_print("Finished after ", f'{self._ndar_iteration} iterations.', 'blue')
            _best_energy_str = f"{self.best_energy_so_far:.4f}"
            ar = self.input_hamiltonian.calculate_approximation_ratio(energy=self.best_energy_so_far)
            if ar is not None:
                _best_energy_str += f" (AR={ar:.5f})"

            anf.cool_print('Final best energy:', _best_energy_str, 'blue')
            anf.cool_print("Total time:", dt_total, 'yellow')
            anf.cool_print("Out of which calls to local sampler:", dt_optimization, 'yellow')
        optimization_history = self._optimization_history

        optimization_history_values = list(optimization_history.values())
        optimization_history_values_sorted = sorted(optimization_history_values, key=lambda x: x[0])
        best_res = optimization_history_values_sorted[0]

        return best_res, self._ndar_history

    def clean_optimization_history(self):
        self._optimization_history = {}
        self._best_energy_so_far = np.inf
        self._ndar_history = []
        self._ndar_iteration = 0
        self._pbar = None


    def history_to_dataframe(self):

        all_results = []
        #Iterate over NDAR iterations
        for ndar_iteration_index in range(len(self._ndar_history)):
            results_here_iter = self._ndar_history[ndar_iteration_index]
            #Iterate over local samplers
            for results_here_iter in results_here_iter:
                best_energy = np.inf
                best_df = None

                #Iterate over best results within the solver
                for res_here_iter in results_here_iter:
                    res_here_iter: NDARIterationResult

                    if res_here_iter.best_energy < best_energy:
                        best_energy = res_here_iter.best_energy
                        best_df = res_here_iter.to_dataframe_main()

                all_results.append(best_df)

        return pd.concat(all_results, ignore_index=True,axis=0)

            #     result_here_iter.ndar_iteration = ndar_iteration_index


        # return pd.DataFrame(self._optimization_history)