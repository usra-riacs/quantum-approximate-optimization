# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 

import time
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple, Callable
from qiskit.transpiler import StagedPassManager
from qiskit_ibm_runtime.fake_provider import FakeAthensV2

from quapopt.circuits.backend_utilities.qiskit import QiskitSessionManagerMixin
from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.circuits.gates.gate_delays import DelaySchedulerBase
from quapopt.optimization.QAOA import (PhaseSeparatorType,
                                       MixerType,
                                       QubitMappingType)

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError, ModuleNotFoundError):
    import numpy as cp

import numpy as np
import optuna
from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import ClassicalMeasurementNoiseSampler
from quapopt.data_analysis.data_handling import (STANDARD_NAMES_DATA_TYPES as SNDT,
                                                 STANDARD_NAMES_VARIABLES as SNV,
                                                 MAIN_KEY_VALUE_SEPARATOR as MKVS,
                                                 LoggingLevel,
                                                 )
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.meta_algorithms.NDAR import (AttractorModel,
                                          ConvergenceCriterion,
                                          NDARIterationResult)
from quapopt.meta_algorithms.NDAR.NDARRunner import NDARRunner
from quapopt.optimization import EnergyResultMain
from quapopt.optimization.QAOA import QAOAResultsLogger
from quapopt.optimization.QAOA.implementation.QAOARunnerSampler import QAOARunnerSampler

from quapopt.optimization.parameter_setting import OptimizerType
from quapopt.optimization.parameter_setting.variational.QAOAOptimizationRunner import QAOAOptimizationRunner
from quapopt.optimization.QAOA import QAOAResult
from quapopt.optimization import BestResultsContainer
from quapopt.optimization.parameter_setting.non_adaptive_optimization.NonAdaptiveOptimizer import NonAdaptiveOptimizer
from quapopt.optimization.parameter_setting.variational.scipy_tools.ScipyOptimizerWrapped import ScipyOptimizerWrapped
from optuna.samplers import BaseSampler as BaseSamplerOptuna
from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf

class NDARRunnerQAOAQiskit(NDARRunner, QiskitSessionManagerMixin):
    def __init__(self,
                 input_hamiltonian_representations: List[ClassicalHamiltonian],
                 # Qiskit-specific configuration
                 qiskit_pass_manager: StagedPassManager,
                 qiskit_backend,
                 program_gate_builder: AbstractProgramGateBuilder = None,
                 number_of_qubits_device_qiskit: Optional[int] = None,
                 qubit_indices_physical: tuple = None,
                 classical_indices=None,
                 enforce_no_ancilla_qubits: bool = True,
                 # QAOA circuit configuration
                 qaoa_depth: int = 1,
                 time_block_size=None,
                 qubit_mapping_type=QubitMappingType.sabre,
                 phase_separator_type=PhaseSeparatorType.QAOA,
                 mixer_type=MixerType.QAOA,
                 every_gate_has_its_own_parameter: bool = False,
                 add_barriers: bool = False,
                 # Session management
                 simulation: bool = True,
                 qiskit_sampler_options: Optional[dict] = None,
                 mock_context_manager_if_simulated: bool = True,
                 session_ibm=None,
                 # Standard NDAR parameters
                 attractor_model: Optional[AttractorModel] = None,
                 convergence_criterion: Optional[ConvergenceCriterion] = None,
                 logging_level: Optional[LoggingLevel] = None,
                 logger_kwargs: Optional[Dict[str, Any]] = None,
                 delay_scheduler:DelaySchedulerBase=None
                 ) -> None:
        """
        NDAR (Noise-Directed Adaptive Remapping) runner specifically designed for Qiskit backends.
        
        This class combines NDAR meta-algorithm capabilities with direct Qiskit quantum hardware
        execution, providing efficient session management across NDAR iterations.
        
        Features:
        - Master session management for entire NDAR run
        - Automatic QAOARunnerSampler instance creation and initialization of qiskit backend with shared sessions
        - Context manager protocol for resource cleanup
        - Full compatibility with IBM Quantum hardware and simulators
        
        Args:
            input_hamiltonian_representations: List of Hamiltonian representations for NDAR
            qiskit_pass_manager: Qiskit transpilation pass manager
            qiskit_backend: Qiskit backend (defaults to FakeAthensV2 simulator)
            program_gate_builder: Gate builder for parameter resolution
            simulation: Whether to use simulation mode (affects session management)
            session_ibm: External session to reuse (optional)
            
            ... (additional parameters follow QAOARunnerSampler and NDARRunner patterns)
        """

        # Initialize parent NDARRunner
        super().__init__(
            input_hamiltonian_representations=input_hamiltonian_representations,
            attractor_model=attractor_model,
            convergence_criterion=convergence_criterion,
            logging_level=logging_level,
            logger_kwargs=logger_kwargs
        )


        self._delay_scheduler = delay_scheduler

        # Initialize session management via mixin
        self._init_session_management(
            qiskit_backend=qiskit_backend,
            simulation=simulation,
            mock_context_manager_if_simulated=mock_context_manager_if_simulated,
            session_ibm=session_ibm
        )

        if self._current_session is None:
            # Create a new session if not provided externally
            self.start_session()

            # Store Qiskit configuration for creating initializing qiskit backend in QAOARunnerSampler instances
        self._qiskit_config = {
            'qiskit_pass_manager': qiskit_pass_manager,
            'qiskit_backend': qiskit_backend,
            'program_gate_builder': program_gate_builder,
            'number_of_qubits_device_qiskit': number_of_qubits_device_qiskit,
            'qubit_indices_physical': qubit_indices_physical,
            'classical_indices': classical_indices,
            'enforce_no_ancilla_qubits': enforce_no_ancilla_qubits,
            'qaoa_depth': qaoa_depth,
            'time_block_size': time_block_size,
            'qubit_mapping_type': qubit_mapping_type,
            'phase_separator_type': phase_separator_type,
            'mixer_type': mixer_type,
            'every_gate_has_its_own_parameter': every_gate_has_its_own_parameter,
            'add_barriers': add_barriers,
            'simulation': simulation,
            'qiskit_sampler_options': qiskit_sampler_options,
            'session_ibm': self._current_session,
            'delay_scheduler': delay_scheduler
        }

        self._current_sampler = QAOARunnerSampler(hamiltonian_representations_cost=input_hamiltonian_representations,
                                                  logger_kwargs=self._logger_kwargs,
                                                  logging_level=self._logging_level,
                                                  store_n_best_results=1)


        self._current_sampler.initialize_backend_qiskit(**self._qiskit_config)

    @property
    def current_sampler(self):
        return self._current_sampler

    @property
    def results_logger(self) -> Optional[QAOAResultsLogger]:
        """Get current QAOA results logger from the active QAOARunnerQiskit instance."""
        if self._current_sampler is not None:
            return self._current_sampler.results_logger
        return None

    def _get_updated_qaoa_logger_kwargs_for_iteration(self):
        """Generate logger kwargs with NDAR iteration suffix."""
        base_kwargs = self._logger_kwargs
        if base_kwargs is None:
            base_kwargs = {}

        if 'table_name_suffix' not in base_kwargs:
            base_kwargs['table_name_suffix'] = ''

        base_suffix = base_kwargs['table_name_suffix']
        ndar_suffix = f"{SNV.NDARIteration.id}{MKVS}{self.ndar_iteration}"

        # Create a simple concatenation with separator if both exist
        if base_suffix:
            updated_suffix = f"{base_suffix}_{ndar_suffix}"
        else:
            updated_suffix = ndar_suffix

        updated_kwargs = base_kwargs.copy()
        updated_kwargs['table_name_suffix'] = updated_suffix

        return updated_kwargs

    def log_results(self,
                    ndar_result: NDARIterationResult,
                    additional_annotations: Optional[Dict[str, Any]] = None):

        if self._logging_level in [None, LoggingLevel.NONE]:
            return

        ndar_overview_df = ndar_result.to_dataframe_main()
        ndar_overview_datatype = SNDT.NDAROverview
        self.results_logger.write_results(dataframe=ndar_overview_df,
                                          data_type=ndar_overview_datatype,
                                          additional_annotation_dict=additional_annotations)

    # TODO(FBM): fix signatures
    def sample_new_solutions(self,
                             hamiltonian_representations_cost: List[ClassicalHamiltonian],

                             qaoa_depth: int,
                             number_of_function_calls: int,
                             number_of_samples_per_function_call: int = None,

                             analytical_betas_p1=True,
                             transfer_optimal_parameters: bool = False,

                             hamiltonian_representations_phase: List[ClassicalHamiltonian] = None,
                             classical_optimizer: Optional[
                                 Union[NonAdaptiveOptimizer, ScipyOptimizerWrapped, BaseSamplerOptuna]] = None,

                             measurement_noise: ClassicalMeasurementNoiseSampler = None,

                             numpy_rng_sampling: Optional[np.random.Generator] = None,
                             show_progress_bars_optimization: bool = False,

                             store_n_best_results: int = 1,

                             add_qrr=False,
                             qrr_noise: Optional[ClassicalMeasurementNoiseSampler] = None,
                             qrr_noise_samples_overhead=1,

                             add_GA=False,
                             kwargs_GA: Optional[Dict[str, Any]] = None,
                             new_seed: Optional[int] = None,

                             ) -> List[Tuple[float, Tuple[np.ndarray, int, QAOAResult]]]:
        # TODO(FBM): this is a mess, refactor; also should probably inherit from NDARRunnerQAOA

        if hamiltonian_representations_cost[0].default_backend == 'cupy':
            import cupy as bck
        elif hamiltonian_representations_cost[0].default_backend == 'numpy':
            import numpy as bck
        else:
            raise ValueError(f"Backend {hamiltonian_representations_cost[0].default_backend} not supported")

        updated_logger_kwargs = self._get_updated_qaoa_logger_kwargs_for_iteration()
        additional_kwargs_sampler = {
                                    #'numpy_rng_sampling': np.random.default_rng(seed=new_seed)
                                     }
        additional_kwargs_optimizer = {}

        qaoa_sampler = QAOARunnerSampler(
            hamiltonian_representations_cost=hamiltonian_representations_cost,
            hamiltonian_representations_phase=hamiltonian_representations_phase,
            logger_kwargs=updated_logger_kwargs,
            logging_level=self._logging_level,
            store_n_best_results=store_n_best_results,
            **additional_kwargs_sampler
        )

        qaoa_sampler.initialize_backend_qiskit(**self._qiskit_config)

        self._current_sampler = qaoa_sampler

        qaoa_optimizer = QAOAOptimizationRunner(qaoa_runner=qaoa_sampler)

        if isinstance(classical_optimizer, BaseSamplerOptuna):
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        t0 = time.perf_counter()

        # if there's no QRR, the parameter transfer is done from iteration 0.
        # if there's only noiseless QRR, then same.
        # but if there's noisy QRR, then we need to transfer from the last iteration

        classical_optimizer_1 = classical_optimizer
        real_number_of_function_calls = number_of_function_calls

        # TODO(FBM): this will work only for custom optimizer, need to improve
        if transfer_optimal_parameters:
            assert not isinstance(classical_optimizer, BaseSamplerOptuna), ("If using transfer_optimal_parameters, "
                                                                            "classical_optimizer should not be optuna sampler,"
                                                                            " use custom optimizer instead")

            assert classical_optimizer.optimizer_type == OptimizerType.custom, ("If using transfer_optimal_parameters,"
                                                                                " classical_optimizer_type should be custom")
            classical_optimizer_1 = classical_optimizer.copy()
            if qrr_noise is not None and self.ndar_iteration > 1:
                classical_optimizer_1.search_space = self._best_parameters
                real_number_of_function_calls = len(self._best_parameters)
            elif qrr_noise is None and self.ndar_iteration > 0:
                classical_optimizer_1.search_space = self._best_parameters
                real_number_of_function_calls = len(self._best_parameters)

        best_n_results, opt_res_full = qaoa_optimizer.run_optimization(qaoa_depth=qaoa_depth,
                                                                       number_of_function_calls=real_number_of_function_calls,
                                                                       classical_optimizer=classical_optimizer_1,
                                                                       optimizer_seed=new_seed,
                                                                       number_of_samples=number_of_samples_per_function_call,
                                                                       numpy_rng_sampling=numpy_rng_sampling,
                                                                       show_progress_bar=show_progress_bars_optimization,
                                                                       measurement_noise=measurement_noise,
                                                                       **additional_kwargs_optimizer
                                                                       )

        if transfer_optimal_parameters and not add_qrr:
            optimal_params_space = []
            for _, res_i in best_n_results:
                best_qaoa_i = res_i[-1]

                if analytical_betas_p1:
                    flattened_args = [best_qaoa_i[0], best_qaoa_i.hamiltonian_representation_index]
                else:
                    flattened_args = [ang for ang in best_qaoa_i.angles] + [
                        best_qaoa_i.hamiltonian_representation_index]
                optimal_params_space.append(flattened_args)
            self._best_parameters = optimal_params_space

        t1 = time.perf_counter()

        # print("optimization time:", t1-t0)
        if not add_qrr:
            if add_GA:
                raise NotImplementedError()
            else:
                candidate_solutions = []
                for score_i, res_i in best_n_results:
                    best_bts_i = res_i[0]
                    best_qaoa_i = res_i[-1]
                    ham_repr_index_i = best_qaoa_i.hamiltonian_representation_index
                    candidate_solutions.append((score_i, (best_bts_i, ham_repr_index_i, best_qaoa_i)))

                return candidate_solutions

        if add_GA:
            keep_n_results_qrr = kwargs_GA['sol_per_pop']
        else:
            keep_n_results_qrr = store_n_best_results

        best_res_qrr, (opt_res_qrr, _, optimization_history) = qaoa_optimizer.apply_QRR_to_optimization_results(
            return_full_history=qrr_noise is not None,
            show_progress_bar=False,
            store_n_best_solutions=keep_n_results_qrr)

        # for first iteration, we do not add bias to the results
        if qrr_noise is None or self.ndar_iteration == 0:
            # sorted_history = sorted(optimization_history, key=lambda x: x[0])

            if transfer_optimal_parameters and qrr_noise is None:
                optimal_parameters_space = []
                for score_i, (best_bts_i, (angles_i, ham_repr_index_i)) in best_res_qrr:
                    if analytical_betas_p1:
                        flat_args = [angles_i[0], ham_repr_index_i]
                    else:
                        flat_args = [ang for ang in angles_i] + [ham_repr_index_i]
                    optimal_parameters_space.append(flat_args)
                self._best_parameters = optimal_parameters_space

            if not add_GA:
                candidate_solutions = []
                for score_i, (best_bts_i, (angles_i, ham_repr_index_i)) in best_res_qrr:
                    qaoa_res = QAOAResult(angles=angles_i,
                                          hamiltonian_representation_index=ham_repr_index_i,
                                          energy_result=EnergyResultMain(energy_best=float(score_i),
                                                                         bitstring_best=best_bts_i))

                    candidate_solutions.append((float(score_i), (best_bts_i, ham_repr_index_i, qaoa_res)))

                t2 = time.perf_counter()
                return candidate_solutions
            else:
                all_present_ham_indices = set([best_res_qrr[i][1][1][1] for i in range(kwargs_GA['sol_per_pop'])])

                best_results_container_GA = BestResultsContainer(number_of_best_results=store_n_best_results)

                for ham_repr_index_i in all_present_ham_indices:
                    solutions_i = [best_res_qrr[i][1][0] for i in range(kwargs_GA['sol_per_pop']) if
                                   best_res_qrr[i][1][1][1] == ham_repr_index_i]

                    kwargs_GA_i = kwargs_GA.copy()

                    if 'num_parents_mating' in kwargs_GA_i:
                        kwargs_GA_i['num_parents_mating'] = min(kwargs_GA_i['num_parents_mating'], len(solutions_i))

                    (energy, solution), ga_instance_i = self._apply_genetic_algorithm_to_samples(
                        samples=np.array(solutions_i),
                        cost_hamiltonian=hamiltonian_representations_cost[ham_repr_index_i],
                        kwargs_GA=kwargs_GA_i)
                    if store_n_best_results > 1:
                        solutions_ga_i = bck.array(ga_instance_i.best_solutions)
                        energies_here = -bck.array(ga_instance_i.best_solutions_fitness)

                        best_results_container_GA.add_multiple_bitstrings_results(bitstrings_array=solutions_ga_i,
                                                                                  energies_array=energies_here,
                                                                                  additional_global_specifiers_tuple=(
                                                                                  None, ham_repr_index_i),
                                                                                  truncation=None)

                    else:
                        best_results_container_GA.add_result(score=float(energy),
                                                             result_to_add=(tuple(solution), (None, ham_repr_index_i)))

                # print(best_results_container_GA.get_best_results()[0][1], 'yo')
                best_results_container_GA.add_another_heap(other_container=best_res_qrr)
                # angles_i = None
                candidate_solutions = []
                for score_i, (best_bts_i, (angles_i, ham_repr_index_i)) in best_results_container_GA.get_best_results():
                    qaoa_res_i = QAOAResult(angles=angles_i,
                                            hamiltonian_representation_index=ham_repr_index_i,
                                            energy_result=EnergyResultMain(energy_best=float(score_i),
                                                                           bitstring_best=best_bts_i))
                    candidate_solutions.append((float(score_i), (best_bts_i, ham_repr_index_i, qaoa_res_i)))

                return candidate_solutions

        # if keep_n_results_qrr!=1:
        best_results_container = BestResultsContainer(number_of_best_results=keep_n_results_qrr, )

        # If noise is not None, then we need to sample from the noisy distribution for each solution
        candidate_solutions = []

        for _, _, candidate_solutions_i, best_qaoa_i in optimization_history:
            # TODO(FBM): consider whether take just the best X solutions instead of all (some of them are pretty bad);
            # but in practice this shouldn't change the results much, while sorting slows down the code

            ham_rep_index_i = best_qaoa_i.hamiltonian_representation_index
            angles_i = best_qaoa_i.angles
            hamiltonian_i = hamiltonian_representations_cost[ham_rep_index_i]
            candidate_solutions_01 = bck.repeat(candidate_solutions_i,
                                                repeats=qrr_noise_samples_overhead,
                                                axis=0)
            candidate_solutions_2 = qrr_noise.add_noise_to_samples(ideal_samples=candidate_solutions_01)

            energies = hamiltonian_i.evaluate_energy(bitstrings_array=candidate_solutions_2)

            if keep_n_results_qrr != 1:
                add_args = (tuple(angles_i),
                            ham_rep_index_i)
                best_results_container.add_multiple_bitstrings_results(bitstrings_array=candidate_solutions_2,
                                                                       energies_array=energies,
                                                                       additional_global_specifiers_tuple=add_args,
                                                                       truncation=keep_n_results_qrr * 2)
            else:
                best_energy_index = int(bck.argmin(energies))
                best_energy_i = energies[best_energy_index]

                best_solution_i = candidate_solutions_2[best_energy_index]

                if isinstance(best_solution_i, cp.ndarray):
                    best_solution_i = cp.asnumpy(best_solution_i)

                best_solution_i = tuple(best_solution_i)

                best_results_container.add_result(score=best_energy_i,
                                                  result_to_add=(best_solution_i,
                                                                 (tuple(angles_i),
                                                                  ham_rep_index_i))
                                                  )

        t2 = time.perf_counter()
        best_res_qrr_noisy = best_results_container.get_best_results()

        # print(best_res_qrr_noisy)

        if transfer_optimal_parameters:
            optimal_parameters_space = []
            for score_i, (best_bts_i, (angles_i, ham_repr_index_i)) in best_res_qrr_noisy:
                if analytical_betas_p1:
                    flat_args = [angles_i[0], ham_repr_index_i]
                else:
                    flat_args = [ang for ang in angles_i] + [ham_repr_index_i]

                optimal_parameters_space.append(flat_args)
            self._best_parameters = optimal_parameters_space

        if not add_GA:
            for score, (bts, (angles, ham_repr_index)) in best_res_qrr_noisy:
                qaoa_res = QAOAResult(angles=angles,
                                      hamiltonian_representation_index=ham_repr_index,
                                      energy_result=EnergyResultMain(energy_best=float(score),
                                                                     bitstring_best=bts))
                candidate_solutions.append((float(score), (bts, ham_repr_index, qaoa_res)))
            return candidate_solutions


        else:
            t0 = time.perf_counter()

            if len(best_res_qrr_noisy) < kwargs_GA['sol_per_pop']:
                diff = kwargs_GA['sol_per_pop'] - len(best_res_qrr_noisy)
                # add solutions to fill the population
                amount_of_present = len(best_res_qrr_noisy)
                counter = 0
                for i in range(diff):
                    # add random solutions
                    best_res_qrr_noisy.append(best_res_qrr_noisy[counter])
                    counter += 1
                    if counter >= amount_of_present:
                        counter = 0

            all_present_ham_indices = set([best_res_qrr_noisy[i][1][1][1] for i in range(kwargs_GA['sol_per_pop'])])

            best_results_container_GA = BestResultsContainer(number_of_best_results=store_n_best_results)

            for ham_repr_index_i in all_present_ham_indices:
                solutions_i = [best_res_qrr_noisy[i][1][0] for i in range(kwargs_GA['sol_per_pop']) if
                               best_res_qrr_noisy[i][1][1][1] == ham_repr_index_i]

                kwargs_GA_i = kwargs_GA.copy()

                if 'num_parents_mating' in kwargs_GA_i:
                    kwargs_GA_i['num_parents_mating'] = min(kwargs_GA_i['num_parents_mating'], len(solutions_i))

                (energy, solution), ga_instance_i = self._apply_genetic_algorithm_to_samples(
                    samples=np.array(solutions_i),
                    cost_hamiltonian=hamiltonian_representations_cost[ham_repr_index_i],
                    kwargs_GA=kwargs_GA_i)
                if store_n_best_results > 1:
                    solutions_ga_i = bck.array(ga_instance_i.best_solutions)
                    energies_here = -bck.array(ga_instance_i.best_solutions_fitness)
                    best_results_container_GA.add_multiple_bitstrings_results(bitstrings_array=solutions_ga_i,
                                                                              energies_array=energies_here,
                                                                              additional_global_specifiers_tuple=(
                                                                                  None, ham_repr_index_i),
                                                                              truncation=None
                                                                              )
                else:
                    best_results_container_GA.add_result(score=float(energy),
                                                         result_to_add=(tuple(solution), (None, ham_repr_index_i)))

            best_results_container_GA.add_another_heap(other_container=best_res_qrr_noisy)
            # angles_i = None
            candidate_solutions = []
            for score_i, (best_bts_i, (angles_i, ham_repr_index_i)) in best_results_container_GA.get_best_results():
                qaoa_res_i = QAOAResult(angles=angles_i,
                                        hamiltonian_representation_index=ham_repr_index_i,
                                        energy_result=EnergyResultMain(energy_best=float(score_i),
                                                                       bitstring_best=best_bts_i))
                candidate_solutions.append((float(score_i), (best_bts_i, ham_repr_index_i, qaoa_res_i)))

            return candidate_solutions
    def run_NDAR(self,
                 # QAOA-specific kwargs
                 qaoa_depth: int,
                 number_of_function_calls: int,
                 classical_optimizer: Union[NonAdaptiveOptimizer, ScipyOptimizerWrapped, BaseSamplerOptuna],
                 number_of_samples_per_function_call: Optional[int] = None,

                 measurement_noise: ClassicalMeasurementNoiseSampler = None,
                 numpy_rng_sampling: Optional[np.random.Generator] = None,
                 show_progress_bars_optimization: bool = False,
                 transfer_optimal_parameters: bool = False,

                 # exp values simulator kwargs
                 analytical_betas_p1=False,

                 # QRR kwargs
                 add_qrr=False,
                 qrr_noise: Optional[ClassicalMeasurementNoiseSampler] = None,
                 qrr_noise_samples_overhead: int = 1,

                 # GA kwargs
                 add_GA=False,
                 kwargs_GA=None,

                 # Standard NDAR kwargs
                 optimize_over_n_gauges: int = 1,
                 numpy_rng_boltzmann=None,
                 step_seed_generator: Optional[Callable[[int], int]] = None,
                 show_progress_bar_ndar=True,
                 verbosity_NDAR=1,
                 temperature_NDAR=0.0,
                 hamming_distance_local_search=None,

                 store_full_data_additional: bool = False,
                 df_annotation_add: Optional[pd.DataFrame] = None,

                 add_mcndar=False

                 ):

        sampler_kwargs = {'qaoa_depth': qaoa_depth,
                          'number_of_function_calls': number_of_function_calls,
                          'number_of_samples_per_function_call': number_of_samples_per_function_call,
                          'classical_optimizer': classical_optimizer,
                          'measurement_noise': measurement_noise,
                          'numpy_rng_sampling': numpy_rng_sampling,

                          'transfer_optimal_parameters': transfer_optimal_parameters,

                          'show_progress_bars_optimization': show_progress_bars_optimization,
                          'analytical_betas_p1': analytical_betas_p1,
                          'add_qrr': add_qrr,
                          'qrr_noise': qrr_noise,
                          'qrr_noise_samples_overhead': qrr_noise_samples_overhead,

                          'add_GA': add_GA,
                          'kwargs_GA': kwargs_GA,

                          }

        return super()._run_NDAR(sampler_kwargs=sampler_kwargs,
                                 optimize_over_n_gauges=optimize_over_n_gauges,
                                 numpy_rng_boltzmann=numpy_rng_boltzmann,
                                 step_seed_generator=step_seed_generator,
                                 show_progress_bar_ndar=show_progress_bar_ndar,
                                 temperature_NDAR=temperature_NDAR,
                                 hamming_distance_local_search=hamming_distance_local_search,
                                 store_full_data_additional=store_full_data_additional,
                                 df_annotation_add=df_annotation_add,
                                 verbosity=verbosity_NDAR,
                                 add_mcndar=add_mcndar
                                 )
