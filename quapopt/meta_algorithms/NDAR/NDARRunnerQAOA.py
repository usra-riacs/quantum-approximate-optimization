# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import BaseSampler as BaseSamplerOptuna

from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import (
    ClassicalMeasurementNoiseSampler,
)
from quapopt.data_analysis.data_handling import MAIN_KEY_VALUE_SEPARATOR as MKVS
from quapopt.data_analysis.data_handling import (
    STANDARD_NAMES_DATA_TYPES as SNDT,
)  # read_results_standardized
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling import LoggingLevel, ResultsLogger
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)
from quapopt.meta_algorithms.NDAR import (
    AttractorModel,
    ConvergenceCriterion,
    NDARIterationResult,
)
from quapopt.meta_algorithms.NDAR.NDARRunner import NDARRunner
from quapopt.optimization import BestResultsContainer, EnergyResultMain
from quapopt.optimization.parameter_setting import OptimizerType
from quapopt.optimization.parameter_setting.non_adaptive_optimization.NonAdaptiveOptimizer import (
    NonAdaptiveOptimizer,
)
from quapopt.optimization.parameter_setting.variational.QAOAOptimizationRunner import (
    QAOAOptimizationRunner,
)
from quapopt.optimization.parameter_setting.variational.scipy_tools.ScipyOptimizerWrapped import (
    ScipyOptimizerWrapped,
)
from quapopt.optimization.QAOA import QAOAResult
from quapopt.optimization.QAOA.implementation.QAOARunnerSampler import QAOARunnerSampler
from quapopt.optimization.QAOA.simulation.QAOARunnerExpValues import QAOARunnerExpValues


class NDARRunnerQAOA(NDARRunner):
    def __init__(
        self,
        input_hamiltonian_representations: List[ClassicalHamiltonian],
        sampler_class: type(QAOARunnerSampler) | type(QAOARunnerExpValues),
        attractor_model: Optional[AttractorModel] = None,
        convergence_criterion: Optional[ConvergenceCriterion] = None,
        logging_level: Optional[LoggingLevel] = None,
        logger_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        This class implements extended version of Noise-Directed Adaptive Remapping.
        A version of this was implemented in paper [1].
        Main differences from the vanilla version are:
        - it can handle multiple Hamiltonian representations
        - it can handle different attractor models
        - it can handle different convergenca criteria

        Refs:
        [1] Maciejewski, Filip B., Jacob Biamonte, Stuart Hadfield, and Davide Venturelli.
        "Improving quantum approximate optimization by noise-directed adaptive remapping."
        arXiv preprint arXiv:2404.01412 (2024).

        Args:
            input_hamiltonian_representations:
            sampler:
            attractor_model:
        """

        super().__init__(
            input_hamiltonian_representations=input_hamiltonian_representations,
            attractor_model=attractor_model,
            convergence_criterion=convergence_criterion,
            logging_level=logging_level,
            logger_kwargs=logger_kwargs,
        )

        self._sampler_class = sampler_class
        self._current_sampler = self._sampler_class(
            hamiltonian_representations_cost=input_hamiltonian_representations,
            logger_kwargs=self._logger_kwargs,
            logging_level=self._logging_level,
            store_n_best_results=1,
        )
        self._results_logger = None
        self._best_parameters = None

    @property
    def sampler_class(self) -> QAOARunnerSampler | QAOARunnerExpValues:
        return self._sampler_class

    @property
    def current_sampler(self):
        return self._current_sampler

    @property
    def results_logger(self) -> Optional[ResultsLogger]:
        """Get current QAOA results logger, creating one if needed."""
        if self._current_sampler is not None:
            return self._current_sampler.results_logger
        return None

    def _get_updated_qaoa_logger_kwargs_for_iteration(self):
        base_kwargs = self._logger_kwargs
        if base_kwargs is None:
            base_kwargs = {}

        if "table_name_suffix" not in base_kwargs:
            base_kwargs["table_name_suffix"] = ""

        base_suffix = base_kwargs["table_name_suffix"]
        ndar_suffix = f"{SNV.NDARIteration.id}{MKVS}{self.ndar_iteration}"
        updated_suffix = self.results_logger.join_table_name_parts(
            [base_suffix, ndar_suffix]
        )
        updated_kwargs = base_kwargs.copy()
        updated_kwargs["table_name_suffix"] = updated_suffix

        return updated_kwargs

    def log_results(
        self,
        ndar_result: NDARIterationResult,
        additional_annotations: Optional[Dict[str, Any]] = None,
    ):

        if self._logging_level in [None, LoggingLevel.NONE]:
            return

        base_kwargs = self._logger_kwargs
        if base_kwargs is None:
            base_kwargs = {}

        if "table_name_suffix" in base_kwargs:
            table_name_suffix = base_kwargs["table_name_suffix"]
        else:
            table_name_suffix = ""

        ndar_overview_df = ndar_result.to_dataframe_main()
        self.results_logger.write_results(
            dataframe=ndar_overview_df,
            data_type=SNDT.NDAROverview,
            additional_annotation_dict=additional_annotations,
            table_name_suffix=table_name_suffix,  # since it's overview of NDAR, we don't want iteration index in table name
        )

    def _apply_genetic_algorithm_to_samples(
        self, samples, cost_hamiltonian, kwargs_GA=None
    ):

        import pygad

        if kwargs_GA is None:
            kwargs_GA = {
                "num_generations": 100,
                "num_parents_mating": 10,
                "keep_parents": -1,
                "mutation_type": "swap",
                "allow_duplicate_genes": True,
                "save_best_solutions": False,
                "suppress_warnings": True,
            }

        kwargs_GA["sol_per_pop"] = samples.shape[0]
        kwargs_GA["num_genes"] = samples.shape[1]

        if cost_hamiltonian.default_backend == "cupy":

            def _cost_function(instance, sol, solution_idx):
                if np.any(sol) > 1:
                    raise ValueError(f"Solution {sol} has values other than 0 or 1")
                return -float(cost_hamiltonian.evaluate_energy(cp.array([sol]))[0])

        else:

            def _cost_function(instance, sol, solution_idx):
                if np.any(sol) > 1:
                    raise ValueError(f"Solution {sol} has values other than 0 or 1")
                return -float(cost_hamiltonian.evaluate_energy([sol])[0])

        ga_instance = pygad.GA(
            fitness_func=_cost_function,
            initial_population=samples,
            gene_type=int,
            **kwargs_GA,
        )
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        energy = cost_hamiltonian.evaluate_energy([solution])[0]
        solution = tuple(solution)

        return (energy, solution), ga_instance



    # TODO(FBM): fix signatures
    def sample_new_solutions(
        self,
        hamiltonian_representations_cost: List[ClassicalHamiltonian],
        qaoa_depth: int,
        number_of_function_calls: int,
        number_of_samples_per_function_call: int = None,
        analytical_betas_p1=True,
        transfer_optimal_parameters: bool = False,
        hamiltonian_representations_phase: List[ClassicalHamiltonian] = None,
        classical_optimizer: Optional[
            Union[NonAdaptiveOptimizer, ScipyOptimizerWrapped, BaseSamplerOptuna]
        ] = None,
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

        if hamiltonian_representations_cost[0].default_backend == "cupy":
            import cupy as bck
        elif hamiltonian_representations_cost[0].default_backend == "numpy":
            import numpy as bck
        else:
            raise ValueError(
                f"Backend {hamiltonian_representations_cost[0].default_backend} not supported"
            )

        # TODO(FBM): this is a mess, refactor

        updated_logger_kwargs = self._get_updated_qaoa_logger_kwargs_for_iteration()

        additional_kwargs_sampler = {}
        additional_kwargs_optimizer = {}

        if self._sampler_class == QAOARunnerExpValues:
            additional_kwargs_optimizer["store_correlators"] = True
            additional_kwargs_optimizer["analytical_betas"] = analytical_betas_p1
            if not add_qrr:
                add_qrr = True
                print("QRR is required for QAOA with exp values")

            additional_kwargs_sampler["store_full_information_in_history"] = True

        if add_GA:
            raise NotImplementedError("GA is not implemented currently")

        qaoa_sampler = self._sampler_class(
            hamiltonian_representations_cost=hamiltonian_representations_cost,
            hamiltonian_representations_phase=hamiltonian_representations_phase,
            logger_kwargs=updated_logger_kwargs,
            logging_level=self.logging_level,
            store_n_best_results=store_n_best_results,
            **additional_kwargs_sampler,
        )
        self._current_sampler = qaoa_sampler

        qaoa_optimizer = QAOAOptimizationRunner(qaoa_runner=qaoa_sampler)

        if isinstance(classical_optimizer, BaseSamplerOptuna):
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        time.perf_counter()

        # if there's no QRR, the parameter transfer is done from iteration 0.
        # if there's only noiseless QRR, then same.
        # but if there's noisy QRR, then we need to transfer from the last iteration

        classical_optimizer_1 = classical_optimizer
        real_number_of_function_calls = number_of_function_calls

        # TODO(FBM): this will work only for custom optimizer, need to improve
        if transfer_optimal_parameters:

            assert not isinstance(classical_optimizer, BaseSamplerOptuna), (
                "If using transfer_optimal_parameters, "
                "classical_optimizer should not be optuna sampler,"
                " use custom optimizer instead"
            )

            assert classical_optimizer.optimizer_type == OptimizerType.custom, (
                "If using transfer_optimal_parameters,"
                " classical_optimizer_type should be custom"
            )
            classical_optimizer_1 = classical_optimizer.copy()
            if qrr_noise is not None and self.ndar_iteration > 1:
                classical_optimizer_1.search_space = self._best_parameters
                real_number_of_function_calls = len(self._best_parameters)
            elif qrr_noise is None and self.ndar_iteration > 0:
                classical_optimizer_1.search_space = self._best_parameters
                real_number_of_function_calls = len(self._best_parameters)

        best_n_results, opt_res_full = qaoa_optimizer.run_optimization(
            qaoa_depth=qaoa_depth,
            number_of_function_calls=real_number_of_function_calls,
            classical_optimizer=classical_optimizer_1,
            optimizer_seed=new_seed,
            number_of_samples=number_of_samples_per_function_call,
            numpy_rng_sampling=numpy_rng_sampling,
            show_progress_bar=show_progress_bars_optimization,
            measurement_noise=measurement_noise,
            **additional_kwargs_optimizer,
        )

        if transfer_optimal_parameters and not add_qrr:
            optimal_params_space = []
            for _, res_i in best_n_results:
                best_qaoa_i = res_i[-1]

                if analytical_betas_p1:
                    flattened_args = [
                        best_qaoa_i[0],
                        best_qaoa_i.hamiltonian_representation_index,
                    ]
                else:
                    flattened_args = [ang for ang in best_qaoa_i.angles] + [
                        best_qaoa_i.hamiltonian_representation_index
                    ]
                optimal_params_space.append(flattened_args)
            self._best_parameters = optimal_params_space

        time.perf_counter()

        if not add_qrr:
            candidate_solutions = []
            for score_i, res_i in best_n_results:
                best_bts_i = res_i[0]
                best_qaoa_i = res_i[-1]
                ham_repr_index_i = best_qaoa_i.hamiltonian_representation_index
                candidate_solutions.append(
                    (score_i, (best_bts_i, ham_repr_index_i, best_qaoa_i))
                )

            return candidate_solutions


        keep_n_results_qrr = store_n_best_results

        best_res_qrr, (opt_res_qrr, _, optimization_history) = (
            qaoa_optimizer.apply_QRR_to_optimization_results(
                return_full_history=qrr_noise is not None,
                show_progress_bar=False,
                store_n_best_solutions=keep_n_results_qrr,
            )
        )

        # for first iteration, we do not add bias to the results
        if qrr_noise is None or self.ndar_iteration == 0:

            if transfer_optimal_parameters and qrr_noise is None:
                optimal_parameters_space = []
                for score_i, (best_bts_i, (angles_i, ham_repr_index_i)) in best_res_qrr:
                    if analytical_betas_p1:
                        flat_args = [angles_i[0], ham_repr_index_i]
                    else:
                        flat_args = [ang for ang in angles_i] + [ham_repr_index_i]
                    optimal_parameters_space.append(flat_args)
                self._best_parameters = optimal_parameters_space

            candidate_solutions = []
            for score_i, (best_bts_i, (angles_i, ham_repr_index_i)) in best_res_qrr:
                qaoa_res = QAOAResult(
                    angles=angles_i,
                    hamiltonian_representation_index=ham_repr_index_i,
                    energy_result=EnergyResultMain(
                        energy_best=float(score_i), bitstring_best=best_bts_i
                    ),
                )

                candidate_solutions.append(
                    (float(score_i), (best_bts_i, ham_repr_index_i, qaoa_res))
                )

            time.perf_counter()
            return candidate_solutions


        # if keep_n_results_qrr!=1:
        best_results_container = BestResultsContainer(
            number_of_best_results=keep_n_results_qrr,
        )

        # If noise is not None, then we need to sample from the noisy distribution for each solution
        candidate_solutions = []

        for _, _, candidate_solutions_i, best_qaoa_i in optimization_history:
            # TODO(FBM): consider whether take just the best X solutions instead of all (some of them are pretty bad);
            # but in practice this shouldn't change the results much, while sorting slows down the code

            ham_rep_index_i = best_qaoa_i.hamiltonian_representation_index
            angles_i = best_qaoa_i.angles
            hamiltonian_i = hamiltonian_representations_cost[ham_rep_index_i]
            candidate_solutions_01 = bck.repeat(
                candidate_solutions_i, repeats=qrr_noise_samples_overhead, axis=0
            )
            candidate_solutions_2 = qrr_noise.add_noise_to_samples(
                ideal_samples=candidate_solutions_01
            )

            energies = hamiltonian_i.evaluate_energy(
                bitstrings_array=candidate_solutions_2
            )

            if keep_n_results_qrr != 1:
                add_args = (tuple(angles_i), ham_rep_index_i)
                best_results_container.add_multiple_bitstrings_results(
                    bitstrings_array=candidate_solutions_2,
                    energies_array=energies,
                    additional_global_specifiers_tuple=add_args,
                    truncation=keep_n_results_qrr * 2,
                )
            else:
                best_energy_index = int(bck.argmin(energies))
                best_energy_i = energies[best_energy_index]

                best_solution_i = candidate_solutions_2[best_energy_index]

                if isinstance(best_solution_i, cp.ndarray):
                    best_solution_i = cp.asnumpy(best_solution_i)

                best_solution_i = tuple(best_solution_i)

                best_results_container.add_result(
                    score=best_energy_i,
                    result_to_add=(best_solution_i, (tuple(angles_i), ham_rep_index_i)),
                )

        time.perf_counter()
        best_res_qrr_noisy = best_results_container.get_best_results()

        if transfer_optimal_parameters:
            optimal_parameters_space = []
            for score_i, (
                best_bts_i,
                (angles_i, ham_repr_index_i),
            ) in best_res_qrr_noisy:
                if analytical_betas_p1:
                    flat_args = [angles_i[0], ham_repr_index_i]
                else:
                    flat_args = [ang for ang in angles_i] + [ham_repr_index_i]

                optimal_parameters_space.append(flat_args)
            self._best_parameters = optimal_parameters_space

        for score, (bts, (angles, ham_repr_index)) in best_res_qrr_noisy:
            qaoa_res = QAOAResult(
                angles=angles,
                hamiltonian_representation_index=ham_repr_index,
                energy_result=EnergyResultMain(
                    energy_best=float(score), bitstring_best=bts
                ),
            )
            candidate_solutions.append(
                (float(score), (bts, ham_repr_index, qaoa_res))
            )
        return candidate_solutions


    def run_NDAR(
        self,
        # QAOA-specific kwargs
        qaoa_depth: int,
        number_of_function_calls: int,
        classical_optimizer: Union[
            NonAdaptiveOptimizer, ScipyOptimizerWrapped, BaseSamplerOptuna
        ],
        number_of_samples_per_function_call: Optional[int] = None,
        measurement_noise: ClassicalMeasurementNoiseSampler = None,
        numpy_rng_sampling: Optional[np.random.Generator] = None,
        show_progress_bars_optimization: bool = False,
        transfer_optimal_parameters: bool = False,
        # exp values simulator kwargs
        analytical_betas_p1=True,
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
        add_mcndar=False,
    ):

        sampler_kwargs = {
            "qaoa_depth": qaoa_depth,
            "number_of_function_calls": number_of_function_calls,
            "number_of_samples_per_function_call": number_of_samples_per_function_call,
            "classical_optimizer": classical_optimizer,
            "measurement_noise": measurement_noise,
            "numpy_rng_sampling": numpy_rng_sampling,
            "transfer_optimal_parameters": transfer_optimal_parameters,
            "show_progress_bars_optimization": show_progress_bars_optimization,
            "analytical_betas_p1": analytical_betas_p1,
            "add_qrr": add_qrr,
            "qrr_noise": qrr_noise,
            "qrr_noise_samples_overhead": qrr_noise_samples_overhead,
            "add_GA": add_GA,
            "kwargs_GA": kwargs_GA,
        }

        return super()._run_NDAR(
            sampler_kwargs=sampler_kwargs,
            optimize_over_n_gauges=optimize_over_n_gauges,
            numpy_rng_boltzmann=numpy_rng_boltzmann,
            step_seed_generator=step_seed_generator,
            show_progress_bar_ndar=show_progress_bar_ndar,
            temperature_NDAR=temperature_NDAR,
            hamming_distance_local_search=hamming_distance_local_search,
            store_full_data_additional=store_full_data_additional,
            df_annotation_add=df_annotation_add,
            verbosity=verbosity_NDAR,
            add_mcndar=add_mcndar,
        )


if __name__ == "__main__":
    print("hej")
