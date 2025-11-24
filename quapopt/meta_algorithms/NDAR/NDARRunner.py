# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import copy
import time
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from quapopt import ancillary_functions as anf
from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling import LoggingLevel
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)
from quapopt.meta_algorithms.NDAR import (
    AttractorModel,
    AttractorStateType,
    ConvergenceCriterion,
    ConvergenceCriterionNames,
    NDARIterationResult,
)
from quapopt.optimization import EnergyResultMain
from quapopt.optimization.QAOA import QAOAResult


# define class for NDAR implementation
class NDARRunner(ABC):
    def __init__(
        self,
        input_hamiltonian_representations: List[ClassicalHamiltonian],
        # sampler_class: type(HamiltonianSolutionsSampler),
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

        Refs:
        [1]

        Args:
            input_hamiltonian_representations:
            sampler:
            attractor_model:
        """

        if isinstance(input_hamiltonian_representations, ClassicalHamiltonian):
            input_hamiltonian_representations = [input_hamiltonian_representations]

        self.number_of_representations = len(input_hamiltonian_representations)

        self._number_of_qubits = input_hamiltonian_representations[0].number_of_qubits
        zero_tuple = tuple([0] * self._number_of_qubits)
        self._input_hamiltonians = input_hamiltonian_representations
        self._transformations_history = {
            0: (ham, zero_tuple) for ham in self._input_hamiltonians
        }
        self._ndar_iteration = 0

        if attractor_model is None:
            attractor_model = AttractorModel(
                attractor_state_type=AttractorStateType.zero,
                number_of_qubits=self._number_of_qubits,
            )

        self._attractor_model = attractor_model
        if convergence_criterion is None:
            convergence_criterion = ConvergenceCriterion(
                convergence_criterion_name=ConvergenceCriterionNames.best_energy_change,
            )
        self._convergence_criterion = convergence_criterion

        self._optimization_history = {}

        self._best_energy_so_far = np.inf

        if logging_level is None:
            logging_level = LoggingLevel.NONE
        if logger_kwargs is None:
            logger_kwargs = {}

        self._logging_level = logging_level
        self._logger_kwargs = logger_kwargs

    @property
    def number_of_qubits(self) -> int:
        return self._number_of_qubits

    @property
    def optimization_history(self) -> dict:
        return self._optimization_history

    @property
    def logging_level(self) -> LoggingLevel:
        return self._logging_level

    def set_logging_level(self, logging_level: LoggingLevel):
        self._logging_level = logging_level

    @property
    def input_hamiltonians(self) -> List[ClassicalHamiltonian]:
        return self._input_hamiltonians

    @property
    def transformations_history(self) -> dict:
        return self._transformations_history

    @property
    def ndar_iteration(self) -> int:
        return self._ndar_iteration

    def increment_ndar_iteration(self):
        self._ndar_iteration += 1

    @property
    def best_energy_so_far(self) -> float:
        return self._best_energy_so_far

    def update_best_energy_so_far(self, candidate_energy: float):
        if self._best_energy_so_far is None:
            self._best_energy_so_far = candidate_energy
        else:
            if candidate_energy < self.best_energy_so_far:
                self._best_energy_so_far = candidate_energy

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

        previous_optimization_results = self._optimization_history[previous_iteration]
        current_optimization_results = self._optimization_history[current_iteration]

        previous_value = previous_optimization_results[0]
        current_value = current_optimization_results[0]

        iteration_index = None
        if (
            self.convergence_criterion.ConvergenceCriterion
            == ConvergenceCriterionNames.MaxIterations
        ):
            iteration_index = current_iteration

        elif (
            self.convergence_criterion.ConvergenceCriterion
            == ConvergenceCriterionNames.MaxUnsuccessfulTrials
        ):
            # we need to count how many times we have failed since last improvement
            iteration_index = 0

            best_so_far = np.inf
            for i in range(current_iteration):
                E_i = self._optimization_history[i][0]
                if E_i >= best_so_far:
                    iteration_index += 1
                else:
                    best_so_far = E_i
                    iteration_index = 0

                if iteration_index > self.convergence_criterion.ConvergenceValue:
                    break

        elif (
            self.convergence_criterion.ConvergenceCriterion
            == ConvergenceCriterionNames.BestEnergyChange
        ):
            pass

        else:
            raise NotImplementedError(
                f"Convergence criterion: {self.convergence_criterion.ConvergenceCriterion} is not implemented"
            )

        converged = self.convergence_criterion.check_convergence(
            previous_score=previous_value,
            current_score=current_value,
            iteration_index=iteration_index,
        )

        # if converged:

        return converged

    # @abstractmethod
    # def _sample_new_solutions(self,
    #                           hamiltonian_representations_cost: List[ClassicalHamiltonian],
    #                           *args,
    #                           **kwargs) -> List[Tuple[float, Tuple[np.ndarray, int, Any]]]:

    def sample_new_solutions(
        self,
        hamiltonian_representations_cost: List[ClassicalHamiltonian],
        new_seed: Optional[int] = None,
        *args,
        **kwargs,
    ) -> List[Tuple[float, Tuple[np.ndarray, int, Any]]]:
        raise NotImplementedError("This method must be implemented in a subclass.")

        # return self._sample_new_solutions(hamiltonian_representations_cost=hamiltonian_representations_cost,
        #                                   *args,
        #                                   **kwargs)

    @staticmethod
    def _metropolis_check(
        energy_current: float,
        energy_previous: float,
        temperature: float,
        rng: Union[cp.random.Generator, np.random.Generator] = None,
    ):

        if temperature == 0.0:
            return True

        if energy_current <= energy_previous:
            return True

        dE = energy_current - energy_previous

        return np.log(rng.uniform(0, 1)) >= dE / temperature

    def _run_NDAR(
        self,
        sampler_kwargs: dict,
        optimize_over_n_gauges=1,
        numpy_rng_boltzmann=None,
        step_seed_generator: Optional[Callable[[int], int]] = None,
        show_progress_bar_ndar=True,
        temperature_NDAR=0.0,
        hamming_distance_local_search=None,
        store_full_data_additional: Optional[bool] = False,
        df_annotation_add: Optional[pd.DataFrame] = None,
        verbosity=1,
        initial_bitstring=None,
        add_mcndar=False,
    ):

        self.clean_optimization_history()

        t0_total = time.perf_counter()
        if numpy_rng_boltzmann is None:
            numpy_rng_boltzmann = np.random.default_rng()
        if step_seed_generator is None:
            step_seed_generator = lambda integer: None

        hamiltonian_representations_to_optimize = self.input_hamiltonians

        if initial_bitstring is not None:
            for i, hamiltonian_representation_i in enumerate(
                hamiltonian_representations_to_optimize
            ):
                hamiltonian_representations_to_optimize[i] = (
                    hamiltonian_representation_i.apply_bitflip(initial_bitstring)
                )

        if sampler_kwargs is None:
            sampler_kwargs = {}

        sampler_kwargs = sampler_kwargs.copy()

        if optimize_over_n_gauges > 1:
            # TODO(FBM): implement this
            raise NotImplementedError(
                "NDAR does not support optimize_over_n_gauges > 1 yet. "
            )

        if "store_n_best_results" in sampler_kwargs:
            if optimize_over_n_gauges > sampler_kwargs["store_n_best_results"]:
                sampler_kwargs["store_n_best_results"] = optimize_over_n_gauges
                print(
                    f"Overwriting store_n_best_results to {optimize_over_n_gauges} because"
                    f" optimize_over_n_gauges is set to {optimize_over_n_gauges}"
                )
        else:
            sampler_kwargs["store_n_best_results"] = optimize_over_n_gauges

        if show_progress_bar_ndar:
            if (
                self.convergence_criterion.ConvergenceCriterion
                == ConvergenceCriterionNames.MaxIterations
            ):
                max_iterations = self.convergence_criterion.ConvergenceValue
            else:
                max_iterations = 10**3

            pbar = tqdm(total=max_iterations, colour="blue", position=0)

        ground_state_energy = hamiltonian_representations_to_optimize[0].lowest_energy
        highest_energy = hamiltonian_representations_to_optimize[0].highest_energy

        if highest_energy is None:
            highest_energy = 0
        delta = highest_energy - ground_state_energy

        dt_optimization = 0

        if hamming_distance_local_search == 0:
            hamming_distance_local_search = None

        if hamming_distance_local_search is not None:
            from quapopt import AVAILABLE_SIMULATORS

            if "cupy" in AVAILABLE_SIMULATORS:
                bck = cp
            else:
                bck = np

            if hamming_distance_local_search == 1:
                bitstrings_neighbors = bck.eye(self.number_of_qubits, dtype=bck.int8)
            elif hamming_distance_local_search == 2:
                bitstrings_neighbors = em.get_all_two_1s_bitstrings_cython(
                    self.number_of_qubits, True
                )
            else:
                raise ValueError(
                    "Only hamming_distance_local_search in [1,2] is implemented."
                )

            bitstrings_neighbors = bck.asarray(bitstrings_neighbors)

        convergence_description = self.convergence_criterion.get_description_string()

        results_all_add = []
        while not self.check_convergence():
            # TODO FBM: remember about indices for qiskit
            t0 = time.perf_counter()
            hamiltonian_representations_to_optimize_raw = [
                x.copy() for x in hamiltonian_representations_to_optimize
            ]

            new_seed_i = step_seed_generator(self.ndar_iteration)
            best_results_i = self.sample_new_solutions(
                hamiltonian_representations_cost=hamiltonian_representations_to_optimize_raw,
                new_seed=new_seed_i,
                **sampler_kwargs,
            )

            dt_optimization += time.perf_counter() - t0

            if len(best_results_i) < optimize_over_n_gauges:
                diff = optimize_over_n_gauges - len(best_results_i)

                # Let's make copy and attach
                best_results_i = best_results_i.copy()
                for i in range(diff):
                    best_results_i.append(best_results_i[0])

                print(
                    "WARNING:",
                    "Not enough results returned by the optimizer. Filling with copies of the best result.",
                )

            # TODO FBM: maybe make this handled outside this function
            if self.ndar_iteration == 0:
                last_best_energy = np.inf
            else:
                last_best_energy = self._optimization_history[self._ndar_iteration - 1][
                    0
                ]

            best_energy_current_iteration = best_results_i[0][0]
            best_solution_current_iteration = best_results_i[0][1][0]

            hamiltonian_representations_to_optimize = []
            for score_i, res_i in best_results_i:
                best_bitstring_i: Union[Tuple[int, ...], np.ndarray] = res_i[0]
                best_bitstring_i = tuple(best_bitstring_i)

                hamiltonian_representation_index_i: int = res_i[1]

                # TODO(FBM): this is used only for logging. Should generalize to something like "OptimizationResult"
                best_qaoa_i: QAOAResult = res_i[2]
                # if not isinstance(best_qaoa_i, QAOAResult):

                hamiltonian_representation_here: ClassicalHamiltonian = (
                    hamiltonian_representations_to_optimize_raw[
                        hamiltonian_representation_index_i
                    ].copy()
                )

                if self._metropolis_check(
                    energy_current=score_i,
                    energy_previous=last_best_energy,
                    temperature=temperature_NDAR,
                    rng=numpy_rng_boltzmann,
                ):
                    bitflip_transformation_i = (
                        self.attractor_model.return_bitflip_transformation(
                            bitstring=best_bitstring_i
                        )
                    )
                    t0 = time.perf_counter()
                    hamiltonian_transformed_i = (
                        hamiltonian_representation_here.apply_bitflip(
                            bitflip_transformation_i
                        )
                    )
                    time.perf_counter()

                    #

                    # raise KeyboardInterrupt
                else:
                    hamiltonian_transformed_i = hamiltonian_representation_here
                    bitflip_transformation_i = tuple(
                        [0] * hamiltonian_representation_here.number_of_qubits
                    )

                # TODO(FBM): this should not depend on "QAOAResult" object!
                ndar_results_i = NDARIterationResult(
                    iteration_index=self.ndar_iteration,
                    bitflip_transform=bitflip_transformation_i,
                    attractor_model=self.attractor_model,
                    qaoa_result=best_qaoa_i,
                    convergence_criterion=self.convergence_criterion,
                )

                self.log_results(ndar_result=ndar_results_i)

                if add_mcndar:
                    convergence_criterion_mcndar = ConvergenceCriterion(
                        convergence_criterion_name=ConvergenceCriterionNames.MaxUnsuccessfulTrials,
                        convergence_value=20,
                    )

                    mcndar_kwargs = {
                        "number_of_samples_per_trial": 10000,
                        "number_of_trials_max": 1000,
                        "p_01": 1 - 3 / self.number_of_qubits,
                        "verbosity": 0,
                        "temperature": 1.0,
                        "temperature_mult": 1.0,
                        "convergence_criterion": convergence_criterion_mcndar,
                        "show_progress_bar": False,
                        "solver_name": "MCNDAR",
                        "solver_seed": new_seed_i,
                    }
                    from quapopt.optimization.classical_solvers import (
                        SolverNames,
                        solve_ising_hamiltonian,
                    )

                    (best_bitstring_mcndar, best_energy_mcndar), df_trials_ = (
                        solve_ising_hamiltonian(
                            hamiltonian=hamiltonian_transformed_i,
                            solver_name=SolverNames.local_search,
                            solver_kwargs=mcndar_kwargs,
                            repetitions=1,
                            show_progress_bar=False,
                        )
                    )

                    best_energy_mcndar = float(best_energy_mcndar)
                    best_bitstring_mcndar = tuple(best_bitstring_mcndar.tolist())

                    best_qaoa_mcndar = QAOAResult(
                        energy_result=EnergyResultMain(
                            energy_best=best_energy_mcndar,
                            bitstring_best=best_bitstring_mcndar,
                        ),
                        angles=best_qaoa_i.angles,
                        hamiltonian_representation_index=hamiltonian_representation_index_i,
                    )

                    if self._metropolis_check(
                        energy_current=best_energy_mcndar,
                        energy_previous=score_i,
                        temperature=temperature_NDAR,
                        rng=numpy_rng_boltzmann,
                    ):
                        bitflip_transformation_mcndar = (
                            self.attractor_model.return_bitflip_transformation(
                                bitstring=best_bitstring_mcndar
                            )
                        )
                        t0 = time.perf_counter()
                        hamiltonian_transformed_i = (
                            hamiltonian_transformed_i.apply_bitflip(
                                bitflip_transformation_mcndar
                            )
                        )
                        time.perf_counter()
                    else:
                        bitflip_transformation_mcndar = tuple(
                            [0] * hamiltonian_representation_here.number_of_qubits
                        )

                    ndar_results_mcndar_i = NDARIterationResult(
                        iteration_index=self.ndar_iteration,
                        bitflip_transform=bitflip_transformation_mcndar,
                        attractor_model=self.attractor_model,
                        qaoa_result=best_qaoa_mcndar,
                        convergence_criterion=self.convergence_criterion,
                    )
                    self.log_results(ndar_result=ndar_results_mcndar_i)

                    score_i = best_energy_mcndar

                    if score_i < best_energy_current_iteration:
                        best_energy_current_iteration = score_i
                        best_solution_current_iteration = best_bitstring_mcndar

                if hamming_distance_local_search is not None:
                    energies_hdls_i = hamiltonian_transformed_i.evaluate_energy(
                        bitstrings_array=bitstrings_neighbors
                    )
                    best_index_hdls_i = bck.argmin(energies_hdls_i)
                    best_energy_hdls_i = float(energies_hdls_i[best_index_hdls_i])
                    best_bitstring_hdls_i = tuple(
                        bitstrings_neighbors[best_index_hdls_i, :].get().tolist()
                    )

                    best_qaoa_hdls_i = QAOAResult(
                        energy_result=EnergyResultMain(
                            energy_best=best_energy_hdls_i,
                            bitstring_best=best_bitstring_hdls_i,
                        ),
                        angles=best_qaoa_i.angles,
                        hamiltonian_representation_index=hamiltonian_representation_index_i,
                    )

                    if self._metropolis_check(
                        energy_current=best_energy_hdls_i,
                        energy_previous=score_i,
                        temperature=temperature_NDAR,
                        rng=numpy_rng_boltzmann,
                    ):

                        bitflip_transformation_hdls_i = (
                            self.attractor_model.return_bitflip_transformation(
                                bitstring=best_bitstring_hdls_i
                            )
                        )

                        t0 = time.perf_counter()
                        hamiltonian_transformed_i = (
                            hamiltonian_transformed_i.apply_bitflip(
                                bitflip_transformation_hdls_i
                            )
                        )
                        time.perf_counter()
                    else:
                        bitflip_transformation_hdls_i = tuple(
                            [0] * hamiltonian_representation_here.number_of_qubits
                        )

                    ndar_results_hdls_i = NDARIterationResult(
                        iteration_index=self.ndar_iteration,
                        bitflip_transform=bitflip_transformation_hdls_i,
                        attractor_model=self.attractor_model,
                        qaoa_result=best_qaoa_hdls_i,
                        convergence_criterion=self.convergence_criterion,
                    )

                    self.log_results(ndar_result=ndar_results_hdls_i)
                    score_i = best_energy_hdls_i

                    if score_i < best_energy_current_iteration:
                        best_energy_current_iteration = score_i
                        best_solution_current_iteration = best_bitstring_hdls_i

                hamiltonian_representations_to_optimize.append(
                    hamiltonian_transformed_i
                )

                best_energy_i = float(best_energy_current_iteration)
                best_energy_so_far = min(
                    [float(self.best_energy_so_far), best_energy_i]
                )

                ar_best_energy_i = (highest_energy - best_energy_i) / delta
                ar_best_energy_so_far = (highest_energy - best_energy_so_far) / delta

                best_bitstring_so_far = tuple(best_solution_current_iteration)
                # TODO(FBM): mcndar is not considered, fix this
                applied_mask = tuple(bitflip_transformation_i)

                if store_full_data_additional:
                    assert (
                        optimize_over_n_gauges == 1
                    ), "Currently, we only support storing full data for one gauge."
                    df_here = pd.DataFrame(
                        data={
                            #  SNV.Seed.id_long: [sampler_seed],
                            SNV.IterationIndex.id_long: [self._ndar_iteration],
                            # SNV.FlipProbability.id_long: [p_01],
                            # SNV.Temperature.id_long: [temperature],
                            SNV.ApproximationRatio.id_long: [ar_best_energy_i],
                            SNV.ApproximationRatioBest.id_long: [ar_best_energy_so_far],
                            SNV.Energy.id_long: [best_energy_i],
                            SNV.EnergyBest.id_long: [best_energy_so_far],
                            SNV.Runtime.id_long: [time.perf_counter() - t0_total],
                            SNV.ConvergenceCriterion.id_long: [convergence_description],
                            SNV.Bitstring.id_long: [
                                tuple([int(x) for x in best_bitstring_so_far])
                            ],
                            SNV.Bitflip.id_long: [
                                tuple([int(x) for x in applied_mask])
                            ],
                        }
                    )
                    if df_annotation_add is not None:
                        df_here = pd.concat([df_here, df_annotation_add], axis=1)

                    results_all_add.append(df_here)

            self.update_best_energy_so_far(
                candidate_energy=best_energy_current_iteration
            )
            self._optimization_history[self._ndar_iteration] = (
                best_energy_current_iteration,
                best_solution_current_iteration,
            )

            if verbosity > 0:
                anf.cool_print("Iteration: ", self._ndar_iteration, "blue")
                energy_str = f"{self.best_energy_so_far}"
                if ground_state_energy is not None and highest_energy is not None:
                    ar_best_so_far = (highest_energy - self.best_energy_so_far) / (
                        highest_energy - ground_state_energy
                    )
                    energy_str += f" (AR: {np.round(ar_best_so_far, 4)})"

                anf.cool_print("Best energy so far: ", energy_str, "green")

            self._ndar_iteration += 1
            if show_progress_bar_ndar:
                pbar.update(1)

            if ground_state_energy is not None:
                if abs(ground_state_energy - self.best_energy_so_far) < 1e-6:
                    if verbosity > 0:
                        anf.cool_print("FOUND GROUND STATE!", "breaking", "cyan")
                        print()
                    break

        if show_progress_bar_ndar:
            pbar.close()

        if verbosity > 0:
            anf.cool_print(
                "Finished after ", f"{self._ndar_iteration} iterations.", "red"
            )
            anf.cool_print("Final best energy:", self.best_energy_so_far, "red")
            anf.cool_print("Optimization time:", dt_optimization, "red")

        optimization_history = self._optimization_history

        optimization_history_values = list(optimization_history.values())
        optimization_history_values_sorted = sorted(
            optimization_history_values, key=lambda x: x[0]
        )
        best_res = optimization_history_values_sorted[0]

        if store_full_data_additional:
            df_res = pd.concat(results_all_add, axis=0)
            return best_res, df_res
        else:
            return best_res, self._optimization_history

    def run_NDAR(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass.")

    # def _log_results(self,
    #                  ndar_result: NDARIterationResult):

    def log_results(
        self,
        ndar_result: NDARIterationResult,
    ):

        raise NotImplementedError("This method must be implemented in a subclass.")
        # if self._logging_level not in [None, LoggingLevel.NONE]:

    def clean_optimization_history(self):
        self._optimization_history = {}
        self._best_energy_so_far = np.inf
        zero_tuple = tuple([0] * self._number_of_qubits)
        self._transformations_history = {
            0: (ham, zero_tuple) for ham in self._input_hamiltonians
        }
        self._ndar_iteration = 0

    def _print_energy_progress_nicely(
        self,
        current_energy: float,
        best_energy_so_far: float,
        verbosity: int,
        step_information: Optional[str] = None,
    ):
        if verbosity <= 0:
            return

        highest_energy = self._input_hamiltonians[0].highest_energy
        lowest_energy = self._input_hamiltonians[0].lowest_energy

        if step_information is None:
            step_information = ""

        print_ar = False
        if highest_energy is not None and lowest_energy is not None:
            delta = highest_energy - lowest_energy
            ar_current = (highest_energy - current_energy) / delta
            ar_previous = (highest_energy - best_energy_so_far) / delta
            print_ar = True

        if current_energy < best_energy_so_far:
            main_text = f"Found new minimum!"
            if step_information is not None:
                main_text += f"({step_information})"

            if print_ar:
                sub_text = (
                    f"AR: {np.round(ar_previous, 4)} --> {np.round(ar_current, 4)}"
                )
            else:
                sub_text = f"Energy: {np.round(best_energy_so_far, 4)} --> {np.round(current_energy, 4)}"

            anf.cool_print(main_text, sub_text, color_name="yellow")
        elif verbosity >= 2:
            main_text = f"Current value ({step_information}):"

            if print_ar:
                sub_text = f"AR: {np.round(ar_current, 4)}"
            else:
                sub_text = f"Energy: {np.round(current_energy, 4)}"

            anf.cool_print(main_text, sub_text, color_name="blue")
