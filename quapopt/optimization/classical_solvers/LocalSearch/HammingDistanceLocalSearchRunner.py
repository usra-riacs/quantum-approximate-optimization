# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import time
from typing import Optional, Union

import numpy as np
from tqdm.notebook import tqdm

from quapopt import ancillary_functions as anf
from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)
from quapopt.meta_algorithms.NDAR import ConvergenceCriterion, ConvergenceCriterionNames

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

import pandas as pd

from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV


class HammingDistanceLocalSearchRunner:
    def __init__(
        self,
        hamiltonian: ClassicalHamiltonian,
    ):
        self._hamiltonian = hamiltonian
        self._number_of_qubits = hamiltonian.number_of_qubits

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @staticmethod
    def _metropolis_check(
        energy_current: float,
        energy_previous: float,
        temperature: float,
        rng: Union[cp.random.Generator, np.random.Generator] = None,
    ):

        if energy_current <= energy_previous or temperature == 0.0:
            return True

        dE = energy_current - energy_previous

        return np.log(rng.uniform(0, 1)) >= dE / temperature

    def run_HDLS(
        self,
        hamming_distance: int,
        number_of_trials_max: int,
        initial_bitstring: Optional[np.ndarray] = None,
        verbosity=1,
        temperature=1.0,
        convergence_criterion=None,
        show_progress_bar=False,
        solver_timeout=None,
        sampler_seed: Optional[int] = None,
        store_full_data=False,
    ):
        """
        This function runs the Local Distance Landscape Search algorithm.
        Args:
            hamming_distance: the hamming distance to be used in the search
            number_of_steps: the number of steps to be taken in the search
            initial_bitstring: the initial bitstring to be used in the search
        Returns:
            Tuple[str, float]: the bitstring and the energy value of the bitstring

        """

        # TODO FBM: refactor cupy vs numba.cuda vs python etc.
        t0_total = time.perf_counter()

        if convergence_criterion is None:
            convergence_criterion = ConvergenceCriterion(
                convergence_criterion_name=ConvergenceCriterionNames.MaxUnsuccessfulTrials,
                convergence_value=50,
            )

        convergence_description = convergence_criterion.get_description_string()

        highest_energy = self.hamiltonian.highest_energy
        ground_state_energy = self.hamiltonian.ground_state_energy
        delta = highest_energy - ground_state_energy

        # TODO(FBM): fix this

        if hamming_distance > 2:
            import numpy as bck
        else:
            import cupy as bck

        if hamming_distance == 1:
            bitstrings_neighbors = bck.eye(
                self.hamiltonian.number_of_qubits, dtype=bck.int8
            )
        elif hamming_distance == 2:
            bitstrings_neighbors = em.get_all_two_1s_bitstrings_cython(
                self.hamiltonian.number_of_qubits, True
            )
        elif hamming_distance == 3:
            bitstrings_neighbors = em.get_all_three_1s_bitstrings_cython(
                self.hamiltonian.number_of_qubits, True
            )
        else:
            raise ValueError(
                "Only hamming_distance_local_search in [1,2] is implemented."
            )
        numpy_rng = bck.random.default_rng(sampler_seed)

        bitstrings_neighbors = bck.asarray(bitstrings_neighbors)

        zeros_n = bck.zeros(self.hamiltonian.number_of_qubits, dtype=bck.int8)
        if initial_bitstring is None:
            bitstring_mask = zeros_n.copy()
        else:
            bitstring_mask = bck.array(initial_bitstring, dtype=int)

        best_bitstring_so_far = zeros_n.copy()
        best_energy_so_far = np.inf

        unsuccesful_trials = 0

        results_all = []
        for counter in tqdm(
            list(range(number_of_trials_max)),
            colour="blue",
            disable=not show_progress_bar,
        ):

            # find best bitstring -> generate 2-local neighborhood of that bitstring.

            bitstrings_i_masked = bitstrings_neighbors ^ bitstring_mask

            energies_masked_i = self.hamiltonian.evaluate_energy(
                bitstrings_array=bitstrings_i_masked
            )

            best_energy_index = int(bck.argmin(energies_masked_i))
            best_energy_i = float(energies_masked_i[best_energy_index])
            best_bitstring_i = bitstrings_i_masked[best_energy_index]

            applied_mask = np.zeros(self.hamiltonian.number_of_qubits, dtype=int)
            if best_energy_i < best_energy_so_far:
                unsuccesful_trials = 0
                best_energy_so_far = best_energy_i
                best_bitstring_so_far = best_bitstring_i
                bitstring_mask = best_bitstring_i
                applied_mask = bitstring_mask

                if verbosity >= 1:
                    ar_i = (highest_energy - best_energy_so_far) / (
                        highest_energy - ground_state_energy
                    )
                    anf.cool_print(
                        f"STEP {counter + 1} AR (energy): ",
                        f"{np.round(ar_i, 3)} ({np.round(best_energy_so_far, 3)})",
                        "green",
                    )
            else:
                unsuccesful_trials += 1
                # TODO FBM: since we include the original bitstring always, we always have dE = 0
                # maybe let's try taking the next bitstring with different energy
                energies_masked_i[best_energy_index] = np.inf
                second_best_energy_index = np.argmin(energies_masked_i)
                second_best_energy_i = energies_masked_i[second_best_energy_index]

                while second_best_energy_i == best_energy_i:
                    energies_masked_i[second_best_energy_index] = np.inf
                    second_best_energy_index = np.argmin(energies_masked_i)
                    second_best_energy_i = energies_masked_i[second_best_energy_index]

                best_energy_i = energies_masked_i[second_best_energy_index]
                best_bitstring_i = bitstrings_i_masked[second_best_energy_index]

                if self._metropolis_check(
                    energy_current=best_energy_i,
                    energy_previous=best_energy_so_far,
                    temperature=temperature,
                    rng=numpy_rng,
                ):

                    bitstring_mask = best_bitstring_i
                    applied_mask = bitstring_mask

                best_energy_i = float(best_energy_i)
                best_energy_so_far = float(best_energy_so_far)

                ar_best_energy_i = (highest_energy - best_energy_i) / delta
                ar_best_energy_so_far = (highest_energy - best_energy_so_far) / delta
                if store_full_data:
                    df_here = pd.DataFrame(
                        data={
                            SNV.Seed.id_long: [sampler_seed],
                            SNV.IterationIndex.id_long: [counter],
                            "HammingDistance": [hamming_distance],
                            SNV.Temperature.id_long: [temperature],
                            "TemperatureMultiplier": [1.0],
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
                    results_all.append(df_here)

            if (
                convergence_criterion.ConvergenceCriterion
                == ConvergenceCriterionNames.MaxUnsuccessfulTrials
            ):
                if unsuccesful_trials > convergence_criterion.ConvergenceValue:
                    break
            else:
                raise NotImplementedError("Convergence criterion not implemented.")
            if solver_timeout is not None:
                t1 = time.perf_counter()
                if t1 - t0_total > solver_timeout:
                    break

        if store_full_data:
            df_res = pd.concat(results_all, axis=0)
        else:
            df_res = None

        return (bck.asnumpy(best_bitstring_so_far), float(best_energy_so_far)), df_res
