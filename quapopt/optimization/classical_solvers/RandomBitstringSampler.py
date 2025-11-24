# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling import LoggingLevel
from quapopt.optimization import HamiltonianSolutionsSampler


class RandomBitstringSampler(HamiltonianSolutionsSampler):

    def __init__(
        self,
        cost_hamiltonian: ClassicalHamiltonian,
        backend=None,
        max_memory=None,
        logging_level: Optional[LoggingLevel] = None,
        logger_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(
            input_hamiltonian_representations_cost=[cost_hamiltonian],
            solve_at_initialization=False,
            number_of_qubits=cost_hamiltonian.number_of_qubits,
            logging_level=logging_level,
            logger_kwargs=logger_kwargs,
        )

        if backend is None:
            backend = cost_hamiltonian.default_backend

        if cost_hamiltonian.default_backend != backend:
            cost_hamiltonian.reinitialize_backend(backend=backend)

        self._cost_hamiltonian = cost_hamiltonian
        self._backend = backend
        self._number_of_qubits = cost_hamiltonian.number_of_qubits
        if max_memory is None:
            if backend == "cupy":
                max_memory = int(1 * 10**7)
            elif backend == "numpy":
                max_memory = int(1 * 10**9)
            else:
                raise ValueError(f"Backend {backend} not supported")

        self._max_memory = max_memory

    def _sample_solutions(
        self,
        number_of_samples_per_trial: int,
        number_of_trials: int = 1,
        seed: int = None,
        show_progress_bar: bool = False,
        return_all_results: bool = False,
    ):

        # OK, so I can fit "max_memory" of bytes in total
        # this means I can fit this amount of bitsrings:
        max_bitstrings_in_memory = int(self._max_memory // self._number_of_qubits)

        if self._backend == "cupy":
            bck = cp
        elif self._backend == "numpy":
            bck = np
        else:
            raise ValueError(f"Backend {self._backend} not supported")

        rng = bck.random.default_rng(seed=seed)

        lowest_energy = self._cost_hamiltonian.lowest_energy
        highest_energy = self._cost_hamiltonian.highest_energy
        delta = highest_energy - lowest_energy

        best_energy_so_far = np.inf
        best_state_so_far = None

        all_bitstrings = []
        all_energies = []

        results_dfs = []

        t0_total = time.perf_counter()

        pbar_trials = tqdm(
            list(range(number_of_trials)),
            disable=(not show_progress_bar) or number_of_trials == 1,
            position=0,
            colour="cyan",
        )
        for trial_index in pbar_trials:
            # If user needs less samples, I will just use that amount
            if number_of_samples_per_trial <= max_bitstrings_in_memory:
                batch_size = number_of_samples_per_trial
                number_of_batches = 1
            else:
                batch_size = max_bitstrings_in_memory

                number_of_batches = int(
                    np.ceil(number_of_samples_per_trial / max_bitstrings_in_memory)
                )

            pbar_batches = tqdm(
                list(range(number_of_batches)),
                disable=(not show_progress_bar) or number_of_trials > 1,
                position=0,
                colour="cyan",
            )
            for batch_index in pbar_batches:
                if batch_index == number_of_batches - 1:
                    batch_size = number_of_samples_per_trial - batch_size * batch_index
                if self._backend == "cupy":
                    random_bitstrings_i = rng.binomial(
                        n=1, p=1 / 2, size=(batch_size, self._number_of_qubits)
                    )
                elif self._backend == "numpy":
                    random_bitstrings_i = rng.integers(
                        low=0, high=2, size=(batch_size, self._number_of_qubits)
                    )
                else:
                    raise ValueError(f"Backend {self._backend} not supported")

                energies_i = self._cost_hamiltonian.evaluate_energy(
                    bitstrings_array=random_bitstrings_i,
                    backend_computation=self._backend,
                )
                best_energy_index = np.argmin(energies_i)
                best_energy_i = float(energies_i[best_energy_index])
                best_state_i = random_bitstrings_i[best_energy_index]

                if best_energy_i < best_energy_so_far:
                    best_energy_so_far = best_energy_i
                    best_state_so_far = best_state_i

                if return_all_results:
                    all_bitstrings.append(random_bitstrings_i)
                    all_energies.append(energies_i)

                ar_best_energy_i = (highest_energy - best_energy_i) / delta
                ar_best_energy_so_far = (highest_energy - best_energy_so_far) / delta

                df_here = pd.DataFrame(
                    data={
                        SNV.Seed.id_long: [seed],
                        SNV.TrialIndex.id_long: [trial_index],
                        SNV.ApproximationRatio.id_long: [ar_best_energy_i],
                        SNV.ApproximationRatioBest.id_long: [ar_best_energy_so_far],
                        SNV.Energy.id_long: [best_energy_i],
                        SNV.EnergyBest.id_long: [best_energy_so_far],
                        SNV.Runtime.id_long: [time.perf_counter() - t0_total],
                        SNV.Bitstring.id_long: [
                            tuple([int(x) for x in best_state_so_far])
                        ],
                    }
                )
                results_dfs.append(df_here)

            pbar_batches.close()

        pbar_trials.close()
        results_df = pd.concat(results_dfs, axis=0)

        if return_all_results:
            all_bitstrings = bck.concatenate(all_bitstrings, axis=0)
            all_energies = bck.concatenate(all_energies, axis=0)
            return (
                (best_state_so_far, best_energy_so_far),
                results_df,
                (all_bitstrings, all_energies),
            )

        return best_state_so_far, best_energy_so_far, results_df
