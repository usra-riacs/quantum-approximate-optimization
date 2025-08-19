# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

from typing import List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em

from quapopt.optimization.parameter_setting.non_adaptive_optimization import NonAdaptiveOptimizer
from quapopt.optimization.parameter_setting import ParametersBoundType


class SimpleRandomOptimizer(NonAdaptiveOptimizer):
    def __init__(self,
                 parameter_bounds: List[Tuple[ParametersBoundType, Tuple[Any, ...]]],
                 argument_names: List[str] = None,
                 ):

        super().__init__(search_space=None,
                         argument_names=argument_names,
                         parameter_bounds=parameter_bounds)

    def run_optimization(self,
                         objective_function: callable,
                         seed=None,
                         number_of_trials: int = None,
                         verbosity: int = 0,
                         show_progress_bar: bool = False,
                         ):

        numpy_rng = np.random.default_rng(seed=seed)

        local_search_spaces = []
        for i in range(len(self.parameter_bounds)):
            bound_type, bound_specs = self.parameter_bounds[i]

            if bound_type == ParametersBoundType.RANGE:
                min_value, max_value = bound_specs
                random_samples = numpy_rng.uniform(min_value, max_value, number_of_trials)

            elif bound_type == ParametersBoundType.SET:
                if len(bound_specs) == 2:
                    if set(bound_specs) == {0, 1}:
                        random_samples = numpy_rng.integers(0, 2, number_of_trials)
                    else:
                        random_samples = numpy_rng.choice(bound_specs, number_of_trials)
                else:
                    random_samples = numpy_rng.choice(bound_specs, number_of_trials)

            elif bound_type == ParametersBoundType.CONSTANT:
                if isinstance(bound_specs, list) or isinstance(bound_specs, tuple):
                    assert len(bound_specs) == 1, "Fixed parameter should have only one value."
                    bound_specs = bound_specs[0]
                random_samples = [bound_specs] * number_of_trials
            else:
                raise ValueError('Unknown bound type.')
            local_search_spaces.append(random_samples)

        # search_space = list(itertools.product(*local_search_spaces))
        search_space = np.array(local_search_spaces).T

        return super()._run_optimization(objective_function=objective_function,
                                         number_of_function_calls=number_of_trials,
                                         verbosity=verbosity,
                                         show_progress_bar=show_progress_bar,
                                         search_space=search_space)


from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from tqdm import tqdm


class SimpleRandomBitstringSampler:
    def __init__(self,
                 hamiltonian: ClassicalHamiltonian,
                 max_memory: int = int(6 * 10 ** 9),
                 ):

        self._hamiltonian = hamiltonian
        self._number_of_qubits = self._hamiltonian.number_of_qubits

        if max(self._hamiltonian.localities) in [1, 2]:
            self._hamiltonian_repr_eval = self._hamiltonian.get_adjacency_matrix(sparse=False,
                                                                                 matrix_type='sym')
        else:
            self._hamiltonian_repr_eval = self._hamiltonian.hamiltonian_list_representation

        self._max_memory = max_memory

    def run_random_sampling(self,
                            number_of_samples: int,
                            seed: Optional[int] = None,
                            show_progress_bar: bool = False,
                            verbosity=0,
                            return_all_samples: bool = False,
                            ):

        numpy_rng = np.random.default_rng(seed=seed)

        total_number_of_bytes_bitstrings = number_of_samples * self._number_of_qubits

        bitstrings_array_memory = total_number_of_bytes_bitstrings / 10 ** 9
        flipping_array_memory = 4 * bitstrings_array_memory
        needed_memory = flipping_array_memory + bitstrings_array_memory + 1

        if needed_memory > self._max_memory / 10 ** 9:
            # This is the number we can afford to store in memory
            # I can allocate at most allowed_memory_per_batch memory at a time
            # I want to divide all trials into batches of size batch_size_bitstrings
            # (memory usage is batch_size_bitstrings*number_of_qubits_test*5 bytes)
            # this will be number of samples in a single batch.
            batch_size_bitstrings = int(max([1, self._max_memory]) // self._number_of_qubits / 5)
            # HOW MANY BATCHES I WILL NEED? TOTAL NUMBER OF NEEDED BYTES DIVIDED BY THE MEMORY OF A SINGLE BATCH
            number_of_batches = int(
                np.ceil(total_number_of_bytes_bitstrings / batch_size_bitstrings / self._number_of_qubits))

            batch_sizes = [batch_size_bitstrings] * number_of_batches
            total_size = batch_size_bitstrings * number_of_batches

            if total_size > number_of_samples:
                overkill = total_size - number_of_samples
                batch_sizes[-1] -= overkill

            if total_size < number_of_samples:
                batch_sizes.append(number_of_samples - total_size)
                number_of_batches += 1

        else:
            number_of_batches = 1
            batch_size_bitstrings = number_of_samples

            batch_sizes = [batch_size_bitstrings]

        def _evaluate_energy_function(bitstrings_array):
            return em.calculate_energies_from_bitstrings(measurement_values=bitstrings_array,
                                                         observable=self._hamiltonian.hamiltonian)

        if verbosity > 0:
            print(f"Number of batches: {number_of_batches}")
            print(f"Batch sizes: {batch_sizes}")

        if return_all_samples and number_of_batches > 1:
            raise ValueError("Cannot return all samples due to memory constraints.")

        best_energy = np.inf
        best_bitstring = None

        lowest_energy = self._hamiltonian.lowest_energy
        highest_energy = self._hamiltonian.highest_energy

        delta = None
        if lowest_energy is not None and highest_energy is not None:
            delta = highest_energy - lowest_energy

        for counter in tqdm(list(range(number_of_batches)), disable=not show_progress_bar):
            batch_size_i = batch_sizes[counter]
            random_samples_i = numpy_rng.integers(0, 2, (batch_size_i, self._number_of_qubits))

            random_energies_i = _evaluate_energy_function(bitstrings_array=random_samples_i)

            best_energy_index_i = np.argmin(random_energies_i)
            best_energy_i = random_energies_i[best_energy_index_i]
            best_bitstring_i = random_samples_i[best_energy_index_i]

            if best_energy_i < best_energy:
                if verbosity > 0:
                    if delta is None:
                        print(f"Best energy: {best_energy} --> {best_energy_i}")
                    else:
                        best_ar = (highest_energy - best_energy) / delta
                        ar_i = (highest_energy - best_energy_i) / delta
                        print(f"Best AR: {best_ar} --> {ar_i}")

                best_energy = best_energy_i
                best_bitstring = best_bitstring_i

        if return_all_samples:
            # number_of_batches = 1 so I can return all samples
            df_samples = pd.DataFrame(data={'energy': random_energies_i,
                                            'bitstring': random_samples_i.to_list()}, )
            return (best_bitstring, best_energy), df_samples

        return (best_bitstring, best_energy), None
