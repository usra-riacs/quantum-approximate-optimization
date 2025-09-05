# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import time
from typing import Optional, List, Tuple, Any

import numpy as np


try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
    import numpy as cp


import pandas as pd
from tqdm.notebook import tqdm

from quapopt import ancillary_functions as anf

from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.meta_algorithms.NDAR import ConvergenceCriterionNames, ConvergenceCriterion
from quapopt.meta_algorithms.NDAR.NDARRunner import NDARRunner


# from quapopt.optimization.classical_solvers.evolutionary_solvers import solve_ising_hamiltonian_evolutionary

def _cupy_old_version_replace_put_along_axis(arr, indices, values, axis,
                                             expanded_indices=None):
    """
    Efficient CuPy implementation of put_along_axis.

    Args:
        arr (cp.ndarray): The array to modify.
        indices (cp.ndarray): Indices along the specified axis where to put the values.
        values (cp.ndarray): Values to insert into the array.
        axis (int): The axis along which to insert values.

    Returns:
        None. Modifies the input array in-place.
    """
    # Create an index grid for each dimension
    if expanded_indices is None:
        expanded_indices = cp.ogrid[tuple(map(slice, arr.shape))]

    # Replace the indices along the specified axis
    expanded_indices[axis] = indices

    # Perform in-place assignment
    arr[tuple(expanded_indices)] = values


class RandomLocalSearchRunner(NDARRunner):
    def __init__(self,
                 hamiltonian: ClassicalHamiltonian,
                 # sampler_seed: int,
                 convergence_criterion: ConvergenceCriterion = None,
                 backend='numpy',
                 max_memory: Optional[int] = None
                 ):

        self._hamiltonian = hamiltonian
        self._number_of_qubits = hamiltonian.number_of_qubits
        # self._sampler_seed = sampler_seed

        if convergence_criterion is None:
            convergence_criterion = ConvergenceCriterion(
                convergence_criterion_name=ConvergenceCriterionNames.MaxUnsuccessfulTrials,
                convergence_value=50)

        self._backend = backend

        if self._hamiltonian.default_backend != backend:
            print("CHANGING BACKEND:", f'{self._hamiltonian.default_backend} -> {backend}')
            self._hamiltonian.reinitialize_backend(backend=backend)

        if max_memory is None:
            if backend == 'cupy':
                max_memory = int(5 * 10 ** 7)
            elif backend == 'numpy':
                max_memory = int(1 * 10 ** 9)
            else:
                raise ValueError(f"Backend {backend} not supported")

        self._max_memory = max_memory

        super().__init__(input_hamiltonian_representations=[self._hamiltonian],
                         attractor_model=None,
                         convergence_criterion=convergence_criterion,
                         )

    def _sample_new_solutions(self,
                              hamiltonian_representations_cost: List[ClassicalHamiltonian],
                              new_seed: Optional[int] = None,
                              *args,
                              **kwargs) -> List[Tuple[float, Tuple[np.ndarray, int, Any]]]:
        # dummy function
        return None

    def run_NDAR(self,
                 number_of_samples_per_trial: int,
                 number_of_trials_max: int,
                 p_01: float,
                 verbosity=1,
                 solver_timeout=None,
                 show_progress_bar=False,
                 initial_bitstring=None,
                 temperature=0.0,
                 temperature_mult: float = 1.0,
                 store_full_data=False,
                 add_GA=False,
                 kwargs_GA: Optional[dict] = None,
                 sampler_seed: Optional[int] = None,
                 ):

        self.clean_optimization_history()

        if add_GA:
            if kwargs_GA is None:
                kwargs_GA = {'num_generations': 200,
                             'num_parents_mating': 10,
                             'sol_per_pop': 10,
                             'keep_parents': 0,
                             'mutation_type': "swap",
                             'mutation_probability': None,
                             # 'mutation_type':'adaptive',
                             # 'mutation_probability':[0.02,0.07],

                             'allow_duplicate_genes': True,
                             'save_best_solutions': False,
                             # 'suppress_warnings': True,
                             }

        # if verbosity >= 1:
        highest_energy = self._hamiltonian.highest_energy
        ground_state_energy = self._hamiltonian.ground_state_energy
        delta = highest_energy - ground_state_energy
        if verbosity >= 1:
            anf.cool_print("GROUND STATE ENERGY:", ground_state_energy, 'green')

        if self._backend == 'cupy':
            bck = cp
        elif self._backend == 'numpy':
            bck = np
        else:
            raise ValueError('Backend not recognized')

        total_number_of_bytes_bitstrings = number_of_samples_per_trial * number_of_trials_max * self._number_of_qubits

        rng_boltzman = bck.random.default_rng(seed=sampler_seed)
        rng_uniform = bck.random.default_rng(seed=sampler_seed)

        if verbosity >= 2:
            print("MEMORY EFFICIENT FLIPS")
        # This is the number we can afford to store in memory
        # I can allocate at most allowed_memory_per_batch memory at a time
        # I want to divide all trials into batches of size batch_size_bitstrings
        # (memory usage is batch_size_bitstrings*number_of_qubits_test*5 bytes)
        # this will be number of samples in a single batch.
        batch_size_bitstrings = int(max([1, self._max_memory]) // self._number_of_qubits / 5)
        # HOW MANY BATCHES I WILL NEED? TOTAL NUMBER OF NEEDED BYTES DIVIDED BY THE MEMORY OF A SINGLE BATCH
        number_of_batches = int(
            np.ceil(total_number_of_bytes_bitstrings / batch_size_bitstrings / self._number_of_qubits))

        if verbosity >= 1:
            print("RUNNING NDAR")

        dt_noise, dt_energies, dt_mask = 0, 0, 0

        zeros_n = bck.zeros((self._number_of_qubits,), dtype=int)

        if initial_bitstring is None:
            bitstring_mask = zeros_n.copy()
            best_bitstring_so_far = zeros_n.copy()
        else:
            initial_bitstring = bck.array(initial_bitstring, dtype=int)
            bitstring_mask = initial_bitstring.copy()
            best_bitstring_so_far = initial_bitstring.copy()

        best_energy_so_far = self._hamiltonian.evaluate_energy(bitstrings_array=[bitstring_mask])[0]

        best_energy_so_far = float(best_energy_so_far)

        # print(best_energy_so_far)

        t0_total = time.perf_counter()
        batch_index = 0
        samples_investigated_so_far = 0

        # temperature = 1

        t0_total = time.perf_counter()

        results_all = []
        unsuccesful_trials = 0

        # if show_progress_bar:
        pbar = tqdm(total=number_of_trials_max, position=0, colour='blue', disable=not show_progress_bar)

        ndar_iteration_index = 0

        all_dfs = []

        convergence_description = self._convergence_criterion.get_description_string()

        ind_i = -1
        while not self.check_convergence() and ndar_iteration_index < number_of_trials_max:
            # for counter in tqdm(list(range(number_of_trials_max)), colour='blue', disable = not show_progress_bar):
            # If we are using memory efficient flips, we need to generate the bitstrings in batches
            # check if need to generate new batch
            ind_i += 1
            # TODO(FBM): take care of edge cases
            if samples_investigated_so_far % batch_size_bitstrings < number_of_samples_per_trial:
                if verbosity >= 2:
                    print("TRIAL NUMBER:", ndar_iteration_index, ", GENERATING NEW BATCH")
                t0 = time.perf_counter()

                t1 = time.perf_counter()
                # print(batch_size_bitstrings)
                random_bitstrings_flipped = rng_uniform.binomial(n=1,
                                                                 p=1 / 2 * (1 - p_01),
                                                                 size=(batch_size_bitstrings,
                                                                       self._number_of_qubits))

                # random_bitstrings_flipped = self._noisy_sampler.add_noise_to_samples(
                #     ideal_samples=random_bitstrings)
                t2 = time.perf_counter()
                # random_bitstrings = None
                if verbosity >= 2:
                    print("TIME TO GENERATE RANDOM BITSTRINGS: ", t1 - t0)
                    print("TIME TO FLIP RANDOM BITSTRINGS: ", t2 - t1)
                ind_i = 0
                samples_investigated_so_far = 0

            t00 = time.perf_counter()
            t01 = time.perf_counter()
            # print("hejunia",type(random_bitstrings_flipped))
            random_bitstrings_flipped_i = random_bitstrings_flipped[ind_i:ind_i + number_of_samples_per_trial, :]
            if len(random_bitstrings_flipped_i) < number_of_samples_per_trial:
                additional_samples_i = number_of_samples_per_trial - len(random_bitstrings_flipped_i)
                random_additional_i = rng_uniform.binomial(n=1,
                                                           p=1 / 2 * (1 - p_01),
                                                           size=(additional_samples_i,
                                                                 self._number_of_qubits))

                random_bitstrings_flipped_i = bck.concatenate([random_bitstrings_flipped_i, random_additional_i],
                                                              axis=0)
            elif len(random_bitstrings_flipped_i) > number_of_samples_per_trial:
                random_bitstrings_flipped_i = random_bitstrings_flipped_i[:number_of_samples_per_trial, :]

            t02 = time.perf_counter()
            random_bitstrings_flipped_i_masked = random_bitstrings_flipped_i ^ bitstring_mask
            flipped_energies = self._hamiltonian.evaluate_energy(bitstrings_array=random_bitstrings_flipped_i_masked,
                                                                 backend_computation=self._backend)

            t03 = time.perf_counter()
            # print(type(flipped_energies),type(random_bitstrings_flipped_i_masked))
            best_energy_index_i = bck.argmin(flipped_energies)
            best_energy_i = float(flipped_energies[best_energy_index_i])
            best_bitstring_i = random_bitstrings_flipped_i[best_energy_index_i, :]

            self._print_energy_progress_nicely(current_energy=best_energy_i,
                                               best_energy_so_far=best_energy_so_far,
                                               verbosity=verbosity,
                                               step_information=f"STEP {ndar_iteration_index}")

            if add_GA:
                pairs_bts_energy = bck.hstack((random_bitstrings_flipped_i_masked,
                                               flipped_energies.reshape(-1, 1)))
                pairs_bts_energy = pairs_bts_energy[bck.argsort(pairs_bts_energy[:, -1])]
                best_solutions_n = pairs_bts_energy[:kwargs_GA['sol_per_pop'], 0:-1]

                (solution_ga, energy_ga), ga_instance_i = solve_ising_hamiltonian_evolutionary(
                                            hamiltonian=self._hamiltonian,
                                            initial_population=best_solutions_n,
                                            solver_seed=ndar_iteration_index,
                                            **kwargs_GA)
                energy_ga = float(energy_ga)

                # print("GA SOLUTION:", f"{(highest_energy - energy_ga) / delta:.3f}")

                if energy_ga < best_energy_i:
                    ar_prev = (highest_energy - best_energy_i) / delta
                    ar_new = (highest_energy - energy_ga) / delta

                    best_energy_i = energy_ga
                    best_bitstring_i = bck.array(solution_ga) ^ bitstring_mask

                    print("GA IMPROVED SOLUTION:", f"{ar_prev:.3f} -> {ar_new:.3f}")

                # raise KeyboardInterrupt

            applied_mask = bitstring_mask.copy()

            if best_energy_i < best_energy_so_far:
                unsuccesful_trials = 0
                best_energy_so_far = best_energy_i
                best_bitstring_so_far = best_bitstring_i
                if p_01 != 0.0:
                    bitstring_mask = bitstring_mask ^ best_bitstring_i

            else:
                unsuccesful_trials += 1
                if p_01 != 0.0:
                    if self._metropolis_check(energy_current=best_energy_i,
                                              energy_previous=best_energy_so_far,
                                              temperature=temperature,
                                              rng=rng_boltzman):
                        bitstring_mask = bitstring_mask ^ best_bitstring_i

            # TODO FBM: maybe make this handled outside this function

            dt_mask += t01 - t00
            dt_energies += t03 - t02
            samples_investigated_so_far += len(random_bitstrings_flipped_i)
            # print(samples_investigated_so_far,batch_size_bitstrings)

            ar_best_energy_i = (highest_energy - best_energy_i) / delta
            ar_best_energy_so_far = (highest_energy - best_energy_so_far) / delta
            if store_full_data:
                df_here = pd.DataFrame(data={
                    SNV.Seed.id_long: [sampler_seed],
                    SNV.IterationIndex.id_long: [ndar_iteration_index],
                    SNV.FlipProbability.id_long: [p_01],
                    SNV.Temperature.id_long: [temperature],
                    "TemperatureMultiplier": [temperature_mult],
                    SNV.ApproximationRatio.id_long: [ar_best_energy_i],
                    SNV.ApproximationRatioBest.id_long: [ar_best_energy_so_far],
                    SNV.Energy.id_long: [best_energy_i],
                    SNV.EnergyBest.id_long: [best_energy_so_far],
                    SNV.Runtime.id_long: [time.perf_counter() - t0_total],
                    SNV.ConvergenceCriterion.id_long: [convergence_description],
                    SNV.Bitstring.id_long: [tuple([int(x) for x in best_bitstring_so_far])],
                    SNV.Bitflip.id_long: [tuple([int(x) for x in applied_mask])],
                })
                results_all.append(df_here)

            self._optimization_history[ndar_iteration_index] = (best_energy_i, best_bitstring_i)
            self._ndar_iteration = ndar_iteration_index
            ndar_iteration_index += 1
            if show_progress_bar:
                pbar.update(1)

            if solver_timeout is not None:
                if time.perf_counter() - t0_total > solver_timeout:
                    # print("TIME LIMIT REACHED")
                    break

        if store_full_data:
            df_res = pd.concat(results_all, axis=0)
        else:
            df_res = None

        t1_total = time.perf_counter()

        if verbosity >= 2:
            print("Time to calculate energies of noisy bitstrings: ", dt_energies)
            print("Time to calculate mask: ", dt_mask)
            print("TOTAL TIME:", t1_total - t0_total)

        pbar.close()

        return (best_bitstring_so_far, best_energy_so_far), df_res

    def run_fixed_HW_NDAR(self,
                          number_of_samples_per_trial: int,
                          number_of_trials_max: int,
                          hamming_weight: int,
                          verbosity=1,
                          runtime_limit=None,
                          show_progress_bar=False,
                          initial_bitstring=None,
                          temperature=0.0,
                          temperature_mult: float = 1.0,
                          store_full_data=False,
                          sampler_seed: Optional[int] = None,
                          ):
        hamming_weight = int(hamming_weight)
        assert hamming_weight > 0, "Please provide a positive integer for hamming weight"

        self.clean_optimization_history()

        # if verbosity >= 1:
        highest_energy = self._hamiltonian.highest_energy
        ground_state_energy = self._hamiltonian.ground_state_energy
        delta = highest_energy - ground_state_energy
        if verbosity >= 1:
            anf.cool_print("GROUND STATE ENERGY:", ground_state_energy, 'green')

        if self._backend == 'cupy':
            bck = cp
        elif self._backend == 'numpy':
            bck = np
        else:
            raise ValueError('Backend not recognized')

        total_number_of_bytes_bitstrings = number_of_samples_per_trial * number_of_trials_max * self._number_of_qubits

        rng_boltzman = bck.random.default_rng(seed=sampler_seed)
        rng_uniform = bck.random.default_rng(seed=sampler_seed)

        # so cupy does not support choice yet
        rng_choice = np.random.default_rng(seed=sampler_seed)

        if verbosity >= 2:
            print("MEMORY EFFICIENT FLIPS")
        # This is the number we can afford to store in memory
        # I can allocate at most allowed_memory_per_batch memory at a time
        # I want to divide all trials into batches of size batch_size_bitstrings
        # (memory usage is batch_size_bitstrings*number_of_qubits_test*5 bytes)
        # this will be number of samples in a single batch.
        batch_size_bitstrings = int(max([1, self._max_memory]) // self._number_of_qubits / 5)
        # HOW MANY BATCHES I WILL NEED? TOTAL NUMBER OF NEEDED BYTES DIVIDED BY THE MEMORY OF A SINGLE BATCH
        number_of_batches = int(
            np.ceil(total_number_of_bytes_bitstrings / batch_size_bitstrings / self._number_of_qubits))

        if verbosity >= 1:
            print("RUNNING NDAR")

        dt_noise, dt_energies, dt_mask = 0, 0, 0

        zeros_n = bck.zeros((self._number_of_qubits,), dtype=int)
        best_bitstring_so_far = zeros_n.copy()
        bitstring_mask = zeros_n.copy()

        if initial_bitstring is not None:
            initial_bitstring = bck.array(initial_bitstring, dtype=int)
            best_bitstring_so_far = initial_bitstring.copy()
            bitstring_mask = bitstring_mask ^ initial_bitstring

        best_energy_so_far = self._hamiltonian.evaluate_energy(bitstrings_array=[bitstring_mask])[0]

        best_energy_so_far = float(best_energy_so_far)

        t0_total = time.perf_counter()
        batch_index = 0
        samples_investigated_so_far = 0

        # temperature = 1

        t0_total = time.perf_counter()

        results_all = []
        unsuccesful_trials = 0

        # if show_progress_bar:
        pbar = tqdm(total=number_of_trials_max, position=0, colour='blue', disable=not show_progress_bar)

        ndar_iteration_index = 0

        all_dfs = []

        convergence_description = self._convergence_criterion.get_description_string()

        all_qubits = list(range(self._number_of_qubits))

        if bck == cp:
            # TODO(FBM): I'm not sure if this speeds anything up, grid creation is not a bottleneck
            expanded_indices_batch = cp.ogrid[tuple(map(slice, (batch_size_bitstrings, self._number_of_qubits)))]

        ind_i = -1
        while not self.check_convergence() and ndar_iteration_index < number_of_trials_max:
            # for counter in tqdm(list(range(number_of_trials_max)), colour='blue', disable = not show_progress_bar):
            # If we are using memory efficient flips, we need to generate the bitstrings in batches
            # check if need to generate new batch
            ind_i += 1
            # TODO(FBM): take care of edge cases
            if samples_investigated_so_far % batch_size_bitstrings < number_of_samples_per_trial:
                if verbosity >= 2:
                    print("TRIAL NUMBER:", ndar_iteration_index, ", GENERATING NEW BATCH")
                t0 = time.perf_counter()

                t1 = time.perf_counter()

                random_bitstrings_flipped = bck.zeros((batch_size_bitstrings, self._number_of_qubits), dtype=int)
                random_indices_i = rng_choice.choice(a=all_qubits,
                                                     size=(batch_size_bitstrings, hamming_weight),
                                                     replace=True,
                                                     )

                if bck == np:
                    bck.put_along_axis(random_bitstrings_flipped, random_indices_i, 1, axis=1)
                elif bck == cp:
                    _cupy_old_version_replace_put_along_axis(random_bitstrings_flipped, random_indices_i, 1, axis=1,
                                                             expanded_indices=expanded_indices_batch)
                else:
                    raise ValueError('Backend not recognized')

                t2 = time.perf_counter()
                # random_bitstrings = None
                if verbosity >= 2:
                    print("TIME TO GENERATE RANDOM BITSTRINGS: ", t1 - t0)
                    print("TIME TO FLIP RANDOM BITSTRINGS: ", t2 - t1)
                ind_i = 0
                samples_investigated_so_far = 0

            t00 = time.perf_counter()
            t01 = time.perf_counter()
            # print("hejunia",type(random_bitstrings_flipped))
            random_bitstrings_flipped_i = random_bitstrings_flipped[ind_i:ind_i + number_of_samples_per_trial, :]
            if len(random_bitstrings_flipped_i) < number_of_samples_per_trial:
                additional_samples_i = number_of_samples_per_trial - len(random_bitstrings_flipped_i)
                random_additional_i = bck.zeros((additional_samples_i, self._number_of_qubits), dtype=int)
                random_indices_i = rng_choice.choice(a=all_qubits,
                                                     size=(additional_samples_i, hamming_weight),
                                                     replace=True,
                                                     )
                if bck == np:
                    bck.put_along_axis(random_additional_i, random_indices_i, 1, axis=1)
                elif bck == cp:
                    _cupy_old_version_replace_put_along_axis(random_additional_i, random_indices_i, 1, axis=1)
                else:
                    raise ValueError('Backend not recognized')

                random_bitstrings_flipped_i = bck.concatenate([random_bitstrings_flipped_i, random_additional_i],
                                                              axis=0)
            elif len(random_bitstrings_flipped_i) > number_of_samples_per_trial:
                random_bitstrings_flipped_i = random_bitstrings_flipped_i[:number_of_samples_per_trial, :]

            t02 = time.perf_counter()
            random_bitstrings_flipped_i_masked = random_bitstrings_flipped_i ^ bitstring_mask
            flipped_energies = self._hamiltonian.evaluate_energy(bitstrings_array=random_bitstrings_flipped_i_masked,
                                                                 backend_computation=self._backend)
            t03 = time.perf_counter()
            # print(type(flipped_energies),type(random_bitstrings_flipped_i_masked))
            best_energy_index_i = bck.argmin(flipped_energies)
            best_energy_i = float(flipped_energies[best_energy_index_i])
            best_bitstring_i = random_bitstrings_flipped_i[best_energy_index_i, :]

            applied_mask = bitstring_mask.copy()

            self._print_energy_progress_nicely(current_energy=best_energy_i,
                                               best_energy_so_far=best_energy_so_far,
                                               verbosity=verbosity,
                                               step_information=f"STEP {ndar_iteration_index}")

            if best_energy_i < best_energy_so_far:
                unsuccesful_trials = 0
                best_energy_so_far = best_energy_i
                best_bitstring_so_far = best_bitstring_i
                # if p_01 != 0.0:
                bitstring_mask = bitstring_mask ^ best_bitstring_i

            else:
                unsuccesful_trials += 1
                # if p_01 != 0.0:
                if self._metropolis_check(energy_current=best_energy_i,
                                          energy_previous=best_energy_so_far,
                                          temperature=temperature,
                                          rng=rng_boltzman):
                    bitstring_mask = bitstring_mask ^ best_bitstring_i

            # TODO FBM: maybe make this handled outside this function

            dt_mask += t01 - t00
            dt_energies += t03 - t02
            samples_investigated_so_far += len(random_bitstrings_flipped_i)
            # print(samples_investigated_so_far,batch_size_bitstrings)

            ar_best_energy_i = (highest_energy - best_energy_i) / delta
            ar_best_energy_so_far = (highest_energy - best_energy_so_far) / delta

            if store_full_data:
                df_here = pd.DataFrame(data={
                    SNV.Seed.id_long: [sampler_seed],
                    SNV.IterationIndex.id_long: [ndar_iteration_index],
                    # SNV.FlipProbability.id_long: [p_01],
                    "FlipHammingWeight": [hamming_weight],
                    SNV.Temperature.id_long: [temperature],
                    "TemperatureMultiplier": [temperature_mult],
                    SNV.ApproximationRatio.id_long: [ar_best_energy_i],
                    SNV.ApproximationRatioBest.id_long: [ar_best_energy_so_far],
                    SNV.Energy.id_long: [best_energy_i],
                    SNV.EnergyBest.id_long: [best_energy_so_far],
                    SNV.Runtime.id_long: [time.perf_counter() - t0_total],
                    SNV.ConvergenceCriterion.id_long: [convergence_description],
                    SNV.Bitstring.id_long: [tuple([int(x) for x in best_bitstring_so_far])],
                    SNV.Bitflip.id_long: [tuple([int(x) for x in applied_mask])],
                })
                results_all.append(df_here)

            self._optimization_history[ndar_iteration_index] = (best_energy_i, best_bitstring_i)
            self._ndar_iteration = ndar_iteration_index
            ndar_iteration_index += 1
            if show_progress_bar:
                pbar.update(1)

            if runtime_limit is not None:
                if time.perf_counter() - t0_total > runtime_limit:
                    print("TIME LIMIT REACHED")
                    break
        if store_full_data:
            df_res = pd.concat(results_all, axis=0)
        else:
            df_res = None
        t1_total = time.perf_counter()

        if verbosity >= 2:
            print("Time to calculate energies of noisy bitstrings: ", dt_energies)
            print("Time to calculate mask: ", dt_mask)
            print("TOTAL TIME:", t1_total - t0_total)

        pbar.close()

        return (best_bitstring_so_far, best_energy_so_far), df_res

# if __name__ == '__main__':
#     size = (10000, 1000)
#
#     #Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
    import numpy as cp

#     from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import ClassicalMeasurementNoiseSampler, \
#         MeasurementNoiseType
#
#     # import numpy as cp
#
#     seed = 0
#     p01_test = 0.99
#     rng_1 = cp.random.default_rng(seed=seed)
#     rng_2 = cp.random.default_rng(seed=seed)
#     rng_3 = np.random.default_rng(seed=seed)
#
#     noisy_sampler = ClassicalMeasurementNoiseSampler(noise_type=MeasurementNoiseType.TP_1q_identical,
#                                                      noise_description={'p_01': p01_test,
#                                                                         'p_10': None},
#                                                      backend_computation='cupy',
#                                                      rng=rng_1)
#
#     t0 = time.perf_counter()
#     bitstrings_1_not_flipped = rng_1.binomial(n=1,
#                                               p=0.5,
#                                               size=size)
#     # bitstrings_1 = add_identical_amplitude_damping_to_samples(ideal_samples_array=bitstrings_1_not_flipped.copy(),
#     #                                                          p_01=p01,
#     #                                                          rng=rng_1)
#     bitstrings_1 = noisy_sampler.add_noise_to_samples(ideal_samples=bitstrings_1_not_flipped.copy(), )
#
#     t1 = time.perf_counter()
#     bitstrings_2 = rng_2.binomial(n=1,
#                                   p=1 / 2 * (1 - p01_test),
#                                   size=size)
#     t2 = time.perf_counter()
#     print("TIME TO GENERATE BITSTRINGS 1:", t1 - t0)
#     print("TIME TO GENERATE BITSTRINGS 2:", t2 - t1)
#     print('1s:',
#           cp.sum(bitstrings_1_not_flipped),
#           cp.sum(bitstrings_1),
#           cp.sum(bitstrings_2))
#
#
#
#     #size = (10,4)
#     all_qubits = list(range(size[1]))
#
#     zeros = np.zeros(size)
#
#     t3 = time.perf_counter()
#     test = rng_3.choice(a=all_qubits,
#                          size=(size[0],2),
#                          replace=True,
#                                 )
#     #print(test.shape)
#     t4 = time.perf_counter()
#     #print(test)
#     #print(zeros)
#     np.put_along_axis(zeros, test, 1, axis=1)
#
#
#
#     t5 = time.perf_counter()
#
#     print("TIME TO GENERATE CHOICE 1:", t4 - t3)
#     print("TIME TO PUT ALONG AXIS:", t5 - t4)
#     #print(test)
#     #print(test)
#     #print(zeros)
#
#
#     #zeros[test] = 1
#     #print(zeros)
#
