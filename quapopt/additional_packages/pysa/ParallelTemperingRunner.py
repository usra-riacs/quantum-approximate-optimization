# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

"""
This is refactored version of PySA repo (https://github.com/nasa/pysa/) modified by FBM
"""







import time
from typing import List, Optional, Union

import numba
import numpy as np
import pandas as pd

from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian


# warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
#TODO(FBM): working on lists instead of ndarrays might be ~2x faster here; CHECK
@numba.njit(fastmath=True, nogil=True, parallel=False)
def _perform_sweep(couplings: np.ndarray,
                   beta: float,
                   bitstring: np.ndarray,
                   numpy_rng: np.random.Generator,
                   bias_probability: float,
                   qubits_ordering: Optional[np.ndarray] = None,
                   local_fields: Optional[np.ndarray] = None,
                   # bias_probability: Optional[float] = None
                   ):


    log_r = np.log(numpy_rng.uniform(low=0, high=1, size=len(bitstring)))

    if qubits_ordering is None:
        qubits_ordering = np.array(list(range(len(bitstring))))
        numpy_rng.shuffle(qubits_ordering)
        # qubits_ordering = numpy_rng.permutation(list(range(len(bitstring))))



    dE_total = 0.
    for qi in qubits_ordering:
        bit_qi = bitstring[qi]
        second_part = np.dot(couplings[qi], bitstring)

        if local_fields is not None:
            second_part += local_fields[qi]

        dE_qi = 2 * bit_qi * second_part

        if dE_qi >= 0:
            bitstring[qi] *= -1
            dE_total += -dE_qi

        elif log_r[qi] < beta * dE_qi:
            bitstring[qi] *= -1
            dE_total += -dE_qi


    if bias_probability!=0.0:
        biases = numpy_rng.binomial(n=1,
                                    p=bias_probability,
                                    size=len(bitstring))
        for qi in range(len(bitstring)):
            bit_qi = bitstring[qi]
            if biases[qi] == 1 and bit_qi == 1:
                second_part = np.dot(couplings[qi], bitstring)

                if local_fields is not None:
                    second_part += local_fields[qi]
                dE_qi = 2 * bit_qi * second_part
                bitstring[qi] *= -1
                dE_total += -dE_qi


    return dE_total


@numba.njit(cache=True, fastmath=True, nogil=True, parallel=False)
def _perform_parallel_tempering_move(all_bitstrings: np.ndarray,
                                     all_energies: np.ndarray,
                                     betas: np.ndarray,
                                     log_probabilities_PT: np.ndarray, ):
    number_of_replicas = len(betas)
    # Apply PT for each pair of replicas
    for k in range(number_of_replicas - 1):
        # Get first index
        k1 = number_of_replicas - k - 1

        # Get second index
        k2 = number_of_replicas - k - 2

        # Compute delta energy
        de_k = (all_energies[k1] - all_energies[k2]) * (betas[k1] - betas[k2])

        log_r_PT = log_probabilities_PT[k1]

        # Accept/reject following Metropolis
        if de_k >= 0:
            # betas[k1], betas[k2] = betas[k2], betas[k1]
            # instead of betas, we swap the energies and bitstrings
            temp = all_bitstrings[k1, :].copy()
            all_bitstrings[k1, :] = all_bitstrings[k2, :]
            all_bitstrings[k2, :] = temp
            all_energies[k1], all_energies[k2] = all_energies[k2], all_energies[k1]


        elif log_r_PT < de_k:
            # betas[k1], betas[k2] = betas[k2], betas[k1]
            # instead of betas, we swap the energies and bitstrings
            temp = all_bitstrings[k1, :].copy()
            all_bitstrings[k1, :] = all_bitstrings[k2, :]
            all_bitstrings[k2, :] = temp
            all_energies[k1], all_energies[k2] = all_energies[k2], all_energies[k1]


@numba.njit(fastmath=True, nogil=True, parallel=True)
def _run_parallel_tempering(number_of_sweeps: int,
                            betas: np.ndarray,
                            all_bitstrings: np.ndarray,
                            all_energies: np.ndarray,
                            couplings: np.ndarray,
                            bias_probabilities: List[float],
                            numpy_rng: np.random.Generator,
                            local_fields: Optional[np.ndarray] = None,
                            qubits_orderings: Optional[np.ndarray] = None,

                            ):


    number_of_replicas = len(betas)


    log_probabilities_PT_all = numpy_rng.uniform(low=0,
                                                 high=1,
                                                 size=(number_of_sweeps, number_of_replicas - 1))

    # if qubits_orderings is None:
    #     qubits_orderings = np.full(shape=(number_of_replicas,),fill_value=None)

    for sweep_index in range(number_of_sweeps):
        log_probabilities_PT_s = log_probabilities_PT_all[sweep_index]

        for replica_index in numba.prange(number_of_replicas):
            beta_i = betas[replica_index]
            bitstring_i = all_bitstrings[replica_index]
            bias_probability_i = bias_probabilities[replica_index]

            if qubits_orderings is None:
                # qubits_ordering_i = numpy_rng.permutation(list(range(all_bitstrings.shape[1])))
                qubits_ordering_i = None
            else:
                qubits_ordering_i = qubits_orderings[replica_index]


            dE_i = _perform_sweep(couplings=couplings,
                                  beta=beta_i,
                                  bitstring=bitstring_i,
                                  local_fields=local_fields,
                                  qubits_ordering=qubits_ordering_i,
                                  bias_probability=bias_probability_i,
                                  numpy_rng=numpy_rng
                                  )
            all_energies[replica_index] += dE_i

        _perform_parallel_tempering_move(all_bitstrings=all_bitstrings,
                                         all_energies=all_energies,
                                         betas=betas,
                                         log_probabilities_PT=log_probabilities_PT_s
                                         )


@numba.njit(fastmath=True, nogil=True, parallel=False)
def _calculate_energy(couplings: np.ndarray,
                      local_fields: np.ndarray,
                      bitstring: np.ndarray,
                      # precision:type=np.float32
                      ):
    second_vector = np.dot(couplings, bitstring) / 2
    if local_fields is not None:
        second_vector += local_fields

    return np.dot(bitstring, second_vector)


class ParallelTemperingRunner:
    def __init__(self,
                 hamiltonian: ClassicalHamiltonian,
                 precision: type = np.float32,
                 rng_seed=None,
                 ):

        if set(hamiltonian.localities) not in [{1}, {1, 2}, {2}]:
            raise ValueError('Parallel tempering is only supported for 1- and 2-local Hamiltonians')

        self._precision = precision
        self._hamiltonian = hamiltonian
        self._number_of_qubits = hamiltonian.number_of_qubits

        couplings, local_fields = hamiltonian.get_couplings_and_local_fields(matrix_type='SYM',
                                                                             backend='numpy',
                                                                             precision=precision)
        self._couplings: np.ndarray = couplings
        self._local_fields: Optional[np.ndarray] = local_fields

        self._rng = np.random.default_rng(rng_seed)

    @property
    def number_of_qubits(self):
        return self._number_of_qubits

    def run_parallel_tempering(self,
                               number_of_sweeps: int,
                               temperatures: List[float],
                               states_initial: Optional[np.ndarray] = None,
                               qubits_orderings: Optional[Union[List[List[int]], List[int]]] = None,
                               bias_probabilities: Optional[Union[List[float], float]] = None,
                               number_of_threads: int = 1
                               ):

        t0 = time.perf_counter()

        number_of_replicas = len(temperatures)

        if states_initial is not None:
            assert states_initial.shape[
                       1] == self._number_of_qubits, 'Initial states must have the same number of qubits as the Hamiltonian'
            if states_initial.shape[0] == 1:
                states_initial = np.repeat(states_initial, number_of_replicas, axis=0)
            elif states_initial.shape[0] != number_of_replicas:
                raise ValueError('The number of initial states must be equal to the number of replicas')

            # check if 0 is in the states1
            if np.any(states_initial == 0):
                states_initial = 1 - 2 * states_initial

        else:
            states_initial = 1 - 2 * self._rng.integers(low=0,
                                                        high=1,
                                                        size=(number_of_replicas, self._number_of_qubits))

        if bias_probabilities is not None:
            if isinstance(bias_probabilities, float) or isinstance(bias_probabilities, int):
                bias_probabilities = [bias_probabilities] * number_of_replicas

            assert len(bias_probabilities) == number_of_replicas, \
                'The number of bias probabilities must be equal to the number of replicas'
        else:
            bias_probabilities = [0.0] * number_of_replicas

        if qubits_orderings is not None:
            if isinstance(qubits_orderings[0], int):
                qubits_orderings = [qubits_orderings] * number_of_replicas
            qubits_orderings = np.array(qubits_orderings)
        else:
            qubits_orderings = None

        all_bitstrings = np.array(states_initial, dtype=self._precision)
        test_energies = np.array([_calculate_energy(couplings=self._couplings,
                                                    local_fields=self._local_fields,
                                                    bitstring=all_bitstrings[i]
                                                    ) for i in range(number_of_replicas)])
        all_energies = em.calculate_energies_from_bitstrings_2_local(couplings_array=self._couplings,
                                                                     bitstrings_array=all_bitstrings,
                                                                     local_fields=self._local_fields,
                                                                     pm_input=True)

        temperatures = np.array(sorted(temperatures, reverse=True))
        self._rng.shuffle(temperatures)

        betas = np.array([1 / T for T in temperatures], dtype=self._precision)

        numba.set_num_threads(number_of_threads)



        _run_parallel_tempering(number_of_sweeps=number_of_sweeps,
                                betas=betas,
                                all_bitstrings=all_bitstrings,
                                all_energies=all_energies,
                                couplings=self._couplings,
                                local_fields=self._local_fields,
                                qubits_orderings=qubits_orderings,
                                bias_probabilities=bias_probabilities,
                                numpy_rng=self._rng,
                                )
        # numba.set_num_threads(1)
        # The actual run
        # print("ACTUAL RUNTIME:", t1 - t0)

        # TODO(FBM): add proper results logging

        # df_sweep = pd.DataFrame(data={SNV.TrialIndex.id_long:[sweep_index]*number_of_replicas,
        #                               SNV.ReplicaIndex.id_long:list(range(number_of_replicas)),
        #                               SNV.InverseTemperature.id_long: betas.tolist(),
        #                               SNV.Energy.id_long:energies.tolist(),
        #                               SNV.Bitstring.id_long:all_states.tolist(),
        #                               SNV.Runtime.id_long:runtimes.tolist(),
        #                               })

        # all_dataframes.append(df_sweep)

        t_end = time.perf_counter()
        # transform energies and states to numpy
        energies = np.array(all_energies)
        all_bitstrings = np.array(all_bitstrings, dtype=np.int8)
        # Convert back to 0s and 1s
        all_bitstrings = (1 - all_bitstrings) / 2

        best_energy_index = np.argmin(energies)
        best_energy = energies[best_energy_index]
        best_state = all_bitstrings[best_energy_index]

        # print([-1 in row for row in all_bitstrings])

        best_energy_test = em.calculate_energies_from_bitstrings_2_local(couplings_array=self._couplings,
                                                                         bitstrings_array=np.concatenate(
                                                                             [all_bitstrings,
                                                                              1 - all_bitstrings]),
                                                                         local_fields=self._local_fields)
        if np.all(abs(best_energy_test - best_energy) >= 10 ** (-1)):
            print(best_energy, best_energy_test)
            raise ValueError("ENERGY MISMATCH")

        # df_res = pd.concat(all_dataframes)
        # df_res[SNV.Bitstring.id_long] = df_res[SNV.Bitstring.id_long].apply(lambda x: tuple((1-np.array(x))/2))

        t1 = time.perf_counter()

        runtime = t1 - t0


        df_res = pd.DataFrame(data={'solution': [tuple(best_state)],
                                     'energy': [best_energy],
                                     'runtime': [runtime]})

        return (best_state, best_energy), df_res


if __name__ == '__main__':
    from quapopt.hamiltonians.generators import build_hamiltonian_generator
    from quapopt.data_analysis.data_handling import (COEFFICIENTS_TYPE,
                                                     COEFFICIENTS_DISTRIBUTION,
                                                     CoefficientsDistributionSpecifier,
                                                     HAMILTONIAN_MODELS)
    from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf

    number_of_qubits = 2 ** 4
    seed_cost_hamiltonian = 0

    coefficients_type = COEFFICIENTS_TYPE.DISCRETE
    coefficients_distribution = COEFFICIENTS_DISTRIBUTION.Uniform
    coefficients_distribution_properties = {'low': -1, 'high': 1, 'step': 1}
    coefficients_distribution_specifier = CoefficientsDistributionSpecifier(CoefficientsType=coefficients_type,
                                                                            CoefficientsDistributionName=coefficients_distribution,
                                                                            CoefficientsDistributionProperties=coefficients_distribution_properties)

    # We generate a Hamiltonian instance. In this case it's a random Sherrington-Kirkpatrick Hamiltonian
    hamiltonian_model = HAMILTONIAN_MODELS.SherringtonKirkpatrick
    localities = (1, 2)

    generator_cost_hamiltonian = build_hamiltonian_generator(hamiltonian_model=hamiltonian_model,
                                                             localities=localities,
                                                             coefficients_distribution_specifier=coefficients_distribution_specifier)

    cost_hamiltonian = generator_cost_hamiltonian.generate_instance(number_of_qubits=number_of_qubits,
                                                                    seed=seed_cost_hamiltonian,
                                                                    read_from_drive_if_present=True)

    print("Class description (cost):", cost_hamiltonian.hamiltonian_class_description)
    print("Instance description (cost):", cost_hamiltonian.hamiltonian_instance_description)

    if cost_hamiltonian.lowest_energy is None:
        print("SOLVING THE HAMILTONIAN CLASSICALLY")
        # if we wish, we can solve the Hamiltonian classically
        cost_hamiltonian.solve_hamiltonian(both_directions=True)

    ground_state_energy = cost_hamiltonian.ground_state_energy
    highest_energy = cost_hamiltonian.highest_energy

    print("GS ENERGY:", ground_state_energy)
    print("HIGHEST ENERGY:", highest_energy)

    ##############
    seed_test = 0
    pt_runner = ParallelTemperingRunner(hamiltonian=cost_hamiltonian,
                                        precision=np.float32,
                                        rng_seed=seed_test, )

    number_of_sweeps = 1000
    temperatures = [0.1, 0.5, 1, 2, 3, 3.5, 4]

    t0 = time.perf_counter()
    (best_state, best_energy), opt_res = pt_runner.run_parallel_tempering(number_of_sweeps=number_of_sweeps,
                                                                          temperatures=temperatures,
                                                                          show_progress_bar=True,
                                                                          number_of_threads=20,
                                                                          bias_probabilities=0.99
                                                                          )
    t1 = time.perf_counter()
    anf.cool_print("PT ENERGY (HERE):", best_energy)

    from ancillary_functions_usra.hamiltonians import ground_state_approximations as gsa

    adj_matrix = cost_hamiltonian.get_adjacency_matrix(backend='numpy')

    t2 = time.perf_counter()
    (best_state_pysa, best_energy_pysa), opt_res_pysa = gsa.solve_ising_hamiltonian_with_pysa(
        weights_matrix=adj_matrix,
        number_of_qubits=number_of_qubits,
        n_sweeps=number_of_sweeps,
        temps=temperatures,
        n_replicas=len(temperatures),
        n_reads=1,
        solver_seed=seed_test,
        initialize_strategy='random',
        update_strategy='random',
        number_of_threads=len(temperatures))
    t3 = time.perf_counter()
    anf.cool_print("PT ENERGY (Original PySA):", best_energy_pysa)
    # print(opt_res['Energy'])

    print("TIME HERE:", t1 - t0)
    print("TIME PYSA:", t3 - t2)

    print(best_state)
