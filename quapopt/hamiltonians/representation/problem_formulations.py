# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


"""
Code present here is used to map between different representations of 2-local Hamiltonians.
The functions present here are used to map between the following representations:
- Ising
- QUBO
- MAXCUT


The convention used across this repository is the following:

ADJACENCY MATRICES:
The adjacency matrix is a real symmetric matrix.
Off-diagonal elements are the couplings, and diagonal elements are the local fields.

SOLUTIONS:
All solutions are stored as 0s and 1s.

In the case of Ising, it is understood that:
0 -> 1 and 1 -> -1, as per the convention that |0> is a ground state of -Z Hamiltonian (see below)


COST FUNCTIONS:
ISING:
\sum_{i<=j} J_{ij} (1-2*x_i)*(1-2*xj) + \sum_i h_i (1-2xi)

QUBO:
\sum_{i<=j} x_i*x_j*Q_{ij}

MAXCUT:
\sum_{i<j} J_{ij} (x_i+x_j-2*x_i*x_j)

Note that in the above, only the upper triangular (including diagonal) is used,
but the actual calculations are sometimes done using symmetric matrices, hence the convention.

OBJECTIVE:
MAXIMIZATION: QUBO, MAXCUT
MINIMIZATION: ISING
"""

from typing import List, Tuple, Union

import numpy as np

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

from enum import Enum


class ProblemFormulationType(Enum):
    ISING = "Ising"
    QUBO = "QUBO"
    MAXCUT = "MaxCut"


def _calculate_maxcut_objective_direct(
    adjacency_matrix: Union[np.ndarray, cp.ndarray],
    bitstrings_array: Union[np.ndarray, cp.ndarray],
) -> List[float]:
    """
    This function calculates the MAXCUT objective directly from the adjacency matrix and the bitstrings.

    :param adjacency_matrix: real symmetric
    :param bitstrings_array: 0s and 1s solutions
    :return:
    """

    objectives_list = []
    for bitstring in bitstrings_array:
        obj = 0
        for u in range(adjacency_matrix.shape[0]):
            for v in range(u + 1, adjacency_matrix.shape[1]):
                coeff = adjacency_matrix[u, v]
                obj += coeff * (
                    bitstring[u] + bitstring[v] - 2 * bitstring[u] * bitstring[v]
                )

        objectives_list.append(obj)
    return objectives_list


def _calculate_qubo_objective_direct(
    adjacency_matrix: Union[np.ndarray, cp.ndarray],
    bitstrings_array: Union[np.ndarray, cp.ndarray],
) -> List[float]:
    """
    This function calculates the QUBO objective directly from the adjacency matrix and the bitstrings.
    :param adjacency_matrix: real symmetric
    :param bitstrings_array: 0s and 1s solutions
    :return:
    """
    objectives_list = []
    for bitstring in bitstrings_array:
        obj = 0
        for u in range(adjacency_matrix.shape[0]):
            for v in range(u, adjacency_matrix.shape[1]):
                coeff = adjacency_matrix[u, v]
                obj += coeff * bitstring[u] * bitstring[v]
        objectives_list.append(obj)
    return objectives_list


def _calculate_ising_objective_direct(
    adjacency_matrix: Union[np.ndarray, cp.ndarray],
    bitstrings_array: Union[np.ndarray, cp.ndarray],
) -> List[float]:
    """
    This function calculates the ISING objective directly from the adjacency matrix and the bitstrings.
    :param adjacency_matrix: real symmetric
    :param bitstrings_array: 0s and 1s solutions
    :return:
    """
    objectives_list = []
    for bitstring in bitstrings_array:
        obj = 0
        for u in range(adjacency_matrix.shape[0]):
            for v in range(u, adjacency_matrix.shape[1]):
                coeff = adjacency_matrix[u, v]
                if u == v:
                    obj += coeff * (1 - 2 * bitstring[u])
                else:
                    obj += coeff * (1 - 2 * bitstring[u]) * (1 - 2 * bitstring[v])
        objectives_list.append(obj)
    return objectives_list


def _map_maxcut_adjacency_to_ising(
    maxcut_adjacency: Union[np.ndarray, cp.ndarray],
) -> Union[np.ndarray, cp.ndarray]:
    """
    This function maps the MAXCUT adjacency matrix to the ISING adjacency matrix.
    # ASSUMING MAXCUT CORRESPONDS TO MAXIMIZATION
    # AND THAT ISING CORRESPONDS TO MINIMIZATION
    # But we also map si = (1-2*x_i) --> x_i = 1/2*(1-si)
    # Using this convention, the ISING HAMILTONIAN for minimization is the same as the MAXCUT HAMILTONIAN for maximization


    :param maxcut_adjacency: real N x N symmetric
    :return: ising adjacency matrix: real N x N symmetric
    """

    return maxcut_adjacency.copy()


def _map_maxcut_adjacency_to_qubo(
    maxcut_adjacency: Union[np.ndarray, cp.ndarray],
) -> Union[np.ndarray, cp.ndarray]:
    """
    This function maps the MAXCUT adjacency matrix to the QUBO adjacency matrix.
    # ASSUMING MAXCUT CORRESPONDS TO MAXIMIZATION
    # AND THAT QUBO CORRESPONDS TO MAXIMIZATION
    :param maxcut_adjacency: real N x N symmetric
    :return: qubo adjacency matrix: real N x N symmetric
    """

    if isinstance(maxcut_adjacency, cp.ndarray):
        bck = cp
    elif isinstance(maxcut_adjacency, np.ndarray):
        bck = np
    else:
        raise ValueError("Input should be either a numpy or cupy array")

    qubo_adjacency = -2 * maxcut_adjacency
    bck.fill_diagonal(qubo_adjacency, bck.sum(maxcut_adjacency, axis=1))

    return qubo_adjacency


def _map_qubo_adjacency_to_maxcut(
    qubo_adjacency: Union[np.ndarray, cp.ndarray],
) -> Union[np.ndarray, cp.ndarray]:
    """
    This function maps the QUBO adjacency matrix to the MAXCUT adjacency matrix.
    # ASSUMING MAXCUT CORRESPONDS TO MAXIMIZATION
    # AND THAT QUBO CORRESPONDS TO MAXIMIZATION

    Since MAXCUT has no local fields, the mapping requires extending the system by single ancillary qubit.

    :param qubo_adjacency: real N x N symmetric
    :return: maxcut adjacency matrix: (N+1) x (N+1) real symmetric
    """

    if isinstance(qubo_adjacency, cp.ndarray):
        bck = cp
    elif isinstance(qubo_adjacency, np.ndarray):
        bck = np
    else:
        raise ValueError("Input should be either a numpy or cupy array")

    number_of_nodes_qubo = qubo_adjacency.shape[0]
    maxcut_adjacency = bck.pad(
        -qubo_adjacency, pad_width=((0, 1), (0, 1)), mode="constant"
    )
    bck.fill_diagonal(maxcut_adjacency, 0)
    couplings_sums = bck.sum(qubo_adjacency, axis=1) + bck.diag(qubo_adjacency)

    maxcut_adjacency[:number_of_nodes_qubo, number_of_nodes_qubo] = couplings_sums
    maxcut_adjacency[number_of_nodes_qubo, :number_of_nodes_qubo] = couplings_sums

    return maxcut_adjacency


def _map_qubo_adjacency_to_ising(
    qubo_adjacency: Union[np.ndarray, cp.ndarray],
) -> Union[np.ndarray, cp.ndarray]:
    """
    This function maps the QUBO adjacency matrix to the ISING adjacency matrix.
    # ASSUMING ISING CORRESPONDS TO MINIMIZATION
    # AND THAT QUBO CORRESPONDS TO MAXIMIZATION

    :param qubo_adjacency: real N x N symmetric
    :return: ising adjacency matrix: real N x N symmetric
    """

    if isinstance(qubo_adjacency, cp.ndarray):
        bck = cp
    elif isinstance(qubo_adjacency, np.ndarray):
        bck = np
    else:
        raise ValueError("Input should be either a numpy or cupy array")

    ising_adjacency = -1 * qubo_adjacency
    bck.fill_diagonal(
        ising_adjacency, bck.sum(qubo_adjacency, axis=1) + bck.diag(qubo_adjacency)
    )

    return ising_adjacency


def _map_ising_adjacency_to_qubo(
    ising_adjacency: Union[np.ndarray, cp.ndarray],
) -> Union[np.ndarray, cp.ndarray]:
    """
    This function maps the ISING adjacency matrix to the QUBO adjacency matrix.
    # ASSUMING ISING CORRESPONDS TO MINIMIZATION
    # AND THAT QUBO CORRESPONDS TO MAXIMIZATION
    :param ising_adjacency: real N x N symmetric
    :return: qubo adjacency matrix: real N x N symmetric
    """

    if isinstance(ising_adjacency, cp.ndarray):
        bck = cp
    elif isinstance(ising_adjacency, np.ndarray):
        bck = np
    else:
        raise ValueError("Input should be either a numpy or cupy array")

    qubo_adjacency = -2 * ising_adjacency
    # weighted_degrees = #+bck.diagonal(ising_adjacency)
    bck.fill_diagonal(qubo_adjacency, bck.sum(ising_adjacency, axis=1))

    return qubo_adjacency


def _map_ising_adjacency_to_maxcut(
    ising_adjacency: Union[np.ndarray, cp.ndarray],
) -> Union[np.ndarray, cp.ndarray]:
    """
    This function maps the ISING adjacency matrix to the MAXCUT adjacency matrix.
    # ASSUMING ISING CORRESPONDS TO MINIMIZATION
    # AND THAT MAXCUT CORRESPONDS TO MAXIMIZATION
    Since MAXCUT does not have local fields, the mapping requires extending the system by single ancillary qubit.

    :param ising_adjacency: real N x N symmetric
    :return: maxcut adjacency matrix: (N+1) x (N+1) real symmetric
    """

    if isinstance(ising_adjacency, cp.ndarray):
        bck = cp
    elif isinstance(ising_adjacency, np.ndarray):
        bck = np
    else:
        raise ValueError("Input should be either a numpy or cupy array")

    # TODO(FBM): why is this doubled?
    double_ising = 2 * ising_adjacency
    maxcut_adjacency = bck.pad(
        double_ising, pad_width=((0, 1), (0, 1)), mode="constant"
    )
    bck.fill_diagonal(maxcut_adjacency, 0)

    local_fields_ising = bck.diag(double_ising)

    number_of_nodes_ising = ising_adjacency.shape[0]
    maxcut_adjacency[:number_of_nodes_ising, number_of_nodes_ising] = local_fields_ising
    maxcut_adjacency[number_of_nodes_ising, :number_of_nodes_ising] = local_fields_ising

    return maxcut_adjacency


def map_maxcut_solution_to_qubo(
    bitstring: Union[List[int], Tuple[int, ...], np.ndarray], pm_input: bool = False
) -> Tuple[int, ...]:
    """
    This function maps the MAXCUT solution to the QUBO solution, assuming that the original QUBO was mapped to MAXCUT.

    :param bitstring: N-dimensional 0s and 1s vector
    :param pm_input: if True, we assume that the input is in [-1,+1]
    :return: output bitstring: (N-1)-dimensional 0s and 1s vector
    """

    bts_pm = bitstring
    if not pm_input:
        assert (
            not -1 in bitstring
        ), "bitstring should be in {0,1} not {-1,1} if pm_input is False"
        bts_pm = [1 - 2 * x for x in bitstring]
    else:
        assert (
            0 not in bitstring
        ), "bitstring should be in {-1,1} not {0,1} if pm_input is True"

    return tuple(
        [int(1 / 2 * (1 - bts_pm[i] * bts_pm[-1])) for i in range(len(bts_pm) - 1)]
    )


def map_maxcut_solution_to_ising(
    bitstring: Union[List[int], Tuple[int, ...], np.ndarray],
):
    """
    This function maps the MAXCUT solution to the ISING solution, assuming that the original ISING was mapped to MAXCUT.
    (so it reduces the dimension by one qubit).

    :param bitstring: N-dimensional 0s and 1s vector
    :return: output bitstring: (N-1)-dimensional 0s and 1s vector
    """
    # TODO(FBM): add version that does this for many bitstrings at once efficiently
    if bitstring[-1] == 0:
        return bitstring[0 : len(bitstring) - 1]
    else:
        if isinstance(bitstring, (np.ndarray, cp.ndarray)):
            return 1 - bitstring[0 : len(bitstring) - 1]
        return [1 - x for x in bitstring[0 : len(bitstring) - 1]]


def map_adjacency_between_formulations(
    input_adjacency: Union[np.ndarray, cp.ndarray],
    input_formulation: ProblemFormulationType,
    output_formulation: ProblemFormulationType,
):
    """
    This function maps the input adjacency matrix from one representation to another.
    :param input_adjacency:
    :param input_formulation:
    :param output_formulation:
    :return:
    """

    if input_formulation == output_formulation:
        return input_adjacency

    if input_formulation == ProblemFormulationType.ISING:
        if output_formulation == ProblemFormulationType.MAXCUT:
            return _map_ising_adjacency_to_maxcut(input_adjacency)
        elif output_formulation == ProblemFormulationType.QUBO:
            return _map_ising_adjacency_to_qubo(input_adjacency)
        else:
            raise ValueError(
                "Output representation should be either 'MAXCUT' or 'QUBO'"
            )

    elif input_formulation == ProblemFormulationType.MAXCUT:
        if output_formulation == ProblemFormulationType.ISING:
            return _map_maxcut_adjacency_to_ising(input_adjacency)
        elif output_formulation == ProblemFormulationType.QUBO:
            return _map_maxcut_adjacency_to_qubo(input_adjacency)
        else:
            raise ValueError("Output representation should be either 'ISING' or 'QUBO'")

    elif input_formulation == ProblemFormulationType.QUBO:
        if output_formulation == ProblemFormulationType.ISING:
            return _map_qubo_adjacency_to_ising(input_adjacency)
        elif output_formulation == ProblemFormulationType.MAXCUT:
            return _map_qubo_adjacency_to_maxcut(input_adjacency)
        else:
            raise ValueError(
                "Output representation should be either 'ISING' or 'MAXCUT'"
            )

    else:
        raise ValueError(
            "Input representation should be either 'ISING', 'MAXCUT' or 'QUBO'"
        )


if __name__ == "__main__":
    # TODO(FBM): should add proper tests
    import itertools

    import numpy as np

    from quapopt import ancillary_functions as anf

    noq_test = 9

    for seed_test in range(0, 10):
        rng_test = np.random.default_rng(seed=seed_test)

        all_bitstrings_n = list(itertools.product([0, 1], repeat=noq_test))
        all_bitstrings_np = list(itertools.product([0, 1], repeat=noq_test + 1))
        all_bitstrings_n = np.array(all_bitstrings_n)
        all_bitstrings_np = np.array(all_bitstrings_np)

        # define base matrices:
        test_ISING_matrix = rng_test.uniform(-5, 5, (noq_test, noq_test))
        test_ISING_matrix = 0.5 * (test_ISING_matrix + test_ISING_matrix.T)

        test_QUBO_matrix = rng_test.uniform(-5, 5, (noq_test, noq_test))
        test_QUBO_matrix = 0.5 * (test_QUBO_matrix + test_QUBO_matrix.T)

        test_MAXCUT_matrix = rng_test.uniform(-5, 5, (noq_test, noq_test))
        test_MAXCUT_matrix = 0.5 * (test_MAXCUT_matrix + test_MAXCUT_matrix.T)
        np.fill_diagonal(test_MAXCUT_matrix, 0)

        base_matrices = {
            ProblemFormulationType.QUBO: test_QUBO_matrix,
            ProblemFormulationType.ISING: test_ISING_matrix,
            ProblemFormulationType.MAXCUT: test_MAXCUT_matrix,
        }

        mapped_matrices = {}
        for input_matrix_name in ProblemFormulationType:
            for output_matrix_name in ProblemFormulationType:
                input_matrix = base_matrices[input_matrix_name]
                output_matrix = map_adjacency_between_formulations(
                    input_adjacency=input_matrix,
                    input_formulation=input_matrix_name,
                    output_formulation=output_matrix_name,
                )
                mapped_matrices[f"{input_matrix_name} --> {output_matrix_name}"] = (
                    output_matrix
                )

        for input_matrix_name in ProblemFormulationType:

            input_matrix = base_matrices[input_matrix_name]
            if input_matrix_name == ProblemFormulationType.ISING:
                # energies_bruteforce = em2.calculate_energies_from_bitstrings_2_local(bitstrings_array=all_bitstrings_n,
                #                                                          weights_matrix=input_matrix)
                energies_bruteforce = _calculate_ising_objective_direct(
                    adjacency_matrix=input_matrix, bitstrings_array=all_bitstrings_n
                )

            elif input_matrix_name == ProblemFormulationType.QUBO:
                energies_bruteforce = _calculate_qubo_objective_direct(
                    bitstrings_array=all_bitstrings_n,
                    adjacency_matrix=input_matrix,
                )
            elif input_matrix_name == ProblemFormulationType.MAXCUT:
                energies_bruteforce = _calculate_maxcut_objective_direct(
                    bitstrings_array=all_bitstrings_n,
                    adjacency_matrix=input_matrix,
                )
            else:
                raise ValueError(
                    "Input matrix name should be either 'ISING', 'QUBO' or 'MAXCUT'"
                )

            if input_matrix_name in [ProblemFormulationType.ISING]:
                best_energy_index = np.argmin(energies_bruteforce)
                best_bitstring_input = all_bitstrings_n[best_energy_index]
                best_energy_input = energies_bruteforce[best_energy_index]
            elif input_matrix_name in [
                ProblemFormulationType.QUBO,
                ProblemFormulationType.MAXCUT,
            ]:
                best_energy_index = np.argmax(energies_bruteforce)
                best_bitstring_input = all_bitstrings_n[best_energy_index]
                best_energy_input = energies_bruteforce[best_energy_index]
            else:
                raise ValueError(
                    "Input matrix name should be either 'ISING', 'QUBO' or 'MAXCUT"
                )

            best_found_solutions = {
                f"{input_matrix_name} (ORIGINAL) (BRUTEFORCE)": (
                    best_bitstring_input,
                    best_energy_input,
                )
            }
            for output_matrix_name in ProblemFormulationType:
                output_matrix = mapped_matrices[
                    f"{input_matrix_name} --> {output_matrix_name}"
                ]
                if output_matrix_name == ProblemFormulationType.ISING:
                    # energies_mapped = em2.calculate_energies_from_bitstrings_2_local(bitstrings_array=all_bitstrings_n,
                    #                                                                 weights_matrix=output_matrix)
                    energies_mapped = _calculate_ising_objective_direct(
                        adjacency_matrix=output_matrix,
                        bitstrings_array=all_bitstrings_n,
                    )

                elif output_matrix_name == ProblemFormulationType.QUBO:
                    energies_mapped = _calculate_qubo_objective_direct(
                        adjacency_matrix=output_matrix,
                        bitstrings_array=all_bitstrings_n,
                    )
                elif output_matrix_name == ProblemFormulationType.MAXCUT:
                    energies_mapped = _calculate_maxcut_objective_direct(
                        adjacency_matrix=output_matrix,
                        bitstrings_array=all_bitstrings_np,
                    )
                else:
                    raise ValueError(
                        "Output matrix name should be either 'ISING', 'QUBO' or 'MAXCUT'"
                    )

                if output_matrix_name in [ProblemFormulationType.ISING]:
                    best_energy_index = np.argmin(energies_mapped)
                    best_bitstring_output = all_bitstrings_n[best_energy_index]
                elif output_matrix_name in [ProblemFormulationType.QUBO]:
                    best_energy_index = np.argmax(energies_mapped)
                    best_bitstring_output = all_bitstrings_n[best_energy_index]
                elif output_matrix_name in [ProblemFormulationType.MAXCUT]:
                    best_energy_index = np.argmax(energies_mapped)
                    best_bitstring_output_raw = all_bitstrings_np[best_energy_index]
                    if input_matrix_name == ProblemFormulationType.ISING:
                        best_bitstring_output = map_maxcut_solution_to_ising(
                            bitstring=best_bitstring_output_raw
                        )
                    elif input_matrix_name == ProblemFormulationType.QUBO:
                        best_bitstring_output = map_maxcut_solution_to_qubo(
                            bitstring=best_bitstring_output_raw
                        )
                    else:
                        best_bitstring_output = best_bitstring_output_raw
                else:
                    raise ValueError(
                        "Output matrix name should be either 'ISING', 'QUBO' or 'MAXCUT"
                    )

                if input_matrix_name in [ProblemFormulationType.ISING]:
                    # best_energy_output = em2.calculate_energies_from_bitstrings_2_local(bitstrings_array=[best_bitstring_output],
                    #                                                         weights_matrix=input_matrix)[0]
                    best_energy_output = _calculate_ising_objective_direct(
                        adjacency_matrix=input_matrix,
                        bitstrings_array=[best_bitstring_output],
                    )[0]
                elif input_matrix_name in [ProblemFormulationType.QUBO]:
                    best_energy_output = _calculate_qubo_objective_direct(
                        adjacency_matrix=input_matrix,
                        bitstrings_array=[best_bitstring_output],
                    )[0]
                elif input_matrix_name in [ProblemFormulationType.MAXCUT]:
                    best_energy_output = _calculate_maxcut_objective_direct(
                        adjacency_matrix=input_matrix,
                        bitstrings_array=[best_bitstring_output],
                    )[0]
                else:
                    raise ValueError(
                        "Input matrix name should be either 'ISING', 'QUBO' or 'MAXCUT"
                    )

                best_found_solutions[
                    f"{input_matrix_name}->{output_matrix_name} (BRUTEFORCE)"
                ] = (best_bitstring_output, best_energy_output)

            all_energies = [x[1] for x in best_found_solutions.values()]

            correct = np.allclose(all_energies, all_energies[0])

            if correct:
                pass
            else:
                anf.cool_print(
                    "FAILED TEST:",
                    f"{input_matrix_name}-->ALL (BRUTEFORCE)-{seed_test}",
                    "red",
                )
                for key, value in best_found_solutions.items():
                    print(key)
                    print(value)
                    print()
                raise KeyboardInterrupt

    anf.cool_print("ALL TESTS PASSED", "", "green")
