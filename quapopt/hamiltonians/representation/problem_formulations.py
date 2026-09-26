# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


r"""
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

#Lazy monkey-patching of cupy
from quapopt import AVAILABLE_SIMULATORS
if 'cupy' in AVAILABLE_SIMULATORS:
    import cupy as cp
else:
    import numpy as cp


from enum import Enum


class ProblemFormulationType(Enum):
    ISING = "Ising"
    QUBO = "QUBO"
    MAXCUT = "MaxCut"


def _calculate_maxcut_objective_direct(adjacency_matrix: Union[np.ndarray, cp.ndarray],
                                       bitstrings_array: Union[np.ndarray, cp.ndarray]) -> List[float]:
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
                obj += coeff * (bitstring[u] + bitstring[v] - 2 * bitstring[u] * bitstring[v])

        objectives_list.append(obj)
    return objectives_list


def _calculate_qubo_objective_direct(adjacency_matrix: Union[np.ndarray, cp.ndarray],
                                     bitstrings_array: Union[np.ndarray, cp.ndarray]) -> List[float]:
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


def _calculate_ising_objective_direct(adjacency_matrix: Union[np.ndarray, cp.ndarray],
                                      bitstrings_array: Union[np.ndarray, cp.ndarray]) -> List[float]:
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


def _map_maxcut_adjacency_to_ising(maxcut_adjacency: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
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


def _map_maxcut_adjacency_to_qubo(maxcut_adjacency: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
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


def _map_qubo_adjacency_to_maxcut(qubo_adjacency: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
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
    maxcut_adjacency = bck.pad(-qubo_adjacency,
                               pad_width=((0, 1), (0, 1)),
                               mode='constant')
    bck.fill_diagonal(maxcut_adjacency, 0)
    couplings_sums = bck.sum(qubo_adjacency, axis=1) + bck.diag(qubo_adjacency)

    maxcut_adjacency[:number_of_nodes_qubo, number_of_nodes_qubo] = couplings_sums
    maxcut_adjacency[number_of_nodes_qubo, :number_of_nodes_qubo] = couplings_sums

    return maxcut_adjacency


def _map_qubo_adjacency_to_ising(qubo_adjacency: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
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
    bck.fill_diagonal(ising_adjacency, bck.sum(qubo_adjacency, axis=1) + bck.diag(qubo_adjacency))

    return ising_adjacency


def _map_ising_adjacency_to_qubo(ising_adjacency: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
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


def _map_ising_adjacency_to_maxcut(ising_adjacency: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
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


    ising = ising_adjacency.copy()
    maxcut_adjacency = bck.pad(ising,
                               pad_width=((0, 1), (0, 1)),
                               mode='constant')
    bck.fill_diagonal(maxcut_adjacency, 0)
    local_fields_ising = bck.diag(ising)

    number_of_nodes_ising = ising_adjacency.shape[0]
    maxcut_adjacency[:number_of_nodes_ising, number_of_nodes_ising] = local_fields_ising
    maxcut_adjacency[number_of_nodes_ising, :number_of_nodes_ising] = local_fields_ising

    return maxcut_adjacency


def map_maxcut_solution_to_qubo(bitstring: Union[List[int], Tuple[int, ...], np.ndarray],
                                pm_input: bool = False) -> Tuple[int, ...]:
    """
    This function maps the MAXCUT solution to the QUBO solution, assuming that the original QUBO was mapped to MAXCUT.

    :param bitstring: N-dimensional 0s and 1s vector
    :param pm_input: if True, we assume that the input is in [-1,+1]
    :return: output bitstring: (N-1)-dimensional 0s and 1s vector
    """

    bts_pm = bitstring
    if not pm_input:
        assert not -1 in bitstring, "bitstring should be in {0,1} not {-1,1} if pm_input is False"
        bts_pm = [1 - 2 * x for x in bitstring]
    else:
        assert 0 not in bitstring, "bitstring should be in {-1,1} not {0,1} if pm_input is True"

    return tuple([int(1 / 2 * (1 - bts_pm[i] * bts_pm[-1])) for i in range(len(bts_pm) - 1)])




def map_maxcut_solution_to_ising(bitstring: Union[List[int], Tuple[int, ...], np.ndarray|cp.ndarray], ):
    """
    This function maps the MAXCUT solution to the ISING solution, assuming that the original ISING was mapped to MAXCUT.
    (so it reduces the dimension by one qubit).

    :param bitstring: N-dimensional 0s and 1s vector
    :return: output bitstring: (N-1)-dimensional 0s and 1s vector
    """
    #TODO(FBM): add version that does this for many bitstrings at once efficiently
    if bitstring[-1]==0:
        return bitstring[0:len(bitstring)-1]
    else:
        if isinstance(bitstring, (np.ndarray,cp.ndarray)):
            return bitstring[0:len(bitstring)-1]^1
        return [x^1 for x in bitstring[0:len(bitstring)-1]]


def map_maxcut_solutions_array_to_ising(bitstrings_array:np.ndarray|cp.ndarray):

    """
    Same as map_maxcut_solution_to_ising but for many bitstrings at once.
    :param bitstrings_array:
    :return:
    """


    #we make a mask based on the last bit value:
    mask = bitstrings_array[:,-1]==1
    #wherever mask is False, we don't do anything:
    #wherever mask is True, we flip all the bits
    bitstrings_array[mask] ^= 1

    #remove last column:
    return bitstrings_array[:,:-1]




def map_adjacency_between_formulations(input_adjacency: Union[np.ndarray, cp.ndarray],
                                       input_formulation: ProblemFormulationType,
                                       output_formulation: ProblemFormulationType):
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
            raise ValueError("Output representation should be either 'MAXCUT' or 'QUBO'")

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
            raise ValueError("Output representation should be either 'ISING' or 'MAXCUT'")

    else:
        raise ValueError("Input representation should be either 'ISING', 'MAXCUT' or 'QUBO'")
