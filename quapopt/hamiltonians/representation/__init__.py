# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import List, Tuple, Union, Optional
import pydantic as pyd

HamiltonianListRepresentation = List[Tuple[Union[float, int], Tuple[pyd.conint(ge=0), ...]]]

import scipy as sc
import numpy as np

#Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
    import numpy as cp


def convert_list_representation_to_adjacency_matrix(hamiltonian_list_representation: HamiltonianListRepresentation,
                                                    matrix_type: str = 'SYM',
                                                    backend='numpy',
                                                    number_of_qubits: Optional[int] = None,
                                                    precision=np.float32
                                                    ) -> Union[sc.sparse.coo_matrix, np.ndarray, cp.ndarray]:
    """
    Converts a list representation of a Hamiltonian to an adjacency matrix.

    :param hamiltonian_list_representation: [(J_{ij}, (i,j)), ...]

    :param matrix_type:  'UT' for upper triangular, 'SYM' for symmetric
    :param backend: 'scipy', 'cupy', 'numpy'; 'scipy' assumes sparse representation

    :param number_of_qubits: optional, if not provided, it will be inferred from the list representation
    :return:  adjacency_matrix
    """

    if number_of_qubits is None:
        number_of_qubits = max([max(interaction[1]) for interaction in hamiltonian_list_representation]) + 1

    if matrix_type.lower() in ["ut"]:
        symmetric = False
    elif matrix_type.lower() in ["sym"]:
        symmetric = True
    else:
        raise ValueError("Matrix type not recognized. Only 'UT' and 'SYM' are allowed.")

    if backend.lower() in ['scipy']:
        data = []
        row_indices = []
        col_indices = []

        def _add_element(row, col, value):
            if value != 0:
                row_indices.append(row)
                col_indices.append(col)
                data.append(value)

        for coeff, tup in hamiltonian_list_representation:
            if len(tup) == 1:
                i = tup[0]
                _add_element(i, i, coeff)
            elif len(tup) == 2:
                i, j = tup

                if symmetric:
                    _add_element(i, j, coeff)
                    _add_element(j, i, coeff)
                else:
                    if i <= j:
                        _add_element(i, j, coeff)
                    else:
                        _add_element(j, i, coeff)
            else:
                raise ValueError('keys of weights_matrix should be tuples of length 1 or 2')

        # Convert lists to numpy arrays
        data = np.array(data)
        row_indices = np.array(row_indices)
        col_indices = np.array(col_indices)
        # Create the coo_array
        adjacency_matrix = sc.sparse.coo_array((data, (row_indices, col_indices)),
                                               shape=(number_of_qubits, number_of_qubits))
    else:
        if backend == 'cupy':
            bck = cp
        elif backend == 'numpy':
            bck = np
        else:
            raise ValueError("Backend not recognized. Only 'scipy', 'cupy' and 'numpy' are allowed.")

        adjacency_matrix = bck.zeros(shape=(number_of_qubits, number_of_qubits), dtype=precision)
        for coeff, tup in hamiltonian_list_representation:
            if len(tup) == 1:
                i = tup[0]
                adjacency_matrix[i, i] = coeff
            elif len(tup) == 2:
                i, j = tup
                if symmetric:
                    adjacency_matrix[i, j] = coeff
                    adjacency_matrix[j, i] = coeff
                else:
                    if i <= j:
                        adjacency_matrix[i, j] = coeff
                    else:
                        adjacency_matrix[j, i] = coeff
            else:
                raise ValueError('keys of weights_matrix should be tuples of length 1 or 2')

    return adjacency_matrix

def convert_adjacency_matrix_to_list_representation(adjacency_matrix:np.ndarray):
    number_of_qubits = adjacency_matrix.shape[0]

    hamiltonian_list_representation = []

    for i in range(number_of_qubits):
        if adjacency_matrix[i,i]!=0:
            hamiltonian_list_representation.append((adjacency_matrix[i,i], (i,)))
        for j in range(i+1, number_of_qubits):
            if adjacency_matrix[i,j]!=0:
                hamiltonian_list_representation.append((adjacency_matrix[i,j], (i,j)))
    return hamiltonian_list_representation




def convert_networkit_graph_to_list_representation(networkit_graph)->HamiltonianListRepresentation:
    import networkit as nk
    if not isinstance(networkit_graph, nk.Graph):
        raise ValueError("Input should be a networkit graph.")

    hamiltonian_list_representation = []
    for tup in networkit_graph.iterEdgesWeights():
        qi, qj = tup[0], tup[1]
        weight = tup[2]
        if qi==qj:
            hamiltonian_list_representation.append((weight, (qi,)))
        else:
            hamiltonian_list_representation.append((weight, (qi, qj)))

    return hamiltonian_list_representation


