# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

from typing import List, Tuple, Union
from numba import cuda
import numpy as np
import gc

@cuda.jit
def _compute_energies_integers_kernel(coefficient,
                                      position_mask,
                                      energies_array_out):
    idx = cuda.grid(1)

    if idx>=len(energies_array_out):
        return
    parity = cuda.popc(idx & position_mask) & 1
    if parity:
        cuda.atomic.add(energies_array_out, idx, -coefficient)
    else:
        cuda.atomic.add(energies_array_out, idx, coefficient)

def _compute_all_energies_integers(hamiltonian_list_representation:List[Tuple[Union[float,int], Tuple[int,...]]],
                                   dtype=np.float32):
    number_of_qubits = max([max(interaction[1]) for interaction in hamiltonian_list_representation]) + 1
    dim = 2**number_of_qubits
    #print("YOOOO", dim, number_of_qubits)

    energies_array = cuda.to_device(np.zeros(dim, dtype=dtype))

    for coeff, qubit_indices in hamiltonian_list_representation:
        position_mask = sum(2 ** (number_of_qubits - x - 1) for x in qubit_indices)
        _compute_energies_integers_kernel.forall(dim)(coeff,
                                                      position_mask,
                                                      energies_array)

    array = energies_array.copy_to_host()
    cuda.synchronize()
    del energies_array
    gc.collect()

    return array

def cuda_solve_hamiltonian(hamiltonian_list_representation):
    return _compute_all_energies_integers(hamiltonian_list_representation)


@cuda.jit
def _get_all_two_1s_bitstrings_kernel(bitstrings_array_out):
    idx, idy = cuda.grid(2)


    #I have a 2D grid of size (number_of_pairs, number_of_qubits)
    #I would like to generate all bitstrings with two 1s
    #First, I need flattened index:
    number_of_qubits = bitstrings_array_out.shape[1]

    if not (idx < idy < number_of_qubits):
        return

    pair_index = idx * number_of_qubits - idx * (idx + 1) // 2 + (idy - idx - 1)

    if pair_index>=bitstrings_array_out.shape[0]:
        return

    #Now I need to place ones in the right places
    bitstrings_array_out[pair_index, idy] = 1
    bitstrings_array_out[pair_index, idx] = 1


def get_all_two_1s_bitstrings_cuda(number_of_qubits:int,
                              dtype=np.int32,
                              include_one_1s_bitstrings:bool=False):
    number_of_pairs = number_of_qubits*(number_of_qubits-1)//2
    bitstrings_array = cuda.to_device(np.zeros((number_of_pairs, number_of_qubits), dtype=dtype))

    threadsperblock = (32, 32)
    blockspergrid_x = (number_of_pairs + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (number_of_qubits + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _get_all_two_1s_bitstrings_kernel[blockspergrid, threadsperblock](bitstrings_array)

    array = bitstrings_array.copy_to_host()
    cuda.synchronize()
    del bitstrings_array
    gc.collect()

    if include_one_1s_bitstrings:
        array = np.concatenate((array,np.eye(number_of_qubits, dtype=dtype)), axis=0)

    return array




@cuda.jit
def _calculate_2q_marginals_from_bitstrings_kernel(bitstrings_array,
                                                 marginals_array_out):
    idx, idy = cuda.grid(2)
    number_of_qubits = bitstrings_array.shape[1]


    if not (idx < idy < number_of_qubits):
        return

    #First, I need flattened index:
    number_of_qubits = bitstrings_array.shape[1]
    # outcomes_subset = bitstrings_array[:,[idx,idy]]
    col_idx = bitstrings_array[:,idx]
    col_idy = bitstrings_array[:,idy]

    flattened_pair_index = idx * number_of_qubits - idx * (idx + 1) // 2 + (idy - idx - 1)

    for bi,bj in zip(col_idx, col_idy):
       # bi, bj = outcome
        if bi==0 and bj==0:
            cuda.atomic.add(marginals_array_out, (flattened_pair_index,0,0), 1)
        elif bi==0 and bj==1:
            cuda.atomic.add(marginals_array_out, (flattened_pair_index,1,0), 1)
        elif bi==1 and bj==0:
            cuda.atomic.add(marginals_array_out, (flattened_pair_index,2,0), 1)
        elif bi==1 and bj==1:
            cuda.atomic.add(marginals_array_out, (flattened_pair_index,3,0), 1)


def calculate_2q_marginals_from_bitstrings_cuda(bitstrings_array:np.ndarray,
                                            dtype=np.int32):
    number_of_qubits = bitstrings_array.shape[1]
    number_of_pairs = number_of_qubits*(number_of_qubits-1)//2
    marginals_array = cuda.to_device(np.zeros((number_of_pairs, 4,1), dtype=dtype))

    threadsperblock = (32, 32)
    blockspergrid_x = (number_of_pairs + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (number_of_qubits + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _calculate_2q_marginals_from_bitstrings_kernel[blockspergrid, threadsperblock](bitstrings_array,
                                                                                   marginals_array)

    array = marginals_array.copy_to_host()
    cuda.synchronize()
    del marginals_array
    gc.collect()

    return array






