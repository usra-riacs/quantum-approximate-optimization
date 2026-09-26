# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np
from typing import Tuple, Union, List
import time
from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em, ancillary_functions as anf
from quapopt.optimization.classical_solvers.LocalSearch.cython_implementation.local_swap_cython import (
    run_exhaustive_swap_choose_first_cython, run_single_swap_choose_best_cython)



from enum import Enum

class LocalSwapStrategy(Enum):
    choose_first = "ChooseFirst"
    choose_first_shuffle = "ChooseFirstShuffle"
    choose_best = "ChooseBest"

    none = 'None'


class LocalSwapConfig:


    def __init__(self,
                 choice:LocalSwapStrategy,
                 greedy:bool,
                 one_sweep:bool=True,
                 two_sweep:bool=True
                 ):

        self._choice = choice
        self._greedy = greedy



def _run_all1swap(bitstring_pm: np.ndarray | cp.ndarray,
                             adjacency_matrix: np.ndarray | cp.ndarray,
                             bck: Union[np, cp],
                             tolerance:float=0.02,
                             return_adj_matrix:bool=False):

    final_bts = bitstring_pm.copy()
    bitflip_outer = bck.outer(bitstring_pm, bitstring_pm)
    adjacency_matrix_i = adjacency_matrix * bitflip_outer
    best_energy_i = em.calculate_energies_from_bitstrings_2_local(bitstrings_array=bitstring_pm.reshape(1, -1),
                                                                  adjacency_matrix=adjacency_matrix_i)

    move_made = True
    while move_made:
        move_made = False
        t0 = time.perf_counter()
        # This gives: energy_flipped-energy_initial
        deltas_E_i = -2 * adjacency_matrix_i.sum(axis=1)
        t1 = time.perf_counter()
        mask = deltas_E_i < -tolerance
        first_index = bck.argmax(mask)
        if not mask[first_index]:
            continue
        move_made = True
        t2 = time.perf_counter()
        best_energy_i = best_energy_i + deltas_E_i[first_index]

        adjacency_matrix_i[first_index, :] *= -1
        adjacency_matrix_i[:, first_index] *= -1

        final_bts[first_index] *= -1

    if return_adj_matrix:
        return final_bts, adjacency_matrix_i

    return final_bts

def _run_all2swap(bitstring_pm: np.ndarray | cp.ndarray,
                             adjacency_matrix: np.ndarray | cp.ndarray,
                             bck: Union[np, cp],
                             tolerance:float=0.2,
                             return_adj_matrix:bool=False):

    final_bts = bitstring_pm.copy()
    bitflip_outer = bck.outer(bitstring_pm, bitstring_pm)
    adjacency_matrix_i = adjacency_matrix * bitflip_outer
    best_energy_i = em.calculate_energies_from_bitstrings_2_local(bitstrings_array=bitstring_pm.reshape(1, -1),
                                                                  adjacency_matrix=adjacency_matrix_i)

    move_made = True


    while move_made:
        move_made = False
        t0 = time.perf_counter()
        # This gives: energy_flipped-energy_initial
        deltas_E_i = -2 * adjacency_matrix_i.sum(axis=1)
        t0 = time.perf_counter()
        deltas_E_i_j = deltas_E_i[:, None] + deltas_E_i[None, :] + 4 * adjacency_matrix_i
        # Only consider edge pairs (matching C++ MQLib All2Swap)
        deltas_E_i_j *= (adjacency_matrix_i != 0)
        bck.fill_diagonal(deltas_E_i_j, 0.0)
        t1 = time.perf_counter()

        mask = bck.where(deltas_E_i_j < -tolerance)
        if len(mask[0]) == 0:
            continue
        move_made = True

        t2 = time.perf_counter()
        first_index, second_index = mask[0][0], mask[1][0]

        best_energy_i = best_energy_i + deltas_E_i_j[first_index, second_index]
        #best_energy_local = best_energy_i

        idx = [first_index, second_index]
        adjacency_matrix_i[idx, :] *= -1
        adjacency_matrix_i[:, idx] *= -1

        final_bts[first_index] *= -1
        final_bts[second_index] *= -1
    if return_adj_matrix:
        return final_bts, adjacency_matrix_i
    return final_bts


def run_exhaustive_swap_choose_first(
        bitstring_pm: np.ndarray | cp.ndarray,
        adjacency_matrix: np.ndarray | cp.ndarray,
        bck: Union[np, cp],
        tolerances_list: Tuple[float, float] = (0.02, 0.2),
        run_one_sweep:bool=True,
        run_two_sweep:bool=True

):
    final_bts = bitstring_pm.copy()
    adjacency_matrix_i = adjacency_matrix.copy()
    for locality_index, local_tolerance, run_sweep in zip(range(2), tolerances_list, [run_one_sweep, run_two_sweep]):
        if not run_sweep:
            continue

        if locality_index == 0:
            final_bts, adjacency_matrix_i = _run_all1swap(bitstring_pm=bitstring_pm,
                                                                     adjacency_matrix=adjacency_matrix_i,
                                                                     bck=bck,
                                                                     tolerance=local_tolerance,
                                                                     return_adj_matrix=True)
        elif locality_index == 1:
            final_bts = _run_all2swap(bitstring_pm=bitstring_pm, adjacency_matrix=adjacency_matrix_i, bck=bck, tolerance=local_tolerance)
        else:
            raise ValueError("locality_index must be 0 or 1")




    return final_bts


def run_exhaustive_swap_choose_best(bitstring_pm: np.ndarray | cp.ndarray,
                                    adjacency_matrix: np.ndarray | cp.ndarray,
                                    bck:Union[np,cp],
                                    tolerances_list:Tuple[float,float]=(0.02, 0.2),
                                    local_neighborhoods:List[np.ndarray | cp.ndarray ]= None,
                                    run_one_swap: bool = True,
                                    run_two_swap: bool = True

                                    ):


    final_bts = bitstring_pm.copy()

    bitflip_outer = bck.outer(bitstring_pm, bitstring_pm)
    adjacency_matrix_i = adjacency_matrix * bitflip_outer
    products = bck.einsum('ij,ij->i',
                          bck.dot(bitstring_pm.reshape(1,-1), adjacency_matrix_i) / 2,
                          bitstring_pm.reshape(1,-1))

    best_energy_i = products[0]

    for locality_index, local_tolerance, local_neighborhood, run_sweep in zip(range(2), tolerances_list, local_neighborhoods, [run_one_swap, run_two_swap]):

        if not run_sweep:
            continue

        move_made = True
        while move_made:
            move_made = False
            energies_local = bck.einsum('ij,ij->i',
                                  bck.dot(local_neighborhood, adjacency_matrix_i) / 2,
                                  local_neighborhood)

            best_energy_local_index = bck.argmin(energies_local)
            best_solution_local = local_neighborhood[best_energy_local_index].astype(int)
            best_energy_local = energies_local[best_energy_local_index]

            if best_energy_i-best_energy_local > local_tolerance:
                best_energy_i = best_energy_local
                move_made = True
                bitflip_outer = bck.outer(best_solution_local, best_solution_local)
                adjacency_matrix_i = adjacency_matrix_i * bitflip_outer

                final_bts *= best_solution_local

    return final_bts



def run_single_swap_choose_best_over_neighborhood(bitstrings_array:np.ndarray | cp.ndarray,
                                                  cost_hamiltonian,
                                                  tolerance_1swap=0.0,
                                                  tolerance_2swap=0.0,
                                                  pm_input:bool=False,
                                                  show_progress_bar:bool=False
                                                  )->np.ndarray | cp.ndarray:
    from tqdm.notebook import tqdm
    csr_indptr, csr_indices, csr_data = cost_hamiltonian.get_csr_arrays()

    if isinstance(bitstrings_array,np.ndarray):
        output_bck = np
    elif isinstance(bitstrings_array,cp.ndarray):
        output_bck = cp
    else:
        raise ValueError("bitstrings_array must be either numpy or cupy array")

    best_energy, best_bitstring = output_bck.inf, None


    bitstrings_array = em.convert_cupy_numpy_array(array=bitstrings_array.copy(),
                                                   output_backend='numpy')

    if pm_input:
        bts_pm = bitstrings_array
    else:
        # Cast before the spin map: an unsigned array wraps under 1 - 2 * bits (a uint8 1 becomes 255).
        bts_pm = 1 - 2 * bitstrings_array.astype(np.int32)

    for bts in tqdm(bts_pm,disable=not show_progress_bar, desc="Flippin'"):
        best_bitstring_neighbors = run_single_swap_choose_best_cython(bitstring_pm=bts,
                                                                      csr_indptr=csr_indptr,
                                                                      csr_indices=csr_indices,
                                                                      csr_data=csr_data,
                                                                      tolerance_1swap=tolerance_1swap,
                                                                      tolerance_2swap=tolerance_2swap
                                                                      )

        best_energy_neighbors = \
        cost_hamiltonian.evaluate_energy(bitstrings_array=[best_bitstring_neighbors],
                                                          pm_input=True)[0]

        if best_energy_neighbors < best_energy:
            best_energy = best_energy_neighbors
            best_bitstring = best_bitstring_neighbors

    best_bitstring = (1 - best_bitstring) / 2

    return float(best_energy), output_bck.array(best_bitstring,dtype=int)