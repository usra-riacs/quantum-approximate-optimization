# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
from typing import Union, List, Tuple
from tqdm.notebook import tqdm
import numpy as np

from quapopt import AVAILABLE_SIMULATORS
if 'cupy' in AVAILABLE_SIMULATORS:
    import cupy as cp
else:
    import numpy as cp

import pandas as pd
import MQLib


from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.hamiltonians.representation import convert_list_representation_to_adjacency_matrix
from quapopt.hamiltonians.representation.problem_formulations import (map_maxcut_solution_to_ising,
                                                                      ProblemFormulationType,
                                                                      map_adjacency_between_formulations,
                                                                      _calculate_ising_objective_direct,
                                                                      _calculate_qubo_objective_direct,
                                                                      _calculate_maxcut_objective_direct)
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian



def solve_maxcut_problem_with_mqlib(adjacency_matrix: np.ndarray,
                                    solver_timeout: float = 0.1,
                                    solver_name: str = 'BURER2002',
                                    solver_seed=42,):

    dat = adjacency_matrix.copy()
    mqlib_instance = MQLib.Instance(problem="M",
                                    dat=dat)
    # MQLIB MAXIMIZES <x|Q|x> if QUBO is given.
    mqlib_result = MQLib.runHeuristic(heuristic=solver_name,
                                       instance=mqlib_instance,
                                       rtsec=solver_timeout,
                                       # TODO FBM: WHat is this helper function?
                                       cb_fun=lambda x: 1,
                                       seed=solver_seed)

    mqlib_result['solution'] = (1 - mqlib_result['solution']) / 2
    mqlib_result['solution'] = mqlib_result['solution'].astype(int)

    return mqlib_result


def solve_ising_hamiltonian_mqlib(hamiltonian: Union[List[Tuple[float, Tuple[int, ...]]], np.ndarray,ClassicalHamiltonian],
                                  solver_kwargs=None,
                                  number_of_qubits=None,
                                  maximization=False):

    if solver_kwargs is None:
        solver_kwargs = {}

    solver_seed = solver_kwargs.get('solver_seed',42)
    solver_name = solver_kwargs.get('solver_name', 'BURER2002')
    solver_timeout = solver_kwargs.get('solver_timeout', 0.1)

    solver_kwargs['solver_name']=solver_name
    solver_kwargs['solver_seed']=solver_seed
    solver_kwargs['solver_timeout']=solver_timeout



    if isinstance(hamiltonian, list):
        adjacency_matrix = convert_list_representation_to_adjacency_matrix(hamiltonian,
                                                                           matrix_type='SYM',
                                                                           backend='numpy',
                                                                           number_of_qubits=number_of_qubits)

        _local_fields_present = np.count_nonzero(np.diag(adjacency_matrix))>0


    elif isinstance(hamiltonian,np.ndarray):
        adjacency_matrix = hamiltonian.copy()
        _local_fields_present = np.count_nonzero(np.diag(adjacency_matrix))>0

    elif isinstance(hamiltonian,cp.ndarray):
        _local_fields_present = cp.count_nonzero(cp.diag(hamiltonian))>0
        adjacency_matrix = cp.asnumpy(hamiltonian)

    elif isinstance(hamiltonian,ClassicalHamiltonian):
        adjacency_matrix = hamiltonian.get_adjacency_matrix(matrix_type='SYM',
                                                             backend='numpy')

        _local_fields_present = 1 in hamiltonian.localities


    else:
        raise ValueError('Hamiltonian must be either a list or a numpy array.')

    if maximization:
        sign = -1.0
    else:
        sign = 1.0



    if _local_fields_present:
        adjacency_matrix_maxcut = map_adjacency_between_formulations(input_adjacency=sign*adjacency_matrix,
                                                                      input_formulation=ProblemFormulationType.ISING,
                                                                      output_formulation=ProblemFormulationType.MAXCUT)
    else:
        adjacency_matrix_maxcut = sign*adjacency_matrix

    res_mqlib = solve_maxcut_problem_with_mqlib(adjacency_matrix=adjacency_matrix_maxcut,
                                                **solver_kwargs)

    maxcut_solution = res_mqlib['solution']
    #maxcut_energy = res_mqlib['objval']

    if _local_fields_present:
        ising_solution = map_maxcut_solution_to_ising(bitstring=maxcut_solution)
    else:
        ising_solution = maxcut_solution

    ising_energy = em.calculate_energies_from_bitstrings_2_local(bitstrings_array=np.array([ising_solution]),
                                                           adjacency_matrix=adjacency_matrix,
                                                           computation_backend='numpy',
                                                           output_backend='numpy')[0]

    runtime = res_mqlib['bestsolhistory_runtimes'][-1]
    df_here = pd.DataFrame(data={'solution': [tuple(ising_solution)],
                                 'energy': [ising_energy],
                                 'runtime': [runtime]})

    return (ising_solution, ising_energy), df_here
