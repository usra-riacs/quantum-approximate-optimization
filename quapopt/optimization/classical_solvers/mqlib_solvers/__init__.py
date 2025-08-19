# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.
from typing import Union, List, Tuple
from tqdm import tqdm
import numpy as np

try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
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
                                    solver_seed=42):

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
        solver_kwargs = {'solver_name': 'BURER2002',
                         'solver_timeout': 0.1,
                         'solver_seed': 42, }
    else:
        if 'solver_seed' not in solver_kwargs:
            solver_kwargs['solver_seed'] = -1

    if isinstance(hamiltonian,list):
        adjacency_matrix = convert_list_representation_to_adjacency_matrix(hamiltonian,
                                                                           matrix_type='SYM',
                                                                           backend='numpy',
                                                                           number_of_qubits=number_of_qubits)
    elif isinstance(hamiltonian,np.ndarray):
        adjacency_matrix = hamiltonian.copy()
    elif isinstance(hamiltonian,cp.ndarray):
        adjacency_matrix = cp.asnumpy(hamiltonian)
    elif isinstance(hamiltonian,ClassicalHamiltonian):
        adjacency_matrix = hamiltonian.get_adjacency_matrix(matrix_type='SYM',
                                                             backend='numpy')


    else:
        raise ValueError('Hamiltonian must be either a list or a numpy array.')

    if maximization:
        sign = -1.0
    else:
        sign = 1.0

    adjacency_matrix_maxcut = map_adjacency_between_formulations(input_adjacency=sign*adjacency_matrix,
                                                          input_formulation=ProblemFormulationType.ISING,
                                                          output_formulation=ProblemFormulationType.MAXCUT)

    res_mqlib = solve_maxcut_problem_with_mqlib(adjacency_matrix=adjacency_matrix_maxcut,
                                                **solver_kwargs)

    maxcut_solution = res_mqlib['solution']
    maxcut_energy = res_mqlib['objval']


    ising_solution = map_maxcut_solution_to_ising(bitstring=maxcut_solution)
    ising_energy = em.calculate_energies_from_bitstrings_2_local(bitstrings_array=np.array([ising_solution]),
                                                           adjacency_matrix=adjacency_matrix,
                                                           computation_backend='numpy',
                                                           output_backend='numpy')[0]

    runtime = res_mqlib['bestsolhistory_runtimes'][-1]
    df_here = pd.DataFrame(data={'solution': [tuple(ising_solution)],
                                 'energy': [ising_energy],
                                 'runtime': [runtime]})

    return (ising_solution, ising_energy), df_here





if __name__ == '__main__':
    # TODO(FBM): should add proper tests
    import itertools
    import numpy as np
    from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf

    noq_test = 12

    for seed_test in range(0, 10):
        rng_test = np.random.default_rng(seed=seed_test)
        all_bitstrings_n = list(itertools.product([0, 1], repeat=noq_test))
        all_bitstrings_n = np.array(all_bitstrings_n)

        test_ISING_matrix = rng_test.uniform(-5, 5, (noq_test, noq_test))
        test_ISING_matrix = 0.5 * (test_ISING_matrix + test_ISING_matrix.T)

        input_matrix_name = ProblemFormulationType.ISING
        output_matrix_name = ProblemFormulationType.MAXCUT

        input_matrix = test_ISING_matrix
        output_matrix = map_adjacency_between_formulations(input_adjacency=input_matrix,
                                                           input_formulation=input_matrix_name,
                                                           output_formulation=output_matrix_name)
        if input_matrix_name == ProblemFormulationType.ISING:
            energies_bruteforce = em.calculate_energies_from_bitstrings_2_local(adjacency_matrix=input_matrix,
                                                                                bitstrings_array=all_bitstrings_n,
                                                                               computation_backend='numpy',
                                                                               output_backend='numpy')
        else:
            raise ValueError("Input matrix name should be either 'ISING', 'QUBO' or 'MAXCUT'")

        if input_matrix_name in [ProblemFormulationType.ISING]:
            best_energy_index = np.argmin(energies_bruteforce)
            best_bitstring_input = all_bitstrings_n[best_energy_index]
            best_energy_input = energies_bruteforce[best_energy_index]
        else:
            raise ValueError("Input matrix name should be either 'ISING', 'QUBO' or 'MAXCUT")

        best_found_solutions = {
            f"{input_matrix_name} (ORIGINAL) (BRUTEFORCE)": (best_bitstring_input, best_energy_input)}

        (best_bitstring_output, best_energy_output), _ = solve_ising_hamiltonian_mqlib(hamiltonian=input_matrix,
                                                                                        solver_kwargs={
                                                                                             'solver_name': 'BURER2002',
                                                                                             'solver_timeout': 0.1,
                                                                                             'solver_seed': seed_test},
                                                                                        number_of_qubits=noq_test)


        if input_matrix_name in [ProblemFormulationType.ISING]:
            best_energy_output = _calculate_ising_objective_direct(adjacency_matrix=input_matrix,
                                                                   bitstrings_array=[best_bitstring_output])[0]
        else:
            raise ValueError("Input matrix name should be either 'ISING', 'QUBO' or 'MAXCUT")

        best_found_solutions[f"{input_matrix_name}->{output_matrix_name} (MQLIB)"] = (
            best_bitstring_output, best_energy_output)

        all_energies = [x[1] for x in best_found_solutions.values()]
        correct = np.allclose(all_energies,
                              all_energies[0])

        if correct:
            pass
        else:
            anf.cool_print("FAILED TEST:", f"{input_matrix_name}-->ALL (BRUTEFORCE)-{seed_test}", 'red')
            for key, value in best_found_solutions.items():
                print(key)
                print(value)
                print()
            raise KeyboardInterrupt

        # raise ValueError("STOP")
    anf.cool_print("ALL TESTS PASSED", '', 'green')
