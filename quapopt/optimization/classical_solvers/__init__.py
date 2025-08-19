# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 

from typing import List, Tuple, Union
import numpy as np
from tqdm import tqdm
import pandas as pd
from enum import Enum
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian


from quapopt.optimization.parameter_setting.non_adaptive_optimization.SimpleGridOptimizer import SimpleGridOptimizer
from quapopt.optimization.parameter_setting import (ParametersBoundType as PBT,
                                                    OptimizerType)


from quapopt.optimization.classical_solvers.mqlib_solvers import solve_ising_hamiltonian_mqlib

try:
    from quapopt.optimization.classical_solvers.simulated_annealing_solvers import solve_ising_hamiltonian_pysa
except(ImportError, ModuleNotFoundError):
    solve_ising_hamiltonian_pysa = None

try:
    from quapopt.optimization.QAOA.wrapped_solvers import solve_ising_hamiltonian_with_QAOA_simulator_exp_values
except(ImportError, ModuleNotFoundError):
    solve_ising_hamiltonian_with_QAOA_simulator_exp_values = None

try:
    from quapopt.optimization.classical_solvers.LocalSearch import solve_ising_hamiltonian_LocalSearch
except(ImportError, ModuleNotFoundError):
    solve_ising_hamiltonian_LocalSearch = None

class SolverNames(Enum):
    mqlib = 'MQLib'
    pysa = 'PySA'
    local_search = 'LocalSearch'

    qaoa_exp_values = 'QAOAExpValuesSimulator'

    evolutionary = 'Evolutionary'




def solve_ising_hamiltonian(hamiltonian: ClassicalHamiltonian,
                            solver_name:SolverNames,
                            solver_kwargs=None,
                            repetitions: int = 1,
                            show_progress_bar: bool = False):
    all_results = []


    if solver_kwargs is None:
        if solver_name == SolverNames.mqlib:
            solver_kwargs = {'solver_name': 'BURER2002',
                             'solver_timeout': 0.1,
                             'solver_seed': 0}
        elif solver_name == SolverNames.pysa:
            solver_kwargs = {'solver_name': 'PT',
                             'number_of_sweeps': 2**10,
                             'temperatures': np.linspace(0.5,4.0,15),
                             'solver_seed': 0}
        elif solver_name == SolverNames.local_search:
            solver_kwargs = {'hamming_distance': 2,
                             'number_of_trials_max': 2000,
                             'solver_seed': 0,
                             'solver_name':'HDLS'}

        elif solver_name == SolverNames.evolutionary:
            solver_kwargs = {'solver_name':'pygad',
                                'num_generations': 100,
                                'num_parents_mating': 10,
                                'sol_per_pop': 50,
                                'initial_population': None,
                                'keep_parents': -1,
                                'mutation_type': "swap",
                                'allow_duplicate_genes': True,

                                'save_best_solutions': False,
                                'random_seed': None,
                                'mutation_probability': None,
                                'solver_seed': 0,

                             }

        elif solver_name == SolverNames.qaoa_exp_values:
            line_search_size = 100
            single_bound = (0, 0.08)
            solver_kwargs = {'classical_optimizer': SimpleGridOptimizer(parameter_bounds=[(PBT.RANGE, single_bound)],
                                                                        max_trials=line_search_size),
                             'classical_optimizer_type': OptimizerType.custom,
                             'number_of_function_calls': line_search_size,
                             'analytical_betas_p1': True,
                             'solver_seed': 0}





    if 'solver_seed' not in solver_kwargs:
        solver_kwargs['solver_seed'] = 0
    else:
        if solver_kwargs['solver_seed'] is None:
            solver_kwargs['solver_seed'] = np.random.randint(0, 2 ** 32)

    main_solver_seed = solver_kwargs['solver_seed']
    ising_energy_best = np.inf
    ising_solution_best = None
    for seed_add in tqdm(list(range(repetitions)), position=0, disable=not show_progress_bar, colour='blue'):
        solver_seed_i = main_solver_seed+seed_add
        solver_kwargs['solver_seed'] = solver_seed_i

        if solver_name == SolverNames.mqlib:
            (ising_solution, ising_energy), df_here = solve_ising_hamiltonian_mqlib(hamiltonian=hamiltonian,
                                                                                    solver_kwargs=solver_kwargs)
        elif solver_name == SolverNames.pysa:
            (ising_solution, ising_energy), df_here = solve_ising_hamiltonian_pysa(hamiltonian=hamiltonian,
                                                                                  **solver_kwargs)

        elif solver_name == SolverNames.local_search:
            solver_kwargs_2 = solver_kwargs.copy()
            del solver_kwargs_2['solver_seed']
            if 'solver_name' in solver_kwargs_2:
                subsolver_name = solver_kwargs_2['solver_name']
                del solver_kwargs_2['solver_name']
            else:
                subsolver_name = 'HDLS'

            (ising_solution, ising_energy), df_here = solve_ising_hamiltonian_LocalSearch(hamiltonian=hamiltonian,
                                                                                          solver_name=subsolver_name,
                                                                                          solver_seed=solver_seed_i,
                                                                                          solver_kwargs=solver_kwargs_2)
        elif solver_name == SolverNames.evolutionary:
            solver_kwargs_2 = solver_kwargs.copy()
            del solver_kwargs_2['solver_seed']
            if 'solver_name' in solver_kwargs_2:
                subsolver_name = solver_kwargs_2['solver_name']
                del solver_kwargs_2['solver_name']
            else:
                subsolver_name = 'pygad'

            (ising_solution, ising_energy), df_here = solve_ising_hamiltonian_evolutionary(hamiltonian=hamiltonian,
                                                                                          solver_name=subsolver_name,
                                                                                          solver_seed=solver_seed_i,
                                                                                          solver_kwargs=solver_kwargs_2)



        elif solver_name == SolverNames.qaoa_exp_values:
            solver_kwargs_2 = solver_kwargs.copy()
            del solver_kwargs_2['solver_seed']

            if 'solver_name' in solver_kwargs_2:
                subsolver_name = solver_kwargs_2['solver_name']
                del solver_kwargs_2['solver_name']
            else:
                subsolver_name = 'p1+QRR'

            (ising_solution, ising_energy), df_here = solve_ising_hamiltonian_with_QAOA_simulator_exp_values(hamiltonian=hamiltonian,
                                                                                                             solver_name=subsolver_name,
                                                                                                             solver_seed=solver_seed_i,
                                                                                                              solver_kwargs=solver_kwargs_2)

        else:
            raise NotImplementedError(f"Solver {solver_name} not implemented")

        if ising_energy < ising_energy_best:
            ising_energy_best = ising_energy
            ising_solution_best = ising_solution

        all_results.append(df_here)


    return (ising_solution_best, ising_energy_best), pd.concat(all_results, ignore_index=True)

























#
# def solve_ising_hamiltonian_mqlib(hamiltonian_list_representation: List[Tuple[float, Tuple[int, ...]]],
#                                   solver_kwargs:dict=None,
#                                   flip_sign:bool=False,
#                                   repetitions:int=1,
#                                   show_progress_bar:bool=False):
#     from ancillary_functions_usra import graph_tools as gt
#
#     if solver_kwargs is None:
#         solver_kwargs = {'solver_name': 'BURER2002',
#                          'solver_timeout': 10,
#                          'solver_seed': 42, }
#     else:
#         if 'solver_seed' not in solver_kwargs:
#             solver_kwargs['solver_seed'] = 42
#
#     graph_ising = gt.map_list_representation_to_graph(hamiltonian_list_representation)
#     graph_maxcut = gt.map_ising_graph_to_maxcut(ising_graph=graph_ising)
#
#     all_results = []
#     ising_energy = np.inf
#     ising_solution = None
#
#
#     for seed_add in tqdm(list(range(repetitions)), position=0, disable=not show_progress_bar, colour='blue'):
#         solver_kwargs['solver_seed'] += seed_add
#
#         res_mqlib = gt.run_mqlib_solver(problem_graph=graph_maxcut,
#                                         problem_type='M',
#                                         # TODO FBM: Something seems to be wrong with the sign
#                                         flip_sign=not flip_sign,
#                                         **solver_kwargs)
#
#         maxcut_solution = res_mqlib['solution']
#         maxcut_energy = res_mqlib['objval']
#
#         solution = gt.map_maxcut_solution_to_ising(bitstring=maxcut_solution)
#         energy = calculate_energies_from_bitstrings(hamiltonian=hamiltonian_list_representation,
#                                                     bitstrings_array=[solution])[0]
#
#         if energy < ising_energy:
#             ising_energy = energy
#             ising_solution = solution
#
#         runtime = res_mqlib['bestsolhistory_runtimes'][-1]
#         df_here = pd.DataFrame(data={'solution': [tuple(solution)],
#                                      'energy': [energy],
#                                      'runtime': [runtime]})
#         all_results.append(df_here)
#
#     res_all = pd.concat(all_results, ignore_index=True)
#
#     return (ising_solution, ising_energy), res_all
