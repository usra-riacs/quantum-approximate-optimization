# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import time

import numpy as np
import pandas as pd

from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

import pygad


def _solve_ising_hamiltonian_pygad(
    hamiltonian: ClassicalHamiltonian,
    num_generations=100,
    num_parents_mating=10,
    sol_per_pop=50,
    initial_population=None,
    keep_parents=-1,
    mutation_type="swap",
    allow_duplicate_genes=True,
    save_best_solutions=False,
    random_seed=None,
    mutation_probability=None,
):
    t0 = time.perf_counter()
    num_genes = hamiltonian.number_of_qubits

    if initial_population is not None:
        if isinstance(initial_population, np.ndarray):
            initial_population = np.array(initial_population, dtype=int)
        elif isinstance(initial_population, cp.ndarray):
            initial_population = cp.asnumpy(initial_population)
        else:
            raise ValueError("Initial population must be a numpy array or cupy array")

        sol_per_pop = initial_population.shape[0]

    if hamiltonian.default_backend == "cupy":

        def _cost_function(instance, sol, solution_idx):
            if np.any(sol) > 1:
                raise ValueError(f"Solution {sol} has values other than 0 or 1")
            return -float(hamiltonian.evaluate_energy(cp.array([sol]))[0])

    else:

        def _cost_function(instance, sol, solution_idx):
            if np.any(sol) > 1:
                raise ValueError(f"Solution {sol} has values other than 0 or 1")
            return -float(hamiltonian.evaluate_energy([sol])[0])

    ga_instance = pygad.GA(
        fitness_func=_cost_function,
        initial_population=initial_population,
        gene_type=int,
        num_genes=num_genes,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        sol_per_pop=sol_per_pop,
        keep_parents=keep_parents,
        mutation_type=mutation_type,
        mutation_probability=mutation_probability,
        allow_duplicate_genes=allow_duplicate_genes,
        save_best_solutions=save_best_solutions,
        suppress_warnings=True,
        random_seed=random_seed,
        gene_space=[0, 1],
    )
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    best_energy = float(hamiltonian.evaluate_energy([solution])[0])
    best_state = tuple(solution)

    t1 = time.perf_counter()
    runtime = t1 - t0

    df_res = pd.DataFrame(
        data={"solution": [best_state], "energy": [best_energy], "runtime": [runtime]}
    )

    return (best_state, best_energy), df_res


def solve_ising_hamiltonian_evolutionary(
    hamiltonian: ClassicalHamiltonian,
    solver_name: str = "pygad",
    solver_seed=None,
    num_generations=100,
    num_parents_mating=10,
    sol_per_pop=50,
    initial_population=None,
    keep_parents=-1,
    mutation_type="swap",
    allow_duplicate_genes=True,
    save_best_solutions=False,
    mutation_probability=None,
):
    if solver_name == "pygad":
        return _solve_ising_hamiltonian_pygad(
            hamiltonian=hamiltonian,
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            sol_per_pop=sol_per_pop,
            initial_population=initial_population,
            keep_parents=keep_parents,
            mutation_type=mutation_type,
            allow_duplicate_genes=allow_duplicate_genes,
            save_best_solutions=save_best_solutions,
            random_seed=solver_seed,
            mutation_probability=mutation_probability,
        )

    else:
        raise ValueError("Only PT solver is supported for now")
