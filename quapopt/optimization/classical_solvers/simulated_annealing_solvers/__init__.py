# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import List, Optional, Union

import numpy as np

from quapopt.additional_packages.pysa.ParallelTemperingRunner import (
    ParallelTemperingRunner,
)
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)


def _solve_ising_hamiltonian_parallel_tempering(
    hamiltonian: ClassicalHamiltonian,
    number_of_sweeps: int,
    temperatures: List[float],
    states_initial: Optional[np.ndarray] = None,
    qubits_orderings: Optional[Union[List[List[int]], List[int]]] = None,
    bias_probabilities: Optional[Union[List[float], float]] = None,
    number_of_threads: int = 1,
    precision_float=np.float32,
    solver_seed=None,
):

    runner = ParallelTemperingRunner(
        hamiltonian=hamiltonian, rng_seed=solver_seed, precision=precision_float
    )

    (best_state, best_energy), df_res = runner.run_parallel_tempering(
        number_of_sweeps=number_of_sweeps,
        temperatures=temperatures,
        states_initial=states_initial,
        qubits_orderings=qubits_orderings,
        bias_probabilities=bias_probabilities,
        number_of_threads=number_of_threads,
    )

    return (best_state, best_energy), df_res


def solve_ising_hamiltonian_pysa(
    hamiltonian: ClassicalHamiltonian,
    number_of_sweeps: int,
    temperatures: List[float],
    states_initial: Optional[np.ndarray] = None,
    qubits_orderings: Optional[Union[List[List[int]], List[int]]] = None,
    bias_probabilities: Optional[Union[List[float], float]] = None,
    number_of_threads: int = 1,
    precision_float=np.float32,
    solver_seed=None,
    solver_name: str = "PT",
):

    if solver_name == "PT":
        return _solve_ising_hamiltonian_parallel_tempering(
            hamiltonian=hamiltonian,
            number_of_sweeps=number_of_sweeps,
            temperatures=temperatures,
            states_initial=states_initial,
            qubits_orderings=qubits_orderings,
            bias_probabilities=bias_probabilities,
            number_of_threads=number_of_threads,
            precision_float=precision_float,
            solver_seed=solver_seed,
        )

    else:
        raise ValueError("Only PT solver is supported for now")
