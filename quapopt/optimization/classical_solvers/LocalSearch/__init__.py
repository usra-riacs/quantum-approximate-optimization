# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import time
from typing import Any, Dict

import pandas as pd

from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)
from quapopt.optimization.classical_solvers.LocalSearch.HammingDistanceLocalSearchRunner import (
    HammingDistanceLocalSearchRunner,
)
from quapopt.optimization.classical_solvers.LocalSearch.RandomLocalSearchRunner import (
    RandomLocalSearchRunner,
)


def solve_ising_hamiltonian_LocalSearch(
    hamiltonian: ClassicalHamiltonian,
    solver_kwargs: Dict[str, Any],
    solver_seed=None,
    solver_name="HDLS",
):

    # solver_kwargs

    t0 = time.perf_counter()
    if solver_name == "HDLS":
        runner = HammingDistanceLocalSearchRunner(
            hamiltonian=hamiltonian,
        )

        ising_solution, ising_energy = runner.run_HDLS(
            **solver_kwargs,
            sampler_seed=solver_seed,
        )

    elif solver_name == "MCNDAR":
        solver_kwargs_2 = solver_kwargs.copy()
        if "convergence_criterion" in solver_kwargs:
            convergence_criterion = solver_kwargs["convergence_criterion"]
            del solver_kwargs_2["convergence_criterion"]
        else:
            convergence_criterion = None

        runner = RandomLocalSearchRunner(
            hamiltonian=hamiltonian,
            convergence_criterion=convergence_criterion,
            backend=hamiltonian.default_backend,
        )

        (ising_solution, ising_energy), _ = runner.run_NDAR(
            **solver_kwargs_2, sampler_seed=solver_seed
        )

    else:
        raise ValueError(f"Solver name {solver_name} not recognized.")
    t1 = time.perf_counter()
    runtime = t1 - t0

    df_res = pd.DataFrame(
        data={
            "solution": [tuple(ising_solution)],
            "energy": [ising_energy],
            "runtime": [runtime],
        }
    )

    return (ising_solution, ising_energy), df_res
