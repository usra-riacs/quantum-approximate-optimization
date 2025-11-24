# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import numpy as np
import pandas as pd

from quapopt.hamiltonians.representation import HamiltonianListRepresentation
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)
from quapopt.meta_algorithms.QRR import QRR_functions as qrr_fun
from quapopt.optimization import OptimizationResult
from quapopt.optimization.QAOA.simulation.pauli_backprop.one_layer.cython_implementation import (
    cython_grad_p1_qaoa as math_grad_p1,
)
from quapopt.optimization.QAOA.simulation.pauli_backprop.one_layer.cython_implementation import (
    cython_p1_qaoa as math_p1,
)
from quapopt.optimization.QAOA.simulation.QAOARunnerExpValues import QAOARunnerExpValues

_DEFAULT_HYPERPARAMETERS_TPE = {
    "consider_prior": True,
    "prior_weight": 1.0,
    "consider_magic_clip": True,
    "consider_endpoints": False,
    "n_startup_trials": 10,
    "n_ei_candidates": 24,
    "multivariate": False,
    "group": False,
    "constant_liar": False,
}


class QAOAOptimizationAnalytical:
    def __init__(self):
        pass

    def run_optimization(
        self,
        cost_hamiltonian: HamiltonianListRepresentation,
        number_of_qubits: int,
        phase_angle: float,
        mixer_angle: float,
        number_of_function_calls: int,
        learning_rate: float,
    ):
        import cupy as bck

        qaoa_runner_analytical = QAOARunnerExpValues(
            number_of_qubits=number_of_qubits,
            hamiltonian_representations_cost=cost_hamiltonian,
            store_full_information_in_history=False,
            simulator_name=None,
            precision_float=np.float32,
        )
        hamiltonian_representations = ClassicalHamiltonian(
            cost_hamiltonian, number_of_qubits
        )
        coupling, local_fields = (
            hamiltonian_representations.get_couplings_and_local_fields()
        )
        if local_fields is None:
            local_fields = np.array(
                [0.0] * hamiltonian_representations.number_of_qubits, dtype=np.float32
            )
        else:
            local_fields = np.array(local_fields, dtype=np.float32)
        coupling = np.array(coupling, dtype=np.float32)
        data = {f"ARG-0": [], f"ARG-1": [], f"FunctionValue": []}
        qaoa_res = qaoa_runner_analytical.run_qaoa([phase_angle, mixer_angle])
        best_exp_value = qaoa_res.energy_mean
        best_argument = (phase_angle, mixer_angle)
        data[f"ARG-0"].append(phase_angle)
        data[f"ARG-1"].append(mixer_angle)
        data[f"FunctionValue"].append(qaoa_res.energy_mean)
        for ite in range(number_of_function_calls):
            double_correlation_grad = (
                math_grad_p1.cython_analytical_double_grad_QAOA_p1(
                    phase_angle=float(phase_angle),
                    mixer_angle=float(mixer_angle),
                    fields=local_fields,
                    correlations=coupling,
                )
            )
            grad_mixer = np.sum(double_correlation_grad[0])
            grad_phase = np.sum(double_correlation_grad[1])
            mixer_angle = (mixer_angle - grad_mixer * learning_rate) % (np.pi)
            phase_angle = (phase_angle - grad_phase * learning_rate) % (np.pi)
            qaoa_res = qaoa_runner_analytical.run_qaoa([phase_angle, mixer_angle])
            data[f"ARG-0"].append(phase_angle)
            data[f"ARG-1"].append(mixer_angle)
            data[f"FunctionValue"].append(qaoa_res.energy_mean)
            if qaoa_res.energy_mean < best_exp_value:
                best_exp_value = qaoa_res.energy_mean
                best_argument = (phase_angle, mixer_angle)
        correlation_matrix = math_p1.cython_analytical_QAOA_p1(
            phase_angle=float(phase_angle),
            mixer_angle=float(mixer_angle),
            correlations_phase=coupling,
            fields_phase=local_fields,
            correlations_cost=coupling,
            fields_cost=local_fields,
        )[1]
        correlatoion_matrix = bck.asarray(correlation_matrix)
        bck.inf
        best_solution = None
        # Extract and flatten the upper-triangular elements
        candidate_eigenvectors_j, eigvals_j = qrr_fun.find_candidate_solutions_QRR(
            correlations_matrix=correlatoion_matrix,
            return_eigenvectors=True,
            solver="cupy",
        )
        candidate_eigenvectors_j[candidate_eigenvectors_j <= 0] = 0
        candidate_eigenvectors_j[candidate_eigenvectors_j > 0] = 1
        energies_j = hamiltonian_representations.evaluate_energy(
            bitstrings_array=candidate_eigenvectors_j, backend_computation="cupy"
        )

        best_energy_index_j = bck.argmin(energies_j)
        best_energy_j = float(energies_j[best_energy_index_j])
        best_solution_j = bck.asnumpy(candidate_eigenvectors_j[best_energy_index_j])
        best_solution_j = best_solution_j.astype(int)
        best_solution_j = tuple(best_solution_j)

        df = pd.DataFrame(data)
        optimizer_res = OptimizationResult(
            best_value=best_solution, best_arguments=best_argument, trials_dataframe=df
        )
        all_candidates_solution = bck.asnumpy(candidate_eigenvectors_j)
        all_candidates_solution = candidate_eigenvectors_j.astype(int)
        return (best_energy_j, best_solution_j), optimizer_res, all_candidates_solution
