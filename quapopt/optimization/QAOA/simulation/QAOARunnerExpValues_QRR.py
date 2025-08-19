# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

import heapq
import pandas as pd
import numpy as np
from typing import Tuple, List
from quapopt.hamiltonians.representation import HamiltonianListRepresentation
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization import OptimizationResult
from quapopt.optimization.QAOA.simulation.QAOARunnerExpValues import QAOARunnerExpValues
from quapopt.meta_algorithms.QRR import QRR_functions as qrr_fun

class QAOARunnerExpValues_QRR:
    def __init__(self):
        pass

    def run_optimization(self,
                         cost_hamiltonian: HamiltonianListRepresentation,
                         number_of_qubits: int,
                         angles: List[Tuple], # (phase , mixer)
                         n_best_energy: int,):
        import cupy as bck
        qaoa_runner_analytical = QAOARunnerExpValues(number_of_qubits=number_of_qubits,
                                                     hamiltonian_representations_cost=[cost_hamiltonian],
                                                     simulator_name=None,
                                                     precision_float=np.float32)
        hamiltonian_representations = ClassicalHamiltonian(cost_hamiltonian, number_of_qubits)
        coupling, local_fields = hamiltonian_representations.get_couplings_and_local_fields()
        if local_fields is None:
            local_fields = np.array([0.]*hamiltonian_representations.number_of_qubits, dtype= np.float32)
        else:
            local_fields = np.array(local_fields, dtype= np.float32)
            
        couplings_i = qaoa_runner_analytical.couplings_cost_dict[0]['numpy']
        couplings_i = bck.asarray(couplings_i)
        couplings_i_nonzero_mask = couplings_i != 0
        couplings_i_inverse = bck.zeros_like(couplings_i)
        couplings_i_inverse[couplings_i_nonzero_mask] = 1 / couplings_i[couplings_i_nonzero_mask]
        data_list = []
        best_energy = bck.inf
        best_bitstrings = None
        for param_idx in range(len(angles)):
            qaoa_result_j = qaoa_runner_analytical.run_qaoa(list(angles[param_idx]),
                                                            store_correlators=True)
            correlators_j = qaoa_result_j.correlators
            correlators_weighted_j = bck.asarray(qaoa_result_j.correlators)
            correlators_unweighted_j = correlators_weighted_j * couplings_i_inverse
            #TODO(FBM): normal QRR is supposed to round eigenvectors. 
            # Here I return them and round them manually to allow for future extensions to vanilla QRR
            candidate_eigenvectors_j, eigvals_j = qrr_fun.find_candidate_solutions_QRR(correlations_matrix=correlators_unweighted_j,
                                                                                       return_eigenvectors=True,
                                                                                       solver='cupy')

            candidate_eigenvectors_j[candidate_eigenvectors_j<=0] = 0
            candidate_eigenvectors_j[candidate_eigenvectors_j>0] = 1
            cost_hamiltonian_j = qaoa_runner_analytical.hamiltonian_representations_cost[0]

            energies_j = cost_hamiltonian_j.evaluate_energy(bitstrings_array=candidate_eigenvectors_j,
                                                            backend_computation='cupy')
            best_energy_index_j = bck.argmin(energies_j)
            best_k_energy_index_j = heapq.nsmallest(n_best_energy, range(len(energies_j)), key=energies_j.__getitem__)
            best_energy_j = float(energies_j[best_energy_index_j])
            best_k_energy_j = [float(energies_j[index]) for index in best_k_energy_index_j]
            best_solution_j = bck.asnumpy(candidate_eigenvectors_j[best_energy_index_j])
            best_solution_j = best_solution_j.astype(int)
            best_solution_j = tuple(best_solution_j)
            best_k_solution_j = [tuple(bck.asnumpy(candidate_eigenvectors_j[index]).astype(int)) for index in best_k_energy_index_j]
            if best_energy_j < best_energy:
                best_energy = best_energy_j
                best_solution = best_solution_j
                best_arguments = (angles[param_idx], 0)
            data_j = {
                    "TrialIndex": [param_idx]*n_best_energy,
                    "Energy": best_k_energy_j,
                    "Bitstring": best_k_solution_j,
                    "Angles": [angles[param_idx]]*n_best_energy,
                    }
            data_list.append(pd.DataFrame(data_j))
            
        optimizer_res = OptimizationResult(best_value=best_energy,
                                           best_arguments=best_arguments,
                                           trials_dataframe=pd.concat(data_list))
        return (best_energy, best_solution), (optimizer_res, None, None)
    
        