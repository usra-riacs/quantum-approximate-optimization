# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import time
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import \
    ClassicalMeasurementNoiseSampler
from quapopt.data_analysis.data_handling import LoggingLevel
from quapopt.data_analysis.data_handling import STANDARD_NAMES_DATA_TYPES as SNDT
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import QAOAFunctionInputFormat as FIFormat
from quapopt.optimization.QAOA.QAOARunnerBase import QAOARunnerBase
from quapopt.optimization import EnergyResultMain
from quapopt.optimization.QAOA import QAOAResult
from quapopt import ancillary_functions as anf

class QAOARunnerStatevector(QAOARunnerBase):
    """
    Class for running QAOA with expectation values.
    """

    def __init__(self,
                 #number_of_qubits: int,
                 hamiltonian_representations_cost: List[ClassicalHamiltonian],
                 hamiltonian_representations_phase: Optional[List[ClassicalHamiltonian]] = None,
                 store_full_information_in_history=False,
                 solve_at_initialization=True,
                 logging_level: Optional[LoggingLevel] = None,
                 logger_kwargs: Optional[Dict[str, Any]] = None,
                 ) -> None:




        super().__init__(hamiltonian_representations_cost=hamiltonian_representations_cost,
                         hamiltonian_representations_phase=hamiltonian_representations_phase,
                         store_full_information_in_history=store_full_information_in_history,
                         solve_at_initialization=solve_at_initialization,
                         #number_of_qubits=number_of_qubits,
                         logging_level=logging_level,
                         logger_kwargs=logger_kwargs)

    def update_history(self,
                       qaoa_result: QAOAResult):
        self._update_history(qaoa_result=qaoa_result)
        best_energy = qaoa_result.energy_mean
        # This simulator does not store bitstrings so we do not pass it additionaly
        tup_to_store = (None, qaoa_result)
        self._best_results_container.add_result(result_to_add=tup_to_store,
                                                score=best_energy)

    def log_results(self,
                     qaoa_result: QAOAResult):

        if self.results_logger is None:
            return

        if self.logging_level is LoggingLevel.NONE or self.logging_level is None:
            return


        optimization_overview_df = qaoa_result.to_dataframe_main()
        optimization_overview_dt = SNDT.OptimizationOverview
        self.results_logger.write_results(dataframe=optimization_overview_df,
                                          data_type=optimization_overview_dt)

        statevector_df = pd.DataFrame(data={SNDT.StateVectors.id_long: [qaoa_result.statevector]})
        statevector_dt = SNDT.StateVectors

        self.results_logger.write_results(dataframe=qaoa_result.annotate_dataframe(statevector_df),
                                          data_type=statevector_dt)

    def get_best_results(self):
        return self._best_results_container.get_best_results()

    def run_qaoa(self,
                 *args,
                 qaoa_depth: int,
                 measurement_noise: ClassicalMeasurementNoiseSampler = None,
                 input_format: FIFormat = FIFormat.direct_list,
                 number_of_samples=None,
                 numpy_rng_sampling=None,
                 return_only_statevector:bool=False
                 ) -> QAOAResult:
        assert number_of_samples is None or number_of_samples == np.inf,\
            "This method is only implemented for number_of_samples=None or np.inf"

        if self._backends is None:
            self.initialize_backend_qokit()

        angles, hamiltonian_representation_index, trial_index = self._input_handler(args=args,
                                                                                    input_format=input_format,
                                                                                    qaoa_depth=qaoa_depth)

        gammas_j = angles[:qaoa_depth]
        betas_j = angles[qaoa_depth:]

        simulator_i = self._backends[hamiltonian_representation_index]
        hamiltonian_i = self._hamiltonian_representations_cost[hamiltonian_representation_index]

        if hamiltonian_i.spectrum is None:
            self._hamiltonian_representations_cost[hamiltonian_representation_index].solve_hamiltonian()
            hamiltonian_i = self._hamiltonian_representations_cost[hamiltonian_representation_index]

        if self._backend_name.lower() in ['qokit']:
            _result = simulator_i.simulate_qaoa(gammas_j * 2, betas_j)
            statevector_ideal = simulator_i.get_statevector(_result).reshape(-1, 1)

        elif self._backend_name.lower() in ['python']:
            _result = simulator_i.get_qaoa_statevector(angles_PS=gammas_j,
                                                       angles_mixer=betas_j)
            statevector_ideal = _result

        else:
            raise NotImplementedError('Only qokit simulator is supported')

        if return_only_statevector:
            return statevector_ideal

        statevector_ideal = anf.convert_cupy_numpy_array(array=statevector_ideal,
                                                         output_backend='numpy')
        prob_distro_ideal = em.cython_abs_squared(statevector_ideal.reshape(-1))
        exp_value_noiseless = em.cython_vdot(vector1=prob_distro_ideal.astype(np.float32),
                                             vector2=hamiltonian_i.spectrum.astype(np.float32))

        energy_result = EnergyResultMain(energy_mean_noiseless=exp_value_noiseless)
        energy_result.update_main_energy(noisy=False)

        if measurement_noise is None:
            qaoa_result = QAOAResult(statevector=statevector_ideal,
                                     energy_result=energy_result,
                                     trial_index=trial_index)

            self.log_results(qaoa_result=qaoa_result)
            return qaoa_result

        # raise NotImplementedError('Measurement noise is not supported yet')
        t0 = time.perf_counter()
        # prob_distro_ideal = em.cython_abs_squared(statevector_ideal.reshape(-1))

        t1 = time.perf_counter()

        if measurement_noise.noisy_hamiltonian_representations['cost'] is not None:
            hamiltonian_noisy, noisy_constant = measurement_noise.noisy_hamiltonian_representations['cost'][
                hamiltonian_representation_index]
        else:
            print('calculating noisy hamiltonians')
            noiseless_hamiltonian = self.hamiltonian_representations[
                hamiltonian_representation_index].hamiltonian_list_representation
            hamiltonian_noisy, noisy_constant = measurement_noise.transform_hamiltonian_to_noisy_version(
                hamiltonian_list=noiseless_hamiltonian)

        if measurement_noise.noisy_hamiltonian_spectra['cost'] is not None:
            noisy_spectrum = measurement_noise.noisy_hamiltonian_spectra['cost'][hamiltonian_representation_index]
        else:
            print('calculating noisy spectra')
            import quapopt.additional_packages.qokit.fur as qk_fur
            simclass_qokit_noisy = qk_fur.choose_simulator(name='gpu')
            simulator_noisy = simclass_qokit_noisy(self.number_of_qubits,
                                                   terms=hamiltonian_noisy)
            noisy_spectrum = simulator_noisy.get_cost_diagonal()

        t2 = time.perf_counter()

        exp_value_noisy = em.cython_vdot(vector1=prob_distro_ideal.astype(np.float32),
                                         vector2=noisy_spectrum.astype(np.float32)) + noisy_constant
        t3 = time.perf_counter()
        energy_result.energy_mean_noisy = exp_value_noisy
        energy_result.update_main_energy(noisy=True)
        t4 = time.perf_counter()
        times = [t1 - t0, t2 - t1, t3 - t2, t4 - t3]
        names = ['abs_squared', 'noisy_hamiltonian', 'dot_product', 'update_energy']
        # print(' '.join([f'{name}: {time:.5f} s' for name, time in zip(names, times)]))

        qaoa_result = QAOAResult(statevector=statevector_ideal,
                                 energy_result=energy_result,
                                 noise_model=measurement_noise,
                                 trial_index=trial_index)

        # TODO(FBM): consider also adding logging of noisy vs noiseless results
        self.log_results(qaoa_result=qaoa_result)

        return qaoa_result
