# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 


from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
#Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
    import numpy as cp

from quapopt import AVAILABLE_SIMULATORS

import pandas as pd

from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import ClassicalMeasurementNoiseSampler
from quapopt.data_analysis.data_handling import LoggingLevel
from quapopt.data_analysis.data_handling import STANDARD_NAMES_DATA_TYPES as SNDT, \
    STANDARD_NAMES_VARIABLES as SNV
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import QAOAFunctionInputFormat as FIFormat, QAOAResult
from quapopt.optimization.QAOA.QAOARunnerBase import QAOARunnerBase
from quapopt.optimization import EnergyResultMain
from quapopt.optimization import HamiltonianSolutionsSampler

class QAOARunnerSampler(QAOARunnerBase, HamiltonianSolutionsSampler):

    """
    Class for running QAOA with samples.
    """

    def __init__(self,
                 hamiltonian_representations_cost: List[ClassicalHamiltonian],
                 hamiltonian_representations_phase: List[ClassicalHamiltonian] = None,
                 store_n_best_results: int = 1,
                 store_full_information_in_history=False,
                 numpy_rng_sampling=None,
                 solve_at_initialization=False,
                 logging_level: Optional[LoggingLevel] = None,
                 logger_kwargs: Optional[Dict[str, Any]] = None,
                 initialize_backend_kwargs: Optional[dict] = None

                 ) -> None:
        """
        Initializes the QAOA runner for sampling solutions.
        :param hamiltonian_representations_cost:
        :param hamiltonian_representations_phase:
        :param store_n_best_results:
        :param store_full_information_in_history:
        :param numpy_rng_sampling:
        :param solve_at_initialization:
        :param logging_level:
        :param logger_kwargs:
        :param initialize_backend_kwargs: kwargs passed to the backend initialization method if not initialized externally
        """


        super().__init__(hamiltonian_representations_cost=hamiltonian_representations_cost,
                         hamiltonian_representations_phase=hamiltonian_representations_phase,
                         store_full_information_in_history=store_full_information_in_history,
                         numpy_rng_sampling=numpy_rng_sampling,
                         solve_at_initialization=solve_at_initialization,
                         logging_level=logging_level,
                         logger_kwargs=logger_kwargs,
                         store_n_best_results=store_n_best_results)

        self._initialize_backend_kwargs = initialize_backend_kwargs
        if self._initialize_backend_kwargs is None:
            self._initialize_backend_kwargs = {}


    def update_history(self,
                       qaoa_result: QAOAResult):

        self._update_history(qaoa_result=qaoa_result)

        if self._store_n_best_results == 1:
            best_bitstrings = [qaoa_result.bitstring_best]
            best_energies = [qaoa_result.energy_best]
        else:
            # If there is more than one, we need to additionaly sort the results
            qaoa_result.sort_energies_and_bitstrings()
            best_bitstrings = [tuple(x) for x in qaoa_result.bitstrings_array[0:self._store_n_best_results]]
            best_energies = qaoa_result.bitstrings_energies[0:self._store_n_best_results]

        for bts, en in zip(best_bitstrings, best_energies):
            tup_to_store = (tuple(bts),
                            qaoa_result.hamiltonian_representation_index,
                            qaoa_result)
            self._best_results_container.add_result(result_to_add=tup_to_store,
                                                    score=en)

    def get_best_results(self) -> List[Tuple[float, Tuple[Tuple[int, ...], int]]]:
        return self._best_results_container.get_best_results()

    def log_results(self,
                     qaoa_result: QAOAResult,
                     additional_annotations: Optional[Dict[str, Any]] = None) -> None:

        if self.results_logger is None or self.logging_level in [None, LoggingLevel.NONE]:
            return


        optimization_overview_df = qaoa_result.to_dataframe_main()
        self.results_logger.write_results(dataframe=optimization_overview_df,
                                          data_type=SNDT.OptimizationOverview,
                                          additional_annotation_dict=additional_annotations)

        if isinstance(qaoa_result.bitstrings_energies, cp.ndarray):
            energies_save = cp.asnumpy(qaoa_result.bitstrings_energies)
        elif isinstance(qaoa_result.bitstrings_energies[0], cp.ndarray):
            energies_save = [float(x) for x in qaoa_result.bitstrings_energies]
        else:
            energies_save = qaoa_result.bitstrings_energies


        energies_main = pd.DataFrame(data={SNV.Energy.id_long: energies_save})
        energies_dt = SNDT.Energies

        self.results_logger.write_results(dataframe=qaoa_result.annotate_dataframe(energies_main),
                                          data_type=energies_dt,
                                          additional_annotation_dict=additional_annotations)

        bitstrings_main = pd.DataFrame(data={SNV.Bitstring.id_long: qaoa_result.bitstrings_array})


        self.results_logger.write_results(dataframe=qaoa_result.annotate_dataframe(bitstrings_main),
                                          data_type=SNDT.Bitstrings,
                                          additional_annotation_dict=additional_annotations)

    def run_qaoa(self,
                 *args,
                 qaoa_depth: int,
                 number_of_samples: int,
                 measurement_noise: ClassicalMeasurementNoiseSampler = None,
                 numpy_rng_sampling=None,
                 input_format: FIFormat = FIFormat.direct_list,
                 return_raw_result=False,
                 backend_name=None) -> Union[QAOAResult, Tuple[QAOAResult, Any]]:

        if number_of_samples == np.inf:
            raise ValueError('Infinite samples are not supported, please use QAOARunnerStatevector class.')

        if self._backends is None:

            _initialize_backend_kwargs = self._initialize_backend_kwargs.copy()


            if backend_name is None:
                backend_name = _initialize_backend_kwargs.get('backend_name', None)
                if backend_name is None:
                    if 'cuda' in AVAILABLE_SIMULATORS:
                        backend_name = 'qokit'
                    else:
                        backend_name = 'qiskit'

            if 'backend_name' in _initialize_backend_kwargs:
                del _initialize_backend_kwargs['backend_name']

            if 'qaoa_depth' in _initialize_backend_kwargs:
                del _initialize_backend_kwargs['qaoa_depth']

            if backend_name == 'qokit':
                self.initialize_backend_qokit(**_initialize_backend_kwargs)
            elif backend_name == 'qiskit':
                self.initialize_backend_qiskit(qaoa_depth=qaoa_depth,
                                               **_initialize_backend_kwargs)
            else:
                raise ValueError('Only qokit and qiskit backends are supported.')
        else:
            if backend_name is not None:
                assert backend_name == self._backend_name, (
                    'Backend name is different than the one used during initialization.'
                    'Please use a new instance of the class if you wish to change backend_computation.')
        if numpy_rng_sampling is None:
            numpy_rng_sampling = self._numpy_rng_sampling


        angles, hamiltonian_representation_index, trial_index = self._input_handler(args=args,
                                                                                    input_format=input_format,
                                                                                    qaoa_depth=qaoa_depth)



        gammas_j = angles[:qaoa_depth]
        betas_j = angles[qaoa_depth:]

        hamiltonian_i = self._hamiltonian_representations_cost[hamiltonian_representation_index]

        backend_i = self._backends[hamiltonian_representation_index]

        if self._backend_name.lower() in ['qokit']:
            _result = backend_i.simulate_qaoa(np.array(gammas_j) * 2,
                                              betas_j)
            statevector_ideal = backend_i.get_statevector(_result)
            bitstrings_array = em.sample_from_statevector(statevector=statevector_ideal,
                                                                number_of_samples=number_of_samples,
                                                                numpy_rng=numpy_rng_sampling,
                                                                sampling_method='auto')


        elif self._backend_name.lower() in ['qiskit']:
            statevector_ideal = None
            _result, bitstrings_array = backend_i.run_qaoa(angles_PHASE=gammas_j.reshape(qaoa_depth),
                                                                 angles_MIXER=betas_j.reshape(qaoa_depth),
                                                                 number_of_samples=number_of_samples)


        else:
            raise NotImplementedError('Only qokit simulator is supported')


        energies_array = hamiltonian_i.evaluate_energy(bitstrings_array=bitstrings_array)


        exp_value_float = np.mean(energies_array)
        #raise KeyboardInterrupt

        qaoa_result = QAOAResult(statevector=statevector_ideal,
                                 bitstrings_array=bitstrings_array,
                                 bitstrings_energies=energies_array,
                                 trial_index=trial_index,
                                 hamiltonian_representation_index=hamiltonian_representation_index,
                                 angles=angles)
        qaoa_result.sort_energies_and_bitstrings()
        qaoa_result.energy_result = EnergyResultMain(energy_mean_noiseless=exp_value_float,
                                                     energy_best_noiseless=qaoa_result.bitstrings_energies[0],
                                                     bitstring_best_noiseless=qaoa_result.bitstrings_array[0],
                                                     )
        qaoa_result.update_main_energy(noisy=False)

        # log results to file

        if measurement_noise is None:
            self.log_results(qaoa_result=qaoa_result)
            if return_raw_result:
                return qaoa_result, _result
            return qaoa_result

        enegy_result_noiseless = qaoa_result.energy_result
        bitstrings_array_noisy = measurement_noise.add_noise_to_samples(ideal_samples=bitstrings_array,
                                                                        rng=numpy_rng_sampling)
        energies_noisy = hamiltonian_i.evaluate_energy(bitstrings_array=bitstrings_array_noisy)
        # energies_noisy = em.calculate_energies_from_bitstrings_2_local(bitstrings_array=bitstrings_array_noisy,
        #                                           adjacency_matrix=hamiltonian_i.get_adjacency_matrix(),
        #                                           computation_backend=hamiltonian_i._default_backend)
        #

        exp_value_noisy = np.mean(energies_noisy)

        qaoa_result = QAOAResult(statevector=statevector_ideal,
                                 bitstrings_array=bitstrings_array_noisy,
                                 bitstrings_energies=energies_noisy,
                                 trial_index=trial_index,
                                 hamiltonian_representation_index=hamiltonian_representation_index,
                                 energy_result=enegy_result_noiseless,
                                 angles=angles)
        qaoa_result.sort_energies_and_bitstrings()
        qaoa_result.energy_result.energy_mean_noisy = exp_value_noisy
        qaoa_result.energy_result.energy_best_noisy = qaoa_result.bitstrings_energies[0]
        qaoa_result.energy_result.bitstring_best_noisy = qaoa_result.bitstrings_array[0]
        qaoa_result.update_main_energy(noisy=True)

        # TODO(FBM): consider also adding logging of noisy vs noiseless results
        self.log_results(qaoa_result=qaoa_result)

        if return_raw_result:
            return qaoa_result, _result

        return qaoa_result



    def _sample_solutions(self,
                          number_of_samples: int,
                          *args,
                          **kwargs
                          )->Tuple[List[Tuple[float, Tuple[Tuple[int, ...], int]]], QAOAResult]:

        qaoa_result = self.run_qaoa(number_of_samples=number_of_samples,
                                    *args,
                                    **kwargs)

        best_n_results = self.get_best_results()

        return best_n_results, qaoa_result

