# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 

import gc
import time
from typing import List, Optional, Dict, Any

import numpy as np
import warnings

from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import ClassicalMeasurementNoiseSampler
from quapopt.data_analysis.data_handling import (verify_whether_to_log_data,
                                                 STANDARD_NAMES_VARIABLES as SNV,
                                                 STANDARD_NAMES_DATA_TYPES as SNDT,
                                                 LoggingLevel
                                                 )
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization import EnergyResultMain
from quapopt.optimization.QAOA import QAOAFunctionInputFormat as FIFormat
from quapopt.optimization.QAOA import QAOAResult
from quapopt.optimization.QAOA.QAOARunnerBase import QAOARunnerBase

from quapopt import AVAILABLE_SIMULATORS

if 'cuda' in AVAILABLE_SIMULATORS:
    from quapopt.optimization.QAOA.simulation.pauli_backprop.one_layer import (math_functions_cuda as math_p1_cuda)
from quapopt.optimization.QAOA.simulation.pauli_backprop.one_layer.cython_implementation import \
    cython_p1_qaoa as math_p1_cython


class QAOARunnerExpValues(QAOARunnerBase):
    """
    Class for running QAOA with expectation values.
    """

    def __init__(self,
                 hamiltonian_representations_cost: List[ClassicalHamiltonian],
                 hamiltonian_representations_phase: List[ClassicalHamiltonian] = None,
                 store_full_information_in_history=False,
                 solve_at_initialization=False,
                 simulator_name=None,
                 logger_kwargs: Dict[str, Any] = None,
                 logging_level: Optional[LoggingLevel] = None,
                 precision_float=np.float64,
                 store_n_best_results=1,
                 #dummy variable for backwards compatibility
                # number_of_qubits: int=None
                 ) -> None:
        """
        Constructor.
        :param input_hamiltonian:
        :param hamiltonian_transformations_to_optimize:
        :param store_full_information_in_history:
        :param ground_state_energy:
        :param numpy_rng_sampling:
        """

        # self._best_results_container = BestResultsContainerBase()
        self._precision_float = precision_float

        super().__init__(hamiltonian_representations_cost=hamiltonian_representations_cost,
                         hamiltonian_representations_phase=hamiltonian_representations_phase,
                         store_full_information_in_history=store_full_information_in_history,
                         solve_at_initialization=solve_at_initialization,
                         #number_of_qubits=hamiltonian_representations_cost[0].number_of_qubits,
                         logger_kwargs=logger_kwargs,
                         logging_level=logging_level,
                         store_n_best_results=store_n_best_results)

        self._simulator_name = None
        self._simulators = None

        self._fields_cost_dict = None
        self._couplings_cost_dict = None

        self._fields_phase_dict = None
        self._couplings_phase_dict = None

        from quapopt import AVAILABLE_SIMULATORS
        if simulator_name in ['mixed', 'cuda']:
            if 'cuda' not in AVAILABLE_SIMULATORS:
                simulator_name = 'cython'
                print("CUDA simulator is not available. Using cython instead.")

        self._initialize_simulators_analytical(simulator_name=simulator_name)
        self._angles_history = {i: {} for i in range(len(self.hamiltonian_representations_phase))}

        self._constant_zeros = (np.zeros(self.number_of_qubits,
                                         dtype=precision_float),
                                np.zeros((self.number_of_qubits, self.number_of_qubits),
                                         dtype=precision_float))

        self._debug = False

        self._ABC_values_history = {i: {} for i in range(len(self.hamiltonian_representations_phase))}

        if self.simulator_name in ['cuda', 'mixed']:
            from numba.core.errors import NumbaPerformanceWarning

            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

    @property
    def fields_cost_dict(self):
        return self._fields_cost_dict

    @property
    def couplings_cost_dict(self):
        return self._couplings_cost_dict

    @property
    def fields_phase_dict(self):
        if self._fields_phase_dict is None:
            return self.fields_cost_dict
        return self._fields_phase_dict

    @property
    def couplings_phase_dict(self):
        if self._couplings_phase_dict is None:
            return self.couplings_cost_dict
        return self._couplings_phase_dict

    @property
    def simulator_name(self):
        return self._simulator_name

    def update_hamiltonians_cost(self,
                                 hamiltonian_representations_cost: List[ClassicalHamiltonian],
                                 solve=False):
        self._update_hamiltonians_cost(hamiltonian_representations_cost=hamiltonian_representations_cost,
                                       solve=False)

    def update_hamiltonians_phase(self,
                                   hamiltonian_representations_phase: List[ClassicalHamiltonian]):
        self._update_hamiltonians_phase(
            hamiltonian_representations_phase=hamiltonian_representations_phase)

        self._angles_history = {}
        for ind, ham_phase in self._hamiltonian_representations_phase.items():
            self._angles_history[ind] = {}

    def _initialize_simulators_analytical(self,
                                          simulator_name=None,
                                          precision_float=None):
        if precision_float is None:
            precision_float = self._precision_float

        #TODO(FBM): do extensive speed tests for different simulators
        #TODO(FBM): think about rewriting those simulators, they are a bit messy now
        if simulator_name is None:
            if self._number_of_qubits <= 50:
                simulator_name = 'cython'
            else:
                simulator_name = 'mixed'

        from quapopt import AVAILABLE_SIMULATORS
        if simulator_name in ['mixed', 'cuda']:
            if 'cuda' not in AVAILABLE_SIMULATORS:
                simulator_name = 'cython'
                print("CUDA simulator is not available. Using cython instead.")

        if simulator_name in ['cuda','mixed']:
            from numba import cuda

        self._simulator_name = simulator_name

        if self._simulator_name in ['mixed', 'cuda']:
            gc.collect()
            cuda.current_context().deallocations.clear()

        self._fields_cost_dict = {}
        self._couplings_cost_dict = {}
        for ind, ham_cost in self.hamiltonian_representations_cost.items():
            fields_array_np, couplings_array_np = ham_cost.get_fields_and_couplings(precision=precision_float)

            self._fields_cost_dict[ind] = {'numpy': fields_array_np}
            self._couplings_cost_dict[ind] = {'numpy': couplings_array_np}

            if self.simulator_name in ['mixed', 'cuda']:
                fields_array_cuda = cuda.to_device(fields_array_np, copy=True)
                couplings_array_cuda = cuda.to_device(couplings_array_np, copy=True)
                # fields_array_cuda = fields_array_np.copy()
                # couplings_array_cuda = couplings_array_np.copy()
                self._fields_cost_dict[ind]['cuda'] = fields_array_cuda
                self._couplings_cost_dict[ind]['cuda'] = couplings_array_cuda

        if self._hamiltonian_representations_phase is not None:
            #raise NotImplementedError("This method is not implemented for phase Hamiltonians yet")
            self._fields_phase_dict = {}
            self._couplings_phase_dict = {}

            for ind, ham_phase in self.hamiltonian_representations_phase.items():
                fields_array_np, couplings_array_np = ham_phase.get_fields_and_couplings(precision=precision_float)
                self._fields_phase_dict[ind] = {'numpy': fields_array_np}
                self._couplings_phase_dict[ind] = {'numpy': couplings_array_np}

                if self.simulator_name in ['mixed', 'cuda']:
                    fields_array_cuda = cuda.to_device(fields_array_np, copy=True)
                    couplings_array_cuda = cuda.to_device(couplings_array_np, copy=True)
                    self._fields_phase_dict[ind]['cuda'] = fields_array_cuda
                    self._couplings_phase_dict[ind]['cuda'] = couplings_array_cuda

    def _get_trigonometric_functions_arrays(self,
                                            hamiltonian_representation_index,
                                            gamma,
                                            fields_phase=None,
                                            correlations_phase=None,
                                            fields_cost=None,
                                            correlations_cost=None
                                            ):

        if fields_phase is None:
            fields_phase = self.fields_phase_dict[hamiltonian_representation_index]
        if correlations_phase is None:
            correlations_phase = self.couplings_phase_dict[hamiltonian_representation_index]
        if fields_cost is None:
            fields_cost = self.fields_cost_dict[hamiltonian_representation_index]
        if correlations_cost is None:
            correlations_cost = self.couplings_cost_dict[hamiltonian_representation_index]

        # TODO FBM: I think just calculating trigonometric functions is much faster with cython
        if self.simulator_name in ['mixed', 'cython']:
            cos_correlations_phase, sin_correlations_phase = math_p1_cython._get_part_1_cython(
                correlations_phase['numpy'],
                gamma)

            cosine_fields = math_p1_cython._get_part_4_cython(fields_phase['numpy'],
                                                              gamma)

            if self.simulator_name in ['mixed']:
                pass


        elif self.simulator_name == 'cuda':
            cos_correlations_phase, sin_correlations_phase = math_p1_cuda._get_part_1_cuda(
                correlations_phase=correlations_phase['cuda'],
                gamma=gamma,
                float_precision=self._precision_float)

            # TODO FBM: I think just calculating trigonometric functions is much faster with cython?
            cosine_fields = math_p1_cuda._get_part_4_cuda(fields_phase=fields_phase['cuda'],
                                                          gamma=gamma,
                                                          float_precision=self._precision_float)

        else:
            raise ValueError("simulator_name must be either 'cuda' or 'cython' or 'mixed'")

        return cos_correlations_phase, sin_correlations_phase, cosine_fields

    def _get_product_formula_array(self,
                                   correlations_cost,
                                   correlations_phase,
                                   cos_correlations_phase,
                                   gamma):

        if self.simulator_name in ['mixed', 'cuda']:
            product_formulas_array = math_p1_cuda._get_part_3_cuda(correlations_cost=correlations_cost['cuda'],
                                                                   correlations_phase=correlations_phase['cuda'],
                                                                   cos_correlations_phase=cos_correlations_phase,
                                                                   gamma=gamma,
                                                                   float_precision=self._precision_float)

            # product_formulas_array = product_formulas_array.copy_to_host()

            #

        elif self.simulator_name == 'cython':
            # print("HEJUNIA")
            # print(correlations_cost['numpy'])
            # print(correlations_phase['numpy'])
            product_formulas_array = math_p1_cython._get_part_3_cython(correlations_cost['numpy'],
                                                                       correlations_phase['numpy'],
                                                                       cos_correlations_phase,
                                                                       gamma)

        else:
            raise ValueError("simulator_name must be either 'cuda' or 'cython' or 'mixed'")

        return product_formulas_array

    def _get_operators_dict(self,
                            hamiltonian_representation_index,
                            gamma,
                            fields_phase=None,
                            fields_cost=None,
                            correlations_phase=None,
                            correlations_cost=None,
                            store_products_in_memory=True,
                            add_ABC=False
                            ):

        if fields_phase is None:
            fields_phase = self.fields_phase_dict[hamiltonian_representation_index]
        if correlations_phase is None:
            correlations_phase = self.couplings_phase_dict[hamiltonian_representation_index]
        if correlations_cost is None:
            correlations_cost = self.couplings_cost_dict[hamiltonian_representation_index]

        t0 = time.perf_counter()

        t1 = time.perf_counter()

        t2 = time.perf_counter()

        if store_products_in_memory:
            if gamma not in self._angles_history[hamiltonian_representation_index]:
                cos_correlations_phase, sin_correlations_phase, cosine_fields = self._get_trigonometric_functions_arrays(
                    hamiltonian_representation_index=hamiltonian_representation_index,
                    gamma=gamma,
                    fields_phase=fields_phase,
                    correlations_phase=correlations_phase)

                product_formulas_array = self._get_product_formula_array(correlations_cost=correlations_cost,
                                                                         correlations_phase=correlations_phase,
                                                                         cos_correlations_phase=cos_correlations_phase,
                                                                         gamma=gamma)

                self._angles_history[hamiltonian_representation_index][gamma] = [cos_correlations_phase,
                                                                                 sin_correlations_phase,
                                                                                 cosine_fields,
                                                                                 product_formulas_array]
            cos_correlations_phase = self._angles_history[hamiltonian_representation_index][gamma][0]
            sin_correlations_phase = self._angles_history[hamiltonian_representation_index][gamma][1]
            cosine_fields = self._angles_history[hamiltonian_representation_index][gamma][2]
            product_formulas_array = self._angles_history[hamiltonian_representation_index][gamma][3]


        else:
            cos_correlations_phase, sin_correlations_phase, cosine_fields = self._get_trigonometric_functions_arrays(
                hamiltonian_representation_index=hamiltonian_representation_index,
                gamma=gamma,
                fields_phase=fields_phase,
                correlations_phase=correlations_phase)

            product_formulas_array = self._get_product_formula_array(correlations_cost=correlations_cost,
                                                                     correlations_phase=correlations_phase,
                                                                     cos_correlations_phase=cos_correlations_phase,
                                                                     gamma=gamma)
        t3 = time.perf_counter()
        if self._debug:
            time_trig = t1 - t0
            time_product = t3 - t2
            anf.cool_print("SIMULATOR:", self.simulator_name, 'green')
            anf.cool_print("TIME FOR TRIGONOMETRIC FUNCTIONS:", time_trig, 'blue')
            anf.cool_print("TIME FOR PRODUCT FORMULAS:", time_product, 'blue')
            anf.cool_print("RATIO TIME TRIG/PRODUCT:", time_trig / time_product, 'blue')

            # check how much RAM memory they take:
            memory_cos = cos_correlations_phase.nbytes
            memory_sin = sin_correlations_phase.nbytes
            memory_cos1q = cosine_fields.nbytes
            total_memory_trig = memory_cos + memory_sin + memory_cos1q
            memory_product = product_formulas_array.nbytes
            # anf.cool_print("MEMORY FOR TRIGONOMETRIC FUNCTIONS:", total_memory_trig, 'blue')
            # anf.cool_print("MEMORY FOR PRODUCT FORMULAS:", memory_product, 'blue')
            anf.cool_print("RATIO MEMORY TRIG/PRODUCT:", total_memory_trig / memory_product, 'blue')

        dict_here = {'cos_correlations_phase': cos_correlations_phase,
                     'sin_correlations_phase': sin_correlations_phase,
                     'product_formulas_array': product_formulas_array,
                     'cosine_fields': cosine_fields}

        if add_ABC:
            if correlations_cost is None:
                correlations_cost = self.couplings_cost_dict[hamiltonian_representation_index]

            if 1 in self.hamiltonian_representations[0].localities:
                if store_products_in_memory:
                    _evaluate_sines = False
                    if gamma not in self._angles_history[hamiltonian_representation_index]:
                        _evaluate_sines = True
                    elif len(self._angles_history[hamiltonian_representation_index][gamma]) < 5:
                        _evaluate_sines = True

                    if _evaluate_sines:
                        if self.simulator_name in ['cython', 'mixed']:
                            sine_fields_phase = math_p1_cython._get_sine_fields_cython(fields_phase['numpy'],
                                                                                        gamma)
                        else:
                            raise NotImplementedError("This method is not implemented for local fields yet")
                        self._angles_history[hamiltonian_representation_index][gamma].append(sine_fields_phase)

                    sine_fields_phase = self._angles_history[hamiltonian_representation_index][gamma][4]
                else:
                    if self.simulator_name in ['cython', 'mixed']:
                        sine_fields_phase = math_p1_cython._get_sine_fields_cython(fields_phase['numpy'],
                                                                                    gamma)
                    else:
                        sine_fields_phase = math_p1_cuda._get_sine_fields_cuda(fields_phase=fields_phase['cuda'],
                                                                                gamma=gamma,
                                                                                float_precision=self._precision_float)

                dict_here['sine_fields_phase'] = sine_fields_phase

                if fields_cost is None:
                    fields_cost = self.fields_cost_dict[hamiltonian_representation_index]

                fake_beta = np.pi / 4
                fake_sin = 1.0

                if self.simulator_name in ['cuda', 'mixed']:
                    Ci = math_p1_cuda._get_part_2_cuda(fields_phase=fields_phase['cuda'],
                                                       fields_cost=fields_cost['cuda'],
                                                       correlations_phase=correlations_phase['cuda'],
                                                       cos_correlations_phase=cos_correlations_phase,
                                                       gamma=gamma,
                                                       beta=fake_beta,
                                                       float_precision=self._precision_float,
                                                       return_only_sum=True)



                elif self.simulator_name == 'cython':
                    Ci = math_p1_cython._get_part_2_cython(fields_phase['numpy'],
                                                           fields_cost['numpy'],
                                                           correlations_phase['numpy'],
                                                           cos_correlations_phase,
                                                           gamma,
                                                           fake_beta,
                                                           True)

                else:
                    raise ValueError("simulator_name must be either 'cuda' or 'cython' or 'mixed' ")
                A_value = self._precision_float(Ci)

                dict_here['A_value'] = A_value

            t0 = time.perf_counter()

            if self.simulator_name in ['cuda', 'mixed']:
                BC_values = math_p1_cuda._get_BC_cuda(fields_phase=fields_phase['cuda'],
                                                      correlations_cost=correlations_cost['cuda'],
                                                      sin_correlations_phase=sin_correlations_phase,
                                                      cosine_fields=cosine_fields,
                                                      product_formulas_array=product_formulas_array,
                                                      gamma=gamma,
                                                      float_precision=self._precision_float)


            elif self.simulator_name in ['cython']:
                BC_values = math_p1_cython._get_BC_cython(fields_phase['numpy'],
                                                          correlations_cost['numpy'],
                                                          sin_correlations_phase,
                                                          cosine_fields,
                                                          product_formulas_array,
                                                          gamma)

            dict_here['BC_values'] = BC_values
            t1 = time.perf_counter()

            if self._debug:
                print("_______")
                anf.cool_print("TIME FOR BC VALUES:", t1 - t0, 'red')

        return dict_here

    def _get_expected_values(self,
                             gamma,
                             beta,
                             fields_phase,
                             fields_cost,
                             correlations_phase,
                             correlations_cost,
                             operators_dict: dict,
                             simulator_name: str = None,
                             return_only_sum=False):
        if simulator_name is None:
            simulator_name = self.simulator_name

        cos_correlations_phase = operators_dict['cos_correlations_phase']
        sin_correlations_phase = operators_dict['sin_correlations_phase']
        product_formulas_array = operators_dict['product_formulas_array']
        cosine_fields = operators_dict['cosine_fields']

        if simulator_name in ['cuda', 'mixed']:
            Ci = math_p1_cuda._get_part_2_cuda(fields_phase=fields_phase['cuda'],
                                               fields_cost=fields_cost['cuda'],
                                               correlations_phase=correlations_phase['cuda'],
                                               cos_correlations_phase=cos_correlations_phase,
                                               gamma=gamma,
                                               beta=beta,
                                               float_precision=self._precision_float,
                                               return_only_sum=return_only_sum)

            Cij = math_p1_cuda._get_part_5_cuda(fields_phase=fields_phase['cuda'],
                                                correlations_cost=correlations_cost['cuda'],
                                                sin_correlations_phase=sin_correlations_phase,
                                                cosine_fields=cosine_fields,
                                                product_formulas_array=product_formulas_array,
                                                gamma=gamma,
                                                beta=beta,
                                                float_precision=self._precision_float,
                                                return_only_sum=return_only_sum)




        elif simulator_name == 'cython':
            Ci = math_p1_cython._get_part_2_cython(fields_phase['numpy'],
                                                   fields_cost['numpy'],
                                                   correlations_phase['numpy'],
                                                   cos_correlations_phase,
                                                   gamma,
                                                   beta,
                                                   return_only_sum)

            Cij = math_p1_cython._get_part_5_cython(fields_phase['numpy'],
                                                    correlations_cost['numpy'],
                                                    sin_correlations_phase,
                                                    cosine_fields,
                                                    product_formulas_array,
                                                    gamma,
                                                    beta,
                                                    return_only_sum)
        else:
            raise ValueError("simulator_name must be either 'cuda' or 'cython' or 'mixed' ")

        return Ci, Cij

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
        if self.results_logger is None or self.logging_level in [None, LoggingLevel.NONE]:
            return


        optimization_overview_df = qaoa_result.to_dataframe_main()
        optimization_overview_dt = SNDT.OptimizationOverview
        self.results_logger.write_results(dataframe=optimization_overview_df,
                                          data_type=optimization_overview_dt)

        correlators_dt = SNDT.Correlators

        if verify_whether_to_log_data(data_type=correlators_dt,
                                      logging_level=self.results_logger._logging_level):
            # # TODO FBM: check what's the fastest way to save this (it's huge for high N)
            # qubit_indices_physical, values = [], []
            # for i in range(self.number_of_qubits):
            #     for j in range(i, self.number_of_qubits):
            #         val =   qaoa_result.correlators[i, j]
            #         if val == 0:
            #             continue
            #         if i==j:
            #             qub_tup = (i,)
            #         else:
            #             qub_tup = (i, j)
            #         qubit_indices_physical.append(qub_tup)
            #         values.append(val)
            # correlators_df = pd.DataFrame(data={SNV.QubitIndices.id_long: qubit_indices_physical,
            #                                     SNV.EnergyMean.id_long: values})
            self.results_logger.write_results(dataframe=qaoa_result.correlators,
                                              data_type=correlators_dt)

    def get_best_results(self):
        return self._best_results_container.get_best_results()

    def _input_handler_analytical_betas_p1(self,
                                           args,
                                           input_format: FIFormat,
                                           ):
        # OK, so we have four options for arguments here:
        # 1. angles, hamiltonian_representation_index
        # 2. angles_gamma, angles_beta, hamiltonian_representation_index
        # 3. angle_1, angle_2, ..., angle_2*qaoa_depth, hamiltonian_representation_index
        # 4. optuna.Trial object
        trial_index = None
        if input_format in [FIFormat.direct_full]:
            # Arguments are passed as _fun(*args)
            angles = np.array(args[0:1])
            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = args[1]
                assert len(args)==2,"The number of angles must be equal to 1 for analytical betas."
            else:
                hamiltonian_representation_index = 0
                assert len(args)==1,"The number of angles must be equal to 1 for analytical betas."

        elif input_format in [FIFormat.direct_list]:
            # Arguments are passed as _fun(list_of_args)
            angles = np.array(args[0])
            assert len(angles) == 1, 'The number of angles must be equal to 1 for analytical betas'
            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = args[1]
            else:
                hamiltonian_representation_index = 0

        elif input_format in [FIFormat.direct_vector]:
            # Arguments are passed as _fun(vector_of_angles, hamiltonian_representation_index)
            angles = np.array(args[0])
            assert len(angles) == 1, 'The number of angles must be equal to 1 for analytical betas'

            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = args[1]
            else:
                hamiltonian_representation_index = 0

        elif input_format in [FIFormat.direct_QAOA]:
            # Arguments are passed as _fun([vector_gamma, vector_beta], hamiltonian_representation_index)
            angles_gamma = args[0][0]

            assert len(angles_gamma) == 1, 'The number of angles_gamma must be equal to 1 for analytical betas'

            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = args[1]
            else:
                hamiltonian_representation_index = 0

            angles = np.array([ai for ai in angles_gamma])

        elif input_format in [FIFormat.optuna]:
            # Arguments are passed as _fun(optuna.Trial)

            trial = args[0]
            trial_index = trial._trial_id

            # TODO(FBM): Make this more flexible
            __ANGLES_BOUNDS_LAYER_PHASE__ = (-np.pi, np.pi)
            __angles_bounds_layer_MIXER__ = (-np.pi, np.pi)

            angles_bounds = [__ANGLES_BOUNDS_LAYER_PHASE__ for _ in range(1)]

            bounds_optuna_angles = [(f"{SNV.Angles.id}-{index}", bound[0], bound[1])
                                    for index, bound in enumerate(angles_bounds[0:len(angles_bounds)])]
            bounds_optuna_transformations = [
                (SNV.HamiltonianRepresentationIndex.id, tuple(range(len(self.hamiltonian_representations))))]

            angles = np.array([trial.suggest_float(*xxx) for xxx in bounds_optuna_angles])
            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = trial.suggest_categorical(*bounds_optuna_transformations)
            else:
                hamiltonian_representation_index = 0

        else:
            raise ValueError('input_format must be either "simulation" or "optuna"')

        if trial_index is None:
            trial_index = self._trial_index
            # self._trial_index += 1

        return angles, hamiltonian_representation_index, trial_index

    def run_qaoa(self,
                 *args,
                 # qaoa_depth: int,
                 measurement_noise: ClassicalMeasurementNoiseSampler = None,
                 store_correlators=False,
                 input_format: FIFormat = FIFormat.direct_list,
                 memory_intensive=True,
                 debug=False,
                 qaoa_depth=1,
                 number_of_samples=None,
                 numpy_rng_sampling=None,
                 analytical_betas=False,
                 trial_index_offset=0,
                 operators_dict=None,
                 # debug_array=None
                 ) -> QAOAResult:

        assert qaoa_depth == 1, "This method is only implemented for QAOA depth 1"
        assert number_of_samples is None or number_of_samples == np.inf, "This method is only implemented for number_of_samples=None or np.inf"

        t0_betas = time.perf_counter()
        if analytical_betas:
            angles, hamiltonian_representation_index, trial_index = self._input_handler_analytical_betas_p1(args=args,
                                                                                                            input_format=input_format)

            assert len(angles) == 1, "The number of angles must be 1"
            beta_j = None

        else:
            angles, hamiltonian_representation_index, trial_index = self._input_handler(args=args,
                                                                                        input_format=input_format,
                                                                                        qaoa_depth=1, )
            #print(angles)
            assert len(angles) == 2, "The number of angles must be 2"
            beta_j = angles[1:][0]
            beta_j = self._precision_float(beta_j)

        gamma_j = angles[0:1][0]
        gamma_j = self._precision_float(gamma_j)
        t1_betas = time.perf_counter()

        trial_index += trial_index_offset
        # fields_phase, correlations_phase = self.hamiltonian_arrays_phase[hamiltonian_representation_index]

        # print(self.fields_cost_dict)
        fields_phase = self.fields_phase_dict[hamiltonian_representation_index]
        correlations_phase = self.couplings_phase_dict[hamiltonian_representation_index]

        if analytical_betas:
            memory_intensive = False
            # Memory intensitve implementation makes use of the fact that most computations are done for gammas and
            # they can be used for multiple betas,
            # for analytical betas, we just use computations for each gamma once

        # if self.simulator_name in ['cuda','mixed']:
        #     cuda.synchronize()

        if operators_dict is None:
            operators_dict = self._get_operators_dict(
                hamiltonian_representation_index=hamiltonian_representation_index,
                gamma=gamma_j,
                fields_phase=fields_phase,
                correlations_phase=correlations_phase,
                store_products_in_memory=memory_intensive,
                add_ABC=analytical_betas)

        t2_betas = time.perf_counter()

        # print(args)

        if analytical_betas:
            if 1 in self.hamiltonian_representations[0].localities:
                A = operators_dict['A_value']
                B, C = operators_dict['BC_values']

                self._ABC_values_history[hamiltonian_representation_index][trial_index] = np.array([A, B, C])

                # This is based on paper: https://arxiv.org/abs/2501.16419
                # in which the authors noticed that it's easy to solve for optimal beta, having access to all those
                # numbers we already calculated
                #TODO(FBM): add proper citations

                a = A

                #In the paper, they incorporate 1/2 inside B and C. I didn't.
                b = B/2
                c = C/2

                term_4 = 16 * b ** 2 + 4 * c ** 2
                term_2 = a ** 2 - term_4
                term_3 = 8 * a * b
                term_1 = -term_3 / 2
                term_0 = 4 * b ** 2

                coeffs_poly = [term_4, term_3, term_2, term_1, term_0]
                roots = np.roots(coeffs_poly)
                roots = np.array([r.real for r in roots if abs(r.imag) <= 10 ** (-9)
                                  and (abs(r.real) <= 1 or abs(r.real - 1) <= 10 ** (-9))])
                def __edge_cases(x):
                    if x>1:
                        if abs(1-x)<=10**(9):
                            return 1
                        else:
                            raise ValueError("x is not in the range [-1,1]")
                    elif x<-1:
                        if abs(-1-x)<=10**(9):
                            return -1
                        else:
                            raise ValueError("x is not in the range [-1,1]")
                    else:
                        return x
                roots = np.array([__edge_cases(r) for r in roots])



                # #TODO(FBM): what precision level is important?
                #print(roots)
                roots_arccos = 1 / 2 * np.arccos(roots)
                new_betas_list = [root for root in roots_arccos] + [-root for root in roots_arccos]
                new_betas_list = sorted(list(set(new_betas_list)))

                # TODO(FBM): checking different parametrization for solution. Verify that functions are unambiguous
                term_4_alt = 1 * b ** 2
                term_3_alt = -2 * b * c
                term_2_alt = -1 * a ** 2 - 2 * b ** 2 + 1 * c ** 2
                term_1_alt = 2 * b * c
                term_0_alt = 1 * b ** 2 - 1 * a ** 2

                coeffs_poly_alt = [term_4_alt, term_3_alt, term_2_alt, term_1_alt, term_0_alt]
                coeffs_poly_alt = np.round(coeffs_poly_alt, 20)
                #print(coeffs_poly_alt)
                roots_alt = np.roots(coeffs_poly_alt)
                roots_alt = np.array([r for r in roots_alt if abs(r.imag) <= 10 ** (-9)])
                roots_alt = np.round(roots_alt.real, 10)
                roots_arctan = 1 / 2 * np.arctan(roots_alt)
                new_betas_list_alt = [root for root in roots_arctan] + [-root for root in roots_arctan]

                results_new_betas = []

                _original_logging_level = self.logging_level
                # TODO(FBM): maybe we want to log the intermediate results?
                self.set_logging_level(logging_level=LoggingLevel.NONE)

                for new_beta in new_betas_list:
                    if abs(new_beta) <= 10 ** (-40):
                        continue

                    new_args = (gamma_j,
                                new_beta,
                                hamiltonian_representation_index)

                    #TODO(FBM): since expected value is simple function of, A,B,C and beta, we probably should just calculate it here

                    # qaoa_result = self.run_qaoa(*new_args,
                    #                             measurement_noise=measurement_noise,
                    #                             store_correlators=False,
                    #                             input_format=FIFormat.direct_full,
                    #                             memory_intensive=memory_intensive,
                    #                             debug=debug,
                    #                             qaoa_depth=1,
                    #                             number_of_samples=None,
                    #                             numpy_rng_sampling=None,
                    #                             analytical_betas=False,
                    #                             trial_index_offset=trial_index_offset,
                    #                             operators_dict=operators_dict
                    #                             )
                    # test_qaoa = qaoa_result.energy_mean


                    sin_2_beta = np.sin(2 * new_beta)
                    Ci_analytical = sin_2_beta*a

                    sin_4_beta = np.sin(4 * new_beta)
                    sin_2_beta_squared = sin_2_beta**2
                    Cij_analytical = sin_4_beta*b+sin_2_beta_squared*c


                    energy_analytical =  Ci_analytical + Cij_analytical

                    # if abs(energy_analytical - test_qaoa) > 10**(-9):
                    #     raise KeyboardInterrupt("Energy analytical and energy from QAOA are not the same")

                    results_new_betas.append(energy_analytical)

                if len(results_new_betas) == 0:
                    best_beta = 0.0
                else:
                    best_energy_index_beta = np.argmin(results_new_betas)
                    best_beta = new_betas_list[best_energy_index_beta]

                # TODO(FBM): work out the indexing of runs for optimization history in this case
                final_args = (gamma_j,
                              best_beta,
                              hamiltonian_representation_index)

                self.set_logging_level(logging_level=_original_logging_level)

                return self.run_qaoa(*final_args,
                                     measurement_noise=measurement_noise,
                                     store_correlators=store_correlators,
                                     input_format=FIFormat.direct_full,
                                     memory_intensive=memory_intensive,
                                     debug=debug,
                                     qaoa_depth=1,
                                     number_of_samples=None,
                                     numpy_rng_sampling=None,
                                     analytical_betas=False,
                                     trial_index_offset=trial_index_offset,
                                     operators_dict=operators_dict
                                     )

            else:
                BC_values = operators_dict['BC_values']

                self._ABC_values_history[hamiltonian_representation_index][trial_index] = BC_values

                B, C = BC_values[0], BC_values[1]

                # a = B / 2
                # b = -C / 2

                a = B
                b = C

                beta_main = 1 / 4 * np.arctan2(2 * a, b) + np.pi / 4
                new_args = (gamma_j,
                            beta_main,
                            hamiltonian_representation_index)

                return self.run_qaoa(*new_args,
                                     measurement_noise=measurement_noise,
                                     store_correlators=store_correlators,
                                     input_format=FIFormat.direct_full,
                                     memory_intensive=memory_intensive,
                                     debug=debug,
                                     qaoa_depth=1,
                                     number_of_samples=None,
                                     numpy_rng_sampling=None,
                                     analytical_betas=False,
                                     trial_index_offset=trial_index_offset,
                                     # operators_dict=operators_dict
                                     operators_dict=operators_dict,
                                     )

        if measurement_noise is None:
            fields_cost = self.fields_cost_dict[hamiltonian_representation_index]
            correlations_cost = self.couplings_cost_dict[hamiltonian_representation_index]

            Ci_noiseless, Cij_noiseless = self._get_expected_values(gamma=gamma_j,
                                                                    beta=beta_j,
                                                                    fields_phase=fields_phase,
                                                                    fields_cost=fields_cost,
                                                                    correlations_phase=correlations_phase,
                                                                    correlations_cost=correlations_cost,
                                                                    operators_dict=operators_dict,
                                                                    simulator_name=self.simulator_name,
                                                                    return_only_sum=not store_correlators
                                                                    )

            if not store_correlators:
                exp_value_noiseless = Ci_noiseless + Cij_noiseless
                Cij_noiseless = None

            else:
                exp_value_noiseless = np.sum(Ci_noiseless) + np.sum(np.sum(Cij_noiseless))

                # set the diagonal of Cij_noiseless to Ci_noiseless
                Cij_noiseless[np.diag_indices_from(Cij_noiseless)] = Ci_noiseless

            energy_result = EnergyResultMain(energy_mean_noiseless=exp_value_noiseless)
            energy_result.update_main_energy(noisy=False)

            qaoa_result = QAOAResult(energy_result=energy_result,
                                     trial_index=trial_index,
                                     hamiltonian_representation_index=hamiltonian_representation_index,
                                     angles=np.array(angles),
                                     correlators=Cij_noiseless)

            self.log_results(qaoa_result=qaoa_result)

            return qaoa_result


        # TODO(FBM): this should to be refactored for measurement noise
        if measurement_noise.noisy_hamiltonian_representations['cost'] is None:
            # The cost Hamiltonian is changed due to the measurement noise, there's a built-in method for that
            measurement_noise.add_noisy_hamiltonian_representations(
                hamiltonian_representations_dict=self.hamiltonian_representations,
                hamiltonian_identifier='cost')



        t2 = time.perf_counter()

        # # The cost Hamiltonian is changed due to the measurement noise
        fields_cost = measurement_noise.noisy_hamiltonian_fields['cost'][hamiltonian_representation_index]
        correlations_cost = measurement_noise.noisy_hamiltonian_couplings['cost'][hamiltonian_representation_index]



        # # The constant term is changed added due to the measurement noise
        _, noisy_constant = measurement_noise.noisy_hamiltonian_representations['cost'][
            hamiltonian_representation_index]


        Ci_noisy, Cij_noisy = self._get_expected_values(gamma=gamma_j,
                                                        beta=beta_j,
                                                        fields_phase=fields_phase,
                                                        fields_cost=fields_cost,
                                                        correlations_phase=correlations_phase,
                                                        correlations_cost=correlations_cost,
                                                        operators_dict=operators_dict,
                                                        simulator_name=self.simulator_name)

        exp_value_noisy = np.sum(Ci_noisy) + np.sum(np.sum(Cij_noisy))
        #print('1',exp_value_noisy)
        exp_value_noisy+= noisy_constant
        #print('2',exp_value_noisy)


        t3 = time.perf_counter()


        energy_result = EnergyResultMain(energy_mean_noisy=exp_value_noisy)
        energy_result.update_main_energy(noisy=True)

        Cij_noisy[np.diag_indices_from(Cij_noisy)] = Ci_noisy

        qaoa_result = QAOAResult(energy_result=energy_result,
                                 trial_index=trial_index,
                                 hamiltonian_representation_index=hamiltonian_representation_index,
                                 noise_model=measurement_noise,
                                 angles=np.array(angles),
                                 correlators=Cij_noisy)
        self.log_results(qaoa_result=qaoa_result)

        return qaoa_result
