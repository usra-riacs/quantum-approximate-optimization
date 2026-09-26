# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import gc
import time
from typing import List, Optional, Dict, Any, Tuple, Type

import numpy as np
import warnings

from quapopt import ancillary_functions as anf

from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import ClassicalMeasurementNoiseSampler
from quapopt.data_analysis.data_handling import (verify_whether_to_log_data,
                                                 STANDARD_NAMES_VARIABLES as SNV,
                                                 STANDARD_NAMES_DATA_TYPES as SNDT,
                                                 LoggingLevel
                                                 )
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization import EnergyResultMain
from quapopt.optimization.QAOA import QAOAFunctionInputFormat as FIFormat
from quapopt.optimization.QAOA import QAOAResult, QAOAResultSimplified
from quapopt.optimization.QAOA.QAOARunnerBase import QAOARunnerBase

from quapopt import AVAILABLE_SIMULATORS
from quapopt.optimization.QAOA.circuits.time_block_ansatz import (divide_hamiltonian_into_batches,
                                                                  TimeBlockBatchingType,
                                                                  get_hamiltonian_partition_equivalent_to_time_block_ansatz_with_linear_swap_network)

if 'cuda' in AVAILABLE_SIMULATORS:
    from quapopt.optimization.QAOA.simulation.pauli_backprop.one_layer import (math_functions_cuda as math_p1_cuda)
from quapopt.optimization.QAOA.simulation.pauli_backprop.one_layer.cython_implementation import \
    cython_p1_qaoa as math_p1_cython


class QAOARunnerExpValues(QAOARunnerBase):
    """
    QAOA expectation value computation using for depth-1 QAOA.
    
    This class provides QAOA expectation value calculations without the overhead
    of full quantum circuit simulation. It directly computes expectation values using
    mathematical implementations that can leverage CPU (Cython) or GPU (CUDA).

    - Only p=1 support
    - Option to optimize only over phase separator angle (gamma), while mixer angle (beta) is computed analytically
    - Direct expectation value computation (no wavefunction sampling)
    - Support for both cost and phase Hamiltonian transformations
    - Classical measurement noise simulation

    :param hamiltonian_representations_cost: List of cost Hamiltonians to optimize
    :type hamiltonian_representations_cost: List[ClassicalHamiltonian]
    :param hamiltonian_representations_phase: Optional phase Hamiltonians for the ansatz
    :type hamiltonian_representations_phase: List[ClassicalHamiltonian], optional
    :param store_full_information_in_history: Whether to store complete optimization history
    :type store_full_information_in_history: bool
    :param solve_at_initialization: Whether to solve Hamiltonians during initialization
    :type solve_at_initialization: bool
    :param simulator_name: Override automatic backend selection ('cuda', 'cython', 'numpy')
    :type simulator_name: str, optional
    :param logger_kwargs: Additional keyword arguments for logger configuration
    :type logger_kwargs: Dict[str, Any], optional
    :param logging_level: Verbosity level for result logging
    :type logging_level: LoggingLevel, optional
    :param precision_float: Floating-point precision for computations
    :type precision_float: numpy.dtype
    :param store_n_best_results: Number of best optimization results to retain
    :type store_n_best_results: int
    
    Example:
        from quapopt.hamiltonians import ClassicalHamiltonian
        
        # Create QAOA expectation value runner
        ham_cost = [ClassicalHamiltonian([(1.0, (0, 1))], number_of_qubits=2)]
        runner = QAOARunnerExpValues(
            hamiltonian_representations_cost=ham_cost,
            precision_float=np.float64
        )
        
        # Run QAOA with specific angles
        result = runner.run_qaoa([0.5, 0.3])  # [gamma, beta] for p=1
        print(f"Energy: {result.energy_result.energy_mean}")
    
    .. note::
        This implementation computes expectation values analytically and is deterministic
        (no sampling noise). For realistic quantum simulation including measurement
        statistics, use QAOARunnerQiskit or add classical measurement noise.
    
    .. note::
        Backend selection hierarchy: CUDA (if available) > Cython > NumPy.
        CUDA backend provides significant speedup for large problem instances.

        # TODO(FBM): should add possibility of multiple angles for parts of the Hamiltonian (before mixer), (pseudo Time Block Ansatz)


    """

    def __init__(self,
                 hamiltonian_representations_cost: List[ClassicalHamiltonian],
                 hamiltonian_representations_phase: List[ClassicalHamiltonian] = None,
                 store_full_information_in_history=False,
                 solve_at_initialization=False,
                 simulator_name:str=None,
                 logger_kwargs: Dict[str, Any] = None,
                 logging_level: Optional[LoggingLevel] = None,
                 precision_float=float,
                 store_n_best_results=1,
                 time_block_size: float = 1.0,
                 time_block_seed: Optional[int] = -1,
                 time_block_partitions_list: Optional[List[Dict[int, List[Tuple[int, ...]]]]] = None,
                 time_block_batching_type: TimeBlockBatchingType = TimeBlockBatchingType.FRACTIONAL,
                 simplified_data_storage: bool = False,
                 precision: Optional[str] = None,

                 #dummy variable for backwards compatibility
                # number_of_qubits: int=None
                 ) -> None:
        """
        Initialize the QAOA expectation value runner with optimized simulation backends.
        
        Sets up the computational infrastructure for high-performance QAOA expectation
        value calculations, including automatic backend selection and Hamiltonian
        preprocessing for efficient computation.
        
        :param hamiltonian_representations_cost: List of cost Hamiltonians to optimize
        :type hamiltonian_representations_cost: List[ClassicalHamiltonian]
        :param hamiltonian_representations_phase: Optional phase Hamiltonians for the ansatz
        :type hamiltonian_representations_phase: List[ClassicalHamiltonian], optional
        :param store_full_information_in_history: Whether to store complete optimization history
        :type store_full_information_in_history: bool
        :param solve_at_initialization: Whether to solve Hamiltonians during initialization
        :type solve_at_initialization: bool
        :param simulator_name: Override automatic backend selection ('cuda', 'cython', 'mixed')
        :type simulator_name: str, optional
        :param logger_kwargs: Additional keyword arguments for logger configuration
        :type logger_kwargs: Dict[str, Any], optional
        :param logging_level: Verbosity level for result logging
        :type logging_level: LoggingLevel, optional
        :param precision_float: Floating-point precision for computations
        :type precision_float: numpy.dtype
        :param store_n_best_results: Number of best optimization results to retain
        :type store_n_best_results: int
        :param precision: accepted for the base class; this runner computes at its own `precision_float`.
        :type precision: str, optional
        """

        # self._best_results_container = BestResultsContainerBase()
        self._precision_float = precision_float

        apply_TB = time_block_size is not None

        if isinstance(time_block_size, float):
            apply_TB = time_block_size != 1.0
        elif isinstance(time_block_size, (int, np.int32, np.int64)):
            apply_TB = time_block_size != hamiltonian_representations_cost[0].number_of_qubits


        if apply_TB:
            #TODO(FBM): Think whether this logic makes the most sense. Since we're doing Time-Block,
            # we are applying PS that differs from cost. So if user provides PS,
            # maybe we should just time-block that Hamiltonian? Refactor later
            assert hamiltonian_representations_phase is None, "If TimeBlock is applied, phase Hamiltonians should not be provided separately."
            if time_block_partitions_list is None:
                hamiltonian_representations_phase = []
                for ham_cost in hamiltonian_representations_cost:
                    tb_partition_ham = divide_hamiltonian_into_batches(hamiltonian=ham_cost,
                                                                       time_block_size=time_block_size,
                                                                       batching_type=time_block_batching_type,
                                                                       time_block_seed=time_block_seed,
                                                                       max_depth=1)
                    ham_phase = tb_partition_ham[0]
                    hamiltonian_representations_phase.append(ham_phase)
            else:
                hamiltonian_representations_phase = [tb_partition_ham[0] for tb_partition_ham in time_block_partitions_list]
        elif hamiltonian_representations_phase is None:
            #No TB and no user-provided phase Hamiltonians -> phase separator = cost Hamiltonian.
            #(An explicitly provided phase list is honored; it used to be silently overwritten here.)
            hamiltonian_representations_phase = hamiltonian_representations_cost

        super().__init__(hamiltonian_representations_cost=hamiltonian_representations_cost,
                         hamiltonian_representations_phase=hamiltonian_representations_phase,
                         store_full_information_in_history=store_full_information_in_history,
                         solve_at_initialization=solve_at_initialization,
                         #number_of_qubits=hamiltonian_representations_cost[0].number_of_qubits,
                         logger_kwargs=logger_kwargs,
                         logging_level=logging_level,
                         store_n_best_results=store_n_best_results,
                         simplified_data_storage=simplified_data_storage,
                         precision=precision)

        self._simulator_name = simulator_name
        self._simulators = None

        self._fields_cost_dict:Optional[Dict[int,Dict[str,np.ndarray]]] = None
        self._couplings_cost_dict:Optional[Dict[int,Dict[str,np.ndarray]]] = None

        self._fields_phase_dict:Optional[Dict[int,Dict[str,np.ndarray]]] = None
        self._couplings_phase_dict:Optional[Dict[int,Dict[str,np.ndarray]]] = None

        self._angles_history = {i: {} for i in range(len(self.hamiltonian_representations_phase))}
        self._constant_zeros = (np.zeros(self.number_of_qubits,
                                         dtype=precision_float),
                                np.zeros((self.number_of_qubits, self.number_of_qubits),
                                         dtype=precision_float))

        self._debug = False
        self._ABC_values_history = {i: {} for i in range(len(self.hamiltonian_representations_phase))}



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

    @property
    def computes_in_single_precision(self) -> bool:
        """This runner computes at its own `precision_float`, not at `precision`."""
        return np.dtype(self._precision_float) == np.float32

    def update_hamiltonians_cost(self,
                                 hamiltonian_representations_cost: List[ClassicalHamiltonian],
                                 solve=False):
        self._update_hamiltonians_cost(hamiltonian_representations_cost=hamiltonian_representations_cost,
                                       solve=False)

    def update_hamiltonians_phase(self,
                                   hamiltonian_representations_phase: List[ClassicalHamiltonian],
                                  number_of_qubits:Optional[int]=None):
        self._update_hamiltonians_phase(
            hamiltonian_representations_phase=hamiltonian_representations_phase,
        number_of_qubits=number_of_qubits)

        self._angles_history = {}
        for ind, ham_phase in self._hamiltonian_representations_phase.items():
            self._angles_history[ind] = {}

    def _initialize_simulators_analytical(self,
                                          simulator_name=None,
                                          precision_float=None,
                                          ):
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
            import cupy as cp

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

        if self.simulator_name in ['cuda', 'mixed']:
            from numba.core.errors import NumbaPerformanceWarning
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


    def update_history(self,
                       qaoa_result: QAOAResult|QAOAResultSimplified):

        self._update_history(qaoa_result=qaoa_result)

        if self._simplified_data_storage:
            best_energy = qaoa_result[1][0]
        else:
            best_energy = qaoa_result.energy_mean
        # This simulator does not store bitstrings so we do not pass it additionaly
        tup_to_store = (None, qaoa_result)
        self._best_results_container.add_result(result_to_add=tup_to_store,
                                                score=best_energy)

    def log_results(self,
                     qaoa_result: QAOAResult|QAOAResultSimplified):


        if self.results_logger is None or self.logging_level in [None, LoggingLevel.NONE]:
            return
        if not isinstance(qaoa_result, QAOAResult):
            raise NotImplementedError("IMPLEMENT THIS")
            return



        optimization_overview_df = qaoa_result.to_dataframe_main()



        optimization_overview_dt = SNDT.OptimizationOverview
        self.results_logger.write_results(dataframe=optimization_overview_df,
                                          data_type=optimization_overview_dt)

        correlators_dt = SNDT.Correlators

        if verify_whether_to_log_data(data_type=correlators_dt,
                                      logging_level=self.results_logger._logging_level):
            # # TODO FBM: check what's the fastest way to save this (it's huge for high N)
            self.results_logger.write_results(dataframe=qaoa_result.correlators,
                                              data_type=correlators_dt)

    def get_best_results(self):
        return self._best_results_container.get_best_results()

    def _input_handler_analytical_betas_p1(self,
                                           args,
                                           input_format: Optional[FIFormat]=None,
                                           ):
        # OK, so we have four options for arguments here:
        # 1. angles, hamiltonian_representation_index
        # 2. angles_gamma, angles_beta, hamiltonian_representation_index
        # 3. angle_1, angle_2, ..., angle_2*qaoa_depth, hamiltonian_representation_index
        # 4. optuna.Trial object


        if input_format is None:
            # Arguments are passed as _fun(*args)
            angles = np.array(args[0:1])
            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = args[1]
                assert len(args)==2,"The number of angles must be equal to 1 for analytical betas."
            else:
                hamiltonian_representation_index = 0
                assert len(args)==1,"The number of angles must be equal to 1 for analytical betas."
            trial_index = self._trial_index
            # self._trial_index += 1

            return angles, hamiltonian_representation_index, trial_index

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
            __angles_bounds_layer_MIXER__ = (-np.pi/2, np.pi/2)

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
                 ) -> QAOAResult|QAOAResultSimplified:

        raise NotImplementedError("This method must be implemented in a subclass")
