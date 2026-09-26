# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import time
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np

try:
    import cupy as cp
except(ModuleNotFoundError,ImportError):
    import numpy as cp
from quapopt import AVAILABLE_SIMULATORS



from quapopt import AVAILABLE_SIMULATORS
from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import ClassicalMeasurementNoiseSampler
from quapopt.data_analysis.data_handling import (LoggingLevel
                                                 )
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization import EnergyResultMain
from quapopt.optimization.QAOA import QAOAFunctionInputFormat as FIFormat
from quapopt.optimization.QAOA import QAOAResult, QAOAResultSimplified
from quapopt.optimization.QAOA.circuits.time_block_ansatz import (TimeBlockBatchingType)


from quapopt.optimization.QAOA.simulation.qaoa_math import (get_WS_mixer_operator_1q,
                                                            get_initial_state_WS_qaoa_1q,
                                                            p1_beta_fourier_coefficients,
                                                            minimize_p1_beta_trigonometric_polynomial)
from quapopt.optimization.QAOA.simulation.QAOARunnerExpValues import QAOARunnerExpValues

from quapopt.optimization.QAOA.simulation.reduced_density_matrices.one_layer.cython_implementation.cython_p1_rdms_qaoa import (get_precomputed_phase_data_cython,
                                                                                                                               get_all_rho_ij_cython,
                                                                                                                               get_all_ZiZj_cython)


from quapopt.optimization.QAOA.simulation.reduced_density_matrices.one_layer.math_functions_cuda import (get_all_rho_ij_cuda,
                                                                                                         get_all_ZiZj_cuda,
                                                                                                         compute_couplings_sums_cuda,
                                                                                                         get_pauli_overlaps_cupy,
                                                                                                         get_pauli_overlaps_numpy)
class QAOARunnerExpValuesRDMs(QAOARunnerExpValues):
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

    """

    def __init__(self,
                 hamiltonian_representations_cost: List[ClassicalHamiltonian],
                 hamiltonian_representations_phase: List[ClassicalHamiltonian] = None,
                 store_full_information_in_history=False,
                 solve_at_initialization=False,
                 simulator_name=None,
                 logger_kwargs: Dict[str, Any] = None,
                 logging_level: Optional[LoggingLevel] = None,
                 precision_float=np.float32,
                 store_n_best_results=1,
                 time_block_size: float = 1.0,
                 time_block_seed: Optional[int] = -1,
                 time_block_partitions_list: Optional[List[Dict[int, List[Tuple[int, ...]]]]] = None,
                 time_block_batching_type: TimeBlockBatchingType = TimeBlockBatchingType.FRACTIONAL,
                 ws_bias_parameters: Optional[List[float]|float] = None,
                 simplified_data_storage:Optional[bool]=None,
                 mixer_opposite_to_input_state:bool=False,
                 precision: Optional[str] = None
                 # dummy variable for backwards compatibility
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
        :param precision_float: Floating-point precision for computations; np.float64 raises
            NotImplementedError on the 'cython' simulator (the default up to 30 qubits)
        :type precision_float: numpy.dtype
        :param store_n_best_results: Number of best optimization results to retain
        :type store_n_best_results: int
        :param precision: accepted for the base class; not acted on.
        :type precision: str, optional
        """

        # if 1 in hamiltonian_representations_cost[0].localities:
        #     raise NotImplementedError("This class is not implemented for Hamiltonians with local fields yet")


        if simulator_name in [None, 'auto']:
            simulator_name = 'cython'

            from quapopt import AVAILABLE_SIMULATORS

            if hamiltonian_representations_cost[0].number_of_qubits > 30 and 'cuda' in AVAILABLE_SIMULATORS:
                simulator_name = 'cuda'

        if simplified_data_storage is None:
            simplified_data_storage = logging_level in [None, LoggingLevel.NONE]



        super().__init__(hamiltonian_representations_cost=hamiltonian_representations_cost,
                         hamiltonian_representations_phase=hamiltonian_representations_phase,
                         store_full_information_in_history=store_full_information_in_history,
                         solve_at_initialization=solve_at_initialization,
                         # number_of_qubits=hamiltonian_representations_cost[0].number_of_qubits,
                         logger_kwargs=logger_kwargs,
                         logging_level=logging_level,
                         store_n_best_results=store_n_best_results,
                         time_block_size=time_block_size,
                         time_block_seed=time_block_seed,
                         time_block_partitions_list=time_block_partitions_list,
                         time_block_batching_type=time_block_batching_type,
                         precision_float=precision_float,
                         simulator_name=simulator_name,
                         simplified_data_storage=simplified_data_storage,
                         precision=precision

                         )

        # The base class may fall back to cython, so the check reads the simulator it resolved.
        if self.simulator_name == 'cython' and np.dtype(precision_float) == np.float64:
            raise NotImplementedError(
                "QAOARunnerExpValuesRDMs: precision_float=np.float64 is not implemented on the 'cython' "
                "simulator: its kernels take float32 buffers, and the first expectation value fails with "
                "a buffer dtype mismatch. Use precision_float=np.float32, or simulator_name='numpy' or "
                "'cuda', which run at float64 (the states and density matrices stay complex64 on every "
                "simulator).")

        _same_bias = False

        if ws_bias_parameters is None:
            ws_bias_parameters = [0.5] * self.number_of_qubits
            _same_bias = True

        elif isinstance(ws_bias_parameters, (int, float)):
            ws_bias_parameters = [ws_bias_parameters] * self.number_of_qubits
            _same_bias = True
        elif isinstance(ws_bias_parameters, (list, tuple, np.ndarray)):
            assert len(ws_bias_parameters) == self.number_of_qubits, "Number of bias parameters must match number of qubits."

            _same_bias = all(c == ws_bias_parameters[0] for c in ws_bias_parameters)


        else:
            raise NotImplementedError("Non-identical biases are not yet implemented")

        ws_bias_parameters = np.array(ws_bias_parameters)

        assert ws_bias_parameters.shape[0] == self.number_of_qubits, "Number of bias parameters must match number of qubits."


        X_terms = 2 * np.sqrt(ws_bias_parameters * (1 - ws_bias_parameters))

        if not mixer_opposite_to_input_state:
            Z_terms = 1 - 2 * ws_bias_parameters
            _c_1q = ws_bias_parameters[0]

        else:
            #Mixer is opposite to the initial state
            Z_terms = 1 - 2 * (1-ws_bias_parameters)
            _c_1q = 1-ws_bias_parameters[0]

        self._XZ_terms = [(x, z) for x, z in zip(X_terms, Z_terms)]

        if self.simulator_name in ['cuda']:
            from numba import cuda
            ws_bias_parameters = cuda.to_device(np.ascontiguousarray(ws_bias_parameters, dtype=np.float32), copy=True)
        self._ws_bias_parameters = ws_bias_parameters


        self._mixers_dict = {}
        self._initial_states_array = None


        #we use convention:
        #RMDS_dict = {(hamiltonian_representation_index, gamma): local_RDMS}.
        #local_RDMS has shape (number_of_qubits, number_of_qubits, 4, 4),
        # and rho_ij = local_RDMS[i, j, :, :] when i!=j.
        # or rho_i = local_RDMS[i, i, 0:2, 0:2] when i==j.

        self._RDMS_dict = {}
        #Keys of _RDMS_dict whose array was built with no rdm_mask, so every entry is filled.
        #A caller that needs every entry accepts only these; a masked caller accepts any entry that covers its mask.
        self._RDMS_full = set()
        #For every other key of _RDMS_dict: the (normalized) rdm_mask its array was built with.
        self._RDMS_masks = {}
        #Per representation index: the cost-support mask the expected-value paths hand the builders
        #(host bool; a device uint8 view on cuda). Lazily built by _get_cost_rdm_mask.
        self._cost_rdm_masks = {}
        self._couplings_sums_dict = {}
        #All-ones (couplings, fields) arrays in the active backend's layout; lazily built by
        #_get_all_ones_cost_masks to bypass the cost-coefficient skip-optimization in the ZiZj paths.
        self._all_ones_cost_masks = None

        self._initialize_simulators_analytical(simulator_name=simulator_name, )

        cx = 4 * (1 - 2 * _c_1q) * np.sqrt((1 - _c_1q) * _c_1q)
        # cy = 2*Sqrt[(1 - c)*c];
        cy = 2 * np.sqrt((1 - _c_1q) * _c_1q)
        # cz1 =  4 (1 - c)*c;
        cz1 = 4 * (1 - _c_1q) * _c_1q
        # cz0 = (1 - 2 c)^2;
        cz0 = (1 - 2 * _c_1q) ** 2


        # cx = 4*(1-2*ws_bias_parameters)*np.


        #The transformation under WS mixer is:
        # Z --> cx * sin(beta)^2 * X + cy * sin(2*beta) * Y + (cz1*Cos(2*beta)+cz0)*Z

        self._c_coeffs = [cx, cy, cz1, cz0]
        #for c = 0.5,
        # [0., 1.0, 1.0, 0.0]

        #for c = 0.25:
        #[sqrt(3/4), sqrt(3/4), 0.75, 0.25]

        self._same_bias = _same_bias
        self._overlaps_dict = {'cupy':{},
                               'numpy':{}}






    @property
    def computes_in_single_precision(self) -> bool:
        """The initial states, mixers and density matrices of this runner are complex64
        whatever `precision_float` is."""
        return True

    def _set_initial_state(self):
        ws_bias_parameters = self._ws_bias_parameters

        initial_states_array = np.array([get_initial_state_WS_qaoa_1q(bias_parameter_WS=c,
                                                                      backend='numpy') for c in ws_bias_parameters],
                                        dtype=np.complex64)

        # print(initial_states_array[0])

        if self.simulator_name in ['cuda']:
            from numba import cuda

            initial_states_array = [initial_states_array.real, initial_states_array.imag]

            initial_states_array[0], initial_states_array[1] = (cuda.to_device(np.ascontiguousarray(initial_states_array[0]), copy=True),
                                                                cuda.to_device(np.ascontiguousarray(initial_states_array[1]), copy=True))




        self._initial_states_array = initial_states_array

    def _get_mixers_array(self,
                          beta: float):

        if beta not in self._mixers_dict:
            mixers_array = np.array([get_WS_mixer_operator_1q(angle=beta,
                                                              term_X=term_X,
                                                              term_Z=term_Z) for term_X, term_Z in self._XZ_terms])
            # print(mixers_array[0])
            #
            # raise KeyboardInterrupt

            if self.simulator_name in ['cuda']:
                from numba import cuda
                mixers_array = [mixers_array.real, mixers_array.imag]
                mixers_array[0], mixers_array[1] = (cuda.to_device(np.ascontiguousarray(mixers_array[0]), copy=True),
                                                    cuda.to_device(np.ascontiguousarray(mixers_array[1]), copy=True))



            self._mixers_dict[beta] = mixers_array

        return self._mixers_dict[beta]

    def _get_mixer_1q(self,
                      beta: float,
                      qubit_index: int):


        mixers_array = self._get_mixers_array(beta=beta)

        if self.simulator_name in ['cuda']:
            # cuda layout: [real_stack, imag_stack] of device arrays, each (n_qubits, 2, 2)
            return (mixers_array[0][qubit_index], mixers_array[1][qubit_index])

        # numpy/cython layout: complex array (n_qubits, 2, 2)
        return mixers_array[qubit_index]

    def _initialize_simulators_analytical(self,
                                          simulator_name=None,
                                          precision_float=None,
                                          ):
        super()._initialize_simulators_analytical(simulator_name=simulator_name,
                                                  precision_float=precision_float)
        self._set_initial_state()



    def clean_gpu_memory(self):
        # The caches refill lazily, so the runner stays usable after this call. The cuda RDM buffers live in
        # cupy's memory pool, which keeps freed blocks for reuse until asked to hand them back to the driver;
        # without that step other allocators (numba, torch) cannot use the memory.
        self._RDMS_dict.clear()
        self._RDMS_full.clear()
        self._RDMS_masks.clear()
        self._couplings_sums_dict.clear()
        self._mixers_dict.clear()

        import gc
        gc.collect()

        if self.simulator_name == 'cuda':
            cp.get_default_memory_pool().free_all_blocks()



    def _get_RDM_phase_separator_ij_numpy(self,
                                          idx_qi,
                                          idx_qj,
                                          couplings_phase,
                                          minus_four_gamma_1j_couplings,
                                          gamma_1j_couplings,
                                          couplings_sums_gamma_1j,
                                          ):


        qi_ket = self._initial_states_array[idx_qi, :, None].copy()
        qj_ket = self._initial_states_array[idx_qj, :, None].copy()

        qi_ket[0] *= np.conj(couplings_sums_gamma_1j[idx_qi,idx_qj])
        qi_ket[1] *= couplings_sums_gamma_1j[idx_qi,idx_qj]

        qj_ket[0] *= np.conj(couplings_sums_gamma_1j[idx_qj,idx_qi])
        qj_ket[1] *= couplings_sums_gamma_1j[idx_qj,idx_qi]


        # Switch to density matrices and compute the effect of
        # the two-qubit CP gate that comes from qubit k neq i,j
        qi_rho, qj_rho = qi_ket * np.matrix.getH(qi_ket), qj_ket * np.matrix.getH(qj_ket)
        rho_ij = np.kron(qi_rho, qj_rho)


        # Tracing out qubit k, prepared in sqrt(1-c_k)|0> + sqrt(c_k)|1>, mixes the two branches of
        # its controlled phase with the weights (1-c_k) and c_k: the weight is k's own bias.
        for k in range(self.number_of_qubits):
            if k in {idx_qi, idx_qj}:
                continue
            if couplings_phase[idx_qi, k] == 0.0 and couplings_phase[idx_qj, k] == 0.0:
                continue
            c = self._ws_bias_parameters[k]
            one_minus_c = 1.0 - c
            phase_i = minus_four_gamma_1j_couplings[idx_qi, k]
            phase_j = minus_four_gamma_1j_couplings[idx_qj, k]
            u1_ij = np.diag([1.0, phase_j, phase_i, phase_i * phase_j])
            rho_ij = one_minus_c * rho_ij + c * np.dot(u1_ij, np.dot(rho_ij, np.matrix.getH(u1_ij)))

        # Apply the two-qubit Rzz gate between `i` and `j`
        if couplings_phase[idx_qi, idx_qj] != 0.0:
            phase_p = gamma_1j_couplings[idx_qi, idx_qj]
            phase_m = np.conj(gamma_1j_couplings[idx_qi, idx_qj])
            u_ij = np.diag([phase_m, phase_p, phase_p, phase_m])
            rho_ij = np.dot(u_ij, np.dot(rho_ij, np.matrix.getH(u_ij)))

        return rho_ij


    def _get_RDM_phase_separator_i_numpy(self,
                                         idx_qi,
                                         couplings_phase,
                                         minus_four_gamma_1j_couplings,
                                         couplings_sums_gamma_1j,
                                         ):


        qi_ket = self._initial_states_array[idx_qi, :, None].copy()
        qi_ket[0] *= np.conj(couplings_sums_gamma_1j[idx_qi,idx_qi])
        qi_ket[1] *= couplings_sums_gamma_1j[idx_qi,idx_qi]
        rho_i = qi_ket * np.matrix.getH(qi_ket)

        # The weight of qubit k's controlled phase is k's own bias (see _get_RDM_phase_separator_ij_numpy).
        for k in range(self.number_of_qubits):
            if k == idx_qi:
                continue
            if couplings_phase[idx_qi, k] == 0.0:
                continue
            c = self._ws_bias_parameters[k]
            one_minus_c = 1.0 - c
            phase_i = minus_four_gamma_1j_couplings[idx_qi, k]
            #u1_diag = np.diag([1.0, phase_i])

            rho_i[1,0] = one_minus_c * rho_i[1,0] + c*phase_i*rho_i[1,0]
            rho_i[0,1] = np.conj(rho_i[1,0])

            # rho_i = one_minus_c * rho_i + c * np.dot(u1_diag, np.dot(rho_i, np.matrix.getH(u1_diag)))

        return rho_i






    def _get_cost_rdm_mask(self,
                           hamiltonian_representation_index: int):
        #The rdm_mask of the COST Hamiltonian's support: upper triangle = pair with a nonzero coupling,
        #diagonal = qubit with a nonzero field. The expected-value paths read no other entry.
        #Host bool array; on cuda a device uint8 view, made once so no per-call host-to-device copy.
        if hamiltonian_representation_index not in self._cost_rdm_masks:
            couplings_cost = self.couplings_cost_dict[hamiltonian_representation_index]['numpy']
            fields_cost = self.fields_cost_dict[hamiltonian_representation_index]['numpy']
            mask = np.triu((couplings_cost != 0.0) | (couplings_cost.T != 0.0), k=1)
            mask[np.diag_indices_from(mask)] = fields_cost != 0.0
            self._cost_rdm_masks[hamiltonian_representation_index] = self._normalize_rdm_mask(mask)

        return self._cost_rdm_masks[hamiltonian_representation_index]

    def _normalize_rdm_mask(self,
                            rdm_mask):
        #The rdm_mask in the form the active backend's builder reads, or None. Any nonzero entry means True;
        #a shape other than (n, n) is refused. numpy/cython read a host bool array; cuda reads a device uint8
        #array (a numba view). A mask already in that form is returned as the same object, so the cost mask
        #cached per representation index keeps its identity across calls.
        if rdm_mask is None:
            return None
        on_device = hasattr(rdm_mask, '__cuda_array_interface__')
        mask = cp.asarray(rdm_mask) if on_device else np.asarray(rdm_mask)
        expected_shape = (self.number_of_qubits, self.number_of_qubits)
        if mask.shape != expected_shape:
            raise ValueError(f"rdm_mask must have shape {expected_shape}; got {mask.shape}.")

        if self.simulator_name in ['cuda']:
            if on_device:
                if mask.dtype == cp.uint8:
                    return rdm_mask
                mask = (mask != 0).astype(cp.uint8)
            else:
                mask = cp.asarray(np.ascontiguousarray(mask != 0, dtype=np.uint8))
            from numba import cuda
            return cuda.as_cuda_array(mask)

        if on_device:
            return cp.asnumpy(mask) != 0
        if mask.dtype == np.bool_:
            return mask
        return mask != 0

    @staticmethod
    def _rdm_mask_covers(stored,
                         requested):
        #True when every entry the requested mask asks for was built under the stored mask.
        if hasattr(requested, '__cuda_array_interface__'):
            return bool(cp.all((cp.asarray(requested) == 0) | (cp.asarray(stored) != 0)))
        return bool(np.all(~np.asarray(requested, dtype=bool) | np.asarray(stored, dtype=bool)))

    def _get_cached_phase_rdms(self,
                               cache_key,
                               rdm_mask):
        #An array built with no mask serves every caller. A caller with rdm_mask=None needs every entry, so
        #nothing else serves it. A masked caller is served by a masked array only when that array's mask covers
        #the requested one: entries never built are zero, not stale, but they are not the requested values.
        if cache_key not in self._RDMS_dict:
            return None
        if cache_key in self._RDMS_full:
            return self._RDMS_dict[cache_key]
        if rdm_mask is None:
            return None
        stored_mask = self._RDMS_masks[cache_key]
        if stored_mask is rdm_mask or self._rdm_mask_covers(stored=stored_mask, requested=rdm_mask):
            return self._RDMS_dict[cache_key]
        return None

    def _store_phase_rdms(self,
                          cache_key,
                          local_RDMS,
                          rdm_mask,
                          memory_intensive: bool):
        if not memory_intensive:
            return
        self._RDMS_dict[cache_key] = local_RDMS
        if rdm_mask is None:
            self._RDMS_full.add(cache_key)
            self._RDMS_masks.pop(cache_key, None)
        else:
            self._RDMS_full.discard(cache_key)
            self._RDMS_masks[cache_key] = rdm_mask






    def _get_precomputed_phase_data(self,
                                    gamma,
                                    couplings_phase,
                                    fields_phase,
                                    couplings_sums,
                                    backend='numpy'):
        """Precompute all phase-related data for a given gamma."""

        if backend in ['numpy']:
            couplings_sums = couplings_sums[:,None]-couplings_phase#+np.diag(fields_phase)
            couplings_sums_gamma_1j = np.exp(1j*gamma*couplings_sums)
            minus_four_gamma_1j_couplings = np.exp(-4 * gamma * 1j * couplings_phase)
            gamma_1j_couplings = np.exp(gamma * 1j * couplings_phase)
        elif backend in ['cython']:
            #The cython kernels take the angles as a fused float32/float64 scalar before any array, so the
            #scalar's type picks the specialization: a numpy float32 scalar matches none. Pass Python floats.
            couplings_sums_gamma_1j, minus_four_gamma_1j_couplings, gamma_1j_couplings = get_precomputed_phase_data_cython(float(gamma),
                                                                                                                           couplings_phase,
                                                                                                                           fields_phase,
                                                                                                                           couplings_sums)

        return {
            'minus_four_gamma_1j_couplings':minus_four_gamma_1j_couplings,
            'gamma_1j_couplings':gamma_1j_couplings,
            'couplings_sums_gamma_1j':couplings_sums_gamma_1j,
        }


    def _get_couplings_sums(self,
                            hamiltonian_representation_index):

        if hamiltonian_representation_index in self._couplings_sums_dict:
            return self._couplings_sums_dict[hamiltonian_representation_index]



        t0 = time.perf_counter()

        if self.simulator_name in ['cuda']:
            from numba import cuda
            import cupy as cp

            # couplings_phase = cp.asarray(self.couplings_phase_dict[hamiltonian_representation_index]['cuda'])
            # fields_phase = cp.asarray(self.fields_phase_dict[hamiltonian_representation_index]['cuda'])
            # couplings_sums = cp.sum(couplings_phase, axis=1)
            #
            # if fields_phase is not None:
            #     couplings_sums += fields_phase

            # couplings_sums = cuda.to_device(cp.ascontiguousarray(couplings_sums, dtype=np.float32), copy=True)

            couplings_phase = self.couplings_phase_dict[hamiltonian_representation_index]['numpy']
            fields_phase = self.fields_phase_dict[hamiltonian_representation_index]['numpy']
            couplings_sums = np.sum(couplings_phase, axis=1)

            if fields_phase is not None:
                couplings_sums += fields_phase

            couplings_sums = cuda.to_device(np.ascontiguousarray(couplings_sums, dtype=np.float32), copy=True)



        elif self.simulator_name in ['numpy', 'cython']:
            couplings_phase = self.couplings_phase_dict[hamiltonian_representation_index]['numpy']
            fields_phase = self.fields_phase_dict[hamiltonian_representation_index]['numpy']
            couplings_sums = np.sum(couplings_phase, axis=1)

            if fields_phase is not None:
                couplings_sums += fields_phase

        else:
            raise NotImplementedError(f"Backend '{self.simulator_name}' not implemented for couplings sums computation.")

        self._couplings_sums_dict[hamiltonian_representation_index] = couplings_sums
        t1 = time.perf_counter()
        #('summing couplings took:', t1-t0, 'seconds.')


        return couplings_sums







    def _get_ZiZj_2q_numpy(self,
                           hamiltonian_representation_index,
                           gamma,
                           beta,
                           memory_intensive: bool = True,
                           cost_masks_override: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                           rdm_mask: Optional[np.ndarray] = None
                           ):
        #Mirror of the cython/cuda ZiZj methods: the phase-layer RDMs come from the (cached) builder
        #restricted to rdm_mask, and <ZiZj>, <Zi> are read where the cost coefficient (or the override)
        #is nonzero. An entry the cost reads but the mask excludes stays 0.
        if cost_masks_override is None:
            couplings_cost = self.couplings_cost_dict[hamiltonian_representation_index]['numpy']
            fields_cost = self.fields_cost_dict[hamiltonian_representation_index]['numpy']
        else:
            #(couplings, fields) arrays used ONLY as a which-entries-to-compute mask
            couplings_cost, fields_cost = cost_masks_override

        ZiZj_array = np.zeros((self.number_of_qubits, self.number_of_qubits), dtype=float)
        Zi_array = np.zeros((self.number_of_qubits,), dtype=float)

        local_RDMS = self._get_rho_ij_full_numpy(hamiltonian_representation_index=hamiltonian_representation_index,
                                                 gamma=gamma,
                                                 memory_intensive=memory_intensive,
                                                 rdm_mask=rdm_mask)

        for idx_qi in range(self.number_of_qubits):
            mixer_qi = self._get_mixer_1q(beta=beta, qubit_index=idx_qi)

            if fields_cost[idx_qi] != 0.0:
                rho_i_phase = local_RDMS[idx_qi, idx_qi, 0:2, 0:2]
                rho_i = np.dot(mixer_qi, np.dot(rho_i_phase, np.matrix.getH(mixer_qi)))
                Zi_array[idx_qi] = np.real(-rho_i[1, 1] + rho_i[0, 0])

            for idx_qj in range(idx_qi + 1, self.number_of_qubits):
                if couplings_cost[idx_qi, idx_qj] == 0.0:
                    continue
                mixer_qj = self._get_mixer_1q(beta=beta, qubit_index=idx_qj)
                mixer_qiqj = np.kron(mixer_qi, mixer_qj)
                rho_ij = np.dot(mixer_qiqj, np.dot(local_RDMS[idx_qi, idx_qj], np.matrix.getH(mixer_qiqj)))

                ZiZj_array[idx_qi, idx_qj] = np.real(rho_ij[0, 0] - rho_ij[1, 1] - rho_ij[2, 2] + rho_ij[3, 3])

        return ZiZj_array, Zi_array


    def _get_rho_ij_2q_cython(self,
                              hamiltonian_representation_index,
                              gamma,
                              memory_intensive: bool = True,
                              rdm_mask: Optional[np.ndarray] = None
                              ):
        #Phase-layer RDM array (n, n, 4, 4) of the cython backend. rdm_mask ((n, n), nonzero = True; diagonal
        #gates the one-qubit RDMs, upper triangle the pairs) restricts what is computed; None computes every entry.
        rdm_mask = self._normalize_rdm_mask(rdm_mask)
        cache_key = (hamiltonian_representation_index, gamma)
        local_RDMS = self._get_cached_phase_rdms(cache_key=cache_key, rdm_mask=rdm_mask)
        if local_RDMS is not None:
            return local_RDMS

        couplings_phase = self.couplings_phase_dict[hamiltonian_representation_index]['numpy']
        fields_phase = self.fields_phase_dict[hamiltonian_representation_index]['numpy']
        couplings_sums = self._get_couplings_sums(hamiltonian_representation_index=hamiltonian_representation_index)
        ws_bias_parameters = np.array(self._ws_bias_parameters, dtype=couplings_phase.dtype)

        #Python float: a numpy float32 angle matches no specialization of the cython kernel.
        local_RDMS = get_all_rho_ij_cython(float(gamma),
                                           self._initial_states_array,
                                           couplings_phase,
                                           fields_phase,
                                           couplings_sums,
                                           ws_bias_parameters,
                                           rdm_mask=rdm_mask)
        self._store_phase_rdms(cache_key=cache_key,
                               local_RDMS=local_RDMS,
                               rdm_mask=rdm_mask,
                               memory_intensive=memory_intensive)

        return local_RDMS




    def _get_rho_ij_2q_cuda(self,
                           hamiltonian_representation_index,
                           gamma,
                           memory_intensive: bool = True,
                           rdm_mask=None
                           ):
        #Phase-layer RDMs of the cuda backend as (real, imag) device views of shape (n, n, 4, 4).
        #rdm_mask: a host or device (n, n) array, nonzero = True; None computes every entry.
        rdm_mask = self._normalize_rdm_mask(rdm_mask)
        cache_key = (hamiltonian_representation_index, gamma)
        local_RDMS = self._get_cached_phase_rdms(cache_key=cache_key, rdm_mask=rdm_mask)
        if local_RDMS is not None:
            return local_RDMS

        couplings_phase = self.couplings_phase_dict[hamiltonian_representation_index]['cuda']
        fields_phase = self.fields_phase_dict[hamiltonian_representation_index]['cuda']
        couplings_sums = self._get_couplings_sums(hamiltonian_representation_index=hamiltonian_representation_index)

        local_RDMS = get_all_rho_ij_cuda(gamma=gamma,
                                         initial_states_real=self._initial_states_array[0],
                                         initial_states_imag=self._initial_states_array[1],
                                         couplings_phase=couplings_phase,
                                         fields_phase=fields_phase,
                                         couplings_sums=couplings_sums,
                                         ws_bias_parameters=self._ws_bias_parameters,
                                         rdm_mask=rdm_mask
                                         )
        self._store_phase_rdms(cache_key=cache_key,
                               local_RDMS=local_RDMS,
                               rdm_mask=rdm_mask,
                               memory_intensive=memory_intensive)

        return local_RDMS






    def _get_ZiZj_2q_cython(self,
                            hamiltonian_representation_index,
                            gamma,
                            beta,
                            memory_intensive: bool = True,
                            cost_masks_override: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                            rdm_mask: Optional[np.ndarray] = None
                            ):


        couplings_phase = self.couplings_phase_dict[hamiltonian_representation_index]['numpy']
        fields_phase = self.fields_phase_dict[hamiltonian_representation_index]['numpy']
        if cost_masks_override is None:
            couplings_cost = self.couplings_cost_dict[hamiltonian_representation_index]['numpy']
            fields_cost = self.fields_cost_dict[hamiltonian_representation_index]['numpy']
        else:
            #(couplings, fields) arrays used ONLY as a which-entries-to-compute mask
            couplings_cost, fields_cost = cost_masks_override
        couplings_sums = self._get_couplings_sums(hamiltonian_representation_index=hamiltonian_representation_index)

        local_RDMS = self._get_rho_ij_2q_cython(hamiltonian_representation_index=hamiltonian_representation_index,
                                                gamma=gamma,
                                                memory_intensive=memory_intensive,
                                                rdm_mask=rdm_mask)

        #Python floats: a numpy float32 angle matches no specialization of the cython kernel.
        ZiZj_array, Zi_array = get_all_ZiZj_cython(float(gamma),
                                         float(beta),
                                         local_RDMS,
                                         self._get_mixers_array(beta=beta),
                                         couplings_phase,
                                         fields_phase,
                                         couplings_cost,
                                         fields_cost,
                                         self._initial_states_array,
                                         couplings_sums)

        return ZiZj_array, Zi_array

    def _get_ZiZj_2q_cuda(self,
                            hamiltonian_representation_index,
                            gamma,
                            beta,
                            memory_intensive: bool = True,
                            cost_masks_override=None,
                            rdm_mask=None
                            ):

        if cost_masks_override is None:
            fields_cost = self.fields_cost_dict[hamiltonian_representation_index]['cuda']
            couplings_cost = self.couplings_cost_dict[hamiltonian_representation_index]['cuda']
        else:
            #(couplings, fields) device arrays used ONLY as a which-entries-to-compute mask
            couplings_cost, fields_cost = cost_masks_override

        local_RDMS = self._get_rho_ij_2q_cuda(hamiltonian_representation_index=hamiltonian_representation_index,
                                             gamma=gamma,
                                             memory_intensive=memory_intensive,
                                             rdm_mask=rdm_mask)

        t0 = time.perf_counter()
        mixers_array = self._get_mixers_array(beta=beta)
        ZiZj_array, Zi_array = get_all_ZiZj_cuda(rho_ij_real=local_RDMS[0],
                                       rho_ij_imag=local_RDMS[1],
                                         mixers_real = mixers_array[0],
                                       mixers_imag=mixers_array[1],
                                         couplings_cost=couplings_cost,
                                         fields_cost=fields_cost,

                                         )
        t1 = time.perf_counter()
        # print("Time to calculate ZiZj:",t1-t0)




        return ZiZj_array, Zi_array

    def _get_all_ones_cost_masks(self):
        #All-ones (couplings, fields) arrays in the active backend's layout, lazily cached.
        #Passed as cost_masks_override to the _get_ZiZj_2q_* methods: the cost coefficients
        #there are ONLY a which-entries-to-compute mask (never multiplied into the values),
        #so all-ones masks make every <ZiZj> (i<j) and every <Zi> get computed.
        #dtype matches self._precision_float so the cython fused types / numba specializations
        #agree with the regular cost arrays.
        if self._all_ones_cost_masks is None:
            n = self.number_of_qubits
            couplings_ones = np.ones((n, n), dtype=self._precision_float)
            fields_ones = np.ones((n,), dtype=self._precision_float)

            if self.simulator_name in ['cuda']:
                from numba import cuda
                couplings_ones = cuda.to_device(np.ascontiguousarray(couplings_ones), copy=True)
                fields_ones = cuda.to_device(np.ascontiguousarray(fields_ones), copy=True)

            self._all_ones_cost_masks = (couplings_ones, fields_ones)

        return self._all_ones_cost_masks



    def _get_expected_value_cuda(self,
                                 hamiltonian_representation_index:int,
                                 gamma:float,
                                 beta:float|np.ndarray|cp.ndarray,
                                 memory_intensive: bool = False,
                                 return_correlators:bool = False,
                                 return_only_best_beta:bool=True
                                 )->float|Tuple[float, np.ndarray]:
        if self._same_bias and not return_correlators:
            if (hamiltonian_representation_index, gamma) in self._overlaps_dict['cupy']:
                (overlaps_cupy_2q, overlaps_cupy_1q) = self._overlaps_dict['cupy'][(hamiltonian_representation_index, gamma)]


            else:
                t0 = time.perf_counter()
                local_RDMS = self._get_rho_ij_2q_cuda(hamiltonian_representation_index=hamiltonian_representation_index,
                                                      gamma=gamma,
                                                      memory_intensive=memory_intensive,
                                                      rdm_mask=self._get_cost_rdm_mask(hamiltonian_representation_index))
                t1 = time.perf_counter()

                fields_cost = None
                if self.hamiltonian_representations_cost[hamiltonian_representation_index].has_local_fields:
                    fields_cost = self.fields_cost_dict[hamiltonian_representation_index]['cuda']

                overlaps_cupy_2q, overlaps_cupy_1q = get_pauli_overlaps_cupy(rho_ij_real=local_RDMS[0],
                                                                            rho_ij_imag=local_RDMS[1],
                                                                            couplings_cost=self.couplings_cost_dict[hamiltonian_representation_index]['cuda'],
                                                                            fields_cost=fields_cost
                                                                            )
                t2 = time.perf_counter()
                self._overlaps_dict['cupy'][(hamiltonian_representation_index, gamma)] = [overlaps_cupy_2q, overlaps_cupy_1q]




            if beta is None:
                # Closed-form mixer angle: the energy is a trigonometric polynomial in beta whose
                # minimum is a root of a quartic; one-body terms shift the constant and the first
                # harmonic.
                if not return_only_best_beta:
                    raise ValueError("beta=None returns the single minimising angle; a full curve needs a beta array. "
                                     "Pass a beta grid, or drop return_only_best_beta=False.")
                overlaps_2q_host = np.asarray(cp.asnumpy(overlaps_cupy_2q), dtype=np.float64)
                overlaps_1q_host = (None if overlaps_cupy_1q is None
                                    else np.asarray(cp.asnumpy(overlaps_cupy_1q), dtype=np.float64))
                mixer_x, mixer_z = self._XZ_terms[0]
                coefficients = p1_beta_fourier_coefficients(overlaps_2q=overlaps_2q_host,
                                                            mixer_x=mixer_x,
                                                            mixer_z=mixer_z,
                                                            overlaps_1q=overlaps_1q_host)
                best_beta, best_value = minimize_p1_beta_trigonometric_polynomial(*coefficients)
                return best_value, best_beta
            elif isinstance(beta,float):
                overlaps_numpy_2q = cp.asnumpy(overlaps_cupy_2q)

                overlaps_numpy_1q = None
                if overlaps_cupy_1q is not None:
                    overlaps_numpy_1q = cp.asnumpy(overlaps_cupy_1q)

                return self._cost_beta(beta=beta,
                                       overlaps_2q=overlaps_numpy_2q,
                                       overlaps_1q=overlaps_numpy_1q)
            elif isinstance(beta,(np.ndarray, cp.ndarray)):
                if len(beta)>10**4:
                    _beta_pass = cp.array(beta)
                    _overlaps_2q = overlaps_cupy_2q
                    _overlaps_1q = overlaps_cupy_1q
                else:
                    _beta_pass = beta
                    _overlaps_2q = cp.asnumpy(overlaps_cupy_2q)
                    _overlaps_1q = None
                    if overlaps_cupy_1q is not None:
                        _overlaps_1q = cp.asnumpy(overlaps_cupy_1q)

                t2= time.perf_counter()
                # self.clean_gpu_memory()
                t3 = time.perf_counter()
                _values = self._cost_many_betas_cuda(betas_array=_beta_pass,
                                                  overlaps_2q=_overlaps_2q,
                                                  overlaps_1q=_overlaps_1q,
                                                     return_only_best_beta=return_only_best_beta
                                                  )
                t4 = time.perf_counter()



                # print("getting RDMS took:", t1-t0, "\ngetting overlaps took:", t2-t1)
                # print("Cleaning GPU memory took:", t3-t2)
                # print( "Costing betas took:", t4-t3)






                return _values
            else:
                raise ValueError(f"beta must be either a float or a numpy array, not {type(beta)}")



        else:

            if not isinstance(beta,float):
                raise NotImplementedError("array of betas not implemented for with stored correlators and not-same bias.")

            #TODO(FBM): work on this
            couplings_cost = self.couplings_cost_dict[hamiltonian_representation_index]['numpy']
            fields_cost = self.fields_cost_dict[hamiltonian_representation_index]['numpy']

            ZiZj_array, Zi_array = self._get_ZiZj_2q_cuda(hamiltonian_representation_index=hamiltonian_representation_index,
                                                        gamma=gamma,
                                                        beta=beta,
                                                        memory_intensive=memory_intensive,
                                                        rdm_mask=self._get_cost_rdm_mask(hamiltonian_representation_index)
                                                        )

            Cij_noiseless = couplings_cost * ZiZj_array
            Ci_noiseless = fields_cost * Zi_array

            exp_value_noiseless = np.sum(Ci_noiseless) + np.sum(np.sum(Cij_noiseless))

            Cij_noiseless[np.diag_indices_from(Cij_noiseless)] = Ci_noiseless

            if return_correlators:
                return exp_value_noiseless, Cij_noiseless

            return exp_value_noiseless

    def _get_rho_ij_full_numpy(self,
                               hamiltonian_representation_index,
                               gamma,
                               memory_intensive: bool = False,
                               rdm_mask: Optional[np.ndarray] = None):
        #Assemble the (n, n, 4, 4) phase-separator RDM array for the numpy backend
        #(same layout as the cython/cuda builders: [i, j] for pairs, [i, i, 0:2, 0:2] for 1q).
        #rdm_mask ((n, n), nonzero = True): the diagonal gates the one-qubit RDMs, the upper triangle the pairs;
        #None fills every entry. Entries left out stay 0.
        rdm_mask = self._normalize_rdm_mask(rdm_mask)
        cache_key = (hamiltonian_representation_index, gamma)
        rho_full = self._get_cached_phase_rdms(cache_key=cache_key, rdm_mask=rdm_mask)
        if rho_full is not None:
            return rho_full

        couplings_phase = self.couplings_phase_dict[hamiltonian_representation_index]['numpy']
        fields_phase = self.fields_phase_dict[hamiltonian_representation_index]['numpy']
        couplings_sums = self._get_couplings_sums(hamiltonian_representation_index=hamiltonian_representation_index)

        _precomputed_data_phase = self._get_precomputed_phase_data(gamma=gamma,
                                                                   couplings_phase=couplings_phase,
                                                                   fields_phase=fields_phase,
                                                                   couplings_sums=couplings_sums,
                                                                   backend='numpy')

        n = self.number_of_qubits
        allowed = np.ones((n, n), dtype=bool) if rdm_mask is None else rdm_mask

        rho_full = np.zeros((n, n, 4, 4), dtype=np.complex64)
        for idx_qi in range(n):
            if allowed[idx_qi, idx_qi]:
                rho_full[idx_qi, idx_qi, 0:2, 0:2] = self._get_RDM_phase_separator_i_numpy(
                    idx_qi=idx_qi,
                    couplings_phase=couplings_phase,
                    minus_four_gamma_1j_couplings=_precomputed_data_phase['minus_four_gamma_1j_couplings'],
                    couplings_sums_gamma_1j=_precomputed_data_phase['couplings_sums_gamma_1j'])

            for idx_qj in range(idx_qi + 1, n):
                if not allowed[idx_qi, idx_qj]:
                    continue
                rho_full[idx_qi, idx_qj] = self._get_RDM_phase_separator_ij_numpy(
                    idx_qi=idx_qi,
                    idx_qj=idx_qj,
                    couplings_phase=couplings_phase,
                    **_precomputed_data_phase)

        self._store_phase_rdms(cache_key=cache_key,
                               local_RDMS=rho_full,
                               rdm_mask=rdm_mask,
                               memory_intensive=memory_intensive)

        return rho_full

    def _get_expected_value_cpu(self,
                                hamiltonian_representation_index: int,
                                gamma: float,
                                beta: float | np.ndarray,
                                memory_intensive: bool = False,
                                return_correlators: bool = False,
                                return_only_best_beta: bool = True,
                                backend: str = 'cython'
                                ) -> float | Tuple[float, np.ndarray]:
        #CPU (cython/numpy) mirror of _get_expected_value_cuda: same-bias fast path via
        #Pauli overlaps (supports beta arrays through _cost_many_betas_cuda, which is
        #backend-agnostic despite its name), otherwise the explicit ZiZj path.
        if backend not in ['cython', 'numpy']:
            raise ValueError(f"backend must be 'cython' or 'numpy', not {backend}")

        if self._same_bias and not return_correlators:
            _cache_key = (hamiltonian_representation_index, gamma)
            if _cache_key in self._overlaps_dict['numpy']:
                overlaps_2q, overlaps_1q = self._overlaps_dict['numpy'][_cache_key]
            else:
                cost_rdm_mask = self._get_cost_rdm_mask(hamiltonian_representation_index)
                if backend == 'cython':
                    local_RDMS = self._get_rho_ij_2q_cython(
                        hamiltonian_representation_index=hamiltonian_representation_index,
                        gamma=gamma,
                        memory_intensive=memory_intensive,
                        rdm_mask=cost_rdm_mask)
                else:
                    local_RDMS = self._get_rho_ij_full_numpy(
                        hamiltonian_representation_index=hamiltonian_representation_index,
                        gamma=gamma,
                        memory_intensive=memory_intensive,
                        rdm_mask=cost_rdm_mask)

                fields_cost = None
                if self.hamiltonian_representations_cost[hamiltonian_representation_index].has_local_fields:
                    fields_cost = self.fields_cost_dict[hamiltonian_representation_index]['numpy']

                overlaps_2q, overlaps_1q = get_pauli_overlaps_numpy(
                    rho_ij=local_RDMS,
                    couplings_cost=self.couplings_cost_dict[hamiltonian_representation_index]['numpy'],
                    fields_cost=fields_cost)
                self._overlaps_dict['numpy'][_cache_key] = [overlaps_2q, overlaps_1q]

            if beta is None:
                # Closed-form mixer angle; see the cuda path for the derivation note.
                if not return_only_best_beta:
                    raise ValueError("beta=None returns the single minimising angle; a full curve needs a beta array. "
                                     "Pass a beta grid, or drop return_only_best_beta=False.")
                overlaps_2q_host = np.asarray(overlaps_2q, dtype=np.float64)
                overlaps_1q_host = None if overlaps_1q is None else np.asarray(overlaps_1q, dtype=np.float64)
                mixer_x, mixer_z = self._XZ_terms[0]
                coefficients = p1_beta_fourier_coefficients(overlaps_2q=overlaps_2q_host,
                                                            mixer_x=mixer_x,
                                                            mixer_z=mixer_z,
                                                            overlaps_1q=overlaps_1q_host)
                best_beta, best_value = minimize_p1_beta_trigonometric_polynomial(*coefficients)
                return best_value, best_beta
            elif isinstance(beta, float):
                return self._cost_beta(beta=beta,
                                       overlaps_2q=overlaps_2q,
                                       overlaps_1q=overlaps_1q)
            elif isinstance(beta, np.ndarray):
                return self._cost_many_betas_cuda(betas_array=beta,
                                                  overlaps_2q=overlaps_2q,
                                                  overlaps_1q=overlaps_1q,
                                                  return_only_best_beta=return_only_best_beta)
            else:
                raise ValueError(f"beta must be either a float or a numpy array, not {type(beta)}")

        else:
            if not isinstance(beta, float):
                raise NotImplementedError("array of betas not implemented for with stored correlators and not-same bias.")

            couplings_cost = self.couplings_cost_dict[hamiltonian_representation_index]['numpy']
            fields_cost = self.fields_cost_dict[hamiltonian_representation_index]['numpy']
            cost_rdm_mask = self._get_cost_rdm_mask(hamiltonian_representation_index)

            if backend == 'cython':
                ZiZj_array, Zi_array = self._get_ZiZj_2q_cython(
                    hamiltonian_representation_index=hamiltonian_representation_index,
                    gamma=gamma,
                    beta=beta,
                    memory_intensive=memory_intensive,
                    rdm_mask=cost_rdm_mask)
            else:
                ZiZj_array, Zi_array = self._get_ZiZj_2q_numpy(
                    hamiltonian_representation_index=hamiltonian_representation_index,
                    gamma=gamma,
                    beta=beta,
                    memory_intensive=memory_intensive,
                    rdm_mask=cost_rdm_mask)

            Cij_noiseless = couplings_cost * ZiZj_array
            Ci_noiseless = fields_cost * Zi_array

            exp_value_noiseless = np.sum(Ci_noiseless) + np.sum(np.sum(Cij_noiseless))

            Cij_noiseless[np.diag_indices_from(Cij_noiseless)] = Ci_noiseless

            if return_correlators:
                return exp_value_noiseless, Cij_noiseless

            return exp_value_noiseless

    def get_expected_value(self,
                           hamiltonian_representation_index,
                           gamma,
                           beta,
                           memory_intensive: bool = False,
                           return_correlators:bool=False,
                           return_only_best_beta:bool=True
                           )->float|Tuple[float,np.ndarray]:


        if self.simulator_name in ['cuda']:
            return self._get_expected_value_cuda(hamiltonian_representation_index=hamiltonian_representation_index,
                                                 gamma=gamma,
                                                 beta=beta,
                                                 memory_intensive=memory_intensive,
                                                 return_correlators=return_correlators,
                                                 return_only_best_beta=return_only_best_beta)
        elif self.simulator_name in ['cython', 'numpy']:
            return self._get_expected_value_cpu(hamiltonian_representation_index=hamiltonian_representation_index,
                                                gamma=gamma,
                                                beta=beta,
                                                memory_intensive=memory_intensive,
                                                return_correlators=return_correlators,
                                                return_only_best_beta=return_only_best_beta,
                                                backend=self.simulator_name)
        else:
            raise NotImplementedError(f"Backend '{self.simulator_name}' not implemented for ZiZj computation.")



    def get_ZiZj_2q(self,
                   hamiltonian_representation_index,
                   gamma,
                   beta,
                   memory_intensive: bool = True
                   ):
        #<ZiZj> and <Zi> on the COST Hamiltonian's support only; the builders compute nothing else.
        cost_rdm_mask = self._get_cost_rdm_mask(hamiltonian_representation_index)

        if self.simulator_name in ['cuda']:
            return self._get_ZiZj_2q_cuda(hamiltonian_representation_index=hamiltonian_representation_index,
                                         gamma=gamma,
                                         beta=beta,
                                         memory_intensive=memory_intensive,
                                         rdm_mask=cost_rdm_mask)
        elif self.simulator_name in ['cython']:
            return self._get_ZiZj_2q_cython(hamiltonian_representation_index=hamiltonian_representation_index,
                                            gamma=gamma,
                                            beta=beta,
                                            memory_intensive=memory_intensive,
                                            rdm_mask=cost_rdm_mask)

        elif self.simulator_name in ['numpy']:
            return self._get_ZiZj_2q_numpy(hamiltonian_representation_index=hamiltonian_representation_index,
                                           gamma=gamma,
                                           beta=beta,
                                           memory_intensive=memory_intensive,
                                           rdm_mask=cost_rdm_mask
                                           )
        else:
            raise NotImplementedError(f"Backend '{self.simulator_name}' not implemented for ZiZj computation.")

    def get_all_ZiZj_Zi(self,
                        hamiltonian_representation_index,
                        gamma,
                        beta,
                        memory_intensive: bool = True
                        ):
        """
        Compute ALL two-body <ZiZj> (i<j) and ALL one-body <Zi> expectation values,
        independently of the cost Hamiltonian's coefficients.

        Unlike get_ZiZj_2q, entries where the cost Hamiltonian has J_ij = 0 or h_i = 0
        are NOT skipped -- all n*(n-1)/2 correlators and n magnetizations are computed.
        The values themselves depend only on the phase separator (of the given
        representation index) and the mixer, never on cost coefficients.

        Returns:
        --------
        ZiZj_array : array, shape (n_qubits, n_qubits)
            Upper triangle (i<j) holds <ZiZj>; diagonal and lower triangle are 0.
        Zi_array : array, shape (n_qubits,)
            <Zi> values.
        """
        cost_masks_override = self._get_all_ones_cost_masks()

        #rdm_mask=None: every phase-layer RDM is built (a cost-masked cached array is never reused here).
        if self.simulator_name in ['cuda']:
            return self._get_ZiZj_2q_cuda(hamiltonian_representation_index=hamiltonian_representation_index,
                                          gamma=gamma,
                                          beta=beta,
                                          memory_intensive=memory_intensive,
                                          cost_masks_override=cost_masks_override,
                                          rdm_mask=None)
        elif self.simulator_name in ['cython']:
            return self._get_ZiZj_2q_cython(hamiltonian_representation_index=hamiltonian_representation_index,
                                            gamma=gamma,
                                            beta=beta,
                                            memory_intensive=memory_intensive,
                                            cost_masks_override=cost_masks_override,
                                            rdm_mask=None)
        elif self.simulator_name in ['numpy']:
            return self._get_ZiZj_2q_numpy(hamiltonian_representation_index=hamiltonian_representation_index,
                                           gamma=gamma,
                                           beta=beta,
                                           memory_intensive=memory_intensive,
                                           cost_masks_override=cost_masks_override,
                                           rdm_mask=None)
        else:
            raise NotImplementedError(
                f"Backend '{self.simulator_name}' not implemented for cost-independent ZiZj computation.")

    def get_RDMs(self,
                 gamma: float,
                 beta: float,
                 hamiltonian_representation_index: int = 0,
                 memory_intensive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """One- and two-qubit reduced density matrices of the p=1 state at (gamma, beta).

        Returns host numpy arrays (complex64) on every backend:
          rdms_1q: shape (n, 2, 2); rdms_1q[i] is the RDM of qubit i.
          rdms_2q: shape (n, n, 4, 4); rdms_2q[i, j] for i < j is the RDM of qubits (i, j) in the basis
                   |q_i q_j> (index 2*a + b for bits a of q_i and b of q_j); the diagonal and lower blocks are 0.
        All pairs and all qubits are computed, independent of the cost Hamiltonian.
        beta = 0.0 gives the state after the phase separator, before the mixer.

        :param gamma: phase separator angle
        :param beta: mixer angle
        :param hamiltonian_representation_index: which phase Hamiltonian representation to use
        :param memory_intensive: cache the phase-layer RDMs under (index, gamma) for later calls
        """
        n = self.number_of_qubits

        if self.simulator_name in ['cuda']:
            rho_real, rho_imag = self._get_rho_ij_2q_cuda(hamiltonian_representation_index=hamiltonian_representation_index,
                                                          gamma=gamma,
                                                          memory_intensive=memory_intensive,
                                                          rdm_mask=None)
            rho_phase = (cp.asnumpy(cp.asarray(rho_real)).astype(np.complex64)
                         + 1j * cp.asnumpy(cp.asarray(rho_imag)).astype(np.complex64))
        elif self.simulator_name in ['cython']:
            rho_phase = np.asarray(self._get_rho_ij_2q_cython(hamiltonian_representation_index=hamiltonian_representation_index,
                                                              gamma=gamma,
                                                              memory_intensive=memory_intensive,
                                                              rdm_mask=None), dtype=np.complex64)
        elif self.simulator_name in ['numpy']:
            rho_phase = np.asarray(self._get_rho_ij_full_numpy(hamiltonian_representation_index=hamiltonian_representation_index,
                                                               gamma=gamma,
                                                               memory_intensive=memory_intensive,
                                                               rdm_mask=None), dtype=np.complex64)
        else:
            raise NotImplementedError(f"Backend '{self.simulator_name}' not implemented for RDM computation.")

        # Host mixers per qubit, each with that qubit's own mixer axis.
        mixers = np.array([get_WS_mixer_operator_1q(angle=beta, term_X=term_X, term_Z=term_Z, backend='numpy')
                           for term_X, term_Z in self._XZ_terms], dtype=np.complex64)
        mixers_dagger = np.conj(np.swapaxes(mixers, 1, 2))

        diagonal = np.arange(n)
        rdms_1q = np.einsum('iab,ibc,icd->iad', mixers, rho_phase[diagonal, diagonal, 0:2, 0:2], mixers_dagger)

        # kron(M_i, M_j)[2a + c, 2b + d] = M_i[a, b] * M_j[c, d]
        mixers_2q = np.einsum('iab,jcd->ijacbd', mixers, mixers).reshape(n, n, 4, 4)
        mixers_2q_dagger = np.conj(np.swapaxes(mixers_2q, 2, 3))
        rdms_2q = np.einsum('ijab,ijbc,ijcd->ijad', mixers_2q, rho_phase, mixers_2q_dagger)
        # The builders keep one-qubit data in the diagonal blocks and nothing below the diagonal.
        rdms_2q[~np.triu(np.ones((n, n), dtype=bool), k=1)] = 0.0

        return rdms_1q.astype(np.complex64), rdms_2q.astype(np.complex64)


    def _cost_beta(self,
                   beta:float,
                   overlaps_2q:np.ndarray,
                   overlaps_1q:Optional[np.ndarray]=None
                   ):
        _sinb2 = np.sin(beta) ** 2
        _sin2b = np.sin(2 * beta)
        _cos2b = np.cos(2 * beta)


        _funs = [_sinb2, _sin2b, _cos2b]

        _funval = 0.0
        for i in range(3):
            for j in range(3):
                overlap_ij = overlaps_2q[i, j]
                if overlap_ij == 0.0:
                    continue
                coeff_i, coeff_j = self._c_coeffs[i], self._c_coeffs[j]
                fun_i, fun_j = _funs[i], _funs[j]
                if i == 2:
                    _prod_ij = fun_i*coeff_i+self._c_coeffs[3]
                else:
                    _prod_ij = fun_i*coeff_i
                if j == 2:
                    _prod_ij *= (coeff_j * fun_j + self._c_coeffs[3])
                else:
                    _prod_ij *= coeff_j * fun_j

                #ci*fun_i*fun_j*c_j

                _funval += overlap_ij * _prod_ij


        if overlaps_1q is None:
            return _funval

        cx, cy, cz1, cz0 = self._c_coeffs[0], self._c_coeffs[1], self._c_coeffs[2], self._c_coeffs[3]

        #The transformation under WS mixer is:
        # Z --> cx * sin(beta)^2 * X + cy * sin(2*beta) * Y + (cz1*Cos(2*beta)+cz0)*Z
        cost_1q = cx*overlaps_1q[0]*_sinb2 + cy*overlaps_1q[1]*_sin2b + (cz1*_cos2b+cz0)*overlaps_1q[2]
        return _funval + cost_1q









    def _cost_many_betas_cuda(self,
                              betas_array: np.ndarray,
                              overlaps_2q: np.ndarray,
                              overlaps_1q: Optional[np.ndarray] = None,
                              return_only_best_beta:bool=True
                              ):

        t0 = time.perf_counter()

        if isinstance(betas_array, np.ndarray):
            import numpy as _bck
        else:
            import cupy as _bck

        _sinb2 = _bck.sin(betas_array) ** 2
        _sin2b = _bck.sin(2 * betas_array)
        _cos2b = _bck.cos(2 * betas_array)

        # shape (3, number_of_betas)
        _funs_array = _bck.array([_sinb2, _sin2b, _cos2b])
        _coeffs_xyz = _bck.array(self._c_coeffs[0:3])[:, None]
        _fun_coeff_array = _funs_array * _coeffs_xyz

        # shape (3, 3, number_of_betas)
        _fun_coeff_array_ij = _fun_coeff_array[:, None, :] * _fun_coeff_array[None, :, :]

        # Main term: shape (number_of_betas,)
        _funvals_betas = (_fun_coeff_array_ij * overlaps_2q[:, :, None]).sum(axis=(0, 1))

        # Cross terms with cz0: shape (number_of_betas,)
        cz0 = self._c_coeffs[3]
        cross_coeff = overlaps_2q[:, 2] + overlaps_2q[2, :]  # shape (3,)
        _cross_terms = cz0 * (_fun_coeff_array * cross_coeff[:, None]).sum(axis=0)

        # Constant term: scalar
        _constant = cz0 ** 2 * overlaps_2q[2, 2]

        _funvals_betas = _funvals_betas + _cross_terms + _constant

        if overlaps_1q is not None:
            cx, cy, cz1, cz0 = self._c_coeffs
            cost_1q = cx * overlaps_1q[0] * _sinb2 + cy * overlaps_1q[1] * _sin2b + (cz1 * _cos2b + cz0) * overlaps_1q[2]

            _funvals_betas+=cost_1q
        t1 = time.perf_counter()
        if return_only_best_beta:


            best_beta_index = _bck.argmin(_funvals_betas)

            t2 = time.perf_counter()
            best_beta = float(betas_array[best_beta_index])
            best_value = float(_funvals_betas[best_beta_index])
            t3 = time.perf_counter()
            # print("Calculating funvals took:",t1-t0)
            # print('Finding best beta:', t2-t1)
            # print('Communicating to CPU:',t3-t2)


            return best_value, best_beta


        return _funvals_betas



        #_funs_array = cp.array(_funs)











    def run_qaoa(self,
                 *args,
                 # qaoa_depth: int,
                 measurement_noise: ClassicalMeasurementNoiseSampler = None,
                 store_correlators:bool=False,
                 input_format: Optional[FIFormat] = None,
                 memory_intensive:Optional[bool]=None,
                 debug=False,
                 qaoa_depth=1,
                 number_of_samples=None,
                 numpy_rng_sampling=None,
                 trial_index_offset=0,
                 operators_dict=None,
                 find_best_beta_only:bool=False,
                 betas_search_space_size:Optional[int]=None,
                 betas_search_space:Optional[np.ndarray]=None,
                 analytical_betas:Optional[bool]=None,
                 # debug_array=None
                 ) -> QAOAResult|QAOAResultSimplified:
        """
        With find_best_beta_only=True the call takes the gamma alone and the mixer angle is
        chosen for it. How it is chosen depends on analytical_betas:

        - None (default): the exact closed form for a two-local Hamiltonian (with or without
          local fields) with a uniform bias and no caller-supplied betas_search_space; otherwise
          the numerical grid.
        - True: the closed form, raising NotImplementedError where it does not apply.
        - False: always the numerical grid (betas_search_space or a linspace of
          betas_search_space_size points over [-pi/2, pi/2], default 10**6).

        analytical_betas is meaningful only together with find_best_beta_only=True.

        The closed form and the grid agree on the energy, not always on the angle: at bias 0.5
        the energy is pi/2-periodic in beta, so the returned beta may sit in a different branch
        than the grid's (they differ by a multiple of pi/2). Compare energies, not stored angles.
        """

        assert qaoa_depth == 1, "This method is only implemented for QAOA depth 1"
        assert number_of_samples is None or number_of_samples == np.inf, "This method is only implemented for number_of_samples=None or np.inf"
        assert measurement_noise is None, "This method does not support measurement noise"
        t0_input= time.perf_counter()

        if analytical_betas is not None and not find_best_beta_only:
            raise ValueError("analytical_betas applies only together with find_best_beta_only=True; "
                             "pass both, or drop analytical_betas.")

        if find_best_beta_only:
            angles, hamiltonian_representation_index, trial_index = self._input_handler_analytical_betas_p1(args=args,
                                                                                                            input_format=input_format)

            self._trial_index+=1


            assert len(angles) == 1, "The number of angles must be 1"

            use_closed_form = ((analytical_betas is not False)
                               and betas_search_space is None
                               and self._same_bias)
            if analytical_betas is True and not use_closed_form:
                if not self._same_bias:
                    obstacle = 'a non-uniform bias'
                else:
                    obstacle = 'an explicit betas_search_space'
                raise NotImplementedError("Closed-form beta is implemented for two-local Hamiltonians with a uniform bias and no "
                                          f"caller-supplied betas_search_space; this call has {obstacle}. "
                                          "Pass analytical_betas=None or False to use the grid search.")

            if use_closed_form:
                # beta=None selects the closed-form branch of get_expected_value; no grid is built.
                beta_j = None
            else:
                beta_j = betas_search_space
                if beta_j is None:
                    if betas_search_space_size is None:
                        betas_search_space_size = 10**6

                    beta_j = np.linspace(-np.pi/2,
                                         np.pi/2,
                                         betas_search_space_size)

            if store_correlators:
                #TODO(FBM): implement taking the best beta and then finding correlators. easy to do
                raise NotImplementedError("Storing correlators is not implemented when finding best beta only.")

        else:

            angles, hamiltonian_representation_index, trial_index = self._input_handler(args=args,
                                                                                        input_format=input_format,
                                                                                        qaoa_depth=1, )
            assert len(angles) == 2, "The number of angles must be 2"
            beta_j = angles[1:][0]



        gamma_j = angles[0:1][0]

        trial_index += trial_index_offset

        t1_input = time.perf_counter()



        # gamma_j = couplings_cost.dtype.type(gamma_j)
        # beta_j = couplings_cost.dtype.type(beta_j)

        t0 = time.perf_counter()
        if memory_intensive is None:
            memory_intensive = store_correlators



        _res = self.get_expected_value(hamiltonian_representation_index=hamiltonian_representation_index,
                                        gamma=gamma_j,
                                        beta=beta_j,
                                        memory_intensive=memory_intensive,
                                        return_correlators=store_correlators
                                        )

        if store_correlators:
            exp_value_noiseless, Cij_noiseless = _res
        else:

            if find_best_beta_only:
                (exp_value_noiseless, best_beta), Cij_noiseless = _res, None
                angles = np.array([gamma_j, best_beta])
            else:
                exp_value_noiseless, Cij_noiseless = _res, None

        t1 = time.perf_counter()

        if self._simplified_data_storage:
            qaoa_result:QAOAResultSimplified = ((trial_index, hamiltonian_representation_index, angles), (exp_value_noiseless, Cij_noiseless))

        else:

            energy_result = EnergyResultMain(energy_mean_noiseless=exp_value_noiseless)
            energy_result.update_main_energy(noisy=False)

            qaoa_result:QAOAResult = QAOAResult(energy_result=energy_result,
                                     trial_index=trial_index,
                                     hamiltonian_representation_index=hamiltonian_representation_index,
                                     angles=np.array(angles),
                                     correlators=Cij_noiseless)

        t2 = time.perf_counter()

        self.log_results(qaoa_result=qaoa_result)

        t3 = time.perf_counter()
        total_dt0 = t3-t0_input

        # print("INPUT TOOK:", t1_input - t0_input)
        # print("COMPUTATION TOOK:",t1-t0,'iters per second:', 1/(t1-t0))
        # print("processing took:",t2-t1,'iters per second:', 1/(t2-t1))
        # print("logging took:",t3-t2,'iters per second:', 1/(t3-t2))
        # total_dt = time.perf_counter()-t0_input
        # print("TOTAL DT:",total_dt0)
        # print("TOTAL DT (with printing):",total_dt,"")


        return qaoa_result
