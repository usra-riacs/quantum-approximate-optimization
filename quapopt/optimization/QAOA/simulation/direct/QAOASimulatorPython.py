# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from typing import List, Optional, Dict

import numpy as np
from tqdm.notebook import tqdm

from quapopt import AVAILABLE_SIMULATORS
from quapopt import ancillary_functions as anf
from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA.circuits.time_block_ansatz import divide_hamiltonian_into_batches, TimeBlockBatchingType
from quapopt.optimization.QAOA.simulation.direct.cupy_kernels import apply_phase_separator_inplace, abs_squared_float64
from quapopt.optimization.QAOA.simulation.direct.cython_implementation.cython_qaoa_statevector_simulator import \
    apply_full_qaoa_circuit_cython, apply_full_qaoa_circuit_cython_WS
from quapopt.optimization.QAOA.simulation.qaoa_math import (get_mixer_operator,
                                                            multiply_by_mixer_operator,
                                                            multiply_by_mixer_operator_WS,
                                                            get_initial_state_WS_qaoa)

try:
    import cupy as cp
except(ModuleNotFoundError, ImportError):
    import numpy as cp

from quapopt.ancillary_functions import get_correlators_from_probability_distribution
from quapopt.precision import Precision, resolve_precision

# The smallest n at which the cupy backend beats cython on this machine, per precision
# (tests/misc/benchmark_precision_policy.py, crossover sweep); below it 'auto' is cython.
# Measured on an RTX 3060 Laptop GPU: the p = 1 call at n = 14 took 0.65 ms on cupy against
# 1.59 ms on cython ('single') and 0.66 against 1.42 ms ('double'); at n = 12 cupy was slower.
AUTO_CUPY_THRESHOLD = {'single': 14, 'double': 14}


class QAOASimulatorPython:
    """
    Basic QAOA simulator. It is not optimized, the main purpose is to provide a reference implementation.

    :param hamiltonian_phase: Phase Hamiltonian to be implemented
    :param backend: Backend to use for simulation. Choose from 'numpy', 'cupy', 'cython' or 'auto'.
    :param time_block_size: Number of linear chains per QAOA layer
    :param time_block_seed: Seed for shuffling the Hamiltonian terms
    :param time_block_partition: Dictionary specifying the partition of the Hamiltonian terms into time blocks.
    :param precision: 'single', 'double' or None for the package default at construction.

    Precision. `precision` ('single' = complex64 statevectors, 'double' = complex128, None = the
    package default at construction) is honored by all three backends: the returned statevector
    and the cached initial state carry its complex dtype; spectra are float64 under both settings.
    'auto' picks cython below AUTO_CUPY_THRESHOLD[precision] qubits and cupy at or above it when
    cupy is present, cython for every size otherwise; numpy is the reference branch and is never
    picked automatically. A caller-supplied input_state of another dtype is cast to the policy's
    dtype once, at entry.
    """

    def __init__(self,
                 hamiltonian_phase: ClassicalHamiltonian,
                 time_block_size: Optional[float] = None,
                 time_block_seed: Optional[int] = -1,
                 time_block_batching_type: TimeBlockBatchingType = TimeBlockBatchingType.FRACTIONAL,
                 time_block_partition: Optional[Dict[int, ClassicalHamiltonian]] = None,
                 backend: Optional[str] = 'auto',
                 input_state_always_the_same=True,
                 precision: Optional[str | Precision] = None
                 ):

        self.precision = resolve_precision(precision)
        self.complex_dtype = self.precision.complex_dtype

        # we want to precompute spectrum
        self._number_of_qubits = hamiltonian_phase.number_of_qubits
        self._dimension = 2 ** self._number_of_qubits

        if isinstance(backend, str):
            backend = backend.lower()

        if backend in [None, 'auto']:
            if 'cupy' in AVAILABLE_SIMULATORS and self._number_of_qubits >= AUTO_CUPY_THRESHOLD[self.precision.name]:
                backend = 'cupy'
            else:
                backend = 'cython'

        assert backend.lower() in ['numpy', 'cupy',
                                   'cython'], f"Backend {backend} not recognised. Choose from ['numpy', 'cupy', 'cython']."

        self.backend_name: str = backend.lower()

        if self.backend_name in ['numpy', 'cython']:
            self._bck = np
        elif self.backend_name == 'cupy':
            self._bck = cp
        else:
            raise ValueError(f"Backend {self.backend_name} not recognised. Choose from 'numpy' or 'cupy'.")

        self._hamiltonian_phase = hamiltonian_phase

        if time_block_partition is None:
            time_block_partition = divide_hamiltonian_into_batches(hamiltonian=hamiltonian_phase,
                                                                   time_block_size=time_block_size,
                                                                   batching_type=time_block_batching_type,
                                                                   time_block_seed=time_block_seed)

        self._time_block_partition = time_block_partition

        self._batches_spectra: List[np.ndarray | cp.ndarray] = [None] * len(time_block_partition)

        self._hamiltonian_spectrum = None

        self.input_state_always_the_same = input_state_always_the_same

        self.fixed_input_state = None
        # The bias the cached initial state was constructed from (see _get_cached_input_state).
        self._fixed_input_state_bias_key = None

    @property
    def hamiltonian_phase(self):
        return self._hamiltonian_phase

    @property
    def batches_spectra(self):
        return self._batches_spectra

    def update_batches_spectra(self,
                               spectrum: np.ndarray | cp.ndarray,
                               index: int):
        """Store a spectrum as a float64 array of this simulator's backend: a host array
        handed to a cupy simulator is moved to the device, and a spectrum of another dtype
        is widened here, once."""

        self._batches_spectra[index] = self._bck.asarray(spectrum, dtype=self._bck.float64)

    def solve_hamiltonian(self,
                          hamiltonian: ClassicalHamiltonian,
                          solving_backend: str = None):
        if solving_backend is None:
            if 'cuda' in AVAILABLE_SIMULATORS:
                solving_backend = 'cuda'
            else:
                solving_backend = 'python'

        # anf.cool_print("SOLVING HAMILTONIAN", '...','green')

        if solving_backend == 'cuda':
            spectrum = anf.cuda_solve_hamiltonian(hamiltonian)
        else:
            spectrum = anf.solve_hamiltonian_python(hamiltonian)

        return spectrum

    def _update_spectra(self,
                        depth: int,
                        solving_backend: str = None
                        ):

        number_of_batches = len(self._time_block_partition)

        how_many_spectra = min([number_of_batches, depth])
        for batch_index in range(how_many_spectra):
            if self._batches_spectra[batch_index] is None:
                spectrum = self.solve_hamiltonian(hamiltonian=self._time_block_partition[batch_index],
                                                  solving_backend=solving_backend)

                spectrum = anf.convert_cupy_numpy_array(array=spectrum,
                                                        output_backend=self._bck.__name__)

                # Spectra are float64 under both precisions: every consumer is a sum or a phase argument.
                spectrum = spectrum.astype(self._bck.float64, copy=False)

                # print(spectrum.shape,self.hamiltonian_phase.number_of_qubits**2,self.hamiltonian_phase.number_of_qubits)
                assert spectrum.shape[0] == 2 ** self._number_of_qubits

                self.update_batches_spectra(spectrum=spectrum,
                                            index=batch_index)

    def _spectra_per_layer(self, depth: int) -> List[np.ndarray]:
        """One spectrum per layer, cycling through the time-block batches.

        The cython kernels read one spectrum per layer from the list they are given,
        so a list shorter than the depth is read past its end.
        """
        number_of_batches = len(self._time_block_partition)
        return [self._batches_spectra[layer_index % number_of_batches] for layer_index in range(depth)]

    def get_qaoa_unitary(self,
                         angles_PS: List[float],
                         angles_mixer: List[float],
                         show_progress_bar: bool = False):

        # TODO(FBM): add cython implementation of this

        backend = self.backend_name
        _bck = self._bck

        self._update_spectra(depth=len(angles_PS), )

        number_of_batches = len(self._time_block_partition)
        unitary_qaoa = _bck.diag(-1j * angles_PS[0] * self._batches_spectra[0])
        unitary_qaoa = get_mixer_operator(angle_mixer=angles_mixer[0],
                                          number_of_qubits=self._number_of_qubits,
                                          backend=backend) @ unitary_qaoa

        for layer_index, (angle_PS, angle_mixer) in tqdm(enumerate(list(zip(angles_PS[1:], angles_mixer[1:]))),
                                                         disable=not show_progress_bar):
            batch_index = (layer_index + 1) % number_of_batches
            unitary_qaoa = _bck.diag(-1j * angle_PS[0] * self._batches_spectra[batch_index]) @ unitary_qaoa
            unitary_qaoa = get_mixer_operator(angle_mixer=angle_mixer,
                                              number_of_qubits=self._number_of_qubits,
                                              backend=backend) @ unitary_qaoa

        return unitary_qaoa

    def get_standard_input_state(self,
                                 bias_parameters_WS: Optional[float | List[float]] = None):

        return get_initial_state_WS_qaoa(number_of_qubits=self._number_of_qubits,
                                         bias_parameters_WS=bias_parameters_WS,
                                         backend='cupy' if self.backend_name == 'cupy' else 'numpy',
                                         dtype=self.complex_dtype)

    @staticmethod
    def _bias_cache_key(bias_parameters_WS: Optional[float | List[float] | np.ndarray | cp.ndarray]):
        """Hashable identity of a warm-start bias: None, a float, or a tuple of floats.

        A scalar and a one-element sequence are different keys because they construct
        different states (identical bias on every qubit vs. a per-qubit list).
        """
        if bias_parameters_WS is None:
            return None
        if isinstance(bias_parameters_WS, (float, int)):
            return float(bias_parameters_WS)
        bias_host = anf.convert_cupy_numpy_array(array=bias_parameters_WS, output_backend='numpy')
        return tuple(float(x) for x in np.asarray(bias_host, dtype=np.float64).ravel())

    def _get_cached_input_state(self,
                                bias_parameters_WS: Optional[float | List[float] | np.ndarray | cp.ndarray] = None):
        """The standard initial state for this bias, cached across calls.

        The cache is valid only for the bias it was constructed from: a call with a
        different bias replaces it. Callers copy the returned array before modifying it.
        """
        bias_key = self._bias_cache_key(bias_parameters_WS)
        if self.fixed_input_state is None or bias_key != self._fixed_input_state_bias_key:
            self.fixed_input_state = self.get_standard_input_state(bias_parameters_WS=bias_parameters_WS)
            self._fixed_input_state_bias_key = bias_key
        return self.fixed_input_state

    def _get_qaoa_statevector_vanilla(self,
                                      angles_PS: List[float],
                                      angles_mixer: List[float],
                                      input_state: Optional[np.ndarray | cp.ndarray] = None,
                                      show_progress_bar: bool = False):

        backend = self.backend_name

        _bck = self._bck
        if input_state is None:
            if self.input_state_always_the_same:
                input_state = self._get_cached_input_state()
            else:
                input_state = self.get_standard_input_state()

        number_of_batches = len(self._time_block_partition)

        # The one cast to the policy's dtype. astype copies, so neither the caller's array nor
        # the cached input state is ever modified.
        input_state = anf.convert_cupy_numpy_array(array=input_state,
                                                   output_backend=self._bck.__name__).astype(self.complex_dtype)

        # Use optimized compiled circuit implementations
        if backend == 'cython':
            # The kernels take complex64 or complex128 states; angles and spectra are float64.
            input_state = apply_full_qaoa_circuit_cython(
                input_state=input_state,
                angles_PS=np.array(angles_PS, dtype=np.float64),
                angles_mixer=np.array(angles_mixer, dtype=np.float64),
                spectra_list=self._spectra_per_layer(depth=len(angles_PS)),
                number_of_qubits=self._number_of_qubits
            )
        elif backend == 'cupy' and 'cupy' in AVAILABLE_SIMULATORS:
            # The kernels work in place on the policy's dtype.
            input_state = cp.ascontiguousarray(input_state)
            angles_PS = [float(a) for a in cp.asnumpy(angles_PS)] if isinstance(angles_PS, cp.ndarray) else angles_PS
            angles_mixer = [float(a) for a in cp.asnumpy(angles_mixer)] if isinstance(angles_mixer, cp.ndarray) else angles_mixer
            for layer_index, (angle_PS, angle_mixer) in tqdm(enumerate(list(zip(angles_PS, angles_mixer))),
                                                             disable=not show_progress_bar):
                batch_index = layer_index % number_of_batches
                apply_phase_separator_inplace(input_state, angle_PS, self._batches_spectra[batch_index])
                input_state = multiply_by_mixer_operator(angle_mixer=angle_mixer,
                                                         number_of_qubits=self._number_of_qubits,
                                                         input_state=input_state,
                                                         backend='cupy')
        else:
            # Layer-by-layer execution for other backends
            for layer_index, (angle_PS, angle_mixer) in tqdm(enumerate(list(zip(angles_PS, angles_mixer))),
                                                             disable=not show_progress_bar):
                # batches reset after number_of_batches, so
                batch_index = layer_index % number_of_batches

                # Phase separation: the product is formed in complex128 and stored at the state's dtype.
                _bck.multiply(input_state, _bck.exp(-1j * float(angle_PS) * self._batches_spectra[batch_index]),
                              out=input_state)

                # Mixer
                input_state = multiply_by_mixer_operator(angle_mixer=angle_mixer,
                                                         number_of_qubits=self._number_of_qubits,
                                                         input_state=input_state,
                                                         backend=backend)

        return input_state

    def _get_qaoa_statevector_WS(self,
                                 angles_PS: List[float],
                                 angles_mixer: List[float],
                                 bias_parameteres_WS: float | List[float],
                                 input_state: Optional[np.ndarray | cp.ndarray] = None,
                                 show_progress_bar: bool = False):

        backend = self.backend_name

        _bck = self._bck

        # The initial state is constructed from the bias as given: a scalar means the same
        # bias on every qubit, which the one-element array below would not express.
        bias_as_given = bias_parameteres_WS
        _identical_bias = False
        if isinstance(bias_parameteres_WS, float):
            _identical_bias = True
            bias_parameteres_WS = [bias_parameteres_WS]

        bias_parameteres_WS = _bck.array(bias_parameteres_WS, dtype=_bck.float64)
        if input_state is None:
            if self.input_state_always_the_same:
                input_state = self._get_cached_input_state(bias_parameters_WS=bias_as_given)
            else:
                input_state = self.get_standard_input_state(bias_parameters_WS=bias_as_given)

        number_of_batches = len(self._time_block_partition)

        # The one cast to the policy's dtype. astype copies, so neither the caller's array nor
        # the cached input state is ever modified.
        input_state = anf.convert_cupy_numpy_array(array=input_state,
                                                   output_backend=self._bck.__name__).astype(self.complex_dtype)

        use_cupy_kernels = backend == 'cupy' and 'cupy' in AVAILABLE_SIMULATORS

        if use_cupy_kernels:
            # The kernels read the mixer terms on the host, one scalar per qubit. Deriving them
            # on the device instead launches a kernel per term and then reads each one back,
            # which costs more than the mixer layer they feed.
            bias_host = np.asarray(anf.convert_cupy_numpy_array(array=bias_parameteres_WS,
                                                                output_backend='numpy'),
                                   dtype=np.float64)
            if _identical_bias:
                c = float(bias_host[0])
                XZ_terms = (2.0 * float(np.sqrt(c * (1.0 - c))), 1.0 - 2.0 * c)
            else:
                XZ_terms = [(2.0 * float(np.sqrt(c * (1.0 - c))), 1.0 - 2.0 * c) for c in bias_host]
        elif _identical_bias:
            c = bias_parameteres_WS[0]
            XZ_terms = (2 * _bck.sqrt(c * (1 - c)), 1 - 2 * c)
        else:
            Z_terms = 1 - 2 * bias_parameteres_WS
            X_terms = 2 * _bck.sqrt(bias_parameteres_WS * (1 - bias_parameteres_WS))
            XZ_terms = [(x, z) for x, z in zip(X_terms, Z_terms)]

        # Use optimized compiled circuit implementations
        if backend == 'cython':
            # The kernels take complex64 or complex128 states; angles and spectra are float64.
            # For Cython, XZ_terms must be a 2D array of shape (1, 2) or (n_qubits, 2)
            if _identical_bias:
                XZ_terms_array = np.array([[XZ_terms[0], XZ_terms[1]]], dtype=np.float64)
            else:
                XZ_terms_array = np.array(XZ_terms, dtype=np.float64)
            input_state = apply_full_qaoa_circuit_cython_WS(
                input_state=input_state,
                angles_PS=np.array(angles_PS, dtype=np.float64),
                angles_mixer=np.array(angles_mixer, dtype=np.float64),
                XZ_terms=XZ_terms_array,
                spectra_list=self._spectra_per_layer(depth=len(angles_PS)),
                number_of_qubits=self._number_of_qubits
            )
        elif use_cupy_kernels:
            # XZ_terms are already host floats here, which is what the kernels need: reading a
            # 0-d cupy array costs a device sync, once per qubit and per layer.
            input_state = cp.ascontiguousarray(input_state)
            angles_PS = [float(a) for a in cp.asnumpy(angles_PS)] if isinstance(angles_PS, cp.ndarray) else angles_PS
            angles_mixer = [float(a) for a in cp.asnumpy(angles_mixer)] if isinstance(angles_mixer, cp.ndarray) else angles_mixer
            for layer_index, (angle_PS, angle_mixer) in tqdm(enumerate(list(zip(angles_PS, angles_mixer))),
                                                             disable=not show_progress_bar):
                batch_index = layer_index % number_of_batches
                apply_phase_separator_inplace(input_state, angle_PS, self._batches_spectra[batch_index])
                input_state = multiply_by_mixer_operator_WS(angle_mixer=angle_mixer,
                                                            number_of_qubits=self._number_of_qubits,
                                                            input_state=input_state,
                                                            XZ_terms=XZ_terms,
                                                            backend='cupy')
        else:
            # Layer-by-layer execution for other backends
            for layer_index, (angle_PS, angle_mixer) in tqdm(enumerate(list(zip(angles_PS, angles_mixer))),
                                                             disable=not show_progress_bar):
                # batches reset after number_of_batches, so
                batch_index = layer_index % number_of_batches

                # Phase separation: the product is formed in complex128 and stored at the state's dtype.
                _bck.multiply(input_state, _bck.exp(-1j * float(angle_PS) * self._batches_spectra[batch_index]),
                              out=input_state)

                # Mixer
                input_state = multiply_by_mixer_operator_WS(angle_mixer=angle_mixer,
                                                            number_of_qubits=self._number_of_qubits,
                                                            input_state=input_state,
                                                            XZ_terms=XZ_terms,
                                                            backend=backend)

        return input_state

    def get_qaoa_statevector(self,
                             angles_PS: List[float] | np.ndarray | cp.ndarray,
                             angles_mixer: List[float] | np.ndarray | cp.ndarray,
                             bias_parameters_WS: Optional[float | List[float]] = None,
                             input_state: Optional[np.ndarray | cp.ndarray] = None,
                             show_progress_bar: bool = False) -> np.ndarray | cp.ndarray:
        """The QAOA statevector at the given angles, in the simulator's complex dtype.

        A caller-supplied `input_state` of another dtype is cast to the policy's dtype once, at
        entry: under 'single' a complex128 input is rounded to complex64 there. The caller's
        array is never modified."""

        self._update_spectra(depth=len(angles_PS))

        if input_state is None and self.input_state_always_the_same:
            input_state = self._get_cached_input_state(bias_parameters_WS=bias_parameters_WS)

        if bias_parameters_WS is None:
            return self._get_qaoa_statevector_vanilla(angles_PS=angles_PS,
                                                      angles_mixer=angles_mixer,
                                                      input_state=input_state,
                                                      show_progress_bar=show_progress_bar)
        else:
            return self._get_qaoa_statevector_WS(angles_PS=angles_PS,
                                                 angles_mixer=angles_mixer,
                                                 bias_parameteres_WS=bias_parameters_WS,
                                                 input_state=input_state,
                                                 show_progress_bar=show_progress_bar)

    def _probabilities_float64(self, quantum_state: np.ndarray | cp.ndarray):
        """|amplitude|^2 as float64 on this simulator's backend, each square formed in double,
        so a probability-weighted sum never accumulates float32 terms under 'single'."""
        if self._bck is np:
            return em.cython_abs_squared(np.ascontiguousarray(quantum_state).reshape(-1),
                                         output_precision_if_complex=np.float64)
        return abs_squared_float64(cp.ascontiguousarray(quantum_state))

    def get_exp_value(self,
                      quantum_state: np.ndarray | cp.ndarray,
                      ):

        prob_distro = self._probabilities_float64(quantum_state)

        if self._hamiltonian_spectrum is None:
            if self.hamiltonian_phase.spectrum is None:
                self.hamiltonian_phase.solve_hamiltonian()
            self._hamiltonian_spectrum = self.hamiltonian_phase.spectrum

        spectrum = self._bck.array(self._hamiltonian_spectrum)

        exp_value = self._bck.sum(prob_distro * spectrum)

        return exp_value

    def get_correlators_from_probability_distribution(self,
                                                      probability_distribution: np.ndarray | cp.ndarray,
                                                      cost_hamiltonian: Optional[ClassicalHamiltonian] = None
                                                      ):

        _bck = self._bck

        if cost_hamiltonian is None:
            cost_hamiltonian = self._hamiltonian_phase

        correlators = get_correlators_from_probability_distribution(probability_distribution=probability_distribution,
                                                                    cost_hamiltonian=cost_hamiltonian)

        if isinstance(correlators, np.ndarray):
            pass
        else:
            correlators = cp.asnumpy(correlators)

        return correlators

    def get_correlators_from_statevector(self,
                                         quantum_state: np.ndarray | cp.ndarray,
                                         cost_hamiltonian: Optional[ClassicalHamiltonian] = None
                                         ):

        probability_distribution = self._probabilities_float64(quantum_state)

        return self.get_correlators_from_probability_distribution(probability_distribution=probability_distribution,
                                                                  cost_hamiltonian=cost_hamiltonian)
