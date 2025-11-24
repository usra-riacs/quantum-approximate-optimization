# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import Dict, List, Optional

import numpy as np
from tqdm.notebook import tqdm

from quapopt import AVAILABLE_SIMULATORS
from quapopt import ancillary_functions as anf
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)
from quapopt.optimization.QAOA.circuits.time_block_ansatz import (
    divide_hamiltonian_into_batches,
)
from quapopt.optimization.QAOA.simulation.direct.cython_implementation.cython_qaoa_statevector_simulator import (
    apply_full_qaoa_circuit_cython,
)

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp

try:
    import numba

    NUMBA_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    NUMBA_AVAILABLE = False


def get_exp_X_operator(angle, backend="numpy"):
    """

    :param angle:
    :param backend:
    :return:
    """
    _bck = cp if backend == "cupy" else np
    return _bck.array(
        [
            [_bck.cos(angle), -1j * _bck.sin(angle)],
            [-1j * _bck.sin(angle), _bck.cos(angle)],
        ]
    )


def get_mixer_operator(angle_mixer, number_of_qubits, backend="numpy"):
    """

    :param angle_mixer:
    :param number_of_qubits:
    :param backend:
    :return:
    """

    _bck = cp if backend == "cupy" else np
    _1q_mixer = get_exp_X_operator(angle_mixer, backend=backend)

    big_mixer = _1q_mixer.copy()
    for _ in range(1, number_of_qubits):
        big_mixer = _bck.kron(big_mixer, _1q_mixer)

    return big_mixer


def multiply_by_mixer_operator(
    angle_mixer,
    number_of_qubits,
    input_state,
    backend="numpy",
):
    """
    Apply mixer operator to statevector without storing full matrix.

    Exploits tensor product structure: U_mixer = exp(-i*angle*X)^⊗n
    Instead of building 2^n × 2^n matrix, applies single-qubit operations sequentially.

    :param angle_mixer: Mixer angle β
    :param number_of_qubits: Number of qubits
    :param input_state: Input statevector of shape (2^n,)
    :param backend: 'numpy' or 'cupy'
    :return: Statevector after mixer application
    """
    _bck = cp if backend == "cupy" else np

    cos_beta, sin_beta = _bck.cos(angle_mixer), _bck.sin(angle_mixer)

    # Reshape statevector to tensor form: (2, 2, ..., 2) with n indices
    state = input_state.reshape([2] * number_of_qubits)

    # Apply single-qubit mixer to each qubit index
    for qubit_idx in range(number_of_qubits):
        # Move qubit_idx axis to position 0
        state = _bck.moveaxis(state, qubit_idx, 0)

        # Apply exp(-i*β*X) = [[cos(β), -i*sin(β)], [-i*sin(β), cos(β)]]
        state[0], state[1] = (
            cos_beta * state[0] - 1j * sin_beta * state[1],
            -1j * sin_beta * state[0] + cos_beta * state[1],
        )

        # Move axis back
        state = _bck.moveaxis(state, 0, qubit_idx)

    # Flatten back to vector form
    return state.reshape(2**number_of_qubits)


# Numba-optimized CPU implementation
if NUMBA_AVAILABLE:

    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _apply_full_qaoa_circuit_numba(
        input_state, angles_PS, angles_mixer, spectra_list, number_of_qubits
    ):
        """
        Complete QAOA circuit execution in a single Numba-compiled function.

        This function applies the entire QAOA circuit (all layers) without returning to Python.
        Eliminates Python loop overhead and function call overhead between layers.

        :param input_state: Initial statevector of shape (2^n,) - will be modified in-place
        :param angles_PS: Array of phase separation angles γ, shape (depth,)
        :param angles_mixer: Array of mixer angles β, shape (depth,)
        :param spectra_list: List of eigenvalue spectra, one per batch (handles time blocking)
        :param number_of_qubits: Number of qubits
        :return: Final statevector after all QAOA layers
        """
        depth = len(angles_PS)
        number_of_batches = len(spectra_list)
        total_dim = 2**number_of_qubits

        for layer_idx in range(depth):
            angle_PS = angles_PS[layer_idx]
            angle_mixer = angles_mixer[layer_idx]

            # Select spectrum for this layer (time blocking)
            batch_idx = layer_idx % number_of_batches
            spectrum = spectra_list[batch_idx]

            # Phase separation: apply exp(-i*γ*H)
            for i in range(total_dim):
                input_state[i] = input_state[i] * np.exp(-1j * angle_PS * spectrum[i])

            # Mixer operator: apply exp(-i*β*X)^⊗n
            cos_beta = np.cos(angle_mixer)
            sin_beta = np.sin(angle_mixer)

            for qubit_idx in range(number_of_qubits):
                stride = 2**qubit_idx
                block_size = 2 * stride

                for block_start in range(0, total_dim, block_size):
                    for i in range(stride):
                        idx_0 = block_start + i
                        idx_1 = block_start + i + stride

                        state_0 = input_state[idx_0]
                        state_1 = input_state[idx_1]

                        input_state[idx_0] = (
                            cos_beta * state_0 - 1j * sin_beta * state_1
                        )
                        input_state[idx_1] = (
                            -1j * sin_beta * state_0 + cos_beta * state_1
                        )

        return input_state

    def apply_full_qaoa_circuit_numba(
        input_state, angles_PS, angles_mixer, batches_spectra, number_of_qubits
    ):
        """
        Wrapper for full QAOA circuit execution with Numba.

        :param input_state: Initial statevector of shape (2^n,)
        :param angles_PS: Array of phase separation angles, shape (depth,)
        :param angles_mixer: Array of mixer angles, shape (depth,)
        :param batches_spectra: Dictionary or list of spectra arrays for time blocking
        :param number_of_qubits: Number of qubits
        :return: Final statevector after all QAOA layers
        """
        # Convert dict to list if needed
        if isinstance(batches_spectra, dict):
            spectra_list = [batches_spectra[i] for i in range(len(batches_spectra))]
        else:
            spectra_list = list(batches_spectra)
        return _apply_full_qaoa_circuit_numba(
            input_state,
            np.array(angles_PS, dtype=np.float64),
            np.array(angles_mixer, dtype=np.float64),
            spectra_list,
            number_of_qubits,
        )


class QAOASimulatorPython:
    """
    Basic QAOA simulator. It is not optimized, the main purpose is to provide a reference implementation.

    :param hamiltonian_phase: Phase Hamiltonian to be implemented
    :param backend: Backend to use for simulation. Choose from 'numpy' or 'cupy'.
    :param time_block_size: Number of linear chains per QAOA layer
    :param time_block_seed: Seed for shuffling the Hamiltonian terms
    :param time_block_partition: Dictionary specifying the partition of the Hamiltonian terms into time blocks.


    """

    def __init__(
        self,
        hamiltonian_phase: ClassicalHamiltonian,
        time_block_size: Optional[float] = None,
        time_block_seed: int = 0,
        time_block_type: str = "fractional",
        time_block_partition: Optional[Dict[int, ClassicalHamiltonian]] = None,
        backend: Optional[str] = "auto",
    ):

        # we want to precompute spectrum
        self._number_of_qubits = hamiltonian_phase.number_of_qubits
        self._dimension = 2**self._number_of_qubits

        if backend in [None, "auto"]:
            if self._number_of_qubits <= 16:
                backend = "cython"
            else:
                if "cupy" in AVAILABLE_SIMULATORS:
                    backend = "cupy"
                else:
                    backend = "cython"

        self.backend_name: str = backend

        if backend == "numba" and not NUMBA_AVAILABLE:
            raise ImportError("Numba package not available. Please change backend.")

        if self.backend_name in ["numpy", "numba", "cython"]:
            self._bck = np
        elif self.backend_name == "cupy":
            self._bck = cp
        else:
            raise ValueError(
                f"Backend {self.backend_name} not recognised. Choose from 'numpy' or 'cupy'."
            )

        self._hamiltonian_phase = hamiltonian_phase

        if time_block_partition is None:
            time_block_partition = divide_hamiltonian_into_batches(
                hamiltonian=hamiltonian_phase,
                time_block_size=time_block_size,
                batching_type=time_block_type,
                seed=time_block_seed,
            )

        self._time_block_partition = time_block_partition

        self._batches_spectra: List[np.ndarray | cp.ndarray] = [None] * len(
            time_block_partition
        )

    @property
    def hamiltonian_phase(self):
        return self._hamiltonian_phase

    @property
    def batches_spectra(self):
        return self._batches_spectra

    def update_batches_spectra(self, spectrum: np.ndarray | cp.ndarray, index: int):

        self._batches_spectra[index] = spectrum

    def solve_hamiltonian(
        self, hamiltonian: ClassicalHamiltonian, solving_backend: str = None
    ):
        if solving_backend is None:
            if "cuda" in AVAILABLE_SIMULATORS:
                solving_backend = "cuda"
            else:
                solving_backend = "python"

        if solving_backend == "cuda":
            spectrum = anf.cuda_solve_hamiltonian(hamiltonian)
        else:
            spectrum = anf.solve_hamiltonian_python(hamiltonian)

        return spectrum

    def _update_spectra(self, depth: int, solving_backend: str = None):

        number_of_batches = len(self._time_block_partition)

        how_many_spectra = min([number_of_batches, depth])
        for batch_index in range(how_many_spectra):
            if self._batches_spectra[batch_index] is None:
                spectrum = self.solve_hamiltonian(
                    hamiltonian=self._time_block_partition[batch_index],
                    solving_backend=solving_backend,
                )

                spectrum = anf.convert_cupy_numpy_array(
                    array=spectrum, output_backend=self._bck.__name__
                )

                # Ensure correct dtype for compiled backends (Cython requires float64)
                if self.backend_name in ["cython", "numba"]:
                    spectrum = spectrum.astype(np.float64, copy=False)

                assert spectrum.shape[0] == 2**self._number_of_qubits

                self.update_batches_spectra(spectrum=spectrum, index=batch_index)

    def get_qaoa_unitary(
        self,
        angles_PS: List[float],
        angles_mixer: List[float],
        show_progress_bar: bool = False,
    ):

        # TODO(FBM): add cython implementation of this

        backend = self.backend_name
        _bck = self._bck

        self._update_spectra(
            depth=len(angles_PS),
        )

        number_of_batches = len(self._time_block_partition)
        unitary_qaoa = _bck.diag(-1j * angles_PS[0] * self._batches_spectra[0])
        unitary_qaoa = (
            get_mixer_operator(
                angle_mixer=angles_mixer[0],
                number_of_qubits=self._number_of_qubits,
                backend=backend,
            )
            @ unitary_qaoa
        )

        for layer_index, (angle_PS, angle_mixer) in tqdm(
            enumerate(list(zip(angles_PS[1:], angles_mixer[1:]))),
            disable=not show_progress_bar,
        ):
            batch_index = (layer_index + 1) % number_of_batches
            unitary_qaoa = (
                _bck.diag(-1j * angle_PS[0] * self._batches_spectra[batch_index])
                @ unitary_qaoa
            )
            unitary_qaoa = (
                get_mixer_operator(
                    angle_mixer=angle_mixer,
                    number_of_qubits=self._number_of_qubits,
                    backend=backend,
                )
                @ unitary_qaoa
            )

        return unitary_qaoa

    def get_qaoa_statevector(
        self,
        angles_PS: List[float],
        angles_mixer: List[float],
        input_state: Optional[np.ndarray | cp.ndarray] = None,
        show_progress_bar: bool = False,
    ):

        backend = self.backend_name

        _bck = self._bck
        norm = None
        if input_state is None:
            input_state = self._bck.ones(self._dimension, dtype=_bck.complex64)
            norm = 1.0 / self._bck.sqrt(self._dimension)
        else:
            input_state = input_state.copy()

        self._update_spectra(
            depth=len(angles_PS),
        )

        number_of_batches = len(self._time_block_partition)

        input_state = anf.convert_cupy_numpy_array(
            array=input_state, output_backend=self._bck.__name__
        )

        # Use optimized compiled circuit implementations
        if backend == "cython":
            # Ensure correct dtypes for Cython (complex128 for state, float64 for angles)
            input_state = apply_full_qaoa_circuit_cython(
                input_state=input_state.astype(np.complex128, copy=False),
                angles_PS=np.array(angles_PS, dtype=np.float64),
                angles_mixer=np.array(angles_mixer, dtype=np.float64),
                spectra_list=self._batches_spectra,  # Already converted to float64 in _update_spectra
                number_of_qubits=self._number_of_qubits,
            )
        elif backend == "numba":
            input_state = apply_full_qaoa_circuit_numba(
                input_state=input_state,
                angles_PS=angles_PS,
                angles_mixer=angles_mixer,
                batches_spectra=self._batches_spectra,
                number_of_qubits=self._number_of_qubits,
            )
        else:
            # Layer-by-layer execution for other backends
            for layer_index, (angle_PS, angle_mixer) in tqdm(
                enumerate(list(zip(angles_PS, angles_mixer))),
                disable=not show_progress_bar,
            ):
                # batches reset after number_of_batches, so
                batch_index = layer_index % number_of_batches

                # Phase separation
                input_state = input_state * _bck.exp(
                    -1j * angle_PS * self._batches_spectra[batch_index]
                )

                # Mixer
                input_state = multiply_by_mixer_operator(
                    angle_mixer=angle_mixer,
                    number_of_qubits=self._number_of_qubits,
                    input_state=input_state,
                    backend=backend,
                )

        if norm is not None:
            input_state = norm * input_state

        return input_state
