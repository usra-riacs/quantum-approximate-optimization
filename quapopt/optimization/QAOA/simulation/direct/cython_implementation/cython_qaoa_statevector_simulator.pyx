# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# distutils: language = c++

"""
Cython-optimized QAOA statevector simulator.
Copyright 2025 USRA
Authors: Filip B. Maciejewski
Use, duplication, or disclosure without authors' permission is strictly prohibited.
"""

import numpy as np
cimport numpy as np
from libc.math cimport cos, sin, exp
from libc.stdlib cimport malloc, free

# Type definitions for clarity
ctypedef double complex complex128
ctypedef np.complex128_t COMPLEX_t
ctypedef np.float64_t FLOAT_t


cdef void apply_phase_separation_inplace(
    complex128* state,
    double angle_PS,
    const double* spectrum,
    int dim
) noexcept nogil:
    """
    Apply phase separation operator in-place: state *= exp(-i * gamma * spectrum).

    This is a diagonal operator in the computational basis.

    :param state: Statevector array (modified in-place)
    :param angle_PS: Phase separation angle gamma
    :param spectrum: Eigenvalue spectrum of phase Hamiltonian
    :param dim: Dimension of the Hilbert space (2^n)
    """
    cdef int i
    cdef double phase
    cdef double real_part, imag_part

    for i in range(dim):
        # Compute phase: -i * gamma * E_i = -i * angle_PS * spectrum[i]
        # exp(-i * x) = cos(x) - i*sin(x)
        phase = angle_PS * spectrum[i]

        # Multiply: state[i] *= exp(-i * phase)
        real_part = state[i].real * cos(phase) + state[i].imag * sin(phase)
        imag_part = state[i].imag * cos(phase) - state[i].real * sin(phase)

        state[i].real = real_part
        state[i].imag = imag_part


cdef void apply_mixer_operator_inplace(
    complex128* state,
    double angle_mixer,
    int number_of_qubits
) noexcept nogil:
    """
    Apply mixer operator in-place: state = exp(-i * beta * X^tensor_n) * state.
    Applies single-qubit X rotation to each qubit sequentially.

    :param state: Statevector array (modified in-place)
    :param angle_mixer: Mixer angle beta
    :param number_of_qubits: Number of qubits
    """
    cdef int qubit_idx, block_start, i, idx_0, idx_1
    cdef int stride, block_size
    cdef int total_dim = 1 << number_of_qubits  # 2^number_of_qubits
    cdef double cos_beta = cos(angle_mixer)
    cdef double sin_beta = sin(angle_mixer)
    cdef complex128 state_0, state_1
    cdef double real_0, imag_0, real_1, imag_1

    # Apply single-qubit X rotation to each qubit
    for qubit_idx in range(number_of_qubits):
        stride = 1 << qubit_idx  # 2^qubit_idx
        block_size = stride << 1  # 2 * stride

        # Iterate through blocks where this qubit alternates between |0> and |1>
        block_start = 0
        while block_start < total_dim:
            # Within each block, process pairs (|...0...>, |...1...>)
            for i in range(stride):
                idx_0 = block_start + i           # Index where qubit is |0>
                idx_1 = block_start + i + stride  # Index where qubit is |1>

                # Read current values
                state_0 = state[idx_0]
                state_1 = state[idx_1]

                # Apply rotation matrix [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
                real_0 = cos_beta * state_0.real + sin_beta * state_1.imag
                imag_0 = cos_beta * state_0.imag - sin_beta * state_1.real

                # state[1] = -i * sin_beta * state_0 + cos_beta * state_1
                real_1 = sin_beta * state_0.imag + cos_beta * state_1.real
                imag_1 = -sin_beta * state_0.real + cos_beta * state_1.imag

                # Write back
                state[idx_0].real = real_0
                state[idx_0].imag = imag_0
                state[idx_1].real = real_1
                state[idx_1].imag = imag_1

            block_start = block_start + block_size


cdef void apply_full_qaoa_circuit_inplace(
    complex128* state,
    const double* angles_PS,
    const double* angles_mixer,
    const double** spectra_list,
    int number_of_qubits,
    int depth,
    int number_of_batches
) noexcept nogil:
    """
    Apply complete QAOA circuit (all layers) in a single function.

    This eliminates all Python overhead by doing the entire depth loop in compiled C code.

    :param state: Initial statevector (modified in-place to final state)
    :param angles_PS: Array of phase separation angles, length = depth
    :param angles_mixer: Array of mixer angles, length = depth
    :param spectra_list: Array of pointers to spectra (for time blocking)
    :param number_of_qubits: Number of qubits
    :param depth: QAOA circuit depth
    :param number_of_batches: Number of time-blocking batches
    """
    cdef int layer_idx, batch_idx
    cdef int dim = 1 << number_of_qubits  # 2^number_of_qubits
    cdef const double* spectrum

    for layer_idx in range(depth):
        # Select spectrum for this layer (time blocking)
        batch_idx = layer_idx % number_of_batches
        spectrum = spectra_list[batch_idx]

        # Apply phase separation
        apply_phase_separation_inplace(
            state,
            angles_PS[layer_idx],
            spectrum,
            dim
        )

        # Apply mixer
        apply_mixer_operator_inplace(
            state,
            angles_mixer[layer_idx],
            number_of_qubits
        )


def apply_full_qaoa_circuit_cython(
    np.ndarray[COMPLEX_t, ndim=1] input_state,
    np.ndarray[FLOAT_t, ndim=1] angles_PS,
    np.ndarray[FLOAT_t, ndim=1] angles_mixer,
    list spectra_list,
    int number_of_qubits
):
    """
    Python-accessible wrapper for the full QAOA circuit implementation.

    This function provides a clean Python interface to the optimized C implementation.

    :param input_state: Initial statevector of shape (2^n,)
    :param angles_PS: Phase separation angles, shape (depth,)
    :param angles_mixer: Mixer angles, shape (depth,)
    :param spectra_list: List of numpy arrays containing spectra (for time blocking)
    :param number_of_qubits: Number of qubits
    :return: Final statevector after QAOA circuit
    """
    # Input validation and variable declarations (all cdef must be at top)
    cdef int depth = len(angles_PS)
    cdef int number_of_batches = len(spectra_list)
    cdef int dim = 1 << number_of_qubits
    cdef int i
    cdef np.ndarray[COMPLEX_t, ndim=1] output_state
    cdef np.ndarray[FLOAT_t, ndim=1] spectrum
    cdef double** spectra_ptrs
    cdef double* angles_PS_ptr
    cdef double* angles_mixer_ptr
    cdef complex128* state_ptr

    if len(angles_mixer) != depth:
        raise ValueError(f"angles_mixer length {len(angles_mixer)} != depth {depth}")

    if input_state.shape[0] != dim:
        raise ValueError(f"input_state dimension {input_state.shape[0]} != 2^{number_of_qubits} = {dim}")

    # Create output array (copy of input)
    output_state = input_state.copy()

    # Convert spectra list to C array of pointers
    spectra_ptrs = <double**>malloc(number_of_batches * sizeof(double*))

    if spectra_ptrs == NULL:
        raise MemoryError("Failed to allocate memory for spectra pointers")

    try:
        # Store pointers to spectrum data
        for i in range(number_of_batches):
            spectrum = spectra_list[i]
            if spectrum.shape[0] != dim:
                raise ValueError(f"Spectrum {i} has wrong dimension: {spectrum.shape[0]} != {dim}")
            spectra_ptrs[i] = <double*>spectrum.data

        # Get pointer to angles arrays
        angles_PS_ptr = <double*>angles_PS.data
        angles_mixer_ptr = <double*>angles_mixer.data
        state_ptr = <complex128*>output_state.data

        # Call the optimized C function (releases GIL!)
        with nogil:
            apply_full_qaoa_circuit_inplace(
                state_ptr,
                angles_PS_ptr,
                angles_mixer_ptr,
                <const double**>spectra_ptrs,
                number_of_qubits,
                depth,
                number_of_batches
            )

    finally:
        # Clean up
        free(spectra_ptrs)

    return output_state


def apply_phase_separation_cython(
    np.ndarray[COMPLEX_t, ndim=1] input_state,
    double angle_PS,
    np.ndarray[FLOAT_t, ndim=1] spectrum
):
    """
    Python wrapper for phase separation operation.

    :param input_state: Input statevector
    :param angle_PS: Phase separation angle
    :param spectrum: Eigenvalue spectrum
    :return: Statevector after phase separation
    """
    cdef int dim = input_state.shape[0]

    if spectrum.shape[0] != dim:
        raise ValueError(f"Spectrum dimension {spectrum.shape[0]} != state dimension {dim}")

    cdef np.ndarray[COMPLEX_t, ndim=1] output_state = input_state.copy()
    cdef complex128* state_ptr = <complex128*>output_state.data
    cdef double* spectrum_ptr = <double*>spectrum.data

    with nogil:
        apply_phase_separation_inplace(state_ptr, angle_PS, spectrum_ptr, dim)

    return output_state


def apply_mixer_operator_cython(
    np.ndarray[COMPLEX_t, ndim=1] input_state,
    double angle_mixer,
    int number_of_qubits
):
    """
    Python wrapper for mixer operator.

    :param input_state: Input statevector
    :param angle_mixer: Mixer angle
    :param number_of_qubits: Number of qubits
    :return: Statevector after mixer application
    """
    cdef int dim = 1 << number_of_qubits

    if input_state.shape[0] != dim:
        raise ValueError(f"State dimension {input_state.shape[0]} != 2^{number_of_qubits}")

    cdef np.ndarray[COMPLEX_t, ndim=1] output_state = input_state.copy()
    cdef complex128* state_ptr = <complex128*>output_state.data

    with nogil:
        apply_mixer_operator_inplace(state_ptr, angle_mixer, number_of_qubits)

    return output_state