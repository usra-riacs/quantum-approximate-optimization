# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt import AVAILABLE_SIMULATORS
from quapopt import ancillary_functions as anf
from typing import List, Optional, Dict, Tuple
from tqdm.notebook import tqdm
import numpy as np
from quapopt.optimization.QAOA.circuits.time_block_ansatz import divide_hamiltonian_into_batches, TimeBlockBatchingType
from quapopt.optimization.QAOA.simulation.direct.cython_implementation.cython_qaoa_statevector_simulator import (apply_full_qaoa_circuit_cython,
                                                                                                                 apply_full_qaoa_circuit_cython_WS)



try:
    import cupy as cp
except(ModuleNotFoundError,ImportError):
    import numpy as cp

from quapopt.optimization.QAOA.simulation.direct.cupy_kernels import apply_xz_rotations_inplace


def _as_cupy_contiguous(state):
    """The kernels work in place on a C-contiguous complex64 or complex128 cupy array."""
    if state.dtype not in (cp.complex64, cp.complex128):
        raise TypeError(f"the cupy mixer kernels take a complex64 or complex128 state, got {state.dtype}; "
                        f"cast with state.astype(cp.complex64) or cp.complex128 first")
    return cp.ascontiguousarray(state)



def get_exp_X_operator_1q(angle,
                          backend='numpy'):
    """

    :param angle:
    :param backend:
    :return:
    """
    _bck = cp if backend == 'cupy' else np
    cos_beta = _bck.cos(angle)
    sin_beta = -1j*_bck.sin(angle)

    return _bck.array([[cos_beta, sin_beta],
                       [sin_beta, cos_beta]]
                      )

def get_WS_mixer_operator_1q(angle,
                             term_Z:float,
                             term_X:float,
                             backend='numpy'):
    """

    :param angle:
    :param backend:
    :return:
    """
    _bck = cp if backend == 'cupy' else np

    cos_beta = _bck.cos(angle)
    sin_beta = -1j*_bck.sin(angle)

    # term_Z = (1 - 2 * bias_parameter)
    # term_X =2*_bck.sqrt((1 - bias_parameter) * bias_parameter)
    # term_off = sin_beta*term_X

    term_off = sin_beta*term_X
    term_diag = cos_beta+term_Z*sin_beta

    # WS mixer: exp(-i*β*(term_X*X + term_Z*Z))
    # Matrix: [[cos(β) - i*sin(β)*Z,   -i*sin(β)*2*X],
    #          [-i*sin(β)*2*X,          cos(β) + i*sin(β)*Z]]


    return _bck.array([[term_diag, term_off],
                       [term_off, _bck.conj(term_diag)]],
                      dtype=_bck.complex64
                      )



def get_mixer_operator_WS(angle_mixer,
                          XZ_terms: Tuple[float, float] | List[Tuple[float, float]],
                          number_of_qubits:int,
                          backend='numpy'):
    """

    :param angle_mixer:
    :param number_of_qubits:
    :param backend:
    :return:
    """
    _identical_bias = False
    if isinstance(XZ_terms,tuple) and len(XZ_terms)==2:
        _identical_bias = True
        XZ_terms = [XZ_terms]*number_of_qubits


    _bck = cp if backend == 'cupy' else np
    _1q_mixer = get_WS_mixer_operator_1q(angle=angle_mixer,
                                         term_X=XZ_terms[0][0],
                                         term_Z=XZ_terms[0][1],
                                         backend=backend)

    big_mixer = _1q_mixer.copy()
    for (term_X,term_Z) in XZ_terms[1:]:
        if _identical_bias:
            big_mixer = _bck.kron(big_mixer,
                                    _1q_mixer)
        else:
            big_mixer = _bck.kron(big_mixer,
                                  get_WS_mixer_operator_1q(angle=angle_mixer,
                                                           term_X=term_X,
                                                           term_Z=term_Z,
                                                           backend=backend))

    return big_mixer




def get_mixer_operator(angle_mixer,
                        number_of_qubits,
                        backend='numpy'):
    """

    :param angle_mixer:
    :param number_of_qubits:
    :param backend:
    :return:
    """


    _bck = cp if backend == 'cupy' else np
    _1q_mixer = get_exp_X_operator_1q(angle_mixer, backend=backend)

    big_mixer = _1q_mixer.copy()
    for _ in range(1, number_of_qubits):
        big_mixer = _bck.kron(big_mixer, _1q_mixer)

    return big_mixer


def multiply_by_mixer_operator(angle_mixer,
                               number_of_qubits,
                               input_state,
                               backend='numpy',):
    """
    Apply mixer operator to statevector without storing full matrix.

    Exploits tensor product structure: U_mixer = exp(-i*angle*X)^⊗n
    Uses tensordot for efficient computation.

    With cupy present, backend='cupy' works in place on a C-contiguous complex64 or complex128
    cupy array, k = 4 bits per launch, and returns that same array; a non-contiguous input is
    made contiguous first, and any other dtype raises TypeError.

    :param angle_mixer: Mixer angle β
    :param number_of_qubits: Number of qubits
    :param input_state: Input statevector of shape (2^n,)
    :param backend: 'numpy' or 'cupy'
    :return: Statevector after mixer application
    """

    if backend == 'cupy' and 'cupy' in AVAILABLE_SIMULATORS:
        assert input_state.size == 1 << number_of_qubits, (
            f"input_state has {input_state.size} amplitudes, expected 2**{number_of_qubits}")
        input_state = _as_cupy_contiguous(input_state)
        beta = float(angle_mixer)
        cos_beta, sin_beta = float(np.cos(beta)), float(np.sin(beta))
        params = [(cos_beta, sin_beta, 0.0)] * number_of_qubits
        return apply_xz_rotations_inplace(input_state, params)

    _bck = cp if backend == 'cupy' else np
    #
    cos_beta, sin_beta = _bck.cos(angle_mixer), _bck.sin(angle_mixer)

    # Reshape statevector to tensor form: (2, 2, ..., 2) with n indices
    state = input_state.reshape([2] * number_of_qubits)

    # Apply single-qubit mixer to each qubit index
    for qubit_idx in range(number_of_qubits):
        # Move qubit_idx axis to position 0
        state = _bck.moveaxis(state, qubit_idx, 0)

        # Apply exp(-i*β*X) = [[cos(β), -i*sin(β)], [-i*sin(β), cos(β)]]
        state[0], state[1] = (cos_beta * state[0] - 1j * sin_beta * state[1],
                              -1j * sin_beta * state[0] + cos_beta * state[1])

        # Move axis back
        state = _bck.moveaxis(state, 0, qubit_idx)

    # Flatten back to vector form
    return state.reshape(2**number_of_qubits)


def multiply_by_mixer_operator_WS(angle_mixer,
                                  number_of_qubits,
                                  input_state,
                                  XZ_terms:Tuple[float,float]|List[Tuple[float,float]],
                                  backend='numpy',):
    """
    Apply warm-started mixer operator to statevector without storing full matrix.

    Exploits tensor product structure: U_mixer = exp(-i*β*(X_term*X + Z_term*Z))^⊗n
    Uses tensordot for efficient computation.

    :param angle_mixer: Mixer angle β
    :param number_of_qubits: Number of qubits
    :param input_state: Input statevector of shape (2^n,)
    With cupy present, backend='cupy' works in place on a C-contiguous complex64 or complex128
    cupy array, k = 4 bits per launch, and returns that same array; a non-contiguous input is
    made contiguous first, and any other dtype raises TypeError. The XZ terms are read on the
    host, so pass Python floats or numpy scalars; a 0-d cupy array costs one device sync per read.

    :param XZ_terms: Either a single tuple (term_X, term_Z) for identical bias,
                     or a list of tuples for per-qubit bias
    :param backend: 'numpy' or 'cupy'
    :return: Statevector after mixer application
    """

    if backend == 'cupy' and 'cupy' in AVAILABLE_SIMULATORS:
        assert input_state.size == 1 << number_of_qubits, (
            f"input_state has {input_state.size} amplitudes, expected 2**{number_of_qubits}")
        input_state = _as_cupy_contiguous(input_state)
        beta = float(angle_mixer)
        cos_beta, sin_beta = float(np.cos(beta)), float(np.sin(beta))
        identical_bias = isinstance(XZ_terms, tuple)
        # Qubit q is bit n - 1 - q of the flat index.
        params = [None] * number_of_qubits
        for qubit_idx in range(number_of_qubits):
            term_X, term_Z = XZ_terms if identical_bias else XZ_terms[qubit_idx]
            params[number_of_qubits - 1 - qubit_idx] = (cos_beta, sin_beta * float(term_X), sin_beta * float(term_Z))
        return apply_xz_rotations_inplace(input_state, params)

    _bck = cp if backend == 'cupy' else np

    cos_beta, sin_beta = _bck.cos(angle_mixer), _bck.sin(angle_mixer)

    # Check if bias is identical for all qubits (single tuple) or per-qubit (list of tuples)
    identical_bias = isinstance(XZ_terms, tuple)

    # Reshape statevector to tensor form: (2, 2, ..., 2) with n indices
    state = input_state.reshape([2] * number_of_qubits)

    # Apply single-qubit mixer to each qubit index
    for qubit_idx in range(number_of_qubits):
        # Move qubit_idx axis to position 0
        state = _bck.moveaxis(state, qubit_idx, 0)

        # Get XZ terms for this qubit
        if identical_bias:
            term_X, term_Z = XZ_terms  # Same for all qubits
        else:
            term_X, term_Z = XZ_terms[qubit_idx]  # Per-qubit

        a, b = state[0], state[1]

        # Apply WS mixer: exp(-i*β*(term_X*X + term_Z*Z))
        # Matrix: [[cos(β) - i*sin(β)*Z,   -i*sin(β)*2*X],
        #          [-i*sin(β)*2*X,          cos(β) + i*sin(β)*Z]]
        state[0], state[1] = (a*cos_beta - 1j*sin_beta*(term_Z*a + b*term_X),
                              b*cos_beta - 1j*sin_beta*(-term_Z*b + a*term_X))

        # Move axis back
        state = _bck.moveaxis(state, 0, qubit_idx)

    # Flatten back to vector form
    return state.reshape(2**number_of_qubits)



def get_initial_state_WS_qaoa_1q(bias_parameter_WS:Optional[float]=None,
                                 backend='numpy',
                                 dtype=np.complex128):
    """The 1-qubit warm-start state [sqrt(1 - c), sqrt(c)], formed in complex128 and returned
    at `dtype`."""

    if backend == 'numpy':
        _bck = np
    elif backend == 'cupy':
        _bck = cp
    else:
        raise ValueError(f'Unknown backend: {backend}')


    if bias_parameter_WS is None:
        bias_parameter_WS = 0.5

    sqrt_c = _bck.sqrt(bias_parameter_WS)
    sqrt_1c = _bck.sqrt(1 - bias_parameter_WS)

    return _bck.array([sqrt_1c, sqrt_c], dtype=_bck.complex128).astype(dtype, copy=False)


def get_initial_state_WS_qaoa(number_of_qubits:int,
                              bias_parameters_WS:Optional[float | List[float]]=None,
                              backend='numpy',
                              dtype=np.complex128):
    """The warm-start product state (uniform when `bias_parameters_WS` is None), returned at
    `dtype`. The uniform state is built directly at `dtype`: its one value, 2 ** (-n / 2), is
    rounded once either way. The product state is formed in complex128 and cast once, at the
    end: a complex64 product formed factor by factor would carry up to (n - 1) / 2 eps, one
    cast carries 0.5 eps."""

    if backend == 'numpy':
        _bck = np
    elif backend == 'cupy':
        _bck = cp
    else:
        raise ValueError(f'Unknown backend: {backend}')

    dimension = 2**number_of_qubits

    if bias_parameters_WS is None:
        return _bck.full(dimension, 2.0 ** (-number_of_qubits / 2), dtype=dtype)


    else:
        _identical_bias = False
        if isinstance(bias_parameters_WS, float):
            _identical_bias = True
            bias_parameters_WS = [bias_parameters_WS]

        if _identical_bias:
            _1q_state = get_initial_state_WS_qaoa_1q(bias_parameter_WS=bias_parameters_WS[0],
                                                     backend=backend)

            input_state = _1q_state.copy()
            for _ in range(number_of_qubits-1):
                input_state = _bck.kron(input_state,_1q_state)

        else:

            _1q_state = get_initial_state_WS_qaoa_1q(bias_parameter_WS=bias_parameters_WS[0],
                                                     backend=backend)

            input_state = _1q_state.copy()

            for c in bias_parameters_WS[1:]:
                input_state = _bck.kron(input_state,
                                        get_initial_state_WS_qaoa_1q(bias_parameter_WS=c,
                                                                     backend=backend))

    return input_state.astype(dtype, copy=False)


def p1_beta_fourier_coefficients(overlaps_2q,
                                 mixer_x: float,
                                 mixer_z: float,
                                 overlaps_1q=None) -> Tuple[float, float, float, float, float]:
    """Fourier coefficients of the p=1 warm-start QAOA energy as a function of the mixer angle.

    For a two-local Hamiltonian, with or without local fields, and a uniform warm-start bias,
    the energy after the mixer exp(-i beta (mixer_x X + mixer_z Z)) on every qubit is

        E(beta) = A0 + A1 cos(2 beta) + B1 sin(2 beta) + A2 cos(4 beta) + B2 sin(4 beta).

    The rotated Z operator is f(beta) . sigma with f = u + v cos(2 beta) + w sin(2 beta),
    u = (n_x n_z, 0, n_z^2), v = (-n_x n_z, 0, n_x^2), w = (0, n_x, 0), and
    E = f^T O f + f . o, where O is the coupling-weighted two-body Pauli overlap matrix and o the
    field-weighted one-body overlap vector. Only the symmetric part of O contributes. The
    one-body term is linear in f, so it shifts A0, A1 and B1 only.

    :param overlaps_2q: (3, 3) array, overlaps_2q[mu, nu] = sum_{i<j} J_ij <sigma_mu^(i) sigma_nu^(j)>,
                        Pauli indices 0 = X, 1 = Y, 2 = Z.
    :param mixer_x: X component of the mixer axis, 2 sqrt(c (1 - c)) for bias c.
    :param mixer_z: Z component of the mixer axis, 1 - 2c (sign flipped for a mixer opposite to
                    the input state).
    :param overlaps_1q: None for a Hamiltonian without local fields, else a length-3 array,
                        overlaps_1q[mu] = sum_i h_i <sigma_mu^(i)>, same Pauli indices.
    :return: (A0, A1, B1, A2, B2) as Python floats, computed in float64.
    """
    O = np.asarray(overlaps_2q, dtype=np.float64)
    S = 0.5 * (O + O.T)
    nx, nz = float(mixer_x), float(mixer_z)

    u = np.array([nx * nz, 0.0, nz ** 2])
    v = np.array([-nx * nz, 0.0, nx ** 2])
    w = np.array([0.0, nx, 0.0])

    def q(a, b):
        return float(a @ S @ b)

    A0 = q(u, u) + 0.5 * (q(v, v) + q(w, w))
    A1 = 2.0 * q(u, v)
    B1 = 2.0 * q(u, w)
    A2 = 0.5 * (q(v, v) - q(w, w))
    B2 = q(v, w)
    if overlaps_1q is not None:
        # f . o with f = u + v cos(2 beta) + w sin(2 beta) moves only the constant and the
        # first harmonic.
        o = np.asarray(overlaps_1q, dtype=np.float64).reshape(3)
        A0 += float(u @ o)
        A1 += float(v @ o)
        B1 += float(w @ o)
    return A0, A1, B1, A2, B2


def minimize_p1_beta_trigonometric_polynomial(A0: float,
                                              A1: float,
                                              B1: float,
                                              A2: float,
                                              B2: float) -> Tuple[float, float]:
    """Global minimiser of E(beta) = A0 + A1 cos2b + B1 sin2b + A2 cos4b + B2 sin4b over beta.

    With z = exp(2 i beta), k1 = A1 - i B1 and k2 = A2 - i B2, the stationary points of E are the
    roots of 2 k2 z^4 + k1 z^3 - conj(k1) z - 2 conj(k2) = 0 on the unit circle. Every root's
    argument is taken as a candidate (robust to roots that drift off the circle numerically) and
    the candidate with the smallest E is returned. A constant E (no roots) returns beta = 0.

    :return: (beta_star, E_star), beta_star in (-pi/2, pi/2]. NOTE the order: angle first.
    """
    k1 = A1 - 1j * B1
    k2 = A2 - 1j * B2
    roots = np.roots([2.0 * k2, k1, 0.0, -np.conj(k1), -2.0 * np.conj(k2)])
    if roots.size == 0:
        return 0.0, float(A0)

    betas = np.angle(roots) / 2.0
    values = (A0
              + A1 * np.cos(2.0 * betas) + B1 * np.sin(2.0 * betas)
              + A2 * np.cos(4.0 * betas) + B2 * np.sin(4.0 * betas))
    k = int(np.argmin(values))
    return float(betas[k]), float(values[k])