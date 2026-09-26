# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import time

import numpy as np
from numba import cuda

# The CPU environments install no cupy. The GPU functions here need it, but importing this
# module must not: the RDM runner imports it on every backend.
try:
    import cupy as cp
    from cupy import ndarray
except (ModuleNotFoundError, ImportError):
    import numpy as cp
    from numpy import ndarray


@cuda.jit
def _compute_couplings_sums_kernel(couplings_phase, fields_phase, couplings_sums):
    """
    Compute couplings_sums[i] = sum_j(couplings_phase[i,j]) + fields_phase[i].

    This is independent of gamma/beta and can be precomputed once.
    """
    n = couplings_phase.shape[0]
    i = cuda.grid(1)

    if i >= n:
        return

    sum_i = fields_phase[i]
    for j in range(n):
        if i != j:
            sum_i += couplings_phase[i, j]

    couplings_sums[i] = sum_i


@cuda.jit
def _precompute_phase_kernel(couplings_phase,
                             gamma,
                             minus_four_gamma_couplings_real,
                             minus_four_gamma_couplings_imag,
                             gamma_couplings_real,
                             gamma_couplings_imag):
    """Precompute phase-dependent data for all qubit pairs."""
    n = couplings_phase.shape[0]
    i = cuda.grid(1)

    if i >= n:
        return

    # Compute phase factors for all pairs involving qubit i
    for j in range(n):
        if i == j:
            continue

        cij = couplings_phase[i, j]

        # minus_four_gamma_1j_couplings[i,j] = exp(-4j * gamma * cij)
        arg_4 = -4.0 * gamma * cij
        minus_four_gamma_couplings_real[i, j] = np.cos(arg_4)
        minus_four_gamma_couplings_imag[i, j] = np.sin(arg_4)

        # gamma_1j_couplings[i,j] = exp(1j * gamma * cij)
        arg_1 = gamma * cij
        gamma_couplings_real[i, j] = np.cos(arg_1)
        gamma_couplings_imag[i, j] = np.sin(arg_1)


def compute_couplings_sums_cuda(couplings_phase, fields_phase):
    """
    Compute couplings_sums[i] = sum_j(couplings_phase[i,j]) + fields_phase[i].

    This is independent of gamma/beta and should be precomputed once.

    Parameters:
    -----------
    couplings_phase : cuda device array, shape (n_qubits, n_qubits)
        Phase Hamiltonian couplings
    fields_phase : cuda device array, shape (n_qubits,)
        Phase Hamiltonian local fields

    Returns:
    --------
    couplings_sums : cuda device array, shape (n_qubits,)
        Sum of couplings and fields for each qubit
    """
    n_qubits = fields_phase.shape[0]

    # Allocate output on device
    couplings_sums = cuda.to_device(np.zeros(n_qubits, dtype=np.float32))

    # Launch kernel
    threads_1d = 256
    blocks_1d = (n_qubits + threads_1d - 1) // threads_1d
    _compute_couplings_sums_kernel[blocks_1d, threads_1d](
        couplings_phase,
        fields_phase,
        couplings_sums
    )

    return couplings_sums


@cuda.jit
def _compute_couplings_sums_gamma_kernel(couplings_phase,
                                         gamma,
                                         couplings_sums,
                                         couplings_sums_gamma_real,
                                         couplings_sums_gamma_imag):
    """Compute couplings_sums_gamma_1j[i,j] = exp(1j*gamma*(couplings_sums[i] - couplings_phase[i,j]))

    For diagonal elements [i,i]: exp(1j*gamma*couplings_sums[i]) (used for single-qubit RDMs)
    For off-diagonal [i,j]: exp(1j*gamma*(couplings_sums[i] - couplings_phase[i,j]))
    """
    n = couplings_phase.shape[0]
    i, j = cuda.grid(2)

    if not (i <= j < n):
        return

    # Diagonal case: exp(1j * gamma * couplings_sums[i])
    if i == j:
        val_ii = gamma * couplings_sums[i]
        couplings_sums_gamma_real[i, i] = np.cos(val_ii)
        couplings_sums_gamma_imag[i, i] = np.sin(val_ii)
        return

    cij = couplings_phase[i, j]

    # Entry [i,j] uses couplings_sums[i]
    val_ij = gamma * (couplings_sums[i] - cij)
    couplings_sums_gamma_real[i, j] = np.cos(val_ij)
    couplings_sums_gamma_imag[i, j] = np.sin(val_ij)

    # Entry [j,i] uses couplings_sums[j]
    val_ji = gamma * (couplings_sums[j] - cij)
    couplings_sums_gamma_real[j, i] = np.cos(val_ji)
    couplings_sums_gamma_imag[j, i] = np.sin(val_ji)


@cuda.jit
def _calculate_rho_ij_kernel(initial_states_real,
                             initial_states_imag,
                             couplings_phase,
                             couplings_sums_gamma_real,
                             couplings_sums_gamma_imag,
                             minus_four_gamma_couplings_real,
                             minus_four_gamma_couplings_imag,
                             gamma_couplings_real,
                             gamma_couplings_imag,
                             ws_bias_parameters,
                             rdm_mask,
                             rho_ij_real_out,
                             rho_ij_imag_out):
    """
    Main kernel: each thread computes one RDM rho_ij for pair (i,j).

    This implements the full p=1 QAOA phase separator RDM computation including:
    - Initial state preparation with phase factors
    - Kron product to form 2-qubit density matrix
    - CP gates from all other qubits k
    - Final Rzz gate between i and j

    rdm_mask is a uint8 (n, n) array: a thread whose entry (diagonal for one-qubit RDMs,
    upper triangle for pairs) is 0 returns before any work, leaving the zero-filled output block.
    """
    n = couplings_phase.shape[0]
    idx_qi, idx_qj = cuda.grid(2)

    # Diagonal case: single-qubit RDM for local fields
    if idx_qi == idx_qj:
        if idx_qi >= n:
            return
        if rdm_mask[idx_qi, idx_qi] == 0:
            return

        rho_real = cuda.local.array((2, 2), dtype=np.float32)
        rho_imag = cuda.local.array((2, 2), dtype=np.float32)

        # Get initial state components
        qi_0_real = initial_states_real[idx_qi, 0]
        qi_0_imag = initial_states_imag[idx_qi, 0]
        qi_1_real = initial_states_real[idx_qi, 1]
        qi_1_imag = initial_states_imag[idx_qi, 1]

        # Get phase factor exp(1j * gamma * couplings_sums[i])
        cij_qi_real = couplings_sums_gamma_real[idx_qi, idx_qi]
        cij_qi_imag = couplings_sums_gamma_imag[idx_qi, idx_qi]

        # qi_ket[0] *= conj(cij_qi)
        temp_real = qi_0_real * cij_qi_real + qi_0_imag * cij_qi_imag
        temp_imag = qi_0_imag * cij_qi_real - qi_0_real * cij_qi_imag
        qi_0_real = temp_real
        qi_0_imag = temp_imag

        # qi_ket[1] *= cij_qi
        temp_real = qi_1_real * cij_qi_real - qi_1_imag * cij_qi_imag
        temp_imag = qi_1_real * cij_qi_imag + qi_1_imag * cij_qi_real
        qi_1_real = temp_real
        qi_1_imag = temp_imag

        # Form 2x2 density matrix rho_i = qi_ket @ qi_ket†
        # rho[a,b] = qi_ket[a] * conj(qi_ket[b])
        # [0,0]: qi_0 * conj(qi_0)
        rho_real[0, 0] = qi_0_real * qi_0_real + qi_0_imag * qi_0_imag
        rho_imag[0, 0] = 0.0  # Always real (diagonal)

        # [0,1]: qi_0 * conj(qi_1)
        rho_real[0, 1] = qi_0_real * qi_1_real + qi_0_imag * qi_1_imag
        rho_imag[0, 1] = qi_0_imag * qi_1_real - qi_0_real * qi_1_imag

        # [1,0]: qi_1 * conj(qi_0)
        rho_real[1, 0] = rho_real[0, 1]
        rho_imag[1, 0] = -rho_imag[0, 1]

        # [1,1]: qi_1 * conj(qi_1)
        rho_real[1, 1] = 1 - rho_real[0, 0]
        rho_imag[1, 1] = 0.0  # Always real (diagonal)

        # Apply partial trace formula for each other qubit k, weighted by k's own bias c_k
        # rho_i[1,0] = (1-c_k) * rho_i[1,0] + c_k * phase_i * rho_i[1,0]
        # rho_i[0,1] = conj(rho_i[1,0])
        for k in range(n):
            if k == idx_qi:
                continue
            if couplings_phase[idx_qi, k] == 0.0:
                continue

            c = ws_bias_parameters[k]
            one_minus_c = 1.0 - c

            phase_i_real = minus_four_gamma_couplings_real[idx_qi, k]
            phase_i_imag = minus_four_gamma_couplings_imag[idx_qi, k]

            # rho[1,0] = (1-c) * rho[1,0] + c * phase_i * rho[1,0]
            # = rho[1,0] * ((1-c) + c * phase_i)
            # = rho[1,0] * (1-c + c*phase_i_real + i*c*phase_i_imag)
            old_10_real = rho_real[1, 0]
            old_10_imag = rho_imag[1, 0]

            # Compute (1-c + c*phase_i_real) + i*(c*phase_i_imag)
            factor_real = one_minus_c + c * phase_i_real
            factor_imag = c * phase_i_imag

            # Complex multiply: rho[1,0] * factor
            rho_real[1, 0] = old_10_real * factor_real - old_10_imag * factor_imag
            rho_imag[1, 0] = old_10_real * factor_imag + old_10_imag * factor_real

            # rho[0,1] = conj(rho[1,0])
            rho_real[0, 1] = rho_real[1, 0]
            rho_imag[0, 1] = -rho_imag[1, 0]

        # Write output to 2x2 submatrix of the 4x4 output
        for row in range(2):
            for col in range(2):
                rho_ij_real_out[idx_qi, idx_qi, row, col] = rho_real[row, col]
                rho_ij_imag_out[idx_qi, idx_qi, row, col] = rho_imag[row, col]

        return

    if not (idx_qi < idx_qj < n):
        return
    if rdm_mask[idx_qi, idx_qj] == 0:
        return

    # Local storage for 4x4 density matrix (real and imag parts)
    rho_real = cuda.local.array((4, 4), dtype=np.float32)
    rho_imag = cuda.local.array((4, 4), dtype=np.float32)

    # Get initial states for qubits i and j
    qi_0_real = initial_states_real[idx_qi, 0]
    qi_0_imag = initial_states_imag[idx_qi, 0]
    qi_1_real = initial_states_real[idx_qi, 1]
    qi_1_imag = initial_states_imag[idx_qi, 1]

    qj_0_real = initial_states_real[idx_qj, 0]
    qj_0_imag = initial_states_imag[idx_qj, 0]
    qj_1_real = initial_states_real[idx_qj, 1]
    qj_1_imag = initial_states_imag[idx_qj, 1]

    # Apply phase factors to qi_ket and qj_ket (if coupling exists)
    # if couplings_phase[idx_qi, idx_qj] != 0.0:
    cij_qi_real = couplings_sums_gamma_real[idx_qi, idx_qj]
    cij_qi_imag = couplings_sums_gamma_imag[idx_qi, idx_qj]
    cij_qj_real = couplings_sums_gamma_real[idx_qj, idx_qi]
    cij_qj_imag = couplings_sums_gamma_imag[idx_qj, idx_qi]

    # qi_ket[0] *= conj(cij_qi)
    temp_real = qi_0_real * cij_qi_real + qi_0_imag * cij_qi_imag
    temp_imag = qi_0_imag * cij_qi_real - qi_0_real * cij_qi_imag
    qi_0_real = temp_real
    qi_0_imag = temp_imag

    # qi_ket[1] *= cij_qi
    temp_real = qi_1_real * cij_qi_real - qi_1_imag * cij_qi_imag
    temp_imag = qi_1_real * cij_qi_imag + qi_1_imag * cij_qi_real
    qi_1_real = temp_real
    qi_1_imag = temp_imag

    # qj_ket[0] *= conj(cij_qj)
    temp_real = qj_0_real * cij_qj_real + qj_0_imag * cij_qj_imag
    temp_imag = qj_0_imag * cij_qj_real - qj_0_real * cij_qj_imag
    qj_0_real = temp_real
    qj_0_imag = temp_imag

    # qj_ket[1] *= cij_qj
    temp_real = qj_1_real * cij_qj_real - qj_1_imag * cij_qj_imag
    temp_imag = qj_1_real * cij_qj_imag + qj_1_imag * cij_qj_real
    qj_1_real = temp_real
    qj_1_imag = temp_imag

    # Compute rho_ij = kron(qi_ket @ qi_ket.H, qj_ket @ qj_ket.H)
    # qi_rho[a,b] = qi_ket[a] * conj(qi_ket[b])
    # kron gives: rho_ij[2*a+c, 2*b+d] = qi_rho[a,b] * qj_rho[c,d]

    # Manual kron product - 16 elements
    # Ordering: |00>, |01>, |10>, |11>
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):

                    # Complex multiply: qi_rho * qj_rho
                    row = 2 * a + c
                    col = 2 * b + d

                    if col < row:
                        continue

                    # qi_ket[a] * conj(qi_ket[b])
                    if a == 0:
                        qi_a_real, qi_a_imag = qi_0_real, qi_0_imag
                    else:
                        qi_a_real, qi_a_imag = qi_1_real, qi_1_imag

                    if b == 0:
                        qi_b_real, qi_b_imag = qi_0_real, qi_0_imag
                    else:
                        qi_b_real, qi_b_imag = qi_1_real, qi_1_imag

                    # qi_rho[a,b] = qi_ket[a] * conj(qi_ket[b])
                    qi_rho_real = qi_a_real * qi_b_real + qi_a_imag * qi_b_imag
                    qi_rho_imag = qi_a_imag * qi_b_real - qi_a_real * qi_b_imag

                    # qj_ket[c] * conj(qj_ket[d])
                    if c == 0:
                        qj_c_real, qj_c_imag = qj_0_real, qj_0_imag
                    else:
                        qj_c_real, qj_c_imag = qj_1_real, qj_1_imag

                    if d == 0:
                        qj_d_real, qj_d_imag = qj_0_real, qj_0_imag
                    else:
                        qj_d_real, qj_d_imag = qj_1_real, qj_1_imag

                    # qj_rho[c,d] = qj_ket[c] * conj(qj_ket[d])
                    qj_rho_real = qj_c_real * qj_d_real + qj_c_imag * qj_d_imag
                    qj_rho_imag = qj_c_imag * qj_d_real - qj_c_real * qj_d_imag

                    rho_real[row, col] = qi_rho_real * qj_rho_real - qi_rho_imag * qj_rho_imag
                    rho_imag[row, col] = qi_rho_real * qj_rho_imag + qi_rho_imag * qj_rho_real

                    # rho_real[col,row] = rho_real[row,col]
                    # rho_imag[col,row] = -rho_imag[row,col]

    # Apply CP gates from all other qubits k
    # rho_new = 0.5 * rho + 0.5 * U @ rho @ U^dag
    # where U = diag([1, phase_j, phase_i, phase_i*phase_j])

    temp_real = cuda.local.array((4, 4), dtype=np.float32)
    temp_imag = cuda.local.array((4, 4), dtype=np.float32)

    for k in range(n):
        if k == idx_qi or k == idx_qj:
            continue
        if couplings_phase[idx_qi, k] == 0.0 and couplings_phase[idx_qj, k] == 0.0:
            continue

        phase_i_real = minus_four_gamma_couplings_real[idx_qi, k]
        phase_i_imag = minus_four_gamma_couplings_imag[idx_qi, k]
        phase_j_real = minus_four_gamma_couplings_real[idx_qj, k]
        phase_j_imag = minus_four_gamma_couplings_imag[idx_qj, k]

        # Diagonal elements: [1.0, phase_j, phase_i, phase_i * phase_j]
        # For diag(d) @ rho @ diag(d)^†: result[i,j] = d[i] * rho[i,j] * conj(d[j])

        # Compute phase_i * phase_j (complex multiply)
        phase_ij_real = phase_i_real * phase_j_real - phase_i_imag * phase_j_imag
        phase_ij_imag = phase_i_real * phase_j_imag + phase_i_imag * phase_j_real

        # Apply diagonal transformation: temp = U @ rho @ U^dag
        for row in range(4):
            for col in range(row, 4):
                # Get diagonal elements d_row and conj(d_col)
                if row == 0:
                    d_row_real, d_row_imag = 1.0, 0.0
                elif row == 1:
                    d_row_real, d_row_imag = phase_j_real, phase_j_imag
                elif row == 2:
                    d_row_real, d_row_imag = phase_i_real, phase_i_imag
                else:  # row == 3
                    d_row_real, d_row_imag = phase_ij_real, phase_ij_imag

                if col == 0:
                    d_col_conj_real, d_col_conj_imag = 1.0, 0.0
                elif col == 1:
                    d_col_conj_real, d_col_conj_imag = phase_j_real, -phase_j_imag
                elif col == 2:
                    d_col_conj_real, d_col_conj_imag = phase_i_real, -phase_i_imag
                else:  # col == 3
                    d_col_conj_real, d_col_conj_imag = phase_ij_real, -phase_ij_imag

                # temp[row,col] = d_row * rho[row,col] * conj(d_col)
                # First: d_row * rho[row,col]
                tmp1_real = d_row_real * rho_real[row, col] - d_row_imag * rho_imag[row, col]
                tmp1_imag = d_row_real * rho_imag[row, col] + d_row_imag * rho_real[row, col]

                # Then: tmp1 * conj(d_col)
                temp_real[row, col] = tmp1_real * d_col_conj_real - tmp1_imag * d_col_conj_imag
                temp_imag[row, col] = tmp1_real * d_col_conj_imag + tmp1_imag * d_col_conj_real


        # rho = 0.5 * rho + 0.5 * temp

        c = ws_bias_parameters[k]

        for row in range(4):
            for col in range(row, 4):
                rho_real[row, col] = (1 - c) * rho_real[row, col] + c * temp_real[row, col]
                rho_imag[row, col] = (1 - c) * rho_imag[row, col] + c * temp_imag[row, col]

    # Apply final Rzz gate between i and j
    # U = diag([phase_m, phase_p, phase_p, phase_m])
    if couplings_phase[idx_qi, idx_qj] != 0.0:
        phase_p_real = gamma_couplings_real[idx_qi, idx_qj]
        phase_p_imag = gamma_couplings_imag[idx_qi, idx_qj]
        phase_m_real = phase_p_real  # conj: flip sign of imag
        phase_m_imag = -phase_p_imag

        # Apply: rho = U @ rho @ U^dag
        for row in range(4):
            for col in range(row, 4):
                if row == 0 or row == 3:
                    d_row_real, d_row_imag = phase_m_real, phase_m_imag
                else:
                    d_row_real, d_row_imag = phase_p_real, phase_p_imag

                if col == 0 or col == 3:
                    d_col_conj_real, d_col_conj_imag = phase_m_real, -phase_m_imag
                else:
                    d_col_conj_real, d_col_conj_imag = phase_p_real, -phase_p_imag

                tmp1_real = d_row_real * rho_real[row, col] - d_row_imag * rho_imag[row, col]
                tmp1_imag = d_row_real * rho_imag[row, col] + d_row_imag * rho_real[row, col]

                temp_real[row, col] = tmp1_real * d_col_conj_real - tmp1_imag * d_col_conj_imag
                temp_imag[row, col] = tmp1_real * d_col_conj_imag + tmp1_imag * d_col_conj_real

                # rho_real[row, col] = temp_real[row, col]
                # rho_imag[row, col] = -temp_imag[row, col]





        # Copy temp back to rho
        for row in range(4):
            for col in range(row, 4):
                rho_real[row, col] = temp_real[row, col]
                rho_imag[row, col] = temp_imag[row, col]


    # Write output
    for row in range(4):
        for col in range(row, 4):
            rho_ij_real_out[idx_qi, idx_qj, row, col] = rho_real[row, col]
            rho_ij_imag_out[idx_qi, idx_qj, row, col] = rho_imag[row, col]

            rho_ij_real_out[idx_qi, idx_qj, col, row] = rho_real[row, col]
            rho_ij_imag_out[idx_qi, idx_qj, col, row] = -rho_imag[row, col]


@cuda.jit
def _apply_mixers_and_extract_ZiZj_kernel(mixers_real,
                                          mixers_imag,
                                          rho_ij_real,
                                          rho_ij_imag,
                                          couplings_cost,
                                          fields_cost,
                                          ZiZj_out,
                                          ):
    """
    Apply mixer operators and extract <ZiZj> and <Zi> expectation values.

    For diagonal (i == j): computes single-qubit <Zi> for local fields
    For off-diagonal (i < j): computes two-qubit <ZiZj> for couplings

    Each thread handles one entry (i,j):
    - Diagonal: rho_i = mixer_i @ rho_phase @ mixer_i^†, <Zi> = rho[1,1] - rho[0,0]
    - Off-diagonal: rho_ij = mixer_ij @ rho_phase @ mixer_ij^†, <ZiZj> = rho[0,0] - rho[1,1] - rho[2,2] + rho[3,3]
    """
    n = couplings_cost.shape[0]
    idx_qi, idx_qj = cuda.grid(2)

    # Diagonal case: single-qubit <Zi> for local fields
    if idx_qi == idx_qj:
        if idx_qi >= n:
            return

        # Skip if no local field in cost Hamiltonian
        if fields_cost[idx_qi] == 0.0:
            return

        # Get single-qubit mixer (2x2 matrix)
        mixer_qi_real = cuda.local.array((2, 2), dtype=np.float32)
        mixer_qi_imag = cuda.local.array((2, 2), dtype=np.float32)

        for i in range(2):
            for j in range(2):
                mixer_qi_real[i, j] = mixers_real[idx_qi, i, j]
                mixer_qi_imag[i, j] = mixers_imag[idx_qi, i, j]

        # Load 2x2 rho_i_phase from diagonal of RDM array
        rho_phase_real = cuda.local.array((2, 2), dtype=np.float32)
        rho_phase_imag = cuda.local.array((2, 2), dtype=np.float32)

        for i in range(2):
            for j in range(2):
                rho_phase_real[i, j] = rho_ij_real[idx_qi, idx_qi, i, j]
                rho_phase_imag[i, j] = rho_ij_imag[idx_qi, idx_qi, i, j]

        # Compute temp = mixer_qi @ rho_phase
        temp_real = cuda.local.array((2, 2), dtype=np.float32)
        temp_imag = cuda.local.array((2, 2), dtype=np.float32)

        for i in range(2):
            for j in range(2):
                sum_real = 0.0
                sum_imag = 0.0
                for k in range(2):
                    sum_real += (mixer_qi_real[i, k] * rho_phase_real[k, j] -
                                 mixer_qi_imag[i, k] * rho_phase_imag[k, j])
                    sum_imag += (mixer_qi_real[i, k] * rho_phase_imag[k, j] +
                                 mixer_qi_imag[i, k] * rho_phase_real[k, j])
                temp_real[i, j] = sum_real
                temp_imag[i, j] = sum_imag

        # Compute rho_final = temp @ mixer_qi^†
        rho_final_real = cuda.local.array((2, 2), dtype=np.float32)

        for i in range(2):
            for j in range(2):
                sum_real = 0.0
                for k in range(2):
                    # temp[i,k] * conj(mixer_qi[j,k])
                    sum_real += (temp_real[i, k] * mixer_qi_real[j, k] +
                                 temp_imag[i, k] * mixer_qi_imag[j, k])
                rho_final_real[i, j] = sum_real

        # Extract <Zi> = rho[1,1] - rho[0,0]
        Zi_value = -rho_final_real[1, 1] + rho_final_real[0, 0]
        ZiZj_out[idx_qi, idx_qi] = Zi_value

        return

    # Off-diagonal case: two-qubit <ZiZj>
    if not (idx_qi < idx_qj < n):
        return

    # Skip if no coupling in cost Hamiltonian (Time-Block ansatz)
    if couplings_cost[idx_qi, idx_qj] == 0.0:
        return

    # Get single-qubit mixers (2x2 matrices)
    mixer_qi_real = cuda.local.array((2, 2), dtype=np.float32)
    mixer_qi_imag = cuda.local.array((2, 2), dtype=np.float32)
    mixer_qj_real = cuda.local.array((2, 2), dtype=np.float32)
    mixer_qj_imag = cuda.local.array((2, 2), dtype=np.float32)

    for i in range(2):
        for j in range(2):
            mixer_qi_real[i, j] = mixers_real[idx_qi, i, j]
            mixer_qi_imag[i, j] = mixers_imag[idx_qi, i, j]
            mixer_qj_real[i, j] = mixers_real[idx_qj, i, j]
            mixer_qj_imag[i, j] = mixers_imag[idx_qj, i, j]

    # Compute mixer_qiqj = kron(mixer_qi, mixer_qj)
    mixer_qiqj_real = cuda.local.array((4, 4), dtype=np.float32)
    mixer_qiqj_imag = cuda.local.array((4, 4), dtype=np.float32)

    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    row = 2 * a + c
                    col = 2 * b + d
                    mixer_qiqj_real[row, col] = (mixer_qi_real[a, b] * mixer_qj_real[c, d] -
                                                 mixer_qi_imag[a, b] * mixer_qj_imag[c, d])
                    mixer_qiqj_imag[row, col] = (mixer_qi_real[a, b] * mixer_qj_imag[c, d] +
                                                 mixer_qi_imag[a, b] * mixer_qj_real[c, d])

    # Load rho_ij_phase
    rho_phase_real = cuda.local.array((4, 4), dtype=np.float32)
    rho_phase_imag = cuda.local.array((4, 4), dtype=np.float32)

    for i in range(4):
        for j in range(4):
            rho_phase_real[i, j] = rho_ij_real[idx_qi, idx_qj, i, j]
            rho_phase_imag[i, j] = rho_ij_imag[idx_qi, idx_qj, i, j]

    # Compute temp = mixer_qiqj @ rho_phase
    temp_real = cuda.local.array((4, 4), dtype=np.float32)
    temp_imag = cuda.local.array((4, 4), dtype=np.float32)

    for i in range(4):
        for j in range(4):
            sum_real = 0.0
            sum_imag = 0.0
            for k in range(4):
                sum_real += (mixer_qiqj_real[i, k] * rho_phase_real[k, j] -
                             mixer_qiqj_imag[i, k] * rho_phase_imag[k, j])
                sum_imag += (mixer_qiqj_real[i, k] * rho_phase_imag[k, j] +
                             mixer_qiqj_imag[i, k] * rho_phase_real[k, j])
            temp_real[i, j] = sum_real
            temp_imag[i, j] = sum_imag

    # Compute rho_final = temp @ mixer_qiqj^†
    rho_final_real = cuda.local.array((4, 4), dtype=np.float32)

    for i in range(4):
        for j in range(4):
            sum_real = 0.0
            for k in range(4):
                # temp[i,k] * conj(mixer_qiqj[j,k])
                sum_real += (temp_real[i, k] * mixer_qiqj_real[j, k] +
                             temp_imag[i, k] * mixer_qiqj_imag[j, k])
            rho_final_real[i, j] = sum_real

    # Extract <ZiZj> = real(rho[0,0] - rho[1,1] - rho[2,2] + rho[3,3])
    ZiZj_value = (rho_final_real[0, 0] - rho_final_real[1, 1] -
                  rho_final_real[2, 2] + rho_final_real[3, 3])

    ZiZj_out[idx_qi, idx_qj] = ZiZj_value


def get_all_rho_ij_cuda(gamma,
                        initial_states_real,
                        initial_states_imag,
                        couplings_phase,
                        fields_phase,
                        couplings_sums,
                        ws_bias_parameters,
                        threadsperblock=(16, 16),
                        rdm_mask=None):
    # rdm_mask: device uint8 (n, n) array; the diagonal gates the one-qubit RDMs, the upper triangle the pairs.
    # None computes every entry. The default is allocated on the device (no host-to-device copy).
    n_qubits = couplings_sums.shape[0]

    # The buffers below are allocated by cupy and consumed by numba kernels, so both libraries must point
    # at the same device. numba reports its device as a CUdevice object; compare as integers.
    numba_device = int(cuda.get_current_device().id)
    cupy_device = cp.cuda.Device().id
    if numba_device != cupy_device:
        raise RuntimeError(f"numba's current device ({numba_device}) differs from cupy's current device "
                           f"({cupy_device}). get_all_rho_ij_cuda allocates its buffers with cupy and fills them "
                           "with numba kernels, so select the same device in both libraries before calling it.")

    # Scratch and output buffers come from cupy's pooled allocator, zero-filled on the device, and reach
    # the kernels as numba views over that memory. The zero fill is load-bearing: the phase kernel leaves the
    # diagonals untouched and the RDM kernel writes only the (i <= j) blocks, while get_pauli_overlaps_cupy
    # reduces over the whole output with zero weight on the rest. The views carry the legacy default stream.
    # All eight are allocated before the first kernel launch so no allocation waits on queued work.
    def _device_zeros(shape):
        return cuda.as_cuda_array(cp.zeros(shape, dtype=cp.float32))

    minus_four_gamma_couplings_real = _device_zeros((n_qubits, n_qubits))
    minus_four_gamma_couplings_imag = _device_zeros((n_qubits, n_qubits))
    gamma_couplings_real = _device_zeros((n_qubits, n_qubits))
    gamma_couplings_imag = _device_zeros((n_qubits, n_qubits))

    couplings_sums_gamma_real = _device_zeros((n_qubits, n_qubits))
    couplings_sums_gamma_imag = _device_zeros((n_qubits, n_qubits))

    rho_ij_real_out = _device_zeros((n_qubits, n_qubits, 4, 4))
    rho_ij_imag_out = _device_zeros((n_qubits, n_qubits, 4, 4))

    if rdm_mask is None:
        rdm_mask = cuda.as_cuda_array(cp.ones((n_qubits, n_qubits), dtype=cp.uint8))

    # Launch precomputation kernel (1D over qubits)
    threads_1d = 256
    blocks_1d = (n_qubits + threads_1d - 1) // threads_1d
    _precompute_phase_kernel[blocks_1d, threads_1d](
        couplings_phase,
        gamma,
        minus_four_gamma_couplings_real,
        minus_four_gamma_couplings_imag,
        gamma_couplings_real,
        gamma_couplings_imag
    )

    # t0 = time.perf_counter()
    # couplings_sums_gamma_r = cp.asarray(couplings_sums_gamma_real)
    # couplings_sums_gamma_i = cp.asarray(couplings_sums_gamma_imag)
    # t1 = time.perf_counter()
    # couplings_sums_gamma = couplings_sums_gamma_r + 1j * couplings_sums_gamma_i
    # products_gamma = couplings_sums_gamma[:, :, None] * couplings_sums_gamma[:, None, :]
    # t2 = time.perf_counter()
    # products_gamma_r = cuda.to_device(cp.ascontiguousarray(products_gamma.real))
    # products_gamma_i = cuda.to_device(cp.ascontiguousarray(products_gamma.imag))
    #
    # t3 = time.perf_counter()







    # Launch couplings_sums_gamma kernel (2D over pairs)
    blockspergrid_x = (n_qubits + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n_qubits + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _compute_couplings_sums_gamma_kernel[blockspergrid, threadsperblock](
        couplings_phase,
        gamma,
        couplings_sums,
        couplings_sums_gamma_real,
        couplings_sums_gamma_imag
    )

    # Launch main RDM computation kernel (2D over pairs)
    _calculate_rho_ij_kernel[blockspergrid, threadsperblock](
        initial_states_real,
        initial_states_imag,
        couplings_phase,
        couplings_sums_gamma_real,
        couplings_sums_gamma_imag,
        minus_four_gamma_couplings_real,
        minus_four_gamma_couplings_imag,
        gamma_couplings_real,
        gamma_couplings_imag,
        ws_bias_parameters,
        rdm_mask,
        rho_ij_real_out,
        rho_ij_imag_out
    )

    # Combine back to complex array
    # rho_ij_array = rho_ij_real_out + 1j * rho_ij_imag_out

    return rho_ij_real_out, rho_ij_imag_out


def get_all_ZiZj_cuda(rho_ij_real,
                      rho_ij_imag,
                      mixers_real,
                      mixers_imag,
                      fields_cost,
                      couplings_cost,
                      threadsperblock=(16, 16)):
    """
    Apply mixer operators and extract <ZiZj> and <Zi> expectation values.

    Parameters:
    -----------
    rho_ij_real, rho_ij_imag : cuda device arrays, shape (n_qubits, n_qubits, 4, 4)
        Phase separator RDMs (real and imaginary parts)
    mixers_real, mixers_imag : cuda device arrays, shape (n_qubits, 2, 2)
        Single-qubit mixer operators
    fields_cost : cuda device array, shape (n_qubits,)
        Local fields in cost Hamiltonian (for <Zi> computation)
    couplings_cost : cuda device array, shape (n_qubits, n_qubits)
        Couplings in cost Hamiltonian (for <ZiZj> computation)

    Returns:
    --------
    ZiZj_array : numpy array, shape (n_qubits, n_qubits)
        Diagonal contains <Zi> values, upper triangle contains <ZiZj> values
    """
    n_qubits = couplings_cost.shape[0]

    # Allocate output
    ZiZj_out = cuda.to_device(np.zeros((n_qubits, n_qubits), dtype=np.float32))

    # Launch kernel
    blockspergrid_x = (n_qubits + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n_qubits + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _apply_mixers_and_extract_ZiZj_kernel[blockspergrid, threadsperblock](
        mixers_real,
        mixers_imag,
        rho_ij_real,
        rho_ij_imag,
        couplings_cost,
        fields_cost,
        ZiZj_out
    )

    ZiZj_out = ZiZj_out.copy_to_host()

    Zi_out = np.diag(ZiZj_out).copy()

    np.fill_diagonal(ZiZj_out, 0.0)

    # ZiZj_out.fill_diagonal(0.0)

    return ZiZj_out, Zi_out


@cuda.jit
def _compute_pauli_overlaps_kernel(rho_ij_real,
                                   rho_ij_imag,
                                   couplings_cost,
                                   pauli_overlaps_out):
    """
    Compute weighted sum of Pauli overlaps Tr(P_a ⊗ P_b @ ρ_ij) * J_ij for all 9 Pauli pairs.

    For Hermitian ρ, all traces are real and given by:
        Tr(X⊗X @ ρ) = 2*(Re(ρ₀₃) + Re(ρ₁₂))
        Tr(X⊗Y @ ρ) = 2*(Im(ρ₁₂) - Im(ρ₀₃))
        Tr(X⊗Z @ ρ) = 2*(Re(ρ₀₂) - Re(ρ₁₃))
        Tr(Y⊗X @ ρ) = -2*(Im(ρ₀₃) + Im(ρ₁₂))
        Tr(Y⊗Y @ ρ) = 2*(Re(ρ₁₂) - Re(ρ₀₃))
        Tr(Y⊗Z @ ρ) = -2*(Im(ρ₀₂) - Im(ρ₁₃))
        Tr(Z⊗X @ ρ) = 2*(Re(ρ₀₁) - Re(ρ₂₃))
        Tr(Z⊗Y @ ρ) = 2*(-Im(ρ₀₁) + Im(ρ₂₃))
        Tr(Z⊗Z @ ρ) = ρ₀₀ - ρ₁₁ - ρ₂₂ + ρ₃₃ = 2*(ρ₀₀+ρ₃₃)-1

    Each thread handles one qubit pair (i,j) and atomically adds to the 3x3 output.

    Parameters:
    -----------
    rho_ij_real, rho_ij_imag : device arrays, shape (n_qubits, n_qubits, 4, 4)
        Real and imaginary parts of phase separator RDMs
    couplings_cost : device array, shape (n_qubits, n_qubits)
        Cost Hamiltonian couplings (J_ij)
    pauli_overlaps_out : device array, shape (3, 3)
        Output: weighted sum of Pauli overlaps (X=0, Y=1, Z=2)
    """
    n = couplings_cost.shape[0]
    idx_qi, idx_qj = cuda.grid(2)

    # Only process upper triangle (i < j)
    if not (idx_qi < idx_qj < n):
        return

    # Skip if no coupling
    Jij = couplings_cost[idx_qi, idx_qj]
    if Jij == 0.0:
        return

    # Load the required density matrix elements
    # Real parts
    rho_real_00 = rho_ij_real[idx_qi, idx_qj, 0, 0]
    rho_real_01 = rho_ij_real[idx_qi, idx_qj, 0, 1]
    rho_real_02 = rho_ij_real[idx_qi, idx_qj, 0, 2]
    rho_real_03 = rho_ij_real[idx_qi, idx_qj, 0, 3]
    # rho_real_11 = rho_ij_real[idx_qi, idx_qj, 1, 1]
    rho_real_12 = rho_ij_real[idx_qi, idx_qj, 1, 2]
    rho_real_13 = rho_ij_real[idx_qi, idx_qj, 1, 3]
    # rho_real_22 = rho_ij_real[idx_qi, idx_qj, 2, 2]
    rho_real_23 = rho_ij_real[idx_qi, idx_qj, 2, 3]
    rho_real_33 = rho_ij_real[idx_qi, idx_qj, 3, 3]

    # Imaginary parts
    rho_imag_01 = rho_ij_imag[idx_qi, idx_qj, 0, 1]
    rho_imag_02 = rho_ij_imag[idx_qi, idx_qj, 0, 2]
    rho_imag_03 = rho_ij_imag[idx_qi, idx_qj, 0, 3]
    rho_imag_12 = rho_ij_imag[idx_qi, idx_qj, 1, 2]
    rho_imag_13 = rho_ij_imag[idx_qi, idx_qj, 1, 3]
    rho_imag_23 = rho_ij_imag[idx_qi, idx_qj, 2, 3]

    # Compute all 9 Pauli overlaps (indices: X=0, Y=1, Z=2)
    # Row 0: X⊗X, X⊗Y, X⊗Z
    tr_XX = 2.0 * (rho_real_03 + rho_real_12)
    tr_XY = 2.0 * (rho_imag_12 - rho_imag_03)
    tr_XZ = 2.0 * (rho_real_02 - rho_real_13)

    # Row 1: Y⊗X, Y⊗Y, Y⊗Z
    tr_YX = -2.0 * (rho_imag_03 + rho_imag_12)
    tr_YY = 2.0 * (rho_real_12 - rho_real_03)
    tr_YZ = -2.0 * (rho_imag_02 - rho_imag_13)

    # Row 2: Z⊗X, Z⊗Y, Z⊗Z
    tr_ZX = 2.0 * (rho_real_01 - rho_real_23)
    tr_ZY = 2.0 * (-rho_imag_01 + rho_imag_23)
    # 2*(ρ₀₀+ρ₃₃)-1
    tr_ZZ = 2.0 * (rho_real_00 + rho_real_33) - 1.
    # tr_ZZ = rho_real_00 - rho_real_11 - rho_real_22 + rho_real_33

    # Atomic add to output (weighted by coupling)
    cuda.atomic.add(pauli_overlaps_out, (0, 0), tr_XX * Jij)
    cuda.atomic.add(pauli_overlaps_out, (0, 1), tr_XY * Jij)
    cuda.atomic.add(pauli_overlaps_out, (0, 2), tr_XZ * Jij)
    cuda.atomic.add(pauli_overlaps_out, (1, 0), tr_YX * Jij)
    cuda.atomic.add(pauli_overlaps_out, (1, 1), tr_YY * Jij)
    cuda.atomic.add(pauli_overlaps_out, (1, 2), tr_YZ * Jij)
    cuda.atomic.add(pauli_overlaps_out, (2, 0), tr_ZX * Jij)
    cuda.atomic.add(pauli_overlaps_out, (2, 1), tr_ZY * Jij)
    cuda.atomic.add(pauli_overlaps_out, (2, 2), tr_ZZ * Jij)


def get_pauli_overlaps_cuda(rho_ij_real,
                            rho_ij_imag,
                            couplings_cost,
                            threadsperblock=(32, 32)):
    """
    Compute weighted Pauli overlaps: overlaps[a,b] = Σ_{i<j} J_ij * Tr(P_a⊗P_b @ ρ_ij)

    This function computes the weighted sum of two-qubit Pauli expectation values
    from the phase separator reduced density matrices. The output is a 3x3 matrix
    where indices correspond to X=0, Y=1, Z=2.

    Parameters:
    -----------
    rho_ij_real : cuda device array, shape (n_qubits, n_qubits, 4, 4)
        Real part of phase separator RDMs
    rho_ij_imag : cuda device array, shape (n_qubits, n_qubits, 4, 4)
        Imaginary part of phase separator RDMs
    couplings_cost : cuda device array, shape (n_qubits, n_qubits)
        Cost Hamiltonian couplings
    threadsperblock : tuple of int, optional
        CUDA thread block dimensions

    Returns:
    --------
    pauli_overlaps : numpy array, shape (3, 3)
        Weighted Pauli overlaps. Index mapping: 0=X, 1=Y, 2=Z
        pauli_overlaps[a,b] = Σ_{i<j} J_ij * Tr(P_a⊗P_b @ ρ_ij)
    """
    n_qubits = couplings_cost.shape[0]

    # Allocate output array on device (initialized to zero)
    pauli_overlaps_out = cuda.to_device(np.zeros((3, 3), dtype=np.float32))

    # Launch kernel
    blockspergrid_x = (n_qubits + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n_qubits + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _compute_pauli_overlaps_kernel[blockspergrid, threadsperblock](
        rho_ij_real,
        rho_ij_imag,
        couplings_cost,
        pauli_overlaps_out
    )

    return pauli_overlaps_out.copy_to_host()


def get_pauli_overlaps_cupy(rho_ij_real,
                            rho_ij_imag,
                            couplings_cost,
                            fields_cost=None):
    """
    Compute weighted Pauli overlaps using CuPy vectorized operations.

    This is faster than the CUDA kernel version because it uses optimized
    CuPy reduction operations instead of atomic adds.

    Parameters:
    -----------
    rho_ij_real : numba.cuda or cupy device array, shape (n_qubits, n_qubits, 4, 4)
        Real part of phase separator RDMs
    rho_ij_imag : numba.cuda or cupy device array, shape (n_qubits, n_qubits, 4, 4)
        Imaginary part of phase separator RDMs
    couplings_cost : numba.cuda or cupy device array, shape (n_qubits, n_qubits)
        Cost Hamiltonian couplings (only upper triangle used)

    Returns:
    --------
    pauli_overlaps : numpy array, shape (3, 3)
        Weighted Pauli overlaps. Index mapping: 0=X, 1=Y, 2=Z
    """
    t000 = time.perf_counter()
    # Zero-copy conversion from numba.cuda to cupy (via __cuda_array_interface__)
    rho_r = cp.asarray(rho_ij_real)
    rho_i = cp.asarray(rho_ij_imag)
    J = cp.asarray(couplings_cost)

    # Use upper triangle only (i < j) to avoid double counting
    J_upper = cp.triu(J, k=1)

    # Pre-weight the RDM elements by J (broadcast J over the 4x4 dimensions)
    # Shape: (n, n, 4, 4) * (n, n, 1, 1) -> (n, n, 4, 4)
    J_expanded = J_upper[:, :, None, None]

    t111 = time.perf_counter()
    t0 = time.perf_counter()



    t1 = time.perf_counter()

    average_rho_r = cp.sum(rho_r * J_expanded, axis=(0, 1))
    average_rho_i = cp.sum(rho_i * J_expanded, axis=(0, 1))

    overlaps = cp.zeros((3, 3), dtype=cp.float32)

    # X⊗X: 2*(Re(ρ₀₃) + Re(ρ₁₂))
    overlaps[0, 0] = 2.0 * (average_rho_r[0, 3] + average_rho_r[1, 2])
    # X⊗Y: 2*(Im(ρ₁₂) - Im(ρ₀₃))
    overlaps[0, 1] = 2.0 * (average_rho_i[1, 2] - average_rho_i[0, 3])
    # X⊗Z: 2*(Re(ρ₀₂) - Re(ρ₁₃))
    overlaps[0, 2] = 2.0 * (average_rho_r[0, 2] - average_rho_r[1, 3])
    # Y⊗X: -2*(Im(ρ₀₃) + Im(ρ₁₂))
    overlaps[1, 0] = -2.0 * (average_rho_i[0, 3] + average_rho_i[1, 2])
    # Y⊗Y: 2*(Re(ρ₁₂) - Re(ρ₀₃))
    overlaps[1, 1] = 2.0 * (average_rho_r[1, 2] - average_rho_r[0, 3])
    # Y⊗Z: -2*(Im(ρ₀₂) - Im(ρ₁₃))
    overlaps[1, 2] = -2.0 * (average_rho_i[0, 2] - average_rho_i[1, 3])
    # Z⊗X: 2*(Re(ρ₀₁) - Re(ρ₂₃))
    overlaps[2, 0] = 2.0 * (average_rho_r[0, 1] - average_rho_r[2, 3])
    # Z⊗Y: -2*(Im(ρ₀₁) - Im(ρ₂₃))
    overlaps[2, 1] = -2.0 * (average_rho_i[0, 1] - average_rho_i[2, 3])
    # Z⊗Z: ρ₀₀ - ρ₁₁ - ρ₂₂ + ρ₃₃ = ρ₀₀  + ρ₃₃ - (ρ₁₁ + ρ₂₂) = ρ₀₀  + ρ₃₃ - (1-ρ₀₀ -ρ₃₃ ) = 2*(ρ₀₀+ρ₃₃)-1
    overlaps[2, 2] = average_rho_r[0, 0] - average_rho_r[1, 1] - average_rho_r[2, 2] + average_rho_r[3, 3]

    t2 = time.perf_counter()

    if fields_cost is None:
        return overlaps, None

    fields_cost = cp.asarray(fields_cost)
    n = fields_cost.shape[0]
    idx: ndarray = cp.arange(n)
    #
    # # Extract diagonal 2x2 blocks: shape (n, 2, 2)
    rho_1q_real = rho_r[idx, idx, :2, :2]
    rho_1q_imag = rho_i[idx, idx, :2, :2]

    # t3 = time.perf_counter()

    t4 = time.perf_counter()

    fields_expanded = fields_cost[:, None, None]
    average_rho_1q_r = cp.sum(fields_expanded * rho_1q_real, axis=0)
    average_rho_1q_i = cp.sum(fields_expanded * rho_1q_imag, axis=0)
    overlaps_1q = cp.zeros((3,), dtype=cp.float32)

    # X: Re(ρ₀₁) + Re(ρ₁₀)
    overlaps_1q[0] = 2 * average_rho_1q_r[0, 1]
    # Y: Im(ρ₀₁) - Im(ρ₁₀)
    overlaps_1q[1] = -2 * average_rho_1q_i[0, 1]
    # Z: Re(ρ₀₀) - Re(ρ₁₁)
    overlaps_1q[2] = average_rho_1q_r[0, 0] - average_rho_1q_r[1, 1]

    t5 = time.perf_counter()


    return overlaps, overlaps_1q



def get_pauli_overlaps_numpy(rho_ij,
                             couplings_cost,
                             fields_cost=None):
    """
    NumPy analogue of get_pauli_overlaps_cupy for CPU backends (cython/numpy).

    Parameters:
    -----------
    rho_ij : numpy complex array, shape (n_qubits, n_qubits, 4, 4)
        Phase separator RDMs; [i, j] (i<j) holds the two-qubit RDM and the
        diagonal [i, i, 0:2, 0:2] holds the single-qubit RDM (same layout as the
        cython/cuda builders). Entries with zero cost coefficient may be left
        as zeros -- they are weighted out of the sums anyway.
    couplings_cost : numpy array, shape (n_qubits, n_qubits)
        Cost Hamiltonian couplings (only upper triangle used)
    fields_cost : numpy array, shape (n_qubits,), optional
        Cost Hamiltonian local fields; if None, the 1q overlaps are skipped.

    Returns:
    --------
    (pauli_overlaps, pauli_overlaps_1q) :
        (3, 3) array with overlaps[a,b] = Σ_{i<j} J_ij * Tr(P_a⊗P_b @ ρ_ij)
        and (3,) array with overlaps_1q[a] = Σ_i h_i * Tr(P_a @ ρ_i)
        (None if fields_cost is None). Index mapping: 0=X, 1=Y, 2=Z.
    """
    rho_r = np.real(rho_ij)
    rho_i = np.imag(rho_ij)

    # Use upper triangle only (i < j) to avoid double counting
    J_upper = np.triu(np.asarray(couplings_cost), k=1)
    J_expanded = J_upper[:, :, None, None]

    average_rho_r = np.sum(rho_r * J_expanded, axis=(0, 1))
    average_rho_i = np.sum(rho_i * J_expanded, axis=(0, 1))

    overlaps = np.zeros((3, 3), dtype=np.float32)

    # X⊗X: 2*(Re(ρ₀₃) + Re(ρ₁₂))
    overlaps[0, 0] = 2.0 * (average_rho_r[0, 3] + average_rho_r[1, 2])
    # X⊗Y: 2*(Im(ρ₁₂) - Im(ρ₀₃))
    overlaps[0, 1] = 2.0 * (average_rho_i[1, 2] - average_rho_i[0, 3])
    # X⊗Z: 2*(Re(ρ₀₂) - Re(ρ₁₃))
    overlaps[0, 2] = 2.0 * (average_rho_r[0, 2] - average_rho_r[1, 3])
    # Y⊗X: -2*(Im(ρ₀₃) + Im(ρ₁₂))
    overlaps[1, 0] = -2.0 * (average_rho_i[0, 3] + average_rho_i[1, 2])
    # Y⊗Y: 2*(Re(ρ₁₂) - Re(ρ₀₃))
    overlaps[1, 1] = 2.0 * (average_rho_r[1, 2] - average_rho_r[0, 3])
    # Y⊗Z: -2*(Im(ρ₀₂) - Im(ρ₁₃))
    overlaps[1, 2] = -2.0 * (average_rho_i[0, 2] - average_rho_i[1, 3])
    # Z⊗X: 2*(Re(ρ₀₁) - Re(ρ₂₃))
    overlaps[2, 0] = 2.0 * (average_rho_r[0, 1] - average_rho_r[2, 3])
    # Z⊗Y: -2*(Im(ρ₀₁) - Im(ρ₂₃))
    overlaps[2, 1] = -2.0 * (average_rho_i[0, 1] - average_rho_i[2, 3])
    # Z⊗Z: ρ₀₀ - ρ₁₁ - ρ₂₂ + ρ₃₃
    overlaps[2, 2] = average_rho_r[0, 0] - average_rho_r[1, 1] - average_rho_r[2, 2] + average_rho_r[3, 3]

    if fields_cost is None:
        return overlaps, None

    fields_cost = np.asarray(fields_cost)
    n = fields_cost.shape[0]
    idx = np.arange(n)

    # Extract diagonal 2x2 blocks: shape (n, 2, 2)
    rho_1q_real = rho_r[idx, idx, :2, :2]
    rho_1q_imag = rho_i[idx, idx, :2, :2]

    fields_expanded = fields_cost[:, None, None]
    average_rho_1q_r = np.sum(fields_expanded * rho_1q_real, axis=0)
    average_rho_1q_i = np.sum(fields_expanded * rho_1q_imag, axis=0)
    overlaps_1q = np.zeros((3,), dtype=np.float32)

    # X: Re(ρ₀₁) + Re(ρ₁₀)
    overlaps_1q[0] = 2 * average_rho_1q_r[0, 1]
    # Y: Im(ρ₀₁) - Im(ρ₁₀)
    overlaps_1q[1] = -2 * average_rho_1q_i[0, 1]
    # Z: Re(ρ₀₀) - Re(ρ₁₁)
    overlaps_1q[2] = average_rho_1q_r[0, 0] - average_rho_1q_r[1, 1]

    return overlaps, overlaps_1q
