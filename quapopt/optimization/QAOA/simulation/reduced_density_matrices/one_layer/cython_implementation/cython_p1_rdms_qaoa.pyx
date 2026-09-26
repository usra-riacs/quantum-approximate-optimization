import numpy as np
cimport numpy as np

ctypedef fused floating:
    np.float32_t
    np.float64_t
from libc.math cimport sin, cos

def get_precomputed_phase_data_cython(floating phase_angle,
                                      np.ndarray[floating, ndim=2] couplings_phase,
                                      np.ndarray[floating, ndim=1] fields_phase,
                                      np.ndarray[floating, ndim=1] couplings_sums,

                                      ):
    cdef unsigned int number_of_qubits = fields_phase.shape[0]


    cdef np.ndarray[np.complex64_t, ndim=2] couplings_sums_gamma_1j = np.zeros((number_of_qubits, number_of_qubits),
                                                                               dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=2] minus_four_gamma_1j_couplings = np.zeros(
        (number_of_qubits, number_of_qubits),
        dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=2] gamma_1j_couplings = np.zeros((number_of_qubits, number_of_qubits),
                                                                          dtype=np.complex64)

    cdef unsigned int i, j, k
    cdef floating gamma = phase_angle
    cdef floating four_gamma_1j = 4.0 * gamma
    cdef floating gamma_1j = gamma
    cdef floating cij_4
    cdef floating cij_gamma

    for i in range(number_of_qubits):
        for j in range(number_of_qubits):
            if i == j:
                continue
            cij_4 = four_gamma_1j * couplings_phase[i, j]
            cij_gamma = gamma_1j * couplings_phase[i, j]
            minus_four_gamma_1j_couplings[i, j].real = cos(cij_4)
            minus_four_gamma_1j_couplings[i, j].imag = -sin(cij_4)

            gamma_1j_couplings[i, j].real = cos(cij_gamma)
            gamma_1j_couplings[i, j].imag = sin(cij_gamma)


    # Compute exp(i*gamma*(couplings_sums[i] - couplings_phase[i,j])) for all i,j
    cdef floating val_ij, val_ji, cos_ij, sin_ij, cos_ji, sin_ji, cij, val_ii
    for i in range(number_of_qubits):
        # Diagonal element for single-qubit case: exp(i*gamma*couplings_sums[i])
        # couplings_sums[i] = sum_k J_ik + h_i (already includes local field)
        val_ii = gamma * couplings_sums[i]
        couplings_sums_gamma_1j[i, i].real = cos(val_ii)
        couplings_sums_gamma_1j[i, i].imag = sin(val_ii)

        for j in range(i + 1, number_of_qubits):
            cij = couplings_phase[i, j]  # symmetric: couplings_phase[i,j] == couplings_phase[j,i]

            # Entry [i,j] uses couplings_sums[i]
            val_ij = gamma * (couplings_sums[i] - cij)
            cos_ij = cos(val_ij)
            sin_ij = sin(val_ij)
            couplings_sums_gamma_1j[i, j].real = cos_ij
            couplings_sums_gamma_1j[i, j].imag = sin_ij

            # Entry [j,i] uses couplings_sums[j] - DIFFERENT!
            val_ji = gamma * (couplings_sums[j] - cij)
            cos_ji = cos(val_ji)
            sin_ji = sin(val_ji)
            couplings_sums_gamma_1j[j, i].real = cos_ji
            couplings_sums_gamma_1j[j, i].imag = sin_ji

    return couplings_sums_gamma_1j, minus_four_gamma_1j_couplings, gamma_1j_couplings

# def get_RDM_phase_separator_ij_cython(int idx_qi,
#                                       int idx_qj,
#                                       floating gamma,
#                                       np.ndarray[np.complex64_t, ndim=2] initial_states_array,
#                                       np.ndarray[floating, ndim=2] couplings_phase,
#                                       np.ndarray[floating, ndim=1] fields_phase,
#                                       np.ndarray[np.complex64_t, ndim=2] couplings_sums_gamma_1j,
#                                       np.ndarray[np.complex64_t, ndim=2] minus_four_gamma_1j_couplings,
#                                       np.ndarray[np.complex64_t, ndim=2] gamma_1j_couplings,
#
#                                       ):
#     cdef unsigned int number_of_qubits = fields_phase.shape[0]
#     cdef np.ndarray[np.complex64_t, ndim=2] qi_ket = initial_states_array[idx_qi, :, None].copy()
#     cdef np.ndarray[np.complex64_t, ndim=2] qj_ket = initial_states_array[idx_qj, :, None].copy()
#     cdef np.ndarray[np.complex64_t, ndim=2] qi_rho
#     cdef np.ndarray[np.complex64_t, ndim=2] qj_rho
#
#     if couplings_phase[idx_qi, idx_qj] == 0.0:
#         qi_rho, qj_rho = qi_ket * np.matrix.getH(qi_ket), qj_ket * np.matrix.getH(qj_ket)
#         return np.kron(qi_rho, qj_rho)
#
#     qi_ket[0] *= np.conj(couplings_sums_gamma_1j[idx_qi, idx_qj])
#     qi_ket[1] *= couplings_sums_gamma_1j[idx_qi, idx_qj]
#
#     qj_ket[0] *= np.conj(couplings_sums_gamma_1j[idx_qj, idx_qi])
#     qj_ket[1] *= couplings_sums_gamma_1j[idx_qj, idx_qi]
#
#     # Switch to density matrices and compute the effect of
#     # the two-qubit CP gate that comes from qubit k neq i,j
#     qi_rho, qj_rho = qi_ket * np.matrix.getH(qi_ket), qj_ket * np.matrix.getH(qj_ket)
#     rho_ij = np.kron(qi_rho, qj_rho)
#
#     for k in range(number_of_qubits):
#         if k in {idx_qi, idx_qj}:
#             continue
#         if couplings_phase[idx_qi, k] == 0.0 and couplings_phase[idx_qj, k] == 0.0:
#             continue
#         phase_i = minus_four_gamma_1j_couplings[idx_qi, k]
#         phase_j = minus_four_gamma_1j_couplings[idx_qj, k]
#         u1_ij = np.diag([1.0, phase_j, phase_i, phase_i * phase_j])
#         rho_ij = 0.5 * rho_ij + 0.5 * np.dot(u1_ij, np.dot(rho_ij, np.matrix.getH(u1_ij)))
#
#     # Apply the two-qubit Rzz gate between `i` and `j`
#     if couplings_phase[idx_qi, idx_qj] != 0.0:
#         phase_p = gamma_1j_couplings[idx_qi, idx_qj]
#         phase_m = np.conj(gamma_1j_couplings[idx_qi, idx_qj])
#         u_ij = np.diag([phase_m, phase_p, phase_p, phase_m])
#         rho_ij = np.dot(u_ij, np.dot(rho_ij, np.matrix.getH(u_ij)))
#
#     return rho_ij

def get_all_rho_ij_cython(
        floating gamma,
        np.ndarray[np.complex64_t, ndim=2] initial_states_array,
        np.ndarray[floating, ndim=2] couplings_phase,
        np.ndarray[floating, ndim=1] fields_phase,
        np.ndarray[floating, ndim=1] couplings_sums,
        np.ndarray[floating, ndim=1] ws_bias_parameters,
        rdm_mask=None,
):
    # rdm_mask: boolean (n, n) array; the diagonal gates the one-qubit RDMs, the upper triangle the pairs.
    # None computes every entry. Entries left out stay 0.
    couplings_sums_gamma_1j, minus_four_gamma_1j_couplings, gamma_1j_couplings = get_precomputed_phase_data_cython(
        gamma,
        couplings_phase,
        fields_phase,
    couplings_sums)
    cdef unsigned int number_of_qubits = fields_phase.shape[0]
    cdef np.ndarray[np.complex64_t, ndim=4] rho_ij_array = np.zeros((number_of_qubits, number_of_qubits, 4, 4),
                                                                    dtype=np.complex64)
    cdef np.ndarray[np.uint8_t, ndim=2] rdm_mask_u8
    if rdm_mask is None:
        rdm_mask_u8 = np.ones((number_of_qubits, number_of_qubits), dtype=np.uint8)
    else:
        rdm_mask_u8 = np.ascontiguousarray(rdm_mask, dtype=np.uint8)
    cdef unsigned int idx_qi, idx_qj, k
    cdef np.complex64_t phase_i, phase_j, phase_p, phase_m, cij_qi, cij_qj

    cdef np.ndarray[np.complex64_t, ndim=2] qi_ket
    cdef np.ndarray[np.complex64_t, ndim=2] qj_ket

    cdef np.ndarray[np.complex64_t, ndim=2] qi_rho
    cdef np.ndarray[np.complex64_t, ndim=2] qj_rho

    cdef np.ndarray[np.complex64_t, ndim=2] rho_ij
    cdef np.ndarray[np.complex64_t, ndim=2] u1_ij
    cdef np.ndarray[np.complex64_t, ndim=2] u_ij

    cdef np.ndarray[np.complex64_t, ndim=2] intermediate = np.empty((4, 4), dtype=np.complex64)


    cdef floating c, one_minus_c


    for idx_qi in range(number_of_qubits):
        for idx_qj in range(idx_qi, number_of_qubits):

            if rdm_mask_u8[idx_qi, idx_qj] == 0:
                continue

            cij_qi = couplings_sums_gamma_1j[idx_qi, idx_qj]


            if idx_qi == idx_qj:
                qi_ket = np.array([[initial_states_array[idx_qi, 0] * np.conj(cij_qi)],
                                   [initial_states_array[idx_qi, 1] * cij_qi]], dtype=np.complex64)

                rho_i = qi_ket * np.matrix.getH(qi_ket)

                # Tracing out qubit k weights the two branches of its controlled phase by k's own bias.
                for k in range(number_of_qubits):
                    if k == idx_qi:
                        continue
                    if couplings_phase[idx_qi, k] == 0.0:
                        continue
                    c = ws_bias_parameters[k]
                    one_minus_c = 1.0 - c
                    phase_i = minus_four_gamma_1j_couplings[idx_qi, k]

                    rho_i[1, 0] = one_minus_c * rho_i[1, 0] + c * phase_i * rho_i[1, 0]
                    rho_i[0, 1] = np.conj(rho_i[1, 0])

                rho_ij_array[idx_qi,idx_qj,0:2,0:2] = rho_i
                continue


            cij_qj = couplings_sums_gamma_1j[idx_qj, idx_qi]  # Note: swapped indices!

            qi_ket = np.array([[initial_states_array[idx_qi, 0]*np.conj(cij_qi)],
                               [initial_states_array[idx_qi, 1]*cij_qi]], dtype=np.complex64)
            qj_ket = np.array([[initial_states_array[idx_qj, 0]*np.conj(cij_qj)],
                               [initial_states_array[idx_qj, 1]*cij_qj]], dtype=np.complex64)

            # Switch to density matrices and compute the effect of
            # the two-qubit CP gate that comes from qubit k neq i,j
            qi_rho = qi_ket * np.matrix.getH(qi_ket)
            qj_rho = qj_ket * np.matrix.getH(qj_ket)
            rho_ij = np.kron(qi_rho, qj_rho)

            for k in range(number_of_qubits):
                if k in {idx_qi, idx_qj}:
                    continue
                if couplings_phase[idx_qi, k] == 0.0 and couplings_phase[idx_qj, k] == 0.0:
                    continue
                c = ws_bias_parameters[k]
                one_minus_c = 1.0 - c
                phase_i = minus_four_gamma_1j_couplings[idx_qi, k]
                phase_j = minus_four_gamma_1j_couplings[idx_qj, k]


                u1_ij = np.array([[1.0], [phase_j], [phase_i], [phase_i * phase_j]],dtype=np.complex64)
                rho_ij = rho_ij * one_minus_c + c* rho_ij * u1_ij * np.matrix.getH(u1_ij)

            # Apply the two-qubit Rzz gate between `i` and `j`
            if couplings_phase[idx_qi, idx_qj] != 0.0:
                phase_p = gamma_1j_couplings[idx_qi, idx_qj]
                phase_m = np.conj(gamma_1j_couplings[idx_qi, idx_qj])

                u_ij = np.array([[phase_m], [phase_p], [phase_p], [phase_m]],dtype=np.complex64)
                rho_ij = rho_ij*u_ij*np.matrix.getH(u_ij)

            rho_ij_array[idx_qi, idx_qj, :, :] = rho_ij

    return rho_ij_array

def get_all_ZiZj_cython(floating gamma,
                        floating beta,
                        np.ndarray[np.complex64_t, ndim=4] local_RDMS,
                        np.ndarray[np.complex64_t, ndim=3] mixers_array,
                        np.ndarray[floating, ndim=2] couplings_phase,
                        np.ndarray[floating, ndim=1] fields_phase,
                        np.ndarray[floating, ndim=2] couplings_cost,
                        np.ndarray[floating, ndim=1] fields_cost,
                        np.ndarray[np.complex64_t, ndim=2] initial_states_array,
                        np.ndarray[floating, ndim=1] couplings_sums,
                       # np.ndarray[floating, ndim=1] ws_bias_parameters,
                        ):
    cdef unsigned int number_of_qubits = fields_phase.shape[0]
    cdef np.ndarray[floating, ndim=2] ZiZj_array = np.zeros((number_of_qubits, number_of_qubits),
                                                            dtype=couplings_phase.dtype)
    cdef np.ndarray[floating, ndim=1] Zi_array = np.zeros((number_of_qubits,), dtype=couplings_phase.dtype)
    cdef unsigned int idx_qi, idx_qj, k

    cdef np.ndarray[np.complex64_t, ndim=2] rho_ij
    cdef np.ndarray[np.complex64_t, ndim=2] mixer_qiqj
    cdef np.ndarray[np.complex64_t, ndim=2] rho_ij_phase
    cdef np.ndarray[np.complex64_t, ndim=2] mixer_qi
    cdef np.ndarray[np.complex64_t, ndim=2] mixer_qj

    # Variables for single-qubit Zi computation
    cdef np.ndarray[np.complex64_t, ndim=2] qi_ket
    cdef np.ndarray[np.complex64_t, ndim=2] rho_i
    cdef np.ndarray[np.complex64_t, ndim=2] rho_i_final
    cdef np.ndarray[np.complex64_t, ndim=2] u1_diag
    cdef np.complex64_t phase_sum, phase_i
    cdef floating c_k, one_minus_c_k, val_ii, cij_4
    cdef floating cos_ii, sin_ii

    for idx_qi in range(number_of_qubits):
        mixer_qi = mixers_array[idx_qi, :, :]

        # Single-qubit Zi computation (when fields_cost[idx_qi] != 0)
        if fields_cost[idx_qi] != 0.0:

            rho_i_phase = local_RDMS[idx_qi, idx_qi, 0:2, 0:2]
            rho_i = mixer_qi @ rho_i_phase @ np.matrix.getH(mixer_qi)
            # Compute <Zi> = rho[1,1] - rho[0,0]
            Zi_array[idx_qi] = np.real(-rho_i[1, 1] + rho_i[0, 0])

        # Two-qubit ZiZj computation
        for idx_qj in range(idx_qi + 1, number_of_qubits):
            if couplings_cost[idx_qi, idx_qj] == 0.0:
                continue
            mixer_qj = mixers_array[idx_qj, :, :]
            rho_ij_phase = local_RDMS[idx_qi, idx_qj, :, :]
            mixer_qiqj = np.kron(mixer_qi, mixer_qj)
            rho_ij = mixer_qiqj @ rho_ij_phase @ np.matrix.getH(mixer_qiqj)
            ZiZj_array[idx_qi, idx_qj] = np.real(rho_ij[0, 0] - rho_ij[1, 1] - rho_ij[2, 2] + rho_ij[3, 3])

    return ZiZj_array, Zi_array
