# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 

import time

import numpy as np
from numba import cuda


@cuda.jit
def _product_kernel_main_local_terms(fields_phase,
                                     fields_cost,
                                     correlations_phase,
                                     cos_correlations_phase,
                                     gamma,
                                     sin_2_beta,
                                     return_only_sum,
                                     Ci_expected_values_out):
    i = cuda.grid(1)
    if i >= fields_phase.shape[0]:
        return

    # both those situations mean that the result is 0
    if fields_cost[i] != 0 and fields_phase[i] != 0:
        product_of_cosines = 1.
        for k in range(fields_phase.shape[0]):
            # we check if the edge exists
            # for i=k correlations_phase[i,k] is 0, so we don't need to check it
            # if correlations_phase[i, k] == 0:
            if correlations_phase[i, k] == 0:
                continue
            # if despite edge being present, the cosine term is 0, the result is 0
            if cos_correlations_phase[i, k] == 0:
                product_of_cosines = 0.0
                break
            # print('cuda',i,k,cos_ik_matrix[i,k])
            product_of_cosines *= cos_correlations_phase[i, k]
        # print('cuda', i, product_of_cosines)
        if product_of_cosines != 0.0:
            the_part = fields_cost[i] * sin_2_beta * np.sin(
                gamma * fields_phase[i]) * product_of_cosines
            if return_only_sum:
                cuda.atomic.add(Ci_expected_values_out, 0, the_part)
            else:
                Ci_expected_values_out[i] = the_part


@cuda.jit
def _kernel_cosine_and_sine_correlations(correlations_phase,
                                         gamma,
                                         cos_correlations_phase_out,
                                         sin_correlations_phase_out):
    n = correlations_phase.shape[0]
    i, j = cuda.grid(2)

    if not (i < j < n):
        return

    arg = gamma * correlations_phase[i, j]
    if arg == 0:
        return

    cos_2_gamma = np.cos(arg)
    sin_2_gamma = np.sin(arg)

    cos_correlations_phase_out[i, j] = cos_2_gamma
    cos_correlations_phase_out[j, i] = cos_2_gamma
    sin_correlations_phase_out[i, j] = sin_2_gamma
    sin_correlations_phase_out[j, i] = sin_2_gamma


@cuda.jit
def _kernel_cosine_vector_fields(fields_phase,
                                 gamma,
                                 cosine_fields_out):
    i = cuda.grid(1)
    if i >= fields_phase.shape[0]:
        return
    # cosine_fields_out[i] = np.cos(2. * gamma * fields_phase[i])
    cosine_fields_out[i] = np.cos(gamma * fields_phase[i])


@cuda.jit
def _kernel_sine_vector_fields(fields_phase,
                               gamma,
                               sine_fields_out):
    i = cuda.grid(1)
    if i >= fields_phase.shape[0]:
        return
    # sine_fields_out[i] = np.sin(2. * gamma * fields_phase[i])
    sine_fields_out[i] = np.sin(gamma * fields_phase[i])






@cuda.jit
def _kernel_product_formulas_C_ij(correlations_cost,
                                  correlations_phase,
                                  cos_correlations_phase,
                                  gamma,
                                  product_formulas_array_out):
    ind_1, ind_2 = cuda.grid(2)
    n = correlations_cost.shape[0]

    run_check = (ind_1 < ind_2 < n)

    if not run_check:
        return

    # First term
    product1, product2 = 1., 1.
    # Second term
    product3, product4 = 1., 1.
    # Third term
    product5, product6 = 1., 1.
    for k in range(n):
        if k == ind_1 or k == ind_2:
            continue
        # if correlations_phase[ind_1, k] != 0:
        if correlations_phase[ind_1, k] != 0:
            cos_ik = cos_correlations_phase[ind_1, k]
            if cos_ik == 0.0:
                product1 = 0.0
                product3 = 0.0
            else:
                if product1 != 0.0:
                    product1 *= cos_ik
                # if correlations_phase[ind_2, k] == 0 and product3 != 0:
                if correlations_phase[ind_2, k] == 0 and product3 != 0:
                    product3 *= cos_ik

        # if correlations_phase[ind_2, k] != 0:
        if correlations_phase[ind_2, k] != 0:
            cos_jk = cos_correlations_phase[ind_2, k]
            if cos_jk == 0.0:
                product2 = 0.0
                product4 = 0.0
            else:
                if product2 != 0.0:
                    product2 *= cos_jk
                # if correlations_phase[ind_1, k] == 0 and product4 != 0:
                if correlations_phase[ind_1, k] == 0 and product4 != 0:
                    product4 *= cos_jk

        if correlations_phase[ind_1, k] != 0 and correlations_phase[ind_2, k] != 0:
            # product5 *= np.cos(2. * gamma * (correlations_phase[ind_1, k] + correlations_phase[ind_2, k]))
            # product6 *= np.cos(2. * gamma * (correlations_phase[ind_1, k] - correlations_phase[ind_2, k]))
            product5 *= np.cos(gamma * (correlations_phase[ind_1, k] + correlations_phase[ind_2, k]))
            product6 *= np.cos(gamma * (correlations_phase[ind_1, k] - correlations_phase[ind_2, k]))

    product_formulas_array_out[ind_1, ind_2, 0] = product1
    product_formulas_array_out[ind_1, ind_2, 1] = product2
    product_formulas_array_out[ind_1, ind_2, 2] = product3
    product_formulas_array_out[ind_1, ind_2, 3] = product4
    product_formulas_array_out[ind_1, ind_2, 4] = product5
    product_formulas_array_out[ind_1, ind_2, 5] = product6


@cuda.jit
def _kernel_calculate_C_ij(fields_phase,
                           correlations_cost,
                           sin_correlations_phase,
                           cosine_fields,
                           product_formulas_array,
                           gamma,
                           sin_4_beta_by_2,
                           sin_2_beta_squared_by_2,
                           return_only_sum,
                           C_ij_expected_values_out):
    idx_1, idx_2 = cuda.grid(2)
    n = fields_phase.shape[0]

    run_check = idx_1 < idx_2 < n and correlations_cost[idx_1, idx_2] != 0
    if not run_check:
        return

    h_i_phase = fields_phase[idx_1]
    h_j_phase = fields_phase[idx_2]
    J_ij_cost = correlations_cost[idx_1, idx_2]

    product1 = product_formulas_array[idx_1, idx_2][0]
    product2 = product_formulas_array[idx_1, idx_2][1]
    product3 = product_formulas_array[idx_1, idx_2][2]
    product4 = product_formulas_array[idx_1, idx_2][3]
    product5 = product_formulas_array[idx_1, idx_2][4]
    product6 = product_formulas_array[idx_1, idx_2][5]

    # First term
    B = J_ij_cost*sin_correlations_phase[idx_1, idx_2]
    B *= cosine_fields[idx_1] * product1 + cosine_fields[idx_2] * product2

    # Second term
    C = J_ij_cost * product3 * product4

    C *= (np.cos(gamma * (h_i_phase + h_j_phase)) * product5 - np.cos(
        gamma * (h_i_phase - h_j_phase)) * product6)

    C*= -1

    term_ij = sin_4_beta_by_2*B + sin_2_beta_squared_by_2*C

    if return_only_sum:
        cuda.atomic.add(C_ij_expected_values_out, (0, 0), term_ij)
    else:
        C_ij_expected_values_out[idx_1, idx_2] = term_ij


@cuda.jit
def _kernel_calculate_BC(fields_phase,
                           correlations_cost,
                           sin_correlations_phase,
                           cosine_fields,
                           product_formulas_array,
                           gamma,
                           BC_values_out):
    idx_1, idx_2 = cuda.grid(2)
    n = fields_phase.shape[0]

    run_check = idx_1 < idx_2 < n and correlations_cost[idx_1, idx_2] != 0
    if not run_check:
        return

    h_i_phase = fields_phase[idx_1]
    h_j_phase = fields_phase[idx_2]
    J_ij_cost = correlations_cost[idx_1, idx_2]

    product1 = product_formulas_array[idx_1, idx_2][0]
    product2 = product_formulas_array[idx_1, idx_2][1]
    product3 = product_formulas_array[idx_1, idx_2][2]
    product4 = product_formulas_array[idx_1, idx_2][3]
    product5 = product_formulas_array[idx_1, idx_2][4]
    product6 = product_formulas_array[idx_1, idx_2][5]

    # First term
    B = J_ij_cost*sin_correlations_phase[idx_1, idx_2]
    B *= cosine_fields[idx_1] * product1 + cosine_fields[idx_2] * product2

    # Second term
    C = J_ij_cost * product3 * product4
    C *= np.cos(gamma * (h_i_phase + h_j_phase)) * product5 - np.cos(
        gamma * (h_i_phase - h_j_phase)) * product6
    C*=-1

    cuda.atomic.add(BC_values_out, 0, B)
    cuda.atomic.add(BC_values_out, 1, C)



#
#
#
# @cuda.jit
# def _kernel_calculate_BC(fields_phase,
#                          correlations_cost,
#                          sin_correlations_phase,
#                          cosine_fields,
#                          product_formulas_array,
#                          gamma,
#                          BC_values_out):
#     idx_u, idx_v = cuda.grid(2)
#     n = fields_phase.shape[0]
#
#     if idx_v <= idx_u:
#         return
#     if idx_v >= n:
#         return
#     if correlations_cost[idx_u, idx_v] == 0:
#         return
#
#     # print('hej', i, j, C_ij_expected_values_out[0, 0])
#
#     h_u_phase = fields_phase[idx_u]
#     h_v_phase = fields_phase[idx_v]
#
#     J_uv_cost = correlations_cost[idx_u, idx_v]
#     sin_uv_phase = sin_correlations_phase[idx_u, idx_v]
#
#     cos_u = cosine_fields[idx_u]
#     cos_v = cosine_fields[idx_v]
#
#     product1 = product_formulas_array[idx_u, idx_v, 0]
#     product2 = product_formulas_array[idx_u, idx_v, 1]
#     product3 = product_formulas_array[idx_u, idx_v, 2]
#     product4 = product_formulas_array[idx_u, idx_v, 3]
#     product5 = product_formulas_array[idx_u, idx_v, 4]
#     product6 = product_formulas_array[idx_u, idx_v, 5]
#
#     B_uv = J_uv_cost * sin_uv_phase * (cos_u * product1 + cos_v * product2)
#
#     term_2 = np.cos(gamma * (h_u_phase + h_v_phase)) * product5
#     term_3 = np.cos(gamma * (h_u_phase - h_v_phase)) * product6
#
#
#     C_uv = -J_uv_cost * product3 * product4 * (term_2 - term_3)
#
#     # BC_values_out[idx_u, idx_v, 0] = B_uv
#     # BC_values_out[idx_u, idx_v, 1] = C_uv
#
#     # BC_values_out[0] += B_uv
#     # BC_values_out[1] += C_uv
#     cuda.atomic.add(BC_values_out, 0, B_uv)
#     cuda.atomic.add(BC_values_out, 1, C_uv)


def _get_part_1_cuda(correlations_phase,
                     gamma,
                     float_precision=np.float64):
    gamma = 2 * gamma

    number_of_qubits = correlations_phase.shape[0]
    n = number_of_qubits

    threads_per_block = (32, 32)
    blocks_per_grid = ((n + threads_per_block[0] - 1) // threads_per_block[0],
                       (n + threads_per_block[1] - 1) // threads_per_block[1])

    cos_correlations_phase = cuda.to_device(np.ones((n, n), dtype=float_precision))
    sin_correlations_phase = cuda.to_device(np.zeros((n, n), dtype=float_precision))

    _kernel_cosine_and_sine_correlations[blocks_per_grid, threads_per_block](correlations_phase,
                                                                             gamma,
                                                                             cos_correlations_phase,
                                                                             sin_correlations_phase)

    cuda.synchronize()

    return cos_correlations_phase, sin_correlations_phase


def _get_part_2_cuda(fields_phase,
                     fields_cost,
                     correlations_phase,
                     cos_correlations_phase,
                     gamma,
                     beta,
                     float_precision=np.float64,
                     return_only_sum=False
                     ):
    gamma = 2 * gamma
    beta = 2 * beta

    number_of_qubits = fields_phase.shape[0]
    n = number_of_qubits

    threads_per_block = 256 * 4
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    if return_only_sum:
        Ci_expected_values = cuda.to_device(np.zeros(1, dtype=float_precision))
    else:
        Ci_expected_values = cuda.to_device(np.zeros(n, dtype=float_precision))

    _product_kernel_main_local_terms[blocks_per_grid, threads_per_block](fields_phase,
                                                                         fields_cost,
                                                                         correlations_phase,
                                                                         cos_correlations_phase,
                                                                         gamma,
                                                                         np.sin(beta),
                                                                         return_only_sum,
                                                                         Ci_expected_values)
    cuda.synchronize()

    if return_only_sum:
        return Ci_expected_values.copy_to_host()[0]
    return Ci_expected_values.copy_to_host()


def _get_part_3_cuda(correlations_cost,
                     correlations_phase,
                     cos_correlations_phase,
                     gamma,
                     float_precision=np.float64
                     ):
    gamma = 2 * gamma

    # number_of_qubits = correlations_cost.shape[0]
    n = correlations_cost.shape[0]
    threads_per_block = (32, 32)
    blocks_per_grid = ((n + threads_per_block[0] - 1) // threads_per_block[0],
                       (n + threads_per_block[1] - 1) // threads_per_block[1])
    product_formulas_array = cuda.to_device(np.zeros((n, n, 6), dtype=float_precision))
    _kernel_product_formulas_C_ij[blocks_per_grid, threads_per_block](correlations_cost,
                                                                      correlations_phase,
                                                                      cos_correlations_phase,
                                                                      gamma,
                                                                      product_formulas_array)
    cuda.synchronize()

    return product_formulas_array


def _get_part_4_cuda(fields_phase,
                     gamma,
                     float_precision=np.float64
                     ):
    gamma = 2 * gamma

    number_of_qubits = fields_phase.shape[0]
    n = number_of_qubits

    threads_per_block = 256 * 4
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    cosine_fields = cuda.to_device(np.zeros(n, dtype=float_precision))

    _kernel_cosine_vector_fields[blocks_per_grid, threads_per_block](fields_phase, gamma, cosine_fields)

    cuda.synchronize()

    return cosine_fields

def _get_sine_fields_cuda(fields_phase,
                          gamma,
                          float_precision=np.float64):
    gamma = 2 * gamma
    number_of_qubits = fields_phase.shape[0]
    n = number_of_qubits


    threads_per_block = 256 * 4
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    sine_fields = cuda.to_device(np.zeros(n, dtype=float_precision))

    _kernel_sine_vector_fields[blocks_per_grid, threads_per_block](fields_phase, gamma, sine_fields)

    cuda.synchronize()

    return sine_fields









def _get_part_5_cuda(fields_phase,
                     correlations_cost,
                     sin_correlations_phase,
                     cosine_fields,
                     product_formulas_array,
                     gamma,
                     beta,
                     float_precision=np.float64,
                     return_only_sum=False
                     ):
    gamma = 2 * gamma
    number_of_qubits = fields_phase.shape[0]
    n = number_of_qubits
    if return_only_sum:
        C_ij_expected_values = cuda.to_device(np.zeros((1, 1), dtype=float_precision))
    else:
        C_ij_expected_values = cuda.to_device(np.zeros((n, n), dtype=float_precision))
    # print("HEJ",return_only_sum)

    sin_4_beta_by_2 = np.sin(4. * beta) / 2
    sin_2_beta_squared_by_2 = (np.sin(2. * beta) ** 2) / 2

    threads_per_block = (32, 32)
    blocks_per_grid = ((n + threads_per_block[0] - 1) // threads_per_block[0],
                       (n + threads_per_block[1] - 1) // threads_per_block[1])
    _kernel_calculate_C_ij[blocks_per_grid, threads_per_block](fields_phase,
                                                               correlations_cost,
                                                               sin_correlations_phase,
                                                               cosine_fields,
                                                               product_formulas_array,
                                                               gamma,
                                                               sin_4_beta_by_2,
                                                               sin_2_beta_squared_by_2,
                                                               return_only_sum,
                                                               C_ij_expected_values)
    cuda.synchronize()

    if return_only_sum:
        return C_ij_expected_values.copy_to_host()[0, 0]
    return C_ij_expected_values.copy_to_host()


def _get_BC_cuda(fields_phase,
                 correlations_cost,
                 sin_correlations_phase,
                 cosine_fields,
                 product_formulas_array,
                 gamma,
                 float_precision=np.float64
                 ):
    gamma = 2 * gamma
    number_of_qubits = fields_phase.shape[0]
    n = number_of_qubits

    BC_values_out = cuda.to_device(np.zeros(2, dtype=float_precision), copy=True)

    threads_per_block = (32, 32)
    blocks_per_grid = ((n + threads_per_block[0] - 1) // threads_per_block[0],
                       (n + threads_per_block[1] - 1) // threads_per_block[1])
    _kernel_calculate_BC[blocks_per_grid, threads_per_block](fields_phase,
                                                             correlations_cost,
                                                             sin_correlations_phase,
                                                             cosine_fields,
                                                             product_formulas_array,
                                                             gamma,
                                                             BC_values_out)
    # print('uhm',BC_values_out.copy_to_host())
    cuda.synchronize()

    return BC_values_out


def analytical_QAOA_cuda(angle_phase: float,
                         angle_mixer: float,
                         fields_phase: np.ndarray,
                         fields_cost: np.ndarray,
                         correlations_phase: np.ndarray,
                         correlations_cost: np.ndarray,
                         float_precision=np.float64
                         ):
    # TODO(FBM): please note that gamma is multiplied by 2 because it always appears as 2*gamma in the formulas
    gamma = 2 * angle_phase
    beta = angle_mixer

    t0 = time.perf_counter()
    cos_correlations_phase, sin_correlations_phase = _get_part_1_cuda(correlations_phase,
                                                                        correlations_cost,
                                                                        fields_phase,
                                                                        fields_cost,
                                                                        gamma,
                                                                        float_precision=float_precision)
    t1 = time.perf_counter()
    Ci_expected_values = _get_part_2_cuda(fields_phase,
                                          fields_cost,
                                          correlations_phase,
                                          cos_correlations_phase,
                                          gamma,
                                          beta,
                                          float_precision=float_precision)

    t2 = time.perf_counter()
    product_formulas_array = _get_part_3_cuda(correlations_cost,
                                              correlations_phase,
                                              cos_correlations_phase,
                                              gamma,
                                              float_precision=float_precision)
    t3 = time.perf_counter()
    cosine_fields = _get_part_4_cuda(fields_phase,
                                     gamma,
                                     float_precision=float_precision)
    t4 = time.perf_counter()

    C_ij_expected_values = _get_part_5_cuda(fields_phase,
                                            correlations_cost,
                                            sin_correlations_phase,
                                            cosine_fields,
                                            product_formulas_array,
                                            gamma,
                                            beta,
                                            float_precision=float_precision)
    t5 = time.perf_counter()

    # times = [t0, t1, t2, t3, t4, t5]
    # dts = [times[i+1]-times[i] for i in range(len(times)-1)]
    # names = ['part 1', 'part 2', 'part 3', 'part 4', 'part 5']
    # for i in range(len(dts)):
    #     print(names[i], dts[i])
    #
    # total_time = times[-1]-times[0]
    #
    # print("LONGEST PART:", names[np.argmax(dts)], np.max(dts), f"({np.max(dts)/total_time*100:.2f}%)")

    return Ci_expected_values, C_ij_expected_values
