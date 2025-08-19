# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos

ctypedef fused floating:
    np.float32_t
    np.float64_t



@cython.boundscheck(False)
@cython.wraparound(False)
def cython_analytical_QAOA_p1(
                         floating phase_angle,
                         floating mixer_angle,
                         np.ndarray[floating, ndim=2] correlations_phase,
                         np.ndarray[floating, ndim=1] fields_phase,
                         np.ndarray[floating, ndim=2] correlations_cost,
                         np.ndarray[floating, ndim=1] fields_cost):
    cdef unsigned int number_of_qubits = fields_phase.shape[0]
    cdef floating gamma = phase_angle
    cdef floating beta = mixer_angle

    cdef floating sin_4_beta_by_2 = sin(4. * beta) / 2
    cdef floating sin_2_beta_squared_by_2 = (sin(2. * beta) ** 2) / 2

    cdef np.ndarray[floating, ndim=2] cosine_correlations = np.zeros((number_of_qubits, number_of_qubits),
                                                                    dtype=correlations_phase.dtype)
    cdef np.ndarray[floating, ndim=2] sine_correlations = np.zeros((number_of_qubits, number_of_qubits),
                                                                  dtype=correlations_phase.dtype)
    cdef np.ndarray[floating, ndim=1] cosine_fields = np.zeros(number_of_qubits,
                                                              dtype=correlations_phase.dtype)
    cdef np.ndarray[floating, ndim=1] sine_fields = np.zeros(number_of_qubits,
                                                            dtype=correlations_phase.dtype)
    cdef unsigned int i, j, k


    for i in range(number_of_qubits):
        cosine_fields[i] = cos(2. * gamma * fields_phase[i])
        sine_fields[i] = sin(2. * gamma * fields_phase[i])
        for j in range(i + 1, number_of_qubits):
            if correlations_cost[i, j] == 0 and fields_cost[i]==0 and fields_phase[i]==0:
                continue
            cos_2_gamma = cos(2. * gamma * correlations_phase[i, j])
            cosine_correlations[i, j] = cos_2_gamma
            cosine_correlations[j, i] = cos_2_gamma

            sin_2_gamma = sin(2. * gamma * correlations_phase[i, j])

            sine_correlations[i, j] = sin_2_gamma
            sine_correlations[j, i] = sine_correlations[i, j]


    cdef np.ndarray[floating, ndim=1] C_i_expected_values = np.zeros(number_of_qubits,
                                                                    dtype=correlations_phase.dtype)
    # Calculate <C_i>
    cdef floating sin_2beta = sin(2. * beta)
    cdef floating product

    for i in range(number_of_qubits):
        h_i_phase = fields_phase[i]
        h_i_cost = fields_cost[i]
        if h_i_cost == 0.0 or h_i_phase == 0.0:
            continue

        product = 1.0
        for k in range(number_of_qubits):
            if correlations_phase[i, k] == 0:
                continue
            cos_2_gamma = cosine_correlations[i, k]
            if cos_2_gamma == 0:
                product = 0
                break
            product *= cos_2_gamma
        if product!=0:
            C_i_expected_values[i] = h_i_cost * sin_2beta * sin(2. * gamma * h_i_phase) * product

    # Calculate <C_i,j>
    cdef np.ndarray[floating, ndim=2] C_ij_expected_values = np.zeros((number_of_qubits, number_of_qubits),
                                                                     dtype=correlations_phase.dtype)


    for i in range(number_of_qubits):
        h_i_phase = fields_phase[i]
        for j in range(i + 1, number_of_qubits):
            if correlations_cost[i, j] == 0:
                continue

            h_j_phase = fields_phase[j]

            J_ij_cost = correlations_cost[i, j]

            # First term
            product1, product2 = 1., 1.
            # Second term
            product3, product4 = 1., 1.
            # Third term
            product5, product6 = 1., 1.

            for k in range(number_of_qubits):
                if k == i or k == j:
                    continue

                if correlations_phase[i, k] != 0:
                    cos_ik = cosine_correlations[i, k]

                    if cos_ik == 0:
                        product1 = 0
                        product3 = 0

                    else:
                        if product1 != 0:
                            product1 *= cos_ik
                        if correlations_phase[j, k] == 0 and product3 != 0:
                            product3 *= cos_ik

                if correlations_phase[j, k] != 0:
                    cos_jk = cosine_correlations[j, k]
                    if cos_jk == 0:
                        product2 = 0
                        product4 = 0
                    else:
                        if product2 != 0:
                            product2 *= cos_jk
                        if correlations_phase[i, k] == 0 and product4 != 0:
                            product4 *= cos_jk

                if correlations_phase[i, k] != 0 and correlations_phase[j, k] != 0:
                    product5 *= cos(2. * gamma * (correlations_phase[i, k] + correlations_phase[j, k]))
                    product6 *= cos(2. * gamma * (correlations_phase[i, k] - correlations_phase[j, k]))

            # First term
            term1 = J_ij_cost * sin_4_beta_by_2 * sine_correlations[i, j]
            term1 *= cosine_fields[i] * product1 + cosine_fields[j] * product2

            # Second term
            term2 = -J_ij_cost * sin_2_beta_squared_by_2 * product3 * product4

            term3 = cos(2. * gamma * (h_i_phase + h_j_phase)) * product5 - cos(
                2. * gamma * (h_i_phase - h_j_phase)) * product6

            C_ij_expected_values[i, j] = term1 + term2 * term3


    return C_i_expected_values, C_ij_expected_values




"""

@cuda.jit
def _kernel_cosine_and_sine_correlations(correlations_phase,
                                         correlations_cost,
                                         fields_phase,
                                         fields_cost,
                                         gamma,
                                         cos_correlations_phase_out,
                                         sin_correlations_phase_out):
    n = correlations_phase.shape[0]
    i, j = cuda.grid(2)

    if not i < j < n:
        return

    #if correlations_cost[i,j] = 0, then the expression does not contribute.
    #if fields_phase[i] = 0 or fields_cost[j] = 0, then the expression does not contribute.
    if correlations_cost[i, j]!=0 or fields_phase[i]!=0 or fields_cost[j]!=0:
        #print(gamma, correlations_phase[i, j])

        #arg = 2. * gamma * correlations_phase[i, j]
        arg = gamma * correlations_phase[i, j]


        cos_2_gamma = np.cos(arg)
        sin_2_gamma = np.sin(arg)

        cos_correlations_phase_out[i, j] = cos_2_gamma
        cos_correlations_phase_out[j, i] = cos_2_gamma
        sin_correlations_phase_out[i, j] = sin_2_gamma
        sin_correlations_phase_out[j, i] = sin_2_gamma
        

def _get_part_1_cuda(correlations_phase,
                     correlations_cost,
                     fields_phase,
                     fields_cost,
                     gamma, ):
    number_of_qubits = fields_phase.shape[0]
    n = number_of_qubits

    threads_per_block = (32, 32)
    blocks_per_grid = ((n + threads_per_block[0] - 1) // threads_per_block[0],
                       (n + threads_per_block[1] - 1) // threads_per_block[1])

    cos_correlations_phase = cuda.to_device(np.zeros((n, n), dtype=np.float64))
    sin_correlations_phase = cuda.to_device(np.zeros((n, n), dtype=np.float64))

    _kernel_cosine_and_sine_correlations[blocks_per_grid, threads_per_block](correlations_phase,
                                                                             correlations_cost,
                                                                             fields_phase,
                                                                             fields_cost,
                                                                             gamma,
                                                                             cos_correlations_phase,
                                                                             sin_correlations_phase)
    return cos_correlations_phase, sin_correlations_phase



"""

#I want C version of the above CUDA code
@cython.boundscheck(False)
@cython.wraparound(False)
def _get_part_1_cython(np.ndarray[floating, ndim=2] correlations_phase,
                       floating gamma):
    cdef unsigned int number_of_qubits = correlations_phase.shape[0]
    cdef floating cos_2_gamma, sin_2_gamma
    cdef np.ndarray[floating, ndim=2] cos_correlations_phase = np.ones((number_of_qubits, number_of_qubits),
                                                                        dtype=correlations_phase.dtype)
    cdef np.ndarray[floating, ndim=2] sin_correlations_phase = np.zeros((number_of_qubits, number_of_qubits),
                                                                        dtype=correlations_phase.dtype)
    cdef unsigned int i, j

    #multiply gamma by 2; this is how it appears in all the formulas
    gamma = 2. * gamma

    for i in range(number_of_qubits):
        for j in range(i + 1, number_of_qubits):
            arg_ij = gamma * correlations_phase[i, j]

            if arg_ij == 0:
                continue

            cos_2_gamma = cos(arg_ij)
            cos_correlations_phase[i, j] = cos_2_gamma
            cos_correlations_phase[j, i] = cos_2_gamma

            sin_2_gamma = sin(arg_ij)
            sin_correlations_phase[i, j] = sin_2_gamma
            sin_correlations_phase[j, i] = sin_2_gamma

    return cos_correlations_phase, sin_correlations_phase



@cython.boundscheck(False)
@cython.wraparound(False)
def _get_part_2_cython(np.ndarray[floating, ndim=1] fields_phase,
                       np.ndarray[floating, ndim=1] fields_cost,
                       np.ndarray[floating, ndim=2] correlations_phase,
                       np.ndarray[floating, ndim=2] cos_correlations_phase,
                       floating gamma,
                       floating beta,
                       return_only_sum):
    cdef unsigned int number_of_qubits = fields_phase.shape[0]
    cdef floating sin_2_beta = sin(2. * beta)
    cdef floating product

    cdef np.ndarray[floating, ndim=1] Ci_expected_values

    if return_only_sum:
        Ci_expected_values = np.zeros(1, dtype=fields_phase.dtype)

    else:
        Ci_expected_values = np.zeros(number_of_qubits,
                                                                    dtype=fields_phase.dtype)

    cdef unsigned int i, k

    #multiply gamma by 2; this is how it appears in all the formulas
    gamma = 2. * gamma


    for i in range(number_of_qubits):
        h_i_phase = fields_phase[i]
        h_i_cost = fields_cost[i]
        if h_i_cost == 0.0 or h_i_phase == 0.0:
            continue

        product = 1.0
        for k in range(number_of_qubits):
            if correlations_phase[i, k] == 0:
                continue
            cos_2_gamma = cos_correlations_phase[i, k]
            if cos_2_gamma == 0:
                product = 0
                break
            product *= cos_2_gamma
        if product!=0:
            term_i = h_i_cost * sin_2_beta * sin(gamma * h_i_phase) * product
            if return_only_sum:
                Ci_expected_values[0] += term_i
            else:
                Ci_expected_values[i] = term_i

    if return_only_sum:
        return Ci_expected_values[0]

    return Ci_expected_values

#I want C version of the above CUDA code
@cython.boundscheck(False)
@cython.wraparound(False)
def _get_part_3_cython(np.ndarray[floating, ndim=2] correlations_cost,
                       np.ndarray[floating, ndim=2] correlations_phase,
                       np.ndarray[floating, ndim=2] cos_correlations_phase,
                       floating gamma):
    cdef unsigned int number_of_qubits = correlations_cost.shape[0]
    cdef floating product1, product2, product3, product4, product5, product6
    cdef np.ndarray[floating, ndim=3] product_formulas_array = np.zeros((number_of_qubits, number_of_qubits, 6),
                                                                        dtype=correlations_cost.dtype)
    cdef unsigned int i, j, k

    #multiply gamma by 2; this is how it appears in all the formulas
    gamma = 2. * gamma

    for i in range(number_of_qubits):
        for j in range(i + 1, number_of_qubits):
            # if correlations_cost[i, j] == 0:
            #     continue

            # First term
            product1, product2 = 1., 1.
            # Second term
            product3, product4 = 1., 1.
            # Third term
            product5, product6 = 1., 1.

            for k in range(number_of_qubits):
                if k == i or k == j:
                    continue
                if correlations_phase[i, k] != 0:
                    cos_ik = cos_correlations_phase[i, k]
                    if cos_ik == 0:
                        product1 = 0.0
                        product3 = 0.0
                    else:
                        if product1 != 0:
                            product1 *= cos_ik
                        if correlations_phase[j, k] == 0 and product3 != 0:
                            product3 *= cos_ik

                if correlations_phase[j, k] != 0:
                    cos_jk = cos_correlations_phase[j, k]
                    if cos_jk == 0:
                        product2 = 0
                        product4 = 0
                    else:
                        if product2 != 0:
                            product2 *= cos_jk
                        if correlations_phase[i, k] == 0 and product4 != 0:
                            product4 *= cos_jk

                if correlations_phase[i, k] != 0 and correlations_phase[j, k] != 0:
                    product5 *= cos(gamma * (correlations_phase[i, k] + correlations_phase[j, k]))
                    product6 *= cos(gamma * (correlations_phase[i, k] - correlations_phase[j, k]))

            product_formulas_array[i, j, 0] = product1
            product_formulas_array[i, j, 1] = product2
            product_formulas_array[i, j, 2] = product3
            product_formulas_array[i, j, 3] = product4
            product_formulas_array[i, j, 4] = product5
            product_formulas_array[i, j, 5] = product6

    return product_formulas_array

"""

@cuda.jit
def _kernel_cosine_vector_fields(fields_phase,
                                 gamma,
                                 cosine_fields_out):
    i = cuda.grid(1)
    if i >= fields_phase.shape[0]:
        return
    #cosine_fields_out[i] = np.cos(2. * gamma * fields_phase[i])
    cosine_fields_out[i] = np.cos(gamma * fields_phase[i])

def _get_part_4_cuda(fields_phase,
                     gamma):

    number_of_qubits = fields_phase.shape[0]
    n = number_of_qubits

    threads_per_block = 256*4
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    cosine_fields = cuda.to_device(np.zeros(n, dtype=np.float64))

    _kernel_cosine_vector_fields[blocks_per_grid, threads_per_block](fields_phase, gamma, cosine_fields)

    return cosine_fields


"""

#I want C version of the above CUDA code

@cython.boundscheck(False)
@cython.wraparound(False)
def _get_part_4_cython(np.ndarray[floating, ndim=1] fields_phase,
                       floating gamma):
    cdef unsigned int number_of_qubits = fields_phase.shape[0]
    cdef np.ndarray[floating, ndim=1] cosine_fields = np.zeros(number_of_qubits,
                                                              dtype=fields_phase.dtype)
    cdef unsigned int i

    #multiply gamma by 2; this is how it appears in all the formulas
    gamma = 2. * gamma

    for i in range(number_of_qubits):
        cosine_fields[i] = cos(gamma * fields_phase[i])

    return cosine_fields






"""

@cuda.jit
def _kernel_calculate_C_ij(fields_phase,
                           correlations_cost,
                           sin_correlations_phase,
                           cosine_fields,
                           product_formulas_array,
                           gamma,
                           sin_4_beta_by_2,
                           sin_2_beta_squared_by_2,
                           C_ij_expected_values_out):
    idx_1, idx_2 = cuda.grid(2)
    n = fields_phase.shape[0]

    run_check = idx_1 < idx_2 < n and correlations_cost[idx_1, idx_2]!=0
    if not run_check:
        return

    # print('hej', i, j, C_ij_expected_values_out[0, 0])

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
    term1 = J_ij_cost * sin_4_beta_by_2 * sin_correlations_phase[idx_1, idx_2]
    term1 *= cosine_fields[idx_1] * product1 + cosine_fields[idx_2] * product2
    # Second term
    term2 = -J_ij_cost * sin_2_beta_squared_by_2 * product3 * product4
    # Third term
    #term3 = np.cos(2. * gamma * (h_i_phase + h_j_phase)) * product5 - np.cos(
       # 2. * gamma * (h_i_phase - h_j_phase)) * product6
    term3 = np.cos(gamma * (h_i_phase + h_j_phase)) * product5 - np.cos(
        gamma * (h_i_phase - h_j_phase)) * product6

    term_ij = term1 + term2 * term3

    # print('cuda', idx_1, idx_2, product1, product2, product3, product4, product5, product6)
    C_ij_expected_values_out[idx_1, idx_2] = term_ij
    

def _get_part_5_cuda(fields_phase,
                                           correlations_cost,
                                           sin_correlations_phase,
                                           cosine_fields,
                                           product_formulas_array,
                                           gamma,
                                           beta
                                           ):
    number_of_qubits = fields_phase.shape[0]
    n = number_of_qubits


    C_ij_expected_values = cuda.to_device(np.zeros((n, n), dtype=np.float64))

    sin_4_beta_by_2 = np.sin(4. * beta) / 2
    sin_2_beta_squared_by_2 = (np.sin(2. * beta) ** 2) / 2

    threads_per_block = (29, 29)
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
                                                               C_ij_expected_values)

    return C_ij_expected_values.copy_to_host()


"""

#I want C version of the above CUDA code
@cython.boundscheck(False)
@cython.wraparound(False)
def _get_part_5_cython(np.ndarray[floating, ndim=1] fields_phase,
                       np.ndarray[floating, ndim=2] correlations_cost,
                       np.ndarray[floating, ndim=2] sin_correlations_phase,
                       np.ndarray[floating, ndim=1] cosine_fields,
                       np.ndarray[floating, ndim=3] product_formulas_array,
                       floating gamma,
                       floating beta,
                       return_only_sum):
    cdef unsigned int number_of_qubits = fields_phase.shape[0]
    cdef floating sin_4_beta_by_2 = sin(4. * beta) / 2
    cdef floating sin_2_beta_squared_by_2 = (sin(2. * beta) ** 2) / 2
    cdef floating term1, term2, term3
    cdef floating product1, product2, product3, product4, product5, product6

    cdef np.ndarray[floating, ndim=2] C_ij_expected_values

    if return_only_sum:
        C_ij_expected_values = np.zeros((1, 1), dtype=fields_phase.dtype)
    else:
        C_ij_expected_values = np.zeros((number_of_qubits, number_of_qubits),
                                                                     dtype=fields_phase.dtype)
    cdef unsigned int i, j

    #multiply gamma by 2; this is how it appears in all the formulas
    gamma = 2. * gamma

    for i in range(number_of_qubits):
        h_i_phase = fields_phase[i]
        for j in range(i + 1, number_of_qubits):
            if correlations_cost[i, j] == 0:
                continue

            h_j_phase = fields_phase[j]

            J_ij_cost = correlations_cost[i, j]

            product1 = product_formulas_array[i, j, 0]
            product2 = product_formulas_array[i, j, 1]
            product3 = product_formulas_array[i, j, 2]
            product4 = product_formulas_array[i, j, 3]
            product5 = product_formulas_array[i, j, 4]
            product6 = product_formulas_array[i, j, 5]

            # First term
            term1 = J_ij_cost * sin_4_beta_by_2 * sin_correlations_phase[i, j]
            term1 *= cosine_fields[i] * product1 + cosine_fields[j] * product2
            # Second term
            term2 = -J_ij_cost * sin_2_beta_squared_by_2 * product3 * product4
            # Third term
            term3 = cos(gamma * (h_i_phase + h_j_phase)) * product5 - cos(
                gamma * (h_i_phase - h_j_phase)) * product6

            term_ij = term1 + term2 * term3

            if return_only_sum:
                C_ij_expected_values[0,0] += term_ij
            else:
                C_ij_expected_values[i, j] = term_ij

    if return_only_sum:
        return C_ij_expected_values[0,0]

    return C_ij_expected_values



@cython.boundscheck(False)
@cython.wraparound(False)
def _get_A_cython(
                #gamma,
                np.ndarray[floating,ndim=1] fields_cost,
                np.ndarray[floating, ndim=1] sin_fields_phase,
                np.ndarray[floating,ndim=2] correlations_phase,
                np.ndarray[floating, ndim=2] cos_correlations_phase,
                np.ndarray[floating,ndim=3] product_formulas_array
                  ):

    cdef unsigned int number_of_qubits = fields_cost.shape[0]
    cdef unsigned int i, j

    cdef floating A_value
    cdef floating product_i
    cdef floating h_i, sin_i

    #gamma = 2. * gamma
    A_value = 0.0
    for i in range(number_of_qubits):
        h_i = fields_cost[i]
        sin_i = sin_fields_phase[i]

        if h_i == 0.0 or sin_i == 0.0:
            continue

        product_i = h_i * sin_i
        #Looking for j connected to i
        for j in range(number_of_qubits):
            if correlations_phase[i, j] !=0 and i!=j:
                #product_formulas_array[i,j,0] contains the product of cosines of interactions between i and everyone except j
                #so we are adding here the cosine of the interaction between i and j
                #TODO(FBM): in this particular part, it would probably be more efficient to calculate the product of cosines for everyone and divide by cos(cij) during computation
                if i<j:
                    product_i *= cos_correlations_phase[i,j]*product_formulas_array[i, j, 0]
                else:
                    product_i *= cos_correlations_phase[j,i]*product_formulas_array[j, i, 0]

                break

        A_value += product_i


    return A_value



@cython.boundscheck(False)
@cython.wraparound(False)
def _get_sine_fields_cython(np.ndarray[floating, ndim=1] fields_phase,
                       floating gamma):
    cdef unsigned int number_of_qubits = fields_phase.shape[0]
    cdef np.ndarray[floating, ndim=1] sin_fields = np.zeros(number_of_qubits,
                                                              dtype=fields_phase.dtype)
    cdef unsigned int i

    #multiply gamma by 2; this is how it appears in all the formulas
    gamma = 2. * gamma

    for i in range(number_of_qubits):
        sin_fields[i] = sin(gamma * fields_phase[i])

    return sin_fields






#HERE's CUDA FUNCTION THAT I WISH TO REWRITE IN C
"""

@cuda.jit
def _kernel_calculate_BC(fields_phase,
                       correlations_cost,
                       sin_correlations_phase,
                       cosine_fields,
                       product_formulas_array,
                       gamma,
                       BC_values_out):
    idx_u, idx_v = cuda.grid(2)
    n = fields_phase.shape[0]

    run_check = idx_u < idx_v < n and correlations_cost[idx_u, idx_v]!=0
    if not run_check:
        return

    # print('hej', i, j, C_ij_expected_values_out[0, 0])

    h_u_phase = fields_phase[idx_u]
    h_v_phase = fields_phase[idx_v]
    J_uv_cost = correlations_cost[idx_u, idx_v]
    sin_uv_phase = sin_correlations_phase[idx_u, idx_v]

    product1 = product_formulas_array[idx_u, idx_v][0]
    product2 = product_formulas_array[idx_u, idx_v][1]
    product3 = product_formulas_array[idx_u, idx_v][2]
    product4 = product_formulas_array[idx_u, idx_v][3]
    product5 = product_formulas_array[idx_u, idx_v][4]
    product6 = product_formulas_array[idx_u, idx_v][5]

    B_uv = J_uv_cost*sin_uv_phase*(cosine_fields[idx_u]*product1+cosine_fields[idx_v]*product2)

    term_2 = np.cos(gamma * (h_u_phase + h_v_phase))*product5
    term_3 = np.cos(gamma * (h_u_phase - h_v_phase))*product6

    C_uv = -J_uv_cost*product3*product4*(term_2-term_3)

    BC_values_out[idx_u, idx_v, 0] = B_uv
    BC_values_out[idx_u, idx_v, 1] = C_uv



"""


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_BC_cython(np.ndarray[floating, ndim=1] fields_phase,
                    np.ndarray[floating, ndim=2] correlations_cost,
                    np.ndarray[floating, ndim=2] sin_correlations_phase,
                    np.ndarray[floating, ndim=1] cosine_fields,
                    np.ndarray[floating, ndim=3] product_formulas_array,
                       floating gamma):
    cdef unsigned int number_of_qubits = correlations_cost.shape[0]
    cdef floating B_uv, C_uv
    # cdef np.ndarray[floating, ndim=3] BC_values = np.zeros(shape=(number_of_qubits, number_of_qubits, 2),
    #                                                        dtype=fields_phase.dtype)
    cdef np.ndarray[floating, ndim=1] BC_values = np.zeros(shape=2,
                                                           dtype=fields_phase.dtype)
    cdef unsigned int i, j

    #multiply gamma by 2; this is how it appears in all the formulas
    gamma = 2. * gamma

    for i in range(number_of_qubits):
        h_i_phase = fields_phase[i]
        cos_i = cosine_fields[i]
        for j in range(i + 1, number_of_qubits):
            if correlations_cost[i, j] == 0:
                continue

            cos_j = cosine_fields[j]

            h_j_phase = fields_phase[j]
            J_ij_cost = correlations_cost[i, j]

            sin_ij_phase = sin_correlations_phase[i, j]

            product1 = product_formulas_array[i, j, 0]
            product2 = product_formulas_array[i, j, 1]
            product3 = product_formulas_array[i, j, 2]
            product4 = product_formulas_array[i, j, 3]
            product5 = product_formulas_array[i, j, 4]
            product6 = product_formulas_array[i, j, 5]

            # First term
            B_uv = J_ij_cost * sin_ij_phase * (cos_i * product1 + cos_j * product2)
            # Second term
            term_2 = cos(gamma * (h_i_phase + h_j_phase)) * product5
            term_3 = cos(gamma * (h_i_phase - h_j_phase)) * product6

            C_uv = -J_ij_cost * product3 * product4 * (term_2 - term_3)

            # BC_values[i, j, 0] = B_uv
            # BC_values[i, j, 1] = C_uv
            BC_values[0] += B_uv
            BC_values[1] += C_uv



    return BC_values









@cython.boundscheck(False)
@cython.wraparound(False)
def _get_ABC_direct_cython(np.ndarray[floating, ndim=1] local_fields,
                            np.ndarray[floating, ndim=2] couplings,
                           floating gamma):

    cdef floating A, B, C
    cdef unsigned int number_of_qubits = local_fields.shape[0]

    cdef np.ndarray[floating, ndim=1] sine_fields = np.zeros(number_of_qubits, dtype=local_fields.dtype)
    cdef np.ndarray[floating, ndim=1] cosine_fields = np.zeros(number_of_qubits, dtype=local_fields.dtype)
    cdef np.ndarray[floating, ndim=2] cosine_correlations = np.zeros((number_of_qubits, number_of_qubits), dtype=local_fields.dtype)
    cdef np.ndarray[floating, ndim=2] sine_correlations = np.zeros((number_of_qubits, number_of_qubits), dtype=local_fields.dtype)

    cdef floating h_i, sin_i, J_ij, J_uv, cos_i, cos_j, sin_ij, sin_uv, cos_chi_plus, cos_chi_minus, cos_wv, cos_uw, cos_vw, cos_uv_f_plus, cos_uv_f_minus

    cdef unsigned int q_i, q_j, q_u, q_v, q_w


    gamma = 2. * gamma

    A = 0.0
    for q_i in range(number_of_qubits):
        h_i = local_fields[q_i]
        sin_i = sin(gamma*h_i)
        sine_fields[q_i] = sin_i
        cos_i = cos(gamma*h_i)
        cosine_fields[q_i] = cos_i

        prod_qi = 1.0
        for q_j in range(number_of_qubits):

            if q_i == q_j:
                continue

            J_ij = couplings[q_i, q_j]

            cos_ij = cos(gamma*J_ij)
            cosine_correlations[q_i, q_j] = cos_ij
            cosine_correlations[q_j, q_i] = cos_ij

            sin_ij = sin(gamma*J_ij)
            sine_correlations[q_i, q_j] = sin_ij
            sine_correlations[q_j, q_i] = sin_ij

            if J_ij == 0:
                continue
            if prod_qi == 0.0:
                continue
            if cos_ij == 0:
                prod_qi = 0
            else:
                prod_qi *= cos_ij

        A += h_i * sin_i * prod_qi

    B = 0.0
    C = 0.0
    for q_u in range(number_of_qubits):
        cos_u = cosine_fields[q_u]
        for q_v in range(q_u+1, number_of_qubits):
            cos_v = cosine_fields[q_v]
            J_uv = couplings[q_u, q_v]

            if J_uv == 0:
                continue

            sin_uv = sine_correlations[q_u, q_v]

            prod_e = cos_v
            prod_d = cos_u

            prod_e_F = 1.0
            prod_d_F = 1.0

            prod_f_plus = 1.0
            prod_f_minus = 1.0

            for q_w in range(number_of_qubits):
                #B part; neighbors of v that are not u
                if couplings[q_v, q_w] != 0 and q_w!=q_u and prod_e != 0:
                    cos_wv = cosine_correlations[q_v, q_w]
                    if cos_wv == 0:
                        prod_e = 0
                    else:
                        prod_e *= cos_wv
                #B part; neighbors of u that are not v
                if couplings[q_u, q_w] != 0 and q_w != q_v and prod_d != 0:
                    cos_uw = cosine_correlations[q_u, q_w]
                    if cos_uw == 0:
                        prod_d = 0
                    else:
                        prod_d *= cos_uw

                #C part; neighbors of v that are not u and are not neighbors of u:
                if couplings[q_v, q_w] != 0 and q_w != q_u and couplings[q_u, q_w] == 0 and prod_e_F != 0:
                    cos_vw = cosine_correlations[q_v, q_w]
                    if cos_vw == 0:
                        prod_e_F = 0
                    else:
                        prod_e_F *= cos_vw

                #C part; neighbors of u that are not v and are not neighbors of v:
                if couplings[q_u, q_w] != 0 and q_w != q_v and couplings[q_v, q_w] == 0 and prod_d_F != 0:
                    cos_uw = cosine_correlations[q_u, q_w]
                    if cos_uw == 0:
                        prod_d_F = 0
                    else:
                        prod_d_F *= cos_uw

                #C part 2; neighbors of both u and v, excluding uw
                if couplings[q_v, q_w]!=0 and couplings[q_w, q_u]!=0 and q_w != q_v and q_w != q_u:
                    if prod_f_plus!=0.0:
                        cos_uv_f_plus = cos(couplings[q_w, q_u]+couplings[q_w, q_v])
                        if cos_uv_f_plus == 0:
                            prod_f_plus = 0
                        else:
                            prod_f_plus *= cos_uv_f_plus
                    if prod_f_minus != 0.0:
                        cos_uv_f_minus = cos(couplings[q_w, q_u]-couplings[q_w, q_v])
                        if cos_uv_f_minus == 0:
                            prod_f_minus = 0
                        else:
                            prod_f_minus *= cos_uv_f_minus




            B += J_uv*sin_uv*(prod_e+prod_d)

            cos_chi_plus = cos(local_fields[q_u]+local_fields[q_v])
            cos_chi_minus = cos(local_fields[q_u]-local_fields[q_v])

            main_term = J_uv*prod_e_F*prod_d_F
            C += main_term*(cos_chi_plus*prod_f_plus - cos_chi_minus*prod_f_minus)


    C *= -1.0
    return A, B, C









