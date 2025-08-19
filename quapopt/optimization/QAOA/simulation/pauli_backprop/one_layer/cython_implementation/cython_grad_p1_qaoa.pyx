# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos

ctypedef fused floating:
    np.float32_t
    np.float64_t


"""
    Here we consider the ising model with fields and correlations
"""

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_analytical_single_grad_QAOA_p1(
                                     floating phase_angle,
                                     floating mixer_angle,
                                     np.ndarray[floating, ndim=1] fields,
                                     np.ndarray[floating, ndim=2] correlations):
    cdef unsigned int number_of_qubits = fields.shape[0]
    cdef floating gamma = phase_angle
    cdef floating beta = mixer_angle

    cdef floating cos_2_beta = cos(2. * beta)
    cdef floating sin_2_beta = sin(2. * beta)

    cdef np.ndarray[floating, ndim=2] cosine_correlations = np.zeros((number_of_qubits, number_of_qubits),
                                                                    dtype=correlations.dtype)
    cdef np.ndarray[floating, ndim=2] sine_correlations = np.zeros((number_of_qubits, number_of_qubits),
                                                                  dtype=correlations.dtype)
    cdef np.ndarray[floating, ndim=1] cosine_fields = np.zeros(number_of_qubits,
                                                              dtype=correlations.dtype)
    cdef np.ndarray[floating, ndim=1] sine_fields = np.zeros(number_of_qubits,
                                                            dtype=correlations.dtype)
    cdef unsigned int i, j, k


    for i in range(number_of_qubits):
        cosine_fields[i] = cos(2. * gamma * fields[i])
        sine_fields[i] = sin(2. * gamma * fields[i])
        for j in range(i + 1, number_of_qubits):
            if correlations[i, j] == 0 and fields[i]==0:
                continue
            cos_2_gamma = cos(2. * gamma * correlations[i, j])
            cosine_correlations[i, j] = cos_2_gamma
            cosine_correlations[j, i] = cos_2_gamma

            sin_2_gamma = sin(2. * gamma * correlations[i, j])

            sine_correlations[i, j] = sin_2_gamma
            sine_correlations[j, i] = sine_correlations[i, j]


    cdef np.ndarray[floating, ndim=1] grad_beta_C_i = np.zeros(number_of_qubits,
                                                               dtype=correlations.dtype)
    cdef np.ndarray[floating, ndim=1] grad_gamma_C_i = np.zeros(number_of_qubits,
                                                                dtype=correlations.dtype)
    # Calculate d<C_i>/dbeta and d<C_i>/dgamma
    cdef floating product_ik, sum_product_ikprime
    
    for i in range(number_of_qubits):
        h_i = fields[i]
        if h_i == 0.0:
            continue
        # Calculate d<C_i>/dbeta
        product_ik = 1.0
        for k in range(number_of_qubits):
            if correlations[i, k] == 0:
                continue
            if cosine_correlations[i, k] == 0:
                break
            product_ik *= cosine_correlations[i, k]
        if product_ik!= 0:
            grad_beta_C_i[i] = 2. * h_i * cos_2_beta * sine_fields[i] * product_ik
        
        # Calculate d<C_i>/dgamma
        sum_product_ikprime = 0.0
        for k in range(number_of_qubits):
            if correlations[i, k] == 0:
                continue
            if cosine_correlations[i, k] == 0:
                break
            sum_product_ikprime += -2. * correlations[i, k] * sine_correlations[i, k] *(product_ik/cosine_correlations[i, k])
        grad_gamma_C_i[i] = h_i * sin_2_beta * (2. * h_i * cosine_fields[i] * product_ik + sine_fields[i] * sum_product_ikprime)
        
    return grad_beta_C_i, grad_gamma_C_i

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_analytical_double_grad_QAOA_p1(
                                     floating phase_angle,
                                     floating mixer_angle,
                                     np.ndarray[floating, ndim=1] fields,
                                     np.ndarray[floating, ndim=2] correlations):
    cdef unsigned int number_of_qubits = fields.shape[0]
    cdef floating product1, product2, product3, product4, product5, product6
    cdef floating d_product1, d_product2, d_product3, d_product4, d_product5, d_product6
    cdef floating gamma = phase_angle
    cdef floating beta = mixer_angle

    cdef floating cos_2_beta = cos(2. * beta)
    cdef floating sin_2_beta = sin(2. * beta)

    cdef np.ndarray[floating, ndim=2] cosine_correlations = np.zeros((number_of_qubits, number_of_qubits),
                                                                    dtype=correlations.dtype)
    cdef np.ndarray[floating, ndim=2] sine_correlations = np.zeros((number_of_qubits, number_of_qubits),
                                                                  dtype=correlations.dtype)
    cdef np.ndarray[floating, ndim=1] cosine_fields = np.zeros(number_of_qubits,
                                                              dtype=correlations.dtype)
    cdef np.ndarray[floating, ndim=1] sine_fields = np.zeros(number_of_qubits,
                                                            dtype=correlations.dtype)
    cdef unsigned int i, j, k
    
    for i in range(number_of_qubits):
        cosine_fields[i] = cos(2. * gamma * fields[i])
        sine_fields[i] = sin(2. * gamma * fields[i])
        for j in range(i + 1, number_of_qubits):
            if correlations[i, j] == 0 and fields[i]==0:
                continue
            cos_2_gamma = cos(2. * gamma * correlations[i, j])
            cosine_correlations[i, j] = cos_2_gamma
            cosine_correlations[j, i] = cos_2_gamma

            sin_2_gamma = sin(2. * gamma * correlations[i, j])

            sine_correlations[i, j] = sin_2_gamma
            sine_correlations[j, i] = sine_correlations[i, j]
    
    cdef np.ndarray[floating, ndim=2] grad_beta_C_ij = np.zeros((number_of_qubits, number_of_qubits),
                                                                 dtype=correlations.dtype)

    cdef np.ndarray[floating, ndim=2] grad_gamma_C_ij = np.zeros((number_of_qubits, number_of_qubits),
                                                                  dtype=correlations.dtype)
    # Calculate d<C_i,j>/dbeta and d<C_i,j>/dgamma
    for i in range(number_of_qubits):
        h_i = fields[i]
        for j in range(i + 1, number_of_qubits):
            if correlations[i, j] == 0:
                continue

            h_j = fields[j]

            J_ij = correlations[i, j]

            # First term
            product1, product2 = 1., 1.
            
            # Second term
            product3, product4 = 1., 1. 
            
            # Third term
            product5, product6 = 1., 1.
            
            # Calculate d<C_i,j>/dbeta
            for k in range(number_of_qubits):
                if k == i or k == j:
                    continue

                if correlations[i, k] != 0:
                    cos_ik = cosine_correlations[i, k]

                    if cos_ik == 0:
                        product1 = 0
                        product3 = 0

                    else:
                        if product1 != 0:
                            product1 *= cos_ik
                        if correlations[j, k] == 0 and product3 != 0:
                            product3 *= cos_ik

                if correlations[j, k] != 0:
                    cos_jk = cosine_correlations[j, k]
                    if cos_jk == 0:
                        product2 = 0
                        product4 = 0
                    else:
                        if product2 != 0:
                            product2 *= cos_jk
                        if correlations[i, k] == 0 and product4 != 0:
                            product4 *= cos_jk

                if correlations[i, k] != 0 and correlations[j, k] != 0:
                    product5 *= cos(2. * gamma * (correlations[i, k] + correlations[j, k]))
                    product6 *= cos(2. * gamma * (correlations[i, k] - correlations[j, k]))

            # First term
            term1 = 2. * J_ij * cos(4. * beta) * sine_correlations[i, j]
            term1 *= cosine_fields[i] * product1 + cosine_fields[j] * product2

            # Second term
            term2 = -2. * J_ij * cos_2_beta * sin_2_beta * product3 * product4

            term3 = cos(2. * gamma * (h_i + h_j)) * product5 - cos(2. * gamma * (h_i - h_j)) * product6

            grad_beta_C_ij[i, j] = term1 + term2 * term3
            grad_beta_C_ij[j, i] = grad_beta_C_ij[i, j] 
        
            #------------------------------------------------------------------------------------------
            # Calculate d<C_i,j>/dgamma

            # First term derivative
            d_product1, d_product2 = 0., 0. 
            # Second term derivative
            d_product3, d_product4 = 0., 0.
            # Third term derivative
            d_product5, d_product6 = 0., 0.

            for k in range(number_of_qubits):
                if k == i or k == j:
                    continue

                if correlations[i, k] != 0:
                    cos_ik = cosine_correlations[i, k]

                    if cos_ik == 0:
                        continue
                    else:
                        d_product1 += -2. * correlations[i, k] * sine_correlations[i, k] * (product1/cos_ik)
                        if correlations[j, k] == 0:
                            d_product3 += -2. * correlations[i, k] * sine_correlations[i, k] * (product3/cos_ik)

                if correlations[j, k] != 0:
                    cos_jk = cosine_correlations[j, k]
                    if cos_jk == 0:
                        continue
                    else:
                        d_product2 += -2. * correlations[j, k] * sine_correlations[j, k] * (product2/cos_jk)
                        if correlations[i, k] == 0:
                            d_product4 += -2. * correlations[j, k] * sine_correlations[j, k] * (product4/cos_jk)

                if correlations[i, k] != 0 and correlations[j, k] != 0:
                    d_product5 += -2. * (correlations[i, k] + correlations[j, k]) * sin(2. * gamma * (correlations[i, k] + correlations[j, k])) * (product5 / cos(2. * gamma * (correlations[i, k] + correlations[j, k])))
                    d_product6 += -2. * (correlations[i, k] - correlations[j, k]) * sin(2. * gamma * (correlations[i, k] - correlations[j, k])) * (product6 / cos(2. * gamma * (correlations[i, k] - correlations[j, k])))
            # First term
            term1 = J_ij * sin(4. * beta)/2
            term1a = 2. * J_ij * cosine_correlations[i, j] * (cosine_fields[i] * product1 + cosine_fields[j] * product2)
            term1b = sine_correlations[i, j]
            term1b *= (-2. * h_i * sine_fields[i] * product1 + cosine_fields[i] * d_product1) + (-2. * h_j * sine_fields[j] * product2 + cosine_fields[j] * d_product2)
            term1 *= (term1a + term1b)

            # Second term
            term2 = (d_product3 * product4 + product3 * d_product4) * (cos(2. * gamma * (h_i + h_j)) * product5 - cos(2. * gamma * (h_i - h_j)) * product6)

            # Third term
            term3 = product3 * product4
            term3a = -2 * (h_i + h_j) * sin(2 * gamma * (h_i + h_j)) * product5 + cos(2 * gamma * (h_i + h_j)) * d_product5
            term3b = -2 * (h_i - h_j) * sin(2 * gamma * (h_i - h_j)) * product6 + cos(2 * gamma * (h_i - h_j)) * d_product6
            term3 *= (term3a + term3b)
            
            sum_term2_term3 = -J_ij * sin(2 * beta)**2/2 
            sum_term2_term3 *= (term2 + term3)
            grad_gamma_C_ij[i, j] = term1 + sum_term2_term3
            grad_gamma_C_ij[j, i] = grad_gamma_C_ij[i, j]
            
    return grad_beta_C_ij, grad_gamma_C_ij
