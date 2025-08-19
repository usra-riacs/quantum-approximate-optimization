# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

 
import numpy as np

def analytical_qaoa_python(
                         phase_angle:float,
                         mixer_angle:float,
                         correlations_phase:np.ndarray,
                         fields_phase:np.ndarray,
                         correlations_cost:np.ndarray,
                         fields_cost:np.ndarray):
    number_of_qubits = len(fields_phase)
    gamma = phase_angle
    beta = mixer_angle

    sin_4_beta_by_2 = np.sin(4. * beta) / 2
    sin_2_beta_squared_by_2 = (np.sin(2. * beta) ** 2) / 2

    cosine_correlations = np.zeros((number_of_qubits, number_of_qubits))
    sine_correlations = np.zeros((number_of_qubits, number_of_qubits))
    cosine_fields = np.zeros(number_of_qubits)
    sine_fields = np.zeros(number_of_qubits)

    for i in range(number_of_qubits):
        cosine_fields[i] = np.cos(2. * gamma * fields_phase[i])
        sine_fields[i] = np.sin(2. * gamma * fields_phase[i])
        for j in range(i + 1, number_of_qubits):
            if correlations_cost[i, j] == 0 and fields_cost[i]==0 and fields_phase[i]==0:
                continue
            cos_2_gamma = np.cos(2. * gamma * correlations_phase[i, j])
            cosine_correlations[i, j] = cos_2_gamma
            cosine_correlations[j, i] = cos_2_gamma

            sin_2_gamma = np.sin(2. * gamma * correlations_phase[i, j])

            sine_correlations[i, j] = sin_2_gamma
            sine_correlations[j, i] = sine_correlations[i, j]


    C_i_expected_values = np.zeros(number_of_qubits)
    # Calculate <C_i>
    sin_2beta = np.sin(2. * beta)
    for i in range(number_of_qubits):
        h_i_phase = fields_phase[i]
        h_i_cost = fields_cost[i]
        if h_i_cost == 0.0 or h_i_phase == 0.0:
            continue
        product = 1
        for k in range(number_of_qubits):
            if correlations_phase[i, k] == 0:
                continue
            #cos_2_gamma = np.cos(2 * gamma * correlations_phase[i, k])
            cos_2_gamma = cosine_correlations[i, k]
            if cos_2_gamma == 0:
                product = 0
                break
            product *= cos_2_gamma
        if product!=0:
            C_i_expected_values[i] = h_i_cost * sin_2beta * np.sin(2. * gamma * h_i_phase) * product

    C_ij_expected_values = np.zeros((number_of_qubits, number_of_qubits))
    # Calculate <C_i,j>
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
                    product5 *= np.cos(2. * gamma * (correlations_phase[i, k] + correlations_phase[j, k]))
                    product6 *= np.cos(2. * gamma * (correlations_phase[i, k] - correlations_phase[j, k]))

            # First term
            term1 = J_ij_cost * sin_4_beta_by_2 * sine_correlations[i, j]
            term1 *= cosine_fields[i] * product1 + cosine_fields[j] * product2

            # Second term
            term2 = -J_ij_cost * sin_2_beta_squared_by_2 * product3 * product4

            term3 = np.cos(2. * gamma * (h_i_phase + h_j_phase)) * product5 - np.cos(
                2. * gamma * (h_i_phase - h_j_phase)) * product6


            C_ij_expected_values[i, j] = term1 + term2 * term3

    return C_i_expected_values, C_ij_expected_values

def _get_ABC_direct_python(local_fields:np.ndarray,
                           couplings:np.ndarray,
                           gamma:float):
    from numpy import sin, cos

    number_of_qubits = local_fields.shape[0]

    sine_fields = np.zeros(number_of_qubits, dtype=local_fields.dtype)
    cosine_fields = np.ones(number_of_qubits, dtype=local_fields.dtype)
    cosine_correlations = np.ones((number_of_qubits, number_of_qubits), dtype=local_fields.dtype)
    sine_correlations = np.zeros((number_of_qubits, number_of_qubits), dtype=local_fields.dtype)

    gamma = 2. * gamma

    A = 0.0
    for q_i in range(number_of_qubits):
        h_i = local_fields[q_i]
        sin_i = sin(gamma*h_i)
        cos_i = cos(gamma*h_i)

        cosine_fields[q_i] = cos_i
        sine_fields[q_i] = sin_i

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

            if J_ij == 0 or prod_qi == 0.0:
                continue

            if cos_ij == 0:
                prod_qi = 0
            else:
                prod_qi *= cos_ij
        #print('yay', prod_qi*sin_i, h_i)
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
                        cos_uv_f_plus = cos(gamma*(couplings[q_w, q_u]+couplings[q_w, q_v]))
                        if cos_uv_f_plus == 0:
                            prod_f_plus = 0
                        else:
                            prod_f_plus *= cos_uv_f_plus
                    if prod_f_minus != 0.0:
                        cos_uv_f_minus = cos(gamma*(couplings[q_w, q_u]-couplings[q_w, q_v]))
                        if cos_uv_f_minus == 0:
                            prod_f_minus = 0
                        else:
                            prod_f_minus *= cos_uv_f_minus

            B += J_uv*sin_uv*(prod_e+prod_d)

            cos_chi_plus = cos(gamma*(local_fields[q_u]+local_fields[q_v]))
            cos_chi_minus = cos(gamma*(local_fields[q_u]-local_fields[q_v]))

            main_term = J_uv*prod_e_F*prod_d_F
            #print('hejka', prod_e_F, prod_d_F, prod_f_plus, prod_f_minus)

            C += main_term*(cos_chi_plus*prod_f_plus - cos_chi_minus*prod_f_minus)


    C *= -1.0

    # B*=1/2
    # C*=1/2


    return A, B, C






def _get_A_python(
                #gamma,
                fields_cost:np.ndarray,
                sin_fields_phase:np.ndarray,
                correlations_phase:np.ndarray,
                cos_correlations_phase:np.ndarray,
               product_formulas_array:np.ndarray
                  ):

    number_of_qubits = fields_cost.shape[0]



    #gamma = 2. * gamma
    A_value = 0.0
    for i in range(number_of_qubits):
        h_i = fields_cost[i]
        sin_i = sin_fields_phase[i]

        if h_i == 0.0 or sin_i == 0.0:
            continue

        product_i = 1.0
        #Looking for j connected to i
        #print(i,product_i)
        for j in range(number_of_qubits):
            # if correlations_phase[i, j] !=0 and i!=j:
            #     #product_formulas_array[i,j,0] contains the product of cosines of interactions between i and everyone except j
            #     #so we are adding here the cosine of the interaction between i and j
            #     #TODO(FBM): in this particular part, it would probably be more efficient to calculate the product of cosines for everyone and divide by cos(cij) during computation
            #     if i<j:
            #         product_i *= cos_correlations_phase[i,j]*product_formulas_array[i, j, 0]
            #     else:
            #         product_i *= cos_correlations_phase[j,i]*product_formulas_array[j, i, 0]
            #     break
            if i!=j:
                product_i *= cos_correlations_phase[i,j]
                jjj = j
                #print(j,product_i, product_formulas_array[i, j, 0])


        A_value += h_i * sin_i*product_i


    return A_value






