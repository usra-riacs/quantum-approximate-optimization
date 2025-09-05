import time
from typing import List, Union, Tuple

import numpy as np
from tqdm.notebook import tqdm

from quapopt import ancillary_functions as anf

from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.circuits.noise.simulation import MeasurementNoiseType
from quapopt.circuits.noise.simulation.classical_noise import add_1q_tensor_product_noise_to_samples
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian

from quapopt.hamiltonians.representation.ClassicalHamiltonianBase import get_fields_and_couplings_from_hamiltonian_list
#Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
    import numpy as cp




class ClassicalMeasurementNoiseSampler:
    def __init__(self,
                 noise_type: MeasurementNoiseType,
                 noise_description: dict,
                 rng=None,
                 backend='cupy'):
        self.noise_type = noise_type
        self.noise_description = noise_description
        self._dummy_noise = False

        if rng is None:
            if backend == 'cupy':
                rng = cp.random.default_rng()
            elif backend == 'numpy':
                _rng = np.random.default_rng()
            else:
                raise ValueError('Backend not recognized')
        elif isinstance(rng, int):
            if backend == 'cupy':
                rng = cp.random.default_rng(seed=rng)
            elif backend == 'numpy':
                rng = np.random.default_rng(seed=rng)
            else:
                raise ValueError('Backend not recognized')

        self._rng = rng
        self._backend = backend

        if self.noise_type in [MeasurementNoiseType.TP_1q_identical,
                               MeasurementNoiseType.TP_1q_general]:

            if 'p_01' not in noise_description:
                raise ValueError('p_01 must be specified for TensorProduct1QIdentical noise')
            if 'p_10' not in noise_description:
                raise ValueError('p_10 must be specified for TensorProduct1QIdentical noise')

            p01_list_or_float = noise_description['p_01']
            p10_list_or_float = noise_description['p_10']

            if p01_list_or_float is not None:
                if self.noise_type == MeasurementNoiseType.TP_1q_identical:
                    assert isinstance(p01_list_or_float,float),"p_01 must be a float for TensorProduct1QIdentical noise"
                elif self.noise_type == MeasurementNoiseType.TP_1q_general:
                    assert not isinstance(p01_list_or_float,float),"p_01 must be a list or array for TensorProduct1QGeneral noise"
                if p01_list_or_float == 0.0:
                    p01_list_or_float = None

            if p10_list_or_float is not None:
                if self.noise_type == MeasurementNoiseType.TP_1q_identical:
                    assert isinstance(p10_list_or_float,float),"p_10 must be a float for TensorProduct1QIdentical noise"
                elif self.noise_type == MeasurementNoiseType.TP_1q_general:
                    assert not isinstance(p10_list_or_float,float),"p_10 must be a list or array for TensorProduct1QGeneral noise"
                if p10_list_or_float == 0.0:
                    p10_list_or_float = None

            if p01_list_or_float is None and p10_list_or_float is None:
                self._dummy_noise = True

            def _sampling_function(ideal_samples: Union[np.ndarray, cp.ndarray],
                                   rng=self._rng) -> np.ndarray:

                return add_1q_tensor_product_noise_to_samples(ideal_samples_array=ideal_samples,
                                                                         p_01_errors=p01_list_or_float,
                                                                         p_10_errors=p10_list_or_float,
                                                                         rng=rng)

            self._noise_matrices_1q = self._get_1q_noise_matrices(
                                            p_01_errors=p01_list_or_float,
                                            p_10_errors=p10_list_or_float
                                        )

            noise_description = {'p_01': p01_list_or_float,
                                 'p_10': p10_list_or_float}

            self.noise_description = noise_description


        elif self.noise_type == MeasurementNoiseType.TP_general:
            raise NotImplementedError('TensorProductGeneral noise not yet implemented')
        else:
            raise ValueError('Noise type not recognized')

        self._sampling_function = _sampling_function
        self.noise_description = noise_description

        self._noisy_hamiltonian_representations = {'cost': None, 'phase': None}
        # self._noisy_hamiltonian_arrays = {'cost': None, 'phase': None}
        self._noisy_hamiltonian_fields = {'cost': None, 'phase': None}
        self._noisy_hamiltonian_couplings = {'cost': None, 'phase': None}

        self._noisy_hamiltonian_spectra = {'cost': None, 'phase': None}

    @property
    def noisy_hamiltonian_representations(self):
        return self._noisy_hamiltonian_representations

    @property
    def noisy_hamiltonian_spectra(self):
        return self._noisy_hamiltonian_spectra

    @property
    def dummy_noise(self):
        return self._dummy_noise

    @property
    def noisy_hamiltonian_fields(self):
        return self._noisy_hamiltonian_fields
    @property
    def noisy_hamiltonian_couplings(self):
        return self._noisy_hamiltonian_couplings

    def add_noise_to_samples(self,
                             ideal_samples: Union[cp.ndarray, np.ndarray],
                             rng=None):
        if rng is None:
            rng = self._rng
        if self._dummy_noise:
            return ideal_samples

        return self._sampling_function(ideal_samples=ideal_samples.copy(),
                                       rng=rng)

    def _get_1q_noise_matrices(self,
                             p_01_errors:Union[float,List[float]],
                             p_10_errors:Union[float,List[float]]):

        if self._backend == 'cupy':
            bck = cp
        elif self._backend == 'numpy':
            bck = np

        if p_01_errors is None:
            p_01_errors = 0.0
        if p_10_errors is None:
            p_10_errors = 0.0
        if isinstance(p_01_errors, list):
            p_01_errors = [0.0 if x is None else x for x in p_01_errors]
        if isinstance(p_10_errors, list):
            p_10_errors = [0.0 if x is None else x for x in p_10_errors]

        if self.noise_type == MeasurementNoiseType.TP_1q_identical:

            assert isinstance(p_10_errors,float) and isinstance(p_01_errors,float),\
                'p_01 and p_10 must be floats for TensorProduct1QIdentical noise'

            lam_1q = bck.array([[1 - p_10_errors, p_01_errors],
                               [p_10_errors, 1 - p_01_errors]])
            return lam_1q
        elif self.noise_type == MeasurementNoiseType.TP_1q_general:
            if len(p_01_errors) != len(p_10_errors):
                raise ValueError('p01_list and p10_list must be of the same length')
            return [bck.array([[1 - p_10, p_01],
                               [p_10, 1 - p_01]]) for p_01, p_10 in zip(p_01_errors, p_10_errors)]


    def get_full_noise_matrix_representation(self,
                                             number_of_qubits,
                                             transpose=False):

        if self.noise_type == MeasurementNoiseType.TP_1q_identical:
            lam_1q = self._noise_matrices_1q
            if transpose:
                lam_1q = lam_1q.T

            full_matrix = 1.0
            for _ in tqdm(list(range((number_of_qubits))), disable=number_of_qubits < 10):
                full_matrix = np.kron(full_matrix, lam_1q)
            return full_matrix

        elif self.noise_type == MeasurementNoiseType.TP_1q_general:
            lam_1q_list = self._noise_matrices_1q

            full_matrix = 1.0
            for i in tqdm(list(range(number_of_qubits)), disable=number_of_qubits < 10):
                lam_1q = lam_1q_list[i]
                if transpose:
                    lam_1q = lam_1q.T
                full_matrix = np.kron(full_matrix, lam_1q)

        elif self.noise_type == MeasurementNoiseType.TP_general:
            noise_description = self.noise_description
            full_matrix = 1.0
            for i in range(number_of_qubits):
                full_matrix = np.kron(full_matrix, noise_description[i])
            return full_matrix

    def get_noise_matrix_element(self,
                                 output_state,
                                 input_state,
                                 ):

        if self.noise_type == MeasurementNoiseType.TP_1q_identical:
            lam_1q = self._noise_matrices_1q
            # input state labels the row,
            # output state labels the column
            p_xy = 1.0
            for qi_input, qi_output in zip(input_state, output_state):
                p_xy = p_xy * lam_1q[qi_input, qi_output]
            return p_xy
        elif self.noise_type == MeasurementNoiseType.TP_1q_general:
            if len(input_state) != len(output_state):
                raise ValueError('Input and output states must be of the same length')

            p_xy = 1.0
            for qi_index, qi_input, qi_output in enumerate(zip(input_state, output_state)):
                p_xy = p_xy * self._noise_matrices_1q[qi_input][qi_output]
            return p_xy

        else:
            raise NotImplementedError('Only implemented for TensorProduct1QIdentical noise')

    def get_probability_of_particular_output_state(self,
                                                   output_state,
                                                   input_probability_distribution,
                                                   number_of_qubits=None,
                                                   classical_register=None
                                                   ):
        # TODO FBM: implement GPU implementation of handling manipulations of the noise matrix

        if self.noise_type == MeasurementNoiseType.TP_1q_identical:
            assert number_of_qubits is not None, 'number_of_qubits must be specified for TensorProduct1QIdentical noise'

            lam_1q = self._noise_matrices_1q
            p_01 = self.noise_description['p_01']
            p_10 = self.noise_description['p_10']
            if p_10 is None:
                p_10 = 0

            # we need to calculate the probability of the output state
            # this is result of summing over all possible input states to which the noise was added
            if classical_register is None:
                classical_register = em.cuda_generate_classical_register(number_of_qubits)

            if p_10 == 0:
                # TODO(FBM) extend this to the case where p_10 is not 0
                # situations = {(0, 0): 1,
                #               (0, 1): p_01,
                #               (1, 0): 0,
                #               (1, 1): 1-p_01
                #               }
                # situations = {0:1,
                #               3:p_01,
                #               2:0,
                #               1:1-p_01}

                all_possible_relabel_probs = {}
                for i in range(number_of_qubits + 1):
                    for j in range(number_of_qubits + 1):
                        # X - output state
                        # Y - input state
                        # i counts the number of situations, where Y_i = 1 and X_i = 0, so error; situation no 2
                        # j counts the number of situations, where Y_i = 1 and X_i = 1, so no error; situation no 3
                        all_possible_relabel_probs[(i, j)] = p_01 ** i * (1 - p_01) ** j

                # trick to get the situations counter
                # mapping is following:
                # 0->0 is 0 (no error, 100% certainty)
                # 1->1 is 3 (no error)
                # 1->0 is 2 (error)
                # 0->1 is 1 (impossible)
                y_shift_x = classical_register << output_state
                x_or_y = output_state | classical_register
                indicator = y_shift_x + x_or_y

                # Now we want to count 0s, 1s, 2s, and 3s for each bitstring in register_generator
                # I will use this to calculate the probability of the output state
                # X - output state
                # Y - input state
                # If Y_i = 0 and X_i = 1, then indicator = 1, and this is impossible so skipping (there is no 0->1 transition)
                ones_all = np.sum(indicator == 1, axis=1)
                # check which bitstrings matter (possible transitions are 0->0, 1->0, 1->1)
                nonzero_indices = np.where(ones_all == 0)[0]
                # cut the indicator matrix
                indicator_cut = indicator[nonzero_indices]
                # If Y_i=1 and X_i=0, then indicator = 2, and this is error
                twos_all_cut = np.sum(indicator_cut == 2, axis=1)
                # If Y_i=1 and X_i=1, then indicator = 3, and this is no error
                threes_all_cut = np.sum(indicator_cut == 3, axis=1)
                # the situation 1 does not affect the probability of the output state
                p_output = 0
                for ind_cut, ind_global in enumerate(nonzero_indices):
                    pi_ideal = input_probability_distribution[ind_global]
                    twos_i = twos_all_cut[ind_cut]
                    threes_i = threes_all_cut[ind_cut]
                    pi_relabel = all_possible_relabel_probs[(twos_i, threes_i)]
                    p_output += pi_ideal * pi_relabel

                return p_output
            else:
                raise NotImplementedError('Only implemented for p_10 = 0')

    def transform_hamiltonian_to_noisy_version(self,
                                               hamiltonian_list: List[Tuple[Union[float,int], Tuple[int, ...]]]) -> Tuple[
        List[Tuple[Union[float,int], Tuple[int, ...]]],
        Union[float, int]]:

        if isinstance(hamiltonian_list, ClassicalHamiltonian):
            hamiltonian_list = hamiltonian_list.hamiltonian_list_representation

        if self._dummy_noise:
            return hamiltonian_list, 0.0

        number_of_qubits = int(max([max(x[1]) for x in hamiltonian_list]) + 1)
        correlations_matrix = np.zeros((number_of_qubits, number_of_qubits),
                                       dtype=float)
        diag_terms = np.zeros(number_of_qubits, dtype=float)
        for coeff, tup in hamiltonian_list:
            if len(tup) == 2:
                correlations_matrix[tup[0], tup[1]] = coeff
                correlations_matrix[tup[1], tup[0]] = coeff
            elif len(tup) == 1:
                diag_terms[tup[0]] = coeff
            else:
                raise ValueError('Only implemented for 1 and 2 qubit terms')

        if self.noise_type == MeasurementNoiseType.TP_1q_identical:

            if self.noise_description['p_10'] != 0 and self.noise_description['p_10'] is not None:
                raise NotImplementedError('Only implemented for p_10 = 0')

            p_01 = self.noise_description['p_01']
            p_11 = 1 - p_01
            p_01_p_11 = p_01 * p_11

            #\sum_{j} M_j = \iden

            #p(j) = #tr((sum_{i} K_i \rho K_i^{\dag}) M_j)
            #\sum_{j}p(j) = tr((sum_{i} K_i \rho K_i^{\dag}) \sum_{j}M_{j}) =
            #tr((sum_{i} K_i \rho K_i^{\dag}) \iden ) ->
            #tr(\rho \sum_{i}K_i^{\dag}) \iden K_i ) = #tr(\rho \sum_{i} K_{i}^{\dag}K_{i}) = 1.0
            #hence \sum_{i}K_i^{\dag}K_i = \iden

            #SOME CHANNELS also have
            #\sum_{i}K_iK_i^{\dag} = \iden --> those channels are called "UNITAL"

            # noise transforms operators in the following way:
            # I -> I (note that this is a map dual to amplitude damping channel, so it is indeed unital)
            # J_{i} Z_i -> J_{i)* (p_01*\iden + p_11*Z_i)
            #         const.bias   unchanged
            # J_{ij} Z_iZ_j -> J_{ij}(p_01**2*\iden + p_11**2*Z_iZ_j + p_01*p_11*(Z_i + Z_j))
            #           const. bias       unchanged       qubit-dependent bias

            # So you're adding to your Hamiltonian:
            #a) Identity -- this is just constant bias, doesn't matter that much
            #b) \sum_{i} h_{i} Z_i
            #h_i = f(p_01, p_11) * real_correlations

            # constant bias (identity term):
            identity_factor = p_01 * np.sum(diag_terms)
            identity_factor += (p_01 ** 2) * np.sum(np.sum(correlations_matrix / 2, axis=0))

            # contribution from 1 qubit terms
            single_qubit_terms_noisy = diag_terms * p_11

            for qi in range(number_of_qubits):
                # contribution from 2 qubit terms
                # additional contribution from 2-qubit terms
                coeffs_2q_noisy = np.sum(correlations_matrix[qi, :]) * p_01_p_11
                single_qubit_terms_noisy[qi] += coeffs_2q_noisy

            correlations_matrix_noisy = correlations_matrix * (p_11 ** 2)

            hamiltonian_noisy = [(ci, (qi,)) for qi, ci in enumerate(single_qubit_terms_noisy) if ci != 0.0]
            for coeff, tup in hamiltonian_list:
                if len(tup) == 2:
                    qi, qj = tup
                    coeff_qi_qj_noisy = correlations_matrix_noisy[qi, qj]
                    hamiltonian_noisy.append((coeff_qi_qj_noisy, (qi, qj)))

            return hamiltonian_noisy, identity_factor


        else:
            raise NotImplementedError('Only implemented for TensorProduct1QIdentical noise')

    def add_noisy_hamiltonian_representations(self,
                                              hamiltonian_representations_dict: dict,
                                              hamiltonian_identifier='cost'):
        from quapopt import AVAILABLE_SIMULATORS



        if self.noisy_hamiltonian_representations[hamiltonian_identifier] is None:
            self._noisy_hamiltonian_representations[hamiltonian_identifier] = {}
        if self._noisy_hamiltonian_fields[hamiltonian_identifier] is None:
            self._noisy_hamiltonian_fields[hamiltonian_identifier] = {key:{} for key in hamiltonian_representations_dict}
        if self._noisy_hamiltonian_couplings[hamiltonian_identifier] is None:
            self._noisy_hamiltonian_couplings[hamiltonian_identifier] = {key:{} for key in hamiltonian_representations_dict}

        for key, hamiltonian_representation in hamiltonian_representations_dict.items():
            hamiltonian_noisy, noisy_constant = self.transform_hamiltonian_to_noisy_version(hamiltonian_representation)


            fields_cost, correlations_cost = get_fields_and_couplings_from_hamiltonian_list(hamiltonian=hamiltonian_noisy,
                                                                                            precision=np.float64,
                                                                                            )



            if 'cuda' in AVAILABLE_SIMULATORS:
                import numba.cuda as cuda
                fields_cost_cuda = cuda.to_device(fields_cost)
                correlations_cost_cuda = cuda.to_device(correlations_cost)
            else:
                fields_cost_cuda = None
                correlations_cost_cuda = None


            #print(hamiltonian_representations_dict.keys())
            #raise KeyboardInterrupt
            self._noisy_hamiltonian_representations[hamiltonian_identifier][key] = (hamiltonian_noisy, noisy_constant)
            self._noisy_hamiltonian_fields[hamiltonian_identifier][key] = {'numpy': fields_cost,
                                                                           'cuda': fields_cost_cuda}
            self._noisy_hamiltonian_couplings[hamiltonian_identifier][key] = {'numpy': correlations_cost,
                                                                                'cuda': correlations_cost_cuda}


    def solve_noisy_hamiltonians(self,
                                 number_of_qubits,
                                 hamiltonian_identifier='cost'):
        if self._noisy_hamiltonian_representations is None:
            raise ValueError('No noisy hamiltonians added')

        from quapopt.additional_packages.qokit import fur as qk_fur
        simulator = qk_fur.choose_simulator(name='gpu')

        if self._noisy_hamiltonian_spectra[hamiltonian_identifier] is None:
            self._noisy_hamiltonian_spectra[hamiltonian_identifier] = {}

        for key, (hamiltonian_representation, noisy_constant) in self._noisy_hamiltonian_representations[
            hamiltonian_identifier].items():
            simulator_key = simulator(number_of_qubits,
                                      terms=hamiltonian_representation)
            self._noisy_hamiltonian_spectra[hamiltonian_identifier][key] = simulator_key.get_cost_diagonal()


if __name__ == '__main__':


    seed = 0

    size = (10000, 1000)
    #Lazy monkey-patching of cupy

    numpy_rng_1 = np.random.default_rng(seed=seed)
    numpy_rng_2 = np.random.default_rng(seed=seed)
    cupy_rng_1 = cp.random.default_rng(seed=seed)
    cupy_rng_2 = cp.random.default_rng(seed=seed)

    t0 = time.perf_counter()
    bitstrings_1 = numpy_rng_1.integers(0, 2, size=size)
    t1 = time.perf_counter()
    bitstrings_2 = numpy_rng_2.binomial(n=1, p=0.5, size=size)
    t2 = time.perf_counter()
    bitstrings_1_cupy = cupy_rng_1.integers(0, 2, size=size)
    t3 = time.perf_counter()
    bitstrings_2_cupy = cupy_rng_2.binomial(n=1, p=0.5, size=size)
    t4 = time.perf_counter()



    print('numpy_rng.integers', t1 - t0)
    print('numpy_rng.binomial', t2 - t1)
    print('0s:',np.sum(bitstrings_1 == 0), np.sum(bitstrings_2 == 0))
    print('cupy_rng.integers', t3 - t2)
    print('cupy_rng.binomial', t4 - t3)
    print('0s:',cp.sum(bitstrings_1_cupy == 0), cp.sum(bitstrings_2_cupy == 1))


    # print(bitstrings_2)
    # print(bitstrings_2_cupy)

    print(cp.argmin(bitstrings_1_cupy))


    bitstrings_biased_1 = 1-numpy_rng_1.binomial(n=1,p=0.1, size=size)
    bitstrings_biased_2 = 1-cupy_rng_1.binomial(n=1,p=0.1, size=size)

    print('0s:',np.sum(bitstrings_biased_1 == 0), cp.sum(bitstrings_biased_2 == 0))









    raise KeyboardInterrupt




    import numpy as np
    from numba import cuda
    import math

    import numpy as np
    from numba import cuda
    import math

    x = np.array([0, 1, 1, 0])
    y = np.array([
        [1, 1, 1, 1],
        [0, 1, 0, 1],
    ]
    )
    # print(x^y)
    # print(x<<y)
    # print(x | y)
    print('input:\n', y)
    print('output:', x)

    y_shift_x = y << x
    x_or_y = x | y
    print('indicator\n', y_shift_x + x_or_y)
    # raise KeyboardInterrupt

    p_01 = 0.5
    CMNS = ClassicalMeasurementNoiseSampler(noise_type=MeasurementNoiseType.TP_1q_identical,
                                            noise_description={'p_01': p_01,
                                                               'p_10': None})

    noq_test = 3

    prob_distro_zero = np.zeros(2 ** noq_test)
    prob_distro_zero[0] = 1
    prob_distro_one = np.zeros(2 ** noq_test)
    prob_distro_one[-1] = 1
    prob_distro_plus = np.full(2 ** noq_test, 1 / (2 ** noq_test))

    print(prob_distro_one)
    p_noisy_one = CMNS.get_probability_of_particular_output_state(output_state=tuple([0] * noq_test),
                                                                  input_probability_distribution=prob_distro_one,
                                                                  number_of_qubits=noq_test)

    p_noisy_zero = CMNS.get_probability_of_particular_output_state(output_state=tuple([0] * noq_test),
                                                                   input_probability_distribution=prob_distro_zero,
                                                                   number_of_qubits=noq_test)

    p_noisy_plus = CMNS.get_probability_of_particular_output_state(output_state=tuple([0] * noq_test),
                                                                   input_probability_distribution=prob_distro_plus,
                                                                   number_of_qubits=noq_test)
    print("p(00..0|0...0)", p_noisy_zero)
    print("p(00..0|11...1)", p_noisy_one)
    print("p(00..0|iden/d)", p_noisy_plus)
