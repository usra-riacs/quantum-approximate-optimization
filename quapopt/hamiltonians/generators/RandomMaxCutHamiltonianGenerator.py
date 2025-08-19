# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.


from quapopt.data_analysis.data_handling import (
    CoefficientsDistributionSpecifier,
    ERDOS_RENYI_TYPES)
from quapopt.data_analysis.data_handling import HamiltonianClassSpecifierMaxCut
from quapopt.hamiltonians.generators.RandomErdosRenyiHamiltonianGenerator import \
    RandomErdosRenyiHamiltonianGenerator


class RandomMaxCutHamiltonianGenerator(RandomErdosRenyiHamiltonianGenerator):
    def __init__(self,
                 coefficients_distribution_specifier: CoefficientsDistributionSpecifier = None,
                 erdos_renyi_type: ERDOS_RENYI_TYPES = ERDOS_RENYI_TYPES.Gnp
                 ):
        """
        Initializes a random Max-Cut Hamiltonian generator.
        It's a 2-local Hamiltonian generator based on the Erdos-Renyi model.
        :param coefficients_distribution_specifier:
        If None, it defaults to constant coefficients of 1.
        :param erdos_renyi_type:
        Specifies the type of Erdos-Renyi graph to use.

        """


        hamiltonian_class_specifier = HamiltonianClassSpecifierMaxCut(ErdosRenyiType=erdos_renyi_type,
                                                                      CoefficientsDistributionSpecifier=coefficients_distribution_specifier)

        super().__init__(hamiltonian_class_specifier=hamiltonian_class_specifier, )


if __name__ == '__main__':
    # Example of usage
    from quapopt.data_analysis.data_handling import (COEFFICIENTS_TYPE,
                                                     COEFFICIENTS_DISTRIBUTION,
                                                     CoefficientsDistributionSpecifier)

    cdp = CoefficientsDistributionSpecifier(CoefficientsType=COEFFICIENTS_TYPE.DISCRETE,
                                            CoefficientsDistributionName=COEFFICIENTS_DISTRIBUTION.Uniform,
                                            CoefficientsDistributionProperties={'low': -10, 'high': 10, 'step': 1})

    RSKHG = RandomSKHamiltonianGenerator(coefficients_distribution_specifier=cdp,
                                         localities=(1, 2,))

    number_of_qubits = 10 ** 2
    random_hamiltonian = RSKHG.generate_instance(number_of_qubits=number_of_qubits,
                                                 seed=42)
    print('got hamiltonian')
    # print(random_hamiltonian)

    from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
    import numpy as np

    import time
    import numba.cuda as nb_cuda

    n_bitstrings = 10 ** 4
    numpy_rng = np.random.default_rng(seed=42)
    random_bitstrings = numpy_rng.integers(low=0, high=2, size=(n_bitstrings, number_of_qubits), dtype=np.int32)
    print('got random bitstrings')
    d_energies = nb_cuda.device_array(n_bitstrings, dtype=np.float32)
    t0 = time.perf_counter()
    energies_1 = em.calculate_energies_from_bitstrings_2_local(adjacency_matrix=random_hamiltonian,
                                                               bitstrings_array=random_bitstrings,
                                                               computation_method='numpy_einsum'
                                                               )
    print('done with numpy einsum')
    t1 = time.perf_counter()
    energies_2 = em.calculate_energies_from_bitstrings_2_local(adjacency_matrix=random_hamiltonian,
                                                               bitstrings_array=random_bitstrings,
                                                               computation_method='cuda_einsum'
                                                               )
    print('done with cuda einsum')
    t2 = time.perf_counter()

    energies_3 = em.calculate_energies_from_bitstrings_2_local(adjacency_matrix=random_hamiltonian,
                                                               bitstrings_array=random_bitstrings,
                                                               computation_method='cython_einsum'
                                                               )
    print('done with cython 2-local einsum')
    t3 = time.perf_counter()
    energies_4 = em.cython_calculate_energies_from_bitstrings(hamiltonian=random_hamiltonian,
                                                              bitstrings=random_bitstrings)
    print('done with cython')

    t4 = time.perf_counter()

    # out = nb.cuda.device_array(2**number_of_qubits, dtype=np.float32)
    # test = precompute_gpu(rank=0,
    #                       n_local_qubits=number_of_qubits,
    #                       terms=random_hamiltonian,
    #                       out=out,
    #                       first_qubit_first_bit=True)
    # out= out.copy_to_host()
    # t4 = time.perf_counter()

    # print(np.allclose(energies_1, energies_2))
    print("Test CUDA", np.allclose(energies_1, energies_2))
    print("TEST CYTHON", np.allclose(energies_1, energies_3))

    print(np.allclose(energies_1, energies_4))

    print("Numpy einsum time: ", t1 - t0)
    print("CUDA einsum: ", t2 - t1)
    print("Cython einsum: ", t3 - t2)
    print("Cython: ", t4 - t3)
    # print("Cython2: ", t5-t4)
    # print("Precompute: ", t4-t3)

    # print(energies_1)
    # print(energies_3)
