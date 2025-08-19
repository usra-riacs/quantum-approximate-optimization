# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

from typing import Optional, Union, Tuple

from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.hamiltonians.generators.RandomClassicalHamiltonianGeneratorBase import RandomClassicalHamiltonianGeneratorBase
from quapopt.data_analysis.data_handling import HamiltonianClassSpecifierSK, CoefficientsDistributionSpecifier




class RandomSKHamiltonianGenerator(RandomClassicalHamiltonianGeneratorBase):
    def __init__(self,
                 coefficients_distribution_specifier:CoefficientsDistributionSpecifier=None,
                 localities:Tuple[int,...]=(2,),
                 ):

        """
        Generates a random Sherrington-Kirkpatrick Hamiltonian.
        :param coefficients_distribution_specifier:
        The coefficients distribution specifier. If None, the default distribution is used.
        :param localities:
        Possible values: (1,2), or (2,). If (1,2), the Hamiltonian will have both 1-local and 2-local terms.
        """

        assert set(localities) in [{2}, {1,2}], "Localities must be either (2,) or (1, 2)."



        hamiltonian_class_specifier = HamiltonianClassSpecifierSK(Localities=localities,
                                                                  CoefficientsDistributionSpecifier=coefficients_distribution_specifier)

        super().__init__(hamiltonian_class_specifier=hamiltonian_class_specifier,)


    def generate_instance(self,
                          number_of_qubits:int,
                          seed: Optional[int] = None,
                          read_from_drive_if_present:bool=True,
                          default_backend:Optional[str]=None) -> ClassicalHamiltonian:

        return self._generate_instance(number_of_qubits=number_of_qubits,
                                       seed=seed,
                                       read_from_drive_if_present=read_from_drive_if_present,
                                       default_backend=default_backend)





if __name__ == '__main__':
    # Example of usage
    from quapopt.data_analysis.data_handling import (COEFFICIENTS_TYPE,
                                                     COEFFICIENTS_DISTRIBUTION,
                                                     CoefficientsDistributionSpecifier,
                                                     HamiltonianClassSpecifierGeneral,
                                                     HAMILTONIAN_MODELS)



    cdp = CoefficientsDistributionSpecifier(CoefficientsType=COEFFICIENTS_TYPE.DISCRETE,
                                            CoefficientsDistributionName=COEFFICIENTS_DISTRIBUTION.Uniform,
                                            CoefficientsDistributionProperties={'low': -10, 'high': 10, 'step':1})

    RSKHG = RandomSKHamiltonianGenerator(coefficients_distribution_specifier=cdp,
                                            localities=(1,2,))


    number_of_qubits = 10**2
    random_hamiltonian = RSKHG.generate_instance(number_of_qubits=number_of_qubits,
                                                 seed=42).hamiltonian
    print('got hamiltonian')
    #print(random_hamiltonian)

    from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
    import numpy as np

    import numba as nb
    import time
    import numba.cuda as nb_cuda
    n_bitstrings = 10**4
    numpy_rng = np.random.default_rng(seed=42)
    random_bitstrings = numpy_rng.integers(low=0, high=2, size=(n_bitstrings, number_of_qubits), dtype=np.int32)
    print('got random bitstrings')
    d_energies  = nb_cuda.device_array(n_bitstrings, dtype=np.float32)
    t0 = time.perf_counter()
    energies_1 = em.calculate_energies_from_bitstrings_2_local(adjacency_matrix=random_hamiltonian,
                                                               bitstrings_array=random_bitstrings,
                                                               computation_method='numpy_einsum'
                                                               )
    print('done with numpy einsum')
    t1 = time.perf_counter()
    energies_2= em.calculate_energies_from_bitstrings_2_local(adjacency_matrix=random_hamiltonian,
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

    #print(np.allclose(energies_1, energies_2))
    print("Test CUDA",np.allclose(energies_1, energies_2))
    print("TEST CYTHON", np.allclose(energies_1, energies_3))


    print(np.allclose(energies_1, energies_4))

    print("Numpy einsum time: ", t1-t0)
    print("CUDA einsum: ", t2-t1)
    print("Cython einsum: ", t3-t2)
    print("Cython: ", t4-t3)
    #print("Cython2: ", t5-t4)
    #print("Precompute: ", t4-t3)


    #print(energies_1)
    #print(energies_3)