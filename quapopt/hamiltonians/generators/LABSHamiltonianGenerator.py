# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
from quapopt.hamiltonians.generators.RandomClassicalHamiltonianGeneratorBase import RandomClassicalHamiltonianGeneratorBase
from quapopt.data_analysis.data_handling import (CoefficientsDistributionSpecifier,
                                                 HamiltonianClassSpecifierLABS)
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian

import numpy as np
from typing import Optional

_NORMALIZATION_LABS = 2


# approximate optimal merit factor and energy for small Ns
# from Table 1 of https://arxiv.org/abs/1512.02475
_KNOWN_OPTIMAL_MF_LABS = {
    3: 4.500,
    4: 4.000,
    5: 6.250,
    6: 2.571,
    7: 8.167,
    8: 4.000,
    9: 3.375,
    10: 3.846,
    11: 12.100,
    12: 7.200,
    13: 14.083,
    14: 5.158,
    15: 7.500,
    16: 5.333,
    17: 4.516,
    18: 6.480,
    19: 6.224,
    20: 7.692,
    21: 8.481,
    22: 6.205,
    23: 5.628,
    24: 8.000,
    25: 8.681,
    26: 7.511,
    27: 9.851,
    28: 7.840,
    29: 6.782,
    30: 7.627,
    31: 7.172,
    32: 8.000,
    33: 8.508,
    34: 8.892,
    35: 8.390,
}

_KNOWN_OPTIMAL_ENERGIES_LABS = {
    3: 1,
    4: 2,
    5: 2,
    6: 7,
    7: 3,
    8: 8,
    9: 12,
    10: 13,
    11: 5,
    12: 10,
    13: 6,
    14: 19,
    15: 15,
    16: 24,
    17: 32,
    18: 25,
    19: 29,
    20: 26,
    21: 26,
    22: 39,
    23: 47,
    24: 36,
    25: 36,
    26: 45,
    27: 37,
    28: 50,
    29: 62,
    30: 59,
    31: 67,
    32: 64,
    33: 64,
    34: 65,
    35: 73,
}







class LABSHamiltonianGenerator(RandomClassicalHamiltonianGeneratorBase):
    def __init__(self):
        """
        Generates Low Autocorrelation Binary Sequences Hamiltonian
        """

        hamiltonian_class_specifier = HamiltonianClassSpecifierLABS()
        super().__init__(hamiltonian_class_specifier=hamiltonian_class_specifier)






    def generate_instance(self,
                          number_of_qubits:int,
                          read_from_drive_if_present=True,
                          default_backend:Optional[str]=None,
                          print_progress_bar:bool=False,
                          seed: Optional[int] = 0,
                          ) -> ClassicalHamiltonian:
        """

        :param number_of_qubits:
        :param read_from_drive_if_present:
        :param default_backend:
        :param print_progress_bar: unused; the build takes well under a second
        :param seed:
        THIS IS IGNORED -- LABS is deterministic Hamiltonian. Argument for compatiblity of all other generators.
        :return:
        """
        assert seed == 0, "LABS is deterministic, so seed is ignored, use seed=0 please."

        hamiltonian_class_specifier = self._hamiltonian_class_specifier
        hamiltonian_instance_specifier = hamiltonian_class_specifier.instance_specifier_constructor(NumberOfQubits=number_of_qubits,
                                                                                                    HamiltonianInstanceIndex=0)


        if read_from_drive_if_present:
            hamiltonian = self._read_from_drive(hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                                hamiltonian_class_specifier=hamiltonian_class_specifier)
            if hamiltonian is not None:
                return hamiltonian

        hamiltonian_list_representation = self.get_LABS_terms(number_of_qubits=number_of_qubits)
        known_energies_dict = None
        if number_of_qubits in _KNOWN_OPTIMAL_ENERGIES_LABS:
            energy = self.get_known_optimal_energy(number_of_qubits=number_of_qubits)
            known_energies_dict = {'lowest_energy': energy}





        return ClassicalHamiltonian(hamiltonian_list_representation=hamiltonian_list_representation,
                                    number_of_qubits=number_of_qubits,
                                    default_backend=default_backend,
                                    hamiltonian_class_specifier=hamiltonian_class_specifier,
                                    hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                    known_energies_dict=known_energies_dict,
                                    class_specific_data={'offset': self.get_LABS_offset(number_of_qubits=number_of_qubits),
                                                         'normalization':_NORMALIZATION_LABS})

    @staticmethod
    def get_LABS_offset(number_of_qubits:int):
        return np.sum([number_of_qubits-k for k in range(1,number_of_qubits)])

    @staticmethod
    def get_LABS_terms(number_of_qubits: int):
        """
        Terms of the LABS energy E(s) = sum_{k=1}^{N-1} C_k(s)^2, C_k(s) = sum_{i=0}^{N-k-1} s_i s_{i+k}
        (arXiv:1512.02475), without its constant part (get_LABS_offset) and divided by
        _NORMALIZATION_LABS, sorted by qubit indices.

        Expanding C_k^2 leaves two kinds of terms, each generated exactly once:
        s_a s_{a+2k}, weight 2: the cross term s_a s_{a+k} * s_{a+k} s_{a+2k} of C_k^2;
        s_a s_{a+k1} s_{a+k2} s_{a+k1+k2} with 1 <= k1 < k2, weight 4: a cross term of both
        C_{k1}^2 and C_{k2}^2.
        """
        two_body = [(2 / _NORMALIZATION_LABS, (a, a + 2 * k))
                    for k in range(1, (number_of_qubits - 1) // 2 + 1)
                    for a in range(number_of_qubits - 2 * k)]
        four_body = [(4 / _NORMALIZATION_LABS, (a, a + k1, a + k2, a + k1 + k2))
                     for k1 in range(1, number_of_qubits)
                     for k2 in range(k1 + 1, number_of_qubits - k1)
                     for a in range(number_of_qubits - k1 - k2)]
        return sorted(two_body + four_body, key=lambda weighted_term: weighted_term[1])

    @staticmethod
    def get_full_energy_value(energy:float,
                                number_of_qubits:int):
        return _NORMALIZATION_LABS * energy + LABSHamiltonianGenerator.get_LABS_offset(number_of_qubits=number_of_qubits)

    @staticmethod
    def get_merit_factor(energy:float,
                         number_of_qubits:int):
        full_energy = LABSHamiltonianGenerator.get_full_energy_value(energy=energy,
                                                          number_of_qubits=number_of_qubits)
        return number_of_qubits**2/(2*full_energy)


    @staticmethod
    def get_known_optimal_MF(number_of_qubits:int):
        return _KNOWN_OPTIMAL_MF_LABS[number_of_qubits] if number_of_qubits in _KNOWN_OPTIMAL_MF_LABS else None

    @staticmethod
    def get_known_optimal_energy(
                                 number_of_qubits:int):

        if number_of_qubits not in _KNOWN_OPTIMAL_ENERGIES_LABS:
            return None
        return (_KNOWN_OPTIMAL_ENERGIES_LABS[number_of_qubits] -
                LABSHamiltonianGenerator.get_LABS_offset(number_of_qubits=number_of_qubits)) / _NORMALIZATION_LABS

