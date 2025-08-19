# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.
from quapopt.hamiltonians.generators.RandomClassicalHamiltonianGeneratorBase import RandomClassicalHamiltonianGeneratorBase
from quapopt.data_analysis.data_handling import (CoefficientsDistributionSpecifier,
                                                 HamiltonianClassSpecifierLABS)
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian

import numpy as np
from typing import Optional
from itertools import combinations
from collections import Counter
from tqdm import tqdm

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
                          print_progress_bar:bool=False) -> ClassicalHamiltonian:

        hamiltonian_class_specifier = self._hamiltonian_class_specifier
        hamiltonian_instance_specifier = hamiltonian_class_specifier.instance_specifier_constructor(NumberOfQubits=number_of_qubits,
                                                                                                    HamiltonianInstanceIndex=0)


        if read_from_drive_if_present:
            hamiltonian = self._read_from_drive(hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                                hamiltonian_class_specifier=hamiltonian_class_specifier)
            if hamiltonian is not None:
                return hamiltonian


        ##here is super explicit generation:
        # all_terms_with_duplicates = {}
        # for j in range(1, number_of_qubits):
        #     for i in range(1,number_of_qubits-j+1):
        #         for k in range(1,number_of_qubits-j+1):
        #             subset = [i-1,i+j-1,k-1,k+j-1]
        #             #if there are duplicate indices, we wish to remove them because Z^2 = I
        #             counts = Counter(subset)
        #             subset = tuple(sorted([k for k, v in counts.items() if v % 2 == 1]))
        #             if subset == ():
        #                 continue
        #             if subset not in all_terms_with_duplicates:
        #                 all_terms_with_duplicates[subset] = 1
        #             else:
        #                 all_terms_with_duplicates[subset] +=1
        # hamiltonian_list_representation = [(v,k) for k,v in all_terms_with_duplicates.items()]


        ## Here is generation developed by Daniel
        # for idxi in range(1, number_of_qubits - 1):
        #     for j in range(1, int(np.floor((number_of_qubits - idxi) / 2)) + 1):
        #         hamiltonian_list_representation.append((1.0, (idxi - 1, idxi + 2 * j - 1)))
        #
        # for idxi in range(1, number_of_qubits - 2):
        #     for idxt in range(1, int(np.floor((number_of_qubits - idxi - 1) / 2)) - 1):
        #         for idxk in range(idxt + 1, number_of_qubits - idxt - idxi - 1):
        #             hamiltonian_list_representation.append(
        #                 (2.0, (idxi - 1, idxi + idxt - 1, idxi + idxk - 1, idxi + idxt + idxk - 1)))



        #Here is generation borrowed from qokit
        subsets_set = set()
        for k in tqdm(list(range(1, number_of_qubits)), disable= not print_progress_bar):
            for i, j in combinations(range(1, number_of_qubits - k + 1), 2):
                # Drop duplicate terms, e.g. Z1Z2Z2Z3 should be just Z1Z3

                #We substract 1 from indices because we typically index qubits from 0
                if i + k == j:
                    tup = tuple(sorted((i - 1, j + k - 1)))
                else:
                    tup = tuple(sorted((i - 1, i + k - 1, j - 1, j + k - 1)))

                subsets_set = subsets_set.union({tup})

        hamiltonian_list_representation = sorted([(len(subset)/_NORMALIZATION_LABS, subset) for subset in subsets_set],
                                                 key = lambda x:x[1])
        offset = LABSHamiltonianGenerator.get_LABS_offset(number_of_qubits=number_of_qubits)
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

