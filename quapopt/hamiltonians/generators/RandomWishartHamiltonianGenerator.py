# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

 
#import all types from typing
from typing import Optional

from quapopt.hamiltonians.generators.RandomClassicalHamiltonianGeneratorBase import RandomClassicalHamiltonianGeneratorBase
#TODO FBM: move chook_wrapper_wishart_planted to quapopt
# from quapopt.additional_packages.ancillary_functions_usra.hamiltonians.wrapped_solvers import wrapped_chook_wrapper_wishart_planted

from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian


from quapopt.data_analysis.data_handling import (
    HamiltonianClassSpecifierWishartPlantedEnsemble,)


class RandomWishartHamiltonianGenerator(RandomClassicalHamiltonianGeneratorBase):

    def __init__(self,
                 planted_ground_state=None):

        hamiltonian_class_specifier = HamiltonianClassSpecifierWishartPlantedEnsemble()

        super().__init__(hamiltonian_class_specifier=hamiltonian_class_specifier)

        self._planted_ground_state = planted_ground_state

    @property
    def planted_ground_state(self):
        return self._planted_ground_state
    def generate_instance(self,
                          number_of_qubits:int,
                          wishart_density:Optional[int]=None,
                          seed=None,
                          read_from_drive_if_present=False,
                          default_backend:Optional[str]=None)->ClassicalHamiltonian:


        if wishart_density is None:
            wishart_density = 1/number_of_qubits



        hamiltonian_class_specifier:HamiltonianClassSpecifierWishartPlantedEnsemble = self._hamiltonian_class_specifier
        hamiltonian_instance_specifier = hamiltonian_class_specifier.instance_specifier_constructor(NumberOfQubits=number_of_qubits,
                                                                                                    HamiltonianInstanceIndex=seed,
                                                                                                    WishartDensity=wishart_density)


        if read_from_drive_if_present:
            hamiltonian = self._read_from_drive(hamiltonian_instance_specifier=hamiltonian_instance_specifier)
            if hamiltonian is not None:
                return hamiltonian


        number_of_columns = int(wishart_density*number_of_qubits)

        hamiltonian_list, ground_state_energy, ground_state = wrapped_chook_wrapper_wishart_planted(number_of_qubits=number_of_qubits,
                                                         number_of_columns=number_of_columns,
                                                         planted_ground_state=self._planted_ground_state,
                                                         seed=seed,
                                                         discretize_couplers=False,
                                                         gauge_transform=False,
                                                         convert_to_hobo=False)

        #TODO FBM: add the description of the class and instance
        class_specific_data = dict(planted_solution=ground_state)

        hamiltonian = ClassicalHamiltonian(number_of_qubits=number_of_qubits,
                                           hamiltonian_list_representation=hamiltonian_list,
                                           hamiltonian_class_specifier=hamiltonian_class_specifier,
                                           hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                           class_specific_data=class_specific_data)

        hamiltonian._lowest_energy = ground_state_energy
        hamiltonian._lowest_energy_state = ground_state

        return self._generate_instance(number_of_qubits=number_of_qubits,
                                       random_instance=hamiltonian,
                                       default_backend=default_backend, )
