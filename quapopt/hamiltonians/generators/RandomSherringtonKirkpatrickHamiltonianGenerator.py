# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

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
