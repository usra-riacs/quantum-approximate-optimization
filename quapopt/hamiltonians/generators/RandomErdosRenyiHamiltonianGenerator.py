# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.
 
import itertools
from typing import Optional, Union
import numpy as np
from quapopt.data_analysis.data_handling import (
    HamiltonianClassSpecifierErdosRenyi,
    CoefficientsDistributionSpecifier,
    ERDOS_RENYI_TYPES)
from quapopt.hamiltonians.generators.RandomClassicalHamiltonianGeneratorBase import \
    RandomClassicalHamiltonianGeneratorBase, _get_default_coefficient_sampling_function
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian


class RandomErdosRenyiHamiltonianGenerator(RandomClassicalHamiltonianGeneratorBase):
    def __init__(self,
                 coefficients_distribution_specifier: CoefficientsDistributionSpecifier = None,
                 localities=(2,),
                 erdos_renyi_type: Optional[ERDOS_RENYI_TYPES] = None,
                 hamiltonian_class_specifier=None
                 ):

        """
        Initializes a random Erdos-Renyi Hamiltonian generator.
        :param coefficients_distribution_specifier:
        :param localities:
        A tuple indicating the localities of the Hamiltonian terms.
        :param erdos_renyi_type:
        :param hamiltonian_class_specifier:
        """


        if erdos_renyi_type is None:
            #default to Gnp model
            erdos_renyi_type = ERDOS_RENYI_TYPES.Gnp

        assert set(localities) in [{2}, {1,2}], "Localities must be either (2,) or (1, 2)."

        if hamiltonian_class_specifier is None:
            #generic ER graph
            hamiltonian_class_specifier = HamiltonianClassSpecifierErdosRenyi(Localities=localities,
                                                                              CoefficientsDistributionSpecifier=coefficients_distribution_specifier,
                                                                              ErdosRenyiType=erdos_renyi_type)

        super().__init__(hamiltonian_class_specifier=hamiltonian_class_specifier)

    def generate_instance(self,
                          number_of_qubits: int,
                          p_or_M: Union[int, float],
                          seed: Optional[int] = None,
                          read_from_drive_if_present: bool = True,
                          default_backend:Optional[str]=None) -> ClassicalHamiltonian:

        hamiltonian_class_specifier:HamiltonianClassSpecifierErdosRenyi = self._hamiltonian_class_specifier
        hamiltonian_instance_specifier = hamiltonian_class_specifier.instance_specifier_constructor(NumberOfQubits=number_of_qubits,
                                                                                                    HamiltonianInstanceIndex=seed,
                                                                                                    EdgeProbabilityOrAmount=p_or_M)



        erdos_renyi_type = hamiltonian_class_specifier.ErdosRenyiType


        if erdos_renyi_type == ERDOS_RENYI_TYPES.Gnp:
            #In this model, graph is constructed by adding an edge between each pair of nodes with probability p
            subsets_generator_function = itertools.combinations(range(number_of_qubits), 2)
            probability_of_adding_edge = p_or_M

            def term_addition_function(numpy_rng, size: int):
                return numpy_rng.uniform(size=size,
                                         low=0.0,
                                         high=1.0) < probability_of_adding_edge

        else:
            #In this model, graph is constructed by adding M edges randomly
            number_of_edges = p_or_M
            term_addition_function = None

            def subsets_generator_function(numpy_rng):
                subsets_generator = list(itertools.combinations(range(number_of_qubits), 2))
                numpy_rng.shuffle(subsets_generator)
                return subsets_generator[0:number_of_edges]

        localities = hamiltonian_class_specifier.Localities

        ham = self._generate_instance(number_of_qubits=number_of_qubits,
                                      term_addition_function=term_addition_function,
                                      subsets_generator=subsets_generator_function,
                                      seed=seed,
                                      read_from_drive_if_present=read_from_drive_if_present,
                                      hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                      default_backend=default_backend, )


        if set(localities) == {2}:
            return ham

        #otherwise, we add single_qubit terms
        _coefficient_sampling_function = _get_default_coefficient_sampling_function(coefficients_distribution=self.coefficients_distribution,
                                                                              coefficients_distribution_properties=self.coefficients_distribution_properties,
                                                                                 coefficients_type=self.coefficients_type)
        numpy_rng = np.random.default_rng(seed)

        coeffs_1q = _coefficient_sampling_function(numpy_rng, number_of_qubits)

        terms_1q = [(coeffs_1q[i], (i,) ) for i in range(number_of_qubits)]

        return ClassicalHamiltonian(hamiltonian_list_representation=terms_1q+ham.hamiltonian_list_representation,
                                    hamiltonian_class_specifier=ham.hamiltonian_class_specifier,
                                    hamiltonian_instance_specifier=ham.hamiltonian_instance_specifier,
                                    number_of_qubits=number_of_qubits,
                                    default_backend=default_backend,)

