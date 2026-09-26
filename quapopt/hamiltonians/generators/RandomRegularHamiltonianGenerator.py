# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import Tuple, Optional, ClassVar, Dict

import networkx as nx

from quapopt.data_analysis.data_handling import (CoefficientsDistributionSpecifier,
                                                 HamiltonianClassSpecifierRegular)
from quapopt.hamiltonians.generators.RandomClassicalHamiltonianGeneratorBase import \
    RandomClassicalHamiltonianGeneratorBase
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian


class RandomRegularHamiltonianGenerator(RandomClassicalHamiltonianGeneratorBase):
    _SPECIFIER_FIELD_TO_KWARG: ClassVar[Dict[str, Optional[str]]] = {'AverageDegree': 'graph_degree',
                                                                     }

    def __init__(self,
                 coefficients_distribution_specifier: CoefficientsDistributionSpecifier = None,
                 localities: Tuple[int, ...] = (2,),
                 hamiltonian_class_specifier: Optional[HamiltonianClassSpecifierRegular] = None
                 ):
        """
        Generates a random regular Hamiltonian.
        :param coefficients_distribution_specifier:
        Specifies the distribution of coefficients for the Hamiltonian.
        :param localities:
        Possible values: (1,2), or (2,). If (1,2), the Hamiltonian will have both 1-local and 2-local terms.
        """
        assert set(localities) in [{2}, {1, 2}], "Localities must be either (2,) or (1, 2)."

        if hamiltonian_class_specifier is None:
            hamiltonian_class_specifier = HamiltonianClassSpecifierRegular(
                CoefficientsDistributionSpecifier=coefficients_distribution_specifier,
                Localities=localities)

        super().__init__(hamiltonian_class_specifier=hamiltonian_class_specifier)

    def generate_instance(self,
                          number_of_qubits: int,
                          graph_degree: int,
                          seed: Optional[int] = None,
                          read_from_drive_if_present=True,
                          default_backend: Optional[str] = None) -> ClassicalHamiltonian:

        hamiltonian_class_specifier = self._hamiltonian_class_specifier
        hamiltonian_instance_specifier = hamiltonian_class_specifier.instance_specifier_constructor(number_of_qubits,
                                                                                                    seed,
                                                                                                    graph_degree)

        if read_from_drive_if_present:
            hamiltonian = self._read_from_drive(hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                                hamiltonian_class_specifier=hamiltonian_class_specifier)
            if hamiltonian is not None:
                return hamiltonian

        random_graph_structure = nx.random_regular_graph(d=graph_degree,
                                                         n=number_of_qubits,
                                                         seed=seed)

        if 1 in self.localities:
            for node in range(number_of_qubits):
                random_graph_structure.add_edge(node, node)

        # TODO FBM: make sure that the data structure is compatible with the one used in the other generators
        subsets_generator = list(random_graph_structure.edges())

        return self._generate_instance(number_of_qubits=number_of_qubits,
                                       term_addition_function=None,
                                       subsets_generator=subsets_generator,
                                       seed=seed,
                                       read_from_drive_if_present=read_from_drive_if_present,
                                       hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                       default_backend=default_backend)
