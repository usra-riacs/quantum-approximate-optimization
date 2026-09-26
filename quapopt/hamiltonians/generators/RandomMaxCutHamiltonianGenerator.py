# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from quapopt.data_analysis.data_handling import (
    CoefficientsDistributionSpecifier,
    ERDOS_RENYI_TYPES,
BaseName)
from quapopt.data_analysis.data_handling import HamiltonianClassSpecifierMaxCut
from quapopt.hamiltonians.generators.RandomErdosRenyiHamiltonianGenerator import \
    RandomErdosRenyiHamiltonianGenerator


class RandomMaxCutHamiltonianGenerator(RandomErdosRenyiHamiltonianGenerator):
    def __init__(self,
                 coefficients_distribution_specifier: CoefficientsDistributionSpecifier = None,
                 erdos_renyi_type: BaseName = ERDOS_RENYI_TYPES.Gnp
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
