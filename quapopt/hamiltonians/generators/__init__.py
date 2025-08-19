# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.



from typing import Union
from quapopt.hamiltonians.generators.RandomErdosRenyiHamiltonianGenerator import RandomErdosRenyiHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomSherringtonKirkpatrickHamiltonianGenerator import RandomSKHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomMAX2SATHamiltonianGenerator import RandomMAX2SATHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomWishartHamiltonianGenerator import RandomWishartHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomRegularHamiltonianGenerator import RandomRegularHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomMaxCutHamiltonianGenerator import RandomMaxCutHamiltonianGenerator
from quapopt.hamiltonians.generators.LABSHamiltonianGenerator import LABSHamiltonianGenerator
from quapopt.data_analysis.data_handling import HAMILTONIAN_MODELS, ERDOS_RENYI_TYPES

import attr

class _AllHamiltonianGenerators:
    #Require coefficients_distribution_specifier and localities
    sherrington_kirkpatrick = RandomSKHamiltonianGenerator
    regular = RandomRegularHamiltonianGenerator

    # Require coefficients_distribution_specifier, localities and ErdosRenyiType
    erdos_renyi = RandomErdosRenyiHamiltonianGenerator

    # Require coefficients_distribution_specifier and ErdosRenyiType
    maxcut = RandomMaxCutHamiltonianGenerator

    #require nothing
    max2sat = RandomMAX2SATHamiltonianGenerator
    wishart = RandomWishartHamiltonianGenerator
    labs = LABSHamiltonianGenerator




#TODO(FBM): perhaps should create a HamiltonianGeneratorFactory class instead of this function
def build_hamiltonian_generator(hamiltonian_model:HAMILTONIAN_MODELS,
                                coefficients_distribution_specifier=None,
                                localities=None,
                                erdos_renyi_type=None)->Union[RandomSKHamiltonianGenerator,
                                                        RandomRegularHamiltonianGenerator,
                                                        RandomErdosRenyiHamiltonianGenerator,
                                                        RandomMaxCutHamiltonianGenerator,
                                                        RandomWishartHamiltonianGenerator,
                                                        RandomMAX2SATHamiltonianGenerator]:
    arguments_dict_builder = {}
    if hamiltonian_model in [HAMILTONIAN_MODELS.SherringtonKirkpatrick,
                             HAMILTONIAN_MODELS.RegularGraph]:
        arguments_dict_builder['coefficients_distribution_specifier'] = coefficients_distribution_specifier
        arguments_dict_builder['localities'] = localities

    elif hamiltonian_model in [HAMILTONIAN_MODELS.ErdosRenyi]:
        arguments_dict_builder['coefficients_distribution_specifier'] = coefficients_distribution_specifier
        arguments_dict_builder['localities'] = localities
        arguments_dict_builder['ErdosRenyiType'] = erdos_renyi_type

    elif hamiltonian_model in [HAMILTONIAN_MODELS.MaxCut]:
        arguments_dict_builder['coefficients_distribution_specifier'] = coefficients_distribution_specifier

    elif hamiltonian_model in [HAMILTONIAN_MODELS.WishartPlantedEnsemble]:
        pass
    elif hamiltonian_model in [HAMILTONIAN_MODELS.MAX2SAT]:
        pass
    elif hamiltonian_model in [HAMILTONIAN_MODELS.LABS]:
        pass

    if hamiltonian_model == HAMILTONIAN_MODELS.MaxCut:
        builder = _AllHamiltonianGenerators.maxcut
    elif hamiltonian_model == HAMILTONIAN_MODELS.RegularGraph:
        builder = _AllHamiltonianGenerators.regular
    elif hamiltonian_model == HAMILTONIAN_MODELS.WishartPlantedEnsemble:
        builder = _AllHamiltonianGenerators.wishart
    elif hamiltonian_model == HAMILTONIAN_MODELS.MAX2SAT:
        builder = _AllHamiltonianGenerators.max2sat
    elif hamiltonian_model == HAMILTONIAN_MODELS.SherringtonKirkpatrick:
        builder = _AllHamiltonianGenerators.sherrington_kirkpatrick
    elif hamiltonian_model == HAMILTONIAN_MODELS.ErdosRenyi:
        builder = _AllHamiltonianGenerators.erdos_renyi
    elif hamiltonian_model == HAMILTONIAN_MODELS.LABS:
        builder = _AllHamiltonianGenerators.labs
    else:
        raise ValueError(f"Unknown hamiltonian model {hamiltonian_model}")

    return builder(**arguments_dict_builder)