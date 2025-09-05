# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)




from typing import Union, Optional
from quapopt.hamiltonians.generators.RandomErdosRenyiHamiltonianGenerator import RandomErdosRenyiHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomSherringtonKirkpatrickHamiltonianGenerator import RandomSKHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomMAX2SATHamiltonianGenerator import RandomMAX2SATHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomWishartHamiltonianGenerator import RandomWishartHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomRegularHamiltonianGenerator import RandomRegularHamiltonianGenerator
from quapopt.hamiltonians.generators.RandomMaxCutHamiltonianGenerator import RandomMaxCutHamiltonianGenerator
from quapopt.hamiltonians.generators.LABSHamiltonianGenerator import LABSHamiltonianGenerator
from quapopt.data_analysis.data_handling import (HamiltonianModels, CoefficientsDistributionSpecifier,
                                                 CoefficientsType,
                                                 CoefficientsDistribution,
ERDOS_RENYI_TYPES)


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
def build_hamiltonian_generator(hamiltonian_model:HamiltonianModels,
                                coefficients_distribution_specifier=None,
                                localities=None,
                                erdos_renyi_type:ERDOS_RENYI_TYPES=None)->Union[RandomSKHamiltonianGenerator,
                                                        RandomRegularHamiltonianGenerator,
                                                        RandomErdosRenyiHamiltonianGenerator,
                                                        RandomMaxCutHamiltonianGenerator,
                                                        RandomWishartHamiltonianGenerator,
                                                        RandomMAX2SATHamiltonianGenerator]:
    arguments_dict_builder = {}
    if hamiltonian_model in [HamiltonianModels.SherringtonKirkpatrick,
                             HamiltonianModels.RegularGraph]:
        arguments_dict_builder['coefficients_distribution_specifier'] = coefficients_distribution_specifier
        arguments_dict_builder['localities'] = localities

    elif hamiltonian_model in [HamiltonianModels.ErdosRenyi]:
        arguments_dict_builder['coefficients_distribution_specifier'] = coefficients_distribution_specifier
        arguments_dict_builder['localities'] = localities
        arguments_dict_builder['erdos_renyi_type'] = erdos_renyi_type

    elif hamiltonian_model in [HamiltonianModels.MaxCut]:
        arguments_dict_builder['coefficients_distribution_specifier'] = coefficients_distribution_specifier

    elif hamiltonian_model in [HamiltonianModels.WishartPlantedEnsemble]:
        pass
    elif hamiltonian_model in [HamiltonianModels.MAX2SAT]:
        pass
    elif hamiltonian_model in [HamiltonianModels.LABS]:
        pass

    if hamiltonian_model == HamiltonianModels.MaxCut:
        builder = _AllHamiltonianGenerators.maxcut
    elif hamiltonian_model == HamiltonianModels.RegularGraph:
        builder = _AllHamiltonianGenerators.regular
    elif hamiltonian_model == HamiltonianModels.WishartPlantedEnsemble:
        builder = _AllHamiltonianGenerators.wishart
    elif hamiltonian_model == HamiltonianModels.MAX2SAT:
        builder = _AllHamiltonianGenerators.max2sat
    elif hamiltonian_model == HamiltonianModels.SherringtonKirkpatrick:
        builder = _AllHamiltonianGenerators.sherrington_kirkpatrick
    elif hamiltonian_model == HamiltonianModels.ErdosRenyi:
        builder = _AllHamiltonianGenerators.erdos_renyi
    elif hamiltonian_model == HamiltonianModels.LABS:
        builder = _AllHamiltonianGenerators.labs
    else:
        raise ValueError(f"Unknown hamiltonian model {hamiltonian_model}")

    return builder(**arguments_dict_builder)


def create_hamiltonian_from_descriptions(class_description: str, 
                                       instance_description: str,
                                       read_from_drive_if_present: bool = True,
                                       default_backend: Optional[str] = None):
    """
    Create a Hamiltonian instance from class and instance description strings.
    
    Args:
        class_description: Class description string (e.g., "HMN=ER;LOC=(2,);CFD=CT~DIS_CDN~UNI...")
        instance_description: Instance description string (e.g., "NOQ=8;HII=3;EPA=0.5")
        read_from_drive_if_present: Whether to read from drive if present
        default_backend: Backend to use for computation
        
    Returns:
        ClassicalHamiltonian instance
    """
    from quapopt.data_analysis.data_handling import (
        reconstruct_hamiltonian_class_specifier, reconstruct_hamiltonian_instance_specifier
    )
    import inspect
    
    # Reconstruct the class and instance specifiers
    class_specifier = reconstruct_hamiltonian_class_specifier(class_description)
    instance_specifier = reconstruct_hamiltonian_instance_specifier(instance_description, class_specifier)
    
    # Map class specifier to HAMILTONIAN_MODEL
    hamiltonian_model_name = class_specifier.HamiltonianModelName.id
    model_mapping = {
        HamiltonianModels.ErdosRenyi.id: HamiltonianModels.ErdosRenyi,
        HamiltonianModels.MaxCut.id: HamiltonianModels.MaxCut,
        HamiltonianModels.SherringtonKirkpatrick.id: HamiltonianModels.SherringtonKirkpatrick,
        HamiltonianModels.RegularGraph.id: HamiltonianModels.RegularGraph,
        HamiltonianModels.MAX2SAT.id: HamiltonianModels.MAX2SAT,
        HamiltonianModels.MAXkSAT.id: HamiltonianModels.MAXkSAT,
        HamiltonianModels.WishartPlantedEnsemble.id: HamiltonianModels.WishartPlantedEnsemble,
        HamiltonianModels.LABS.id: HamiltonianModels.LABS
    }
    
    hamiltonian_model = model_mapping.get(hamiltonian_model_name)
    if hamiltonian_model is None:
        raise ValueError(f"Unknown hamiltonian model: {hamiltonian_model_name}")
    
    # Extract parameters from specifiers
    coefficients_distribution_specifier = class_specifier.CoefficientsDistributionSpecifier
    localities = class_specifier.Localities
    
    # Handle Erdos-Renyi type if present
    erdos_renyi_type = None
    if hasattr(class_specifier, 'ErdosRenyiType'):
        erdos_renyi_type = class_specifier.ErdosRenyiType
    
    # Create generator using existing function
    generator = build_hamiltonian_generator(
        hamiltonian_model=hamiltonian_model,
        coefficients_distribution_specifier=coefficients_distribution_specifier,
        localities=localities,
        erdos_renyi_type=erdos_renyi_type
    )
    
    # Extract instance parameters
    number_of_qubits = instance_specifier.NumberOfQubits
    seed = instance_specifier.HamiltonianInstanceIndex
    
    # Get generator's generate_instance signature to handle different parameter requirements
    sig = inspect.signature(generator.generate_instance)
    kwargs = {
        'number_of_qubits': number_of_qubits,
        'seed': seed,
        'read_from_drive_if_present': read_from_drive_if_present,
        'default_backend': default_backend
    }
    
    # Add EdgeProbabilityOrAmount if the instance specifier has it and the generator expects it
    if hasattr(instance_specifier, 'EdgeProbabilityOrAmount') and 'p_or_M' in sig.parameters:
        kwargs['p_or_M'] = instance_specifier.EdgeProbabilityOrAmount
    
    # Only include parameters that exist in the signature
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    
    return generator.generate_instance(**valid_kwargs)


if __name__ == '__main__':
    hamiltonian_model_test = HamiltonianModels.SherringtonKirkpatrick
    localities_test = (1,2)

    coefficients_type_test = CoefficientsType.CONSTANT
    coefficients_distribution_test = CoefficientsDistribution.Constant
    coefficients_distribution_properties_test = {'value': -1.0}
    coefficients_distribution_specifier_test = CoefficientsDistributionSpecifier(
        CoefficientsType=coefficients_type_test,
        CoefficientsDistributionName=coefficients_distribution_test,
        CoefficientsDistributionProperties=coefficients_distribution_properties_test)


    generator_cost_hamiltonian_test = build_hamiltonian_generator(
        hamiltonian_model=hamiltonian_model_test,
        localities=localities_test,
        coefficients_distribution_specifier=coefficients_distribution_specifier_test
    )

    system_size_problem_test = 12
    seed_cost_hamiltonian_test = 0

    cost_hamiltonian_test = generator_cost_hamiltonian_test.generate_instance(
        number_of_qubits=system_size_problem_test,
        seed=seed_cost_hamiltonian_test,
        read_from_drive_if_present=True,
        default_backend='numpy'
    )

    cost_hamiltonian_class_description_test = cost_hamiltonian_test.hamiltonian_class_description
    cost_hamiltonian_instance_description_test = cost_hamiltonian_test.hamiltonian_instance_description

    print("Class description (cost):", cost_hamiltonian_class_description_test)
    print("Instance description (cost):", cost_hamiltonian_instance_description_test)

    cost_hamiltonian_reconstructed_test = create_hamiltonian_from_descriptions(class_description=cost_hamiltonian_class_description_test,
                                                                            instance_description=cost_hamiltonian_instance_description_test,
                                                                            read_from_drive_if_present=True,
                                                                            default_backend='numpy')
    cost_hamiltonian_class_description_test_reconstructed = cost_hamiltonian_reconstructed_test.hamiltonian_class_description
    cost_hamiltonian_instance_description_test_reconstructed = cost_hamiltonian_reconstructed_test.hamiltonian_instance_description
    print("Reconstructed class description (cost):",
          cost_hamiltonian_class_description_test_reconstructed
          )
    print("Reconstructed instance description (cost):",cost_hamiltonian_instance_description_test_reconstructed)
    same_class_test = cost_hamiltonian_class_description_test == cost_hamiltonian_class_description_test_reconstructed
    same_instance_test = cost_hamiltonian_instance_description_test == cost_hamiltonian_instance_description_test_reconstructed
    print("Same class description:", same_class_test)
    print("Same instance description:", same_instance_test)


