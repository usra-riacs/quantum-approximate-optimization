# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import time
from typing import Optional, List, Tuple, Union, Any

from quapopt.data_analysis.data_handling import (CoefficientsType,
                                                 CoefficientsDistribution,
                                                 CoefficientsDistributionSpecifier,
                                                 HamiltonianModels, LoggingLevel)
from quapopt.hamiltonians.generators import build_hamiltonian_generator
from quapopt.hamiltonians.generators.RandomClassicalHamiltonianGeneratorBase import RandomClassicalHamiltonianGeneratorBase

from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import QubitMappingType
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from quapopt.optimization.parameter_setting.non_adaptive_optimization.SimpleGridOptimizer import SimpleGridOptimizer
from quapopt.optimization.parameter_setting.variational.custom_optimizers.DivideAndConquerGridOptimizer import DivideAndConquerGridOptimizer
from quapopt.optimization.parameter_setting.variational.custom_optimizers.ConsecutiveOptimizersRunner import ConsecutiveOptimizersRunner

from quapopt.optimization.parameter_setting.variational.scipy_tools.ScipyOptimizerWrapped import ScipyOptimizerWrapped
from quapopt.optimization.parameter_setting import OptimizerType


from quapopt.optimization.parameter_setting import ParametersBoundType

from quapopt import ancillary_functions as anf
from quapopt.data_analysis.data_handling.schemas.naming import (STANDARD_NAMES_DATA_TYPES as SNDT,
                                                 STANDARD_NAMES_VARIABLES as SNV)

from quapopt.data_analysis.data_handling.schemas.naming import (MAIN_KEY_SEPARATOR as MKS,
                                                                MAIN_KEY_VALUE_SEPARATOR as MKVS)

def get_default_hamiltonian_class_kwargs(localities=None,
                                         coefficients_type=None,
                                         coefficients_distribution=None,
                                         coefficients_distribution_properties=None) -> dict:
    """
    Resolve partially-specified Hamiltonian class arguments into fully-specified ones.
    Single source of truth for the defaults shared by get_hamiltonian_generator,
    generate_random_hamiltonian, and get_hamiltonian_class_description.
    """
    if localities is None:
        localities = (2,)

    if coefficients_type is None:
        coefficients_type = CoefficientsType.CONTINUOUS

    if coefficients_distribution is None:
        if coefficients_type == CoefficientsType.CONTINUOUS:
            coefficients_distribution = CoefficientsDistribution.Normal
        elif coefficients_type == CoefficientsType.DISCRETE:
            coefficients_distribution = CoefficientsDistribution.Uniform
        elif coefficients_type == CoefficientsType.CONSTANT:
            coefficients_distribution = CoefficientsDistribution.Constant

    if coefficients_distribution_properties is None:
        if coefficients_distribution == CoefficientsDistribution.Normal:
            coefficients_distribution_properties = dict(loc=0.0, scale=1.0)
        elif coefficients_distribution == CoefficientsDistribution.Uniform:
            if coefficients_type == CoefficientsType.CONTINUOUS:
                coefficients_distribution_properties = dict(low=-1, high=1)
            elif coefficients_type == CoefficientsType.DISCRETE:
                # Must match CoefficientsDistributionSpecifier.__post_init__: dict(low=-1, high=1, step=1)
                # samples the same set (0 is excluded) but serializes to a different class description.
                coefficients_distribution_properties = dict(values=[-1, 1])
        elif coefficients_distribution == CoefficientsDistribution.Constant:
            coefficients_distribution_properties = dict(value=1)

    return dict(localities=localities,
                coefficients_type=coefficients_type,
                coefficients_distribution=coefficients_distribution,
                coefficients_distribution_properties=coefficients_distribution_properties)


def get_default_hamiltonian_instance_kwargs(hamiltonian_model) -> dict:
    """
    Default model-specific kwargs passed to generator.generate_instance
    (e.g., edge probability for Erdos-Renyi).
    """
    instance_kwargs = {}
    if hamiltonian_model in [HamiltonianModels.ErdosRenyi]:
        instance_kwargs['p_or_M'] = 0.1
    elif hamiltonian_model in [HamiltonianModels.MAX2SAT]:
        instance_kwargs['clause_density'] = 1.5
    elif hamiltonian_model in [HamiltonianModels.RegularGraph]:
        instance_kwargs['graph_degree'] = 3
    return instance_kwargs


def get_hamiltonian_generator(hamiltonian_model,
                              localities=None,
                              coefficients_type=None,
                              coefficients_distribution=None,
                              coefficients_distribution_properties=None,
                              class_specific_kwargs:Optional[dict]=None)->RandomClassicalHamiltonianGeneratorBase:

    class_kwargs = get_default_hamiltonian_class_kwargs(localities=localities,
                                                        coefficients_type=coefficients_type,
                                                        coefficients_distribution=coefficients_distribution,
                                                        coefficients_distribution_properties=coefficients_distribution_properties)


    coefficients_distribution_specifier = CoefficientsDistributionSpecifier(
        CoefficientsType=class_kwargs['coefficients_type'],
        CoefficientsDistributionName=class_kwargs['coefficients_distribution'],
        CoefficientsDistributionProperties=class_kwargs['coefficients_distribution_properties'])



    generator_cost_hamiltonian = build_hamiltonian_generator(hamiltonian_model=hamiltonian_model,
                                                             localities=class_kwargs.get('localities', localities),
                                                             coefficients_distribution_specifier=coefficients_distribution_specifier,
                                                             class_specific_kwargs=class_specific_kwargs)

    return generator_cost_hamiltonian



def generate_random_hamiltonian(number_of_qubits: Optional[int]=None,
                                default_backend='numpy',
                                hamiltonian_model=HamiltonianModels.ErdosRenyi,
                                localities:Optional[Tuple[int,...]]=None,
                                coefficients_type=None,
                                coefficients_distribution=None,
                                coefficients_distribution_properties=None,
                                instance_kwargs=None,
                                seed=0,
                                class_specific_kwargs:Optional[dict]=None
                                )->ClassicalHamiltonian:

    if instance_kwargs is None:
        instance_kwargs = get_default_hamiltonian_instance_kwargs(hamiltonian_model=hamiltonian_model)

    generator_cost_hamiltonian = get_hamiltonian_generator(hamiltonian_model=hamiltonian_model,
                                                           coefficients_type=coefficients_type,
                                                           coefficients_distribution=coefficients_distribution,
                                                           coefficients_distribution_properties=coefficients_distribution_properties,
                                                           localities=localities,
                                                           class_specific_kwargs=class_specific_kwargs)


    cost_hamiltonian = generator_cost_hamiltonian.generate_instance(number_of_qubits=number_of_qubits,
                                                                    seed=seed,
                                                                    read_from_drive_if_present=True,
                                                                    default_backend=default_backend,
                                                                    **instance_kwargs
                                                                    )

    return cost_hamiltonian


def get_hamiltonian_class_description(hamiltonian_model,
                                      localities=None,
                                      coefficients_type=None,
                                      coefficients_distribution=None,
                                      coefficients_distribution_properties=None,
                                      class_specific_kwargs:Optional[dict]=None,):

    generator = get_hamiltonian_generator(hamiltonian_model=hamiltonian_model,
                                          localities=localities,
                                          coefficients_type=coefficients_type,
                                          coefficients_distribution=coefficients_distribution,
                                          coefficients_distribution_properties=coefficients_distribution_properties,
                                          class_specific_kwargs=class_specific_kwargs)

    return generator.hamiltonian_class_description






def generate_random_sampling_results(hamiltonian: ClassicalHamiltonian,
                                     number_of_samples_per_trial: int,
                                     number_of_trials: int,
                                     p_1=0.5,
                                     seeds_range=None,
                                     show_progress_bar: bool = False,
                                     logging_level: Optional[LoggingLevel] = None,
                                     logger_kwargs: Optional[dict] = None,
                                     backend: Optional[str] = None,
                                     max_memory: Optional[int] = None,
                                     return_bitstrings_and_energies=False,
                                     )->List[pd.DataFrame]:
    from quapopt.optimization.classical_solvers.RandomBitstringSampler import RandomBitstringSampler

    _rbs = RandomBitstringSampler(cost_hamiltonian=hamiltonian,
                                  logging_level=logging_level,
                                  logger_kwargs=logger_kwargs,
                                  backend=backend,
                                  max_memory=max_memory,
                                  )
    if seeds_range is None:
        seeds_range = [0]

    all_results = []
    for seed in seeds_range:
        res = _rbs.sample_solutions(number_of_samples=number_of_samples_per_trial,
                                    number_of_trials=number_of_trials,
                                    seed=seed,
                                    show_progress_bar=show_progress_bar,
                                    return_all_results=return_bitstrings_and_energies,
                                    p_1=p_1)

        all_results.append(res[1])



    return all_results



    return local_sampler_callable_random_sampling
def read_random_sampling_results(cost_hamiltonian,
                                  logger_kwargs,
                                 p_1:float=0.5,
                                 return_none_if_not_found:bool=True)->pd.DataFrame:
    from quapopt.optimization.classical_solvers.RandomBitstringSampler import RandomBitstringSampler

    _rbs = RandomBitstringSampler(cost_hamiltonian=cost_hamiltonian,
                                  logging_level=LoggingLevel.VERY_DETAILED,
                                  logger_kwargs=logger_kwargs)

    return _rbs.read_energies_histogram(p_1=p_1,
                                        return_none_if_not_found=return_none_if_not_found)








def read_or_generate_random_sampling_results(hamiltonian: ClassicalHamiltonian,
                                             number_of_samples_per_trial: int,
                                             number_of_trials: int,
                                             p_1=0.5,
                                             seeds_range=None,
                                             show_progress_bar: bool = False,
                                             logging_level: Optional[LoggingLevel] = None,
                                             logger_kwargs: Optional[dict] = None,
                                             backend: Optional[str] = None,
                                             max_memory: Optional[int] = None,
                                             return_bitstrings_and_energies=False)->Union[pd.DataFrame,Tuple[pd.DataFrame, Tuple[pd.DataFrame,pd.DataFrame]]]:



    if logger_kwargs is not None:
        df_try = read_random_sampling_results(cost_hamiltonian=hamiltonian,
                                              logger_kwargs=logger_kwargs,
                                              p_1=p_1,
                                              return_none_if_not_found=True)

        if df_try is not None:
            return df_try

        generate_random_sampling_results(hamiltonian=hamiltonian,
                                         number_of_samples_per_trial=number_of_samples_per_trial,
                                         number_of_trials=number_of_trials,
                                         p_1=p_1,
                                         seeds_range=seeds_range,
                                         show_progress_bar=show_progress_bar,
                                         logging_level=logging_level,
                                         logger_kwargs=logger_kwargs,
                                         backend=backend,
                                         max_memory=max_memory,
                                         return_bitstrings_and_energies=return_bitstrings_and_energies,
                                         )

        return read_random_sampling_results(cost_hamiltonian=hamiltonian,
                                            logger_kwargs=logger_kwargs,
                                            p_1=p_1)

    res_gen = generate_random_sampling_results(hamiltonian=hamiltonian,
                                     number_of_samples_per_trial=number_of_samples_per_trial,
                                     number_of_trials=number_of_trials,
                                     p_1=p_1,
                                     seeds_range=seeds_range,
                                     show_progress_bar=show_progress_bar,
                                     logging_level=logging_level,
                                     logger_kwargs=logger_kwargs,
                                     backend=backend,
                                     max_memory=max_memory,
                                     return_bitstrings_and_energies=return_bitstrings_and_energies,
                                     )


    if return_bitstrings_and_energies:
        dfs_summary, dfs_en, dfs_bts = [], [], []
        for df_res, (df_en, df_bts) in res_gen:
            dfs_summary.append(df_res)
            dfs_en.append(df_en)
            dfs_bts.append(df_bts)

        df_summary = pd.concat(dfs_summary, axis=0,ignore_index=True)
        df_en = pd.concat(dfs_en, axis=0,ignore_index=True)
        df_bts = pd.concat(dfs_bts, axis=0,ignore_index=True)

        return df_summary, (df_en, df_bts)

    return pd.concat(res_gen, axis=0,ignore_index=True)











def get_solutions_away_from_ground_state(cost_hamiltonian:ClassicalHamiltonian,
                                         max_flips:int,
                                         step_size:int=1,
                                         same_size_flips_max_amount:int=5,
                                         max_solutions:Optional[int]=None,
                                         show_progress_bar:bool=False,
                                         return_energies:bool=False,
                                         seed:int=0,
                                         min_ar:float=0.5,
                                         max_ar:float=1.0
                                         )->List[Tuple[int,...]]|Tuple[List[Tuple[int,...]],List[float]]:



    flips_range = list(range(0, max_flips+1, step_size))

    number_of_qubits = cost_hamiltonian.number_of_qubits
    ground_state = cost_hamiltonian.ground_state
    ground_state_array = np.array(ground_state)

    all_gauges = []
    _already_done = set()
    for number_of_flips in tqdm(flips_range,disable = not show_progress_bar):
        numpy_rng_flips = np.random.default_rng(seed=seed)
        if max_solutions is not None:
            if len(all_gauges) >= max_solutions:
                break

        for _ in list(range(same_size_flips_max_amount)):
            if max_solutions is not None:
                if len(all_gauges) >= max_solutions:
                    break

            indices_to_flip = numpy_rng_flips.choice(number_of_qubits,
                                                     size=number_of_flips,
                                                     replace=False)

            ground_state_array_flipped = ground_state_array.copy()

            # now we flip only those indices that are in indices_to_flip
            if number_of_flips != 0:
                ground_state_array_flipped[indices_to_flip] = 1 - ground_state_array_flipped[indices_to_flip]

            ground_state_flipped = tuple(ground_state_array_flipped.tolist())
            cost_hamiltonian_i_gauge = cost_hamiltonian.copy().apply_bitflip(ground_state_flipped)
            zero_energy_i = float(cost_hamiltonian_i_gauge.evaluate_energy(bitstrings_array=[[0] * number_of_qubits])[0])
            ar_zero_energy_i = cost_hamiltonian_i_gauge.calculate_approximation_ratio(energy=zero_energy_i)

            if ar_zero_energy_i < min_ar or ar_zero_energy_i > max_ar:
                continue

            if zero_energy_i not in _already_done:
                all_gauges.append(ground_state_flipped)
                _already_done.add(zero_energy_i)

    if len(all_gauges)==0:
        if return_energies:
            return [], []
        return []

    _all_gauges = np.array(all_gauges)
    _all_energies = cost_hamiltonian.evaluate_energy(bitstrings_array=_all_gauges)
    _all_pairs = [(x, y) for x, y in zip(_all_energies, _all_gauges)]
    _all_pairs = sorted(_all_pairs, key=lambda x: x[0])

    _all_energies = [float(x[0]) for x in _all_pairs]
    _all_gauges = [tuple(x[1].tolist()) for x in _all_pairs]

    if return_energies:
        return _all_gauges, _all_energies

    return _all_gauges


def get_solutions_that_are_approximately_uniformly_spread_out_from_ground_state(cost_hamiltonian:ClassicalHamiltonian,
                                                                                 number_of_solutions:int,
                                                                                 runtime_max:float=10,
                                                                                 show_progress_bar:bool=False,
                                                                                min_ar:float = 0.5,
                                                                                max_ar:float=1.0

                                                                                )->List[List[int]]:

    #TODO(FBM): improve this by estimating number of flips to get from 1.0 to ~0.5


    if number_of_solutions == 0:
        return [[0]*cost_hamiltonian.number_of_qubits]

    if cost_hamiltonian.ground_state is None:
        raise ValueError("Cost hamiltonian ground state is not provided.")


    diff_per_step: float = (max_ar - min_ar) / number_of_solutions


    number_of_qubits = cost_hamiltonian.number_of_qubits
    max_flips = number_of_qubits // 2

    ground_state = np.array(cost_hamiltonian.ground_state)

    best_guess_flip_size = None
    closest_diff = np.inf
    for guess_flip_size in range(1, max_flips+1):
        ground_state_flipped = ground_state.copy()
        indices_to_flips = np.random.choice(number_of_qubits, size=guess_flip_size, replace=False)
        ground_state_flipped[indices_to_flips] = 1 - ground_state_flipped[indices_to_flips]
        energy_here = cost_hamiltonian.evaluate_energy(bitstrings_array=[ground_state_flipped])[0]



        ar_here = cost_hamiltonian.calculate_approximation_ratio(energy=energy_here)
        diff_i = abs(1-ar_here - diff_per_step)
        if diff_i < closest_diff:
            closest_diff = diff_i
            best_guess_flip_size = guess_flip_size

    bgfs = best_guess_flip_size
    pbar = tqdm(list(range(1000)), disable=not show_progress_bar)
    best_gauges, best_spread = None, np.inf

    t_start = time.perf_counter()
    stop = False
    while not stop:
        seed = 0
        while not stop:
            for step_size, same_size_flips_max_amount in zip([bgfs,bgfs,bgfs,bgfs+1,bgfs+1,bgfs+1], [1,2,3,1,2,3]):
                _all_gauges, _all_energies = get_solutions_away_from_ground_state(cost_hamiltonian=cost_hamiltonian,
                                                                                  max_flips=max_flips,
                                                                                  step_size=step_size,
                                                                                  same_size_flips_max_amount=same_size_flips_max_amount,
                                                                                  max_solutions=number_of_solutions,
                                                                                  show_progress_bar=False,
                                                                                  seed=seed,
                                                                                  return_energies=True,
                                                                                  min_ar=min_ar,
                                                                                  max_ar=max_ar,)
                seed += 1

                if len(_all_energies) == 0:
                    continue
                _approx_ars = cost_hamiltonian.calculate_approximation_ratio(energy=_all_energies)
                _approx_ars = np.sort(_approx_ars)

                # _spread_i = np.mean([abs(abs(_approx_ars[i]-_approx_ars[i+1])-diff_per_step) for i in range(len(_approx_ars)-1)])
                _spread_i = np.sum([abs(abs(_approx_ars[i]-_approx_ars[i+1])-diff_per_step) for i in range(len(_approx_ars)-1)])
               # _spread_i-= np.std(_approx_ars)


                if _spread_i<best_spread:
                    best_gauges, best_spread = _all_gauges, _spread_i



                if show_progress_bar:
                    pbar.update(1)
                stop = (time.perf_counter()-t_start) > runtime_max

                if number_of_solutions == 1:
                    stop = True
                    best_gauges = _all_gauges

                if stop:
                    break

    return best_gauges


def create_grid_search_plus_scipy_consecutive_optimizer(parameter_bounds_global: List[Tuple[ParametersBoundType, Tuple[Any, ...]]],
                                                         calls_split:Optional[List[float]]=None,
                                                         shrink_factors:Optional[List[float]]=None,
                                                        optimizer_name_scipy:str='cobyqa',
                                                        specific_points_to_include:Optional[List[Tuple[float, ...]|float]]=None,
basinhopping:bool=True,

optimizer_kwargs:Optional[dict]=None,
basinhopping_kwargs:Optional[dict]=None,
                                                        )->ConsecutiveOptimizersRunner:

    if calls_split is None:
        calls_split = [0.9, 0.1]

    if basinhopping_kwargs is None:
        basinhopping_kwargs = {'niter': 5}

    local_optimizer_builders = []
    for precision_level in range(len(calls_split)):
        if precision_level == 0:
            def _local_builder(parameter_bounds,
                               max_trials,
                               search_space):
                return SimpleGridOptimizer(parameter_bounds=parameter_bounds,
                                           max_trials=max_trials,
                                           specific_points_to_include=specific_points_to_include,
                                           search_space=search_space)

            local_optimizer_builders.append((OptimizerType.custom, _local_builder))
        else:
            def _local_builder(parameter_bounds,
                               starting_point,
                               ):

                return ScipyOptimizerWrapped(parameters_bounds=parameter_bounds,
                                             optimizer_name=optimizer_name_scipy,
                                             optimizer_kwargs=optimizer_kwargs,
                                             starting_point=starting_point,
                                             basinhopping=basinhopping,
                                             basinhopping_kwargs=basinhopping_kwargs,
                                             )

            local_optimizer_builders.append((OptimizerType.scipy, _local_builder))

    optimizer_smarter = ConsecutiveOptimizersRunner(global_parameters_bounds=parameter_bounds_global,
                                                    list_of_optimizer_builders=local_optimizer_builders,
                                                    calls_split=calls_split,
                                                    shrink_factors=shrink_factors, )

    return optimizer_smarter


def get_standard_subfolders_hierarchy_main(experiment_category:str,
                                     experiment_subcategory:str):


    experiments_folders_hierarchy = ['Results',
                                     experiment_category,
                                     experiment_subcategory]

    return experiments_folders_hierarchy
def get_standard_subfolders_hierarchy_sub(backend_name:str,
                                         simulation:bool,
                                         noiseless_simulation:bool,
                                         hamiltonian_class_description:str,
                                          just_random_sampling:bool=False):


    if just_random_sampling:
        return ['RandomSampling', f"{hamiltonian_class_description}"]


    experiments_folders_hierarchy = [f"{SNV.Backend.id}{MKVS}{backend_name}",
                                     f"{SNV.Simulated.id}{MKVS}{simulation}{MKS}Noiseless{MKVS}{noiseless_simulation}",
                                     f"{hamiltonian_class_description}"]

    return experiments_folders_hierarchy

def get_standard_subfolders_hierarchy_full(experiment_category:str,
                                     experiment_subcategory:str,
                                     backend_name:str,
                                     simulation:bool,
                                     noiseless_simulation:bool,
                                     hamiltonian_class_description:str,
                                           just_random_sampling:bool=False):



    return (get_standard_subfolders_hierarchy_main(experiment_category=experiment_category,
                                                  experiment_subcategory=experiment_subcategory)
            +get_standard_subfolders_hierarchy_sub(backend_name=backend_name,
                                       simulation=simulation,
                                       noiseless_simulation=noiseless_simulation,
                                       hamiltonian_class_description=hamiltonian_class_description,
                                                   just_random_sampling=just_random_sampling))
