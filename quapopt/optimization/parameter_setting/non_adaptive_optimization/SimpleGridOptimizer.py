# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import itertools
from typing import List, Tuple, Any, Optional,Set

import numpy as np

from quapopt.optimization.parameter_setting.non_adaptive_optimization.NonAdaptiveOptimizer import NonAdaptiveOptimizer
from quapopt.optimization.parameter_setting import ParametersBoundType



def create_approximately_uniform_search_space_including_a_point(min_value:float,
                                                                  max_value:float,
                                                                  point_of_interest:float,
                                                                  number_of_points:int,
                                                                  )->List[float]:

    if number_of_points == 1:
        return [point_of_interest]

    if min_value == max_value:
        return [point_of_interest]*number_of_points

    if number_of_points == 2:
        if point_of_interest!=(min_value+max_value)/2.0:
            _add_point = (min_value+max_value)/2.0
        elif point_of_interest!=min_value:
            _add_point = min_value
        else:
            _add_point = max_value
        main_space  = np.array([point_of_interest, _add_point])
    elif number_of_points == 3:
        _points = [point_of_interest]
        if point_of_interest!=min_value:
            _points.append(min_value)
        if point_of_interest!=max_value:
            _points.append(max_value)

        main_space = np.array(_points)
    else:

        main_space = np.round(np.linspace(start=min_value,
                                           stop=max_value,
                                           num=number_of_points), 15)




    if point_of_interest in main_space:
        main_space = main_space.tolist()
        # the main point should still be at the beginning of the list
        main_space_pos = main_space.index(point_of_interest)
        main_space[0], main_space[main_space_pos] = main_space[main_space_pos], main_space[0]

        return main_space




    #otherwise, we need to think about this.

    _middle_region = (max_value + min_value) / 2.0
    _points_to_the_left = len(main_space[main_space <= _middle_region])

    if point_of_interest<min_value or point_of_interest>max_value:
        # in this case, we want to take some point that is between min_value and the middle of the region
        _point_index = _points_to_the_left // 2

    else:
        #in this case, point of interest is BETWEEN the min_value and max_value
        _points_to_the_right = len(main_space[main_space > point_of_interest])

        if _points_to_the_left>=_points_to_the_right:
            #we put it on the left
            _point_index = _points_to_the_left // 2

        else:
            _point_index = _points_to_the_left - 1 + _points_to_the_right // 2

    main_space[_point_index] = point_of_interest
    main_space = np.sort(main_space).tolist()
    #the main point should still be at the beginning of the list
    main_space_pos = main_space.index(point_of_interest)
    main_space[0], main_space[main_space_pos] = main_space[main_space_pos], main_space[0]


    return main_space




def _create_uniform_search_space_continuous(parameter_specs:List[Tuple[float,float]|Tuple[int,int]],
                                            max_number_of_points:Optional[int]=None,
                                            local_search_spaces_sizes: Optional[List[int]] = None,
                                            specific_points_to_include:Optional[Tuple[float,...]|float]=None,
                                            ):

    if max_number_of_points is None and local_search_spaces_sizes is None:
        raise ValueError("Either max_number_of_points or local_search_spaces_sizes must be defined.")

    if specific_points_to_include is not None:
        if isinstance(specific_points_to_include, float):
            specific_points_to_include = tuple([specific_points_to_include]*(len(parameter_specs)))

    if local_search_spaces_sizes is None:
        number_of_continuous_parameters = len(parameter_specs)
        size_each_continuous_grid = int(max_number_of_points ** (1 / number_of_continuous_parameters))

        local_search_spaces_sizes = [size_each_continuous_grid for _ in range(number_of_continuous_parameters)]
    else:
        if isinstance(local_search_spaces_sizes,int):
            local_search_spaces_sizes = [local_search_spaces_sizes for _ in range(len(parameter_specs))]

        assert len(local_search_spaces_sizes)==len(parameter_specs), "Number of local search spaces sizes does not match number of parameters."


    if specific_points_to_include is None:
        return [np.linspace(start=min_value,
                            stop=max_value,
                            num=size_param).tolist() for (min_value, max_value), size_param in zip(parameter_specs,local_search_spaces_sizes)]

    return [create_approximately_uniform_search_space_including_a_point(min_value=min_value,
                                                                        max_value=max_value,
                                                                         point_of_interest=point,
                                                                         number_of_points=size_param) for (min_value, max_value), point, size_param in zip(parameter_specs,
                                                                                                                                                           specific_points_to_include,
                                                                                                                                                           local_search_spaces_sizes)]



def _create_uniform_search_spaces_categorical(parameter_specs:List[List[Any]],
                                              local_search_spaces_sizes:Optional[int|List[int]]=None):

    local_search_space = parameter_specs.copy()
    if local_search_spaces_sizes is not None:
        if isinstance(local_search_spaces_sizes,int):
            local_search_spaces_sizes = [local_search_spaces_sizes]*len(parameter_specs)
        local_search_space = [spec[0:size_i] for spec, size_i in zip(parameter_specs,local_search_spaces_sizes)]

    return local_search_space

def _create_uniform_search_spaces_constant(parameter_specs: List[float|int]):

    local_search_spaces_list = []

    for bound_specs in parameter_specs:
        if isinstance(bound_specs, list) or isinstance(bound_specs, tuple):
            assert len(bound_specs) == 1, "Fixed parameter should have only one value."
            bound_specs = bound_specs[0]
        local_search_spaces_list.append([bound_specs])

    return local_search_spaces_list




def create_uniform_search_space(parameter_bounds:List[Tuple[ParametersBoundType,Any]],
                                max_trials:Optional[int]=None,
                                specific_points_to_include:Optional[Tuple[float,...]]=None,
                                local_search_spaces_sizes: Optional[List[int]|int] = None,
                                ):

    if max_trials is None and local_search_spaces_sizes is None:
        raise ValueError("Either max_trials or local_search_spaces_sizes must be defined.")


    enumerated_parameter_bounds = dict(enumerate(parameter_bounds))



    if local_search_spaces_sizes is None:
        number_of_parameters = len(parameter_bounds)
        non_fixed_parameters = [x for x in parameter_bounds if x[0] != ParametersBoundType.CONSTANT]
        categorical_parameters = [x[1] for x in non_fixed_parameters if x[0] == ParametersBoundType.SET]

        total_mult_factor_categorical = None
        if len(categorical_parameters)!=0:
            total_mult_factor_categorical = int(np.prod([len(x) for x in categorical_parameters]))
            assert total_mult_factor_categorical <= max_trials, "Number of trials is too small for minimal grid size."

        continuous_parameters = [x for x in non_fixed_parameters if x[0] == ParametersBoundType.RANGE]
        number_of_continuous_parameters = len(continuous_parameters)


        trials_left_for_continuous = max_trials
        if total_mult_factor_categorical is not None:
            trials_left_for_continuous = trials_left_for_continuous // total_mult_factor_categorical

        size_each_continuous_grid = int(trials_left_for_continuous ** (1 / number_of_continuous_parameters))

        local_search_spaces_sizes = []

        for parameter_index, (bound_type, bound_specs) in enumerated_parameter_bounds.items():
            if bound_type == ParametersBoundType.CONSTANT:
                local_search_spaces_sizes.append(1)
            elif bound_type == ParametersBoundType.SET:
                local_search_spaces_sizes.append(len(bound_specs))
            elif bound_type == ParametersBoundType.RANGE:
                local_search_spaces_sizes.append(size_each_continuous_grid)
            else:
                raise ValueError("Unknown bound type.")



    number_of_parameters = len(local_search_spaces_sizes)

    parameter_indices_by_type = {ParametersBoundType.CONSTANT: [],
                                 ParametersBoundType.RANGE:  [],
                                 ParametersBoundType.SET:  []}
    for parameter_index, (bound_type, bound_specs) in enumerated_parameter_bounds.items():
        parameter_indices_by_type[bound_type].append(parameter_index)

    local_search_spaces_dict = {}

    for bound_type in [ParametersBoundType.CONSTANT, ParametersBoundType.SET, ParametersBoundType.RANGE]:

        parameter_indices_type = parameter_indices_by_type[bound_type]
        if len(parameter_indices_type)==0:
            continue


        parameter_specs_type = [parameter_bounds[i][1] for i in parameter_indices_type]

        local_search_spaces_sizes_bound = [local_search_spaces_sizes[i] for i in parameter_indices_type]


        if bound_type == ParametersBoundType.CONSTANT:
            local_search_spaces_list = _create_uniform_search_spaces_constant(parameter_specs=parameter_specs_type)

        elif bound_type == ParametersBoundType.SET:
            local_search_spaces_list = _create_uniform_search_spaces_categorical(parameter_specs=parameter_specs_type,
                                                                                 local_search_spaces_sizes=local_search_spaces_sizes_bound)

        elif bound_type == ParametersBoundType.RANGE:
            local_search_spaces_list = _create_uniform_search_space_continuous(parameter_specs=parameter_specs_type,
                                                                                specific_points_to_include=specific_points_to_include,
                                                                               local_search_spaces_sizes=local_search_spaces_sizes_bound
                                                                            )

        else:
            raise ValueError('Unknown bound type.')


        for parameter_index, local_search_space in zip(parameter_indices_type, local_search_spaces_list):
            local_search_spaces_dict[parameter_index] = local_search_space


    local_search_spaces = [local_search_spaces_dict[i] for i in range(number_of_parameters)]

    search_space = list(itertools.product(*local_search_spaces))


    return search_space, local_search_spaces_sizes



class SimpleGridOptimizer(NonAdaptiveOptimizer):
    def __init__(self,
                 parameter_bounds: List[Tuple[ParametersBoundType, Tuple[Any, ...]]],
                 max_trials: Optional[int]=None,
                 argument_names: List[str] = None,
                 specific_points_to_include:Optional[Tuple[float,...]|float]=None,
                 search_space:Optional[list]=None,
                 local_search_spaces_sizes:Optional[List[int]]=None,
                 ):

        #print("hejunia2", search_space, max_trials, local_search_spaces_sizes)
        if search_space is None and (max_trials is not None or local_search_spaces_sizes is not None):
            search_space, local_search_spaces_sizes = create_uniform_search_space(parameter_bounds=parameter_bounds,
                                                       max_trials=max_trials,
                                                       specific_points_to_include=specific_points_to_include,
                                                       local_search_spaces_sizes=local_search_spaces_sizes)

        if search_space is not None:

            if max_trials is not None:
                assert len(search_space) <= max_trials, "Number of trials is too small for minimal grid size."
            if argument_names is None:
                argument_names = ['ARG-{}'.format(i) for i in range(len(search_space[0]))]

        super().__init__(search_space=search_space,
                         argument_names=argument_names,
                         optimizer_name="GridOptimizer",
                         parameter_bounds=parameter_bounds,
                         specific_points_to_include=specific_points_to_include,
                         local_search_spaces_sizes=local_search_spaces_sizes)

    def rebuild_instance_with_the_same_search_space(self,
                                max_trials:Optional[int]=None,
                                search_space: Optional[list] = None,
                                local_search_spaces_sizes: Optional[List[int]] = None)->'SimpleGridOptimizer':

        if search_space is None:
            if self._search_space is not None:
                if len(self._search_space)<=max_trials:
                    if local_search_spaces_sizes is None:
                        return self
                    elif local_search_spaces_sizes == self._local_search_spaces_sizes:
                        return self
            assert max_trials is not None or local_search_spaces_sizes is not None, ("Either max_trials or "
                                                                                     "local_search_spaces_sizes"
                                                                                     " must be defined if no search "
                                                                                       "space is provided")
        return SimpleGridOptimizer(parameter_bounds=self._parameter_bounds,
                                   max_trials=max_trials,
                                   argument_names=self._argument_names,
                                   search_space=search_space,
                                   specific_points_to_include=self._specific_points_to_include,
                                   local_search_spaces_sizes=local_search_spaces_sizes)


    def rebuild_instance_with_new_search_space(self,
                                            max_trials:Optional[int]=None,
                                            search_space: Optional[list] = None,
                                            local_search_spaces_sizes: Optional[List[int]] = None)->'SimpleGridOptimizer':

        return SimpleGridOptimizer(parameter_bounds=self._parameter_bounds,
                                   max_trials=max_trials,
                                   argument_names=self._argument_names,
                                   search_space=search_space,
                                   specific_points_to_include=self._specific_points_to_include,
                                   local_search_spaces_sizes=local_search_spaces_sizes)






    @staticmethod
    def create_search_space_around_point(bound_type:ParametersBoundType,
                                         center_point:float|int,
                                         number_of_points:int,
                                         radius:float,
                                         bound_range_global: Optional[Tuple[float, float]]=None,
                                         ):

        if bound_type == ParametersBoundType.RANGE:
            if bound_range_global is None:
                start = center_point - radius
                stop = center_point + radius
            else:
                global_min, global_max = bound_range_global
                start = max(center_point - radius, global_min)
                stop = min(center_point + radius, global_max)

            search_space = np.linspace(start=start,
                                       stop=stop,
                                       num=number_of_points,
                                       endpoint=True).tolist()

            return search_space


        elif bound_type == ParametersBoundType.CONSTANT:
            return [center_point]
        else:
            raise NotImplementedError('This method only works for RANGE and CONSTANT bound types.')





    def run_optimization(self,
                         objective_function: callable,
                         number_of_function_calls: int = None,
                         verbosity: int = 0,
                         show_progress_bar: bool = False,
                         #dummy variable to satisfy interface
                         optimizer_seed: int = None
                         ):


        return super()._run_optimization(objective_function=objective_function,
                                         number_of_function_calls=number_of_function_calls,
                                         verbosity=verbosity,
                                         show_progress_bar=show_progress_bar)
