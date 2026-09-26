# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from enum import Enum, unique
from typing import Union, Tuple, Optional, List, Any

import numpy as np
import pandas as pd

from quapopt.data_analysis.data_handling import (STANDARD_NAMES_VARIABLES as SNV,
                                                 StandardizedSpecifier)
from quapopt.data_analysis.data_handling.schemas.naming import MAIN_KEY_VALUE_SEPARATOR

from dataclasses import dataclass, field

from quapopt.optimization import EnergyResultMain
from quapopt.optimization.QAOA import QAOAResult

try:
    import cupy as cp
except(ModuleNotFoundError,ImportError):
    import numpy as cp


def handle_bitstring_format_inefficient(bitstring: Optional[Tuple[int,...] | List[int] | np.ndarray | cp.ndarray]):
    if bitstring is None:
        return None

    if isinstance(bitstring,(np.ndarray,cp.ndarray)):
        bitstring = bitstring.tolist()

    return tuple([int(x) for x in bitstring])



@unique
class AttractorStateType(Enum):
    zero = "0"
    one = "1"
    adaptive = "Adaptive"
    custom = "Custom"


@unique
class TrialChoiceRule(Enum):
    best_sample_energy = "BestSampleEnergy"
    best_mean_energy = "BestMeanEnergy"
    all = "All"


@unique
class BitstringChoiceRule(Enum):
    frequency = "Frequency"
    hamming_weight = "HammingWeight"
    energy = "Energy"


@unique
class PropertyFunctionRule(Enum):
    mean = "Mean"
    median = "Median"
    variance = "Variance"


@unique
class ConvergenceCriterionNames(Enum):
    # Stop after a certain number of iterations
    MaxIterations = "MI"
    # Stop after a certain number of unsuccessful trials
    MaxUnsuccessfulTrials = "MUT"
    # Stop after the best energy change is below a certain threshold
    BestEnergyChange = "BEC"
    # Stop after the mean energy change is below a certain threshold
    MeanEnergyChange = "MEC"
    # Stop after the median energy change is below a certain threshold
    Mixed = "MIX"


BitstringTypeSignature = Tuple[int, ...] | np.ndarray | List[int]
BestResultSignature = Tuple[Tuple[float, BitstringTypeSignature, int], Optional[Any]]



@dataclass
class ConvergenceCriterion(StandardizedSpecifier):
    ConvergenceCriterion: ConvergenceCriterionNames = field(default=None, init=False)
    ConvergenceValue: Union[int, float] = field(default=None, init=False)
    
    def __init__(self,
                 convergence_criterion_name: ConvergenceCriterionNames,
                 convergence_value: Union[int, float] = None):
        
        # Apply default values based on criterion type
        if convergence_value is None:
            if convergence_criterion_name == ConvergenceCriterionNames.MaxIterations:
                convergence_value = 10
            elif convergence_criterion_name == ConvergenceCriterionNames.MaxUnsuccessfulTrials:
                convergence_value = 3
            elif convergence_criterion_name == ConvergenceCriterionNames.BestEnergyChange:
                convergence_value = 10 ** -6
            elif convergence_criterion_name == ConvergenceCriterionNames.MeanEnergyChange:
                convergence_value = 10 ** -6
            else:
                raise ValueError(f"Value needs to be provided for: {convergence_criterion_name}")
        
        # Use object.__setattr__ for frozen dataclass initialization
        object.__setattr__(self, 'ConvergenceCriterion', convergence_criterion_name)
        object.__setattr__(self, 'ConvergenceValue', convergence_value)
        
        # Call parent init
        super().__init__()

    def check_convergence(self,
                          previous_score=None,
                          current_score=None,
                          iteration_index=None):

        if iteration_index == 0:
            return False

        convergence_criterion = getattr(self, SNV.ConvergenceCriterion.id_long)

        if (convergence_criterion == ConvergenceCriterionNames.MaxIterations
                or convergence_criterion == ConvergenceCriterionNames.MaxUnsuccessfulTrials):
            # if it's MaxTrials then iteration_index is number of unsucesfull trials
            # if it's Maxiterations then iteration_index is the number of iterations
            maximal_iterations = self.ConvergenceValue
            if iteration_index >= maximal_iterations:
                return True
            else:
                return False
        elif (convergence_criterion == ConvergenceCriterionNames.BestEnergyChange
              or convergence_criterion == ConvergenceCriterionNames.MeanEnergyChange):
            if abs(previous_score - current_score) < self.ConvergenceValue:
                return True
            elif previous_score < current_score:
                return True

            return False
        else:
            raise NotImplementedError(f"Convergence criterion: {convergence_criterion} is not implemented")

    def get_description_string(self):
        return super().get_description_string(major_separator=False)


@dataclass
class AttractorModel(StandardizedSpecifier):
    NumberOfQubits: int = field(default=None, init=False)
    AttractorStateType: AttractorStateType = field(default=None, init=False)
    AttractorState: Tuple[int, ...] = field(default=None, init=False)
    
    def __init__(self,
                 number_of_qubits: int,
                 attractor_state_type: AttractorStateType = None,
                 attractor_state: Tuple[int, ...] = None):
        
        # Infer attractor_state_type from attractor_state if provided
        if attractor_state is not None:
            unique_values = list(set(attractor_state))
            if len(unique_values) == 1:
                if unique_values[0] == 0:
                    attractor_state_type = AttractorStateType.zero
                elif unique_values[0] == 1:
                    attractor_state_type = AttractorStateType.one
                else:
                    raise ValueError(f"Invalid attractor state: {attractor_state}")
            else:
                attractor_state_type = AttractorStateType.custom
        
        # Validate custom state
        if attractor_state_type == AttractorStateType.custom and attractor_state is None:
            raise ValueError(f"Invalid attractor state: {attractor_state}")
        
        # Generate attractor_state based on type if not provided
        if attractor_state is None:
            if attractor_state_type == AttractorStateType.zero:
                attractor_state = tuple([0] * number_of_qubits)
            elif attractor_state_type == AttractorStateType.one:
                attractor_state = tuple([1] * number_of_qubits)
            elif attractor_state_type == AttractorStateType.adaptive:
                attractor_state = tuple([0] * number_of_qubits)
            elif attractor_state_type is None:
                # Default to zero state
                attractor_state_type = AttractorStateType.zero
                attractor_state = tuple([0] * number_of_qubits)
            else:
                raise ValueError(f"You have to provide attractor state for {attractor_state_type}")
        
        # Use object.__setattr__ for frozen dataclass initialization
        object.__setattr__(self, 'NumberOfQubits', number_of_qubits)
        object.__setattr__(self, 'AttractorStateType', attractor_state_type)
        object.__setattr__(self, 'AttractorState', attractor_state)
        
        # Call parent init
        super().__init__()
    
    @property
    def attractor_state(self):
        return self.AttractorState
    
    def with_new_attractor_state(self, attractor_state: Tuple[int, ...]) -> 'AttractorModel':
        """Create a new AttractorModel with updated attractor state (for adaptive mode)."""
        # Determine the type based on the new state
        unique_values = list(set(attractor_state))
        if len(unique_values) == 1:
            if unique_values[0] == 0:
                new_type = AttractorStateType.zero
            elif unique_values[0] == 1:
                new_type = AttractorStateType.one
            else:
                new_type = AttractorStateType.custom
        else:
            new_type = AttractorStateType.custom
        
        return AttractorModel(
            number_of_qubits=self.NumberOfQubits,
            attractor_state_type=new_type,
            attractor_state=attractor_state
        )

    def return_bitflip_transformation(self,
                                      bitstring: Union[Tuple[int, ...],np.ndarray]):

        if self.AttractorStateType == AttractorStateType.zero:
            return tuple([bit for bit in bitstring])
        elif self.AttractorStateType == AttractorStateType.one:
            return tuple([1 - bit for bit in bitstring])
        elif self.AttractorStateType in [AttractorStateType.adaptive, AttractorStateType.custom]:
            attractor_state = self.AttractorState
            # I wish to map the attractor state to the bitstring, so the transformation has to be addition modulo 2
            return tuple([0 if attractor_state[i] == bitstring[i] else 1 for i in range(self.NumberOfQubits)])
        else:
            raise NotImplementedError(
                f"Transformation for attractor state: {self.AttractorStateType} is not implemented")

    def get_description_string(self):
        return f"{SNV.AttractorState.id}{MAIN_KEY_VALUE_SEPARATOR}{self.AttractorStateType.value}"


class NDARIterationResult:
    def __init__(self,
                 iteration_index: int,
                 best_result:BestResultSignature,
                 bitflip_transform: BitstringTypeSignature,
                 local_sampler_specific_data: Optional[QAOAResult | EnergyResultMain] = None,
                 attractor_model: AttractorModel = None,
                 convergence_criterion: ConvergenceCriterion = None,
                 local_sampler_name:Optional[str]=None
                 ):

        self.iteration_index = iteration_index
        self.best_result = best_result
        self.local_sampler_specific_data = local_sampler_specific_data
        self.attractor_model = attractor_model
        self.convergence_criterion = convergence_criterion
        self.bitflip_transform =  handle_bitstring_format_inefficient(bitstring=bitflip_transform)
        self.bitflip_hamming_weight = int(sum(self.bitflip_transform))

        if local_sampler_name is None:
            local_sampler_name = "UnnamedSampler"
        self.local_sampler_name = local_sampler_name


    @property
    def best_energy(self) -> float:
        return self.best_result[0][0]
    @property
    def best_bitstring(self) -> BitstringTypeSignature:
        return self.best_result[0][1]
    @property
    def best_hamiltonian_representation_index(self) -> int:
        return self.best_result[0][2]

    @property
    def bitflip(self):
        return self.bitflip_transform

    def _try_to_get_stuff(self):

        c_param = None
        trial_index = None
        angles = None
        energy_mean = None

        if isinstance(self.best_result,QAOAResult):
            c_param = float(self.best_result.bias_parameters_WS)
            trial_index = self.best_result.trial_index
            angles = self.best_result.angles
            energy_mean = self.best_result.energy_mean

        elif isinstance(self.best_result, tuple):
            if isinstance(self.best_result[1], pd.DataFrame):
                if SNV.WSBiasParameters.id_long in self.best_result[1].columns:
                    c_param = float(self.best_result[1][SNV.WSBiasParameters.id_long].iloc[0])
                elif SNV.FlipProbability.id_long in self.best_result[1].columns:
                    c_param = float(self.best_result[1][SNV.FlipProbability.id_long].iloc[0])

                if SNV.TrialIndex.id_long in self.best_result[1].columns:
                    trial_index = int(self.best_result[1][SNV.TrialIndex.id_long].iloc[0])
                if SNV.Angles.id_long in self.best_result[1].columns:
                    angles = self.best_result[1][SNV.Angles.id_long].iloc[0]
                if SNV.EnergyMean.id_long in self.best_result[1].columns:
                    energy_mean = float(self.best_result[1][SNV.EnergyMean.id_long].iloc[0])




        return c_param, trial_index, angles, energy_mean

    def __repr__(self):
        main_str ="Single-Iteration NDAR result with the following data:\n"
        main_str += f"Iteration index: {self.iteration_index}\n"
        main_str += f"Best energy: {self.best_energy}\n"

        c_param, trial_index, angles, energy_mean = self._try_to_get_stuff()

        if energy_mean is not None:
            main_str += f"Best energy mean: {energy_mean}\n"

        if trial_index is not None:
            main_str += f"Best trial index: {trial_index}\n"

        main_str += f"Best hamiltonian representation index: {self.best_hamiltonian_representation_index}\n"

        if angles is not None:
            main_str += f"Best angles: {angles}\n"

        if c_param is not None:
            main_str += f"Best bias parameters: {c_param}\n"

        main_str += f"Bitflip hamming weight: {self.bitflip_hamming_weight}\n"


        main_str += 'Obtained with local sampler: ' + self.local_sampler_name + '\n'

        return main_str



    def annotate_dataframe(self,
                           df: pd.DataFrame):

        if self.attractor_model is not None:
            attractor_string = self.attractor_model.get_description_string()
        else:
            attractor_string = 'None'

        bitflip_transform = self.bitflip_transform
        if isinstance(bitflip_transform,np.ndarray):
            bitflip_transform = bitflip_transform.tolist()

        if self.convergence_criterion is not None:
            convergence_string = self.convergence_criterion.get_description_string()
        else:
            convergence_string = 'None'


        local_sampler_name = str(self.local_sampler_name)

        main_annotation = pd.DataFrame(data={f"{SNV.AttractorModel.id_long}": [attractor_string] * len(df),
                                             f"{SNV.ConvergenceCriterion.id_long}": [convergence_string] * len(df),
                                             f"{SNV.SamplerName.id_long}": [local_sampler_name],
                                             f"{SNV.NDARIteration.id_long}": [self.iteration_index] * len(df),
                                             f"{SNV.BitflipHammingWeight.id_long}": [self.bitflip_hamming_weight]*len(df),
                                             f"{SNV.Bitflip.id_long}": [bitflip_transform] * len(df),
                                             })

        return main_annotation.join(df)

    def to_dataframe_main(self) -> pd.DataFrame:

        (energy_best, bitstring_best, hamiltonian_representation_index_best), _ = self.best_result

        if isinstance(bitstring_best,(np.ndarray,cp.ndarray)):
            bitstring_best_df = bitstring_best.tolist()

        else:
            bitstring_best_df = bitstring_best


        c_param, trial_index, angles, energy_mean = self._try_to_get_stuff()


        df_best = pd.DataFrame(data={
                                    f"{SNV.TrialIndex.id_long}": [trial_index],
                                    f"{SNV.HamiltonianRepresentationIndex.id_long}": [hamiltonian_representation_index_best],
                                    f"{SNV.Angles.id_long}": [angles],
                                    f"{SNV.WSBiasParameters.id_long}":[c_param],
                                    f"{SNV.EnergyBest.id_long}": [energy_best],
                                    f"{SNV.EnergyMean.id_long}": [energy_mean],
                                     f"{SNV.BitstringBest.id_long}": [handle_bitstring_format_inefficient(bitstring=bitstring_best_df)],
                                     })

        df_best = self.annotate_dataframe(df_best)

        nice_order = [
            f"{SNV.NDARIteration.id_long}",
            f"{SNV.EnergyBest.id_long}",
            f"{SNV.BitflipHammingWeight.id_long}",
            f"{SNV.WSBiasParameters.id_long}",
            f"{SNV.EnergyMean.id_long}",
            f"{SNV.SamplerName.id_long}",
            f"{SNV.HamiltonianRepresentationIndex.id_long}",
            f"{SNV.TrialIndex.id_long}",
            f"{SNV.Angles.id_long}",
            f"{SNV.AttractorModel.id_long}",
            f"{SNV.ConvergenceCriterion.id_long}",
            f"{SNV.BitstringBest.id_long}",
            f"{SNV.Bitflip.id_long}"
                      ]

        return df_best[nice_order]




#
# class AdaptiveAttractorModel(AttractorModel):
#     def __init__(self,
#                  trial_choice_rule: TrialChoiceRule,
#                  bitstring_choice_rule: BitstringChoiceRule,
#                  property_function_rule: PropertyFunctionRule):
#         names = [SNDAR.attractor_state_type,
#                  SNDAR.trial_choice_rule,
#                  SNDAR.bitstring_choice_rule,
#                  SNDAR.property_function_rule]
#         values = [AttractorStateType.adaptive,
#                   trial_choice_rule,
#                   bitstring_choice_rule,
#                   property_function_rule]
#
#         super().__init__(names=names, values=values)
#
#     def update_attractor_state(self, new_attractor_state: Tuple[int]):
#         setattr(self, SNDAR.attractor_state, new_attractor_state)
#
#
