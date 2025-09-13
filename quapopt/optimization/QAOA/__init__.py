# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

# import all types from typing
from enum import Enum
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
#Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
    import numpy as cp

import pandas as pd
from pathlib import Path

from quapopt import ancillary_functions as anf

from quapopt.data_analysis.data_handling import (StandardizedSpecifier,
                                                 HamiltonianClassSpecifierGeneral,
                                                 HamiltonianInstanceSpecifierGeneral,
                                                 ResultsLogger,
                                                 STANDARD_NAMES_VARIABLES as SNV,
HamiltonianOptimizationSpecifier
                                                 )
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization import (EnergyResultMain,
                                   HamiltonianOptimizationResultsLogger)


from quapopt.data_analysis.data_handling import LoggingLevel

from dataclasses import dataclass, field, fields


_ANGLES_BOUNDS_LAYER_PHASE = (-np.pi, np.pi)
_ANGLES_BOUNDS_LAYER_MIXER = (-np.pi, np.pi)


class PhaseSeparatorType(Enum):
    QAOA = 'QAOA'
    QAMPA = 'QAMPA'

class MixerType(Enum):
    QAOA = 'QAOA'
    QAMPA = 'QAMPA'

class QubitMappingType(Enum):
    linear_swap_network = 'LSN'
    fully_connected = 'FC'
    sabre = 'SABRE'

#TODO(FBM): we shouldn't allow for that much flexibility, lol
class QAOAFunctionInputFormat(Enum):
    # Arguments are passed as _fun(*args)
    direct_full = 'DirectFull'
    # Arguments are passed as _fun(list_of_args)
    direct_list = 'DirectList'
    # Arguments are passed as _fun(vector_of_angles, *other_args)
    direct_vector = 'DirectVector'
    # Arguments are passed as _fun([vector_gamma, vector_beta], *other_args)
    direct_QAOA = 'QAOA'
    # Arguments are passed as _fun(optuna.Trial)
    optuna = 'Optuna'

@dataclass
class AnsatzSpecifier(StandardizedSpecifier):
    # Define fields with default values, some computed in __post_init__
    PhaseHamiltonianClass: HamiltonianClassSpecifierGeneral = field(default=None, init=True)
    PhaseHamiltonianInstance: HamiltonianInstanceSpecifierGeneral = field(default=None, init=True)
    PhaseSeparatorType: PhaseSeparatorType = field(default=None, init=True)
    MixerType: MixerType = field(default=None, init=True)
    QubitMappingType: QubitMappingType = field(default=None, init=True)
    Depth: int = field(default=None, init=True)
    TimeBlockSize: Optional[int|float] = field(default=None, init=True)

    def __post_init__(self):
        # Set defaults
        if self.PhaseSeparatorType is None:
            self.PhaseSeparatorType = PhaseSeparatorType.QAOA
        if self.MixerType is None:
            self.MixerType = MixerType.QAOA
        if self.QubitMappingType is None:
            self.QubitMappingType = QubitMappingType.linear_swap_network
        if self.Depth is None:
            self.Depth = 1


        if self.TimeBlockSize is None:
            if self.QubitMappingType == QubitMappingType.linear_swap_network:
                if self.PhaseHamiltonianInstance is None:
                    self.TimeBlockSize = None
                else:
                    self.TimeBlockSize = self.PhaseHamiltonianInstance.NumberOfQubits
            else:
                self.TimeBlockSize = 1.0


    @property
    def AlgorithmicDepth(self):
        return self.Depth




class QAOAResult:
    def __init__(self,
                 energy_result: EnergyResultMain = None,
                 trial_index: Optional[int] = None,
                 angles: np.ndarray = None,
                 hamiltonian_representation_index: Optional[int] = None,
                 statevector: np.ndarray = None,
                 bitstrings_array: Optional[Union[np.ndarray, List[Union[Tuple[int, ...], List[int]]]]] = None,
                 bitstrings_energies: Optional[np.ndarray[float]] = None,
                 noise_model=None,
                 sort_energies_and_bitstrings=False,
                 correlators: np.ndarray = None
                 ):
        self.trial_index = trial_index
        self.angles = angles
        self.hamiltonian_representation_index = hamiltonian_representation_index
        self.correlators = correlators

        self.statevector = statevector



        self.bitstrings_array = bitstrings_array
        self.bitstrings_energies = bitstrings_energies

        self.energy_result = energy_result
        if self.bitstrings_array is not None and self.bitstrings_energies is not None and sort_energies_and_bitstrings:
            self.sort_energies_and_bitstrings()

        if self.energy_result is not None:
            self.energy_mean = self.energy_result.energy_mean
            self.energy_best = self.energy_result.energy_best
            self.bitstring_best = self.energy_result.bitstring_best
        else:
            self.energy_mean = None
            self.energy_best = None
            self.bitstring_best = None

        self.noise_model = noise_model

    def annotate_dataframe(self,
                           df: pd.DataFrame):

        main_annotation = pd.DataFrame(data={f"{SNV.TrialIndex.id_long}": [self.trial_index]*len(df),
                                             f"{SNV.HamiltonianRepresentationIndex.id_long}": [
                                                 self.hamiltonian_representation_index]*len(df)},
                                       )


        # Now I want to add separate column for each angle:
        if self.angles is None:
            #TODO(FBM): temporary hack, in general the angles should always be provided
            angle_annotation = pd.DataFrame(data={f"{SNV.Angles.id_long}-{0}": [None]*len(df)})
        else:

            if len(self.angles.shape) == 1:
                angles_list = self.angles
            else:
                angles_list = self.angles.flatten()


            angle_annotation = pd.DataFrame(data={f"{SNV.Angles.id_long}-{i}": [float(val)]*len(df) for i, val in enumerate(angles_list)})


        full_annotation = main_annotation.join(angle_annotation)

        return full_annotation.join(df)

    def to_dataframe_main(self):
        return self.annotate_dataframe(self.energy_result.to_dataframe_main())

    def to_dataframe_full(self):
        return self.annotate_dataframe(self.energy_result.to_dataframe_full())

    def sort_energies_and_bitstrings(self):
        sorted_pairs = sorted(zip(self.bitstrings_energies, self.bitstrings_array), key=lambda x: x[0])
        self.bitstrings_energies, self.bitstrings_array = zip(*sorted_pairs)

        

    def update_main_energy(self,
                           noisy: bool):
        self.energy_result.update_main_energy(noisy)
        self.energy_mean = self.energy_result.energy_mean
        self.energy_best = self.energy_result.energy_best
        self.bitstring_best = self.energy_result.bitstring_best


if __name__ == '__main__':

    print(QubitMappingType.sabre == QubitMappingType.linear_swap_network)
