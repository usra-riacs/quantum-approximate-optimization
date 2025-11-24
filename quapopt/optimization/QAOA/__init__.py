# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

# import all types from typing
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

from dataclasses import dataclass, field

import pandas as pd

from quapopt.ancillary_functions import convert_cupy_numpy_array
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling import (
    HamiltonianClassSpecifierGeneral,
    HamiltonianInstanceSpecifierGeneral,
    StandardizedSpecifier,
)
from quapopt.optimization import EnergyResultMain

_ANGLES_BOUNDS_LAYER_PHASE = (-np.pi, np.pi)
_ANGLES_BOUNDS_LAYER_MIXER = (-np.pi, np.pi)


class PhaseSeparatorType(Enum):
    QAOA = "QAOA"
    QAMPA = "QAMPA"


class MixerType(Enum):
    QAOA = "QAOA"
    QAMPA = "QAMPA"

    ws_qaoa_zero_biased = "WSQAOAZeroBiased"
    ws_qaoa_one_biased = "WSQAOAOneBiased"


class InitialStateType(Enum):
    QAOA = "QAOA"

    ws_qaoa_zero_biased = "WSQAOAZeroBiased"
    ws_qaoa_one_biased = "WSQAOAOneBiased"
    zero = "Zero"
    one = "One"


# class InitialStateDescription:


class QubitMappingType(Enum):
    linear_swap_network = "LSN"
    fully_connected = "FC"
    sabre = "SABRE"


# TODO(FBM): we shouldn't allow for that much flexibility, lol
class QAOAFunctionInputFormat(Enum):
    # Arguments are passed as _fun(*args)
    direct_full = "DirectFull"
    # Arguments are passed as _fun(list_of_args)
    direct_list = "DirectList"
    # Arguments are passed as _fun(vector_of_angles, *other_args)
    direct_vector = "DirectVector"
    # Arguments are passed as _fun([vector_gamma, vector_beta], *other_args)
    direct_QAOA = "QAOA"
    # Arguments are passed as _fun(optuna.Trial)
    optuna = "Optuna"


@dataclass
class AnsatzSpecifier(StandardizedSpecifier):
    # Define fields with default values, some computed in __post_init__
    PhaseHamiltonianClass: HamiltonianClassSpecifierGeneral = field(
        default=None, init=True
    )
    PhaseHamiltonianInstance: HamiltonianInstanceSpecifierGeneral = field(
        default=None, init=True
    )
    PhaseSeparatorType: PhaseSeparatorType = field(default=None, init=True)
    MixerType: MixerType = field(default=None, init=True)
    QubitMappingType: QubitMappingType = field(default=None, init=True)
    Depth: int = field(default=None, init=True)
    TimeBlockSize: Optional[int | float] = field(default=None, init=True)

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


#
# @dataclass(frozen=False)
# class QuantumOptimizationSpecifier(HamiltonianOptimizationSpecifier):
#     # Additional field for ansatz specification
#
#     def _get_dataframe_annotation(self, long_names: bool = True) -> dict:
#         """Override to include ansatz specifier in annotations."""
#
#         # Get parent annotations, including description of the Hamiltonian that is optimized
#
#         # Add ansatz specifier annotation
#         if self.AnsatzSpecifier is None:
#
#
#
#
# class QAOAResultsLogger(HamiltonianOptimizationResultsLogger):
#     """Specialized logger for QAOA optimization experiments."""
#
#     def __init__(self,
#                  cost_hamiltonian: ClassicalHamiltonian,
#                  ):
#         """
#         Initialize QAOAResultsLogger.
#
#         This logger automatically sets up QAOA-specific folder hierarchy and
#         creates appropriate experiment specifiers that include both Hamiltonian
#         and ansatz information.
#
#         :param cost_hamiltonian: The Hamiltonian to optimize
#         :param ansatz_specifier: QAOA ansatz specification
#         :param table_name_prefix: Optional prefix for table names
#         :param table_name_suffix: Optional suffix for table names
#         :param experiment_specifier: Optional additional experiment specifier
#         :param experiment_folders_hierarchy: Optional folder hierarchy
#         :param directory_main: Main directory for storing results
#         :param logging_level: Logging verbosity level
#         :param experiment_set_name: Name of the experiment set
#         :param experiment_set_id: ID of the experiment set
#         :param experiment_instance_id: ID of this experiment instance
#         """
#         warnings.warn(
#             "QAOAResultsLogger is deprecated. Use ResultsLogger instead and handle "
#             "metadata in runner/analyzer classes.",
#             DeprecationWarning,
#
#
#         # Create QAOA-specific experiment specifier that includes ansatz information
#
#
#         # Merge with any additional experiment specifier if provided
#         if experiment_specifier is not None:
#
#         # Initialize parent class with updated parameters
#         super().__init__(
#             experiment_instance_id=experiment_instance_id,)
#
#


class QAOAResult:
    def __init__(
        self,
        energy_result: EnergyResultMain = None,
        trial_index: Optional[int] = None,
        angles: np.ndarray = None,
        hamiltonian_representation_index: Optional[int] = None,
        statevector: np.ndarray = None,
        bitstrings_array: Optional[
            Union[np.ndarray, List[Union[Tuple[int, ...], List[int]]]]
        ] = None,
        bitstrings_energies: Optional[np.ndarray | cp.ndarray] = None,
        noise_model=None,
        sort_energies_and_bitstrings=False,
        correlators: np.ndarray = None,
    ):

        self.trial_index = trial_index
        self.angles = angles
        self.hamiltonian_representation_index = hamiltonian_representation_index
        self.correlators = correlators

        self.statevector = statevector

        self.bitstrings_array = bitstrings_array
        self.bitstrings_energies = bitstrings_energies
        self.energies_are_sorted = False

        self.energy_result = energy_result

        self._bck = np

        if self.bitstrings_array is not None and self.bitstrings_energies is not None:
            if isinstance(self.bitstrings_array, np.ndarray):
                _bck_name = "numpy"
            elif isinstance(self.bitstrings_array, cp.ndarray):
                self._bck = cp
                _bck_name = "cupy"
            else:
                raise ValueError(
                    "Unknown type of bitstrings_array:", type(self.bitstrings_array)
                )

            self.bitstrings_energies = convert_cupy_numpy_array(
                array=self.bitstrings_energies, output_backend=_bck_name
            )

        if sort_energies_and_bitstrings:
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

    def annotate_dataframe(self, df: pd.DataFrame):

        main_annotation = pd.DataFrame(
            data={
                f"{SNV.TrialIndex.id_long}": [self.trial_index] * len(df),
                f"{SNV.HamiltonianRepresentationIndex.id_long}": [
                    self.hamiltonian_representation_index
                ]
                * len(df),
            },
        )

        # Now I want to add separate column for each angle:
        if self.angles is None:
            # TODO(FBM): temporary hack, in general the angles should always be provided
            angle_annotation = pd.DataFrame(
                data={f"{SNV.Angles.id_long}": [None] * len(df)}
            )
        else:

            if len(self.angles.shape) == 1:
                angles_list = np.array(self.angles)
            else:
                angles_list = np.array(self.angles.flatten())

            #     data={f"{SNV.Angles.id_long}-{i}": [float(val)] * len(df) for i, val in enumerate(angles_list)})

            angle_annotation = pd.DataFrame(
                data={f"{SNV.Angles.id_long}": [angles_list.tolist()] * len(df)}
            )

        full_annotation = main_annotation.join(angle_annotation)

        return full_annotation.join(df)

    def to_dataframe_main(self):
        return self.annotate_dataframe(self.energy_result.to_dataframe_main())

    def to_dataframe_full(self):
        return self.annotate_dataframe(self.energy_result.to_dataframe_full())

    def sort_energies_and_bitstrings(self):

        if self.energies_are_sorted:
            return

        idx_sort = self._bck.argsort(self.bitstrings_energies)
        self.bitstrings_energies = self.bitstrings_energies[idx_sort]
        self.bitstrings_array = self.bitstrings_array[idx_sort]
        self.energies_are_sorted = True

    def update_main_energy(self, noisy: bool):
        self.energy_result.update_main_energy(noisy)
        self.energy_mean = self.energy_result.energy_mean
        self.energy_best = self.energy_result.energy_best
        self.bitstring_best = self.energy_result.bitstring_best


if __name__ == "__main__":
    print(QubitMappingType.sabre == QubitMappingType.linear_swap_network)
