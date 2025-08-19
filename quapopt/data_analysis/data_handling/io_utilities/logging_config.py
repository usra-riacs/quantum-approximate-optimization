# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

"""
Configuration dataclasses for the logging system.
Provides immutable, type-safe configuration objects for different logger types.
"""

import datetime
import enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.data_analysis.data_handling.standard_names.data_hierarchy import (
    StandardizedSpecifier,
    HamiltonianOptimizationSpecifier,
    MAIN_KEY_SEPARATOR as MKS,
    DEFAULT_TABLE_NAME_PARTS_SEPARATOR,
    DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR
)
from quapopt.data_analysis.data_handling.io_utilities import DEFAULT_STORAGE_DIRECTORY
from quapopt.data_analysis.data_handling.standard_names import (STANDARD_NAMES_DATA_TYPES as SNDT,
                                                                STANDARD_NAMES_VARIABLES as SNV,
                                                                )
from quapopt.data_analysis.data_handling.io_utilities import IOMixin

class LoggingLevel(enum.Enum):
    """Enumeration for different logging verbosity levels."""
    NONE = 0
    # Just expected values etc.
    MINIMAL = 1
    # Expected values, energies, etc.
    BASIC = 2
    # Expected values, energies, bitstrings etc.
    DETAILED = 3
    # Expected values, energies, bitstrings, for multiple levels of optimization (e.g., both noiseless and noisy).
    VERY_DETAILED = 4


@dataclass(frozen=True)
class LoggerConfig:
    """
    Base configuration dataclass for all logger types.
    Provides immutable, type-safe configuration.

    The full main path for storing results is:
    <default_storage_directory>/<base_path>/
    """
    logging_level: LoggingLevel = LoggingLevel.BASIC
    table_name_prefix: Optional[str] = None
    table_name_suffix: Optional[str] = None
    directory_main: Optional[Path] = None

    default_storage_directory: Optional[Path] = field(default_factory=lambda: Path(DEFAULT_STORAGE_DIRECTORY))
    table_name_parts_separator: str = DEFAULT_TABLE_NAME_PARTS_SEPARATOR
    dataframe_type_name_separator: str = DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR

    def __post_init__(self):
        """Auto-generate IDs if not provided."""
        if self.directory_main is None:
            object.__setattr__(self, 'base_path', Path(""))


@dataclass(frozen=True)
class ExperimentLoggerConfig(LoggerConfig):
    """
    Configuration for experiment-based logging with standardized specifiers.
    """
    experiment_specifier: Optional[StandardizedSpecifier] = field(default=StandardizedSpecifier(), init=True)
    experiment_folders_hierarchy: List[str] = field(default_factory=list)

    experiment_set_name: Optional[str] = None
    experiment_set_id: Optional[str] = None

    experiment_instance_id: Optional[str] = None

    def __post_init__(self):
        if self.experiment_set_name is None:
            object.__setattr__(self, 'experiment_set_name', f"ExpSet{anf.get_current_date()}")
        if self.experiment_set_id is None:
            object.__setattr__(self, 'experiment_set_id', anf.create_random_uuid())
        if self.experiment_instance_id is None:
            object.__setattr__(self, 'experiment_instance_id', anf.create_random_uuid())








@dataclass(frozen=True)
class HamiltonianOptimizationLoggerConfig(ExperimentLoggerConfig):
    """
    Configuration for Hamiltonian-specific logging.
    User passes cost_hamiltonian for convenience, but only specifiers are stored.
    """
    CostHamiltonianClass: Any = field(default=None, init=False)
    CostHamiltonianInstance: Any = field(default=None, init=False)

    def __init__(self,
                 cost_hamiltonian: Any,  # Full Hamiltonian object (not stored)
                 **kwargs
                 ):
        """
        Initialize HamiltonianOptimizationLoggerConfig with a cost Hamiltonian.
        :param cost_hamiltonian:
        :param logging_level:
        :param table_name_prefix:
        :param table_name_suffix:
        :param experiment_folders_hierarchy:
        :param directory_main:
        :param id_logger:
        :param id_logging_session:
        """
        # Extract specifiers from Hamiltonian (but don't store the full object)
        cost_hamiltonian_class_specifier = cost_hamiltonian.hamiltonian_class_specifier
        cost_hamiltonian_instance_specifier = cost_hamiltonian.hamiltonian_instance_specifier

        # Store only the extracted specifiers (not the full Hamiltonian)
        object.__setattr__(self, 'CostHamiltonianClass', cost_hamiltonian_class_specifier)
        object.__setattr__(self, 'CostHamiltonianInstance', cost_hamiltonian_instance_specifier)

        # call super
        super().__init__(**kwargs)

    def __post_init__(self):
        cost_hamiltonian_class_specifier = self.CostHamiltonianClass
        cost_hamiltonian_instance_specifier = self.CostHamiltonianInstance
        # Create Hamiltonian-specific experiment specifier
        hamiltonian_experiment_specifier = HamiltonianOptimizationSpecifier(
            CostHamiltonianClass=cost_hamiltonian_class_specifier,
            CostHamiltonianInstance=cost_hamiltonian_instance_specifier
        )
        # Get existing experiment_specifier from kwargs
        existing_specifier = self.experiment_specifier

        # Merge with the Hamiltonian specifier
        merged_specifier = existing_specifier.merge_with(other=hamiltonian_experiment_specifier)

        # Set the merged specifier
        object.__setattr__(self, 'experiment_specifier', merged_specifier)



    def __repr__(self):
        class_specifier = self.CostHamiltonianClass
        instance_specifier = self.CostHamiltonianInstance

        return (f"HamiltonianOptimizationLoggerConfig for Hamiltonian:\n "
                f"Class:{class_specifier.get_description_string()};\n "
                f"Instance:{instance_specifier.get_description_string()}")

if __name__ == '__main__':
    from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
    from quapopt.data_analysis.data_handling import (HamiltonianOptimizationSpecifier,
                                                     HamiltonianClassSpecifierSK,
                                                     HamiltonianInstanceSpecifierSK)

    from quapopt.optimization.QAOA import AnsatzSpecifier


    _hcs = HamiltonianClassSpecifierSK(Localities=(2,))
    _his = HamiltonianInstanceSpecifierSK(NumberOfQubits=3,
                                          HamiltonianInstanceIndex=0,)

    _hcs2 = HamiltonianClassSpecifierSK(Localities=(1,2))
    _his2 = HamiltonianInstanceSpecifierSK(NumberOfQubits=3,
                                            HamiltonianInstanceIndex=1,)


    #let's see if we can do a simple test of the dataclass
    config = HamiltonianOptimizationLoggerConfig(
        cost_hamiltonian=ClassicalHamiltonian(hamiltonian_list_representation=[(-5,(0,1)), (2, (1,2))],
                                              number_of_qubits=3,
                                              hamiltonian_class_specifier=_hcs,
                                              hamiltonian_instance_specifier=_his),  # Placeholder, since we don't have a full Hamiltonian object here
        logging_level=LoggingLevel.BASIC,
        table_name_prefix="test",
        table_name_suffix="log",
        experiment_folders_hierarchy=["experiment1", "run1"],
        directory_main=Path("/tmp/logs"),
    )

    hamiltonian_experiment_specifier = HamiltonianOptimizationSpecifier(
        CostHamiltonianClass=_hcs,
        CostHamiltonianInstance=_his
    )

    ansatz_specifier = AnsatzSpecifier(phase_hamiltonian_class_specifier=_hcs2,
                                      phase_hamiltonian_instance_specifier=_his2,
                                      depth=2,
                                      )
    merged_specifier = hamiltonian_experiment_specifier.merge_with(other=ansatz_specifier)

    print(merged_specifier)