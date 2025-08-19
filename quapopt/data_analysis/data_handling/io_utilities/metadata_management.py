# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

"""
Hierarchical metadata management system for organizing related experiments.
Provides experiment set and instance configurations with metadata persistence.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Optional

import pandas as pd

from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.data_analysis.data_handling.io_utilities import IOMixin, DEFAULT_STORAGE_DIRECTORY, add_file_format_suffix
from quapopt.data_analysis.data_handling.standard_names import (
    STANDARD_NAMES_DATA_TYPES as SNDT,
    STANDARD_NAMES_VARIABLES as SNV
)


# The point of the code here is the following:
# Sometimes, we implement multiple experiments (over multiple days) that share a bunch of the same METADATA
# (e.g., Hamiltonian class, qiskit compilation details, QPU).
# Those belong to the same experiment set, and we want to save that metadata once (in json file).
# Then, we can reference that metadata in each experiment instance without saving it many times for each
# experiment.

# For a single experiment set, the idea is to store generic metadata types in json, while keeping track
# of the experiment ids in the set in csv dataframe.


# ================================================================================
# HIERARCHICAL METADATA MANAGEMENT SYSTEM
# ================================================================================


@dataclass(frozen=True)
class ExperimentSetMetadataConfig:
    """
    Configuration for an experiment set metadata manager.

    This is a frozen dataclass that holds the configuration parameters for the ExperimentSetMetadataManager.
    It does not store any metadata itself, but provides a structure for initializing the manager.
    """

    ExperimentSetName: Optional[str] = "DefaultExperimentSet"
    ExperimentSetId: Optional[str] = None

    directory_main: Optional[Path] = None
    default_storage_directory: Optional[Path] = Path(DEFAULT_STORAGE_DIRECTORY)

    description: Optional[str] = None
    created_timestamp: Optional[str] = None

    def __post_init__(self):
        if self.ExperimentSetName is None and self.ExperimentSetId is None:
            raise ValueError("At least one of the: ExperimentSetName and ExperimentSetId should be provided")

        # Ensure created_timestamp is set to current time if not provided
        if self.created_timestamp is None:
            object.__setattr__(self, 'created_timestamp', anf.get_current_date_time())

        # Generate a random ID if not provided
        if self.ExperimentSetId is None:
            object.__setattr__(self, 'ExperimentSetId', anf.create_random_uuid())

        if self.ExperimentSetName is None:
            object.__setattr__(self, 'ExperimentSetName', self.ExperimentSetId[:10])


    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        data = {
            f'{SNV.ExperimentSetID.id_long}': [self.ExperimentSetId],
            f'{SNV.ExperimentSetName.id_long}': [self.ExperimentSetName],
            'main_directory': [str(self.directory_main)],
            'default_storage_directory': [str(self.default_storage_directory)],
            'created_timestamp': [self.created_timestamp],
            'description': [self.description],
        }

        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame):
        """Load configuration from pandas DataFrame."""
        if dataframe.empty or len(dataframe) == 0:
            raise ValueError("Provided DataFrame is empty")


        df = dataframe.iloc[0]
        return cls(ExperimentSetName=df[SNV.ExperimentSetName.id_long],
                   ExperimentSetId=df[SNV.ExperimentSetID.id_long],
                   directory_main=df['main_directory'] if pd.notna(df['main_directory']) else None,
                   default_storage_directory=df['default_storage_directory'] if pd.notna(df['default_storage_directory']) else Path(DEFAULT_STORAGE_DIRECTORY),
                   description=df['description'] if pd.notna(df['description']) else None,
                   created_timestamp=df['created_timestamp'] if pd.notna(df['created_timestamp']) else anf.get_current_date_time())



class ExperimentSetMetadataManager(IOMixin):
    """
    Manager for a set of related experiments sharing common metadata.
    
    This class tracks experiment instances and provides methods to save/load shared metadata,
    but does NOT store the metadata as an attribute. Metadata is provided when saving.
    """

    def __init__(self,
                 experiment_set_name: Optional[str] = "DefaultExperimentSet",
                 experiment_set_id: Optional[str] = None,
                 directory_main: Optional[str | Path] = None,
                 default_storage_directory: Optional[str | Path] = DEFAULT_STORAGE_DIRECTORY,
                 description: Optional[str] = None,
                 created_timestamp: Optional[str] = None):
        """
        Initialize ExperimentSetMetadataManager.
        
        Args:
            experiment_set_name: Human-readable name for the experiment set
            experiment_set_id: Unique identifier (auto-generated if None)
            metadata_base_path: Directory for metadata files
            description: Optional description of the experiment set
            created_timestamp: Creation timestamp (auto-generated if None)
        """

        # Set main attributes with defaults

        if experiment_set_name is None and experiment_set_id is None:
            raise ValueError("At least one of the: ExperimentSetName and ExperimentSetId should be provided")

        config = ExperimentSetMetadataConfig(ExperimentSetName=experiment_set_name,
                                             ExperimentSetId=experiment_set_id,
                                             directory_main=directory_main,
                                             default_storage_directory=default_storage_directory,
                                             description=description,
                                             created_timestamp=created_timestamp)

        self.experiment_set_name = config.ExperimentSetName
        self.experiment_set_id = config.ExperimentSetId

        self.main_directory = config.directory_main
        self.default_storage_directory = config.default_storage_directory

        self.base_path = self.construct_base_path()

        self.description = description
        self.created_timestamp = created_timestamp or anf.get_current_date_time()

        # Initialize collections
        self._experiment_instances_ids: List[str] = []
        self._experiment_instances_names: List[str] = []

        # Initialize internal attributes
        self.persistence_strategy: MetadataPersistenceStrategy = SplitFilePersistence()
        self.metadata_version: str = "1.0.0"

        # Handle name fallback
        if self.experiment_set_name is None:
            self.experiment_set_name = self.experiment_set_id[:10]

        # Auto-save if we have existing experiment instances
        if len(self._experiment_instances_ids) > 0:
            self._write_set_tracking()

    @classmethod
    def from_config(cls,
                    config: ExperimentSetMetadataConfig) -> 'ExperimentSetMetadataManager':
        """
        Create an ExperimentSetMetadataManager from a configuration object.
        :param config:
        :return:
        """
        return cls(experiment_set_name=config.ExperimentSetName,
                   experiment_set_id=config.ExperimentSetId,
                   directory_main=config.directory_main,
                   default_storage_directory=config.default_storage_directory,
                   description=config.description,
                   created_timestamp=config.created_timestamp)

    def construct_base_path(self):
        return super().construct_base_path(directory_main=self.main_directory,
                                           default_storage_directory=self.default_storage_directory)

    @classmethod
    def from_name_and_id(cls,
                         experiment_set_name: str,
                         experiment_set_id: str,
                         directory_main=None,
                         default_storage_directory=DEFAULT_STORAGE_DIRECTORY,
                         ) -> 'ExperimentSetMetadataManager':

        dummy_instance = cls(
            experiment_set_name=experiment_set_name,
            experiment_set_id=experiment_set_id,
            directory_main=directory_main,
            default_storage_directory=default_storage_directory,
        )

        # let's try to read the metadata_manager from file
        existing_config_df: pd.DataFrame = dummy_instance.read_set_tracking()

        if isinstance(existing_config_df, pd.DataFrame):

            if existing_config_df.empty:
                existing_config_df = None

        if existing_config_df is not None:
            read_exp_names_list = existing_config_df[SNV.ExperimentSetName.id_long].tolist()
            read_exp_ids_list = existing_config_df[SNV.ExperimentSetID.id_long].tolist()

            read_exp_name = read_exp_names_list[0]
            read_exp_id = read_exp_ids_list[0]

            # Verify the loaded metadata_manager has the expected name
            if read_exp_name != experiment_set_name:
                raise ValueError(
                    f"Loaded experiment set has name '{read_exp_name}' "
                    f"but expected '{experiment_set_name}'. This suggests a naming conflict."
                )
            # Verify ID matches
            if read_exp_id != experiment_set_id:
                raise ValueError(
                    f"Loaded experiment set has ID '{read_exp_id}' "
                    f"but expected '{experiment_set_id}'. This suggests a naming conflict."
                )

            manager = cls.from_config(config=ExperimentSetMetadataConfig.from_dataframe(dataframe=existing_config_df))
            # Skip the first row which is experiment set info, get instance IDs
            instance_ids = existing_config_df[SNV.ExperimentInstanceID.id_long].tolist()
            if instance_ids:  # Only add if there are actual instance IDs
                manager.add_experiment_instances(
                                                 experiment_instance_ids=instance_ids,
                                                 write_to_tracking=False)
            return manager

        print("No existing metadata found for the specified experiment set. ")
        return None

    @property
    def experiment_instances_ids(self) -> List[str]:
        """Read-only access to experiment instance IDs."""
        return self._experiment_instances_ids

    @property
    def experiment_instances_names(self) -> List[str]:
        """Read-only access to experiment instance names."""
        return self._experiment_instances_names

    def __repr__(self):
        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""

        _l = len(self._experiment_instances_ids)
        # If no instances yet, create at least one row with experiment set info
        if _l == 0:
            _l = 1
            instance_ids = [str(None)]  # Use None for empty instance
        else:
            instance_ids = self.experiment_instances_ids
            
        data = {
            f'{SNV.ExperimentSetID.id_long}': [self.experiment_set_id] * _l,
            f'{SNV.ExperimentSetName.id_long}': [self.experiment_set_name] * _l,
            'main_directory': [str(self.main_directory)] * _l,
            'default_storage_directory': [str(self.default_storage_directory)] * _l,
            'base_path': [str(self.base_path)] * _l,
            'created_timestamp': [self.created_timestamp] * _l,
            'description': [self.description] * _l,
            'metadata_version': [self.metadata_version] * _l,
            f'{SNV.ExperimentInstanceID.id_long}': instance_ids,
        }

        return pd.DataFrame(data)

    def add_experiment_instances(self,
                                 experiment_instance_ids: List[str],
                                 write_to_tracking: bool = True
                                 ) -> Optional[Path]:


        for exp_id in experiment_instance_ids:
            self._experiment_instances_ids.append(exp_id)

        if write_to_tracking:
            # Write to tracking file if requested
            return self._write_set_tracking()

        return None

    def get_experiment_set_prefix(self) -> str:
        return self.get_key_value_pair(key_id=SNV.ExperimentSetName.id,
                                       value=self.experiment_set_name)

    @classmethod
    def get_metadata_filename(cls,
                              experiment_set_name: str,
                              metadata_data_type: SNDT) -> str:
        prefix = cls.get_key_value_pair(key_id=SNV.ExperimentSetName.id,
                                        value=experiment_set_name)

        return cls.get_full_table_name(table_name_parts=[prefix],
                                       data_type=metadata_data_type)

    def get_metadata_path(self,
                          metadata_data_type: SNDT) -> Path:
        """Get the metadata directory path."""

        return self.get_absolute_path_of_data_type(data_type=metadata_data_type)

    def _write_set_tracking(self) -> Path:
        return self.persistence_strategy.write_set_tracking(self)

    def read_set_tracking(self) -> pd.DataFrame:
        return self.persistence_strategy.read_set_tracking(metadata_manager=self)

    def read_shared_metadata(self,
                             metadata_data_type: SNDT,
                             persistence_strategy: Optional['MetadataPersistenceStrategy'] = None) -> Any:

        if persistence_strategy is None:
            persistence_strategy = SplitFilePersistence()

        if not isinstance(persistence_strategy, SplitFilePersistence):
            raise ValueError("load_shared_metadata only works with SplitFilePersistence strategy")

        return persistence_strategy.read_set_metadata(self, metadata_data_type)

    def write_shared_metadata(self,
                              metadata_data_type: SNDT,
                              shared_metadata: Any,
                              overwrite_existing: bool = False) -> Path:

        persistence_strategy = self.persistence_strategy

        if not isinstance(persistence_strategy, SplitFilePersistence):
            raise ValueError("save_shared_metadata only works with SplitFilePersistence strategy")

        return persistence_strategy.write_set_metadata(
            self,
            metadata_data_type,
            shared_metadata,
            overwrite_existing
        )


#
# @dataclass(frozen=True)
# class ExperimentInstanceMetadataManager(LoggerConfig, MetadataPathMixin):
#     """
#     Configuration for a specific experiment instance within an experiment set.
#     References the parent ExperimentSetMetadataManager and adds instance-specific metadata.
#
#     This is domain-agnostic - specific domains should extend this class.
#     """
#     # Reference to the parent experiment set
#     # Note this helps to recover metadata that is shared across experiments
#     ExperimentSetId: Optional[str] = None
#     ExperimentSetName: Optional[str] = None
#
#     # Main directory where metadata for this experiment instance will be stored.
#     metadata_main_directory: Optional[Path] = field(default=None, init=False)
#
#     # Human-readable name for the experiment instance.
#     # Note: this will be used to generate metadata filenames, so it should be valid for file naming.
#     experiment_instance_name: str = "DefaultExperimentInstance"
#     # Unique identifier for this experiment instance
#     # Note: this will be used to identify the instance if multiple instances are used with the same name
#     experiment_instance_id: str = field(default_factory=lambda: anf.create_random_uuid())
#
#
#     # Timestamp for this specific instance
#     instance_timestamp: str = field(default_factory=lambda: anf.get_current_date_time())
#
#     def __post_init__(self):
#         # For frozen dataclass, we need to use object.__setattr__ in __post_init__
#         # This is the only acceptable place for this pattern
#         if self.experiment_instance_name is None:
#             object.__setattr__(self, 'experiment_instance_name', self.experiment_instance_id[0:10])
#
#         object.__setattr__(self, 'metadata_base_path', self.base_path)
#
#     def load_experiment_set_metadata(self,
#                                      persistence_strategy: Optional['MetadataPersistenceStrategy'] = None,
#                                      data_type: SNDT = None) -> 'ExperimentSetMetadataManager':
#         """Load the parent experiment set configuration."""
#         return ExperimentSetMetadataManager.read_set_tracking(
#             ExperimentSetName=self.ExperimentSetName,
#             metadata_main_directory=self.metadata_main_directory,
#             persistence_strategy=persistence_strategy,
#             data_type=data_type
#         )
#
#     @classmethod
#     def get_experiment_instance_prefix(cls, experiment_instance_name: str) -> str:
#         """Generate prefix for experiment instance metadata files."""
#         return cls.generate_prefix(SNV.ExperimentInstanceName.id, experiment_instance_name)
#
#     def _get_experiment_instance_prefix(self) -> str:
#         """Non-static version to use instance's experiment instance name."""
#         return self.get_experiment_instance_prefix(self.experiment_instance_name)
#
#     @classmethod
#     def get_metadata_filename(cls, experiment_instance_name: str,
#                               data_type: SNDT) -> str:
#         """Generate filename for storing this instance's metadata."""
#         prefix = cls.get_experiment_instance_prefix(experiment_instance_name)
#         return cls.generate_metadata_filename(prefix, data_type)
#
#     def _get_metadata_filename(self, data_type: SNDT) -> str:
#         """Non-static version to use instance's experiment instance name."""
#         return self.get_metadata_filename(self.experiment_instance_name, data_type)
#

# ================================================================================
# METADATA PERSISTENCE STRATEGY
# ================================================================================

class MetadataPersistenceStrategy(ABC):
    """Abstract strategy for saving/loading experiment set metadata."""

    @staticmethod
    @abstractmethod
    def write_set_metadata(
            metadata_manager: ExperimentSetMetadataManager,
            metadata_data_type: SNDT,
            shared_metadata: Any,
            overwrite_existing: bool) -> Path:
        pass

    @staticmethod
    @abstractmethod
    def read_set_metadata(
            metadata_manager: ExperimentSetMetadataManager,
            metadata_data_type: SNDT) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def write_set_tracking(
            metadata_manager: ExperimentSetMetadataManager) -> Path:
        pass

    @staticmethod
    @abstractmethod
    def read_set_tracking(
            metadata_manager: ExperimentSetMetadataManager,
            return_none_if_not_found: bool = True) -> ExperimentSetMetadataManager:
        pass


class SplitFilePersistence(MetadataPersistenceStrategy):
    """
    Split-file persistence: CSV for metadata_manager, JSON for shared metadata.
    
    Architecture:
    1. Config CSV (ExperimentSetTracking data type): Lightweight experiment set info
    2. Shared metadata JSON (user-specified data type): Complex experimental data
    """

    @staticmethod
    def _get_csv_file_path(#cls,
                          metadata_manager: ExperimentSetMetadataManager) -> Path:
        """
        Get the path for the CSV file storing experiment set tracking information.

        Args:
            metadata_manager: ExperimentSetMetadataManager instance
            data_type: Data type for the metadata file

        Returns:
            Path to the CSV file
        """

        config_dir = metadata_manager.get_metadata_path(SNDT.ExperimentSetTracking)
        config_filename = metadata_manager.get_metadata_filename(
            metadata_manager.experiment_set_name, SNDT.ExperimentSetTracking)
        config_csv_path = config_dir / config_filename

        #from . import add_file_format_suffix
        config_csv_path = add_file_format_suffix(string=config_csv_path, suffix='.csv')
        return config_csv_path
    @staticmethod
    def _get_json_file_path(#self,
                            metadata_manager: ExperimentSetMetadataManager,
                           metadata_data_type: SNDT) -> Path:
        """
        Get the path for the JSON file storing shared metadata.

        Args:
            metadata_manager: ExperimentSetMetadataManager instance
            metadata_data_type: Data type for the shared metadata

        Returns:
            Path to the JSON file
        """
        metadata_dir = metadata_manager.get_metadata_path(metadata_data_type)
        metadata_filename = metadata_manager.get_metadata_filename(
            metadata_manager.experiment_set_name, metadata_data_type)
        metadata_json_path = metadata_dir / metadata_filename

        #from . import add_file_format_suffix
        metadata_json_path = add_file_format_suffix(string=metadata_json_path, suffix='.json')
        return metadata_json_path



    @staticmethod
    def write_set_tracking(#cls,
            metadata_manager: ExperimentSetMetadataManager) -> Path:

        config_csv_path = SplitFilePersistence._get_csv_file_path(metadata_manager=metadata_manager)

        # Ensure directory exists
        config_csv_path.parent.mkdir(parents=True, exist_ok=True)


        #First, we wish to check if the given instance already exists
        existing_df = SplitFilePersistence.read_set_tracking(metadata_manager=metadata_manager,
                                             return_none_if_not_found=True)

        if isinstance(existing_df, pd.DataFrame):
            if existing_df.empty:
                existing_df = None

        if existing_df is not None:
            existing_instance_ids = set(existing_df[SNV.ExperimentInstanceID.id_long].tolist())
            if set(metadata_manager.experiment_instances_ids).difference(existing_instance_ids) == set():
                # If all experiment instance IDs already exist, do not overwrite
                return config_csv_path


        # Create DataFrame with metadata_manager data
        df = metadata_manager.to_dataframe()


        #set infinite display options for pandas
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)



        df.to_csv(config_csv_path, index=False)



        return config_csv_path

    @staticmethod
    def read_set_tracking(#cls,
                          metadata_manager: ExperimentSetMetadataManager,
                          return_none_if_not_found: bool = True) -> pd.DataFrame:

        config_csv_path = SplitFilePersistence._get_csv_file_path(metadata_manager=metadata_manager)

        if not config_csv_path.exists():
            if return_none_if_not_found:
                return None
            else:
                raise FileNotFoundError(f"Metadata tracking file not found: {config_csv_path}")

        # Read metadata_manager CSV
        df = pd.read_csv(config_csv_path)
        if df.empty or len(df)==0:
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(f"Empty metadata_manager file: {config_csv_path}")

        df[SNV.ExperimentSetID.id_long] = df[SNV.ExperimentSetID.id_long].astype(str)
        df[SNV.ExperimentSetName.id_long] = df[SNV.ExperimentSetName.id_long].astype(str)
        df[SNV.ExperimentInstanceID.id_long] = df[SNV.ExperimentInstanceID.id_long].astype(str)


        df = df[df[SNV.ExperimentSetID.id_long] == metadata_manager.experiment_set_id]


        return df





    @staticmethod
    def write_set_metadata(#cls,
            metadata_manager: ExperimentSetMetadataManager,
            metadata_data_type: SNDT,
            shared_metadata: Any,
            overwrite_existing: bool) -> Path:
        """Save shared metadata to JSON file."""
        # Generate shared metadata JSON path
        metadata_json_path   = SplitFilePersistence._get_json_file_path(metadata_manager=metadata_manager,
                                                                    metadata_data_type=metadata_data_type)

        # Check if file exists and handle overwrite logic
        if metadata_json_path.exists() and not overwrite_existing:
            raise ValueError(f"Metadata file already exists: {metadata_json_path}")
        elif metadata_json_path.exists() and overwrite_existing:
            # Remove existing file if overwrite is requested
            metadata_json_path.unlink()
        # Ensure directory exists
        metadata_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare metadata for JSON serialization
        if isinstance(shared_metadata, pd.DataFrame):
            metadata_to_save = {
                '_type': 'pandas_dataframe',
                'data': shared_metadata.to_dict('records'),
                'columns': list(shared_metadata.columns)
            }
        else:
            metadata_to_save = shared_metadata

        # Save to JSON
        with open(metadata_json_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2, default=str)

        return metadata_json_path

    @staticmethod
    def read_set_metadata(#cls,
                          metadata_manager: ExperimentSetMetadataManager,
                          metadata_data_type: SNDT) -> Any:
        """
        Load shared metadata JSON file for given data type.
        
        Args:
            metadata_manager: ExperimentSetMetadataManager to determine file paths
            metadata_data_type: Data type for shared metadata file
            
        Returns:
            Loaded shared metadata (dict, DataFrame, etc.)
        """
        # Generate shared metadata JSON path
        metadata_json_path   = SplitFilePersistence._get_json_file_path(metadata_manager=metadata_manager,
                                                         metadata_data_type=metadata_data_type)

        if not metadata_json_path.exists():
            raise FileNotFoundError(f"Shared metadata file not found: {metadata_json_path}")

        # Load JSON
        with open(metadata_json_path, 'r') as f:
            metadata = json.load(f)

        # Handle pandas DataFrame deserialization
        if isinstance(metadata, dict) and metadata.get('_type') == 'pandas_dataframe':
            return pd.DataFrame(
                metadata['data'],
                columns=metadata['columns']
            )

        return metadata
