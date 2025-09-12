# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


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

from quapopt import ancillary_functions as anf

from quapopt.data_analysis.data_handling.io_utilities import IOMixin, DEFAULT_STORAGE_DIRECTORY, add_file_format_suffix
from quapopt.data_analysis.data_handling.schemas import (
    STANDARD_NAMES_DATA_TYPES as SNDT,
    STANDARD_NAMES_DATA_TYPES_EXPERIMENT_SETS as SNDTES,
    STANDARD_NAMES_VARIABLES as SNV,
 BaseNameDataType

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


@dataclass
class ExperimentSetMetadataConfig:
    """
    Configuration for an experiment set metadata manager.

    This is a frozen dataclass that holds the configuration parameters for the ExperimentSetMetadataManager.
    It does not store any metadata itself, but provides a structure for initializing the manager.
    """

    ExperimentSetName: Optional[str] = None
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
            self.created_timestamp = anf.get_current_date_time()

        # Generate a random ID if not provided
        if self.ExperimentSetId is None:
            self.ExperimentSetId = anf.create_random_uuid()

        if self.ExperimentSetName is None:
            self.ExperimentSetName = self.ExperimentSetId


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
                 experiment_set_id: Optional[str] = None,
                 experiment_set_name: Optional[str] = None,
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
            self.experiment_set_name = self.experiment_set_id

        # Auto-save if we have existing experiment instances
        if len(self._experiment_instances_ids) > 0:
            self._write_set_tracking()

    @staticmethod
    def _convert_to_set_level_metadata(metadata_data_type: BaseNameDataType) -> BaseNameDataType:
        """
        Convert instance-level metadata type to experiment set level.
        
        Since this manager handles experiment set metadata, we automatically
        convert any metadata types to their set-level equivalents.
        
        :param metadata_data_type: Original data type (potentially instance-level)
        :return: Corresponding set-level data type
        """
        # Get the attribute name from the instance-level type
        for attr_name in dir(SNDT):
            if not attr_name.startswith('_'):
                attr_value = getattr(SNDT, attr_name)
                if attr_value == metadata_data_type:
                    # Found the matching attribute, now get set-level version
                    if hasattr(SNDTES, attr_name):
                        return getattr(SNDTES, attr_name)
                    break
        
        # If no conversion found, return original (for non-metadata types)
        return metadata_data_type

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

    def get_experiment_set_suffix(self) -> str:
        return self.get_key_value_pair(key_id=SNV.ExperimentSetID.id,
                                       value=self.experiment_set_id)

    @classmethod
    def get_metadata_filename(cls,
                              metadata_data_type: BaseNameDataType,
                              experiment_set_id: str,
                              table_name_prefix: Optional[str] = None,
                              table_name_suffix: Optional[str] = None) -> str:

        # Convert to set-level metadata type
        metadata_data_type = cls._convert_to_set_level_metadata(metadata_data_type)

        experiment_set_suffix = cls.get_key_value_pair(key_id=SNV.ExperimentSetID.id,
                                                       value=experiment_set_id)

        table_name_parts = [table_name_prefix, table_name_suffix, experiment_set_suffix]
        
        return cls.get_full_table_name(table_name_parts=table_name_parts,
                                       data_type=metadata_data_type)

    def get_metadata_path(self,
                          metadata_data_type: BaseNameDataType) -> Path:
        """Get the metadata directory path."""
        
        # Convert to set-level metadata type
        metadata_data_type = self._convert_to_set_level_metadata(metadata_data_type)
        
        return self.get_absolute_path_of_data_type(data_type=metadata_data_type)

    def _write_set_tracking(self) -> Path:
        return self.persistence_strategy.write_set_tracking(self)

    def read_set_tracking(self) -> pd.DataFrame:
        return self.persistence_strategy.read_set_tracking(metadata_manager=self)

    def read_shared_metadata(self,
                             metadata_data_type: BaseNameDataType,
                             persistence_strategy: Optional['MetadataPersistenceStrategy'] = None,
                             table_name_prefix: Optional[str] = None,
                             table_name_suffix: Optional[str] = None) -> Any:

        # Convert to set-level metadata type
        metadata_data_type = self._convert_to_set_level_metadata(metadata_data_type)

        if persistence_strategy is None:
            persistence_strategy = SplitFilePersistence()

        if not isinstance(persistence_strategy, SplitFilePersistence):
            raise ValueError("load_shared_metadata only works with SplitFilePersistence strategy")

        return persistence_strategy.read_set_metadata(self, metadata_data_type, table_name_prefix, table_name_suffix)

    def write_shared_metadata(self,
                              metadata_data_type: BaseNameDataType,
                              shared_metadata: Any,
                              overwrite_existing: bool = False,
                              table_name_prefix: Optional[str] = None,
                              table_name_suffix: Optional[str] = None) -> Path:

        # Convert to set-level metadata type
        metadata_data_type = self._convert_to_set_level_metadata(metadata_data_type)

        persistence_strategy = self.persistence_strategy

        if not isinstance(persistence_strategy, SplitFilePersistence):
            raise ValueError("save_shared_metadata only works with SplitFilePersistence strategy")

        return persistence_strategy.write_set_metadata(
            self,
            metadata_data_type,
            shared_metadata,
            overwrite_existing,
            table_name_prefix,
            table_name_suffix
        )


# ================================================================================
# METADATA PERSISTENCE STRATEGY
# ================================================================================

class MetadataPersistenceStrategy(ABC):
    """Abstract strategy for saving/loading experiment set metadata."""

    @staticmethod
    @abstractmethod
    def write_set_metadata(
            metadata_manager: ExperimentSetMetadataManager,
            metadata_data_type: BaseNameDataType,
            shared_metadata: Any,
            overwrite_existing: bool,
            table_name_prefix: Optional[str] = None,
            table_name_suffix: Optional[str] = None) -> Path:
        pass

    @staticmethod
    @abstractmethod
    def read_set_metadata(
            metadata_manager: ExperimentSetMetadataManager,
            metadata_data_type: BaseNameDataType,
            table_name_prefix: Optional[str] = None,
            table_name_suffix: Optional[str] = None) -> Any:
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
            metadata_data_type=SNDT.ExperimentSetTracking, 
            experiment_set_id=metadata_manager.experiment_set_id)
        config_csv_path = config_dir / config_filename

        #from . import add_file_format_suffix
        config_csv_path = add_file_format_suffix(string=config_csv_path, suffix='.csv')
        return config_csv_path
    @staticmethod
    def _get_json_file_path(#self,
                            metadata_manager: ExperimentSetMetadataManager,
                           metadata_data_type: BaseNameDataType,
                           table_name_prefix: Optional[str] = None,
                           table_name_suffix: Optional[str] = None) -> Path:
        """
        Get the path for the JSON file storing shared metadata.

        Args:
            metadata_manager: ExperimentSetMetadataManager instance
            metadata_data_type: Data type for the shared metadata
            table_name_prefix: Optional prefix for the table name
            table_name_suffix: Optional suffix for the table name

        Returns:
            Path to the JSON file
        """
        metadata_dir = metadata_manager.get_metadata_path(metadata_data_type)
        metadata_filename = metadata_manager.get_metadata_filename(
            metadata_data_type=metadata_data_type, 
            experiment_set_id=metadata_manager.experiment_set_id,
            table_name_prefix=table_name_prefix, 
            table_name_suffix=table_name_suffix)
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

        # print("hejka",config_csv_path)
        # print('hejka parent:',config_csv_path.parent)


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
        df = df[df[SNV.ExperimentSetName.id_long] == metadata_manager.experiment_set_name]

        return df





    @staticmethod
    def write_set_metadata(#cls,
            metadata_manager: ExperimentSetMetadataManager,
            metadata_data_type: BaseNameDataType,
            shared_metadata: Any,
            overwrite_existing: bool,
            table_name_prefix: Optional[str] = None,
            table_name_suffix: Optional[str] = None) -> Path:
        """Save shared metadata to JSON file."""
        # Generate shared metadata JSON path
        metadata_json_path   = SplitFilePersistence._get_json_file_path(metadata_manager=metadata_manager,
                                                                    metadata_data_type=metadata_data_type,
                                                                    table_name_prefix=table_name_prefix,
                                                                    table_name_suffix=table_name_suffix)

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
                          metadata_data_type: BaseNameDataType,
                          table_name_prefix: Optional[str] = None,
                          table_name_suffix: Optional[str] = None) -> Any:
        """
        Load shared metadata JSON file for given data type.
        
        Args:
            metadata_manager: ExperimentSetMetadataManager to determine file paths
            metadata_data_type: Data type for shared metadata file
            table_name_prefix: Optional prefix for the table name
            table_name_suffix: Optional suffix for the table name
            
        Returns:
            Loaded shared metadata (dict, DataFrame, etc.)
        """
        # Generate shared metadata JSON path
        metadata_json_path   = SplitFilePersistence._get_json_file_path(metadata_manager=metadata_manager,
                                                         metadata_data_type=metadata_data_type,
                                                         table_name_prefix=table_name_prefix,
                                                         table_name_suffix=table_name_suffix)

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
