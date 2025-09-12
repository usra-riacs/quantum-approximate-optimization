# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from pathlib import Path
from typing import List, Optional, Union, Any

import numpy as np
import pandas as pd

from quapopt import ancillary_functions as anf

from quapopt.data_analysis.data_handling.io_utilities import DEFAULT_STORAGE_DIRECTORY
from quapopt.data_analysis.data_handling.schemas.configurations import LoggingLevel, ExperimentLoggerConfig
from quapopt.data_analysis.data_handling.io_utilities.metadata_management import ExperimentSetMetadataManager
from quapopt.data_analysis.data_handling.io_utilities.standardized_io import (ResultsIO)
from quapopt.data_analysis.data_handling.schemas.naming import (
    DEFAULT_TABLE_NAME_PARTS_SEPARATOR,
    DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR,
STANDARD_NAMES_DATA_TYPES as SNDT,
STANDARD_NAMES_VARIABLES as SNV,
)
from quapopt.data_analysis.data_handling.schemas.naming import (StandardizedSpecifier,
                                                                BaseNameDataType, )

_DATA_TYPES_LOGGING_MINIMAL = [SNDT.VariationalParameters,
                               SNDT.ExpectedValues,
                               SNDT.OptimizationOverview,
                               SNDT.TestResults,
                               SNDT.HamiltonianTransformations,
                               # SNDT.ExperimentSetLoggingMetadata,
                               SNDT.ExperimentSetTracking,
                               # SNDT.ExperimentInstanceTracking,
                               # SNDT.ExperimentSetMetadataManager,
                               SNDT.JobMetadata,
                               SNDT.BackendData,
                               SNDT.Unspecified
                               ]
_DATA_TYPES_LOGGING_BASIC = _DATA_TYPES_LOGGING_MINIMAL + [SNDT.Bitstrings,
                                                           SNDT.BitstringsHistograms,
                                                           SNDT.VariableValues]
_DATA_TYPES_LOGGING_DETAILED = _DATA_TYPES_LOGGING_BASIC + [SNDT.Energies,
                                                            SNDT.EnergiesHistograms]
_DATA_TYPES_LOGGING_VERY_DETAILED = _DATA_TYPES_LOGGING_DETAILED + [SNDT.Unspecified,
                                                                    SNDT.StateVectors,
                                                                    SNDT.Correlators]

_ALL_DATA_TYPES = (_DATA_TYPES_LOGGING_MINIMAL +
                   _DATA_TYPES_LOGGING_BASIC +
                   _DATA_TYPES_LOGGING_DETAILED +
                   _DATA_TYPES_LOGGING_VERY_DETAILED)


def verify_whether_to_log_data(logging_level: LoggingLevel,
                               data_type: BaseNameDataType):
    if logging_level == LoggingLevel.NONE:
        return False

    if data_type not in _ALL_DATA_TYPES or data_type is None:
        # If the data type is not in the list of all data types, we assume it should be logged because it's some custom
        # data type we didn't account for.
        return True

    if logging_level == LoggingLevel.MINIMAL:
        if data_type not in _DATA_TYPES_LOGGING_MINIMAL:
            return False
    if logging_level == LoggingLevel.BASIC:
        if data_type not in _DATA_TYPES_LOGGING_BASIC:
            return False
    if logging_level == LoggingLevel.DETAILED:
        if data_type not in _DATA_TYPES_LOGGING_DETAILED:
            return False
    if logging_level == LoggingLevel.VERY_DETAILED:
        if data_type not in _DATA_TYPES_LOGGING_VERY_DETAILED:
            return False

    # If we got here, it means the data type is in the list of data types for the given logging level.
    return True


class ResultsLogger(ResultsIO):
    def __init__(self,
                 experiment_set_name: Optional[str] = None,
                 experiment_instance_id: Optional[str] = None,
                 experiment_set_id: Optional[str] = None,
                 experiment_specifier: Optional[StandardizedSpecifier] = None,
                 experiment_folders_hierarchy: Optional[List[str]] = None,
                 table_name_prefix: Optional[str] = None,
                 table_name_suffix: Optional[str] = None,
                 directory_main: Optional[str | Path] = None,
                 logging_level: LoggingLevel = LoggingLevel.BASIC,
                 config: Optional[ExperimentLoggerConfig] = None, ):

        """
        The file-naming convention is as follows:

        <table_name_prefix>;<table_name_prefix>;<table_name_prefix>dat=<data_type>.csv

        In general, we imagine that:
        1. "table_name_prefix" = a description of set of experiments,
           e.g., "BenchmarkingNDAR;Hamiltonian=SK;NumberOfQubits=5"
        2. "table_name_prefix" = a description of the particular experimental instance,
              e.g., "HamiltonianSeed=0;Optimizer=Adam;LearningRate=0.01"
        3. "table_name_prefix" = Whatever additional information we want to add that the two others do not account for.

        """

        if config is None:
            config = ExperimentLoggerConfig(experiment_set_name=experiment_set_name,
                                            experiment_set_id=experiment_set_id,
                                            experiment_instance_id=experiment_instance_id,
                                            experiment_specifier=experiment_specifier,
                                            experiment_folders_hierarchy=experiment_folders_hierarchy,
                                            table_name_prefix=table_name_prefix,
                                            table_name_suffix=table_name_suffix,
                                            directory_main=directory_main,
                                            logging_level=logging_level)
        self.config = config

        experiment_folders_hierarchy = config.experiment_folders_hierarchy
        experiment_specifier = config.experiment_specifier
        experiment_set_name = config.experiment_set_name
        directory_main = config.directory_main
        table_name_prefix = config.table_name_prefix
        table_name_suffix = config.table_name_suffix
        experiment_set_id = config.experiment_set_id
        experiment_instance_id = config.experiment_instance_id
        logging_level = config.logging_level

        self._experiment_folders_hierarchy = experiment_folders_hierarchy
        self._experiment_specifier = experiment_specifier

        # Build the directory path manually without relying on parent methods
        if directory_main is not None:
            directory_main = Path(directory_main)
        else:
            directory_main = Path("")

        if experiment_folders_hierarchy:
            directory_main = directory_main / Path("/".join(experiment_folders_hierarchy))

        # print(directory_main, 'yo')
        # raise KeyboardInterrupt
        super().__init__(table_name_prefix=table_name_prefix,
                         table_name_suffix=table_name_suffix,
                         directory_main=directory_main,
                         default_storage_directory=DEFAULT_STORAGE_DIRECTORY,
                         table_name_parts_separator=DEFAULT_TABLE_NAME_PARTS_SEPARATOR,
                         dataframe_type_name_separator=DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR)

        # Merged functionality from ResultsLoggerBase
        self._logging_level = logging_level
        self._files_history = []

        metadata_manager = ExperimentSetMetadataManager.from_name_and_id(experiment_set_name=experiment_set_name,
                                                                         experiment_set_id=experiment_set_id,
                                                                         directory_main=directory_main,
                                                                         default_storage_directory=self.get_default_storage_directory())
        if metadata_manager is None:
            metadata_manager = ExperimentSetMetadataManager(experiment_set_id=experiment_set_id,
                                                            experiment_set_name=experiment_set_name,
                                                            directory_main=directory_main,
                                                            default_storage_directory=self.get_default_storage_directory())

        self._metadata_manager = metadata_manager

        if experiment_instance_id is None:
            experiment_instance_id = anf.create_random_uuid()

        self._experiment_instance_id = experiment_instance_id

        self.metadata_manager.add_experiment_instances(experiment_instance_ids=[self.experiment_instance_id],
                                                       write_to_tracking=True)

        # self.config = config

    def set_logging_level(self, level):
        """Set the logging level for the results logger."""
        self._logging_level = level

    @property
    def logging_level(self):
        """Get the current logging level."""
        return self._logging_level

    @property
    def files_history(self):
        """Get the history of files written by this logger."""
        return self._files_history

    def __repr__(self):
        return (f"ResultsLogger\n"
                f"experiment_set_name={self.experiment_set_name}\n"
                f"experiment_set_id={self.experiment_set_id}\n"
                f"experiment_instance_id={self.experiment_instance_id}\n"
                f"experiment_specifier={self.experiment_specifier.get_description_string()}\n"
                f"experiment_folders_hierarchy={self.experiment_folders_hierarchy}\n"
                f"base_path={self.base_path}\n"
                f"table_name_prefix={self.table_name_prefix}\n"
                f"table_name_suffix={self.table_name_suffix}\n"
                )

    @classmethod
    def from_config(cls, config: ExperimentLoggerConfig) -> 'ResultsLogger':
        """
        Factory method to create a ResultsLogger instance from a configuration object.
        :param config:
        :return:
        """

        return cls(
            experiment_set_name=config.experiment_set_name,
            experiment_set_id=config.experiment_set_id,
            experiment_instance_id=config.experiment_instance_id,
            experiment_specifier=config.experiment_specifier,
            experiment_folders_hierarchy=config.experiment_folders_hierarchy,
            table_name_prefix=config.table_name_prefix,
            table_name_suffix=config.table_name_suffix,
            directory_main=config.directory_main,
            logging_level=config.logging_level,
            config=config

        )

    def get_absolute_path_of_data_type(self, data_type: BaseNameDataType) -> Path:
        """
        Returns the absolute path of the data type directory.
        :param data_type:
        :return:
        """
        if data_type is None:
            data_type = SNDT.Unspecified
        # TODO(FBM): it shouldn't be necessary -- bug
        return self.base_path / self.get_subpath_of_data_type(data_type=data_type)

    @property
    def experiment_specifier(self):
        return self._experiment_specifier

    @property
    def experiment_folders_hierarchy(self):
        return self._experiment_folders_hierarchy

    @property
    def table_name_prefix(self):
        return self._table_name_prefix

    @table_name_prefix.setter
    def table_name_prefix(self, value: str):
        if not isinstance(value, str):
            raise TypeError("table_name_prefix must be a string")
        self._table_name_prefix = value

    @property
    def experiment_instance_id(self):
        return self._experiment_instance_id

    @experiment_instance_id.setter
    def experiment_instance_id(self, value: str):
        if not isinstance(value, str):
            raise TypeError("experiment_instance_id must be a string")
        if not value:
            raise ValueError("experiment_instance_id cannot be an empty string")
        self._experiment_instance_id = value

    @property
    def metadata_manager(self):
        return self._metadata_manager

    @metadata_manager.setter
    def metadata_manager(self, value: ExperimentSetMetadataManager):
        if not isinstance(value, ExperimentSetMetadataManager):
            raise TypeError("metadata_manager must be an instance of ExperimentSetMetadataManager")
        self._metadata_manager = value

    @property
    def experiment_set_name(self):
        return self.metadata_manager.experiment_set_name

    @property
    def experiment_set_id(self):
        return self.metadata_manager.experiment_set_id

    def extend_table_name_suffix(self,
                                 additional_part: str,
                                 appending: bool = True, ):

        if not isinstance(additional_part, str):
            raise TypeError("additional_part must be a string")

        if appending:
            parts = [self.table_name_suffix, additional_part]
        else:
            parts = [additional_part, self.table_name_suffix]

        self.table_name_suffix = self.join_table_name_parts(table_name_parts=parts)

    def extend_table_name_prefix(self,
                                 additional_part: str,
                                 appending: bool = True, ):

        if not isinstance(additional_part, str):
            raise TypeError("additional_part must be a string")

        if appending:
            parts = [self.table_name_prefix, additional_part]
        else:
            parts = [additional_part, self.table_name_prefix]

        self.table_name_prefix = self.join_table_name_parts(table_name_parts=parts)

    def read_experiment_set_tracking(self):
        return self.metadata_manager.read_set_tracking()

    def _read_shared_metadata(self,
                              data_type: SNDT,
                              table_name_prefix: Optional[str] = None,
                              table_name_suffix: Optional[str] = None
                              ):
        return self.metadata_manager.read_shared_metadata(metadata_data_type=data_type,
                                                          table_name_prefix=table_name_prefix,
                                                          table_name_suffix=table_name_suffix, )

    def _write_shared_metadata(self,
                               metadata_data_type: SNDT,
                               shared_metadata: Any,
                               overwrite_existing: bool = False,
                               table_name_prefix: Optional[str] = None,
                               table_name_suffix: Optional[str] = None
                               ) -> Path:
        return self.metadata_manager.write_shared_metadata(metadata_data_type=metadata_data_type,
                                                           shared_metadata=shared_metadata,
                                                           overwrite_existing=overwrite_existing,
                                                           table_name_prefix=table_name_prefix,
                                                           table_name_suffix=table_name_suffix, )

    def write_metadata(self,
                       metadata: Any,
                       shared_across_experiment_set: bool,
                       data_type: BaseNameDataType = None,
                       overwrite_existing_non_csv: bool = False,
                       table_name_prefix: Optional[str] = None,
                       table_name_suffix: Optional[str] = None,
                       annotate_with_experiment_metadata: bool = False,
                       additional_annotation_dict: Optional[dict] = None,
                       ignore_logging_level: bool = False,
                       append_experiment_set_suffix: bool = True,
                       ) -> Optional[Path]:
        """
        Write metadata either as shared (experiment set level) or instance-specific metadata.

        :param metadata: The metadata to write (DataFrame, dict, or other serializable object)
        :param data_type: The data type for categorizing the metadata
        :param shared_across_experiment_set: If True, writes as shared metadata at experiment set level.
                                            If False, writes as instance-specific metadata.
        :param overwrite_existing: For shared metadata, whether to overwrite existing files
        :param table_name_prefix: Optional prefix for the table name
        :param table_name_suffix: Optional suffix for the table name
        :param annotate_with_experiment_metadata: For instance metadata, whether to annotate with experiment metadata
        :param additional_annotation_dict: Additional annotations to add
        :param ignore_logging_level: Whether to ignore the logging level check
        :param append_experiment_set_suffix: Whether to append experiment set suffix to table name
        :return: Path to the written file, or None if not written
        """
        if data_type is None:
            data_type = SNDT.Unspecified

        if shared_across_experiment_set:
            # Write as shared metadata at experiment set level
            return self._write_shared_metadata(
                metadata_data_type=data_type,
                shared_metadata=metadata,
                overwrite_existing=overwrite_existing_non_csv,
                table_name_prefix=table_name_prefix,
                table_name_suffix=table_name_suffix)
        else:
            return self.write_results(
                dataframe=metadata,
                data_type=data_type,
                table_name_prefix=table_name_prefix,
                table_name_suffix=table_name_suffix,
                annotate_with_experiment_metadata=annotate_with_experiment_metadata,
                additional_annotation_dict=additional_annotation_dict,
                ignore_logging_level=ignore_logging_level,
                append_experiment_set_suffix=append_experiment_set_suffix,
                overwrite_existing_non_csv=overwrite_existing_non_csv
            )


    def read_metadata(self,
                      shared_across_experiment_set: bool,
                      data_type: BaseNameDataType = None,
                      table_name_prefix: Optional[str] = None,
                      table_name_suffix: Optional[str] = None,
                      experiment_instance_ids: Optional[List[str]] = None,
                      annotate_with_experiment_metadata: bool = False,
                      additional_annotation_dict: Optional[dict] = None,
                      format_type: str = 'dataframe',
                      return_none_if_not_found: bool = False,
                      filter_by_experiment_set: bool = True,
                      append_experiment_set_suffix: bool = True) -> Any:
        """
        Read metadata either as shared (experiment set level) or instance-specific metadata.

        :param data_type: The data type for categorizing the metadata
        :param shared_across_experiment_set: If True, reads shared metadata at experiment set level.
                                            If False, reads instance-specific metadata.
        :param table_name_prefix: Optional prefix for the table name
        :param table_name_suffix: Optional suffix for the table name
        :param experiment_instance_ids: For instance metadata, specific instance IDs to read
        :param annotate_with_experiment_metadata: For instance metadata, whether to annotate with experiment metadata
        :param additional_annotation_dict: Additional annotations to filter by
        :param format_type: Format type for reading ('dataframe' or other)
        :param return_none_if_not_found: Whether to return None if file not found
        :param filter_by_experiment_set: Whether to filter by experiment set
        :param append_experiment_set_suffix: Whether to append experiment set suffix to table name
        :return: The read metadata
        """
        if data_type is None:
            data_type = SNDT.Unspecified

        if shared_across_experiment_set:
            # Read as shared metadata at experiment set level
            return self._read_shared_metadata(
                data_type=data_type,
                table_name_prefix=table_name_prefix,
                table_name_suffix=table_name_suffix
            )
        else:
            # Read as instance-specific metadata using read_results
            return self.read_results(
                data_type=data_type,
                table_name_prefix=table_name_prefix,
                table_name_suffix=table_name_suffix,
                experiment_instance_ids=experiment_instance_ids,
                annotate_with_experiment_metadata=annotate_with_experiment_metadata,
                additional_annotation_dict=additional_annotation_dict,
                format_type=format_type,
                return_none_if_not_found=return_none_if_not_found,
                filter_by_experiment_set=filter_by_experiment_set,
                append_experiment_set_suffix=append_experiment_set_suffix
            )

    def get_dataframe_annotation(self) -> dict:

        annotations = self._experiment_specifier.get_dataframe_annotation()
        if annotations is None:
            annotations = {}

        return annotations

    def _create_annotations_dict(self,
                                 annotate_with_ids: bool = True,
                                 annotate_with_experiment_metadata=False,
                                 additional_annotation_dict: Optional[dict] = None,
                                 ):
        annotation_dicts = []
        if annotate_with_ids:
            annotation_dicts.append({f'{SNV.ExperimentInstanceID.id_long}': self.experiment_instance_id,
                                     })
        if annotate_with_experiment_metadata:
            annotations_dict_metadata = self.get_dataframe_annotation()
            if annotations_dict_metadata != {}:
                annotation_dicts.append(annotations_dict_metadata)
        if additional_annotation_dict is not None:
            annotation_dicts.append(additional_annotation_dict)
        annotations_dict = {}
        for ann_dict in annotation_dicts:
            annotations_dict = {**annotations_dict, **ann_dict}

        # print('annotate_with_ids:',annotate_with_ids)
        return annotations_dict

    def write_results(self,
                      dataframe: Union[pd.DataFrame, np.ndarray],
                      table_name: Optional[str] = None,
                      data_type: BaseNameDataType = None,
                      directory_subpath: Optional[str | Path] = None,
                      # annotate_with_ids: bool = True,
                      annotate_with_experiment_metadata=False,
                      additional_annotation_dict: Optional[dict] = None,
                      ignore_logging_level=False,
                      table_name_prefix: Optional[str] = None,
                      table_name_suffix: Optional[str] = None,
                      append_experiment_set_suffix:bool=True,
                      overwrite_existing_non_csv:bool=False,
                      ):
        """
        Writes the results to a file. The file is saved in the directory specified by the directory_subpath.
        :param dataframe:
        :param data_type:
        :param directory_subpath:
        :param annotate_with_experiment_metadata:
        :param additional_annotation_dict:
        :param ignore_logging_level:
        :param table_name_prefix:
        :param table_name_suffix:
        :param append_experiment_set_suffix
        :return:
        """

        if isinstance(dataframe, pd.DataFrame):
            if dataframe.empty:
                # print("DataFrame is empty. Not writing to file.")
                return

        if not ignore_logging_level:
            if not verify_whether_to_log_data(logging_level=self.logging_level,
                                              data_type=data_type):
                # print("Data type", data_type, "is not being logged at the current logging level",)
                return

        if table_name_prefix is None:
            table_name_prefix = self.table_name_prefix
        if table_name_suffix is None:
            table_name_suffix = self.table_name_suffix

        if directory_subpath is None:
            directory_subpath = Path()

        directory_subpath = Path(directory_subpath) / self.get_subpath_of_data_type(data_type=data_type)

        annotations_dict = self._create_annotations_dict(annotate_with_ids=True,
                                                         annotate_with_experiment_metadata=annotate_with_experiment_metadata,
                                                         additional_annotation_dict=additional_annotation_dict)
        combined_suffix = table_name_suffix
        if append_experiment_set_suffix:
            # Add experiment_set_id as suffix right before data type
            experiment_set_suffix = self.metadata_manager.get_experiment_set_suffix()  # This creates "ExperimentSetName=value" format
            if combined_suffix:
                combined_suffix = self.join_table_name_parts([combined_suffix, experiment_set_suffix])
            else:
                combined_suffix = experiment_set_suffix


        if isinstance(dataframe,pd.DataFrame):
            format_type = 'dataframe'
        else:
            format_type = 'json'

        file_path = super().write_results(dataframe=dataframe,
                                          table_name=table_name,
                                          directory_subpath=directory_subpath,
                                          table_name_prefix=table_name_prefix,
                                          table_name_suffix=combined_suffix,
                                          data_type=data_type,
                                          df_annotations_dict=annotations_dict,
                                          format_type=format_type,
                                          overwrite_existing_non_csv=overwrite_existing_non_csv)

        # Track file history (merged from ResultsLoggerBase)
        self._files_history.append((data_type, file_path))

        return file_path

    def read_results(self,
                     table_name: Optional[str] = None,
                     data_type: SNDT = None,
                     directory_subpath: Optional[str | Path] = None,
                     # annotate_with_ids: bool = False,
                     experiment_instance_ids: Optional[List[str]] = None,
                     annotate_with_experiment_metadata=False,
                     additional_annotation_dict: Optional[dict] = None,
                     format_type: str = 'dataframe',
                     excluded_trials=None,
                     number_of_threads=1,
                     return_none_if_not_found: bool = False,
                     table_name_prefix: Optional[str | Path] = None,
                     table_name_suffix: Optional[str | Path] = None,
                     full_absolute_path_to_the_file: Optional[str] = None,
                     filter_by_experiment_set: bool = True,
                     append_experiment_set_suffix:bool=True
                     ):
        """
        Reads the results from a file. The file is read from the directory specified by the directory_subpath.
        :param table_name:
        :param data_type:
        :param directory_subpath:
        :param annotate_with_experiment_metadata:
        :param additional_annotation_dict:
        :param excluded_trials:
        :param number_of_threads:
        :param return_none_if_not_found:
        :param table_name_prefix:
        :param table_name_suffix:
        :return:
        """

        if full_absolute_path_to_the_file is not None:
            # If a full path is provided, we read directly from that path
            df_read = super().read_results(full_absolute_path_to_the_file=full_absolute_path_to_the_file,
                                           return_none_if_not_found=return_none_if_not_found)



        else:
            if directory_subpath is None:
                directory_subpath = Path()

            directory_subpath = Path(directory_subpath) / self.get_subpath_of_data_type(data_type=data_type)

            annotations_dict = self._create_annotations_dict(annotate_with_ids=False,
                                                             annotate_with_experiment_metadata=annotate_with_experiment_metadata,
                                                             additional_annotation_dict=additional_annotation_dict)

            if table_name_prefix is None:
                table_name_prefix = self.table_name_prefix
            if table_name_suffix is None:
                table_name_suffix = self.table_name_suffix
                
            combined_suffix = table_name_suffix
            if append_experiment_set_suffix:
                # Add experiment_set_id as suffix right before data type
                experiment_set_suffix = self.metadata_manager.get_experiment_set_suffix()  # This creates "ExperimentSetName=value" format
                if combined_suffix:
                    combined_suffix = self.join_table_name_parts([combined_suffix, experiment_set_suffix])
                else:
                    combined_suffix = experiment_set_suffix

            df_read = super().read_results(directory_subpath=directory_subpath,
                                           table_name=table_name,
                                           table_name_prefix=table_name_prefix,
                                           table_name_suffix=combined_suffix,
                                           data_type=data_type,
                                           df_annotations_dict=annotations_dict,
                                           format_type=format_type,
                                           excluded_trials=excluded_trials,
                                           number_of_threads=number_of_threads,
                                           return_none_if_not_found=return_none_if_not_found,
                                           )

        if df_read is None or df_read.empty:
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(f"No data found for data type {data_type} with table name {table_name} "
                                 f"in directory {directory_subpath}.")

        df_read[SNV.ExperimentInstanceID.id_long] = df_read[SNV.ExperimentInstanceID.id_long].astype(str)

        # Determine which experiment instance IDs to include
        valid_experiment_instance_ids = None
        if filter_by_experiment_set:
            tracking_data = self.read_experiment_set_tracking()
            if tracking_data is None or tracking_data.empty:
                error_msg = (f"ExperimentSetTracking data is empty or missing for experiment set "
                             f"'{self.experiment_set_name}' (ID: {self.experiment_set_id}). "
                             f"Cannot filter by experiment set without tracking data.")
                if return_none_if_not_found:
                    return None
                else:
                    raise ValueError(error_msg)

            # Extract experiment instance IDs from tracking data
            valid_experiment_instance_ids = set(tracking_data[SNV.ExperimentInstanceID.id_long].tolist())

        # If user provided specific experiment_instance_ids, intersect with valid ones
        if experiment_instance_ids is not None:
            if valid_experiment_instance_ids is not None:
                # Filter user-provided IDs to only include valid ones
                valid_experiment_instance_ids = valid_experiment_instance_ids.intersection(set(experiment_instance_ids))
            else:
                # No experiment set filtering, use user-provided IDs directly
                valid_experiment_instance_ids = set(experiment_instance_ids)

        # Apply filtering if we have valid experiment instance IDs
        if valid_experiment_instance_ids is not None:
            df_read = df_read[df_read[SNV.ExperimentInstanceID.id_long].isin(valid_experiment_instance_ids)]

        if df_read.empty:
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(f"No data found for experiment instance ID {valid_experiment_instance_ids} in "
                                 f"data type {data_type} with table name {table_name}.")

        return df_read

    def gather_results(self,
                       data_type: SNDT,
                       table_name_prefix: Optional[str] = None,
                       table_name_suffix: Optional[str] = None,
                       directory_subpath: Optional[str | Path] = None,
                       return_none_if_not_found: bool = False,
                       experiment_instance_ids: Optional[List[str]] = None,
                       filter_by_experiment_set: bool = True) -> Optional[pd.DataFrame]:
        """
        Gather and merge all pandas DataFrames of a specified data type from the storage directory.

        This method scans the data type directory, detects all CSV files matching the data type,
        reads them as pandas DataFrames, and concatenates them into a single merged DataFrame.
        Only supports pandas DataFrames stored as CSV files.

        By default, this method filters results to only include experiment instances that belong
        to the current experiment set (as determined by ExperimentSetTracking data). This ensures
        data integrity and prevents mixing results from different experiment sets.

        The merged DataFrame will include two additional columns:
        - 'SourceFileName': The name of the source file (e.g., 'results.csv')
        - 'SourceFilePath': The full absolute path to the source file

        :param data_type: The data type to gather (e.g., SNDT.VariationalParameters)
        :param table_name_prefix: Optional prefix filter for table names, defaults to self.table_name_prefix
        :param table_name_suffix: Optional suffix filter for table names, defaults to self.table_name_suffix
        :param directory_subpath: Optional subdirectory within data type folder
        :param return_none_if_not_found: If True, return None if no files found; if False, raise ValueError
        :param experiment_instance_ids: Optional list of specific experiment instance IDs to include.
                                       If provided, only these IDs will be gathered (intersected with
                                       experiment set IDs if filter_by_experiment_set=True)
        :param filter_by_experiment_set: If True (default), only gather results belonging to the current
                                        experiment set. If False, gather all files regardless of experiment set
        :return: Concatenated DataFrame or None if no files found and return_none_if_not_found=True
        :raises ValueError: If no files found and return_none_if_not_found=False, or if ExperimentSetTracking
                           data is missing when filter_by_experiment_set=True
        :raises TypeError: If attempting to gather non-DataFrame data (pickle files)
        """

        if data_type is None:
            data_type = SNDT.Unspecified

        # Construct the full directory path for this data type
        if directory_subpath is None:
            directory_subpath = Path()
        directory_subpath = Path(directory_subpath)

        data_type_subpath = self.get_subpath_of_data_type(data_type=data_type)
        full_directory = self.base_path / directory_subpath / data_type_subpath

        if not full_directory.exists():
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(f"Directory not found: {full_directory}")

        if table_name_prefix is None:
            table_name_prefix = self.table_name_prefix
        if table_name_suffix is None:
            table_name_suffix = self.table_name_suffix

        # Get data type suffix for file detection
        data_type_suffix = self.get_data_type_suffix(data_type=data_type)

        # Find all CSV files in the directory that match the data type
        csv_files = []
        for file_path in full_directory.glob("*.csv"):
            file_stem = file_path.stem

            # Parse the file name to check if it matches our data type
            try:
                parsed_parts = self.parse_table_name(full_table_name=file_stem,
                                                     name_parts_separator=self._tnps)
                #print(parsed_parts)


                # Check if the file has the correct data type suffix
                if data_type_suffix in parsed_parts:
                    # Apply prefix/suffix filters if provided
                    matches_filters = True
                    if table_name_prefix is not None and table_name_prefix != '':
                        if table_name_prefix not in parsed_parts:
                            matches_filters = False

                    if table_name_suffix is not None and table_name_suffix!= '':
                        if table_name_suffix not in parsed_parts:
                            matches_filters = False

                    if matches_filters:
                        csv_files.append(file_path)

            except Exception as e:
                # Skip files that don't follow the naming convention
                continue

        if not csv_files:
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(f"No CSV files found for data type {data_type.id_long} in {full_directory}")

        # Read and concatenate all DataFrames
        dataframes = []
        failed_files = []

        for file_path in csv_files:
            try:
                # Use read_results method with filtering parameters passed through
                df = self.read_results(full_absolute_path_to_the_file=file_path,
                                       return_none_if_not_found=True,
                                       experiment_instance_ids=experiment_instance_ids,
                                       filter_by_experiment_set=filter_by_experiment_set,
                                       )

                if df is not None and not df.empty:
                    # Add metadata about source file
                    df['SourceFileName'] = file_path.name
                    df['SourceFilePath'] = str(file_path)
                    dataframes.append(df)

            except Exception as e:
                failed_files.append((file_path.name, str(e)))
                continue

        if failed_files and len(dataframes) == 0:
            error_msg = f"Failed to read all files for data type {data_type.id_long}. Errors: {failed_files}"
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(error_msg)

        if not dataframes:
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(f"No valid DataFrames found for data type {data_type.id_long}")

        # Concatenate all DataFrames
        try:
            merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
            return merged_df

        except Exception as e:
            error_msg = (f"Failed to concatenate DataFrames for data type {data_type.id_long}. "
                         f"This may be due to incompatible column structures. Error: {str(e)}")
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(error_msg)


if __name__ == '__main__':
    # Example usage
    logger = ResultsLogger(experiment_set_name="TestExperiment1",
                           experiment_set_id="123456",
                           experiment_instance_id="67890v1",
                           directory_main="test_results/",
                           logging_level=LoggingLevel.BASIC)

    some_set_metadata = {'maybe_adding_this': np.array(['hehe', 'haha', 'hoho']),
                         'and_that': 42,
                         'pi': 3.14159,
                         'is_fun': True, }

    df = pd.DataFrame({"a": [1, 2, 8], "b": [4, 5, 6]})
    logger.write_results(dataframe=df,
                         table_name="TestTable",
                         data_type=SNDT.Unspecified)

    logger._write_shared_metadata(metadata_data_type=SNDT.Unspecified,
                                  shared_metadata=some_set_metadata,
                                  overwrite_existing=True
                                  )

    read_df = logger.read_results(data_type=SNDT.Unspecified,
                                  table_name="TestTable",
                                  experiment_instance_ids=["67890v1", "67890v2", "67890v3"])

    read_metadata = logger._read_shared_metadata(data_type=SNDT.Unspecified)

    read_tracking = logger.read_experiment_set_tracking()

    print("Read Tracking:")
    print(read_tracking)
    print("Read Metadata:")
    print(read_metadata)
    print("Read DataFrame:")
    print(read_df)
    print("HEJ")
    logger2 = logger.__class__(experiment_set_name="TestExperiment2",
                               experiment_set_id="12345",
                               experiment_instance_id="67890v4",
                               directory_main="test_results/",
                               logging_level=LoggingLevel.BASIC)

    some_set_metadata2 = {'maybe_adding_this': np.array(['hehe', 'haha', 'hoho']),
                          'and_that': 42,
                          'pi': 3.14159,
                          'is_fun': True, }

    df2 = pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})
    logger2._write_shared_metadata(metadata_data_type=SNDT.Unspecified,
                                   shared_metadata=some_set_metadata2,
                                   overwrite_existing=True
                                   )

    logger2.write_results(dataframe=df2,
                          data_type=SNDT.Unspecified,
                          table_name="TestTable")
    read_df2 = logger2.read_results(data_type=SNDT.Unspecified,
                                    table_name="TestTable",
                                    experiment_instance_ids=["67890v1", "67890v2", "67890v3", "67890v4"],
                                    )

    read_tracking2 = logger2.read_experiment_set_tracking()
    print("Read Tracking from logger2:")
    print(read_tracking2)
    print("Read Metadata from logger2:")
    read_metadata2 = logger2._read_shared_metadata(data_type=SNDT.Unspecified)
    print(read_metadata2)
    print("Read DataFrame from logger2:")
    print(read_df2)
