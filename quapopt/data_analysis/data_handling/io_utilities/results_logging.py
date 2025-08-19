# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 

from pathlib import Path
from typing import List, Optional, Union, Any

import numpy as np
import pandas as pd

from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.data_analysis.data_handling.io_utilities import DEFAULT_STORAGE_DIRECTORY
from quapopt.data_analysis.data_handling.io_utilities.logging_config import LoggingLevel, ExperimentLoggerConfig
from quapopt.data_analysis.data_handling.io_utilities.metadata_management import ExperimentSetMetadataManager
from quapopt.data_analysis.data_handling.io_utilities.standardized_io import (ResultsIO)
from quapopt.data_analysis.data_handling.standard_names import (STANDARD_NAMES_DATA_TYPES as SNDT,
                                                                STANDARD_NAMES_VARIABLES as SNV,
                                                                )
from quapopt.data_analysis.data_handling.standard_names.data_hierarchy import (
    DEFAULT_TABLE_NAME_PARTS_SEPARATOR,
    DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR
)
from quapopt.data_analysis.data_handling.standard_names.data_hierarchy import (StandardizedSpecifier,
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


class ResultsLoggerBase(ResultsIO):
    """
    Base class for logging results in a standardized way.
    It provides methods for writing and reading results to/from files,
    It offers a lot of flexibility.
    The file-naming convention is as follows:
    <table_name_prefix>;<table_name_prefix>;<table_name_prefix>dat=<data_type>.csv



    """

    def __init__(self,
                 table_name_prefix: Optional[str] = None,
                 table_name_suffix: Optional[str] = None,
                 directory_main: Optional[str | Path] = None,
                 default_storage_directory: Optional[str | Path] = DEFAULT_STORAGE_DIRECTORY,
                 table_name_parts_separator: str = DEFAULT_TABLE_NAME_PARTS_SEPARATOR,
                 dataframe_type_name_separator: str = DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR,
                 logging_level: LoggingLevel = LoggingLevel.BASIC,
                 ):
        """
        Initializes the ResultsLoggerBase class.

        The full main path for storing results is
        <default_storage_directory>/<base_path>/


        :param table_name_prefix:
        :param table_name_suffix:
        :param directory_main:
        The main directory where the results will be stored.
        defaults to empty string
        :param default_storage_directory:
        The default storage directory for all data.
        Defaults to whatever is set in DEFAULT_STORAGE_DIRECTORY.

        :param table_name_parts_separator:
        The separator to be used between parts of the table name.
        Helps to differentiate between different parts of the table name.
        :param dataframe_type_name_separator:
        The separator to be used between the column name and the data type in the dataframe column names.
        :param logging_level:
        The logging level to be used for the results logger.
        """

        super().__init__(directory_main=directory_main,
                         table_name_prefix=table_name_prefix,
                         table_name_suffix=table_name_suffix,
                         default_storage_directory=default_storage_directory,
                         table_name_parts_separator=table_name_parts_separator,
                         dataframe_type_name_separator=dataframe_type_name_separator)

        self._logging_level = logging_level
        self._files_history = []

    def set_logging_level(self, level):
        self._logging_level = level

    @property
    def logging_level(self):
        return self._logging_level

    @property
    def files_history(self):
        return self._files_history

    def write_results(self,
                      dataframe: Union[pd.DataFrame, np.ndarray],
                      directory_subpath: Optional[str | Path] = None,
                      table_name: Optional[str] = None,
                      table_name_prefix: Optional[str | Path] = None,
                      table_name_suffix: Optional[str | Path] = None,
                      data_type: Optional[SNDT] = None,
                      df_annotations_dict: Optional[dict] = None,
                      format_type: str = 'dataframe',
                      ) -> Path:
        """
        Writes the results to a file. The file is saved in the directory specified by the directory_subpath

        The full path will be something like:
        <default_storage_directory>/<base_path>/<directory_subpath>/<directory_datatype>/FullTableName.csv


        :param dataframe:
        Data to be saved
        :param directory_subpath:
        The subpath within the main directory where the file will be saved.
        If None, it will be set to the standard sub-path for the given data type.
        If provided, we anyway append the standard sub-path for the given data type.

        :param table_name:
        The name of the table to be saved. If None, it will be set to empty string
        :param table_name_prefix:
        The prefix to be added to the table name. If None, it will be set to self.
        :param table_name_suffix:
        The suffix to be added to the table name. If None, it will be set to self.
        :param data_type:
        The data type of the results to be saved. If None, it will be set to SNDT.Unspecified.
        :param df_annotations_dict:
        A dictionary containing additional annotations to be added to the DataFrame.
        :param as_pickle:
        If True, the results will be saved as a pickle file. If False, they will be saved as a CSV file.
        NOTE: this is not recommended, as it may lead to issues with data types and serialization.
        :param id_logger:
        The ID of the logger. If None, it will not be added to the annotation dictionary.
        If provided, it will be added to the annotation dictionary under the key "IDLogger"
        :param id_logging_session:
        The ID of the logging session. If None, it will not be added to the annotation dictionary.
        If provided, it will be added to the annotation dictionary under the key "IDLoggingSession"


        :return:
        """

        if directory_subpath is None:
            directory_subpath = Path()

        directory_subpath = Path(directory_subpath)

        id_key = SNV.ID.id_long
        if df_annotations_dict is None:
            df_annotations_dict = {}

        # TODO FBM: think about it. Maybe it is better to save as pickle if the data type is unspecified
        if data_type is None:
            format_type = 'pickle'

        file_path = super().write_results(dataframe=dataframe,
                                          directory_subpath=directory_subpath,
                                          table_name=table_name,
                                          table_name_prefix=table_name_prefix,
                                          table_name_suffix=table_name_suffix,
                                          data_type=data_type,
                                          df_annotations_dict=df_annotations_dict,
                                          format_type=format_type)

        self._files_history.append((data_type, file_path))

        return file_path

    def read_results(self,
                     directory_subpath: Optional[str | Path] = None,
                     table_name: Optional[str] = None,
                     table_name_prefix: Optional[str | Path] = None,
                     table_name_suffix: Optional[str | Path] = None,
                     data_type: SNDT = None,
                     df_annotation: dict = None,
                     format_type: str = 'dataframe',
                     excluded_trials=None,
                     number_of_threads=1,
                     return_none_if_not_found: bool = False, ):
        """
        Reads the results from a file. The file is read from the directory specified by the directory_subpath.

        The full path will be something like:
        <default_storage_directory>/<base_path>/<directory_subpath>/<directory_datatype>/FullTableName.csv




        :param directory_subpath:
        :param table_name:
        :param table_name_prefix:
        :param table_name_suffix:
        :param data_type:
        :param df_annotation:
        :param as_pickle:
        :param excluded_trials:
        :param number_of_threads:
        :param return_none_if_not_found:
        :return:
        """

        if directory_subpath is None:
            directory_subpath = Path()

        directory_subpath = Path(directory_subpath)

        # if data_type is not None:
        # directory_subpath = directory_subpath / self.get_standard_sub_path_of_data_type(data_type=data_type)

        return super().read_results(directory_subpath=directory_subpath,
                                    table_name=table_name,
                                    table_name_prefix=table_name_prefix,
                                    table_name_suffix=table_name_suffix,
                                    data_type=data_type,
                                    df_annotations_dict=df_annotation,
                                    format_type=format_type,
                                    excluded_trials=excluded_trials,
                                    number_of_threads=number_of_threads,
                                    return_none_if_not_found=return_none_if_not_found)


class ResultsLogger(ResultsLoggerBase):
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
                         dataframe_type_name_separator=DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR,
                         logging_level=logging_level)

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

    def read_shared_metadata(self,
                             data_type: SNDT):
        return self.metadata_manager.read_shared_metadata(metadata_data_type=data_type)

    def write_shared_metadata(self,
                              metadata_data_type: SNDT,
                              shared_metadata: Any,
                              overwrite_existing: bool = False) -> Path:
        return self.metadata_manager.write_shared_metadata(metadata_data_type=metadata_data_type,
                                                           shared_metadata=shared_metadata,
                                                           overwrite_existing=overwrite_existing)

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
                      ):
        """
        Writes the results to a file. The file is saved in the directory specified by the directory_subpath.
        :param dataframe:
        :param data_type:
        :param directory_subpath:
        :param annotate_with_ids:
        :param annotate_with_experiment_metadata:
        :param additional_annotation_dict:
        :param ignore_logging_level:
        :param table_name_prefix:
        :param table_name_suffix:
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

        return super().write_results(dataframe=dataframe,
                                     table_name=table_name,
                                     directory_subpath=directory_subpath,
                                     table_name_prefix=table_name_prefix,
                                     table_name_suffix=table_name_suffix,
                                     data_type=data_type,
                                     df_annotations_dict=annotations_dict,
                                     format_type='dataframe')

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
                     ):
        """
        Reads the results from a file. The file is read from the directory specified by the directory_subpath.
        :param table_name:
        :param data_type:
        :param directory_subpath:
        :param annotate_with_ids:
        :param annotate_with_experiment_metadata:
        :param additional_annotation_dict:
        :param as_pickle:
        :param excluded_trials:
        :param number_of_threads:
        :param return_none_if_not_found:
        :param table_name_prefix:
        :param table_name_suffix:
        :return:
        """

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

        df_read = super().read_results(directory_subpath=directory_subpath,
                                       table_name=table_name,
                                       table_name_prefix=table_name_prefix,
                                       table_name_suffix=table_name_suffix,
                                       data_type=data_type,
                                       df_annotation=annotations_dict,
                                       format_type=format_type,
                                       excluded_trials=excluded_trials,
                                       number_of_threads=number_of_threads,
                                       return_none_if_not_found=return_none_if_not_found)

        if df_read is None:
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(f"No data found for data type {data_type} with table name {table_name}.")

        df_read[SNV.ExperimentInstanceID.id_long] = df_read[SNV.ExperimentInstanceID.id_long].astype(str)

        if experiment_instance_ids is None:
            experiment_instance_ids = [self.experiment_instance_id]

        df_read = df_read[df_read[SNV.ExperimentInstanceID.id_long].isin(experiment_instance_ids)]
        if df_read.empty:
            if return_none_if_not_found:
                return None
            else:
                raise ValueError(f"No data found for experiment instance ID {experiment_instance_ids} in "
                                 f"data type {data_type} with table name {table_name}.")

        return df_read


if __name__ == '__main__':
    # Example usage
    logger = ResultsLogger(experiment_set_name="TestExperiment2",
                           experiment_set_id="12345",
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

    logger.write_shared_metadata(metadata_data_type=SNDT.Unspecified,
                                 shared_metadata=some_set_metadata,
                                 overwrite_existing=True
                                 )

    read_df = logger.read_results(data_type=SNDT.Unspecified,
                                  table_name="TestTable",
                                  experiment_instance_ids=["67890v1", "67890v2", "67890v3"])

    read_metadata = logger.read_shared_metadata(data_type=SNDT.Unspecified)

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
    logger2.write_results(dataframe=df,
                          data_type=SNDT.Unspecified,
                          table_name="TestTable")
    read_df2 = logger2.read_results(data_type=SNDT.Unspecified,
                                    table_name="TestTable",
                                    experiment_instance_ids=["67890v1", "67890v2", "67890v3", "67890v4"])

    read_tracking2 = logger.read_experiment_set_tracking()
    print("Read Tracking from logger2:")
    print(read_tracking2)
    print("Read Metadata from logger2:")
    read_metadata2 = logger2.read_shared_metadata(data_type=SNDT.Unspecified)
    print(read_metadata2)
    print("Read DataFrame from logger2:")
    print(read_df2)
