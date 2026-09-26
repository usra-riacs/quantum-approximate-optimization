# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import copy
import csv
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Optional, List, Union, Any, Dict, Tuple
from enum import Enum
import numpy as np
import pandas as pd

from quapopt import ancillary_functions as anf
from quapopt.data_analysis.data_handling.schemas.naming import (
    DEFAULT_TABLE_NAME_PARTS_SEPARATOR,
    DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR,
    STANDARD_NAMES_DATA_TYPES as SNDT,
    BaseNameDataType,
    STANDARD_NAMES_VARIABLES as SNV,
    MAIN_KEY_VALUE_SEPARATOR as MKVS,
    MAIN_KEY_SEPARATOR as MKS,
    SUB_KEY_VALUE_SEPARATOR as SKVS,
    HamiltonianInstanceSpecifierGeneral,
    HamiltonianClassSpecifierGeneral
)
from quapopt.data_analysis.data_handling.io_utilities import DEFAULT_STORAGE_DIRECTORY
from quapopt.data_analysis.data_handling.schemas.parsing import parse_description_string


SUPPORTED_DATATYPES_BASIC = ['float', 'float16', 'float32', 'float64',
                             'int', 'int8', 'int16', 'int32', 'int64',
                             'str', 'bool', 'uint', 'uint8', 'uint16', 'uint32', 'uint64',
                             ]
SUPPORTED_DATATYPES_COMPLEX = ['list', 'tuple',
                               'dict', 'set',
                               'ndarray','np.ndarray', 'None']


SUPPORTED_DATATYPES = SUPPORTED_DATATYPES_BASIC + SUPPORTED_DATATYPES_COMPLEX

INTEGER_LISTLIKE_TYPES = [SNV.Bitflip.id, SNV.Bitflip.id_long,
                          SNV.Bitstring.id, SNV.Bitstring.id_long,
                          SNV.Permutation.id, SNV.Permutation.id_long,
                          ]
FLOAT_LISTLIKE_TYPES = [SNV.Angles.id, SNV.Angles.id_long]

def _string_mod_fun1(string: str | Path,
                     suffix: str):
    input_format = type(string)
    string = str(string)

    if not suffix.startswith('.'):
        proper_suffix = f'.{suffix}'
    else:
        proper_suffix = suffix
    return input_format, string, proper_suffix


def _string_mod_fun2(string: str | Path,
                     input_format: type):
    if issubclass(input_format, Path):
        return Path(string)
    elif input_format == str:
        return string
    else:
        raise TypeError(f"Unsupported input type: {input_format}. Expected str or Path.")


# def with_suffix_patch(string:str|Path,
#                       suffix:str)->Path|str:
#     input_format, string, proper_suffix = _string_mod_fun1(string=string,
#                                                             suffix=suffix)
#     l = len(proper_suffix)
#
#     if string.endswith(proper_suffix):
#         string = string[:-l]
#     else:
#         string = string + proper_suffix
#
#     return _string_mod_fun2(string=string,
#                             input_format=input_format)

def add_file_format_suffix(string: str | Path,
                           suffix: str) -> Path:
    """
    A patch for the Path.with_suffix method to ensure it works correctly with strings that contain floats (i.e., they
    include dot characters so the Path.with_suffix method does not work correctly).
    :param string:
    :param suffix:
    :return:
    """

    input_format, string, proper_suffix = _string_mod_fun1(string=string,
                                                           suffix=suffix)

    if not string.endswith(proper_suffix):
        string = string + proper_suffix

    return _string_mod_fun2(string=string,
                            input_format=input_format)


def remove_file_format_suffix(string: str | Path,
                              suffix: str):
    input_format, string, proper_suffix = _string_mod_fun1(string=string,
                                                           suffix=suffix)

    if string.endswith(proper_suffix):
        string = string[:-len(proper_suffix)]

    return _string_mod_fun2(string=string,
                            input_format=input_format)


def serialize_for_logging(obj: Any, max_depth: int = 10) -> Any:
    """
    Recursively convert objects to JSON-serializable format.

    Handles:
    - Enums → .value (or .name if value not serializable)
    - numpy/cupy arrays → list
    - numpy scalars → python scalars
    - Callables → "<callable: name>" or skip
    - Objects with get_description_string() → call it
    - Objects with __class__ → "<ClassName>"
    - Dicts/lists/tuples → recursive
    - Primitives → pass through
    """
    if max_depth <= 0:
        return "<max_depth_exceeded>"

    # Primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Enums
    if isinstance(obj, Enum):
        return obj.value if isinstance(obj.value, (str, int, float)) else obj.name

    # Numpy/CuPy scalars
    if hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass

    # Arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'get'):  # CuPy array
        try:
            return obj.get().tolist()
        except:
            pass

    # Tuples/Lists
    if isinstance(obj, (list, tuple)):
        serialized = [serialize_for_logging(item, max_depth - 1) for item in obj]
        return tuple(serialized) if isinstance(obj, tuple) else serialized

    # Dicts
    if isinstance(obj, dict):
        return {
            str(k): serialize_for_logging(v, max_depth - 1)
            for k, v in obj.items()
        }

    # Callables (lambdas, functions)
    if callable(obj):
        name = getattr(obj, '__name__', 'anonymous')
        return f"<callable:{name}>"

    # Objects with description method
    if hasattr(obj, 'get_description_string'):
        return obj.get_description_string()

    # Context managers, backends, complex objects → class name
    return f"<{obj.__class__.__name__}>"
class IOMixin:
    """Mixin providing common path and filename generation utilities."""

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def get_key_value_pair(cls,
                           key_id: str,
                           value: str,
                           major=True) -> str:
        """Generate a standardized prefix for metadata files."""

        if major:
            return f"{key_id}{MKVS}{value}"
        else:
            return f"{key_id}{SKVS}{value}"

    @classmethod
    def get_data_type_suffix(cls,
                             data_type: BaseNameDataType) -> str:
        """Get the standardized suffix for a given data type."""
        return cls.get_key_value_pair(key_id=SNDT.DataType.id,
                                      value=data_type.id)

    @classmethod
    def get_default_storage_directory(cls,
                                      default_storage_directory: Optional[
                                          str | Path] = DEFAULT_STORAGE_DIRECTORY) -> Path:
        """
        Get the default storage directory for all data.
        It is a folder where ALL data is stored, including hamiltonians, results, etc.
        The default is "DEFAULT_STORAGE_DIRECTORY".
        :param default_storage_directory:
        If None, the current working directory is used.
        :return:
        """

        if default_storage_directory is None:
            return Path.cwd()
        else:
            return Path(default_storage_directory)

    @classmethod
    def construct_base_path(cls,
                            directory_main: Optional[str | Path] = None,
                            default_storage_directory: Optional[str | Path] = DEFAULT_STORAGE_DIRECTORY) -> Path:

        default_storage_directory = default_storage_directory

        if default_storage_directory is None:
            if directory_main is None:
                directory_main = Path().cwd()
            default_storage_directory = Path("")
        else:
            if directory_main is None:
                directory_main = Path("output")

        if directory_main == Path().cwd():
            print("Warning: 'base_path' is set to the current working directory, but "
                  "'default_storage_directory' is provided. We ignore 'default_storage_directory' "
                  "and use cwd as the main directory.")
            default_storage_directory = Path()



        default_storage_directory = Path(default_storage_directory)
        directory_main = Path(directory_main)
        base_path = default_storage_directory / directory_main


        return base_path

    @classmethod
    def get_hamiltonian_data_base_path(cls,
                                       storage_directory: Optional[str | Path] = DEFAULT_STORAGE_DIRECTORY) -> Path:
        """
        Get the main path for hamiltonian data storage.
        :param storage_directory:
        :return:
        """
        path_main = cls.get_default_storage_directory(default_storage_directory=storage_directory)
        return path_main / "HamiltonianData" / "Hamiltonians"

    @classmethod
    def get_hamiltonian_class_base_path(cls,
                                        hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral|str,
                                        storage_directory: Optional[
                                            str | Path] = DEFAULT_STORAGE_DIRECTORY) -> Path:
        if hamiltonian_class_specifier is None:
            raise ValueError("Hamiltonian class specifier is None.")
        if isinstance(hamiltonian_class_specifier,str):
            hamiltonian_class_description = hamiltonian_class_specifier
        else:
            hamiltonian_class_description = hamiltonian_class_specifier.get_description_string()
        path_full = cls.get_hamiltonian_data_base_path(storage_directory=storage_directory)
        path_full = path_full / hamiltonian_class_description
        path_full.mkdir(parents=True, exist_ok=True)
        return path_full

    @classmethod
    def get_hamiltonian_instance_filename(cls,
                                          hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral) -> str:
        """
        Get the filename for a hamiltonian instance based on its specifier.
        In this case, it's just a wrapper for another function, but maybe in the future it will be more complex.
        :param hamiltonian_instance_specifier:
        :return:
        """

        assert hamiltonian_instance_specifier is not None, "Hamiltonian instance specifier is None."
        return hamiltonian_instance_specifier.get_description_string()

    @classmethod
    def get_subpath_of_data_type(cls,
                                 data_type: BaseNameDataType) -> Path:

        if data_type is None:
            data_type = SNDT.Unspecified

        data_category = data_type.data_category
        data_subcategory = data_type.data_subcategory

        return Path() / data_category / data_subcategory / data_type.id_long

    def get_absolute_path_of_data_type(self,
                                       data_type: BaseNameDataType) -> Path:
        """
        Get the absolute path for a given data type.
        :param data_type:
        :return:
        """
        return self.construct_base_path() / self.get_subpath_of_data_type(data_type=data_type)

    @classmethod
    def parse_table_name(cls,
                         full_table_name: str,
                         name_parts_separator: str = DEFAULT_TABLE_NAME_PARTS_SEPARATOR):
        """

        :param full_table_name:
        :param name_parts_separator:
        :return:
        """

        # We wish to decompose full_table_name into its components:
        # 1. prefix_table_name
        # 2. table_name
        # 3. table_name_suffix
        # 4. data_type_suffix

        # note that some compontents might not be present.
        # User is supposed to know whether given parts of table were present or not

        return full_table_name.split(name_parts_separator)
    @staticmethod
    def parse_file_name_to_dict(file_name:str,
                                return_base_names:bool=False):

        return parse_description_string(description=file_name,
                                        return_base_names=return_base_names)


    @classmethod
    def update_dataframe_with_variables_from_string(cls,
                                           file_name:str,
                                           df:pd.DataFrame,
                                            overwrite_existing:bool=False
                                            ):
        variable_values_dict = cls.parse_file_name_to_dict(file_name=file_name,
                                                            return_base_names=True)

        columns_disambiugous = [s.lower() for s in df.columns]

        for original_name, (column_value, base_name) in variable_values_dict.items():
            if base_name is None:
                variable_id = original_name
                variable_format = None
            else:
                variable_id = base_name.id_long
                variable_format = base_name.value_format

            if variable_format is None:
                variable_format = 'Unknown'
            else:
                variable_format = variable_format.__name__

            column_name = f"{variable_id}|{variable_format}"

            if overwrite_existing:
                df[column_name] = column_value
            elif variable_id.lower() not in columns_disambiugous and column_name.lower() not in columns_disambiugous:
                #print(variable_id, 'adding')
                df[column_name] = column_value


        return df


    @classmethod
    def parse_file_type_suffix(cls,
                               file_name: str | Path, ):
        """
        Parse the file name to extract the stem and suffix.
        :param file_name:
        :return:
        """
        file_name = Path(file_name)
        return file_name.stem, file_name.suffix

    @classmethod
    def join_table_name_parts(cls,
                              table_name_parts: List[Optional[str]],
                              name_parts_separator: str = DEFAULT_TABLE_NAME_PARTS_SEPARATOR):

        joined_name = name_parts_separator.join(
            [part for part in table_name_parts if (part is not None and part != "")]) if table_name_parts else ""

        #print('hejunia',joined_name, table_name_parts)


        return joined_name

    @classmethod
    def get_full_table_name(cls,
                            table_name_parts: List[Optional[str]],
                            data_type: Optional[BaseNameDataType] = None,
                            name_parts_separator: str = DEFAULT_TABLE_NAME_PARTS_SEPARATOR):

        data_type_suffix = cls.get_data_type_suffix(data_type=data_type)
        if data_type_suffix not in table_name_parts:
            table_name_parts.append(data_type_suffix)

        return cls.join_table_name_parts(table_name_parts=table_name_parts,
                                         name_parts_separator=name_parts_separator)

    @classmethod
    def write_pickled_results(cls,
                              object_to_save: Any,
                              file_path: str | Path,
                              add_timestamp_if_exists: bool = True,
                              overwrite_if_exists: bool = False) -> None:
        """
        Save an object to a pickle file at the specified file path.
        If file exists, the default behavior is to add a timestamp to the file name.
        If `overwrite_if_exists` is set to True, the existing file will be overwritten instead

        :param object_to_save:
        :param file_path:
        :param add_timestamp_if_exists:
        If True, a timestamp will be added to the file name if the file already exists.
        :param overwrite_if_exists:
        If True, the existing file will be overwritten.
        :return:
        """

        file_path = Path(file_path)
        file_path = remove_file_format_suffix(string=file_path,
                                              suffix='.pkl')

        if file_path.exists():
            if overwrite_if_exists:
                file_path.unlink()  # Remove the existing file
            elif add_timestamp_if_exists:
                file_path = file_path + Path(f"Time{MKVS}{time.strftime(f'%Y-%m-%d-%H-%M-%S')}")
            else:
                raise FileExistsError(f"File {file_path} already exists. "
                                      f"Use 'overwrite_if_exists' or 'add_timestamp_if_exists' to handle this.")

        file_path = add_file_format_suffix(string=file_path, suffix=".pkl")
        with open(f"{file_path}", 'wb') as f:
            pickle.dump(object_to_save, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def write_json_results(cls,
                           data: Any,
                           full_path: str | Path,
                           overwrite_existing: bool = False,
                           add_timestamp_if_exists: bool = True):
        """
        Save metadata to a JSON file in a standardized format.
        :param data:
        :param full_path:
        :param overwrite_existing:
        :param add_timestamp_if_exists:

        """
        import json
        full_path = Path(full_path)

        full_path = add_file_format_suffix(string=full_path, suffix='.json')

        if full_path.exists():
            if not overwrite_existing:
                raise ValueError(f"Metadata file already exists: {full_path}")
            elif overwrite_existing:
                # Remove existing file if overwrite is requested
                full_path.unlink()
            elif add_timestamp_if_exists:
                full_path = full_path.with_name(
                    f"{full_path.stem}_Time{MKVS}{time.strftime(f'%Y-%m-%d-%H-%M-%S')}{full_path.suffix}")

        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare metadata for JSON serialization
        if isinstance(data, pd.DataFrame):
            metadata_to_save = {
                '_type': 'pandas_dataframe',
                'data': data.to_dict('records'),
                'columns': list(data.columns)
            }
        else:
            metadata_to_save = data

        # Save to JSON
        with open(full_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2, default=str)

        return full_path


    @classmethod
    def write_numpy_results(cls,
                           data: np.ndarray,
                           full_path: str | Path,
                            overwrite_existing: bool = False,
                            add_timestamp_if_exists: bool = True):
        """
        Save metadata to a JSON file in a standardized format.
        :param data:
        :param full_path:

        """
        import numpy as np


        full_path = Path(full_path)
        full_path = add_file_format_suffix(string=full_path, suffix='.npy')

        if full_path.exists():
            if not overwrite_existing:
                raise ValueError(f"Metadata file already exists: {full_path}")
            elif overwrite_existing:
                # Remove existing file if overwrite is requested
                full_path.unlink()
            elif add_timestamp_if_exists:
                full_path = full_path.with_name(
                    f"{full_path.stem}_Time{MKVS}{time.strftime(f'%Y-%m-%d-%H-%M-%S')}{full_path.suffix}")

        full_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(full_path, data)

        return full_path


    @classmethod
    def write_dataframe_results(cls,
                                data: pd.DataFrame,
                                full_path: str | Path):
        """
        Save a pandas DataFrame to a CSV file in a standardized format.
        :param data:
        :param full_path:
        :return:
        """
        # print('timestop 3', full_path)
        full_path = Path(full_path)

        full_path = add_file_format_suffix(string=full_path, suffix='.csv')

        # print('timestop 4', full_path)

        full_path.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(str(full_path),
                    index=False,
                    mode='a',
                    header=not os.path.exists(full_path),
                    quoting=csv.QUOTE_NONNUMERIC)

        return full_path

    @classmethod
    def write_results(cls,
                      data: Union[pd.DataFrame, np.ndarray, Any],
                      full_path: str | Path,
                      format_type: str = 'dataframe',
                      overwrite_existing_non_csv: bool = False,
                      add_timestamp_if_exists: bool = True
                      ):

        # print('timestop 2', full_path)


        if format_type.lower() in ['dataframe']:
            return cls.write_dataframe_results(data=data,
                                               full_path=full_path)
        elif format_type.lower() in ['pickle']:
            return cls.write_pickled_results(object_to_save=data,
                                             file_path=full_path,
                                             overwrite_if_exists=overwrite_existing_non_csv)
        elif format_type.lower() in ['json']:
            return cls.write_json_results(data=data,
                                          full_path=full_path,
                                          overwrite_existing=overwrite_existing_non_csv,
                                          add_timestamp_if_exists=True)
        elif format_type.lower() in ['numpy']:
            return cls.write_numpy_results(data=data,
                                             full_path=full_path,
                                           overwrite_existing=overwrite_existing_non_csv,
                                           add_timestamp_if_exists=True)

        else:
            raise ValueError(f"Unsupported format_type: {format_type}. Supported types are 'dataframe' and 'pickle'.")

    @classmethod
    def read_pickled_results(cls,
                             file_path: str | Path,
                             return_none_if_not_found=False) -> Any:
        """
          Read an object from a pickle file at the specified file path.
          :param file_path:
          :param return_none_if_not_found:
          :return:
          """

        file_path = add_file_format_suffix(string=file_path, suffix=".pkl")

        try:
            with open(file_path, 'rb') as f:
                object_read = pickle.load(f)
            return object_read
        except(FileNotFoundError) as e:
            if return_none_if_not_found:
                return None
            else:
                raise e

    @classmethod
    def _convert_value_basic(cls,
                             column_series: pd.DataFrame,
                             type_name: str, ):
        """
        Convert a pandas Series to a specific basic data type.
        :param column_series:
        :param type_name:
        :return:
        """

        if type_name not in SUPPORTED_DATATYPES_BASIC:
            return column_series

        if type_name == 'float':
            return column_series.astype(float)
        elif type_name == 'float64':
            return column_series.astype(np.float64)
        elif type_name == 'float32':
            return column_series.astype(np.float32)
        elif type_name == 'float16':
            return column_series.astype(np.float16)
        elif type_name == 'int':
            return column_series.astype(int)
        elif type_name == 'int8':
            return column_series.astype(np.int8)
        elif type_name == 'int16':
            return column_series.astype(np.int16)
        elif type_name == 'int32':
            return column_series.astype(np.int32)
        elif type_name == 'int64':
            return column_series.astype(np.int64)
        elif type_name == 'str':
            return column_series.astype(str)
        elif type_name == 'bool':
            return column_series.astype(bool)

    @classmethod
    def annotate_dataframe(cls,
                           dataframe: pd.DataFrame,
                           annotation: Dict[str, Any]):
        """
        Annotate a pandas DataFrame with new columns based on the provided annotation dictionary.
        :param dataframe:
        :param annotation:
        :return:
        """

        if dataframe is None:
            return dataframe

        dataframe = dataframe.copy()

        # First check if columns are not present:
        for key in annotation:
            if key in dataframe.columns:
                raise ValueError(f"Column '{key}' already present in dataframe.")
        # Add new columns and annotate ALL rows:
        for key in annotation:
            dataframe[key] = [annotation[key]]*len(dataframe)
        return dataframe

    @classmethod
    def _df_rename_datatypes_columns(cls,
                                     dataframe: pd.DataFrame):
        dataframe = dataframe.copy()
        columns_renamed = {}
        for col in dataframe.columns:
            # take first , middle, and last value for each column and infer type
            value_indices = [0, len(dataframe) // 2, -1]
            values_col = dataframe[col].iloc[value_indices]
            types_col = [type(value) for value in values_col]

            # if data_type is provided, use it
            if len(set(types_col)) == 1:
                column_type_detected = types_col[0]
                column_type_detected = column_type_detected.__name__
                if column_type_detected in SUPPORTED_DATATYPES:
                    col_type = column_type_detected
                else:
                    col_type = 'Unknown'
            # print(types_col, column_type_detected, col_type)
            #
            else:
                col_type = 'Unknown'

            col_new_name = f'{col}{DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR}{col_type}'
            columns_renamed[col] = col_new_name
        dataframe_renamed = dataframe.rename(columns=columns_renamed)
        return dataframe_renamed


    @classmethod
    def evaluate_column_values(cls,
                               df_read:pd.DataFrame,
                               column_name:str,
                               column_type:str,
                               column_name_in_df:str,
                               number_of_threads:int=1
                               ):




        if column_type is None:
            for _, base_name in SNV.get_all_attributes().items():
                id_short, id_long = base_name.id, base_name.id_long
                if column_name == id_short or column_name == id_long:

                    if base_name.value_format is None:
                        column_type = 'str'
                    else:
                        column_type = base_name.value_format.__name__

                    break

        if column_type is None:
            column_type = 'Unknown'


        def _parallelize_function(fun):
            def __parallelized_function(x):
                return anf.df_column_apply_function_parallelized(
                    series=x,
                    function_to_apply=fun,
                    number_of_threads=number_of_threads)
            return __parallelized_function

        _convert_function_alternative = None
        _convert_function = None
        if column_type == 'Unknown':
            pass
        elif column_type in SUPPORTED_DATATYPES_BASIC:
            _convert_function = lambda x: cls._convert_value_basic(x, column_type)
        elif column_type in SUPPORTED_DATATYPES_COMPLEX:
            if column_name in INTEGER_LISTLIKE_TYPES:
                output_numbers_type = int
            else:
                output_numbers_type = float





            if column_type in ['tuple', 'list', 'ndarray', 'np.ndarray']:
                _convert_function = _parallelize_function(anf.get_eval_string_listlike_function(output_container_type=column_type,
                                                                                                input_numbers_type=int,
                                                                                                output_numbers_type=output_numbers_type
                                                                                                ))


                if column_type not in ['ndarray', 'np.ndarray']:
                    _convert_function_alternative = _parallelize_function(anf.get_eval_string_listlike_function(output_container_type=column_type,
                                                                                                    input_numbers_type=float,
                                                                                                    output_numbers_type=output_numbers_type
                                                                                                    ))
            else:
                _convert_function = lambda x:eval(x)

        else:
            raise TypeError(f"Unsupported type: {column_type} (of type {type(column_type)}).")

        try:
            df_read[column_name_in_df] = _convert_function(df_read[column_name_in_df])
        except(ValueError, TypeError) as e:

            if _convert_function_alternative is not None:
                try:
                    df_read[column_name_in_df] = _convert_function_alternative(df_read[column_name_in_df])
                except(ValueError) as e:
                    return df_read

        return df_read


    @classmethod
    def _post_process_dataframe_results(cls,
                                        df_read:pd.DataFrame,
                                        number_of_threads:int=1,
                                        type_separator=DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR,

                                        ):

        columns_renamed, columns_types = {}, {}
        column_names_all = df_read.columns
        for col_name_full in column_names_all:
            split_type = col_name_full.split(type_separator)
            if len(split_type) == 2:
                col_name, col_type = col_name_full.split(type_separator)
            elif len(split_type) == 1:
                col_name = col_name_full
                col_type = None

            #Whoops
            elif col_name_full in ['|0...0> Energy|float']:
                col_name = '|0...0> Energy'
                col_type = 'float'
            else:
                raise ValueError(f"Column name '{col_name_full}' is not properly formatted.")

            try:
                df_read = cls.evaluate_column_values(df_read=df_read,
                                                     column_name=col_name,
                                                     column_type=col_type,
                                                     column_name_in_df=col_name_full,
                                                     number_of_threads=number_of_threads)
                columns_renamed[col_name_full] = col_name

            except(ValueError) as val_err:
                print("ERROR READING COLUMN:", col_name_full)
                print("MESSAGE:", val_err)
                columns_renamed[col_name_full] = col_name_full

        df_read = df_read.rename(columns=columns_renamed)

        return df_read


    # @classmethod
    # def parse_table


    @classmethod
    def read_pandas_dataframe(cls,
                              full_path: str | Path,
                              number_of_threads=1,
                              type_separator=DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR,
                              return_none_if_not_found=False,
                              post_process_dataframe:bool=True,
                              float_precision: Optional[str] = None):
        """
        Read a pandas DataFrame from a file in a standardized format.
        :param full_path:
        The full path to the file where results are stored.
        :param number_of_threads:
        The number of threads to use for parallel processing.
        :param type_separator:
        The separator used in the column names to separate the name and the data type.
        :param return_none_if_not_found:
        If True, the function will return None if the file is not found.
        If False, it will raise a FileNotFoundError.
        :return:
        """

        full_path = add_file_format_suffix(string=full_path, suffix='.csv')


        t0 = time.perf_counter()
        try:
            # float_precision='round_trip' returns the exact float64 that was written, at
            # the cost of a slower parse. The default parser is off by up to one ULP on
            # some values, which is invisible in analysis and fatal to an equality test.
            df_read = pd.read_csv(str(full_path), float_precision=float_precision)
        except(FileNotFoundError) as e:
            if return_none_if_not_found:
                return None
            else:
                raise e

        if not post_process_dataframe:
            return df_read


        t1 = time.perf_counter()

        df_read = cls._post_process_dataframe_results(df_read=df_read,
                                                      number_of_threads=number_of_threads,
                                                      type_separator=type_separator)

        t2= time.perf_counter()


        return df_read

    @classmethod
    def read_results(cls,
                     full_path: str | Path,
                     format_type='dataframe',
                     df_annotations_dict: dict = None,
                     excluded_trials=None,
                     name_type_separator=DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR,
                     return_none_if_not_found=False,
                     number_of_threads=1,
                     post_process_dataframe: bool = True,
                     float_precision: Optional[str] = None
                     ):
        """
        Function to read results from a file in a standardized format.
        :param full_path:
        The full path to the file where results are stored.
        :param as_pickle:
        If True, the results will be read as a pickle file.
        If False, the results will be read as a CSV file.
        :param df_annotations_dict:
        A dictionary with annotations to be added to the dataframe after reading.
        :param excluded_trials:
        A list of trials to be excluded from the results.
        If provided, the dataframe will be filtered to exclude these trials.
        If there is no column with trial ids, this parameter is ignored.
        :param name_type_separator:
        The separator used in the column names to separate the name and the data type.
        :param return_none_if_not_found:
        :param number_of_threads:
        :return:
        """

        full_path = Path(full_path)



        if format_type.lower() == 'dataframe':
            df_read = cls.read_pandas_dataframe(full_path=full_path,
                                                number_of_threads=number_of_threads,
                                                type_separator=name_type_separator,
                                                return_none_if_not_found=return_none_if_not_found,
                                                post_process_dataframe = post_process_dataframe,
                                                float_precision=float_precision
                                                )

        elif format_type.lower() == 'pickle':
            return cls.read_pickled_results(file_path=full_path,
                                            return_none_if_not_found=return_none_if_not_found)

        elif format_type.lower() == 'numpy':
            return np.load(full_path.with_suffix('.npy'))

        else:
            raise ValueError(f"Unsupported format_type: {format_type}. Supported types are 'dataframe' and 'pickle'.")

        if excluded_trials is not None:
            if SNV.TrialIndex.id_long in df_read.columns:
                df_read = df_read[~df_read[SNV.TrialIndex.id_long].isin(excluded_trials)]
            elif SNV.TrialIndex.id in df_read.columns:
                df_read = df_read[~df_read[SNV.TrialIndex.id].isin(excluded_trials)]


        if df_annotations_dict is not None:
            df_read = cls.annotate_dataframe(dataframe=df_read,
                                             annotation=df_annotations_dict)




        return df_read




class IOHamiltonianMixin(IOMixin):
    @classmethod
    def _write_hamiltonian_to_text_file(cls,
                                        hamiltonian: List[Tuple[Union[float, int], Tuple[int, ...]]],
                                        file_path: str | Path,
                                        overwrite_if_exists: bool = False,
                                        ignore_if_exists: bool = True) -> None:
        """
        Write a hamiltonian to a text file in a standard format
        :param hamiltonian:
        :param file_path:
        :param overwrite_if_exists:
        :param ignore_if_exists:
        :return:
        """

        file_path = Path(file_path)
        file_path = add_file_format_suffix(string=file_path, suffix=".txt")

        if file_path.exists():
            if overwrite_if_exists:
                print("Hamiltonian exists, overwriting.")
                file_path.unlink()
            elif ignore_if_exists:
                print("Hamiltonian exists, not overwriting.")
                return
            else:
                raise FileExistsError(f"File {file_path} already exists. "
                                      f"Use 'overwrite_if_exists' to handle this or 'ignore_if_exists' to ignore it.")

        with open(file_path, 'w') as file:
            for weight, edge in hamiltonian:
                file_line = f' '.join([str(qubit) for qubit in edge])
                file_line = f"{file_line}|{weight}\n"
                file.write(file_line)

    @classmethod
    def _load_hamiltonian_from_text_file(cls,
                                         file_path: str | Path) -> List[Tuple[float, Tuple[int, ...]]]:
        """
        Load a hamiltonian from a text file in a standard format.
        :param file_path:
        :return:
        """

        file_path = add_file_format_suffix(string=file_path, suffix='.txt')

        hamiltonian = []
        with open(f"{file_path}", 'r') as file:
            for line in file:
                line = line.strip()
                edge, weight = line.split('|')
                edge = tuple([int(qubit) for qubit in edge.split()])
                weight = float(weight)
                hamiltonian.append((weight, edge))
        return hamiltonian

    # The columns a solutions file (KnownSolutions, SolutionsArchive) carries today,
    # and the two columns the first version of the store wrote.
    SOLUTION_COLUMNS = (SNV.SolverName.id_long,
                        SNV.Runtime.id_long,
                        SNV.Energy.id_long,
                        SNV.Bitstring.id_long)
    SOLUTION_COLUMNS_LEGACY = (SNV.Bitstring.id_long,
                               SNV.Energy.id_long)
    SOLVER_NAME_UNKNOWN = "Unknown"

    @classmethod
    def _migrate_legacy_solutions_file(cls,
                                       full_path: str | Path) -> None:
        """Bring ONE solutions file to the current column schema, in place.

        Appending a row of the current schema to a file written with the legacy
        two-column one leaves a CSV that no parser can read, so the file must be
        converted before the append. Only the file about to be written is
        touched: the store holds hundreds of files and a bulk rewrite is neither
        needed nor wanted. The conversion is atomic — a complete file is built
        beside the original and then replaces it — so an interrupted run leaves
        either the old file or the new one, never a half-written one.

        A file that already carries the current columns returns untouched, and so
        does a file whose header is neither schema.

        Legacy rows carry no solver and no runtime, so they migrate with the same
        placeholders the writer uses when a caller names none. The migration changes the
        schema and nothing else: every row survives it, duplicates included.
        """

        full_path = add_file_format_suffix(string=Path(full_path), suffix='.csv')
        if not full_path.exists():
            return

        with open(full_path, 'r', newline='') as file:
            rows_read = [row for row in csv.reader(file) if row]

        if not rows_read:
            return
        header = rows_read[0]
        if header == list(cls.SOLUTION_COLUMNS) or header != list(cls.SOLUTION_COLUMNS_LEGACY):
            return

        rows_converted = []
        for row in rows_read[1:]:
            if len(row) == len(cls.SOLUTION_COLUMNS_LEGACY):
                bitstring, energy = row
                row_converted = [cls.SOLVER_NAME_UNKNOWN, 0.0, float(energy), bitstring]
            elif len(row) == len(cls.SOLUTION_COLUMNS):
                # A row appended by the current writer before the file was migrated.
                solver_name, solver_runtime, energy, bitstring = row
                row_converted = [solver_name, float(solver_runtime), float(energy), bitstring]
            else:
                raise ValueError(f"Cannot migrate {full_path}: row {row} has {len(row)} fields, "
                                 f"expected {len(cls.SOLUTION_COLUMNS_LEGACY)} or "
                                 f"{len(cls.SOLUTION_COLUMNS)}.")
            rows_converted.append(row_converted)

        # A leftover of this name is the tell that a migration was interrupted.
        path_temporary = full_path.with_name(f"{full_path.name}.migrating")
        with open(path_temporary, 'w', newline='') as file:
            # '\n', not csv's default '\r\n': every other writer of these files is pandas,
            # which appends '\n', and a migrated file must not end up with mixed terminators.
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
            writer.writerow(list(cls.SOLUTION_COLUMNS))
            writer.writerows(rows_converted)
        os.replace(str(path_temporary), str(full_path))

    @classmethod
    def _write_single_solution(cls,
                               full_path: str | Path,
                               bitstring: Union[Tuple[int, ...], str, np.ndarray, List[int]],
                               energy: float,
                               solver_name:str="Unknown",
                               solver_runtime:float=0.0,
                               ):
        bitstring = tuple([int(x) for x in bitstring])
        energy = float(energy)

        cls._migrate_legacy_solutions_file(full_path=full_path)

        # Deduplicate on (state, energy): repeated solves of the same instance must not
        # grow the file with identical rows. The same state under a different energy
        # (e.g. the same path written to after a Hamiltonian transformation) still appends.
        # The comparison below is an equality test on float64, so the read has to return
        # the value that was written: the default parser is up to one ULP off on some
        # decimal strings, and a missed match appends a duplicate row on every solve.
        existing_df = cls.read_results(full_path=full_path,
                                       return_none_if_not_found=True,
                                       format_type='dataframe',
                                       float_precision='round_trip')
        if existing_df is not None:
            import ast
            for _bs, _en in zip(existing_df[SNV.Bitstring.id_long].values,
                                existing_df[SNV.Energy.id_long].values):
                try:
                    _bs_tuple = tuple(int(x) for x in
                                      (ast.literal_eval(_bs) if isinstance(_bs, str) else _bs))
                    _matches = (_bs_tuple == bitstring and float(_en) == energy)
                except (ValueError, TypeError, SyntaxError):
                    # An unparseable row (e.g. a half-written line in a corrupt file)
                    # cannot match; it must never crash the writer or block the append.
                    continue
                if _matches:
                    return

        df_save = pd.DataFrame(data={
                                     SNV.SolverName.id_long: [solver_name],
                                     SNV.Runtime.id_long: [solver_runtime],
                                     SNV.Energy.id_long: [energy],
                                     SNV.Bitstring.id_long: [bitstring],
                                     })
        cls.write_results(data=df_save,
                          full_path=full_path,
                          format_type='dataframe')

    @classmethod
    def _write_hamiltonian_solutions(cls,
                                     file_path_main: str | Path,
                                     known_energies_dict: dict,
                                     which:str='all'
                                     ):
        file_path_main = str(file_path_main)

        if which.lower() == 'all':
            _names = ['lowest', 'highest']
        elif which.lower() == 'lowest':
            _names = ['lowest']
        elif which.lower() == 'highest':
            _names = ['highest']
        else:
            raise ValueError(f"Unsupported value for 'which': {which}. Supported values are 'all', 'lowest' and 'highest'.")


        for _name in _names:
            state = known_energies_dict.get(f'{_name}_energy_state', None)
            energy = known_energies_dict.get(f'{_name}_energy', None)
            solver = known_energies_dict.get(f'{_name}_energy_state_solver', "Unknown")
            runtime = known_energies_dict.get(f'{_name}_energy_state_runtime', 0.0)

            file_path_known_solutions = f"{file_path_main}{MKS}KnownSolutions"
            if energy is not None:
                cls._write_single_solution(full_path=file_path_known_solutions,
                                           bitstring=tuple([int(x) for x in state]),
                                           energy=energy,
                                           solver_name=solver,
                                           solver_runtime=runtime
                                           )



        #
        # highest_energy_state = known_energies_dict.get('highest_energy_state', None)
        # highest_energy = known_energies_dict.get('highest_energy', None)
        # highest_energy_state_solver = known_energies_dict.get('highest_energy_state_solver', "Unknown")
        # highest_energy_state_runtime = known_energies_dict.get('highest_energy_state_runtime', 0.0)
        #
        #
        # if highest_energy_state is not None:
        #     highest_energy_state = tuple([int(x) for x in highest_energy_state])
        #     cls._write_single_solution(full_path=file_path_known_solutions,
        #                                bitstring=highest_energy_state,
        #                                energy=highest_energy,
        #                                solver_name=highest_energy_state_solver,
        #                                solver_runtime=highest_energy_state_runtime
        #                                )


class ResultsIO(IOMixin):
    """
    A class for handling input/output operations for results data.

    It provides methods to read and write results in a standardized format,
    including support for various data types and annotations.
    """

    def __init__(self,
                 table_name_prefix: Optional[str] = None,
                 table_name_suffix: Optional[str] = None,
                 directory_main: Optional[str | Path] = None,
                 default_storage_directory: Optional[str | Path] = DEFAULT_STORAGE_DIRECTORY,
                 table_name_parts_separator: str = DEFAULT_TABLE_NAME_PARTS_SEPARATOR,
                 dataframe_type_name_separator: str = DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR
                 ):
        """
        Initialize the ResultsIO object with the specified parameters.


        ###########DIRECTORY CONVENTIONS
        The `base_path` parameter specifies the main directory where ALL results will be stored.
        If "default_storage_directory" is not provided, it defaults to "DEFAULT_STORAGE_DIRECTORY".

        The final directory for storing results will be:
        <default_storage_directory>/<base_path>

        if both are None, then the current working directory is used.

        If `base_path` is None, it defaults to the current working directory.
        If `default_storage_directory` is None, then it doesn't add any prefix to the base_path.

        NOTE: If 'default_storage_directory' is provided, but 'base_path' is None, then the whole path defaults to
        the working directory anyway, and default_storage_directory is ignored.

        :param directory_main:
        :param default_storage_directory:


        ###########Table naming conventions
        The `table_name_prefix` and `table_name_suffix` parameters are used to create a standardized table name.
        The final table name will be constructed as follows:
        tnps = name_parts_separator
        <table_name_prefix><tnps><table_name><tnps><table_name_suffix><tnps><data_type_suffix>
        where <data_type_suffix> is the data type of the table, if provided.

        :param table_name_prefix:
        If `table_name_prefix` is None, it later defaults to an empty string.
        :param table_name_suffix:
        If `table_name_suffix` is None, it later defaults to an empty string.
        :param table_name_parts_separator:
        The key separator used in the table name. Defaults to `MAIN_KEY_SEPARATOR`.

        :param dataframe_type_name_separator:

        """
        super().__init__()

        self._base_path = self.construct_base_path(directory_main=directory_main,
                                                   default_storage_directory=default_storage_directory)
        self._table_name_prefix = table_name_prefix
        self._table_name_suffix = table_name_suffix

        # tnps = table name parts separator
        self._tnps = table_name_parts_separator
        # dnts  = dataframe name/type separator
        self._dts = dataframe_type_name_separator

        os.makedirs(self._base_path, exist_ok=True)

    @property
    def base_path(self):
        return self._base_path

    @base_path.setter
    def base_path(self,
                  base_path: str | Path):

        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    @property
    def table_name_prefix(self):
        return self._table_name_prefix

    @table_name_prefix.setter
    def table_name_prefix(self, table_name_main: Optional[str]):
        self._table_name_prefix = table_name_main

    @property
    def table_name_suffix(self):
        return self._table_name_suffix

    @table_name_suffix.setter
    def table_name_suffix(self, table_name_suffix: Optional[str]):
        self._table_name_suffix = table_name_suffix

    def get_full_table_name(self,
                            table_name: Optional[str],
                            table_name_prefix: Optional[str] = None,
                            table_name_suffix: Optional[str] = None,
                            data_type: Optional[BaseNameDataType] = None):
        """
        Full table name should look like this:

        <table_name_prefix>%<table_name>%<table_name_suffix>%<data_type_suffix>
        :param table_name:
        The name of the table, e.g. 'TestTable'.
        :param table_name_prefix:
        The prefix for the table name, e.g. 'TestPrefix'.
        :param table_name_suffix:
        The suffix for the table name, e.g. 'TestSuffix'.
        :param data_type:
        The data type of the table, e.g. SNDT.Results.
        :return:
        """

        table_name_parts = [table_name_prefix, table_name, table_name_suffix]

        return super().get_full_table_name(table_name_parts=table_name_parts,
                                           data_type=data_type,
                                           name_parts_separator=self._tnps)

    def get_cleaned_table_name(self,
                               table_name: str,
                               data_type: Optional[BaseNameDataType] = None,
                               remove_prefix=True,
                               remove_suffix=True,
                               remove_data_type_suffix=True,
                               remove_file_type_suffix=True,
                               ensure_consistency_with_class_instance=True, ):
        """
        Get the cleaned table name based on the current settings.
        The full table name looks something like:
        <table_name_prefix><MAIN_KEY_SEPARATOR><table_name><MAIN_KEY_SEPARATOR><data_type_suffix><MAIN_KEY_SEPARATOR><table_name_suffix>
        And we wish to clean it up by removing the prefix, suffix, and data type suffix if requested.
        :param table_name:
        The full table name to be cleaned, e.g. 'TestPrefix%TestTable%TestSuffix%Results.dat'.
        :param data_type:
        The data type of the table, e.g. SNDT.Results.
        :param remove_prefix:
        If True, the prefix will be removed from the table name.
        :param remove_suffix:
        If True, the suffix will be removed from the table name.
        :param remove_data_type_suffix:
        If True, the data type suffix will be removed from the table name.
        :param remove_file_type_suffix:
        If True, the file type suffix (e.g. '.csv') will be removed from the table name.
        :param ensure_consistency_with_class_instance:
        If True, the method will assert that the cleaned table name is consistent with the class instance settings.
        :return:
        """

        table_name, _original_file_type_suffix = self.parse_file_type_suffix(file_name=table_name)

        parsed_table_name = self.parse_table_name(full_table_name=table_name,
                                                  name_parts_separator=self._tnps)
        _expected_prefix = self.table_name_prefix
        _expected_suffix = self.table_name_suffix
        _expected_dt_suffix = self.get_data_type_suffix(data_type=data_type)

        cleaned_parts: List[str] = []

        idx = 0

        if _expected_prefix is not None:
            prefix = parsed_table_name[idx]
            if ensure_consistency_with_class_instance and prefix != _expected_prefix:
                raise ValueError(f"Expected prefix '{_expected_prefix}' "
                                 f"but got '{prefix}'")

            if not remove_prefix:
                cleaned_parts.append(prefix)
            idx += 1

        main_name = parsed_table_name[idx]
        cleaned_parts.append(main_name)
        idx += 1

        if _expected_suffix is not None:
            suffix = parsed_table_name[idx]

            if ensure_consistency_with_class_instance and suffix != _expected_suffix:
                raise ValueError(f"Expected suffix '{_expected_suffix}' "
                                 f"but got '{suffix}'")

            if not remove_suffix:
                cleaned_parts.append(suffix)
            idx += 1

        if _expected_dt_suffix is not None:
            dt_suffix = parsed_table_name[idx]

            if ensure_consistency_with_class_instance and dt_suffix != _expected_dt_suffix:
                raise ValueError(f"Expected data type suffix '{_expected_dt_suffix}' "
                                 f"but got '{dt_suffix}'")

            if not remove_data_type_suffix:
                cleaned_parts.append(dt_suffix)
            idx += 1

        cleaned_name = self._tnps.join(cleaned_parts)

        if not remove_file_type_suffix:
            # from . import add_file_format_suffix
            cleaned_name = str(add_file_format_suffix(string=cleaned_name, suffix=_original_file_type_suffix))

        return cleaned_name

    def get_full_file_name_table(self,
                                 table_name: Optional[str],
                                 table_name_prefix: Optional[str] = None,
                                 table_name_suffix: Optional[str] = None,
                                 data_type: Optional[BaseNameDataType] = None,
                                 ):
        """
        Get the full file name for a table, including prefix, suffix, and data type.
        The full file name will be constructed as follows:
        <table_name_prefix><tnps><table_name><tnps><data_type_suffix><tnps><table_name_suffix>
        where <data_type_suffix> is the data type of the table, if provided.

        :param table_name:
        :param table_name_prefix:
        if None, it defaults to the value of self.table_name_prefix.
        :param table_name_suffix:
        if None, it defaults to the value of self.table_name_suffix.
        :param data_type:
        :return:
        """

        if table_name_prefix is None:
            table_name_prefix = self.table_name_prefix
        if table_name_suffix is None:
            table_name_suffix = self.table_name_suffix

        return self.get_full_table_name(table_name_prefix=table_name_prefix,
                                        table_name_suffix=table_name_suffix,
                                        table_name=table_name,
                                        data_type=data_type)

    def get_save_directory(self,
                           directory_subpath: str | Path = None):
        """
        Get the full path to the directory where results will be saved.
        :param directory_subpath:
        :return:
        """
        if directory_subpath is None:
            directory_subpath = Path()
        directory_subpath = Path(directory_subpath)
        return self._base_path / directory_subpath

    def write_dense_array(self,
                          array: np.ndarray,
                          full_path: str | Path,
                          ):
        """
        Write a dense numpy array to a file in .npy format.
        :param array:
        :param full_path:
        :return:
        """

        # from . import add_file_format_suffix
        full_path = str(add_file_format_suffix(string=full_path, suffix='.npy'))

        np.save(f"{full_path}", array, allow_pickle=False)

    def write_results(self,
                      dataframe: Union[pd.DataFrame, Any],
                      directory_subpath: Optional[str | Path] = None,
                      table_name: Optional[str] = None,
                      table_name_prefix: Optional[str | Path] = None,
                      table_name_suffix: Optional[str | Path] = None,
                      data_type: BaseNameDataType = None,
                      df_annotations_dict: dict = None,
                      format_type='dataframe',
                      overwrite_existing_non_csv: bool = False,
                      ):
        """
        Write results to a file in a standardized format.
        The final path will be:
        <full_save_directory>/<full_table_name>
        created based on conventions defined in this class.

        :param dataframe:
        The data to be saved
        :param directory_subpath:
        A subpath within the main directory where the results will be saved.
        If none, it defaults to the main directory.
        :param table_name:
        The name of the table to be saved. If None, it defaults to an empty string.
        :param table_name_prefix:
        The prefix for the table name. If None, it defaults to the value of self.table_name_prefix.
        :param table_name_suffix:
        The suffix for the table name. If None, it defaults to the value of self.table_name_suffix.
        :param data_type:
        The data type of the table. If None, no suffix is added to the table name.
        :param df_annotations_dict:
        A dictionary with annotations to be added to the dataframe before saving.
        :param as_pickle:
        If True, the results will be saved as a pickle file.
        If False, it will be saved as a CSV file.

        :return:
        """

        full_table_name = self.get_full_file_name_table(table_name=table_name,
                                                        data_type=data_type,
                                                        table_name_prefix=table_name_prefix,
                                                        table_name_suffix=table_name_suffix)

        full_save_directory = self.get_save_directory(directory_subpath=directory_subpath)

        full_path = full_save_directory / full_table_name

        # This is special type of data that is saved to dense array.
        # TODO(FBM): this should probably be handled in a different way
        if data_type == SNDT.Correlators:
            return self.write_dense_array(array=dataframe,
                                          full_path=full_path)

        if df_annotations_dict is not None:
            if isinstance(dataframe, pd.DataFrame):
                dataframe = self.annotate_dataframe(dataframe=dataframe,
                                                    annotation=df_annotations_dict)
            else:
                print("WARNING: df_annotations_dict is provided, but the data is not a pandas DataFrame. "
                      "Annotations will be ignored.")

        if isinstance(dataframe, pd.DataFrame):
            dataframe_renamed = self._df_rename_datatypes_columns(dataframe=dataframe)
        else:
            dataframe_renamed = dataframe
        # print('timestop 1', full_path)

        return super().write_results(data=dataframe_renamed,
                                     full_path=full_path,
                                     format_type=format_type,
                                     overwrite_existing_non_csv=overwrite_existing_non_csv)

    def read_results(self,
                     directory_subpath: Optional[str | Path] = None,
                     table_name: Optional[str] = None,
                     table_name_prefix: Optional[str | Path] = None,
                     table_name_suffix: Optional[str | Path] = None,
                     data_type: BaseNameDataType = None,
                     df_annotations_dict: dict = None,
                     format_type='dataframe',
                     excluded_trials=None,
                     number_of_threads=1,
                     return_none_if_not_found: bool = False,
                     full_absolute_path_to_the_file: Optional[str] = None,
                     post_process_dataframe: bool = True

                     ):
        """
        Read results from a file in a standardized format.
        It assumes that the file is saved in a standardized format using `write_results_standardized`.

        :param directory_subpath:
        A subpath within the main directory where the results are saved.
        :param table_name:
        The name of the table to be read. If None, it defaults to an empty string.
        :param table_name_prefix:
        The prefix for the table name. If None, it defaults to the value of self.table_name_prefix.
        :param table_name_suffix:
        The suffix for the table name. If None, it defaults to the value of self.table_name_suffix.
        :param data_type:
        The data type of the table. If None, no suffix is added to the table name.
        :param df_annotations_dict:
        A dictionary with annotations to be added to the dataframe after reading.
        :param as_pickle:
        If True, the results will be read as a pickle file.
        :param excluded_trials:
        A list of trials to be excluded from the results.
        :param number_of_threads:
        The number of threads to use for parallel processing.
        :param return_none_if_not_found:
        If True, the function will return None if the file is not found.
        :return:
        """
        if full_absolute_path_to_the_file is not None:
            return super().read_results(full_path=full_absolute_path_to_the_file,
                                        return_none_if_not_found=return_none_if_not_found,
                                        post_process_dataframe=post_process_dataframe)

        full_table_name = self.get_full_file_name_table(table_name=table_name,
                                                        data_type=data_type,
                                                        table_name_prefix=table_name_prefix,
                                                        table_name_suffix=table_name_suffix)

        full_save_directory = self.get_save_directory(directory_subpath=directory_subpath)

        full_path = full_save_directory / full_table_name

        return super().read_results(full_path=full_path,
                                    format_type=format_type,
                                    df_annotations_dict=df_annotations_dict,
                                    excluded_trials=excluded_trials,
                                    number_of_threads=number_of_threads,
                                    name_type_separator=self._dts,
                                    return_none_if_not_found=return_none_if_not_found,
                                    post_process_dataframe=post_process_dataframe)
