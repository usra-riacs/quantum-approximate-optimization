# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.
import copy
import csv
import os
import pickle
import time
from pathlib import Path
from typing import Optional, List, Any, Tuple, Union, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.data_analysis.data_handling.standard_names import (STANDARD_NAMES_DATA_TYPES as SNDT,
                                                                HamiltonianClassSpecifierGeneral,
                                                                HamiltonianInstanceSpecifierGeneral, )
from quapopt.data_analysis.data_handling.standard_names import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling.standard_names.data_hierarchy import (BaseNameDataType, )
from quapopt.data_analysis.data_handling.standard_names.data_hierarchy import (
    MAIN_KEY_SEPARATOR as MKS,
    DEFAULT_TABLE_NAME_PARTS_SEPARATOR,
    DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR
)
from quapopt.data_analysis.data_handling.standard_names.data_hierarchy import (MAIN_KEY_VALUE_SEPARATOR as MKVS,
                                                                               SUB_KEY_VALUE_SEPARATOR as SKVS)

load_dotenv()

try:
    DEFAULT_STORAGE_DIRECTORY = Path(os.getenv('DEFAULT_STORAGE_DIRECTORY'))
except Exception as e:
    # in case the environment variable is not set, we will use the default path
    import quapopt

    project_root_abs = os.path.dirname(os.path.abspath(quapopt.__file__))
    project_root_abs = os.path.dirname(project_root_abs)

    DEFAULT_STORAGE_DIRECTORY = Path(project_root_abs) / "output"

if DEFAULT_STORAGE_DIRECTORY is None:
    DEFAULT_STORAGE_DIRECTORY = Path().cwd() / "output"

DEFAULT_STORAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)

SUPPORTED_DATATYPES_BASIC = ['float', 'float16', 'float32', 'float64',
                             'int', 'int8', 'int16', 'int32', 'int64',
                             'str', 'bool', 'uint', 'uint8', 'uint16', 'uint32', 'uint64',
                             ]
SUPPORTED_DATATYPES_COMPLEX = ['list', 'tuple',
                               'dict', 'set',
                               'ndarray', 'None']
SUPPORTED_DATATYPES = SUPPORTED_DATATYPES_BASIC + SUPPORTED_DATATYPES_COMPLEX

INTEGER_TUPLE_TYPES = [SNV.Bitflip.id, SNV.Bitflip.id_long,
                       SNV.Bitstring.id, SNV.Bitstring.id_long,
                       SNV.Permutation.id, SNV.Permutation.id_long,
                       ]


def _string_mod_fun1(string:str|Path,
                    suffix:str):

    input_format = type(string)
    string = str(string)

    if not suffix.startswith('.'):
        proper_suffix = f'.{suffix}'
    else:
        proper_suffix = suffix
    return input_format, string, proper_suffix

def _string_mod_fun2(string:str|Path,
                    input_format:type):

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

def add_file_format_suffix(string: str|Path,
                      suffix: str)->Path:
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


def remove_file_format_suffix(string:str|Path,
                                suffix:str):
    input_format, string, proper_suffix = _string_mod_fun1(string=string,
                                                           suffix=suffix)

    if string.endswith(proper_suffix):
        string = string[:-len(proper_suffix)]

    return _string_mod_fun2(string=string,
                            input_format=input_format)




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
                             data_type: SNDT) -> str:
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

        directory_main = directory_main
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
                                        hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral,
                                        storage_directory: Optional[
                                            str | Path] = DEFAULT_STORAGE_DIRECTORY) -> Path:
        if hamiltonian_class_specifier is None:
            raise ValueError("Hamiltonian class specifier is None.")
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


        return joined_name



    @classmethod
    def get_full_table_name(cls,
                            table_name_parts: List[Optional[str]],
                            data_type: Optional[SNDT] = None,
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
    def write_dataframe_results(cls,
                                data: pd.DataFrame,
                                full_path: str | Path):
        """
        Save a pandas DataFrame to a CSV file in a standardized format.
        :param data:
        :param full_path:
        :return:
        """
        #print('timestop 3', full_path)
        full_path = Path(full_path)

        full_path = add_file_format_suffix(string=full_path, suffix='.csv')

        #print('timestop 4', full_path)


        full_path.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(str(full_path),
                    index=False,
                    mode='a',
                    header=not os.path.exists(full_path),
                    quoting=csv.QUOTE_NONNUMERIC)

        return full_path

    @classmethod
    def write_results(cls,
                      data: pd.DataFrame | Any,
                      full_path: str | Path,
                      format_type: str = 'dataframe'):

        #print('timestop 2', full_path)


        if format_type.lower() in ['dataframe']:
            return cls.write_dataframe_results(data=data,
                                               full_path=full_path)
        elif format_type.lower() in ['pickle']:
            return cls.write_pickled_results(object_to_save=data,
                                             file_path=full_path)
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
            dataframe[key] = annotation[key]
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
    def read_pandas_dataframe(cls,
                              full_path: str | Path,
                              number_of_threads=1,
                              type_separator=DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR,
                              return_none_if_not_found=False):
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
        try:
            df_read = pd.read_csv(str(full_path))
        except(FileNotFoundError) as e:
            if return_none_if_not_found:
                return None
            else:
                raise e

        # THis is a hack to make sure that the columns are read correctly
        # TODO(FBM): We should refactor this
        columns_renamed, columns_types = {}, {}
        for col in df_read.columns:

            split_type = col.split(type_separator)
            if len(split_type) == 2:
                col_name, col_type = col.split(type_separator)
                columns_renamed[col] = col_name
                columns_types[col_name] = col_type

            elif len(split_type) == 1:
                col_name = col

                col_type = None
                if col_name in INTEGER_TUPLE_TYPES:
                    col_type = 'tuple'
                else:
                    for _, base_name in SNV.get_all_attributes().items():
                        id_short, id_long = base_name.id, base_name.id_long
                        if col_name == id_short or col_name == id_long:
                            col_type = str(base_name.value_format)
                            break

                if col_type is None:
                    col_type = 'Unknown'
                columns_renamed[col] = col_name
                columns_types[col_name] = col_type
            else:
                raise ValueError(f"Column name '{col}' is not properly formatted.")
            try:
                if col_type == 'Unknown':
                    pass
                elif col_type in SUPPORTED_DATATYPES_BASIC:
                    # print("CONVERTING TO:",col_type)
                    df_read[col] = cls._convert_value_basic(column_series=df_read[col],
                                                            type_name=col_type)
                elif col_type in SUPPORTED_DATATYPES_COMPLEX:
                    if col_name in INTEGER_TUPLE_TYPES and col_type == 'tuple':
                        try:
                            df_read[col] = anf.df_column_apply_function_parallelized(
                                series=df_read[col],
                                function_to_apply=anf.eval_string_tuple_to_tuple,
                                number_of_threads=number_of_threads)
                        except(ValueError) as e:
                            df_read[col] = anf.df_column_apply_function_parallelized(
                                series=df_read[col],
                                function_to_apply=anf.eval_string_float_tuple_to_int_tuple,
                                number_of_threads=number_of_threads)
                    elif col_name in INTEGER_TUPLE_TYPES and col_type == 'ndarray':
                        # print("HEJKA HERE")
                        # print(df_read[col].values[0])
                        df_read[col] = anf.df_column_apply_function_parallelized(
                            series=df_read[col],
                            function_to_apply=lambda x: np.fromstring(x[1:-1], dtype=int, sep=' '),
                            number_of_threads=number_of_threads)


            except(ValueError) as val_err:
                print("ERROR READING COLUMN:", col)
                print("MESSAGE:", val_err)
                columns_renamed[col] = col
                columns_types[col] = 'Unknown'

        df_read = df_read.rename(columns=columns_renamed)
        return df_read

    @classmethod
    def read_results(cls,
                     full_path: str | Path,
                     format_type='dataframe',
                     df_annotations_dict: dict = None,
                     excluded_trials=None,
                     name_type_separator=DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR,
                     return_none_if_not_found=False,
                     number_of_threads=1, ):
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
                                                return_none_if_not_found=return_none_if_not_found)

        elif format_type.lower() == 'pickle':
            return cls.read_pickled_results(file_path=full_path,
                                            return_none_if_not_found=return_none_if_not_found)

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
                                        overwrite_if_exists: bool = False) -> None:
        """
        Write a hamiltonian to a text file in a standard format
        :param hamiltonian:
        :param file_path:
        :return:
        """

        file_path = Path(file_path)
        file_path = add_file_format_suffix(string=file_path, suffix=".txt")

        if file_path.exists():
            if overwrite_if_exists:
                file_path.unlink()
            else:
                raise FileExistsError(f"File {file_path} already exists. "
                                      f"Use 'overwrite_if_exists' to handle this.")

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

    @classmethod
    def _write_single_solution(cls,
                               full_path: str | Path,
                               bitstring: Union[Tuple[int, ...], str, np.ndarray, List[int]],
                               energy: float):
        bitstring = tuple([int(x) for x in bitstring])
        df_save = pd.DataFrame(data={SNV.Bitstring.id_long: [bitstring],
                                     SNV.Energy.id_long: [energy],
                                     })
        cls.write_results(data=df_save,
                          full_path=full_path,
                          format_type='dataframe')

    @classmethod
    def _write_hamiltonian_solutions(cls,
                                     file_path_main: str|Path,
                                     known_energies_dict: dict, ):
        file_path_main = str(file_path_main)

        if 'spectrum' in known_energies_dict:
            spectrum = known_energies_dict['spectrum']
            if spectrum is not None:
                file_path_spectrum = f"{file_path_main}{MKS}Spectrum"
                with open(f"{file_path_spectrum}.txt", 'w') as file:
                    for eigenvalue in spectrum:
                        file_line = f"{eigenvalue}\n"
                        file.write(file_line)
            del known_energies_dict['spectrum']

        lowest_energy_state = known_energies_dict.get('lowest_energy_state', None)
        lowest_energy = known_energies_dict.get('lowest_energy', None)

        file_path_known_solutions = f"{file_path_main}{MKS}KnownSolutions"
        if lowest_energy_state is not None:
            lowest_energy_state = tuple([int(x) for x in lowest_energy_state])
            cls._write_single_solution(full_path=file_path_known_solutions,
                                       bitstring=lowest_energy_state,
                                       energy=lowest_energy)

        highest_energy_state = known_energies_dict.get('highest_energy_state', None)
        highest_energy = known_energies_dict.get('highest_energy', None)
        if highest_energy_state is not None:
            highest_energy_state = tuple([int(x) for x in highest_energy_state])
            cls._write_single_solution(full_path=file_path_known_solutions,
                                       bitstring=highest_energy_state,
                                       energy=highest_energy)
