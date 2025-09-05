# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import os
import uuid
from pathlib import Path
from typing import Optional, List, Union, Any

import numpy as np
import pandas as pd

from quapopt.data_analysis.data_handling.io_utilities import IOMixin, DEFAULT_STORAGE_DIRECTORY, add_file_format_suffix
from quapopt.data_analysis.data_handling.schemas import (STANDARD_NAMES_DATA_TYPES as SNDT,
                                                         )
from quapopt.data_analysis.data_handling.schemas.naming import (
    DEFAULT_TABLE_NAME_PARTS_SEPARATOR,
    DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR
)


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
                            data_type: Optional[SNDT] = None):
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
                               data_type: Optional[SNDT] = None,
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
                                 data_type: Optional[SNDT] = None,
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
                      dataframe: pd.DataFrame|Any,
                      directory_subpath: Optional[str | Path] = None,
                      table_name: Optional[str] = None,
                      table_name_prefix: Optional[str | Path] = None,
                      table_name_suffix: Optional[str | Path] = None,
                      data_type: SNDT = None,
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
                     data_type: SNDT = None,
                     df_annotations_dict: dict = None,
                     format_type='dataframe',
                     excluded_trials=None,
                     number_of_threads=1,
                     return_none_if_not_found: bool = False,
                     full_absolute_path_to_the_file: Optional[str] = None,
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
                                        return_none_if_not_found=return_none_if_not_found)

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
                                    return_none_if_not_found=return_none_if_not_found)


if __name__ == '__main__':
    table_length = 2
    _test_df = pd.DataFrame(data={
        # 'ex_float':[float(x) for x in range(table_length)],
        f'exfloat32': [np.float32(x) for x in range(table_length)],
        f'exfloat64': [np.float64(x) for x in range(table_length)],
        # 'ex_int':[int(x) for x in range(table_length)],
        f'exint64': [np.int64(x) for x in range(table_length)],
        f'exint32': [np.int32(x) for x in range(table_length)],
        # 'ex_str':[str(x) for x in range(table_length)],
        f'exbool': [bool(x) for x in range(table_length)],
        'Permutation': [tuple([x for x in range(4)]) for x in range(table_length)],
        'Bitstring': [tuple([0 for _ in range(4)]) for x in range(table_length)],
        'Bitflip': [tuple([1 for _ in range(4)]) for x in range(table_length)],

    })

    results_io_test = ResultsIO(directory_main='test_directory',
                                table_name_prefix='TestPrefix',
                                table_name_suffix='TestSuffix', )

    # random uuid
    uuid = uuid.uuid4()
    table_name_test = f'TestTable1-' + str(uuid)
    test_subpath = 'test_subpath'

    results_io_test.write_results(dataframe=_test_df,
                                  table_name=table_name_test,
                                  directory_subpath=test_subpath,
                                  data_type=SNDT.TestResults,
                                  as_pickle=False)

    df_read = results_io_test.read_results(table_name=table_name_test,
                                           directory_subpath=test_subpath,
                                           data_type=SNDT.TestResults)
    # now we verify if types match with the original datafrae _test_df
    for col in df_read.columns:
        # take the column values from both dfs
        values_original = _test_df[col].values
        values_read = df_read[col].values
        # check if types are the same
        for i in range(len(values_original)):
            # print(col, i)
            assertion = type(values_original[i]) == type(values_read[i])

            if not assertion:
                print(col, i, 'should be:', type(values_original[i]), 'is:', type(values_read[i]))
        # check if values are the same
        for i in range(len(values_original)):
            # print(col, i)
            assert values_original[i] == values_read[i]

    print(df_read)
