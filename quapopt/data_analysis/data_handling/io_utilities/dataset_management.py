"""
Dataset management utilities for grouping multiple experiment sets.

Provides a lightweight many-to-many mapping between dataset names and experiment set IDs,
stored as one CSV file per dataset. This allows the same experiment set to belong to
multiple datasets, supporting flexible organization of iterative algorithm runs (e.g., NDAR).
"""

import warnings
from pathlib import Path
from typing import Optional, List, Type, Dict, Any

import pandas as pd

from quapopt import ancillary_functions as anf
from quapopt.data_analysis.data_handling.io_utilities import DEFAULT_STORAGE_DIRECTORY
from quapopt.data_analysis.data_handling.io_utilities.standardized_io import IOMixin
from quapopt.data_analysis.data_handling.schemas import (
    STANDARD_NAMES_DATA_TYPES as SNDT,
    STANDARD_NAMES_VARIABLES as SNV,
)
import ast


def _get_dataset_csv_path(
        dataset_name: str,
        default_storage_directory: Optional[str | Path] = DEFAULT_STORAGE_DIRECTORY,
) -> Path:
    base_path = IOMixin.construct_base_path(
        default_storage_directory=default_storage_directory,
        directory_main=''
    )
    subpath = IOMixin.get_subpath_of_data_type(data_type=SNDT.DatasetName)
    return base_path / subpath / f"{SNDT.DatasetName.id}={dataset_name}.csv"


def add_to_dataset(
        dataset_name: str,
        experiment_set_id: str,
        experiment_set_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description:Optional[str]=None,
        default_storage_directory: Optional[str | Path] = DEFAULT_STORAGE_DIRECTORY,
        experiment_folders_hierarchy:Optional[List[str]]=None,
) -> Path:
    """
    Register an experiment set in a dataset. Creates the dataset CSV if it doesn't exist.

    :param dataset_name: Name of the dataset to add to.
    :param experiment_set_id: Unique identifier of the experiment set.
    :param experiment_set_name: Human-readable name of the experiment set.
    :param tags: Optional list of tag strings (e.g., ["ndar", "er", "n20"]). Stored semicolon-separated in CSV.
    :param default_storage_directory: Root storage directory.
    :param directory_main: Main subdirectory for data organization.
    :returns: Path to the dataset CSV file.
    """
    csv_path = _get_dataset_csv_path(
        dataset_name=dataset_name,
        default_storage_directory=default_storage_directory,
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _esi_col = SNV.ExperimentSetID.id_long

    # Check for existing entry to avoid duplicates
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        existing_df[_esi_col] = existing_df[_esi_col].astype(str)
        if str(experiment_set_id) in existing_df[_esi_col].values:
            return csv_path

    new_row = pd.DataFrame({
        f'{SNDT.DatasetName.id_long}': [dataset_name],
        f'{SNV.ExperimentSetID.id_long}': [str(experiment_set_id)],
        f'{SNV.ExperimentSetName.id_long}': [experiment_set_name],
        f'OriginalStorageDirectory':[default_storage_directory],
        f'OriginalExperimentFoldersHierarchy':[experiment_folders_hierarchy if experiment_folders_hierarchy else ''],
        f'{SNV.Timestamp.id_long}': [anf.get_current_date_time()],
        'description':[description if description else ''],
        'tags': [';'.join(tags) if tags else ''],
        
    })

    if csv_path.exists():
        df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(csv_path, index=False)
    return csv_path


def read_dataset_info(
        dataset_name: str,
        default_storage_directory: Optional[str | Path] = DEFAULT_STORAGE_DIRECTORY,
        return_just_ids:bool=True,
        ids_to_ignore:Optional[List[str]]=None,
) -> Optional[pd.DataFrame|List[str]]:
    """
    Read the contents of a dataset CSV.

    :param dataset_name: Name of the dataset to read.
    :param default_storage_directory: Root storage directory.
    :param directory_main: Main subdirectory for data organization.
    :returns: DataFrame with dataset entries, or None if the dataset doesn't exist.
    """
    csv_path = _get_dataset_csv_path(
        dataset_name=dataset_name,
        default_storage_directory=default_storage_directory,
    )

    if not csv_path.exists():
        if return_just_ids:
            return []
        return None

    df = pd.read_csv(csv_path)
    df[f'{SNV.ExperimentSetID.id_long}'] = df[f'{SNV.ExperimentSetID.id_long}'].astype(str)
    df = df.drop_duplicates(subset=[f'{SNV.ExperimentSetID.id_long}'])

    if ids_to_ignore is not None:
        df = df[~df[f'{SNV.ExperimentSetID.id_long}'].isin(ids_to_ignore)]


    if return_just_ids:
        return sorted(df[f'{SNV.ExperimentSetID.id_long}'].tolist())

    return df


def gather_dataset_results(dataset_name: str,
                           data_types:List[Type[SNDT]],
                            default_storage_directory: Optional[str | Path] = DEFAULT_STORAGE_DIRECTORY,
                            ids_to_ignore:Optional[List[str]]=None,
                           overwrite_storage_directory:Optional[str]=None,
                           overwrite_experiment_folders_hierarchy:Optional[List[str]]=None,
                           read_kwargs_results:Optional[Dict[str,Any]]=None,
                           read_kwargs_metadata:Optional[Dict[str,Any]]=None,
                           return_none_if_not_found:bool=False,
                           ):
    from quapopt.data_analysis.data_handling import ResultsLogger

    dataset_info = read_dataset_info(dataset_name=dataset_name,
                                     default_storage_directory=default_storage_directory,
                                     ids_to_ignore=ids_to_ignore,
                                     return_just_ids=False)
    if dataset_info is None:
        raise ValueError(f"Dataset info not found: {dataset_name}")


    for _, row in dataset_info.iterrows():
        experiment_set_id = row[f'{SNV.ExperimentSetID.id_long}']

        if ids_to_ignore is not None and experiment_set_id in ids_to_ignore:
            continue



        storage_directory = (
            overwrite_storage_directory
            if overwrite_storage_directory is not None
            else row['OriginalStorageDirectory']
        )
        # A dataset produced on one machine and copied to another carries the *producing* machine's
        # absolute storage path, which does not exist on the reading machine. Redirect to the local
        # store instead of handing a dead path to ResultsLogger: its constructor calls os.makedirs
        # on the base path, so a dead path raises (or, worse, silently resolves to a directory that
        # exists but is empty, which reads back as "no experiment sets found").
        if not Path(storage_directory).exists():
            warnings.warn(f"Recorded storage directory '{storage_directory}' does not exist on this "
                          f"machine; reading dataset '{dataset_name}' from "
                          f"'{default_storage_directory}' instead.")
            storage_directory = default_storage_directory

        folders_hierarchy = (
            overwrite_experiment_folders_hierarchy
            if overwrite_experiment_folders_hierarchy is not None
            else ast.literal_eval(row['OriginalExperimentFoldersHierarchy'])
        )

        results_reader = ResultsLogger(experiment_set_id=experiment_set_id,
                                       experiment_folders_hierarchy=folders_hierarchy,
                                       default_storage_directory=storage_directory,
                                       do_not_create_experiment_ids=True)

        if read_kwargs_results is None:
            read_kwargs_results = {}
        if read_kwargs_metadata is None:
            read_kwargs_metadata = {}

        read_kwargs_results_proper = read_kwargs_results.copy()
        read_kwargs_results_proper['return_none_if_not_found'] = read_kwargs_results_proper.get('return_none_if_not_found', return_none_if_not_found)
        read_kwargs_results_proper['drop_experiment_instance_id'] = read_kwargs_results_proper.get('drop_experiment_instance_id', False)

        read_kwargs_metadata_proper = read_kwargs_metadata.copy()
        read_kwargs_metadata_proper['return_none_if_not_found'] = read_kwargs_metadata_proper.get('return_none_if_not_found', return_none_if_not_found)

        dfs_to_return = []
        for datatype in data_types:
            try:
                df = results_reader.gather_results(data_type=datatype,
                                                  **read_kwargs_results_proper)

            except Exception as e:
                df = None


            if df is None:

                try:
                    df = results_reader.read_metadata(shared_across_experiment_set=True,
                                                     data_type=datatype,
                                                     **read_kwargs_metadata_proper)
                except Exception as e:
                    if not return_none_if_not_found:
                        raise e
                    else:
                        df = None


            dfs_to_return.append(df)
        yield dfs_to_return


