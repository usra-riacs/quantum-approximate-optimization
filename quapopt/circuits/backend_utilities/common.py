# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
from quapopt.data_analysis.data_handling import (STANDARD_NAMES_VARIABLES as SNV,
                                                 STANDARD_NAMES_DATA_TYPES as SNDT,
                                                 MAIN_KEY_VALUE_SEPARATOR)
def get_backend_data_path_standardized(backend_name: str):
    experiment_folders_hierarchy = ['BackendInformation',
                                    f"{SNV.Backend.id}{MAIN_KEY_VALUE_SEPARATOR}{backend_name}"]

    return experiment_folders_hierarchy
