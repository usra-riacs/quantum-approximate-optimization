# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import pandas as pd

from quapopt.data_analysis.data_handling.schemas.naming import (MAIN_KEY_VALUE_SEPARATOR as MKVS,
                                                                MAIN_KEY_SEPARATOR as MKS,
                                                                STANDARD_NAMES_VARIABLES as SNV,
                                                                STANDARD_NAMES_DATA_TYPES as SNDT)
from quapopt.data_analysis.data_handling import ResultsLogger, LoggingLevel

from typing import List, Optional, Type, Callable, Dict, Any, Mapping
import numpy as np
from quapopt import ancillary_functions as anf
from tqdm.notebook import tqdm
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian


from quapopt.data_analysis.statistics.SamplingResultsAnalyzer import SamplingResultsAnalyzer
from quapopt.hamiltonians.generators import create_hamiltonian_from_descriptions

from quapopt.ancillary_functions.presets import (get_standard_subfolders_hierarchy_full,
                                                 get_standard_subfolders_hierarchy_sub,
                                                 get_standard_subfolders_hierarchy_main)

class QAOAResultsAnalyzer(SamplingResultsAnalyzer):
    def __init__(self,
                 experiment_config:Dict[str,Any],
                 df_processing_functions:List[Callable[[pd.DataFrame], pd.DataFrame]]=None,
                 subset_variables_values_dict:Optional[Dict[str,Any|List[Any]]]=None,
                 ):

        super().__init__(experiment_config=experiment_config)

        if not isinstance(df_processing_functions, (list,tuple)):
            df_processing_functions = [df_processing_functions]

        if df_processing_functions is None:
            df_processing_functions = []

        _file_name_filter = None
        if subset_variables_values_dict is not None:
            def _file_name_filter(file_name:str)->bool:
                parsed_dict = ResultsLogger.parse_file_name_to_dict(file_name)
                for key, value in subset_variables_values_dict.items():
                    if key in parsed_dict:
                        if isinstance(value, list):
                            if parsed_dict[key] not in value:
                                return False
                        else:
                            if parsed_dict[key] != value:
                                return False
                return True
            def _df_processing_function(df:pd.DataFrame)->pd.DataFrame:
                for key, value in subset_variables_values_dict.items():
                    if key in df.columns:

                        if isinstance(value,list):
                            df = df[df[key].isin(value)]
                        else:
                            df = df[df[key]==value]


                return df

            df_processing_functions.append(_df_processing_function)



        self._df_processing_functions = df_processing_functions
        self._file_name_filter = _file_name_filter


        self._cost_hamiltonians = {}



        self._hamiltonians_metadata = None
        self._id_to_hamiltonian_mapping = None


    @property
    def cost_hamiltonians(self) -> Dict[str, Dict[str, ClassicalHamiltonian]]:
        return self._cost_hamiltonians

    def set_cost_hamiltonians(self,
                              cost_hamiltonians:Dict[str, Dict[str, ClassicalHamiltonian]]
                              ):
        self._cost_hamiltonians = cost_hamiltonians





    def initialize_results_from_config(self,
                                        data_types_of_interest:List[Type[SNDT]],
                                        experiment_set_ids_experiments: Optional[List[str]] = None,
                                        experiment_set_ids_simulations: Optional[List[str]] = None,
                                        show_progress_bar:bool=True,
                                        noiseless_simulation:bool=True,
                                        ):



        super()._initialize_results_from_config(
            data_types_of_interest=data_types_of_interest,
            experiment_set_ids_experiments=experiment_set_ids_experiments,
            experiment_set_ids_simulations=experiment_set_ids_simulations,
            show_progress_bar=show_progress_bar,
            noiseless_simulation=noiseless_simulation
        )

        self._add_hamiltonians_metadata(data_type_for_inference=data_types_of_interest[0])



    def _add_hamiltonians_metadata(self,
                                   data_type_for_inference:Type[SNDT]=SNDT.Bitstrings,
                                   experiment_set_ids_experiments: Optional[List[str]] = None,
                                   experiment_set_ids_simulations: Optional[List[str]] = None,
                                   data_type_of_interest:Optional[Type[SNDT]]=None,
                                   ):


        _res_df = self.get_data_type(data_type=data_type_for_inference)

        #print(_res_df.columns)
        if _res_df is None:
            return

        if SNV.HamiltonianClassDescription.id_long not in _res_df.columns:
            return
        if SNV.HamiltonianInstanceDescription.id_long not in _res_df.columns:
            return


        _res_df = _res_df[[SNV.HamiltonianClassDescription.id_long,
                          SNV.HamiltonianInstanceDescription.id_long]].copy().drop_duplicates()


        _unique_class_instance_pairs = _res_df.to_records(index=False).tolist()
        _cost_hamiltonians = {}

        for class_description, instance_description in _unique_class_instance_pairs:
            if class_description not in _cost_hamiltonians:
                _cost_hamiltonians[class_description] = {}
            if instance_description not in _cost_hamiltonians[class_description]:
                hamiltonian_instance = create_hamiltonian_from_descriptions(class_description=class_description,
                                                                            instance_description=instance_description,
                                                                            default_backend='numpy')
                _cost_hamiltonians[class_description][instance_description] = hamiltonian_instance

        self.set_cost_hamiltonians(cost_hamiltonians=_cost_hamiltonians)





    def read_results(self,
                   main_data_type:Type[SNDT],
                   experiment_set_ids_experiments:Optional[List[str]]=None,
                   experiment_set_ids_simulations:Optional[List[str]]=None,
                   show_progress_bar:bool=True,
                   file_name_filter_function: Optional[Mapping[str, bool] | Callable[[str], bool]] = None,
                     noiseless_simulation:bool=True,

                     ):


        def _combined_file_name_filter(file_name: str) -> bool:
            if file_name_filter_function is not None:
                _bool_from_external = file_name_filter_function(file_name)
                if not _bool_from_external:
                    return False

            if self._file_name_filter is not None:
                return self._file_name_filter(file_name)

            return True

        experiment_config = self._experiment_config
        main_folders_hierarchy = experiment_config['main_folders_hierarchy']
       # backend_name = experiment_config['backend_name']
      #  hamiltonian_class_description = experiment_config['hamiltonian_class_description']

        faulty_experiment_ids = experiment_config.get('faulty_experiment_ids', None)

        merge_instances_metadata_data_type = experiment_config.get('merge_instances_metadata_data_type',
                                                                   SNDT.QAOAOptimizationMetadata)

        if experiment_set_ids_experiments is None:
            experiment_set_ids_experiments = experiment_config.get('experiment_set_ids_experiments', [])

        if experiment_set_ids_simulations is None:
            experiment_set_ids_simulations = experiment_config.get('experiment_set_ids_simulations', [])

        assert not (len(experiment_set_ids_experiments)==0 and len(experiment_set_ids_simulations)==0),\
            "At least one of the experiment_set_ids_experiments or experiment_set_ids_simulations must be non-empty"


        all_results_list = []
        for simulation, ids_list in zip([False, True], [experiment_set_ids_experiments, experiment_set_ids_simulations]):
            if len(ids_list) == 0:
                continue

          #  noiseless_simulation = experiment_config.get('noiseless_simulation', simulation)

            experiments_folders_hierarchy = main_folders_hierarchy.copy()#+get_standard_subfolders_hierarchy_sub(backend_name=backend_name,
                                                  # simulation=simulation,
                                                  # noiseless_simulation=noiseless_simulation,
                                                  # hamiltonian_class_description=hamiltonian_class_description)
            for experiment_set_id in tqdm(ids_list,disable = not show_progress_bar, desc=f'Reading Datasets (sim = {simulation})'):

                logger_kwargs_main = {'experiment_folders_hierarchy': experiments_folders_hierarchy,
                                      'experiment_set_id': experiment_set_id,  # Used to group the experiments
                                      }
                results_reader = ResultsLogger(**logger_kwargs_main,
                                               do_not_create_experiment_ids=True)


                _post = self._df_processing_functions.copy()
                results_df_i = results_reader.gather_results(data_type=main_data_type,
                                                            merge_instances_metadata_data_type=merge_instances_metadata_data_type,
                                                            experiment_instance_ids_to_skip=faulty_experiment_ids,
                                                            return_none_if_not_found=True,
                                                             df_procesing_functions=_post,
                                                             show_progress_bar=show_progress_bar,
                                                             number_of_threads=1,
                                                             file_name_filter_function=_combined_file_name_filter
                                                            )
                if results_df_i is None:
                    continue

                results_df_i['Simulation'] = simulation



                all_results_list.append(results_df_i)


        if len(all_results_list)==0:
            raise ValueError("No results found")

        df_res = pd.concat(all_results_list, axis=0, ignore_index=True)

        return df_res




    @staticmethod
    def get_circuit_depth_processing_function_IBM():
            def _circuit_depth_processing_single_instance(results_df_i):
                def __add_depths_keys(_key):
                    if _key not in results_df_i.columns:
                        results_df_i[_key] = 0.0
                    else:
                        results_df_i[_key] = results_df_i[_key].fillna(0.0)
                    return results_df_i

                for _key in ['CircuitDepth', 'CZCount', 'SXCount']:
                    __add_depths_keys(_key)
                return results_df_i

            return _circuit_depth_processing_single_instance




