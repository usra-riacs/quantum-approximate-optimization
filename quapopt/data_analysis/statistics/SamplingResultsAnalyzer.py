# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import pandas as pd

from quapopt.data_analysis.statistics.EnergiesTailsAnalyzer import EnergiesTailsAnalyzer
from quapopt.data_analysis.data_handling.schemas.naming import (MAIN_KEY_VALUE_SEPARATOR as MKVS,
                                                                MAIN_KEY_SEPARATOR as MKS,
                                                                STANDARD_NAMES_VARIABLES as SNV,
                                                                STANDARD_NAMES_DATA_TYPES as SNDT)
from quapopt.data_analysis.data_handling import ResultsLogger, LoggingLevel

from typing import List, Optional, Type, Callable, Dict, Any
import numpy as np
from quapopt import ancillary_functions as anf
from quapopt.ancillary_functions.presets import read_or_generate_random_sampling_results
from tqdm.notebook import tqdm
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.data_analysis.visualization.PlotlySubplotsPlotter import PlotlySubplotsPlotter



class SamplingResultsAnalyzer:
    def __init__(self,
                 experiment_config:Optional[Dict[str,Any]]=None,
                 ):

        self._results_overview = None

        self._energies_histograms = None
        self._bitstrings_histograms = None

        self._energies_flat = None
        self._bitstrings_flat = None

        self._experiment_config = experiment_config
        self._results_plotter:Optional[PlotlySubplotsPlotter] = None



    @property
    def results_overview(self)->Optional[pd.DataFrame]:
        return self._results_overview

    @property
    def energies_histograms(self)->Optional[pd.DataFrame]:
        return self._energies_histograms

    @property
    def bitstrings_histograms(self)->Optional[pd.DataFrame]:
        return self._bitstrings_histograms

    @property
    def energies_flat(self)->Optional[pd.DataFrame]:
        return self._energies_flat
    @property
    def bitstrings_flat(self)->Optional[pd.DataFrame]:
        return self._bitstrings_flat


    @property
    def results_plotter(self)->Optional[PlotlySubplotsPlotter]:
        return self._results_plotter

    def set_results_plotter(self,
                            plotter:PlotlySubplotsPlotter):
        self._results_plotter = plotter

    def set_data_type(self,
                      data_type:Type[SNDT],
                      results_df:pd.DataFrame,
                      ):

        if data_type == SNDT.OptimizationOverview:
            self._results_overview = results_df

        elif data_type == SNDT.EnergiesHistograms:
            self._energies_histograms = results_df
            self._energies_flat = anf.expand_histogram_dataframe(df_histogram=self._energies_histograms)
        elif data_type == SNDT.Energies:
            self._energies_flat = results_df
            self._energies_histograms = anf.transform_to_histogram_dataframe(df_flat=results_df,
                                                                             value_column=SNV.Energy.id_long,
                                                                             )
        elif data_type == SNDT.BitstringsHistograms:
            self._bitstrings_histograms = results_df
            self._bitstrings_flat = anf.expand_histogram_dataframe(df_histogram=self._bitstrings_histograms)
        elif data_type == SNDT.Bitstrings:
            self._bitstrings_flat = results_df
            self._bitstrings_histograms = anf.transform_to_histogram_dataframe(df_flat=results_df,
                                                                             value_column=SNV.Bitstring.id_long,
                                                                             )
        else:
            raise ValueError(f"Unsupported data type {data_type}. Only the following data types are supported: OptimizationOverview, EnergiesHistograms, Energies, BitstringsHistograms, Bitstrings")


    def get_data_type(self,
                      data_type:Type[SNDT],
                      )->Optional[pd.DataFrame]:

        if data_type == SNDT.OptimizationOverview:
            return self._results_overview
        elif data_type == SNDT.EnergiesHistograms:
            return self._energies_histograms
        elif data_type == SNDT.Energies:
            return self._energies_flat
        elif data_type == SNDT.BitstringsHistograms:
            return self._bitstrings_histograms
        elif data_type == SNDT.Bitstrings:
            return self._bitstrings_flat
        else:
            raise ValueError(f"Unsupported data type {data_type}. Only the following data types are supported: OptimizationOverview, EnergiesHistograms, Energies, BitstringsHistograms, Bitstrings")


    def _initialize_results_from_config(self,
                                        data_types_of_interest:List[Type[SNDT]],
                                        *args,
                                        **kwargs,
                                            ):

        for data_type in data_types_of_interest:
            df_res = self.read_results(main_data_type=data_type,
                                    *args,
                                        **kwargs
                                            )


            self.set_data_type(data_type=data_type,
                               results_df=df_res)

    def initialize_results_from_data(self,
                                    data_types_of_interest:List[Type[SNDT]],
                                    dataframes:List[pd.DataFrame],

                                          ):

        for data_type,df_res in zip(data_types_of_interest,dataframes):
            self.set_data_type(data_type=data_type,
                               results_df=df_res)

    def read_results(self,
                     main_data_type:Type[SNDT],
                     *args,
                     **kwargs):
        raise NotImplementedError("This method should be implemented in a child class")
