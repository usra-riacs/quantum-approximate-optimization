# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import numpy as np
import pandas as pd
from typing import List, Optional
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV, LoggingLevel
from quapopt import ancillary_functions as anf
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian


class EnergiesTailsAnalyzer:
    def __init__(self,
                 cost_hamiltonian:ClassicalHamiltonian,
                 energies_histograms_df:pd.DataFrame):


        self._energies_histograms_df = energies_histograms_df
        self._energies_expanded_df = None
        self._cost_hamiltonian = cost_hamiltonian

    @property
    def energies_histograms_df(self):
        return self._energies_histograms_df

    @property
    def energies_expanded_df(self):
        if self._energies_expanded_df is None:
            self._energies_expanded_df = self._expand_histogram_df()

        return self._energies_expanded_df


    def get_best_energies(self,
                          grouping_columns:Optional[List[str]]=None,):
        df_res = self.energies_histograms_df
        df_best = anf.contract_dataframe_with_minmax_values(df=df_res,
                                                            variable_name=SNV.Energy.id_long,
                                                            grouping_columns=grouping_columns,
                                                            find_maximal_value=False)
        return df_best


    def _expand_histogram_df(self,
                             df_histogram:Optional[pd.DataFrame]=None,):
        # raise NotImplementedError("FINISH THIS")

        if df_histogram is None:
            df_histogram = self.energies_histograms_df

        return anf.expand_histogram_dataframe(df_histogram=df_histogram,
                                              count_column=SNV.Count.id_long)

    def get_mean_energies(self,
                          grouping_columns:Optional[List[str]]=None,
                          columns_to_skip:Optional[List[str]]=None):


        df_mean = anf.contract_dataframe_with_aggregating_functions(df=self.energies_expanded_df,
                                                                    functions_to_apply=['mean'],
                                                                    grouping_columns=grouping_columns,
                                                                    columns_to_skip=columns_to_skip)

        return df_mean


    def get_energy_quantiles(self,
                             number_of_samples_list:List[int],
                             grouping_columns: Optional[List[str]] = None,
                             #columns_to_skip: Optional[List[str]] = None
                             ):


        df_results = self.energies_expanded_df
        if grouping_columns is None:
            df_grouped = [(None, df_results)]
        else:
            df_grouped = df_results.groupby(grouping_columns)

        all_dfs = []
        #let's iterate over each group
        for _, group in df_grouped:
            group:pd.DataFrame = group.copy()

            energies_group = group[SNV.Energy.id_long].to_numpy()
            energies_group.sort()

            group_header = group.iloc[[0]]
            for number_of_samples in number_of_samples_list:
                #we want only the first "number_of_samples" rows
                df_s = group_header.copy()
                df_s[SNV.TailSize.id_long] = [number_of_samples]
                df_s.drop(columns=[SNV.Energy.id_long], inplace=True)
                tail_mean = np.mean(energies_group[:number_of_samples])
                df_s[f"{SNV.Energy.id_long}_mean"] = tail_mean

                all_dfs.append(df_s)

        df_tails = pd.concat(all_dfs,ignore_index=True)

        return df_tails
