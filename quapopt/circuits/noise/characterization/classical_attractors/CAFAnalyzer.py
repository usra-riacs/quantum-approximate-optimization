# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)



import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from quapopt.circuits.noise.characterization.classical_attractors import (_standardized_table_name_CAF)

from quapopt.data_analysis.data_handling import (STANDARD_NAMES_DATA_TYPES as SNDT)

from quapopt.data_analysis.data_handling.io_utilities.results_logging import ResultsLogger
from quapopt import ancillary_functions as anf


from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import (ClassicalMeasurementNoiseSampler)
from quapopt.circuits.noise.characterization.measurements.DDOT import DDOTAnalyzer
from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go
import plotly.io as pio

from quapopt.data_analysis.data_handling import (STANDARD_NAMES_VARIABLES as SNV)


class CAFAnalyzer:
    def __init__(self,
                 #mirror_circuit_repeats_list: List[int],
                 #delays_list: List[float],
                 results_list=None,
                 results_logger_kwargs: Optional[dict] = None,
                 noisy_sampler_post_processing: ClassicalMeasurementNoiseSampler = None,
                 noisy_sampler_post_processing_rng=None,
                # uuid=None,
                 computation_backend: Optional[str] = None,
                 append_results_df: Optional[List[pd.DataFrame]] = None,
                 experiment_instances_ids:Optional[List[str]]=None,

                 old_version:bool=False):

        if computation_backend is None:
            computation_backend = 'numpy'

        if computation_backend == 'cupy':
            import cupy as bck
        elif computation_backend == 'numpy':
            import numpy as bck
        else:
            raise ValueError(
                f"Computation backend_computation {computation_backend} not supported. Supported backends are: cupy, numpy.")

        self._bck = bck
        self._computation_backend = computation_backend
        #self._delays_list = delays_list
        #self._mirror_circuit_repeats_list = mirror_circuit_repeats_list

        if results_logger_kwargs is None:
            assert results_list is not None, \
                "If you don't provide results logger kwargs, you need to provide circuit labels and bitstrings arrays."
            self._results_logger = None

            raise NotImplementedError("Non-read results not implemented yet")
        else:
            assert results_list is None, ("If results logger is provided, "
                                          "the results list should be None.")

            self._results_logger = ResultsLogger(**results_logger_kwargs)

            #print(self._results_logger.experiment_folders_hierarchy)


            results_df = self.read_results(old_version=old_version,
                                           experiment_instances_ids=experiment_instances_ids)
            anf.cool_print("GOT DATA...", 'Pre-processing', 'green')
            #print(results_df.columns)

            # results_df.drop(columns=['MCType', 'MCAnsatzDescription',

            #                          'MCAngles', 'Backend', 'Simulated',
            #                          'NumberOfQubits', 'NumberOfSamples',
            #                          'RNGSeed-Hamiltonian', 'RNGSeed-Angles',
            #                          'RNGSeed-Circuits',
            #                          'CompilationMetadata',
            #                         # 'ErrorMitigationMetadata',
            #                          'QubitsChain',
            #                          'LoggerMetadata'
            #                          ],
            #                 inplace=True)

            unique_jobs = sorted(results_df[SNV.JobId.id_long].unique())
            all_results = []

            for circuit_index in unique_jobs:
                df_i = results_df[results_df[SNV.JobId.id_long] == circuit_index].copy()

                #print(df_i.columns)
                unique_delay_schedules = sorted(df_i['DSDescription'].unique())
                unique_mcr = sorted(df_i['MCRepeats'].unique())
                unique_circuit_labels = df_i[SNV.CircuitLabel.id_long].unique()

                #print(type(unique_circuit_labels[0]), type(unique_delay_schedules[0]), type(unique_mcr[0]))
                assert len(unique_delay_schedules) == 1, "The delays should be unique for each circuit index."
                assert len(unique_mcr) == 1, "The mirror circuit repeats should be unique for each circuit index."
                assert len(unique_circuit_labels) == 1, "The circuit labels should be unique for each circuit index."

                dsd_i = unique_delay_schedules[0]
                mcr_i = unique_mcr[0]

                #print(dsd_i,mcr_i)

                circuit_label = unique_circuit_labels[0]
                circuit_bitstrings = bck.array(df_i[SNV.Bitstring.id_long].tolist(), dtype=np.int32)
                circuit_counts = bck.array(df_i[SNV.Count.id_long].tolist(), dtype=int)

                #print(circuit_bitstrings.shape)
                #raise KeyboardInterrupt

                df_res_here = pd.DataFrame(data={'DSDescription': [dsd_i] * len(circuit_counts),
                                                 'MCRepeats': [mcr_i] * len(circuit_counts),
                                                 SNV.CircuitLabel.id_long: [tuple(circuit_label)] * len(circuit_counts),
                                                 SNV.Bitstring.id_long: circuit_bitstrings.tolist(),
                                                 SNV.Count.id_long: circuit_counts.tolist()})


                all_results.append(df_res_here)

            if append_results_df is not None:
                for df in append_results_df:
                    all_results.append(df)
          #  print('hej', [type(x) for x in all_results])

            df_results = pd.concat(all_results, axis=0)

            anf.cool_print("Pre-processing finished", '...', 'green')

        self._df_results = df_results

        # self._colors_list = ['forestgreen', 'cornflowerblue', 'gold', 'coral', 'red',
        #                      'darkviolet', 'crimson', 'darkorange', 'darkkhaki', 'darkslateblue',
        #                      ]
        self._colors_list = [
            # greens
            "#00FF00", "#33CC33", "#669966",
            # blues
            "#6699FF", "#3366CC", "#003399",
            # reds
            "#FF6666", "#CC3333", "#990000"]

        # allowed options: ['', '/', '\\', 'x', '-', '|', '+', '.']
        self._patterns_list = [dict(shape=''), dict(shape='x'), dict(shape='\\'),
                               dict(shape='/'), dict(shape='-'), dict(shape='|'),
                               dict(shape='+'), dict(shape='.'), ]

        self._df_noise_matrices = None
        self._number_of_qubits = len(self._df_results['CircuitLabel'].values[0])

    def calculate_noise_matrices(self,
                                 subsets_2q=None,
                                 include_2q=False):

        if subsets_2q is None:
            subsets_2q = [(i, i + 1) for i in range(0, self._number_of_qubits - 1, 2)]
            subsets_2q += [(i, i + 1) for i in range(1, self._number_of_qubits - 1, 2)]

        df_results = self._df_results

        bck = self._bck

        all_input_states = list(df_results['CircuitLabel'].unique())
        #delays_list = self._delays_list
        #mirror_circuit_repeats_list = self._mirror_circuit_repeats_list
        #print(df_results['DSDescription'].unique())
        delays_list = sorted(df_results['DSDescription'].unique())
        mirror_circuit_repeats_list = sorted(df_results['MCRepeats'].unique())


        all_dfs = []
        for delay in delays_list:
            for mcr in mirror_circuit_repeats_list:
                df_i = df_results[(df_results['DSDescription'] == delay) & (df_results['MCRepeats'] == mcr)].copy()

                if df_i.empty:
                    continue

                bitstrings_arrays = []
                for input_state in all_input_states:
                    df_i_j = df_i[df_i['CircuitLabel'] == input_state].copy()
                    bts_ij = bck.array(df_i_j[SNV.Bitstring.id_long].tolist(), dtype=bck.uint8)
                    counts_ij = bck.array(df_i_j[SNV.Count.id_long].tolist(), dtype=int)
                    bts_ij = bck.repeat(bts_ij, counts_ij, axis=0)
                    bitstrings_arrays.append(bts_ij)

                bitstrings_arrays = bck.array(bitstrings_arrays, dtype=bck.uint8)

                ddot_analyzer_i = DDOTAnalyzer(circuits_labels=all_input_states,
                                               bitstrings_arrays=bitstrings_arrays, )

                noise_matrices_1q = list(ddot_analyzer_i.get_1q_noise_matrices())
                # noise_matrix_1q_av = noise_matrices_1q.sum(axis=0) / len(noise_matrices_1q)

                # noise_matrix_2q_av = noise_matrices_2q.sum(axis=0) / len(noise_matrices_2q)

                df_res_here = pd.DataFrame({'DSDescription': [delay] * self._number_of_qubits,
                                            'MCRepeats': [mcr] * self._number_of_qubits,
                                            'Subset': [(i,) for i in range(self._number_of_qubits)],
                                            'NoiseMatrix': noise_matrices_1q})

                if include_2q:
                    noise_matrices_2q = list(ddot_analyzer_i.get_2q_noise_matrices(subsets_2q=subsets_2q,
                                                                                   show_progress_bar=False))

                    df_2q = pd.DataFrame(data={
                        'DSDescription': [delay] * len(subsets_2q),
                        'MCRepeats': [mcr] * len(subsets_2q),
                        'Subset': subsets_2q,
                        'NoiseMatrix': noise_matrices_2q})
                    df_res_here = pd.concat([df_res_here, df_2q], axis=0)
                all_dfs.append(df_res_here)

        self._df_noise_matrices = pd.concat(all_dfs, axis=0)

        return self._df_noise_matrices

    @property
    def results(self):
        """
        Returns the results of the analysis.
        :return:
        """
        return self._df_results

    @property
    def noise_matrices(self):
        """
        Returns the average noise matrices.
        :return:
        """
        if self._df_noise_matrices is None:
            self.calculate_noise_matrices()
        return self._df_noise_matrices

    def read_results(self,
                     experiment_instances_ids=None,
                     old_version=False
                     ):
        if old_version:
            return self._read_results_depreciated()

        #We implemented a new version of results logger, which is more general and robust.
        #So we read the results using the new version.
        df_results = self._results_logger.gather_results(data_type=SNDT.BitstringsHistograms,
                                                         experiment_instance_ids=experiment_instances_ids)
        df_results['CircuitLabel'] = df_results['CircuitLabel'].apply(anf.eval_string_tuple_to_tuple)


        return df_results


    def _read_results_depreciated(self):
        """
        This is a backward-compatibility function for the old version of the results logger.
        :return:
        """


        dataframes = []

        # print(self._results_logger.get_base_path())
        bitstrings_folder = self._results_logger.get_absolute_path_of_data_type(data_type=SNDT.BitstringsHistograms)
        metadata_folder = self._results_logger.get_absolute_path_of_data_type(data_type=SNDT.CircuitsMetadata)

        files_bitstrings = sorted(os.listdir(bitstrings_folder))
        files_metadata = sorted(os.listdir(metadata_folder))

        # TODO(FBM): modernize this function

        for file_name_bts in files_bitstrings:
            split_name_bts = file_name_bts.split(self._results_logger._tnps)
            table_name_main_bitstrings = self._results_logger._tnps.join(split_name_bts[0:-1])

            for file_name_metadata in files_metadata:
                split_name_metadata = file_name_metadata.split(self._results_logger._tnps)
                table_name_main_metadata = self._results_logger._tnps.join(split_name_metadata[0:-1])
                if table_name_main_bitstrings == table_name_main_metadata:
                    break

            if table_name_main_bitstrings != table_name_main_metadata:
                raise FileNotFoundError(
                    f"Metadata file {file_name_metadata} does not match bitstrings file {file_name_bts}.")

            df_here_bts = self._results_logger.read_results(table_name=table_name_main_bitstrings,
                                                            data_type=SNDT.BitstringsHistograms,
                                                            return_none_if_not_found=True,
                                                            filter_by_experiment_set=False,
                                                            append_experiment_set_suffix=False)
            if df_here_bts is None:
                continue
            df_here_metadata = self._results_logger.read_results(table_name=table_name_main_bitstrings,
                                                                 data_type=SNDT.CircuitsMetadata,
                                                                 filter_by_experiment_set=False,
                                                                 append_experiment_set_suffix=False)

            if df_here_metadata.empty:
                raise FileNotFoundError(f"Metadata file {file_name_metadata} is empty.")

            # assert len(df_here_metadata) == 1, ("The metadata dataframe should be of length 1, "
            #                                     "if it is not, you have to provide uuid",
            #                                     f"{df_here_metadata}")

            if df_here_bts.empty:
                continue

            df_here_metadata['MCRepeats'] = df_here_metadata['MCRepeats'].astype(int)

            for col in df_here_metadata.columns:
                if col not in df_here_bts.columns:
                    df_here_bts[col] = df_here_metadata[col].values[0]
            dataframes.append(df_here_bts)

            # assert len(df_here_metadata) == 1, ("The metadata dataframe should be of length 1")

        results_df = pd.concat(dataframes, axis=0)
        results_df['CircuitLabel'] = results_df['CircuitLabel'].apply(anf.eval_string_tuple_to_tuple)

        return results_df



    def plot_hamming_weight_histograms(self,
                                       df_results: Optional[pd.DataFrame] = None,
                                       # size_bin:float = 0.05,
                                       fom_names_list: Tuple[str, ...] = ('HammingWeight',
                                                                          'HammingDistance'),
                                       add_info_title: Optional[str] = None,
                                       name_suffix='',
                                       folder_path: Optional[str] = None,
                                       input_states_to_plot: Optional[Tuple[int, ...]] = None,
                                       grouped_by='mcr',
                                       save_html=True,
                                       delays_to_skip = None

                                       ):
        if delays_to_skip is None:
            delays_to_skip =[]
        bck = self._bck

        pio.templates.default = "plotly_white"  # ← forces light theme globally

        if df_results is None:
            df_results = self._df_results

        patterns_list = self._patterns_list
        colors_list = self._colors_list

        if input_states_to_plot is None:
            input_states_to_plot = df_results['CircuitLabel'].unique()

        delays_list = sorted(df_results['DSDescription'].unique())
        mirror_circuit_repeats_list = sorted(df_results['MCRepeats'].unique())
        number_of_columns = 1
        number_of_rows = 1
        # number_of_rows = int(np.ceil(len(input_states_to_plot) / number_of_columns))

        # xbins_dict = dict(size=size_bin)

        if folder_path is None:
            folder_path = f'../temp/figs/CAF/'

        if add_info_title is None:
            add_info_title = ''

        datapoint_per_plot = len(delays_list) * len(mirror_circuit_repeats_list)

        number_of_qubits = len(input_states_to_plot[0])
        known_input_states = {tuple([0] * number_of_qubits): '|0...0>',
                              tuple([1] * number_of_qubits): '|1...1>',
                              tuple([0, 1] * int(number_of_qubits / 2)): '|01...01>',
                              tuple([1, 0] * int(number_of_qubits / 2)): '|10...10>'
                              }

        lines_list = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

        for fom_name in fom_names_list:
            for idx, input_state in enumerate(input_states_to_plot):
                row_idx = idx // number_of_columns + 1
                col_idx = idx % number_of_columns + 1

                row_idx = 1
                col_idx = 1
                # [
                #     f"Input state:{known_input_states[input_states_to_plot[i]]}; {add_info_title}"
                #     if input_states_to_plot[
                #            i] in known_input_states else f"Input state:|X>; {add_info_title}"
                #     for i in range(len(input_states_to_plot))]

                df_is = df_results[df_results['CircuitLabel'] == input_state]

                input_state_array = bck.array(input_state, dtype=bck.uint8)

                idx_data = 0
                for idx_delay, delay in enumerate(delays_list):
                    if delay in delays_to_skip:
                        continue

                    subplot_titles = [
                        f"Input state:{known_input_states[input_state]}; ({add_info_title})"
                        if input_state in known_input_states else f"Input state:|X>"]

                    fig = make_subplots(rows=number_of_rows,
                                        cols=number_of_columns,
                                        subplot_titles=subplot_titles)

                    # if delay == 10:
                    #     delay_name = '10 microseconds'

                    for idx_mcr, mcr in enumerate(mirror_circuit_repeats_list):

                        width_line = 0
                        trace_name = f"MCR: {mcr}; {delay}" if delay!='RandomSampling' else "Random Sampling"
                        #trace_name = f"MCR: {mcr}" if mcr != -1 else "Random"

                        if grouped_by == 'delay':
                            legendgroup = f"g:DSDescription:{delay}"
                            # print(idx_delay,colors_list[idx_delay])

                            marker_dict = dict(

                                line=dict(color=colors_list[idx_mcr - 1],
                                          width=width_line,
                                          ),
                                # ),

                                pattern=patterns_list[idx_delay],
                                color=colors_list[idx_mcr - 1], )

                        elif grouped_by == 'mcr':
                            legendgroup = f"g:MCR:{mcr}"
                            marker_dict = dict(

                                line=dict(color=colors_list[idx_mcr - 1],
                                          width=width_line,
                                          ),
                                # ),

                                pattern=patterns_list[idx_delay],
                                color=colors_list[idx_mcr - 1], )


                        else:
                            raise ValueError(f"Unknown grouping option: {grouped_by}")

                        df_i = df_is[(df_is['DSDescription'] == delay) & (df_is['MCRepeats'] == mcr)].copy()
                        if df_i.empty:
                            continue

                        bitstrings_array = bck.array(df_i[SNV.Bitstring.id_long].tolist(), dtype=bck.uint8)
                        bitstrings_counts = bck.array(df_i[SNV.Count.id_long].tolist(), dtype=int)

                        # print(df_i)
                        # print(bitstrings_array.shape)
                        # print(bitstrings_counts.shape)

                        if fom_name in ['hamming_weight', 'HammingWeight']:
                            nice_name = 'Hamming Weight'
                            fom_values = np.sum(bitstrings_array, axis=1)
                        elif fom_name in ['hamming_distance', 'HammingDistance']:
                            fom_values = np.sum(bitstrings_array ^ input_state_array, axis=1)
                            nice_name = 'Hamming Distance'
                        else:
                            raise ValueError(f"Unknown fom_name: {fom_name}")

                        if mcr == -1:
                            # code for random data
                            marker_dict = dict(
                                line=dict(width=width_line,
                                          color='black',
                                          ),

                                color='black')
                            legendgroup = 'g:Random'
                            trace_name = "Random"

                        fom_values = bck.repeat(fom_values, bitstrings_counts, axis=0)

                        counts_hist, bin_edges = np.histogram(a=fom_values,
                                                              range=(0, number_of_qubits + 1),
                                                              bins=number_of_qubits + 1,
                                                              density=False)

                        # Calculate bar positions
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        counts_normalized = counts_hist  # / counts_hist.max()
                        # fig.add_trace(go.Scatter(x=bin_centers,
                        #                          y=counts_normalized,
                        #                          mode='lines',
                        #                          marker=marker_dict,
                        #                          line=dict(shape='hv',
                        #                                    color=marker_dict['color'],
                        #                                    width=3),
                        #                          opacity=1.0,
                        #                         #name=trace_name,
                        #                         showlegend=False,
                        #                         legendgroup=legendgroup))
                        fig.add_trace(go.Histogram(x=fom_values,
                                                   xbins=dict(start=0,
                                                              end=number_of_qubits + 1,
                                                              size=1),
                                                   marker=marker_dict,
                                                   opacity=1.0,
                                                   name=trace_name,
                                                   showlegend=True,
                                                   # legendgroup=legendgroup,
                                                   nbinsx=number_of_qubits,
                                                   # update bar width
                                                   # width=0.8,
                                                   ),
                                      row=row_idx,
                                      col=col_idx,
                                      )

                        idx_data += 1

                    if fom_name in ['hamming_weight', 'HammingWeight']:
                        # let's add vertical line in the middle
                        middle_hw = int(number_of_qubits / 2)
                        fig.add_shape(type="line",
                                      x0=middle_hw,
                                      y0=0,
                                      x1=middle_hw,
                                      y1=bck.sum(bitstrings_counts) / 10,
                                      line=dict(color="black", width=2, dash="dash"),
                                      row=row_idx,
                                      col=col_idx,

                                      )

                    color_grid = 'rgba(0,0,0,0.15)'  # light gray
                    linewidht_grid = 2.0  # px

                    # update x axis
                    fig.update_xaxes(title_text=f"{nice_name}",
                                     row=row_idx,
                                     col=col_idx,
                                     range=(-1, self._number_of_qubits + 1),
                                     showgrid=True,  # make sure they’re on
                                     gridcolor=color_grid,
                                     gridwidth=linewidht_grid,  # thicker stroke
                                     griddash="solid",
                                     linecolor="rgba(0,0,0,0.0)",
                                     linewidth=linewidht_grid,  # thickness in px,
                                     zerolinecolor=color_grid,
                                     zerolinewidth=linewidht_grid,  # thickness in px
                                     )
                    # update y axis
                    fig.update_yaxes(title_text="Counts",
                                     row=row_idx,
                                     col=col_idx,
                                     showgrid=True,  # make sure they’re on
                                     gridcolor=color_grid,
                                     gridwidth=linewidht_grid,  # thicker stroke
                                     griddash="solid",
                                     linecolor="rgba(0,0,0,0.0)",
                                     linewidth=linewidht_grid,  # thickness in px,
                                     zerolinecolor=color_grid,
                                     zerolinewidth=linewidht_grid,  # thickness in px
                                     )

                    fig.update_layout(hoverlabel_namelength=-1,
                                      height=number_of_rows * 1600,
                                      width=number_of_columns * 1600,
                                      legend=dict(
                                          # traceorder='grouped',
                                          # groupclick='togglegroup',
                                          # itemdoubleclick='toggle',
                                          yanchor='bottom',
                                          y=1.1,
                                          xanchor='center',
                                          x=0.5,
                                          orientation='h',
                                          # entrywidth= 1000,
                                          font=dict(size=24)

                                      ),
                                      bargap=0.0000001,
                                      font=dict(size=26),

                                      )
                    fig.update_annotations(font_size=26)

                    state_name = known_input_states[input_state] if input_state in known_input_states else '|X>'
                    plot_name = f'Histograms_{fom_name};{state_name};delay={delay}'
                    file_name = f'{folder_path}{plot_name};{name_suffix}'

                    os.makedirs('/'.join(file_name.split('/')[0:-1]), exist_ok=True)
                    fig.write_image(f'{file_name}.png',
                                    format='png',
                                    scale=3.0)
                    if save_html:
                        plotly.offline.plot(fig,
                                            filename=f'{file_name}.html')

                # plotly.offline.plot(fig,
                #                     filename=f'{file_name}.html')

    def plot_average_noise_rates(self,
                                 noise_matrices_df: Optional[pd.DataFrame] = None,
                                 grouped_by='delay',
                                 folder_path: Optional[str] = None,
                                 additional_title_string: Optional[str] = None,
                                 additional_figure_name_string: Optional[str] = None,
                                 save_html=True,
                                 plot_noise_bias=False


                                 ):
        add_2q = False,

        if folder_path is None:
            folder_path = f'../temp/figs/CAF/'

        if noise_matrices_df is None:
            noise_matrices_df = self.noise_matrices

        df_1q = noise_matrices_df[noise_matrices_df['Subset'].apply(lambda x: len(x) == 1)].copy()
        df_2q = noise_matrices_df[noise_matrices_df['Subset'].apply(lambda x: len(x) == 2)].copy()

        from quapopt import ancillary_functions as anf

        unique_variables_columns_names = ['DSDescription', 'MCRepeats']

        df_1q['p(0->0)'] = df_1q['NoiseMatrix'].apply(lambda x: x[0, 0])
        df_1q['p(1->1)'] = df_1q['NoiseMatrix'].apply(lambda x: x[1, 1])

        df_1q['p(0->0)-p(1->1)'] = df_1q.apply(lambda row:row['p(0->0)'] - row['p(1->1)'], axis=1)


        df_1q.drop(columns=['NoiseMatrix', 'Subset'], inplace=True)

        df_1q_av = anf.contract_dataframe_with_aggregating_functions(df=df_1q,
                                                         functions_to_apply=['mean', 'median','std', 'min', 'max'],
                                                         grouping_columns=['DSDescription', 'MCRepeats'])


        # print(df_1q_av)


        df_2q['p(00->00)'] = df_2q['NoiseMatrix'].apply(lambda x: x[0, 0])
        df_2q['p(01->01)'] = df_2q['NoiseMatrix'].apply(lambda x: x[1, 1])
        df_2q['p(10->10)'] = df_2q['NoiseMatrix'].apply(lambda x: x[2, 2])
        df_2q['p(11->11)'] = df_2q['NoiseMatrix'].apply(lambda x: x[3, 3])
        df_2q.drop(columns=['NoiseMatrix', 'Subset'], inplace=True)
        df_2q_av = anf.contract_dataframe_with_aggregating_functions(df=df_2q,
                                                         functions_to_apply=['mean','median', 'std', 'min', 'max'],
                                                         grouping_columns=['DSDescription', 'MCRepeats'])

        # print(df_2q_av)
        # raise KeyboardInterrupt


        delays_list = list(noise_matrices_df['DSDescription'].unique())
        random_sampling_str = None
        for x in delays_list:
            if 'randomsampling' in x.lower():
                random_sampling_str = x
                break

        # if random_sampling_str is not None:
        #remove random_sampling_str from delays_list and put it at the end of the list
        if random_sampling_str is not None:
            delays_list.remove(random_sampling_str)

        def _sorting_function(ds_description:str):
            #we have two strings that identify the delay schedule: a) DelayBase=X; b) mean~X; We want to detect them and sort accordingly
            if 'DelayBase=' in ds_description:
                return ds_description.split('DelayBase=')[1]
            elif 'mean~' in ds_description:
                _split = ds_description.split('mean~')[1]

                float_type = None
                for digits in [1,2,3,4,5]:
                    try:
                        float_type = float(_split[0:digits])
                    except(ValueError, TypeError):
                        continue
               # print(float_type)
                return float_type
            else:
                return ds_description

        delays_list = sorted(delays_list,key = _sorting_function)
        if random_sampling_str is not None:
            delays_list.append(random_sampling_str)


        def _rename_function(ds_description:str):
            if 'RandomSampling' in ds_description:
                return "RandomSampling"


            if 'DelayBase=' in ds_description:
                _split = ds_description.split('DelayBase=')[1]


                return "Delay="+_split
            elif 'mean~' in ds_description:
                _split = ds_description.split('mean~')[1]
                return "Delay="+_split
            else:
                return ds_description

        #delays_list = [_rename_function(x) for x in delays_list]




        number_of_columns = 1
        number_of_rows = 1

        figs_add_list = ['1q', '2q'] if add_2q else ['1q']


        if plot_noise_bias:
            sub_names_1q = ['p(0->0)-p(1->1)']


        else:
            sub_names_1q = ['p(0->0)', 'p(1->1)']


        for sub_name_list, fig_add in zip([sub_names_1q,
                                           ['p(11->11)',
                                            'p(00->00)',
                                            'p(01->01)',
                                            'p(10->10)', ]],
                                          figs_add_list):

            # subplot_titles = [f'1q readout fidelity; {add_info_title}',
            #                   f'2q readout fidelity; {add_info_title}']

            if fig_add == '1q':
                if plot_noise_bias:
                    subplot_titles = [f"1Q Noise Bias<br>({additional_title_string})"]

                else:

                    subplot_titles = [f"1Q 'State Preservation Fidelity'<br>({additional_title_string})"]
            elif fig_add == '2q':
                subplot_titles = [f"2Q 'State Preservation Fidelity'<br>({additional_title_string})"]

            fig = make_subplots(rows=number_of_rows,
                                cols=number_of_columns,
                                subplot_titles=subplot_titles)

            symbols_list = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down']
            colors_list = ['forestgreen', 'cornflowerblue', 'crimson', 'darkviolet', 'coral', 'crimson', 'darkorange', 'darkkhaki', 'darkslateblue']
            lines_list = ['solid', 'dot', 'dash', 'longdash', 'dashdot']


            if plot_noise_bias:
                yrange = (-0.1, 0.2)
            else:
                yrange = (-0.01, 1.01)

            for idx_delay, delay in enumerate(delays_list):
                df_1q_av_i = df_1q_av[df_1q_av['DSDescription'] == delay].copy()
                df_2q_av_i = df_2q_av[df_2q_av['DSDescription'] == delay].copy()

                #print(df_1q_av_i)

                mcrs_i_1q = df_1q_av_i['MCRepeats'].values
                mcrs_i_2q = df_2q_av_i['MCRepeats'].values

                for sub_idx, sub_name in enumerate(sub_name_list):
                    if sub_name in ['p(1->1)', 'p(0->0)', 'p(0->0)-p(1->1)']:
                        fom_i_mean = df_1q_av_i[f"{sub_name}_median"].values
                        fom_i_std = df_1q_av_i[f"{sub_name}_std"].values
                        fom_i_min = df_1q_av_i[f"{sub_name}_min"].values
                        fom_i_max = df_1q_av_i[f"{sub_name}_max"].values
                        #lets plot min, max values
                        # array_err_y = np.array([fom_i_max - fom_i_mean,
                        #                         fom_i_mean - fom_i_min
                        #                         ])
                        # arrayminus_y = np.array([fom_i_min])
                        # arrayplus_y = np.array([fom_i_max])
                        # #
                        #
                        # print(arrayminus_y)
                        # print(fom_i_mean)
                        # print(arrayplus_y)
                        # raise KeyboardInterrupt

                        mcrs_i = mcrs_i_1q
                        row_idx = 1

                    elif sub_name in ['p(00->00)', 'p(11->11)', 'p(01->01)', 'p(10->10)', ]:
                        fom_i_mean = df_2q_av_i[f"{sub_name}_mean"].values
                        fom_i_std = df_2q_av_i[f"{sub_name}_std"].values
                        mcrs_i = mcrs_i_2q
                        row_idx = 1

                    else:
                        raise ValueError(f"Unknown sub_name: {sub_name}")

                    color = colors_list[idx_delay]
                    symbol = symbols_list[sub_idx]
                    line = lines_list[sub_idx]

                    if grouped_by == 'delay':
                        legendgroup = f"g:{row_idx};DSDescription:{delay}"
                    elif grouped_by == 'probability':
                        legendgroup = f"g:{row_idx};Prob:{sub_name}"
                    else:
                        raise ValueError(f"Unknown grouping option: {grouped_by}")
                    # now we will add the scatter plot
                    fig.add_trace(go.Scatter(x=mcrs_i,
                                             y=fom_i_mean,
                                             # add error bars
                                             error_y=dict(type='data',
                                                          array=fom_i_std,
                                                         # arrayminus=arrayminus_y,
                                                          visible=True,
                                                          width=8),
                                             mode='markers+lines',
                                             name=f"{sub_name}; {_rename_function(delay)}",
                                             marker=dict(color=color,
                                                         symbol=symbol,
                                                         size=8
                                                         ),
                                             line=dict(color=color,
                                                       width=8,
                                                       dash=line,
                                                       ),
                                             showlegend=True,
                                             # legendgroup=legendgroup,
                                             ),
                                  row=row_idx,
                                  col=1,
                                  )


                    if plot_noise_bias:
                        #we want to plot a horizontal line from min circuit repetitions to max circuit repetitions,
                        # dotted and grey, corresponding to random sampling
                        #copilot, you there?
                        # well, I'm here
                        fig.add_hline(y=0,
                                      line=dict(color='gray',
                                                width=4,
                                                dash='dot'),
                                     # annotation='RandomSampling',
                                      row=row_idx,
                                      col=1,
                                      )
                    else:
                        fig.add_hline(y=0.5,
                                      line=dict(color='gray',
                                                width=4,
                                                dash='dot'),
                                     # annotation='RandomSampling',

                                      )







                    linewidht_grid = 2.
                    color_grid = "rgba(0,0,0,0.35)"  # any CSS‐style color string
                    fig.update_xaxes(title_text="Mirror Circuit Repetitions",
                                     row=row_idx,
                                     col=1,
                                     showgrid=True,  # make sure they’re on
                                     gridcolor=color_grid,
                                     gridwidth=linewidht_grid,  # thicker stroke
                                     griddash="solid",
                                     linecolor="rgba(0,0,0,0.0)",
                                     linewidth=linewidht_grid,  # thickness in px,
                                     zerolinecolor=color_grid,
                                     zerolinewidth=linewidht_grid,  # thickness in px
                                     # minor_gridcolor= color_grid,
                                     # minor_gridwidth=linewidht_grid,  # thickness in px
                                     )

                    if fig_add == '2q':
                        yname = f"Probability (State Preservation {fig_add})"
                    elif fig_add == '1q':
                        if plot_noise_bias:
                            yname = f"Noise Bias ({fig_add})"
                        else:
                            yname=f"Probability (State Preservation {fig_add})"



                    fig.update_yaxes(title_text=yname,
                                     row=row_idx,
                                     col=1,
                                     range=yrange,
                                     showgrid=True,  # make sure they’re on
                                     gridcolor=color_grid,
                                     gridwidth=linewidht_grid,  # thicker stroke
                                     griddash="solid",
                                     linecolor=color_grid,
                                     linewidth=linewidht_grid,  # thickness in px
                                     zerolinecolor=color_grid,
                                     zerolinewidth=linewidht_grid,  # thickness in px
                                     )

            fig.update_layout(hoverlabel_namelength=-1,
                              height=number_of_rows * 1600,
                              width=number_of_columns * 1600,
                              legend=dict(
                                  # traceorder='grouped',
                                  # groupclick='togglegroup',
                                  # itemdoubleclick='toggle',
                                  # let's make the legend a bit smaller
                                  # let's move it on the top of the figure
                                  yanchor='bottom',
                                  y=1.1 if fig_add == '1q' else 1.1,
                                  xanchor='center',
                                  x=0.5,
                                  # let's change legend's orientation
                                  orientation='h',
                                  # entrywidth= 1000,
                                  font=dict(size=30)

                              ),
                              bargap=0.0000001,
                              font=dict(size=40),
                              )
            fig.update_annotations(font_size=40)
            # fig.for_each_annotation(lambda a: a.update(text=subplot_titles[int(a.text) - 1],
            #                                            font=dict(size=36), ))

            if plot_noise_bias:
                prefix = 'NoiseBias'
            else:
                prefix = 'SpamFidelities'

            file_name = f'{prefix};{fig_add};{additional_figure_name_string}'
            os.makedirs(folder_path, exist_ok=True)

            if save_html:
                plotly.offline.plot(fig,
                                    filename=f'{folder_path}{file_name}.html')

            
            # Create PNG-optimized version with better sizing
            fig_png = go.Figure(fig)  # Create a copy to avoid modifying the original
            fig_png.update_layout(
                height=900,  # Increased height to accommodate legend above
                width=1000,  # Standard width
                legend=dict(
                    yanchor='bottom',
                    y=1.22,  # Place legend above the plot
                    xanchor='center',
                    x=0.5,  # Center horizontally
                    orientation='h',  # Horizontal orientation for space efficiency
                    font=dict(size=30),
                    bgcolor='rgba(255,255,255,0.95)',  # Semi-transparent background
                    bordercolor='rgba(0,0,0,0.2)',
                    borderwidth=1,
                    itemsizing='constant',  # Consistent item sizing
                    tracegroupgap=5,  # Reduce gap between groups
                    itemwidth=30  # Control item width
                ),
                font=dict(size=40),  # General font size
                margin=dict(l=80, r=40, t=200, b=60),  # Large top margin for legend
                title=dict(
                    font=dict(size=40),
                    y=0.98,  # Move title up slightly
                    yanchor='top'
                )
            )

            # Write PNG with explicit dimensions
            fig_png.write_image(file=f'{folder_path}{file_name}.png',
                            format='png',
                            width=1500,  # Explicit width in pixels
                            height=1600,  # Explicit height in pixels
                            scale=1.0)  # Don't scale, use explicit dimensions


            return fig