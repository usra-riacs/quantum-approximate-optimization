# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

import itertools
import os
import time
from typing import List, Tuple, Optional, Any, Union

import numpy as np
from tqdm import tqdm

from quapopt.circuits.noise.characterization.tomography_tools import (TomographyType,
                                                                      TomographyGatesType)
from quapopt.circuits.noise.characterization.tomography_tools.OverlappingTomographyRunnerBase import \
    OverlappingTomographyRunnerBase

try:
    import cupy as cp
except(ImportError, ModuleNotFoundError):
    import numpy as cp
from quapopt.circuits.noise.simulation.ClassicalMeasurementNoiseSampler import (ClassicalMeasurementNoiseSampler,
                                                                                    MeasurementNoiseType)
from quapopt.circuits.gates import AbstractProgramGateBuilder, AbstractCircuit

from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go
from quapopt.data_analysis.data_handling.io_utilities.results_logging import ResultsLogger, LoggingLevel
from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf

from quapopt.data_analysis.data_handling import (STANDARD_NAMES_DATA_TYPES as SNDT,
                                                 STANDARD_NAMES_VARIABLES as SNV)

from qiskit.transpiler.passmanager import StagedPassManager

class DDOTRunner(OverlappingTomographyRunnerBase):
    """
    Class to run the Diagonal Detector Overlapping Tomography (DDOT) experiments.

    In practice, the generated circuits are random combinations of X and I gates.

    Some Refs:
    [1] https://arxiv.org/pdf/2101.02331
    [2] https://arxiv.org/pdf/2311.10661


    """

    def __init__(self,
                 number_of_qubits: int,
                 program_gate_builder: AbstractProgramGateBuilder,
                 sdk_name: str,
                 numpy_rng_seed: int = None,
                qubit_indices_physical=None,
                 results_logger_kwargs: Optional[dict] = None,
                 logging_level: LoggingLevel = LoggingLevel.DETAILED,
                 pass_manager_qiskit_indices: StagedPassManager = None,
                 number_of_qubits_device_qiskit: int = None

                 ):
        tomography_type = TomographyType.DIAGONAL_DETECTOR
        tomography_gates_type = TomographyGatesType.PAULI

        if logging_level is not None and logging_level != LoggingLevel.NONE:
            #TODO(FBM):add defaults
            if results_logger_kwargs is None:
                raise NotImplementedError("If you want to log results, you need to provide results_logger_kwargs.")
                experiment_folders_hierarchy = ['DefaultResultsFolder',
                                                'NoiseCharacterization',
                                                "DDOT",
                                                "DefaultSubFolder"]
                uuid = anf.create_random_uuid()
                directory_main = None
                table_name_prefix = None
                table_name_main = 'DDOTResults'
                results_logger_kwargs = {'experiment_folders_hierarchy': experiment_folders_hierarchy,
                                         'uuid': uuid,
                                         'base_path': directory_main,
                                         'table_name_prefix': table_name_prefix,
                                         'table_name_prefix': table_name_main}

        super().__init__(number_of_qubits=number_of_qubits,
                         program_gate_builder=program_gate_builder,
                         tomography_type=tomography_type,
                         tomography_gates_type=tomography_gates_type,
                         sdk_name=sdk_name,
                         numpy_rng_seed=numpy_rng_seed,
                         qubit_indices_physical=qubit_indices_physical,
                         results_logger_kwargs=results_logger_kwargs,
                         logging_level=logging_level,
                         pass_manager_qiskit_indices=pass_manager_qiskit_indices,
                         number_of_qubits_device_qiskit=number_of_qubits_device_qiskit
                         )

    def generate_tomography_circuits_random(self,
                                            number_of_circuits: int,
                                            enforce_uniqueness=True,
                                            add_measurements=True,
                                            add_barriers=True,
                                            include_standard_ddot_circuits=False,
                                            initial_circuit_labels:Optional[List[Tuple[int, ...]]] = None,
                                            prepend_circuit: Optional[AbstractCircuit] = None,
                                            append_circuit: Optional[AbstractCircuit] = None,
                                            middle_circuit: Optional[AbstractCircuit] = None,
                                            ) -> List[Tuple[Tuple[int, ...], Any]]:
        """

        Generate a list of tomography circuits for the DDOT experiment.
        It is the wrapper of parent class method that has "include_standard_ddot_circuits" option for standardized
        DDOT circuits.

        :param number_of_circuits:
        :param enforce_uniqueness:
        :param add_measurements:
        :param add_barriers:
        :param include_standard_ddot_circuits:
        :param prepend_circuit:
        :param append_circuit:
        :param middle_circuit:
        :return:
        """

        if initial_circuit_labels is None:
            initial_circuit_labels = []

        initial_circuit_labels = initial_circuit_labels.copy()

        #print("HEJUNIA",initial_circuit_labels, include_standard_ddot_circuits)
        if include_standard_ddot_circuits:
            assert number_of_circuits >= 2, "If you want to include standard DDOT circuits, you need to generate at least 2 circuits."
            circuit_00_label = tuple([0] * self._number_of_qubits)
            circuit_11_label = tuple([1] * self._number_of_qubits)
            initial_circuit_labels += [circuit_00_label, circuit_11_label]

            if number_of_circuits >= 4:
                # get 01 on first linear chain
                circuit_01_label = [0, 1] * (int(self._number_of_qubits / 2))
                circuit_10_label = [1, 0] * (int(self._number_of_qubits / 2))
                if self._number_of_qubits % 2 == 1:
                    circuit_01_label.append(0)
                    circuit_10_label.append(1)
                circuit_01_label = tuple(circuit_01_label)
                circuit_10_label = tuple(circuit_10_label)

                initial_circuit_labels += [circuit_01_label, circuit_10_label]


        #print("HEJUNIA2:", initial_circuit_labels, include_standard_ddot_circuits)
        return self._generate_tomography_circuits_random(number_of_circuits=number_of_circuits,
                                                         enforce_uniqueness=enforce_uniqueness,
                                                         add_measurements=add_measurements,
                                                         add_barriers=add_barriers,
                                                         prepended_symbols_list=initial_circuit_labels,
                                                         prepend_circuit=prepend_circuit,
                                                         append_circuit=append_circuit,
                                                         middle_circuit=middle_circuit)

class DDOTAnalyzer:
    def __init__(self,
                 circuits_labels: Union[List[Tuple[int, ...]],np.ndarray] = None,
                 bitstrings_arrays: Union[List[np.ndarray], np.ndarray] = None,
                 computation_backend=None,
                 results_logger_kwargs: Optional[dict] = None,
                 noisy_sampler_post_processing:ClassicalMeasurementNoiseSampler=None,
                 noisy_sampler_post_processing_rng = None,
                 uuid=None
                 ):

        """
        Class to analyze the DDOT experiment results.
        :param circuits_labels:
        The circuit labels should be of shape (number_of_circuits, number_of_qubits).
        :param bitstrings_arrays:
        The bitstrings arrays should be of shape (number_of_circuits, number_of_samples, number_of_qubits).
        :param computation_backend:
        """

        if computation_backend is None:
            computation_backend = 'numpy'

        if computation_backend == 'cupy':
            import cupy as bck
        elif computation_backend == 'numpy':
            import numpy as bck
        else:
            raise ValueError(
                f"Computation backend {computation_backend} not supported. Supported backends are: cupy, numpy.")

        self._bck = bck
        self._computation_backend = computation_backend

        if results_logger_kwargs is None:
            assert circuits_labels is not None and bitstrings_arrays is not None, \
                "If you don't provide results logger kwargs, you need to provide circuit labels and bitstrings arrays."
            self._results_logger = None
        else:
            self._results_logger = ResultsLogger(**results_logger_kwargs)

        if circuits_labels is None:
            assert bitstrings_arrays is None, "If you provide bitstrings arrays, you need to provide circuit labels as well."
        if bitstrings_arrays is None:
            assert circuits_labels is None, "If you provide circuit labels, you need to provide bitstrings arrays as well."

        if self._results_logger is not None:
            anf.cool_print("READING DATA...",'...','blue')
            results_df = self.read_results(table_name=None,
                                           data_type=SNDT.BitstringsHistograms,
                                           uuid=uuid)
            anf.cool_print("GOT DATA...",'Pre-processing','green')

            unique_circuit_indices = results_df[SNV.CircuitIndex.id_long].unique()

            circuits_labels, bitstrings_arrays = [], []
            for circuit_index in tqdm(unique_circuit_indices,
                               desc='PRE-PROCESSING',
                               colour='yellow'):
                results_df_job = results_df[results_df[SNV.CircuitIndex.id_long] == circuit_index]
                unique_circuit_labels = results_df_job[SNV.CircuitLabel.id_long].unique()
                if len(unique_circuit_labels)>1:
                    raise ValueError(f"More than one circuit label found for circuit id {circuit_index}.")

                circuit_label = unique_circuit_labels[0]

                bitstrings_circuit = bck.array(results_df_job[SNV.Bitstring.id_long].tolist(),
                                               dtype=bck.uint8)
                counts_circuit = bck.array(results_df_job[SNV.Count.id_long].tolist(),
                                             dtype=int)

                bitstrings_circuit = bck.repeat(bitstrings_circuit,
                                                counts_circuit,
                                                axis=0)

                bitstrings_arrays.append(bitstrings_circuit)

                circuits_labels.append(bck.array(circuit_label, dtype=bck.uint8))


        assert len(circuits_labels) == len(
            bitstrings_arrays), "Circuit labels and bitstrings arrays have different lengths."

        if noisy_sampler_post_processing is not None:
            bitstrings_arrays = [noisy_sampler_post_processing.add_noise_to_samples(ideal_samples=bts_x,
                                                                                   rng=noisy_sampler_post_processing_rng) for bts_x in bitstrings_arrays]


        self._circuits_labels = bck.array(circuits_labels, dtype=bck.uint)
        self._bitstrings_arrays = bck.array(bitstrings_arrays, dtype=bck.uint8)



        assert len(self._circuits_labels[0]) == self._bitstrings_arrays.shape[
            2], "Circuit labels and bitstrings arrays have different lengths."

        self._number_of_qubits = len(self._circuits_labels[0])
        self._number_of_samples = self._bitstrings_arrays.shape[1]

        self._noise_matrices_1q = None
        self._noise_matrices_2q = None
        self._subsets_2q = None

    def read_results(self,
                     table_name: Optional[str] = None,
                     data_type: SNDT = SNDT.BitstringsHistograms,
                     uuid=None
                     ):

        results_df = self._results_logger.read_results(table_name=table_name,
                                                       data_type=data_type)
        if uuid is not None:
            assert 'UUID' in results_df.columns, "UUID column not found in results dataframe."
            results_df = results_df[results_df['UUID'] == uuid].copy()

        if 'CircuitLabel' in results_df.columns:
            results_df['CircuitLabel'] = results_df['CircuitLabel'].apply(anf.eval_string_tuple_to_tuple)

        return results_df

    @property
    def noise_matrices_1q(self):
        """
        Get the 1-qubit noise matrices from the bitstrings arrays.

        :return: 1-qubit noise matrices
        """

        if self._noise_matrices_1q is None:
            self.get_1q_noise_matrices()

        return self._noise_matrices_1q

    @property
    def noise_matrices_2q(self):
        """
        Get the 2-qubit noise matrices from the bitstrings arrays.

        :return: 2-qubit noise matrices
        """

        if self._noise_matrices_2q is None:
            self.get_2q_noise_matrices()

        return self._noise_matrices_2q

    def get_1q_noise_matrices(self):
        """
        Get the 1-qubit noise matrices from the bitstrings arrays.

        :return: 1-qubit noise matrices
        """

        circuits_labels = self._circuits_labels
        bitstrings_arrays = self._bitstrings_arrays

        bck = self._bck
        noise_matrices = bck.zeros((self._number_of_qubits, 2, 2), dtype=bck.uint)
        number_of_samples = bitstrings_arrays.shape[1]

        input_ones = circuits_labels == 1
        input_zeros = circuits_labels == 0

        x1s_full = bitstrings_arrays.sum(axis=1)
        x0s_full = number_of_samples - x1s_full

        noise_matrices[:, 1, 1] = (x1s_full * input_ones).sum(axis=0)
        noise_matrices[:, 0, 1] = (x0s_full * input_ones).sum(axis=0)

        noise_matrices[:, 1, 0] = (x1s_full * input_zeros).sum(axis=0)
        noise_matrices[:, 0, 0] = (x0s_full * input_zeros).sum(axis=0)

        # Normalize the noise matrices
        noise_matrices = noise_matrices.astype(float)
        for i in range(2):
            noise_matrices[:, :, i] /= noise_matrices[:, :, i].sum(axis=1, keepdims=True)

        #replace nans with zeros
        noise_matrices = bck.nan_to_num(noise_matrices, nan=0.0)

        self._noise_matrices_1q = noise_matrices

        return self.noise_matrices_1q

    def get_2q_noise_matrices(self,
                              subsets_2q=None,
                              show_progress_bar=True):
        """
        Get the 2-qubit noise matrices from the bitstrings arrays.


        :param subsets_2q:
        :param show_progress_bar:
        :return:
        """

        number_of_qubits = self._number_of_qubits
        if subsets_2q is None:
            subsets_2q = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]

        # The circuit labels should be of shape (number_of_circuits, number_of_qubits).
        circuits_labels = self._circuits_labels
        # The bitstrings arrays should be of shape (number_of_circuits, number_of_samples, number_of_qubits).
        bitstrings_arrays = self._bitstrings_arrays

        bck = self._bck

        noise_matrices = bck.zeros((len(subsets_2q), 4, 4), dtype=bck.uint)

        qi_idx = bck.array([pair[0] for pair in subsets_2q], dtype=bck.uint8)
        qj_idx = bck.array([pair[1] for pair in subsets_2q], dtype=bck.uint8)

        integer_idx_input = (circuits_labels << 1)[:, qi_idx] | circuits_labels[:, qj_idx]
        number_of_subsets = len(subsets_2q)

        for circuit_index in tqdm(list(range(len(circuits_labels))),
                                  position=0,
                                  colour='green',
                                  desc='Reconstructing 2q noise matrices',
                                  disable=not show_progress_bar):
            bitstrings_here = bitstrings_arrays[circuit_index, :, :]
            idxs_output = (bitstrings_here[:, qi_idx] << 1) | bitstrings_here[:, qj_idx]

            for output_state_index in range(4):
                marg_i_2q = (idxs_output == output_state_index).sum(axis=0)
                for id in range(number_of_subsets):
                    noise_matrices[id, output_state_index, integer_idx_input[circuit_index, id]] += marg_i_2q[id]


        # Normalize the noise matrices
        noise_matrices = noise_matrices.astype(float)
        for i in range(4):
            noise_matrices[:, :, i] /= noise_matrices[:, :, i].sum(axis=1, keepdims=True)

        #replace nans with zeros
        noise_matrices = bck.nan_to_num(noise_matrices, nan=0.0)

        self._noise_matrices_2q = noise_matrices
        self._subsets_2q = subsets_2q

        return self._noise_matrices_2q

    def _save_figure(self,
                     figure,
                     file_name):

        os.makedirs('/'.join(file_name.split('/')[0:-1]), exist_ok=True)

        plotly.offline.plot(figure,
                            filename=f'{file_name}.html')



    def plot_noise_matrices(self,
                            noise_matrices=None,
                            subset_indices=None,
                            qubit_indices_physical_map: Optional[List[int]] = None,
                            colormap_name='YlGnBu',
                            file_name=None):

        if noise_matrices is None:
            noise_matrices = self.noise_matrices_1q
            subset_indices = list(range(self._number_of_qubits))
        if noise_matrices is not None:
            assert subset_indices is not None, "If you provide noise matrices, you need to provide subset indices as well."

        number_of_columns = 3
        number_of_rows = int(np.ceil(len(subset_indices) / number_of_columns))

        if qubit_indices_physical_map is None:
            physical_names = subset_indices
        else:
            qipm = qubit_indices_physical_map
            physical_names = [tuple([qipm[i] for i in subset]) if not isinstance(subset, int) else (qipm[subset],) for
                              subset in subset_indices]

        fig = make_subplots(rows=number_of_rows,
                            cols=number_of_columns,
                            subplot_titles=[f"Qubit(s):{physical_names[i]}" for i in range(len(noise_matrices))])

        for idx, matrix in enumerate(noise_matrices):
            row = (idx // number_of_columns) + 1
            col = (idx % number_of_columns) + 1
            fig.add_trace(
                go.Heatmap(z=matrix[::-1],
                           colorscale=colormap_name,
                           zmin=0.0,
                           zmax=1.0, ),
                row=row,
                col=col, )

        fig.update_layout(height=number_of_rows * 400,
                          width=number_of_columns * 500,
                          title_text="Reconstructed noise matrices")

        if file_name is None:
            folder_path = f'../temp/figs/readout_noise/'
            plot_name = 'noise_matrices_heatmaps'
            file_name = f'{folder_path}{plot_name}'

        self._save_figure(figure=fig,
                          file_name=file_name)

    def plot_noise_matrices_1q(self,
                               qubit_indices_physical_map: Optional[List[int]] = None,
                               colormap_name='YlGnBu',
                               file_name=None):
        self.plot_noise_matrices(noise_matrices=self.noise_matrices_1q,
                                 subset_indices=list(range(self._number_of_qubits)),
                                 qubit_indices_physical_map=qubit_indices_physical_map,
                                 colormap_name=colormap_name,
                                 file_name=file_name)

    def plot_noise_matrices_2q(self,
                                 qubit_indices_physical_map: Optional[List[int]] = None,
                                    colormap_name='YlGnBu',
                                    file_name=None):
        self.plot_noise_matrices(noise_matrices=self.noise_matrices_2q,
                                 subset_indices=self._subsets_2q,
                                 qubit_indices_physical_map=qubit_indices_physical_map,
                                 colormap_name=colormap_name,
                                 file_name=file_name)


    def plot_errors_histograms(self,
                               noise_matrices=None,
                               file_name=None):

        if noise_matrices is None:
            noise_matrices = self.noise_matrices_1q

        noise_matrices = np.array(noise_matrices)

        dimension = noise_matrices[0].shape[0]
        number_of_qubits = int(np.log2(dimension))

        cls = list(itertools.product([0, 1], repeat=number_of_qubits))

        titles = [f"Prob({''.join([str(x) for x in cls[i]])}->{''.join([str(x) for x in cls[j]])})"
                  for i in range(dimension) for j in
                  range(dimension)]

        fig = make_subplots(rows=dimension,
                            cols=dimension,
                            subplot_titles=titles)

        # start_bin = 0.0
        # stop_bin = 1.0
        size_bin = 0.005

        max_success, max_error = 0.0, 0.0
        min_success, min_error = 1.0, 1.0

        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    max_success = np.max([max_success, np.max(noise_matrices[:, i, j])])
                    min_success = np.min([min_success, np.min(noise_matrices[:, i, j])])
                else:
                    max_error = np.max([max_error, np.max(noise_matrices[:, i, j])])
                    min_error = np.min([min_error, np.min(noise_matrices[:, i, j])])

        for input_int in range(dimension):
            for output_int in range(dimension):
                flat_index = input_int * dimension + output_int
                probs_x_y = noise_matrices[:, output_int, input_int]
                #print(probs_x_y)

                if input_int == output_int:
                    color = 'teal'
                    xrange = [min_success, max_success]
                    bin_min, bin_max = xrange
                else:
                    color = 'crimson'
                    xrange = [min_error, max_error]
                    bin_min, bin_max = xrange

                xbins_arange = np.arange(bin_min, bin_max + size_bin, size_bin)
                xbins_dict = dict(  # start=bin_min,
                    # end=bin_max,
                    size=size_bin)

                fig.add_trace(go.Histogram(x=probs_x_y,
                                           xbins=xbins_dict,
                                           marker_color=color,
                                           opacity=0.75,
                                           name=titles[flat_index],
                                           ),
                              row=input_int + 1,
                              col=output_int + 1,
                              # xrange=xrange,
                              )
                # update layout of this subplot to change xrange
                fig.update_xaxes(
                    row=input_int + 1,
                    col=output_int + 1,
                    range=xrange,

                )

                average = np.mean(probs_x_y)
                #print(probs_x_y)
                #print(np.histogram(probs_x_y, bins=xbins_arange)[0])
                # add dashed vertical line with average value
                fig.add_trace(
                    go.Scatter(
                        x=[average, average],
                        y=[0, max([np.histogram(probs_x_y, bins=xbins_arange)[0]]) * 1.0],
                        mode='lines',
                        line=dict(color='black', width=3, dash='dash'),
                        name=f'Average {titles[flat_index]}'
                    ),
                    row=input_int + 1,
                    col=output_int + 1,
                )

        if file_name is None:
            folder_path = f'../temp/figs/readout_noise/'
            plot_name = 'noise_matrices_histograms'
            file_name = f'{folder_path}{plot_name}'

        self._save_figure(figure=fig,
                          file_name=file_name)


    def plot_errors_histograms_1q(self,
                               file_name=None):
        self.plot_errors_histograms(noise_matrices=self.noise_matrices_1q,
                                    file_name=file_name)
    def plot_errors_histograms_2q(self,
                               file_name=None):
        self.plot_errors_histograms(noise_matrices=self.noise_matrices_2q,
                                    file_name=file_name)



if __name__ == '__main__':
    from quapopt.circuits.gates.logical.LogicalGateBuilderQiskit import LogicalGateBuilderQiskit
    from qiskit_aer.backends.aer_simulator import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import Session, SamplerV2 as SamplerRuntime, QiskitRuntimeService

    number_of_qubits_test = 3
    number_of_samples_test = 10 ** 4
    number_of_circuits_test = 8

    #Number of random X gates
    #Number of different delays per combination of X gates
    #Number of unitary folded circuits per combination of X gates and a delay

    #Total number of circuits
    #n_xs*n_delays*n_folds

    #Input state: some hamming weight
    #Output distribution: some hamming weights

    computation_backend = 'numpy'

    if computation_backend == 'cupy':
        bck = cp
    elif computation_backend == 'numpy':
        bck = np
    else:
        raise ValueError(
            f"Computation backend {computation_backend} not supported. Supported backends are: cupy, numpy.")
    numpy_rng = bck.random.default_rng(0)

    p01_list = list(numpy_rng.uniform(0, 0.2, number_of_qubits_test))
    p10_list = list(numpy_rng.uniform(0, 0.05, number_of_qubits_test))

    experiment_folders_hierarchy = ['noise_characterization', 'overlapping_tomography', "DDOT"]
    uuid = 'TestRuns_01'
    directory_main = None
    table_name_prefix = None
    table_name_main = 'DDOTResults'
    results_logger_kwargs = {'experiment_folders_hierarchy': experiment_folders_hierarchy,
                             'uuid': uuid,
                             'base_path': directory_main,
                             'table_name_prefix': table_name_prefix,
                             'table_name_prefix': table_name_main}

    CMNS = ClassicalMeasurementNoiseSampler(noise_type=MeasurementNoiseType.TP_1q_general,
                                            noise_description={'p_01': p01_list,
                                                               'p_10': p10_list},
                                            rng=numpy_rng, )

    expected_noise_matrices_1q = [bck.array([[1 - p_10, p_01],
                                             [p_10, 1 - p_01]]) for p_01, p_10 in zip(p01_list, p10_list)]


    if number_of_qubits_test <= 24:
        gate_builder = LogicalGateBuilderQiskit()
        sdk_name = 'qiskit'

        start = time.time()

        DDOT_run_test = DDOTRunner(number_of_qubits=number_of_qubits_test,
                                   program_gate_builder=gate_builder,
                                   sdk_name=sdk_name,
                                   numpy_rng_seed=42,
                                   results_logger_kwargs=results_logger_kwargs,
                                   logging_level=LoggingLevel.DETAILED)

        random_circuits = DDOT_run_test.generate_tomography_circuits_random(number_of_circuits=number_of_circuits_test,
                                                                            enforce_uniqueness=True,
                                                                            add_measurements=True,
                                                                            add_barriers=True,
                                                                            include_standard_ddot_circuits=True)

        local_username = 'fbm'
        provider = QiskitRuntimeService(channel='ibm_quantum',
                                        instance='ibm-q-stfc/optimisation/ndar',
                                        filename=f'/home/{local_username}/.ibm_quantum/qiskit_ibm_runtime_credentials.json')

        backend_name = 'ibm_marrakesh'
        qiskit_backend = provider.backend(name=backend_name,
                                  use_fractional_gates=True)
        pass_manager = generate_preset_pass_manager(backend=qiskit_backend,
                                                    optimization_level=0)

        circuits_labels, bitstrings_histograms = DDOT_run_test.run_tomography_circuits_qiskit_session(
            tomography_circuits=random_circuits,
            qiskit_backend=qiskit_backend,
            simulation=True,
            number_of_shots=number_of_samples_test,
            qiskit_pass_manager=pass_manager)

        # print(bitstrings_histograms[0], bitstrings_histograms[1])

        bitstrings_array = anf.transform_histogram_to_bitstrings_array(
            bitstrings_array_histogram=bitstrings_histograms)

        ddot_analyzer_test = DDOTAnalyzer(
            # circuits_labels=circuits_labels,
            # bitstrings_arrays=noisy_bitstrings_array,
            computation_backend=computation_backend,
            results_logger_kwargs=results_logger_kwargs,
            noisy_sampler_post_processing_rng=numpy_rng,
            noisy_sampler_post_processing=None
        )

    else:
        rng_test = bck.random.default_rng(seed=0)
        circuits_labels = rng_test.binomial(n=1, p=1 / 2, size=(number_of_circuits_test, number_of_qubits_test)).astype(
            bck.uint8)
        bitstrings_array = bck.array([[label] * number_of_samples_test for label in circuits_labels])
        noisy_bitstrings_array = [CMNS.add_noise_to_samples(ideal_samples=row,
                                                            rng=numpy_rng) for row in bitstrings_array]

        ddot_analyzer_test = DDOTAnalyzer(
            circuits_labels=circuits_labels,
            bitstrings_arrays=noisy_bitstrings_array,
            computation_backend=computation_backend,
            # results_logger_kwargs=results_logger_kwargs,
            # noisy_sampler_post_processing_rng=numpy_rng,
            # noisy_sampler_post_processing=CMNS
        )

    subsets_2q = [(i, j) for i in range(number_of_qubits_test) for j in range(i + 1, number_of_qubits_test)]

    expected_noise_matrices_2q = [bck.kron(expected_noise_matrices_1q[i], expected_noise_matrices_1q[j]) for i, j in
                                  subsets_2q]

    t0 = time.perf_counter()
    reconstructed_noise_matrices_1q = ddot_analyzer_test.get_1q_noise_matrices()
    t1 = time.perf_counter()
    reconstructed_noise_matrices_2q = ddot_analyzer_test.get_2q_noise_matrices(subsets_2q=subsets_2q)
    t2 = time.perf_counter()

    print("1q reconstruction time: ", t1 - t0)
    print("2q reconstruction time: ", t2 - t1)
    # for qi in list(range(number_of_qubits_test)):
    #     expected_lam = expected_noise_matrices_1q[qi]
    #     reconstructed_lam = ddot_analyzer_test.noise_matrices_1q[qi]
    #     print(f"Qubit: {qi}")
    #     print("Expected:\n", np.round(expected_lam, 3))
    #     print("Reconstructed:\n", np.round(reconstructed_lam, 3))
    #     print("close:", np.allclose(expected_lam, reconstructed_lam, atol=1e-2))

    print('1q matrices agree:', bck.allclose(bck.array(reconstructed_noise_matrices_1q),
                                             bck.array(expected_noise_matrices_1q),
                                             atol=1e-2))
    print('2q matrices agree:',
          bck.allclose(bck.array(reconstructed_noise_matrices_2q),
                       bck.array(expected_noise_matrices_2q),
                       atol=1e-2))

    ddot_analyzer_test.plot_noise_matrices(noise_matrices=reconstructed_noise_matrices_1q,
                                           subset_indices=list(range(number_of_qubits_test)))
    #
    ddot_analyzer_test.plot_noise_matrices(noise_matrices=reconstructed_noise_matrices_2q,
                                           subset_indices=subsets_2q)

    ddot_analyzer_test.plot_errors_histograms(noise_matrices=reconstructed_noise_matrices_1q)
    ddot_analyzer_test.plot_errors_histograms(noise_matrices=reconstructed_noise_matrices_2q)

    raise KeyboardInterrupt()

    marginals_1q = em.get_marginals_from_bitstrings_array(bitstrings_array=random_bitstrings,
                                                          qubits_subsets_by_locality={1: [0, 1, 3]},
                                                          normalize=False
                                                          )
    marginals_2q = em.get_marginals_from_bitstrings_array(bitstrings_array=random_bitstrings,
                                                          qubits_subsets_by_locality={2: [(0, 1), (1, 2), (2, 3)]},
                                                          normalize=False
                                                          )
    marginals_3q_4q = em.get_marginals_from_bitstrings_array(bitstrings_array=random_bitstrings,
                                                             qubits_subsets_by_locality={3: [(0, 1, 2), (1, 2, 3)],
                                                                                         4: [(0, 1, 2, 3)]},
                                                             normalize=False
                                                             )

    finish = time.time()

    print("Time taken: ", finish - start)
    print(random_bitstrings)
    print(marginals_1q)
    print(marginals_2q)
    print(marginals_3q_4q[0][1])
    print(marginals_3q_4q[1][1])
