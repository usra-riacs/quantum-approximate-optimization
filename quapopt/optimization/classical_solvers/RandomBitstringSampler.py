# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import time
from typing import List, Optional, Any, Dict, Mapping, Callable, Type

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path

from quapopt.data_analysis.data_handling.schemas.naming import (
    BaseNameDataType
)
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian

# Lazy monkey-patching of cupy
from quapopt import AVAILABLE_SIMULATORS
if 'cupy' in AVAILABLE_SIMULATORS:
    import cupy as cp
else:
    import numpy as cp

from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.optimization import (HamiltonianSolutionsSampler)
from quapopt.data_analysis.data_handling import LoggingLevel
from quapopt.data_analysis.data_handling import (STANDARD_NAMES_DATA_TYPES as SNDT)

DEFAULT_ID_RANDOM_SAMPLING = 'RandomSampling'
DEFAULT_FOLDER_RANDOM_SAMPLING = ['Results', 'RandomSampling']


class RandomBitstringSampler(HamiltonianSolutionsSampler):

    def __init__(self,
                 cost_hamiltonian: ClassicalHamiltonian,
                 backend=None,
                 max_memory=None,
                 logging_level: Optional[LoggingLevel] = None,
                 logger_kwargs: Optional[Dict[str, Any]] = None,
                 ):

        def _check_if_use_default_value(_key):
            _use_default_value = True
            if _key in logger_kwargs:
                _use_default_value = logger_kwargs[_key] is None
            return _use_default_value

        if logging_level not in [None, LoggingLevel.NONE] and logger_kwargs is None:
            logger_kwargs = {}

        if logger_kwargs is not None:
            logger_kwargs = logger_kwargs.copy()
            if _check_if_use_default_value('experiment_folders_hierarchy'):
                logger_kwargs['experiment_folders_hierarchy'] = DEFAULT_FOLDER_RANDOM_SAMPLING+[cost_hamiltonian.hamiltonian_class_description]
            if _check_if_use_default_value('experiment_set_id'):
                logger_kwargs['experiment_set_id'] = DEFAULT_ID_RANDOM_SAMPLING

        super().__init__(input_hamiltonian_representations_cost=[cost_hamiltonian],
                         solve_at_initialization=False,
                         number_of_qubits=cost_hamiltonian.number_of_qubits,
                         logging_level=logging_level,
                         logger_kwargs=logger_kwargs,
                         )

        if backend is None:
            backend = cost_hamiltonian.default_backend

        if cost_hamiltonian.default_backend != backend:
            cost_hamiltonian.reinitialize_backend(backend=backend)

        self._cost_hamiltonian = cost_hamiltonian
        self._backend = backend
        self._number_of_qubits = cost_hamiltonian.number_of_qubits
        if max_memory is None:
            if backend == 'cupy':
                max_memory = int(1 * 10 ** 7)
            elif backend == 'numpy':
                max_memory = int(1 * 10 ** 9)
            else:
                raise ValueError(f"Backend {backend} not supported")

        self._max_memory = max_memory

    def _sample_solutions(self,
                          number_of_samples: int,
                          number_of_trials: int = 1,
                          seed: int = 0,
                          show_progress_bar: bool = False,
                          return_all_results: bool = False,
                          p_1=0.5):

        # OK, so I can fit "max_memory" of bytes in total
        # this means I can fit this amount of bitsrings:
        max_bitstrings_in_memory = int(self._max_memory // self._number_of_qubits)

        if self._backend == 'cupy':
            bck = cp
        elif self._backend == 'numpy':
            bck = np
        else:
            raise ValueError(f"Backend {self._backend} not supported")

        rng = bck.random.default_rng(seed=seed)

        best_energy_so_far = np.inf
        best_state_so_far = None

        all_bitstrings, all_energies = [], []

        results_dfs = []
        number_of_samples_per_trial = number_of_samples
        t0_total = time.perf_counter()

        pbar_trials = tqdm(list(range(number_of_trials)),
                           disable=(not show_progress_bar) or number_of_trials == 1,
                           position=0,
                           colour='cyan')
        for trial_index in pbar_trials:
            # If user needs less samples, I will just use that amount
            if number_of_samples_per_trial <= max_bitstrings_in_memory:
                batch_size = number_of_samples_per_trial
                number_of_batches = 1
            else:
                batch_size = max_bitstrings_in_memory
                number_of_batches = int(np.ceil(number_of_samples_per_trial / max_bitstrings_in_memory))

            pbar_batches = tqdm(list(range(number_of_batches)),
                                disable=(not show_progress_bar) or number_of_trials > 1,
                                position=0,
                                colour='cyan')
            for batch_index in pbar_batches:
                if batch_index == number_of_batches - 1:
                    batch_size = number_of_samples_per_trial - batch_size * batch_index
                if self._backend == 'cupy':
                    random_bitstrings_i = rng.binomial(n=1,
                                                       p=p_1,
                                                       size=(batch_size,
                                                             self._number_of_qubits))
                elif self._backend == 'numpy':
                    if p_1 == 0.5:
                        random_bitstrings_i = rng.integers(low=0,
                                                           high=2,
                                                           size=(batch_size,
                                                                 self._number_of_qubits))
                    else:
                        random_bitstrings_i = rng.binomial(n=1,
                                                           p=p_1,
                                                           size=(batch_size,
                                                                 self._number_of_qubits))
                else:
                    raise ValueError(f"Backend {self._backend} not supported")

                energies_i = self._cost_hamiltonian.evaluate_energy(bitstrings_array=random_bitstrings_i,
                                                                    backend_computation=self._backend)
                best_energy_index = np.argmin(energies_i)
                best_energy_i = float(energies_i[best_energy_index])
                best_state_i = random_bitstrings_i[best_energy_index]

                if best_energy_i < best_energy_so_far:
                    best_energy_so_far = best_energy_i
                    best_state_so_far = best_state_i



                annotation_i = {
                    SNV.Seed.id_long: seed,
                    SNV.TrialIndex.id_long: trial_index,
                }


                if return_all_results or self.logging_level.value>=1:
                    df_histogram_energies_i = self._get_histogram_df(values_array=energies_i,
                                                                     annotations_dict=annotation_i,
                                                                     value_name=SNV.Energy.id_long)

                    if return_all_results:
                        all_energies.append(energies_i)

                    if self.logging_level.value >= 1:
                        self.log_energies_histogram(energies_histogram_df=df_histogram_energies_i,
                                                    p_1=p_1)

                    if return_all_results or self.logging_level.value>=2:
                        df_histogram_bitstrings_i = self._get_histogram_df(values_array=random_bitstrings_i,
                                                                           annotations_dict=annotation_i,
                                                                           value_name=SNV.Bitstring.id_long)
                        if self.logging_level.value >= 2:
                            self.log_bitstrings_histogram(bitstrings_histogram_df=df_histogram_bitstrings_i,
                                                          p_1=p_1)

                        if return_all_results:
                            all_bitstrings.append(random_bitstrings_i)



                df_here = pd.DataFrame(data={
                    SNV.Seed.id_long: [seed],
                    SNV.TrialIndex.id_long: [trial_index],
                    SNV.EnergyMean.id_long: [np.mean(energies_i)],
                    SNV.EnergySTD.id_long: [np.std(energies_i)],
                    SNV.EnergyBest.id_long: [best_energy_i],
                    SNV.ValueBestSoFar.id_long: [best_energy_so_far],
                    SNV.Runtime.id_long: [time.perf_counter() - t0_total],
                    SNV.BitstringBest.id_long: [tuple([int(x) for x in best_state_so_far])],
                })
                results_dfs.append(df_here)

            pbar_batches.close()

        pbar_trials.close()
        results_df = pd.concat(results_dfs, axis=0, ignore_index=True)

        if return_all_results:
            all_bitstrings = pd.concat(all_bitstrings, axis=0)
            all_energies = pd.concat(all_energies, axis=0)
            return (best_state_so_far, best_energy_so_far), (results_df, (all_bitstrings, all_energies))

        return (best_state_so_far, best_energy_so_far), results_df


    def _get_histogram_df(self,
                          values_array,
                          annotations_dict: dict,
                          value_name: str,
                          ):

        if self._backend == 'cupy':
            bck = cp
        elif self._backend == 'numpy':
            bck = np
        else:
            raise ValueError(f"Backend {self._backend} not supported")

        if self.logging_level in [None, LoggingLevel.NONE]:
            return

        values, counts = bck.unique(values_array, return_counts=True)

        df_hist = pd.DataFrame(data={value_name: values.tolist(),
                                        SNV.Count.id_long: counts.tolist()})
        for col, val in annotations_dict.items():
            df_hist[col] = [val] * len(df_hist)

        return df_hist


    def _log_histogram(self,
                       df_histogram: pd.DataFrame,
                       p_1: float,
                       data_type:Type[SNDT]
                       ):
        table_name_suffix = f'p1={p_1}'
        self.write_results(dataframe=df_histogram,
                           data_type=data_type,
                           table_name_suffix=table_name_suffix)


    def log_energies_histogram(self,
                               energies_histogram_df:pd.DataFrame,
                               p_1:float):

        self._log_histogram(df_histogram=energies_histogram_df,
                            p_1=p_1,
                            data_type=SNDT.EnergiesHistograms)

    def log_bitstrings_histogram(self,
                                 bitstrings_histogram_df: pd.DataFrame,
                                 p_1: float):

        self._log_histogram(df_histogram=bitstrings_histogram_df,
                            p_1=p_1,
                            data_type=SNDT.BitstringsHistograms)





    def read_energies_histogram(self,
                                p_1:float=0.5,
                                table_name_prefix: Optional[str] = None,
                                directory_subpath: Optional[str | Path] = None,
                                return_none_if_not_found: bool = False,
                                experiment_instance_ids: Optional[List[str]] = None,
                                experiment_instance_ids_to_skip: Optional[List[str]] = None,
                                filter_by_experiment_set: bool = True,
                                file_name_filter_function: Optional[Mapping[str, bool] | Callable[[str], bool]] = None,
                                merge_instances_metadata_data_type: Optional[BaseNameDataType] = None,
                                drop_file_source_columns: bool = True,
                                drop_experiment_instance_id: bool = True,
                                columns_to_drop: Optional[List[str]] = None):

        table_name_suffix = f'p1={p_1}'

        return self.gather_results(data_type=SNDT.EnergiesHistograms,
                                   table_name_prefix=table_name_prefix,
                                   table_name_suffix=table_name_suffix,
                                   directory_subpath=directory_subpath,
                                   return_none_if_not_found=return_none_if_not_found,
                                   experiment_instance_ids=experiment_instance_ids,
                                   experiment_instance_ids_to_skip=experiment_instance_ids_to_skip,
                                   filter_by_experiment_set=filter_by_experiment_set,
                                   file_name_filter_function=file_name_filter_function,
                                   merge_instances_metadata_data_type=merge_instances_metadata_data_type,
                                   drop_file_source_columns=drop_file_source_columns,
                                   drop_experiment_instance_id=drop_experiment_instance_id,
                                   columns_to_drop=columns_to_drop)
