# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import itertools
import warnings
from typing import Union, List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError, ModuleNotFoundError):
    import numpy as cp

from pathlib import Path

from quapopt.data_analysis.data_handling import (StandardizedSpecifier,
                                                 ResultsLogger
                                                 )
from quapopt.data_analysis.data_handling import LoggingLevel, \
    HamiltonianOptimizationLoggerConfig
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV, STANDARD_NAMES_DATA_TYPES as SNDT
from quapopt.hamiltonians.representation import HamiltonianListRepresentation as HLR
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian as CLH, \
    ClassicalHamiltonian

_BTS_FORMAT = Union[np.ndarray, cp.ndarray, Tuple[int, ...]]


class HamiltonianOptimizer:
    def __init__(self,
                 input_hamiltonian_representations_cost: Union[List[Union[CLH, HLR]], CLH, HLR],
                 solve_at_initialization=True,
                 number_of_qubits: Optional[int] = None,
                 logging_level: Optional[LoggingLevel] = None,
                 logger_kwargs: Optional[Dict[str, Any]] = None,
                 store_full_information_in_history: bool = False,
                 store_n_best_results: int = 1,
                 ):

        if isinstance(input_hamiltonian_representations_cost, list):
            example_ham = input_hamiltonian_representations_cost[0]
            if isinstance(example_ham, list):
                pass
            elif isinstance(example_ham, CLH):
                pass
            else:
                input_hamiltonian_representations_cost = [input_hamiltonian_representations_cost]
        elif isinstance(input_hamiltonian_representations_cost, CLH):
            input_hamiltonian_representations_cost = [input_hamiltonian_representations_cost]
        else:
            raise ValueError(
                f'input_hamiltonian_representations_cost must be either '
                f'a list of Hamiltonian terms or a ClassicalHamiltonian instance'
                f'but is {type(input_hamiltonian_representations_cost)}')

        self._hamiltonian_representations_cost: Dict[int, CLH] = {}
        if number_of_qubits is None:
            example_ham = input_hamiltonian_representations_cost[0]
            # print(example_ham)
            if isinstance(example_ham, list):
                number_of_qubits = max([max(tup) for coeff, tup in input_hamiltonian_representations_cost[0]]) + 1
            elif isinstance(example_ham, CLH):
                number_of_qubits = example_ham.number_of_qubits

        self._number_of_qubits = number_of_qubits

        if self._number_of_qubits <= 20:
            self._dimension = 2 ** self._number_of_qubits
        else:
            self._dimension = None

        self.update_hamiltonians_cost(input_hamiltonian_representations_cost,
                                      solve=solve_at_initialization)

        if store_n_best_results > 1:
            self._best_results_container = BestResultsContainer(number_of_best_results=store_n_best_results)
        else:
            self._best_results_container = BestResultsContainerBase()

        self._store_n_best_results = store_n_best_results

        if logging_level is None:
            logging_level = LoggingLevel.NONE
        if logger_kwargs is None:
            logger_kwargs = {}

        logger_kwargs = logger_kwargs.copy()
        if 'logging_level' not in logger_kwargs:
            logger_kwargs['logging_level'] = logging_level
        else:
            logging_level = logger_kwargs['logging_level']

        self._logging_level = logging_level

        self._results_logger: Optional[ResultsLogger] = None

        if logging_level != LoggingLevel.NONE:
            self.reinitialize_logger(**logger_kwargs)

        self._optimization_history_full = []
        self._optimization_history_main = []
        self._store_full_information_in_history = store_full_information_in_history

    @property
    def optimization_history_full(self):
        return self._optimization_history_full

    @property
    def optimization_history_main(self):
        return self._optimization_history_main

    def clear_optimization_history(self):

        try:
            self._best_results_container.clear()
        except(AttributeError):
            pass
        self._optimization_history_full = []
        self._optimization_history_main = []
        # self._correlators_history = {}
        self._trial_index = 0

    def _update_hamiltonians_cost(self,
                                  hamiltonian_representations_cost: List[ClassicalHamiltonian],
                                  solve=False):

        if isinstance(hamiltonian_representations_cost, dict):
            hams_range = hamiltonian_representations_cost.items()
        else:
            hams_range = enumerate(hamiltonian_representations_cost)

        for idx, ham in hams_range:
            ham = self._handle_hamiltonian_type(ham,
                                                solve=solve,
                                                number_of_qubits=self._number_of_qubits)
            self._hamiltonian_representations_cost[idx] = ham

    def update_hamiltonians_cost(self,
                                 hamiltonian_representations_cost: List[ClassicalHamiltonian],
                                 solve=False):

        self._update_hamiltonians_cost(hamiltonian_representations_cost,
                                       solve=solve)

    def _update_history(self,
                        *args,
                        **kwargs):

        raise NotImplementedError("This method should be implemented in a subclass")

    def update_history(self,
                       *args,
                       **kwargs):
        self._update_history(*args, **kwargs)

    def reinitialize_logger(self,
                            *args,
                            **kwargs
                            ):

        raise NotImplementedError("This method should be implemented in a subclass")

        # self.initialize_results_logger(
        #     table_name_prefix=table_name_prefix,
        #     table_name_suffix=table_name_suffix,
        #     experiment_specifier=experiment_specifier,
        #     experiment_folders_hierarchy=experiment_folders_hierarchy,
        #     directory_main=directory_main,
        #     logging_level=logging_level,
        #     experiment_set_name=experiment_set_name,
        #     experiment_set_id=experiment_set_id,
        #     experiment_instance_id=experiment_instance_id,
        # )

        # self.set_logging_level(logging_level=logging_level)

    # def reinitialize_logger(self,
    #                         table_name_prefix: Optional[str] = None,
    #                         table_name_suffix: Optional[str] = None,
    #                         experiment_specifier=None,
    #                         experiment_folders_hierarchy: List[str] = None,
    #                         directory_main: Optional[str | Path] = None,
    #                         logging_level: LoggingLevel = LoggingLevel.BASIC,
    #                         experiment_set_name: Optional[str] = None,
    #                         experiment_set_id: Optional[str] = None,
    #                         experiment_instance_id: Optional[str] = None,
    #                         ):
    #
    #     self._reinitialize_logger(table_name_prefix=table_name_prefix,
    #                               table_name_suffix=table_name_suffix,
    #                               experiment_specifier=experiment_specifier,
    #                               experiment_folders_hierarchy=experiment_folders_hierarchy,
    #                               directory_main=directory_main,
    #                               logging_level=logging_level,
    #                               experiment_set_name=experiment_set_name,
    #                               experiment_set_id=experiment_set_id,
    #                               experiment_instance_id=experiment_instance_id,
    #                               )

    @property
    def hamiltonian_representations(self) -> Dict[int, ClassicalHamiltonian]:
        return self._hamiltonian_representations_cost

    @property
    def hamiltonian_representations_cost(self) -> Dict[int, ClassicalHamiltonian]:
        return self._hamiltonian_representations_cost

    @property
    def number_of_qubits(self):
        return self._number_of_qubits

    @property
    def dimension(self):
        return self._dimension

    @staticmethod
    def _handle_hamiltonian_type(ham,
                                 solve=False,
                                 number_of_qubits=None):
        # print("SOLVING:",solve)
        if isinstance(ham, list):
            ham = CLH(ham,
                      number_of_qubits=number_of_qubits)
            if solve:
                ham.solve_hamiltonian()
        elif isinstance(ham, CLH):
            if solve:
                ham.solve_hamiltonian()
        else:
            raise ValueError(
                f'hamiltonian must be either a list of Hamiltonian terms or a ClassicalHamiltonian instance'
                f'but is {type(ham)}')
        return ham

    # def _initialize_results_logger(self,
    #                                *args,
    #                                **kwargs):
    #
    #     raise NotImplementedError("This method should be implemented in a subclass")

    # def initialize_results_logger(self,
    #                               *args,
    #                               **kwargs):
    #     raise NotImplementedError("This method should be implemented in a subclass")

    @property
    def results_logger(self) -> ResultsLogger:
        return self._results_logger

    @property
    def logging_level(self):
        logging_level_outer = self._logging_level
        if self.results_logger is not None:
            self.results_logger._logging_level = logging_level_outer
        return logging_level_outer

    def set_logging_level(self, logging_level: LoggingLevel):
        self._logging_level = logging_level
        if self.results_logger is not None:
            self.results_logger._logging_level = logging_level

    def set_logger_uuid(self, uuid):
        self._results_logger.uuid = uuid


class HamiltonianSolutionsSampler(HamiltonianOptimizer):

    def __init__(self,
                 input_hamiltonian_representations_cost: Union[List[Union[CLH, HLR]], CLH, HLR],
                 solve_at_initialization=True,
                 number_of_qubits: Optional[int] = None,
                 logging_level: Optional[LoggingLevel] = None,
                 logger_kwargs: Optional[Dict[str, Any]] = None,
                 store_full_information_in_history: bool = False,
                 store_n_best_results: int = 1,
                 ):

        super().__init__(input_hamiltonian_representations_cost=input_hamiltonian_representations_cost,
                         solve_at_initialization=solve_at_initialization,
                         number_of_qubits=number_of_qubits,
                         logging_level=logging_level,
                         logger_kwargs=logger_kwargs,
                         store_full_information_in_history=store_full_information_in_history,
                         store_n_best_results=store_n_best_results,
                         )

    def _sample_solutions(self,
                          number_of_samples: int,
                          *args,
                          **kwargs) -> Union[
        Tuple[List[Tuple[float, _BTS_FORMAT]], Any], List[Tuple[float, _BTS_FORMAT]]]:

        """

        :param number_of_samples:
        :param args:
        :param kwargs:
        :return:
        It should return [ (best_bitstring_0, best_energy_0), ... , (best_bitstring_n, best_energy_n) ], other_data
        in accordance with self.store_n_best_results
        """

        raise NotImplementedError("This method should be implemented in a subclass")

    def sample_solutions(self,
                         number_of_samples: int,
                         *args,
                         **kwargs) -> Tuple[List[Tuple[float, _BTS_FORMAT]], Any]:

        tup_return = self._sample_solutions(number_of_samples=number_of_samples,
                                            *args,
                                            **kwargs)

        if isinstance(tup_return, tuple):
            best_n_results, other_data = tup_return
        else:
            best_n_results = tup_return
            other_data = None

        return best_n_results, other_data


# TODO FBM: refactor this, it's terrible
class EnergyResultMain:
    def __init__(self,
                 energy_mean_noiseless: Optional[float] = None,
                 energy_mean_noisy: Optional[float] = None,
                 energy_best_noiseless: Optional[float] = None,
                 energy_best_noisy: Optional[float] = None,
                 energy_mean: Optional[float] = None,
                 energy_best: Optional[float] = None,
                 bitstring_best_noiseless: Optional[Tuple[int, ...]] = None,
                 bitstring_best_noisy: Optional[Tuple[int, ...]] = None,
                 bitstring_best: Optional[Tuple[int, ...]] = None, ):

        self.energy_mean_noiseless = energy_mean_noiseless
        self.energy_mean_noisy = energy_mean_noisy
        self.energy_best_noiseless = energy_best_noiseless
        self.energy_best_noisy = energy_best_noisy
        self.bitstring_best_noiseless = bitstring_best_noiseless
        self.bitstring_best_noisy = bitstring_best_noisy

        self.energy_mean = energy_mean
        self.energy_best = energy_best
        self.bitstring_best = bitstring_best

    def update_main_energy(self,
                           noisy: bool):
        if noisy:
            self.energy_mean = self.energy_mean_noisy
            self.energy_best = self.energy_best_noisy
            self.bitstring_best = self.bitstring_best_noisy
        else:
            self.energy_mean = self.energy_mean_noiseless
            self.energy_best = self.energy_best_noiseless
            self.bitstring_best = self.bitstring_best_noiseless
        return self

    def to_dataframe_full(self):




        return pd.DataFrame(data={
            f"{SNV.EnergyMean.id_long}{SNV.Noiseless.id_long}": [self.energy_mean_noiseless],
            f"{SNV.EnergyMean.id_long}{SNV.Noisy.id_long}": [self.energy_mean_noisy],
            f"{SNV.EnergyBest.id_long}{SNV.Noiseless.id_long}": [self.energy_best_noiseless],
            f"{SNV.EnergyBest.id_long}{SNV.Noisy.id_long}": [self.energy_best_noisy],
            f"{SNV.BitstringBest.id_long}{SNV.Noiseless.id_long}": [self.bitstring_best_noiseless.tolist() if self.bitstring_best_noiseless is not None else None],
            f"{SNV.BitstringBest.id_long}{SNV.Noisy.id_long}": [self.bitstring_best_noisy.tolist() if self.bitstring_best_noisy is not None else None],},
        )

    def to_dataframe_main(self):

        return pd.DataFrame(
            data={f"{SNV.EnergyMean.id_long}": [float(self.energy_mean) if self.energy_mean is not None else None],
                  f"{SNV.EnergyBest.id_long}": [float(self.energy_best) if self.energy_best is not None else None],
                  f"{SNV.BitstringBest.id_long}": [
                      self.bitstring_best.tolist() if self.bitstring_best is not None else None]},
        )


class BestResultsContainerBase:
    def __init__(self):
        self._best_result = (np.inf, None)

    def clear(self):
        self._best_result = (np.inf, None)

    def add_result(self,
                   result_to_add: Union[List[Any], Tuple[Any]],
                   score: float):
        if score < self._best_result[0]:
            self._best_result = (score, result_to_add)
        return self

    def get_best_results(self) -> List[Tuple[float, Any]]:
        return [self._best_result]


class BestResultsContainer(BestResultsContainerBase):
    def __init__(self,
                 number_of_best_results,
                 ):

        super().__init__()

        self.number_of_best_results = number_of_best_results

        if self.number_of_best_results > 1:
            self.results_heap: List[Tuple[float, Union[List[Any], Tuple[Any]]]] = []
            # add counter as workaround for heapq's lack of support for tuples with equal values
            self.counter = itertools.count()

    def add_multiple_bitstrings_results(self,
                                        bitstrings_array,
                                        energies_array,
                                        additional_global_specifiers_tuple: Optional[Tuple[Any, ...]] = None,
                                        truncation: Optional[int] = None,
                                        ):
        # Lazy monkey-patching of cupy
        try:
            import cupy as cp
        except(ImportError, ModuleNotFoundError):
            import numpy as cp

        if isinstance(bitstrings_array, np.ndarray):
            bck = np
        elif isinstance(bitstrings_array, cp.ndarray):
            bck = cp

        pairs_bts_energy = bck.hstack((bitstrings_array, energies_array.reshape(-1, 1)))
        # sort by energy (last column value)
        pairs_bts_energy = pairs_bts_energy[bck.argsort(pairs_bts_energy[:, -1])]
        if truncation is not None:
            best_n_pairs = pairs_bts_energy[:truncation, :]
        else:
            best_n_pairs = pairs_bts_energy

        add_args = additional_global_specifiers_tuple if additional_global_specifiers_tuple is not None else ()
        for x in best_n_pairs:
            bts_x = x[:-1]
            energy_x = x[-1]

            # if bck == np:
            #  bts_x = tuple(bts_x)
            # else:
            bts_x = tuple(bts_x.tolist())

            res_to_add = (bts_x,
                          add_args)

            self.add_result(score=float(energy_x),
                            result_to_add=res_to_add
                            )

    def add_result(self,
                   result_to_add: Union[List[Any], Tuple[Any]],
                   score: float):
        if self.number_of_best_results == 1:
            return super().add_result(result_to_add, score)

        # Check if the result is already in the heap, if yes, do not update
        for _, data_present in self.results_heap:
            if data_present[0:-1] == result_to_add[0:-1]:
                try:
                    if data_present[-1] == result_to_add[-1]:
                        return self
                except(TypeError) as e:
                    return self

        result_entry = (score, result_to_add)

        self.results_heap.append(result_entry)
        self.results_heap.sort(key=lambda x: x[0],
                               reverse=False)

        if len(self.results_heap) > self.number_of_best_results:
            self.results_heap = self.results_heap[:self.number_of_best_results]

        # print('hm',len(self.results_heap))

        return self

    def get_best_results(self) -> List[Tuple[float, Any]]:
        if self.number_of_best_results == 1:
            return super().get_best_results()

        # Return results in sorted order, best first
        return [(score, data) for score, data in sorted(self.results_heap,
                                                        reverse=False,
                                                        key=lambda x: x[0])]

    def add_another_heap(self,
                         other_container: Union[BestResultsContainerBase, List[Tuple[float, Any]]]):

        if isinstance(other_container, BestResultsContainerBase):
            best_results_other = other_container.get_best_results()
        else:
            best_results_other = other_container
        for score, data in best_results_other:
            self.add_result(score=score,
                            result_to_add=data)


class OptimizationResult:

    def __init__(self,
                 best_value,
                 best_arguments,
                 trials_dataframe: pd.DataFrame = None,
                 best_point_data: Any = None):
        self._best_value = best_value
        self._best_arguments = best_arguments
        self._trials_dataframe = trials_dataframe
        self._best_point_data = best_point_data

    @property
    def best_value(self):
        return self._best_value

    @property
    def best_arguments(self):
        return self._best_arguments

    @property
    def trials_dataframe(self):
        return self._trials_dataframe

    @property
    def best_point_data(self):
        return self._best_point_data


class HamiltonianOptimizationResultsLogger(ResultsLogger):
    """Specialized ResultsLogger for Hamiltonian optimization experiments."""

    def __init__(self,
                 cost_hamiltonian: ClassicalHamiltonian,
                 table_name_prefix: Optional[str] = None,
                 table_name_suffix: Optional[str] = None,
                 experiment_specifier: Optional[StandardizedSpecifier] = None,
                 experiment_folders_hierarchy: Optional[List[str]] = None,
                 directory_main: Optional[str | Path] = None,
                 logging_level: LoggingLevel = LoggingLevel.BASIC,
                 experiment_set_name: Optional[str] = None,
                 experiment_set_id: Optional[str] = None,
                 experiment_instance_id: Optional[str] = None,
                 ):
        """
        Initialize HamiltonianOptimizationResultsLogger.
        
        This logger automatically creates appropriate folder hierarchies and table names
        based on the Hamiltonian properties while supporting the new config-based architecture.
        
        :param cost_hamiltonian: The Hamiltonian to optimize
        :param table_name_prefix: Optional prefix for table names
        :param table_name_suffix: Optional suffix for table names
        :param experiment_specifier: Optional experiment specifier (will be merged with Hamiltonian specifier)
        :param experiment_folders_hierarchy: Optional folder hierarchy
        :param directory_main: Main directory for storing results
        :param logging_level: Logging verbosity level
        :param experiment_set_name: Name of the experiment set
        :param experiment_set_id: ID of the experiment set
        :param experiment_instance_id: ID of this experiment instance
        """
        warnings.warn(
            "HamiltonianOptimizationResultsLogger is deprecated. Use ResultsLogger instead and handle "
            "metadata in runner/analyzer classes.",
            DeprecationWarning,
            stacklevel=2
        )

        # Create config using the new architecture
        config = HamiltonianOptimizationLoggerConfig(
            cost_hamiltonian=cost_hamiltonian,
            logging_level=logging_level,
            table_name_prefix=table_name_prefix,
            table_name_suffix=table_name_suffix,
            experiment_folders_hierarchy=experiment_folders_hierarchy,
            directory_main=Path(directory_main) if directory_main else None,
            experiment_set_name=experiment_set_name,
            experiment_set_id=experiment_set_id,
            experiment_instance_id=experiment_instance_id,
            experiment_specifier=experiment_specifier
        )

        # Initialize parent class with config
        super().__init__(config=config)


    def __post_init__(self):
        # in post-init, we wish to save shared metadata regarding Hamiltonian class and instance
        # to the logger's specifier
        hamiltonian_class_description:HamiltonianOptimizationLoggerConfig = self.config.CostHamiltonianClass.get_description_string()
        hamiltonian_instance_description = self.config.CostHamiltonianInstance.get_description_string()
        shared_metadata = pd.DataFrame(data={"CostHamiltonianClass": [hamiltonian_class_description],
                                             "CostHamiltonianInstance": [hamiltonian_instance_description]})
        try:
            # self._write_shared_metadata(shared_metadata=shared_metadata,
            #                             metadata_data_type=SNDT.CostHamiltonianMetadata,
            #                             overwrite_existing=False, )
            #

            self.write_metadata(metadata=shared_metadata,
                               shared_across_experiment_set=True,
                               data_type=SNDT.CostHamiltonianMetadata,
                               overwrite_existing_non_csv=False)

        except(ValueError, KeyError):
            print("Hamiltonian metadata already exists, not overwriting.")
