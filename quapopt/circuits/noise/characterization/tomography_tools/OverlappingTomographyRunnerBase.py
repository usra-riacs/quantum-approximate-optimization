# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import copy
import itertools
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from qiskit.primitives.containers.bit_array import BitArray as QiskitBitArray
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit_aer.backends.aer_simulator import AerSimulator
from tqdm.notebook import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from quapopt import ancillary_functions as anf
from quapopt.circuits.backend_utilities import (
    attempt_to_run_qiskit_circuits,
    get_counts_from_bit_array,
)
from quapopt.circuits.backend_utilities.qiskit import QiskitSessionManagerMixin
from quapopt.circuits.gates import (
    AbstractCircuit,
    AbstractProgramGateBuilder,
    CircuitQiskit,
)
from quapopt.circuits.noise.characterization.tomography_tools import (
    _PAULI_TOMOGRAPHY_COUNTS_STANDARD,
    TomographyGatesType,
    TomographyType,
)
from quapopt.data_analysis.data_handling import STANDARD_NAMES_DATA_TYPES as SNDT
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling.io_utilities.results_logging import (
    LoggingLevel,
    ResultsLogger,
)


class OverlappingTomographyRunnerBase(QiskitSessionManagerMixin):
    """
    This class is used to generate and run overlapping tomography_tools circuits.
    The general idea:
    1. Generate a set of symbols for the tomography_tools circuits. Value of the symbols corresponds to a local operation,
    and position of the symbol corresponds to qubit the local operation acts on.
    2. Generate a set of circuits corresponding to the symbols. In basic version, we'll support only random Pauli
    matrices, but generalizations are possible.
    3. Run the circuits on the quantum computer and get the results.

    This can be later used to analyze the marginal distributions to investigate some aspects of the noise.
    But the details of the analysis should be relegated to the children classes that are made to implement
    analysis specific to the particular tomography_tools type.

    Some Refs:
    [1] https://arxiv.org/pdf/1908.02754
    [2] https://arxiv.org/pdf/2101.02331
    [3] https://arxiv.org/pdf/2311.10661
    [4] https://arxiv.org/pdf/2408.05730
    """

    def __init__(
        self,
        number_of_qubits: int,
        program_gate_builder: AbstractProgramGateBuilder,
        sdk_name: str,
        tomography_type: TomographyType,
        tomography_gates_type: TomographyGatesType,
        simulation: bool,
        numpy_rng_seed: int = None,
        qubit_indices_physical: tuple = None,
        parametric=False,
        results_logger_kwargs: Optional[Dict] = None,
        logging_level: LoggingLevel = LoggingLevel.DETAILED,
        pass_manager_qiskit_indices: StagedPassManager = None,
        number_of_qubits_device_qiskit: int = None,
        # qiskit-specific kwargs
        qiskit_backend=None,
        mock_context_manager_if_simulated: bool = True,
        session_ibm=None,
        qiskit_sampler_options: Optional[dict] = None,
        noiseless_simulation: bool = False,
    ):
        """

        :param number_of_qubits: Physical number of qubits in the circuit.
        :param program_gate_builder: the program gate builder that will be used to generate the circuits.
        :param sdk_name:
        Supported values:
        - 'qiskit

        :param tomography_type: for now, we only allow pre-defined tomography_tools types:
        - DIAGONAL_DETECTOR = reconstruction of diagonal parts of measurement operators
        - DETECTOR = reconstruction of full measurement operators
        - STATE = reconstruction of states
        - PROCESS = reconstruction of processes (#TODO FBM: perhaps this should be separate module)
        #TODO(FBM): in the future, perhaps allow specyfing custom tomography_tools with custom number of symbols and interpreter

        :param tomography_gates_type:
        Supported values:
        - PAULI = random Pauli states (detector tomography) or measurements (state tomography) or both (process tomography_tools)

        :param numpy_rng_seed: seed for the random number generator. If None, a random seed will be used.
        :param qubit_indices_physical: Indices of the qubits on which the circuits will be run. If None, the qubits will be
        numbered from 0 to number_of_qubits - 1.
        :param parametric: HOLDER FOR FUTURE DEVELOPMENT TODO(FBM): Implement parametric tomography_tools
        """

        if sdk_name.lower() in ["qiskit"]:
            assert (
                qiskit_backend is not None
            ), "Qiskit backend must be provided for Qiskit SDK"

        if parametric:
            # TODO FBM: maybe implement this
            raise NotImplementedError(
                "Parametric tomography_tools is not supported in this SDK yet."
            )

        if tomography_gates_type == TomographyGatesType.PAULI:
            number_of_symbols = _PAULI_TOMOGRAPHY_COUNTS_STANDARD[tomography_type]
        else:
            raise ValueError(f"Not implemented yet: {tomography_gates_type}")

        self._number_of_symbols = number_of_symbols
        self._number_of_qubits = number_of_qubits
        self._numpy_rng = np.random.default_rng(numpy_rng_seed)
        self._program_gate_builder = program_gate_builder
        self._sdk_name = sdk_name

        self._pass_manager_qiskit_indices = pass_manager_qiskit_indices

        self._tomography_type = tomography_type
        self._tomography_gates_type = tomography_gates_type

        if qubit_indices_physical is None:
            qubit_indices_physical = tuple(range(number_of_qubits))
        self._qubit_indices_physical = qubit_indices_physical

        if number_of_qubits_device_qiskit is None:
            number_of_qubits_device_qiskit = max(qubit_indices_physical) + 1
        self._number_of_qubits_device_qiskit = number_of_qubits_device_qiskit

        # TODO FBM: maybe implement this
        if sdk_name in ["qiskit"] and parametric:
            parametric = False
            print(
                "Parametric tomography is not supported in Qiskit yet. Setting parametric to False."
            )

        elif parametric:
            raise NotImplementedError(
                "Parametric tomography is not supported in this SDK yet."
            )

        self._parametric = parametric

        self._logging_level = logging_level

        if logging_level is not None and logging_level != LoggingLevel.NONE:
            # TODO(FBM): add default results logger kwargs
            if results_logger_kwargs is None:
                raise NotImplementedError(
                    "Results logger kwargs must be provided if logging level is not NONE."
                )

        self._results_logger = (
            ResultsLogger(**results_logger_kwargs)
            if results_logger_kwargs is not None
            else None
        )

        if self._results_logger is not None:
            self._results_logger.set_logging_level(level=logging_level)

        if self._sdk_name == "qiskit":

            # Initialize session management via mixin
            self._init_session_management(
                qiskit_backend=qiskit_backend,
                simulation=simulation,
                mock_context_manager_if_simulated=mock_context_manager_if_simulated,
                session_ibm=session_ibm,
                qiskit_sampler_options=qiskit_sampler_options,
                noiseless_simulation=noiseless_simulation,
            )

    @property
    def results_logger(self):
        return self._results_logger

    @property
    def logging_level(self):
        return self.results_logger.logging_level

    @logging_level.setter
    def logging_level(self, value):
        if value is None:
            value = LoggingLevel.NONE
        if not isinstance(value, LoggingLevel):
            raise ValueError("Logging level must be an instance of LoggingLevel.")
        self._results_logger.set_logging_level(level=value)

    def append_measurements(
        self,
        circuit: AbstractCircuit,
        qubit_indices: Optional[List[int]] = None,
        reverse_indices: bool = False,
    ) -> AbstractCircuit:
        """
        This function appends measurement instructions to the circuit.
        :param circuit:
        :return:
        """
        if qubit_indices is None:
            qubit_indices = self._qubit_indices_physical

        circuit = self._program_gate_builder.add_measurements(
            quantum_circuit=circuit,
            qubit_indices=qubit_indices,
            reverse_indices=reverse_indices,
        )
        return circuit

    def append_delays(
        self,
        circuit: CircuitQiskit,
        duration: float,
        duration_unit="us",
        qubit_indices: Optional[List[int]] = None,
    ) -> CircuitQiskit:
        """
        This function appends delay instructions to the circuit.
        :param circuit:
        :param duration:
        :param duration_unit:
        :return:
        """

        # TODO(FBM): compilation question: does 0 delay do exactly this or not?
        if duration == 0.0:
            return circuit

        if qubit_indices is None:
            qubit_indices = self._qubit_indices_physical

        if self._sdk_name in ["qiskit"]:
            circuit.delay(duration=duration, unit=duration_unit, qarg=qubit_indices)
        else:
            # TODO FBM: implement this
            raise NotImplementedError("Delays are not supported in this SDK yet.")
        return circuit

    def append_barriers(
        self, circuit: CircuitQiskit, qubit_indices: Optional[List[int]] = None
    ) -> CircuitQiskit:
        """
        This function appends barrier instructions to the circuit.
        :param circuit:
        :return:
        """
        if qubit_indices is None:
            qubit_indices = self._qubit_indices_physical

        if self._sdk_name in ["qiskit"]:
            circuit.barrier(qubit_indices)
        else:
            # TODO FBM: implement this
            raise NotImplementedError("Barriers are not supported in this SDK yet.")
        return circuit

    def _generate_symbols_random(
        self,
        number_of_circuits: int,
        number_of_symbols: int,
        enforce_uniqueness=False,
        prepended_symbols_list=None,
    ) -> List[Tuple[int, ...]]:
        """
        This function generates random symbols for the tomography_tools circuits.
        :param number_of_circuits:
        :param number_of_symbols:
        :param enforce_uniqueness:
        If True, the generated number of symbols will be unique.
        Warning: for small number of qubits and large number of circuits, this might not be possible.
        :param prepended_symbols_list:
        A list of symbols that will be prepended to the generated symbols.
        If None, an empty list will be used.
        :return:
        """
        if number_of_circuits == 0:
            return prepended_symbols_list if prepended_symbols_list is not None else []

        if enforce_uniqueness:
            if self._number_of_qubits < 20:
                possible_combinations = number_of_symbols**self._number_of_qubits

                if (
                    number_of_circuits + len(prepended_symbols_list)
                    > possible_combinations
                ):
                    print("possible combinations", possible_combinations)
                    print("number of circuits", number_of_circuits)
                    print("number of qubits:", self._number_of_qubits)
                    print("number of symbols:", number_of_symbols)
                    raise ValueError(
                        "Cannot enforce uniqnuess for more circuits than possible combinations"
                    )

        if number_of_symbols == 1:
            random_symbols = [[0] * self._number_of_qubits] * number_of_circuits

            if enforce_uniqueness and number_of_circuits > 1:
                raise ValueError("Cannot enforce uniqueness with only one symbol")

        elif number_of_symbols == 2:
            random_symbols = self._numpy_rng.binomial(
                n=1, p=1 / 2, size=(number_of_circuits, self._number_of_qubits)
            )

        else:
            random_symbols = self._numpy_rng.integers(
                low=0,
                high=number_of_symbols,
                size=(number_of_circuits, self._number_of_qubits),
            )

        if prepended_symbols_list is None:
            prepended_symbols_list = []

        # TODO(FBM): this conversion is not very efficient for large objects
        random_symbols = prepended_symbols_list + [
            tuple(x.tolist()) for x in random_symbols
        ]
        if enforce_uniqueness:
            random_symbols = list(set(random_symbols))
            if len(random_symbols) < number_of_circuits + len(prepended_symbols_list):
                missing_symbols = (
                    number_of_circuits
                    + len(prepended_symbols_list)
                    - len(random_symbols)
                )
                return self._generate_symbols_random(
                    number_of_circuits=missing_symbols,
                    number_of_symbols=number_of_symbols,
                    enforce_uniqueness=enforce_uniqueness,
                    prepended_symbols_list=random_symbols,
                )

        return random_symbols

    def _generate_symbols_all_combinations(
        self, number_of_symbols: int
    ) -> List[Tuple[int, ...]]:
        """
        This function generates all combinations of symbols for the tomography_tools circuits on given number of qubits.
        Warning: since this number grows exponentially, it returns an error for over million circuits
        :param number_of_symbols:
        :return:
        """

        if number_of_symbols**self._number_of_qubits > 2**20:
            raise ValueError(
                "The number of combinations is too large, please use a smaller number of symbols or qubits"
            )

        return [
            tuple(x)
            for x in itertools.product(
                range(number_of_symbols), repeat=self._number_of_qubits
            )
        ]

    def _get_tomography_circuit_nonparametric_pauli(
        self,
        circuit_label: Union[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[int, ...]]],
        prepend_circuit_global=None,
        append_circuit_global=None,
        middle_circuit_global=None,
        add_measurements=True,
        add_barriers=True,
    ) -> AbstractCircuit:
        """
        This function generates a tomography_tools circuit for the given circuit label.
        :param circuit_label: List of symbols for the tomography_tools circuit.
        :param prepend_circuit_global: Circuit to be prepended to ALL circuits
        :param append_circuit_global: Circuit to be appended to ALL circuits
        :param middle_circuit_global: Circuit to be inserted in the middle (this makes sense FOR PROCESS TOMOGRAPHY ONLY)


        :param add_measurements: whether to add measurement instructions to the end of the circuit.
        :param add_barriers: whether to add barrier instructions to the end of the circuit.
        :return:
        """
        if middle_circuit_global is not None:
            assert (
                self._tomography_type == TomographyType.PROCESS
            ), "Middle circuit is only supported for process tomography"

        circuit_symbols_prepend, circuit_symbols_append = None, None
        if self._tomography_type == TomographyType.PROCESS:
            circuit_symbols_prepend, circuit_symbols_append = circuit_label
        elif self._tomography_type in [
            TomographyType.DIAGONAL_DETECTOR,
            TomographyType.QUANTUM_DETECTOR,
        ]:
            circuit_symbols_prepend = circuit_label
            circuit_symbols_append = None
        elif self._tomography_type in [TomographyType.STATE]:
            circuit_symbols_prepend = None
            circuit_symbols_append = circuit_label

        _qubit_indices = self._qubit_indices_physical

        if self._sdk_name in ["qiskit"]:
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(
                self._number_of_qubits_device_qiskit, self._number_of_qubits
            )

        else:
            raise NotImplementedError(f"SDK {self._sdk_name} is not supported yet")

        if prepend_circuit_global is not None:
            circuit = self._program_gate_builder.combine_circuits(
                left_circuit=circuit, right_circuit=prepend_circuit_global
            )

        # This is not None for detector tomography or process tomography
        if circuit_symbols_prepend is not None:
            number_of_symbols = self._number_of_symbols
            for symbol_i in range(number_of_symbols):
                qi_physical = [
                    _qubit_indices[index_i]
                    for index_i, label_i in enumerate(circuit_symbols_prepend)
                    if label_i == symbol_i
                ]
                if symbol_i == 0:
                    # Initialize in |0>
                    circuit = self._program_gate_builder.I(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                elif symbol_i == 1:
                    # Initialize in |1>
                    circuit = self._program_gate_builder.X(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                elif symbol_i == 2:
                    # Initialize in |+>
                    circuit = self._program_gate_builder.H(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                elif symbol_i == 3:
                    # Initialize in |->
                    circuit = self._program_gate_builder.X(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                    circuit = self._program_gate_builder.H(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                elif symbol_i == 4:
                    # Initialize in |i+>
                    circuit = self._program_gate_builder.H(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                    circuit = self._program_gate_builder.S(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                elif symbol_i == 5:
                    # Initialize in |i->
                    circuit = self._program_gate_builder.X(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                    circuit = self._program_gate_builder.H(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                    circuit = self._program_gate_builder.S(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                else:
                    raise ValueError(
                        f"Pauli detector tomography_tools supports only 6 labels, got {symbol_i}"
                    )

        if middle_circuit_global is not None:
            circuit = self._program_gate_builder.combine_circuits(
                left_circuit=circuit, right_circuit=middle_circuit_global
            )

        # This is not None for state tomography or process tomography
        if circuit_symbols_append is not None:
            number_of_symbols = self._number_of_symbols

            for symbol_i in range(number_of_symbols):
                qi_physical = [
                    _qubit_indices[index_i]
                    for index_i, label_i in enumerate(circuit_symbols_append)
                    if label_i == symbol_i
                ]
                if symbol_i == 0:
                    # Measure in Z basis (this is default basis typically)
                    continue
                elif symbol_i == 1:
                    # Measure in X basis
                    circuit = self._program_gate_builder.H(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                elif symbol_i == 2:
                    # Measure in Y basis
                    circuit = self._program_gate_builder.Sdag(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                    circuit = self._program_gate_builder.H(
                        quantum_circuit=circuit, qubits_tuple=qi_physical
                    )
                else:
                    raise ValueError(
                        f"Pauli detector tomography_tools supports only 3 labels, got {symbol_i}"
                    )

        if append_circuit_global is not None:
            # print(len(bck_utils.get_nontrivial_physical_indices_from_circuit(quantum_circuit=append_circuit_global)),
            #  len(append_circuit_global.qubits))
            circuit = self._program_gate_builder.combine_circuits(
                left_circuit=circuit, right_circuit=append_circuit_global
            )

        if add_barriers:
            if self._sdk_name != "qiskit":
                raise NotImplementedError(
                    f"Barriers are not supported in {self._sdk_name} yet."
                )
            circuit = self.append_barriers(
                circuit=circuit, qubit_indices=_qubit_indices
            )

        if add_measurements:
            circuit = self.append_measurements(
                circuit=circuit, qubit_indices=_qubit_indices
            )
        return circuit

    def _generate_tomography_circuits_nonparametric_random(
        self,
        number_of_circuits=1,
        enforce_uniqueness=True,
        add_measurements=True,
        add_barriers=True,
        prepend_circuit: Optional[AbstractCircuit] = None,
        append_circuit: Optional[AbstractCircuit] = None,
        middle_circuit: Optional[AbstractCircuit] = None,
        prepended_symbols_list: Optional[List[Tuple[int, ...]]] = None,
    ) -> List[
        Tuple[Union[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[int, ...]]], Any]
    ]:
        """
        This function generates random tomography_tools circuits.

        it calls:
        1. self._generate_symbols_random() to generate random symbols for the tomography_tools circuits.
        and
        2. self._get_tomography_circuit_nonparametric_pauli() to generate the circuits

        :param number_of_circuits:
        :param enforce_uniqueness:
        :param add_measurements:
        :param add_barriers:
        :param prepend_circuit: Circuit to be prepended to ALL circuits
        :param append_circuit: Circuit to be appended to ALL circuits
        :param middle_circuit: Circuit to be inserted in the middle (this makes sense FOR PROCESS TOMOGRAPHY ONLY)
        :param prepended_symbols_list: List of symbols that should be included in the generated symbols.
        :return:
        """

        if prepended_symbols_list is not None:
            assert len(prepended_symbols_list) <= number_of_circuits, (
                f"If you want to prepend symbols, "
                "you need to generate at least as many circuits."
                f"Here we have: {len(prepended_symbols_list)} <= {number_of_circuits}"
            )
            number_of_circuits += -len(prepended_symbols_list)

        if self._tomography_type == TomographyType.PROCESS:
            if enforce_uniqueness:
                # TODO FBM: implement this
                raise NotImplementedError("Not implemented yet: enforce_uniqnuess")
            circuit_symbols_1 = self._generate_symbols_random(
                number_of_circuits=number_of_circuits,
                number_of_symbols=self._number_of_symbols[0],
                prepended_symbols_list=prepended_symbols_list,
            )
            circuit_symbols_2 = self._generate_symbols_random(
                number_of_circuits=number_of_circuits,
                number_of_symbols=self._number_of_symbols[1],
                prepended_symbols_list=prepended_symbols_list,
            )
            circuit_symbols = list(zip(circuit_symbols_1, circuit_symbols_2))
        else:
            circuit_symbols = self._generate_symbols_random(
                number_of_circuits=number_of_circuits,
                number_of_symbols=self._number_of_symbols,
                enforce_uniqueness=enforce_uniqueness,
                prepended_symbols_list=prepended_symbols_list,
            )

        all_circuits = []
        for circuit_label in circuit_symbols:
            circuit_i = self._get_tomography_circuit_nonparametric_pauli(
                circuit_label=circuit_label,
                prepend_circuit_global=prepend_circuit,
                append_circuit_global=append_circuit,
                middle_circuit_global=middle_circuit,
                add_measurements=add_measurements,
                add_barriers=add_barriers,
            )
            all_circuits.append((circuit_label, circuit_i))

        return all_circuits

    def _generate_tomography_circuits_random(
        self,
        number_of_circuits=1,
        enforce_uniqueness=True,
        add_measurements=True,
        add_barriers=True,
        prepend_circuit: Optional[AbstractCircuit] = None,
        append_circuit: Optional[AbstractCircuit] = None,
        middle_circuit: Optional[AbstractCircuit] = None,
        prepended_symbols_list: Optional[List[Tuple[int, ...]]] = None,
    ) -> List[Tuple[Tuple[int, ...], Any]]:
        """
        This function generates random tomography circuits. Simple wrapper that decides between
        parametric and non-parametric tomography circuits.
        Currently it's redundant cause we haven't finished implementing parametric tomography yet.

        :param number_of_circuits:
        :param enforce_uniqueness:
        :param add_measurements:
        :param add_barriers:
        :param prepend_circuit: Circuit to be prepended to ALL circuits
        :param append_circuit: Circuit to be appended to ALL circuits
        :param middle_circuit: Circuit to be inserted in the middle (this makes sense FOR PROCESS TOMOGRAPHY ONLY)
        :param prepended_symbols_list: List of symbols that should be included in the generated symbols.
        :return:
        """

        if self._parametric:
            raise NotImplementedError(
                "Parametric tomography_tools is not supported yet."
            )
        else:
            return self._generate_tomography_circuits_nonparametric_random(
                number_of_circuits=number_of_circuits,
                enforce_uniqueness=enforce_uniqueness,
                add_measurements=add_measurements,
                add_barriers=add_barriers,
                prepend_circuit=prepend_circuit,
                append_circuit=append_circuit,
                middle_circuit=middle_circuit,
                prepended_symbols_list=prepended_symbols_list,
            )

    def run_tomography_circuits_qiskit(
        self,
        tomography_circuits: List[Tuple[Tuple[int, ...], CircuitQiskit]],
        number_of_shots,
        qiskit_pass_manager=None,
        logging_metadata: Optional[List[pd.DataFrame]] = None,
        logging_annotation: Optional[List[dict]] = None,
        logging_table_names: Optional[List[str]] = None,
        show_progress_bar=True,
        max_attempts_run=5,
        return_results: bool = True,
        metadata_experiment_set: Optional[pd.DataFrame] = None,
        confirm_runs_if_on_hardware: bool = True,
        batched_execution: bool = False,
        progress_bar_in_notebook: bool = True,
    ) -> Optional[Tuple[List[Tuple[int, ...]], List[Tuple[np.ndarray, np.ndarray]]]]:
        """
        This function runs the tomography circuits on the given backend_computation.
        :param tomography_circuits:
        List of tuples (circuit_label, circuit) where circuit is a Qiskit circuit.
        :param qiskit_backend:
        Qiskit backend_computation to run the circuits on.
        :param simulation:
        If True, the circuits will be run on a simulator.
        If False, the circuits will be run on a real device.
        :param number_of_shots:
        Number of shots to run the circuits with.
        :param qiskit_sampler_options:
        Options for the Qiskit sampler. If None, default options will be used.
        :param qiskit_pass_manager:
        Qiskit pass manager to use for transpiling the circuits. If None, default pass manager will be used.
        :param logging_metadata:
        Metadata to be logged for each circuit. If None, no metadata will be logged.
        This metadata is a dataframe that is saved in separate file than the results.
        :param logging_annotation:
        Annotation to be logged for each circuit. If None, no annotation will be logged.
        This annotation is a dictionary that is saved in the same file as the results (and potentially logging_metadata)

        ADVICE: if you have a lot of circuits, logging_metadata is recommended to use to annotate data,
        because we expect large number of df rows in the results. Adding anotation to each row can
        highly increase the size of the file.

        :param logging_table_names:
        Table names to be used for logging the results. If None, default table names will be used.
        This is a list of strings that will be used as suffixes for the table names.

        :param show_progress_bar:
        If True, a progress bar will be shown while running the circuits.
        :param max_attempts_run:
        Maximum number of attempts to run the circuit. If the circuit fails, it will be retried
        :param mock_context_manager_if_simulated:
        If True, the context manager will be mocked when running on a simulator.
        This is useful for testing purposes.

        :param return_results
        If True, the results will be returned as a tuple of (labels_list, bitstrings_array_histograms).
        Otherwise, it does not return anything and results can be accessed from the results logger.

        :return:
        """

        if batched_execution:
            # TODO(FBM): add batched execution (watch out for possible memory issues?) (what about error handling)?
            raise NotImplementedError("batched_execution is not implemented yet")

        if not self._simulation and confirm_runs_if_on_hardware:
            if not anf.query_yes_no("WARNING! RUNNING ON REAL HARDWARE! CONTINUE?"):
                raise ValueError("User did not confirm to run on real hardware.")

        circuits_list = [x[1] for x in tomography_circuits]
        labels_list = [x[0] for x in tomography_circuits]

        # just to makes sure that we LOG everything in case of running experiments
        ignore_logging_level = not self._simulation

        circuits_list_isa = circuits_list.copy()

        if qiskit_pass_manager is not None:
            circuits_list_isa = qiskit_pass_manager.run(circuits_list_isa)

        # TODO(FBM): somehow isa manager sometimes adds global phase to
        # the circuit and this returns an error during circuit run; the following is a workaround.
        # It should not change anything unless the circuit is used in some control block
        for circ in circuits_list_isa:
            circ.data = [instr for instr in circ.data if instr.name != "global_phase"]

        if isinstance(logging_annotation, dict):
            logging_annotation = [logging_annotation] * len(circuits_list_isa)
        if isinstance(logging_table_names, str):
            logging_table_names = [logging_table_names] * len(circuits_list_isa)
        if isinstance(logging_metadata, pd.DataFrame):
            logging_metadata = [logging_metadata] * len(circuits_list_isa)

        if ignore_logging_level or self.logging_level != LoggingLevel.NONE:
            if metadata_experiment_set is not None:
                self.results_logger._write_shared_metadata(
                    metadata_data_type=SNDT.CircuitsMetadata,
                    shared_metadata=metadata_experiment_set,
                    overwrite_existing=False,
                )

        if logging_annotation is not None:
            assert len(logging_annotation) == len(
                circuits_list_isa
            ), "Logging annotation length does not match number of circuits"
        if logging_table_names is not None:
            assert len(logging_table_names) == len(
                circuits_list_isa
            ), "Logging table names length does not match number of circuits"
        if logging_metadata is not None:
            assert len(logging_metadata) == len(
                circuits_list_isa
            ), "Logging metadata length does not match number of circuits"

        bitstrings_array_histograms = []

        sampler_ibm = self._ensure_sampler()

        anf.cool_print("RUNNING EXPERIMENTS", "...", "green")

        if progress_bar_in_notebook:
            _tqdm = tqdm_notebook
        else:
            _tqdm = tqdm

        for circuit_index, (circuit_label, circuit_isa) in _tqdm(
            enumerate(zip(labels_list, circuits_list_isa)),
            desc="Running experiments",
            colour="yellow",
            disable=not show_progress_bar,
            total=len(circuits_list_isa),
        ):

            time.perf_counter()
            _success, job_circuit, results_circuit, df_job_metadata_run = (
                attempt_to_run_qiskit_circuits(
                    circuits_isa=circuit_isa,
                    sampler_ibm=sampler_ibm,
                    number_of_shots=number_of_shots,
                    max_attempts_run=max_attempts_run,
                    metadata_for_error_printing=circuit_label,
                )
            )

            job_id = None
            if df_job_metadata_run is not None:
                job_id = df_job_metadata_run.iloc[0][SNV.JobId.id_long]

            if ignore_logging_level or self.logging_level != LoggingLevel.NONE:
                # CircuitLabel is not necessary unique.
                # For random experiments on large system sizes
                # it is extremely unlikely to get two identical labels
                # But sometimes we can intentionally run the same circuit
                # multiple times so we need to have a way of identifying
                # which job it came from.
                # We can use job_id for that, but it's usually a large string.
                # hence, to save memory, instead we use circuit_index and we can match them later.
                df_job_metadata = pd.DataFrame(
                    data={
                        "Success": [_success],
                        SNV.CircuitLabel.id_long: [circuit_label],
                        # SNV.CircuitIndex.id_long: [circuit_index],
                    }
                )
                df_job_metadata = pd.concat(
                    [df_job_metadata, df_job_metadata_run], axis=1
                )

                if logging_metadata is not None:
                    df_job_metadata = pd.concat(
                        [df_job_metadata, logging_metadata[circuit_index]], axis=1
                    )

                self.results_logger.write_results(
                    dataframe=df_job_metadata,
                    data_type=SNDT.JobMetadata,
                    ignore_logging_level=ignore_logging_level,
                    additional_annotation_dict=(
                        logging_annotation[circuit_index]
                        if logging_annotation is not None
                        else None
                    ),
                    table_name_suffix=(
                        logging_table_names[circuit_index]
                        if logging_table_names is not None
                        else None
                    ),
                )

            if not _success:
                print("Skipping circuit:", circuit_label)
                continue

            data_c: QiskitBitArray = results_circuit[0].data.c

            time.perf_counter()

            bitstrings_res, counts_res = get_counts_from_bit_array(bit_array=data_c)
            bitstrings_array_histograms.append((bitstrings_res, counts_res))

            if ignore_logging_level or self.logging_level != LoggingLevel.NONE:
                df_bitstrings = pd.DataFrame(
                    data={
                        SNV.CircuitLabel.id_long: [circuit_label] * len(bitstrings_res),
                        SNV.Bitstring.id_long: [x for x in bitstrings_res],
                        SNV.Count.id_long: counts_res,
                        SNV.JobId.id_long: [job_id] * len(bitstrings_res),
                    }
                )

                self.results_logger.write_results(
                    dataframe=df_bitstrings,
                    data_type=SNDT.BitstringsHistograms,
                    ignore_logging_level=ignore_logging_level,
                    additional_annotation_dict=(
                        logging_annotation[circuit_index]
                        if logging_annotation is not None
                        else None
                    ),
                    table_name_suffix=(
                        logging_table_names[circuit_index]
                        if logging_table_names is not None
                        else None
                    ),
                )

            time.perf_counter()

        if return_results:
            return labels_list, bitstrings_array_histograms


if __name__ == "__main__":
    from quapopt.circuits.gates.logical.LogicalGateBuilderQiskit import (
        LogicalGateBuilderQiskit,
    )

    tomography_type_test = TomographyType.DIAGONAL_DETECTOR
    tomography_gates_type_test = TomographyGatesType.PAULI
    noq_test = 3

    gate_builder = LogicalGateBuilderQiskit()
    sdk_name = "qiskit"

    OTR_test = OverlappingTomographyRunnerBase(
        number_of_qubits=noq_test,
        program_gate_builder=gate_builder,
        sdk_name=sdk_name,
        tomography_type=tomography_type_test,
        tomography_gates_type=tomography_gates_type_test,
        numpy_rng_seed=42,
        parametric=False,
        logging_level=LoggingLevel.DETAILED,
    )

    random_circuits = OTR_test._generate_tomography_circuits_random(
        number_of_circuits=2**noq_test,
        enforce_uniqueness=True,
        add_measurements=False,
        add_barriers=True,
    )

    qiskit_backend = AerSimulator(method="statevector", device="CPU")

    delays_list = [0.0, 0.5, 1.0]

    all_results = []
    for delay in delays_list:
        circuits_delay = copy.deepcopy(random_circuits)
        circuits_delay_run = []
        for circ_label, circ in circuits_delay:
            circ = OTR_test.append_delays(
                circuit=circ, duration=delay, duration_unit="s"
            )
            circ = OTR_test.append_measurements(circuit=circ)
            circuits_delay_run.append((circ_label, circ))

        df_res_delay = OTR_test.run_tomography_circuits_qiskit(
            tomography_circuits=circuits_delay_run,
            qiskit_backend=qiskit_backend,
            shots=1000,
            df_annotation={"delay": delay, "delay_unit": "s"},
        )
        all_results.append(df_res_delay)

    for circuit_label, circuit in random_circuits:
        print(circuit_label)
        print(circuit)
        print("---")

    df_res = pd.concat(all_results, axis=0)
    print(df_res)
    inputs = np.array(df_res["CircuitLabel"].tolist())
    outputs = np.array(df_res["Bitstring"].tolist())
    print("CONSISTENT:", np.allclose(inputs, outputs))
