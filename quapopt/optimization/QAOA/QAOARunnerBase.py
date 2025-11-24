from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit.transpiler import StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from quapopt.circuits.gates.gate_delays import DelaySchedulerBase
from quapopt.data_analysis.data_handling import STANDARD_NAMES_DATA_TYPES as SNDT
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling import LoggingLevel, ResultsLogger
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)
from quapopt.optimization import HamiltonianOptimizer
from quapopt.optimization.QAOA import (
    _ANGLES_BOUNDS_LAYER_MIXER,
    _ANGLES_BOUNDS_LAYER_PHASE,
    AnsatzSpecifier,
    MixerType,
    PhaseSeparatorType,
)
from quapopt.optimization.QAOA import QAOAFunctionInputFormat as FIFormat
from quapopt.optimization.QAOA import QAOAResult, QubitMappingType
from quapopt.optimization.QAOA.implementation.QAOARunnerQiskit import QAOARunnerQiskit
from quapopt.optimization.QAOA.simulation.direct.QAOASimulatorPython import (
    QAOASimulatorPython,
)


class QAOARunnerBase(HamiltonianOptimizer):
    """
    Base class for Quantum Approximate Optimization Algorithm (QAOA) runners.

    This abstract class provides the foundation for implementing QAOA optimization algorithms
    across different quantum computing backends. It handles Hamiltonian management,
    ansatz configuration, logging, and provides the interface for running QAOA circuits.

    :param hamiltonian_representations_cost: List of cost Hamiltonians to optimize over
    :type hamiltonian_representations_cost: List[ClassicalHamiltonian]
    :param hamiltonian_representations_phase: Optional phase Hamiltonians for the QAOA ansatz
    :type hamiltonian_representations_phase: List[ClassicalHamiltonian], optional
    :param store_full_information_in_history: Whether to store complete optimization history
    :type store_full_information_in_history: bool
    :param numpy_rng_sampling: Random number generator for sampling operations
    :type numpy_rng_sampling: numpy.random.Generator, optional
    :param solve_at_initialization: Whether to solve Hamiltonians during initialization
    :type solve_at_initialization: bool
    :param logging_level: Verbosity level for result logging
    :type logging_level: LoggingLevel, optional
    :param logger_kwargs: Additional keyword arguments for logger configuration
    :type logger_kwargs: Dict[str, Any], optional
    :param ansatz_specifier: Configuration for the QAOA ansatz circuit
    :type ansatz_specifier: AnsatzSpecifier, optional
    :param store_n_best_results: Number of best optimization results to retain
    :type store_n_best_results: int

    Example:
        >>> from quapopt.hamiltonians import ClassicalHamiltonian
        >>> from quapopt.optimization.QAOA import QAOARunnerBase, AnsatzSpecifier
        >>> # Subclass implementation required
        >>> class MyQAOARunner(QAOARunnerBase):
        ...     def run_qaoa(self, *args, **kwargs):
        ...         # Implementation here
        ...         pass

    .. note::
        This is an abstract base class. Concrete implementations must override
        the abstract methods: `run_qaoa`, `log_results`, and `get_best_results`.
    """

    def __init__(
        self,
        hamiltonian_representations_cost: List[ClassicalHamiltonian],
        hamiltonian_representations_phase: List[ClassicalHamiltonian] = None,
        store_full_information_in_history=False,
        numpy_rng_sampling=None,
        solve_at_initialization=True,
        logging_level: Optional[LoggingLevel] = None,
        logger_kwargs: Optional[Dict[str, Any]] = None,
        ansatz_specifier: Optional[AnsatzSpecifier] = None,
        store_n_best_results=1,
    ):
        """
        Base class for QAOA runners. It initializes the QAOA runner with the given parameters.
        :param hamiltonian_representations_cost:
        :param hamiltonian_representations_phase:
        :param store_full_information_in_history:
        :param numpy_rng_sampling:
        :param solve_at_initialization:
        :param logging_level:
        :param logger_kwargs:
        :param ansatz_specifier:
        :param store_n_best_results:
        """

        self._angles_bounds_phase = _ANGLES_BOUNDS_LAYER_PHASE
        self._angles_bounds_MIXER = _ANGLES_BOUNDS_LAYER_MIXER

        self._backends = None
        self._backend_name = None
        self._trial_index = 0

        if numpy_rng_sampling is None:
            numpy_rng_sampling = np.random.default_rng()
        self._numpy_rng_sampling = numpy_rng_sampling

        self._ansatz_specifier = ansatz_specifier

        self._hamiltonian_representations_phase: Optional[
            Dict[int, ClassicalHamiltonian]
        ] = None

        if hamiltonian_representations_phase is not None:
            if isinstance(hamiltonian_representations_phase, list):
                example_ham = hamiltonian_representations_phase[0]
                if isinstance(example_ham, list):
                    pass
                elif isinstance(example_ham, ClassicalHamiltonian):
                    pass
                else:
                    hamiltonian_representations_phase = [
                        hamiltonian_representations_phase
                    ]
            else:
                hamiltonian_representations_phase = [hamiltonian_representations_phase]
            self._hamiltonian_representations_phase: Optional[
                Dict[int, ClassicalHamiltonian]
            ] = {}
            self.update_hamiltonians_phase(hamiltonian_representations_phase)

        super().__init__(
            input_hamiltonian_representations_cost=hamiltonian_representations_cost,
            solve_at_initialization=solve_at_initialization,
            logging_level=logging_level,
            logger_kwargs=logger_kwargs,
            store_n_best_results=store_n_best_results,
            store_full_information_in_history=store_full_information_in_history,
        )

    # @property
    # def correlators_history(self):

    @property
    def ansatz_specifier(self):
        """
        Get the current QAOA ansatz configuration.

        :returns: Current ansatz specifier containing circuit configuration details
        :rtype: AnsatzSpecifier
        """
        return self._ansatz_specifier

    # setter
    @ansatz_specifier.setter
    def ansatz_specifier(self, ansatz_specifier: AnsatzSpecifier):
        """
        Set the QAOA ansatz configuration.

        :param ansatz_specifier: New ansatz configuration specifying circuit structure, depth, and mapping
        :type ansatz_specifier: AnsatzSpecifier
        """
        self._ansatz_specifier = ansatz_specifier

    @property
    def hamiltonian_representations_phase(self) -> Dict[int, ClassicalHamiltonian]:
        """
        Get the phase Hamiltonians for the QAOA ansatz.

        If no specific phase Hamiltonians were provided, returns the cost Hamiltonians.

        :returns: Dictionary mapping indices to phase Hamiltonian representations
        :rtype: Dict[int, ClassicalHamiltonian]
        """
        if self._hamiltonian_representations_phase is None:
            return self._hamiltonian_representations_cost

        return self._hamiltonian_representations_phase

    def _update_hamiltonians_phase(
        self, hamiltonian_representations_phase: List[ClassicalHamiltonian]
    ):
        if hamiltonian_representations_phase is None:
            self._hamiltonian_representations_phase = None
            return

        if isinstance(hamiltonian_representations_phase, dict):
            hams_range = hamiltonian_representations_phase.items()
        else:
            hams_range = enumerate(hamiltonian_representations_phase)

        for idx, ham in hams_range:
            ham = self._handle_hamiltonian_type(
                ham, solve=False, number_of_qubits=self._number_of_qubits
            )
            self._hamiltonian_representations_phase[idx] = ham

    def update_hamiltonians_phase(
        self, hamiltonian_representations_phase: List[ClassicalHamiltonian]
    ):
        self._update_hamiltonians_phase(
            hamiltonian_representations_phase=hamiltonian_representations_phase
        )
        self._backends = None
        self._backend_name = None

    def reinitialize_logger(
        self,
        table_name_prefix: Optional[str] = None,
        table_name_suffix: Optional[str] = None,
        experiment_specifier=None,
        experiment_folders_hierarchy: List[str] = None,
        directory_main: Optional[str | Path] = None,
        logging_level: LoggingLevel = LoggingLevel.BASIC,
        experiment_set_name: Optional[str] = None,
        experiment_set_id: Optional[str] = None,
        experiment_instance_id: Optional[str] = None,
    ):
        """
        Initialize the QAOA results logger with the new architecture.

        :param table_name_prefix: Optional prefix for table names
        :param table_name_suffix: Optional suffix for table names
        :param experiment_specifier: Optional experiment specifier
        :param experiment_folders_hierarchy: Optional folder hierarchy
        :param directory_main: Main directory for storing results
        :param logging_level: Logging verbosity level
        :param experiment_set_name: Name of the experiment set
        :param experiment_set_id: ID of the experiment set
        :param experiment_instance_id: ID of this experiment instance
        """

        cost_hamiltonian = self.hamiltonian_representations_cost[0]

        _input_kwargs_dict = {
            "experiment_folders_hierarchy": experiment_folders_hierarchy,
            "logging_level": logging_level,
            "table_name_prefix": table_name_prefix,
            "experiment_set_name": experiment_set_name,
            "experiment_set_id": experiment_set_id,
            "experiment_instance_id": experiment_instance_id,
            "directory_main": directory_main,
            "experiment_specifier": experiment_specifier,
            "table_name_suffix": table_name_suffix,
        }

        if self.results_logger is not None:
            # we wish to go through all existing kwargs, and if user didn't provide it in the reinitialization,
            # we will use the existing values
            for existing_key, existing_value in self.results_logger.__dict__.items():
                if existing_key not in _input_kwargs_dict.keys():
                    continue

                proposed_value = _input_kwargs_dict[existing_key]
                if proposed_value is None:
                    _input_kwargs_dict[existing_key] = existing_value

        # Initialize ResultsLogger directly with basic parameters
        self._results_logger = ResultsLogger(
            table_name_prefix=table_name_prefix,
            table_name_suffix=table_name_suffix,
            experiment_folders_hierarchy=experiment_folders_hierarchy,
            directory_main=directory_main,
            logging_level=logging_level,
            experiment_set_name=experiment_set_name,
            experiment_set_id=experiment_set_id,
            experiment_instance_id=experiment_instance_id,
        )

        self.set_logging_level(logging_level=logging_level)
        ansatz_specifier = self.ansatz_specifier
        # Save shared metadata about the cost Hamiltonian, and ansatz specifier
        ansatz_description = "None"
        if ansatz_specifier is not None:
            ansatz_description = ansatz_specifier.get_description_string()
        hamiltonian_class_description = (
            cost_hamiltonian.hamiltonian_class_specifier.get_description_string()
        )
        hamiltonian_instance_description = (
            cost_hamiltonian.hamiltonian_instance_specifier.get_description_string()
        )
        shared_metadata_cost = pd.DataFrame(
            data={
                "CostHamiltonianClass": [hamiltonian_class_description],
                "CostHamiltonianInstance": [hamiltonian_instance_description],
                "AnsatzSpecifier": [ansatz_description],
            }
        )
        try:
            # self.results_logger._write_shared_metadata(shared_metadata=shared_metadata_cost,
            #                                            overwrite_existing=False)

            self.results_logger.write_metadata(
                metadata=shared_metadata_cost,
                shared_across_experiment_set=True,
                data_type=SNDT.CostHamiltonianMetadata,
                overwrite_existing_non_csv=False,
            )

        except (ValueError, KeyError):
            pass

        # We will now write all hamiltonian transformations applied to the cost Hamiltonian and the phase Hamiltonian
        cost_hamiltonians_all = self.hamiltonian_representations_cost

        data_cost = {
            SNV.HamiltonianRepresentationIndex.id_long: [],
            SNV.HamiltonianTransformationType.id_long: [],
            SNV.HamiltonianTransformationValue.id_long: [],
        }

        for i, ham_i in cost_hamiltonians_all.items():
            applied_transformations_i = ham_i.applied_transformations
            for transformation_i in applied_transformations_i:
                type_i = transformation_i.transformation
                value_i = transformation_i.value
                data_cost[SNV.HamiltonianRepresentationIndex.id_long].append(i)
                data_cost[SNV.HamiltonianTransformationType.id_long].append(
                    type_i.id_long
                )
                data_cost[SNV.HamiltonianTransformationValue.id_long].append(value_i)

        data_type_gauges = SNDT.HamiltonianTransformations
        self.results_logger.write_results(
            dataframe=pd.DataFrame(data_cost),
            data_type=data_type_gauges,
            table_name="CostHamiltonianGauges",
        )

        # TODO FBM: add the same for phase
        phase_hamiltonians_all = self._hamiltonian_representations_phase
        if phase_hamiltonians_all is not None:
            phase_hamiltonian = list(phase_hamiltonians_all.values())[0]
            phase_hamiltonian_class_description = (
                phase_hamiltonian.hamiltonian_class_specifier.get_description_string()
            )
            phase_hamiltonian_instance_description = (
                phase_hamiltonian.hamiltonian_instance_specifier.get_description_string()
            )
            shared_metadata_phase = pd.DataFrame(
                data={
                    "PhaseHamiltonianClass": [phase_hamiltonian_class_description],
                    "PhaseHamiltonianInstance": [
                        phase_hamiltonian_instance_description
                    ],
                }
            )
            try:
                self.results_logger._write_shared_metadata(
                    shared_metadata=shared_metadata_phase,
                    metadata_data_type=SNDT.PhaseHamiltonianMetadata,
                    overwrite_existing=False,
                )
            except (ValueError, KeyError):
                print("Phase Hamiltonian metadata already exists, not overwriting.")

            data_phase = {
                f"Phase{SNV.HamiltonianRepresentationIndex.id_long}": [],
                f"Phase{SNV.HamiltonianTransformationType.id_long}": [],
                f"Phase{SNV.HamiltonianTransformationValue.id_long}": [],
            }

            for i, ham_i in phase_hamiltonians_all.items():
                applied_transformations_i = ham_i.applied_transformations
                for transformation_i in applied_transformations_i:
                    type_i = transformation_i.transformation
                    value_i = transformation_i.value
                    data_phase[
                        f"Phase{SNV.HamiltonianRepresentationIndex.id_long}"
                    ].append(i)
                    data_phase[
                        f"Phase{SNV.HamiltonianTransformationType.id_long}"
                    ].append(type_i.id_long)
                    data_phase[
                        f"Phase{SNV.HamiltonianTransformationValue.id_long}"
                    ].append(value_i)

            self.results_logger.write_results(
                dataframe=pd.DataFrame(data_phase),
                data_type=data_type_gauges,
                table_name="PhaseHamiltonianGauges",
            )

    def log_results(
        self,
        qaoa_result: QAOAResult,
    ):

        raise NotImplementedError("This method must be implemented in a subclass.")

    def initialize_backend_qokit(
        self,
        qokit_backend: str = "gpu",
    ):

        if self.number_of_qubits > 30:
            raise ValueError(
                "The number of qubits is too large for qokit. Please use a different backend_computation."
            )

        # TODO FBM: figure out how to handle reversing indices for qokit
        # TODO(FBM): I think this is already handled; I modified the qokit's source code to handle this
        import quapopt.additional_packages.qokit.fur as qk_fur

        if qokit_backend.lower() in ["gpu"]:
            simclass = qk_fur.choose_simulator(name=qokit_backend)
        elif qokit_backend.lower() in ["cpu"]:
            simclass = qk_fur.choose_simulator(name="python")
        elif qokit_backend.lower() in ["auto"]:
            simclass = qk_fur.choose_simulator(name="auto")
        else:
            raise ValueError(
                "Only gpu backend_computation is supported as of now for qokit."
            )

        qokit_simulators = {}
        for i, ham_i in self.hamiltonian_representations_phase.items():
            simulator_i = simclass(
                ham_i.number_of_qubits, terms=ham_i.hamiltonian_list_representation
            )
            simulator_i.get_cost_diagonal()
            qokit_simulators[i] = simulator_i

        self._backends = qokit_simulators
        self._backend_name = f"qokit"

    def initialize_backend_qiskit(
        self,
        # qiskit kwargs
        simulation: bool,
        qiskit_backend,
        qiskit_pass_manager: StagedPassManager = None,
        qiskit_sampler_options: Optional[Dict[str, Any]] = None,
        session_ibm=None,
        #######
        program_gate_builder=None,
        number_of_qubits_device_qiskit: Optional[int] = None,
        qubit_indices_physical=None,
        classical_indices=None,
        # ansatz kwargs
        qubit_mapping_type: QubitMappingType = QubitMappingType.sabre,
        qaoa_depth=1,
        time_block_size=None,
        phase_separator_type=PhaseSeparatorType.QAOA,
        mixer_type=MixerType.QAOA,
        every_gate_has_its_own_parameter=False,
        add_barriers=False,
        delay_scheduler: DelaySchedulerBase = None,
        noiseless_simulation: bool = False,
        ##########
    ):

        if self.number_of_qubits > 30 and simulation:
            raise ValueError(
                "The number of qubits is too large for qiskit. Please use a different backend_computation."
            )

        if qiskit_pass_manager is None:
            qiskit_pass_manager = generate_preset_pass_manager(
                backend=qiskit_backend, optimization_level=0
            )

        if qiskit_backend is None:
            from qiskit_aer.backends.aer_simulator import AerSimulator

            from quapopt import AVAILABLE_SIMULATORS

            if "cuda" in AVAILABLE_SIMULATORS:
                _device = "GPU"
            else:
                _device = "CPU"

            qiskit_backend = AerSimulator(method="statevector", device=_device)

        if program_gate_builder is None:
            from quapopt.circuits.gates.logical.LogicalGateBuilderQiskit import (
                LogicalGateBuilderQiskit,
            )

            program_gate_builder = LogicalGateBuilderQiskit()

        qiskit_simulators = {}
        for i, ham_i in self.hamiltonian_representations_phase.items():

            simulator_i = QAOARunnerQiskit(
                hamiltonian_phase=ham_i,
                # Compilation kwargs specific to Qiskit
                qiskit_pass_manager=qiskit_pass_manager,
                qiskit_backend=qiskit_backend,
                program_gate_builder=program_gate_builder,
                number_of_qubits_device_qiskit=number_of_qubits_device_qiskit,
                qubit_indices_physical=qubit_indices_physical,
                classical_indices=classical_indices,
                # Ansatz kwargs
                qaoa_depth=qaoa_depth,
                time_block_size=time_block_size,
                qubit_mapping_type=qubit_mapping_type,
                phase_separator_type=phase_separator_type,
                mixer_type=mixer_type,
                every_gate_has_its_own_parameter=every_gate_has_its_own_parameter,
                add_barriers=add_barriers,
                simulation=simulation,
                qiskit_sampler_options=qiskit_sampler_options,
                mock_context_manager_if_simulated=True,
                session_ibm=session_ibm,
                delay_scheduler=delay_scheduler,
                noiseless_simulation=noiseless_simulation,
            )
            # Start session on first instance if no external session provided
            if session_ibm is None and i == 0:
                session_ibm = simulator_i.current_session
            #     # Ensure all instances share the same session

            qiskit_simulators[i] = simulator_i

        self._backends = qiskit_simulators
        self._backend_name = f"qiskit"

    def initialize_backend_python(
        self,
        backend: Optional[str] = "auto",
        time_block_size: float = 1.0,
        time_block_seed: int = 0,
        time_block_partition: Optional[Dict[int, List[Tuple[int, ...]]]] = None,
    ):

        if self.number_of_qubits > 25:
            raise ValueError(
                "The number of qubits is too large for our simulator."
                "Please use a different backend_computation."
            )

        python_simulators = {}
        for i, ham_i in self.hamiltonian_representations_phase.items():
            simulator_i = QAOASimulatorPython(
                hamiltonian_phase=ham_i,
                backend=backend,
                time_block_size=time_block_size,
                time_block_seed=time_block_seed,
                time_block_partition=time_block_partition,
            )

            python_simulators[i] = simulator_i

        self._backends = python_simulators
        self._backend_name = f"python"

    def _update_history(self, qaoa_result: QAOAResult):

        if self._store_full_information_in_history:
            self._optimization_history_full.append(qaoa_result)
        self._optimization_history_main.append(qaoa_result.to_dataframe_main())

    # TODO FBM: probably should not allowed this much flexibility
    def _input_handler(
        self,
        args,
        input_format: FIFormat,
        qaoa_depth: int,
    ):
        # OK, so we have four options for arguments here:
        # 1. angles, hamiltonian_representation_index
        # 2. angles_gamma, angles_beta, hamiltonian_representation_index
        # 3. angle_1, angle_2, ..., angle_2*qaoa_depth, hamiltonian_representation_index
        # 4. optuna.Trial
        trial_index = None
        if input_format in [FIFormat.direct_full]:
            # Arguments are passed as _fun(*args)
            angles = np.array(args[0 : 2 * qaoa_depth])
            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = args[2 * qaoa_depth]
            else:
                hamiltonian_representation_index = 0

        elif input_format in [FIFormat.direct_list]:
            # Arguments are passed as _fun(list_of_args)
            angles = np.array(args[0])
            assert (
                len(angles) == 2 * qaoa_depth
            ), "The number of angles must be equal to 2*qaoa_depth."
            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = args[1]
            else:
                hamiltonian_representation_index = 0

        elif input_format in [FIFormat.direct_vector]:
            # Arguments are passed as _fun(vector_of_angles, hamiltonian_representation_index)
            angles = np.array(args[0])
            assert (
                len(angles) == 2 * qaoa_depth
            ), "The number of angles must be equal to 2*qaoa_depth."

            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = args[1]
            else:
                hamiltonian_representation_index = 0

        elif input_format in [FIFormat.direct_QAOA]:
            # Arguments are passed as _fun([vector_gamma, vector_beta], hamiltonian_representation_index)
            angles_gamma = args[0][0]
            angles_beta = args[0][1]

            assert (
                len(angles_gamma) == qaoa_depth
            ), "The number of angles_gamma must be equal to qaoa_depth."
            assert (
                len(angles_beta) == qaoa_depth
            ), "The number of angles_beta must be equal to qaoo_depth."

            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = args[1]
            else:
                hamiltonian_representation_index = 0

            angles = np.array([ai for ai in angles_gamma] + [bi for bi in angles_beta])

        elif input_format in [FIFormat.optuna]:
            # Arguments are passed as _fun(optuna.Trial)

            trial = args[0]
            trial_index = trial._trial_id

            # TODO(FBM): Make this more flexible
            __ANGLES_BOUNDS_LAYER_PHASE__ = (-np.pi, np.pi)
            __angles_bounds_layer_MIXER__ = (-np.pi, np.pi)

            angles_bounds = [
                __ANGLES_BOUNDS_LAYER_PHASE__ for _ in range(qaoa_depth)
            ] + [__angles_bounds_layer_MIXER__ for _ in range(qaoa_depth)]

            bounds_optuna_angles = [
                (f"{SNV.Angles.id}-{index}", bound[0], bound[1])
                for index, bound in enumerate(angles_bounds[0 : len(angles_bounds)])
            ]
            choices_transformations = tuple(
                self.hamiltonian_representations_cost.keys()
            )

            angles = np.array(
                [trial.suggest_float(*xxx) for xxx in bounds_optuna_angles]
            )
            if len(self.hamiltonian_representations) > 1:
                hamiltonian_representation_index = trial.suggest_categorical(
                    name=SNV.HamiltonianRepresentationIndex.id,
                    choices=choices_transformations,
                )
            else:
                hamiltonian_representation_index = 0

        else:
            raise ValueError("Wrong input specifier")

        if trial_index is None:
            trial_index = self._trial_index
            self._trial_index += 1

        return angles, hamiltonian_representation_index, trial_index

    def run_qaoa(self, *args, **kwargs) -> QAOAResult:

        raise NotImplementedError("This method must be implemented in a subclass.")

    def run_qaoa_wrapped(
        self, *args, input_format: FIFormat = FIFormat.direct_list, **kwargs
    ):

        res_qaoa = self.run_qaoa(*args, input_format=input_format, **kwargs)

        self.update_history(qaoa_result=res_qaoa)

        energy_mean = res_qaoa.energy_result.energy_mean
        return energy_mean

    def get_best_results(self) -> List[Tuple[float, Any]]:
        raise NotImplementedError("This method must be implemented in a subclass.")
