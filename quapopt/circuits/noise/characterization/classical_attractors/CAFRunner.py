# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from quapopt.circuits.gates import AbstractCircuit, AbstractProgramGateBuilder
from quapopt.circuits.noise.characterization.measurements.DDOT import DDOTRunner
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

from qiskit.transpiler.passmanager import StagedPassManager

from quapopt.circuits import backend_utilities as bck_utils
from quapopt.circuits.gates.gate_delays import (
    DelaySchedulerBase,
    add_delays_to_circuit_layers,
)
from quapopt.circuits.noise.characterization.classical_attractors import (
    MirrorCircuitSpecification,
    MirrorCircuitType,
    _standardized_table_name_CAF,
)
from quapopt.data_analysis.data_handling import STANDARD_NAMES_DATA_TYPES as SNDT
from quapopt.data_analysis.data_handling.io_utilities.results_logging import (
    LoggingLevel,
)
from quapopt.optimization.QAOA import MixerType, PhaseSeparatorType
from quapopt.optimization.QAOA.circuits import MappedAnsatzCircuit


class CAFRunner(DDOTRunner):
    """
    This class main function is to generate and run the Classical Attractor Finder (CAF) experiments.

    The current ideas for CAF circuits:

    1. Random X gates with various delays.
    2. Random X gates with a mirror circuit and various delays.
    """

    def __init__(
        self,
        number_of_qubits: int,
        program_gate_builder: AbstractProgramGateBuilder,
        mirror_circuit_specification: MirrorCircuitSpecification,
        sdk_name: str,
        simulation: bool,
        numpy_rng_seed: int = None,
        results_logger_kwargs: Optional[dict] = None,
        logging_level: LoggingLevel = LoggingLevel.DETAILED,
        pass_manager_qiskit_indices: StagedPassManager = None,
        qubit_indices_physical: Optional[List[int] | Tuple[int, ...]] = None,
        number_of_qubits_device_qiskit: int = None,
        # qiskit-specific kwargs
        mock_context_manager_if_simulated: bool = True,
        session_ibm=None,
        qiskit_sampler_options: Optional[dict] = None,
        noiseless_simulation: bool = False,
        qiskit_backend=None,
    ):

        # TODO(FBM): add automated, modernized metadata managment

        if sdk_name.lower() in ["qiskit"]:
            assert (
                qiskit_backend is not None
            ), "For Qiskit SDK, you need to provide a qiskit_backend."

        if logging_level is not None and logging_level != LoggingLevel.NONE:
            if results_logger_kwargs is None:
                # TODO(FBM): add defaults
                raise NotImplementedError(
                    "To use logging, you need to provide results_logger_kwargs."
                )

        mcs = mirror_circuit_specification.copy()

        if mcs.fixed_ansatz is None:
            mc_type = mirror_circuit_specification.mirror_circuit_type

            if mc_type in [
                MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK,
                MirrorCircuitType.QAOA_FULLY_CONNECTED,
            ]:
                fixed_ansatz = CAFRunner.get_qaoa_ansatz_custom(
                    phase_hamiltonian=mcs.phase_hamiltonian_qaoa,
                    program_gate_builder=program_gate_builder,
                    ansatz_kwargs=mcs.mirror_circuit_ansatz_kwargs,
                    use_linear_swap_network=mc_type
                    == MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK,
                    qubit_indices_physical=qubit_indices_physical,
                )
            elif mc_type == MirrorCircuitType.QAOA_SABRE:
                fixed_ansatz = CAFRunner.get_qaoa_ansatz_qiskit_router(
                    phase_hamiltonian=mcs.phase_hamiltonian_qaoa,
                    qiskit_pass_manager=mcs.pass_manager_mirror_circuit,
                    ansatz_kwargs=mcs.mirror_circuit_ansatz_kwargs,
                )
                if qubit_indices_physical is not None:
                    print(
                        "Warning: qubit_indices_physical is provided, but it will be ignored for the SABRE circuit type. "
                    )
                qubit_indices_physical = (
                    bck_utils.get_nontrivial_physical_indices_from_circuit(
                        quantum_circuit=fixed_ansatz.quantum_circuit,
                        filter_ancillas=True,
                    )
                )

                print("PHYSICAL QUBIT INDICES:", qubit_indices_physical)
                print("Number of qubits:", len(qubit_indices_physical))
                print("Expected:", number_of_qubits)

            else:
                raise NotImplementedError(
                    f"Mirror circuit type {mc_type} is not implemented."
                )

            mcs.fixed_ansatz = fixed_ansatz

        self._mirror_circuit_specification = mcs

        super().__init__(
            number_of_qubits=number_of_qubits,
            program_gate_builder=program_gate_builder,
            sdk_name=sdk_name,
            simulation=simulation,
            numpy_rng_seed=numpy_rng_seed,
            results_logger_kwargs=results_logger_kwargs,
            logging_level=logging_level,
            pass_manager_qiskit_indices=pass_manager_qiskit_indices,
            qubit_indices_physical=qubit_indices_physical,
            number_of_qubits_device_qiskit=number_of_qubits_device_qiskit,
            mock_context_manager_if_simulated=mock_context_manager_if_simulated,
            session_ibm=session_ibm,
            qiskit_sampler_options=qiskit_sampler_options,
            noiseless_simulation=noiseless_simulation,
            qiskit_backend=qiskit_backend,
        )

    def _get_mirror_circuit(
        self, circuit: AbstractCircuit, add_barriers: bool = True, repeats: int = 1
    ):
        """
        This function appends the adjoint of the circuit to the end of the circuit.
        :param circuit:
        :param add_barriers: if True, barriers are added between the circuit and the adjoint
        :param repeats: if repeats>1, the circuit is repeated

        :return:
        """
        if repeats == 0:
            return None

        adjoint_circuit = self._program_gate_builder.get_circuit_adjoint(
            quantum_circuit=circuit
        )

        mirror_circuit = self._program_gate_builder.copy_circuit(
            quantum_circuit=circuit
        )
        if add_barriers:
            mirror_circuit = self.append_barriers(circuit=mirror_circuit)
        mirror_circuit = self._program_gate_builder.combine_circuits(
            left_circuit=mirror_circuit, right_circuit=adjoint_circuit
        )

        if repeats > 1:
            mirror_circuit = self._program_gate_builder.repeat_circuit(
                quantum_circuit=mirror_circuit, repeats=repeats
            )

        return mirror_circuit

    def _append_delays_to_circuits(
        self,
        circuits_list: List[AbstractCircuit],
        delays_in_microseconds_list: List[float],
    ):
        """
        This function appends delays to the circuits.
        :param circuits_list:
        :param delays_in_microseconds_list:
        :return:
        """

        delayed_circuits = [[] for _ in range(len(delays_in_microseconds_list))]
        for delay_index, delay in enumerate(delays_in_microseconds_list):
            for circuit in circuits_list:
                circuit_delayed = self.append_delays(
                    circuit=circuit.copy(), duration=delay, duration_unit="us"
                )
                delayed_circuits[delay_index].append(circuit_delayed)

        return delayed_circuits

    @staticmethod
    def get_qaoa_ansatz_custom(  # self,
        phase_hamiltonian: ClassicalHamiltonian,
        program_gate_builder: AbstractProgramGateBuilder,
        use_linear_swap_network: bool = True,
        ansatz_kwargs: Optional[dict] = None,
        qubit_indices_physical: Optional[Tuple[int, ...]] = None,
        sdk_name="qiskit",
    ) -> MappedAnsatzCircuit:
        """
        wrapper for defining basic qaoa ansatz
        :param phase_hamiltonian:
        :param use_linear_swap_network:
        If True, constructs linear swap network ansatz
        if False, assumes arbitrary connectivity
        :param ansatz_kwargs:
        See LinearSwapNetworkQAOACircuit and FullyConnectedQAOACircuit for details
        :return:
        """
        if ansatz_kwargs is not None:
            ansatz_kwargs = ansatz_kwargs.copy()
            if "program_gate_builder" in ansatz_kwargs:
                del ansatz_kwargs["program_gate_builder"]

            if (
                qubit_indices_physical is None
                and "qubit_indices_physical" in ansatz_kwargs
            ):
                qubit_indices_physical = ansatz_kwargs["qubit_indices_physical"]
                del ansatz_kwargs["qubit_indices_physical"]

        if use_linear_swap_network:
            from quapopt.optimization.QAOA.circuits.LinearSwapNetworkQAOACircuit import (
                LinearSwapNetworkQAOACircuit,
            )

            if ansatz_kwargs is None:
                ansatz_kwargs = {
                    "depth": 1,
                    "time_block_size": None,
                    "phase_separator_type": None,
                    "mixer_type": None,
                }
            if "linear_chains_pair_device" not in ansatz_kwargs:
                qbts_indices = qubit_indices_physical

                # If this is not specified, we assume that the chains of qubits correspond to the list of qubits
                tuple_0_device = tuple(
                    [
                        (qbts_indices[i], qbts_indices[i + 1])
                        for i in range(0, phase_hamiltonian.number_of_qubits - 1, 2)
                    ]
                )
                tuple_1_device = tuple(
                    [
                        (qbts_indices[i], qbts_indices[i + 1])
                        for i in range(1, phase_hamiltonian.number_of_qubits - 1, 2)
                    ]
                )

                ansatz_kwargs["linear_chains_pair_device"] = [
                    tuple_0_device,
                    tuple_1_device,
                ]

            ansatz_qaoa = LinearSwapNetworkQAOACircuit(
                sdk_name=sdk_name,
                hamiltonian_phase=phase_hamiltonian,
                program_gate_builder=program_gate_builder,
                **ansatz_kwargs,
                input_state="|0>",
            )
        else:
            from quapopt.optimization.QAOA.circuits.FullyConnectedQAOACircuit import (
                FullyConnectedQAOACircuit,
            )

            if ansatz_kwargs is None:
                ansatz_kwargs = {
                    "depth": 1,
                    "time_block_size": None,
                    "phase_separator_type": None,
                    "mixer_type": None,
                }

            ansatz_qaoa = FullyConnectedQAOACircuit(
                sdk_name=sdk_name,
                depth=1,
                hamiltonian_phase=phase_hamiltonian,
                program_gate_builder=program_gate_builder,
                **ansatz_kwargs,
                initial_state="|0>",
                qubit_indices_physical=qubit_indices_physical,
            )
        return ansatz_qaoa

    @staticmethod
    def get_qaoa_ansatz_qiskit_router(
        phase_hamiltonian: ClassicalHamiltonian,
        qiskit_pass_manager: StagedPassManager,
        ansatz_kwargs: Optional[dict] = None,
    ) -> MappedAnsatzCircuit:
        # TODO(FBM): just generating it this way is likely to cause indexing errors. Should either solve this or remove

        from quapopt.optimization.QAOA.circuits.SabreMappedQAOACircuit import (
            SabreMappedQAOACircuit,
        )

        if ansatz_kwargs is None:
            ansatz_kwargs = {
                "depth": 1,
                "time_block_size": None,
                "phase_separator_type": None,
                "mixer_type": None,
            }

        # if not assert_no_ancilla_qubits:

        ansatz_qaoa = SabreMappedQAOACircuit(
            qiskit_pass_manager=qiskit_pass_manager,
            hamiltonian_phase=phase_hamiltonian,
            **ansatz_kwargs,
        )

        return ansatz_qaoa

    def _bind_parameters_mirror_circuit(
        self,
        mirror_circuit,
        mirror_circuit_type: MirrorCircuitType,
        parameter_values: np.ndarray,
        parameter_names,
    ):
        if mirror_circuit_type in [
            MirrorCircuitType.QAOA_FULLY_CONNECTED,
            MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK,
            MirrorCircuitType.QAOA_SABRE,
        ]:

            values_0 = parameter_values[0 : len(parameter_names[0])]
            values_1 = parameter_values[len(parameter_names[0]) :]

            parameters_map = {
                parameter_names[0]: values_0,
                parameter_names[1]: values_1,
            }
        else:
            raise NotImplementedError(
                "This mirror circuit type is not implemented yet."
            )

        mirror_circuit = self._program_gate_builder.copy_circuit(
            quantum_circuit=mirror_circuit
        )

        # self._program_gate_builder.resolve_parameters(quantum_circuit=mirror_circuit,
        #                                               memory_map=parameters_map)
        # raise KeyboardInterrupt
        return self._program_gate_builder.resolve_parameters(
            quantum_circuit=mirror_circuit, memory_map=parameters_map
        )

    def _generate_ansatz_circuit(
        self,
        mirror_circuit_specification: MirrorCircuitSpecification = None,
        qiskit_pass_manager: Optional[StagedPassManager] = None,
    ):
        """

        :param mirror_circuit_specification:
        :param qiskit_pass_manager:
        If the mirror circuit is QAOA Sabre type, then we require a Qiskit pass manager to be provided so it can perform
        the routing via the Sabre algorithm.


        :return:
        """
        if mirror_circuit_specification is None:
            return None, None

        mirror_circuit_type = mirror_circuit_specification.mirror_circuit_type
        mirror_circuit_ansatz_kwargs = (
            mirror_circuit_specification.mirror_circuit_ansatz_kwargs
        )
        mirror_circuit_parameters_values = (
            mirror_circuit_specification.mirror_circuit_parameters_values
        )
        phase_hamiltonian_qaoa = mirror_circuit_specification.phase_hamiltonian_qaoa

        if mirror_circuit_type is None:
            mirror_circuit_type = MirrorCircuitType.NONE

        if mirror_circuit_type == MirrorCircuitType.NONE:
            return None, None

        ansatz_circuit = mirror_circuit_specification.fixed_ansatz

        # In this case, we need to generate the ansatz
        if ansatz_circuit is None:
            if mirror_circuit_type in [
                MirrorCircuitType.QAOA_FULLY_CONNECTED,
                MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK,
            ]:
                # default phase for those types of circuits
                if phase_hamiltonian_qaoa is None:
                    # If no Hamiltonian was provided, we generate a random one. Its connectivity will depend
                    # on the type of the ansatz circuit.
                    if mirror_circuit_type == MirrorCircuitType.QAOA_FULLY_CONNECTED:
                        all_pairs = [
                            (i, j)
                            for i in range(self._number_of_qubits)
                            for j in range(i + 1, self._number_of_qubits)
                        ]
                    elif (
                        mirror_circuit_type
                        == MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK
                    ):
                        linear_chain_1 = [
                            (i, i + 1) for i in range(0, self._number_of_qubits - 1, 2)
                        ]
                        linear_chain_2 = [
                            (i, i + 1) for i in range(1, self._number_of_qubits - 1, 2)
                        ]
                        all_pairs = linear_chain_1 + linear_chain_2
                    else:
                        raise ValueError(
                            f"Unsupported mirror circuit type: {mirror_circuit_type}"
                        )

                    all_qubits = [(i,) for i in range(self._number_of_qubits)]
                    random_coefficients = (
                        2
                        * self._numpy_rng.binomial(
                            n=1, p=1 / 2, size=len(all_pairs) + len(all_qubits)
                        )
                        - 1
                    )

                    hamiltonian_list = [
                        (coeff, subset)
                        for coeff, subset in zip(
                            random_coefficients, all_pairs + all_qubits
                        )
                    ]
                    phase_hamiltonian_qaoa = ClassicalHamiltonian(
                        hamiltonian_list_representation=hamiltonian_list,
                        number_of_qubits=self._number_of_qubits,
                    )

            if mirror_circuit_type == MirrorCircuitType.QAOA_FULLY_CONNECTED:
                ansatz_circuit = self.get_qaoa_ansatz_custom(
                    phase_hamiltonian=phase_hamiltonian_qaoa,
                    use_linear_swap_network=False,
                    ansatz_kwargs=mirror_circuit_ansatz_kwargs,
                    sdk_name=self._sdk_name,
                    qubit_indices_physical=self._qubit_indices_physical,
                    program_gate_builder=self._program_gate_builder,
                )

            elif mirror_circuit_type == MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK:
                ansatz_circuit = self.get_qaoa_ansatz_custom(
                    phase_hamiltonian=phase_hamiltonian_qaoa,
                    use_linear_swap_network=True,
                    ansatz_kwargs=mirror_circuit_ansatz_kwargs,
                    sdk_name=self._sdk_name,
                    qubit_indices_physical=self._qubit_indices_physical,
                    program_gate_builder=self._program_gate_builder,
                )

            elif mirror_circuit_type == MirrorCircuitType.QAOA_SABRE:
                assert (
                    qiskit_pass_manager is not None
                ), "For SABRE swap network, we need to provide a Qiskit pass manager."
                ansatz_circuit = self.get_qaoa_ansatz_qiskit_router(
                    phase_hamiltonian=phase_hamiltonian_qaoa,
                    qiskit_pass_manager=qiskit_pass_manager,
                    ansatz_kwargs=mirror_circuit_ansatz_kwargs,
                )

            else:
                raise ValueError(
                    f"Unsupported mirror circuit type: {mirror_circuit_type}"
                )

        else:
            ansatz_circuit = ansatz_circuit.copy()

        circuit_to_be_mirrored = ansatz_circuit.quantum_circuit
        parameter_names = ansatz_circuit.parameters

        self._qubit_indices_physical = ansatz_circuit.logical_to_physical_qubits_map

        if mirror_circuit_parameters_values is not None:
            circuit_to_be_mirrored = self._bind_parameters_mirror_circuit(
                mirror_circuit=circuit_to_be_mirrored,
                mirror_circuit_type=mirror_circuit_type,
                parameter_values=mirror_circuit_parameters_values,
                parameter_names=parameter_names,
            )

        return circuit_to_be_mirrored, parameter_names

    def _generate_random_circuits_with_delays(
        self,
        number_of_circuits: int,
        enforce_uniqueness=False,
        add_measurements=True,
        add_barriers=True,
        delays_list: Optional[List[float]] = None,
        circuit_to_be_mirrored: Optional[AbstractCircuit] = None,
        mirror_circuit_repeats: int = 1,
        include_standard_ddot_circuits: bool = False,
        initial_circuit_labels: Optional[List[Tuple[int, ...]]] = None,
    ):
        """
        This function generates random circuits for the CAF experiments, assuming mirror circuit is already provided (or not)
        :param number_of_circuits:
        :param enforce_uniqueness:
        :param add_measurements:
        :param add_barriers:
        :param delays_list:
        :param circuit_to_be_mirrored:
        if not None, at the end of each circuit, we add (UU^*)^l with l=mirror_circuit_repeats
        :param mirror_circuit_repeats:

        :param include_standard_ddot_circuits:
        if True, we include standard ddot circuits in the list of circuits
        :return:
        """

        mirror_circuit = None
        if circuit_to_be_mirrored is not None and mirror_circuit_repeats > 0:
            mirror_circuit = self._get_mirror_circuit(
                circuit=circuit_to_be_mirrored,
                add_barriers=add_barriers,
                repeats=mirror_circuit_repeats,
            )

        random_circuits = self.generate_tomography_circuits_random(
            number_of_circuits=number_of_circuits,
            enforce_uniqueness=enforce_uniqueness,
            add_measurements=False,
            add_barriers=add_barriers,
            prepend_circuit=None,
            append_circuit=mirror_circuit,
            include_standard_ddot_circuits=include_standard_ddot_circuits,
            initial_circuit_labels=initial_circuit_labels,
        )

        circuit_labels, circuits_list = [tup[0] for tup in random_circuits], [
            tup[1] for tup in random_circuits
        ]

        if delays_list is None:
            delays_list = [0.0]

        circuits_with_delays = self._append_delays_to_circuits(
            circuits_list=circuits_list, delays_in_microseconds_list=delays_list
        )

        if add_measurements:

            reverse_qiskit_indices = False
            if mirror_circuit is not None:
                mirror_circuit_type = (
                    self._mirror_circuit_specification.mirror_circuit_type
                )
                if mirror_circuit_type in [
                    MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK,
                    MirrorCircuitType.QAOA_FULLY_CONNECTED,
                ]:
                    reverse_qiskit_indices = True
                elif mirror_circuit_type == MirrorCircuitType.QAOA_SABRE:
                    reverse_qiskit_indices = False
                else:
                    raise ValueError(
                        f"Unsupported mirror circuit type: {mirror_circuit_type}"
                    )

            # raise KeyboardInterrupt
            circuits_with_delays = [
                [
                    self.append_measurements(
                        circuit=circuit, reverse_indices=reverse_qiskit_indices
                    )
                    for circuit in circuits
                ]
                for circuits in circuits_with_delays
            ]

        return circuit_labels, circuits_with_delays

    def _generate_single_random_CAF_experiment(
        self,
        number_of_input_states: int,
        delay_in_microseconds: float,
        enforce_uniqueness=True,
        include_standard_ddot_circuits=True,
        initial_circuit_labels: Optional[List[Tuple[int, ...]]] = None,
        mirror_circuit_specification: MirrorCircuitSpecification = None,
        mirror_circuit_repeats: int = 1,
        qiskit_pass_manager: Optional[StagedPassManager] = None,
    ) -> Tuple[List[Tuple[int, ...]], List[AbstractCircuit], List[Any]]:
        """
        This function generates standardized random CAF experiments with (optional) mirror circuits.

        A single CAF experiments consistutes of the following:
        1. Random X gates as input state.
        2. Optional mirror circuit after the input state.
        3. Additional delay at the end of the circuit.

        The total number of circuits for this experiment is number_of_input_states.

        This function does the following:

        1. Generate random combinations of X and I gates.
        2. Potentially generate a mirror circuit with parameters specified by user.
        3. Append delay to the circuit
        4. Return the circuits and labels.

        :param number_of_input_states:
        Number of input states to be generated. This is the number of circuits that will be generated.
        :param delay_in_microseconds:
        Delay time to be appended to the circuit. This is the delay time that will be used in the experiment.
        UNIT IS MICROSECONDS
        :param enforce_uniqueness:
        If True, the circuits are generated in such a way that they are unique.
        :param qubit_mapping_type:
        If None, no mirror circuit is generated.
        :param mirror_circuit_kwargs:
        If qubit_mapping_type is not None, this dictionary is passed to the ansatz circuit.
        :param mirror_circuit_parameters_values:
        Mirror circuits are parametric, so we can provide the parameters here.
        :param mirror_circuit_repeats:
        :param phase_hamiltonian_qaoa:
        If mirror circuit is of QAOA type, this hamiltonian is used to generate the ansatz circuit.
        :param include_standard_ddot_circuits:
        If True, we include standard ddot circuits in the list of circuits.
        :param qiskit_pass_manager:
        If the mirror circuit is QAOA Sabre type, then we require a Qiskit pass manager to be provided so it can perform
        the routing via the Sabre algorithm.


        :return:

        tuple of:
        -circuit_labels:
        List of circuit labels. Shape: (number_of_circuits,); and each label is a tuple of length number_of_qubits
        -circuits_list:
        List of lists of circuits. Shape: (number_of_circuits, number_of_delays)
        every inner list corresponds to a different delay time.
        all circuits lists correspond to the fixed list of labels (circuit_labels)

        """

        circuit_to_be_mirrored, parameter_names = None, None

        if mirror_circuit_repeats != 0:
            circuit_to_be_mirrored, parameter_names = self._generate_ansatz_circuit(
                mirror_circuit_specification=mirror_circuit_specification,
                qiskit_pass_manager=qiskit_pass_manager,
            )

        # generate random circuits with the ansatz circuit after input state and a delay afterwards
        circuits_labels, circuits_list = self._generate_random_circuits_with_delays(
            number_of_circuits=number_of_input_states,
            enforce_uniqueness=enforce_uniqueness,
            add_measurements=True,
            add_barriers=True,
            delays_list=[delay_in_microseconds],
            circuit_to_be_mirrored=circuit_to_be_mirrored,
            mirror_circuit_repeats=mirror_circuit_repeats,
            include_standard_ddot_circuits=include_standard_ddot_circuits,
            initial_circuit_labels=initial_circuit_labels,
        )
        assert (
            len(circuits_list) == 1
        ), "The circuits list must be of length 1, since we only generate one delay."
        circuits_list = circuits_list[0]

        return circuits_labels, circuits_list, parameter_names

    def generate_set_of_CAF_experiments(
        self,
        # mirror_circuit_specification: MirrorCircuitSpecification,
        mirror_circuit_repeats_list: List[int],
        # delays_in_microseconds_list: List[float],
        delay_schedules_list: List[DelaySchedulerBase],
        number_of_input_states_per_experiment: int,
        qiskit_pass_manager: Optional[StagedPassManager] = None,
        initial_circuit_labels: Optional[List[Tuple[int, ...]]] = None,
        include_standard_DDOT_circuits: bool = False,
    ) -> List[Tuple[List[Tuple[int, ...]], List[AbstractCircuit], pd.DataFrame]]:
        """
        A single CAF experiment set is a CARTESIAN PRODUCT of:
        1. Random X gates as input state.
        2. Mirror circuits after X gates.
        3. Additional delays at the end of the circuit.

        So the total number of generated experiments is:
        len(mirror_circuit_specifications_list) * len(delays_list) * number_of_input_states_per_experiment

        :param mirror_circuit_specifications_list:
        List of mirror circuit specifications. See class MirrorCircuitSpecification for details.
        :param delays_in_microseconds_list:
        List of delay times to be appended to the circuits. UNIT IS MICROSECONDS
        :param number_of_input_states_per_experiment:
        Number of input states to be generated for each experiment.
        :param mirror_circuit_repeats:
        Number of times the mirror circuit is repeated.
        :param qiskit_pass_manager:
        If the mirror circuit is QAOA Sabre type, then we require a Qiskit pass manager to be provided so it can perform
        the routing via the Sabre algorithm.


        :return:
        """
        mcs = self._mirror_circuit_specification

        assert mcs.mirror_circuit_parameters_values is not None, (
            "To generate the experiments, "
            "we need to provide the parameters for the mirror circuit."
        )

        all_experiments = []

        for delay_scheduler in delay_schedules_list:
            # We create schedule of delays for given scheduler
            mcs_delay = mcs.copy()
            ansatz_delay = mcs_delay.fixed_ansatz
            circuit_base_delay = ansatz_delay.quantum_circuit
            circuit_delayed = add_delays_to_circuit_layers(
                quantum_circuit=circuit_base_delay,
                number_of_qubits=self._number_of_qubits,
                delay_scheduler=delay_scheduler,
                for_visualization=False,
                # special flag so the "delay_at_the_end" property is ignored;
                # for mirror circuit, ansatz is not the actual end of the circuit
                ignore_delay_at_the_end=True,
                ignore_add_barriers_flag=False,
            )
            ansatz_delay.quantum_circuit = circuit_delayed
            mcs_delay.fixed_ansatz = ansatz_delay

            for mcr in mirror_circuit_repeats_list:
                (circuits_labels, circuits_list, _) = (
                    self._generate_single_random_CAF_experiment(
                        number_of_input_states=number_of_input_states_per_experiment,
                        delay_in_microseconds=delay_scheduler.delay_at_the_end,
                        enforce_uniqueness=True,
                        mirror_circuit_specification=mcs_delay,
                        mirror_circuit_repeats=mcr,
                        qiskit_pass_manager=qiskit_pass_manager,
                        initial_circuit_labels=initial_circuit_labels,
                        include_standard_ddot_circuits=include_standard_DDOT_circuits,
                    )
                )

                df_metadata_exp = pd.DataFrame(
                    data={
                        "DSDescription": [delay_scheduler.get_string_description()],
                        "MCRepeats": [mcr],
                    }
                )

                all_experiments.append(
                    (circuits_labels, circuits_list, df_metadata_exp)
                )

        return all_experiments

    def run_CAF_experiments_qiskit_session(
        self,
        CAF_experiments_list: List[
            Tuple[List[Tuple[int, ...]], List[AbstractCircuit], pd.DataFrame]
        ],
        number_of_shots,
        qiskit_pass_manager_before_execution=None,
        show_progress_bar=True,
        max_attempts_run=5,
        return_results: bool = False,
        additional_metadata_experiment: Optional[pd.DataFrame] = None,
        batched_execution: bool = False,
        progress_bar_in_notebook: bool = True,
        table_name_suffix: Optional[str] = None,
    ) -> Optional[Tuple[List[Tuple[int, ...]], List[Tuple[np.ndarray, np.ndarray]]]]:
        """
        This function runs the CAF experiments using Qiskit.
        It mainly just flattens the experiments so we can use method from parent class to run qiskit session.
        It also takes care of the metadata, so we can log it properly.

        :param CAF_experiments_list:
        This list contains the circuits and their labels and additional metadata.
        See what is the format returned by self.generate_set_of_CAF_experiments.

        #for the following parameters, see the parent class
        :param qiskit_backend:
        :param simulation:
        :param number_of_shots:
        :param qiskit_sampler_options:
        :param qiskit_pass_manager_before_execution:
        :param show_progress_bar:
        :param max_attempts_run:
        :param mock_context_manager_if_simulated:
        :return:
        """

        # we will want to flatten results because we don't want to re-open session for each experiment
        # (and we'll pass them to parent class for running)
        flat_tomography_circuits = []
        table_names_flat = []
        logging_annotation = []

        if self._simulation:
            ignore_logging_level = False

        else:
            # If we're running on actual device,
            # so this should be always logged
            if self.logging_level == LoggingLevel.NONE or self.logging_level is None:
                print("RUNNING ON REAL DEVICE, CHANGING LOGGING LEVEL TO VERY DETAILED")
                self.logging_level = LoggingLevel.VERY_DETAILED
            # just to makes sure that we LOG everything
            ignore_logging_level = True

        for exp_index, (circuits_labels, circuits_list, df_metadata_exp) in enumerate(
            CAF_experiments_list
        ):
            assert (
                len(df_metadata_exp) == 1
            ), "The metadata dataframe should be of length 1"

            delay_schedule_description = df_metadata_exp["DSDescription"].values[0]
            mcr = df_metadata_exp["MCRepeats"].values[0]
            table_name_add = _standardized_table_name_CAF(
                delay_schedule_description=delay_schedule_description,
                mirror_circuit_repeats=mcr,
            )
            table_name_add = self.results_logger.join_table_name_parts(
                [table_name_add, table_name_suffix]
            )
            table_names_flat += [table_name_add] * len(circuits_labels)

            assert (
                len(df_metadata_exp) == 1
            ), "The metadata dataframe should be of length 1"

            logging_annotation += [df_metadata_exp.to_dict(orient="records")[0]] * len(
                circuits_labels
            )

            # To avoid logging metadata for each circuit (as could be done by passing exploded list to the parent class
            # runner), we just log it once here.
            # Each experiment set is uniquely identified by the delay and mcr values, so this should be unambiguous.

            if ignore_logging_level or self.logging_level != LoggingLevel.NONE:
                # logging the metadata for each experiment
                if additional_metadata_experiment is not None:
                    assert (
                        set(df_metadata_exp.columns).intersection(
                            set(additional_metadata_experiment.columns)
                        )
                        == set()
                    ), "The metadata dataframe should not contain any columns that are already in the experiment metadata."
                    df_metadata_exp = pd.concat(
                        [df_metadata_exp, additional_metadata_experiment], axis=1
                    )

                self.results_logger.write_results(
                    dataframe=df_metadata_exp,
                    data_type=SNDT.CircuitsMetadata,
                    ignore_logging_level=ignore_logging_level,
                    additional_annotation_dict=None,
                    table_name_suffix=table_name_add,
                )

            for circ_l, circ in zip(circuits_labels, circuits_list):
                flat_tomography_circuits.append((circ_l, circ))

        assert len(flat_tomography_circuits) == len(
            table_names_flat
        ), "The number of circuits and the number of table names must be the same."

        return self.run_tomography_circuits_qiskit(
            tomography_circuits=flat_tomography_circuits,
            number_of_shots=number_of_shots,
            qiskit_pass_manager=qiskit_pass_manager_before_execution,
            logging_table_names=table_names_flat,
            logging_annotation=logging_annotation,
            logging_metadata=None,
            show_progress_bar=show_progress_bar,
            max_attempts_run=max_attempts_run,
            return_results=return_results,
            batched_execution=batched_execution,
            progress_bar_in_notebook=progress_bar_in_notebook,
        )


if __name__ == "__main__":

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer.backends.aer_simulator import AerSimulator

    from quapopt.circuits.gates.logical.LogicalGateBuilderQiskit import (
        LogicalGateBuilderQiskit,
    )

    number_of_qubits_test = 3
    number_of_samples_test = 10**4
    number_of_circuits_test = 2

    computation_backend = "numpy"

    if computation_backend == "cupy":
        bck = cp
    elif computation_backend == "numpy":
        bck = np
    else:
        raise ValueError(
            f"Computation backend {computation_backend} not supported. Supported backends are: cupy, numpy."
        )

    gate_builder = LogicalGateBuilderQiskit()
    sdk_name = "qiskit"

    experiment_folders_hierarchy = ["noise_characterization", "CAF_experiments"]
    uuid = "TestUuid10"
    directory_main = None
    table_name_prefix = "TestRuns10"
    table_name_main = "CAFResults"
    results_logger_kwargs = {
        "experiment_folders_hierarchy": experiment_folders_hierarchy,
        "uuid": uuid,
        "base_path": directory_main,
        "table_name_prefix": table_name_prefix,
        "table_name_prefix": table_name_main,
    }

    CAF_test = CAFRunner(
        number_of_qubits=number_of_qubits_test,
        program_gate_builder=gate_builder,
        sdk_name=sdk_name,
        numpy_rng_seed=0,
        results_logger_kwargs=results_logger_kwargs,
        logging_level=LoggingLevel.VERY_DETAILED,
    )

    seed_cost_hamiltonian_test = 0
    numpy_rng = np.random.default_rng(seed=0)

    random_parameters = numpy_rng.uniform(low=-np.pi, high=np.pi, size=2)

    from quapopt.data_analysis.data_handling import (
        CoefficientsDistribution,
        CoefficientsDistributionSpecifier,
        CoefficientsType,
        HamiltonianModels,
    )
    from quapopt.hamiltonians.generators import build_hamiltonian_generator

    coefficients_type = CoefficientsType.DISCRETE
    coefficients_distribution = CoefficientsDistribution.Uniform
    coefficients_distribution_properties = {"low": -1, "high": 1, "step": 1}
    coefficients_distribution_specifier = CoefficientsDistributionSpecifier(
        CoefficientsType=coefficients_type,
        CoefficientsDistributionName=coefficients_distribution,
        CoefficientsDistributionProperties=coefficients_distribution_properties,
    )

    # We generate a Hamiltonian instance. In this case it's a random Sherrington-Kirkpatrick Hamiltonian
    hamiltonian_model = HamiltonianModels.SherringtonKirkpatrick
    localities = (1, 2)
    generator_cost_hamiltonian = build_hamiltonian_generator(
        hamiltonian_model=hamiltonian_model,
        localities=localities,
        coefficients_distribution_specifier=coefficients_distribution_specifier,
    )

    phase_hamiltonian_qaoa = generator_cost_hamiltonian.generate_instance(
        number_of_qubits=number_of_qubits_test,
        seed=seed_cost_hamiltonian_test,
        read_from_drive_if_present=True,
    )

    print(
        "Class description (cost):",
        phase_hamiltonian_qaoa.hamiltonian_class_description,
    )
    print(
        "Instance description (cost):",
        phase_hamiltonian_qaoa.hamiltonian_instance_description,
    )

    MCS = MirrorCircuitSpecification
    delays_list = [0.0, 1.0, 2.0]
    mirror_circuit_repeats = [0, 1, 2, 3, 4]
    mirror_circuit_specification = MCS(
        mirror_circuit_type=MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK,
        mirror_circuit_ansatz_kwargs={
            "depth": 1,
            "time_block_size": int(number_of_qubits_test / 2),
            "phase_separator_type": PhaseSeparatorType.QAOA,
            "mixer_type": MixerType.QAOA,
        },
        mirror_circuit_parameters_values=random_parameters,
        phase_hamiltonian_qaoa=phase_hamiltonian_qaoa,
    )

    all_experiments = CAF_test.generate_set_of_CAF_experiments(
        mirror_circuit_specification=mirror_circuit_specification,
        delays_in_microseconds_list=delays_list,
        number_of_input_states_per_experiment=number_of_circuits_test,
        mirror_circuit_repeats_list=mirror_circuit_repeats,
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    qiskit_backend = AerSimulator(method="statevector", device="GPU")
    pass_manager = generate_preset_pass_manager(
        backend=qiskit_backend, optimization_level=0
    )

    CAF_test.run_CAF_experiments_qiskit_session(
        CAF_experiments_list=all_experiments,
        qiskit_backend=qiskit_backend,
        simulation=True,
        number_of_shots=number_of_samples_test,
        qiskit_sampler_options=None,
        qiskit_pass_manager_before_execution=pass_manager,
        show_progress_bar=True,
        max_attempts_run=5,
        mock_context_manager_if_simulated=True,
        return_results=False,
    )

    caf_analyzer = CAFAnalyzer(
        mirror_circuit_repeats_list=mirror_circuit_repeats,
        delays_list=delays_list,
        results_list=None,
        results_logger_kwargs=results_logger_kwargs,
        noisy_sampler_post_processing=None,
        noisy_sampler_post_processing_rng=None,
        uuid=uuid,
        computation_backend=computation_backend,
    )

    raise KeyboardInterrupt

    counter = 0

    all_inputs = []
    all_outputs = []
    for circ_labels, circ_list, df_meta in all_experiments:
        print(df_meta)

        for circ_l, circ in zip(circ_labels, circ_list):
            label_should_be = final_labels_list[counter]
            res_l = final_bitstrings_list[counter]
            counter += 1
            if circ_l != label_should_be:
                raise ValueError(f"Labels do not match: {circ_l} != {label_should_be}")
            print("label:", circ_l)

            print("results:", res_l)
            print()

            all_inputs.append(np.array(circ_l, dtype=np.int32))

            all_outputs.append(res_l[0][0])

    all_inputs = np.array(all_inputs, dtype=np.int32)
    all_outputs = np.array(all_outputs, dtype=np.int32)
    print("CONSISTENT:", np.allclose(all_inputs, all_outputs))
