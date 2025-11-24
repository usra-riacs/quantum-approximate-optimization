# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from functools import partial
from typing import Optional

from pydantic import conint

from quapopt.circuits.gates import (
    AbstractCircuit,
    AbstractProgramGateBuilder,
    _SUPPORTED_SDKs,
    cirq,
    pyquil,
    qiskit,
)
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)
from quapopt.optimization.QAOA import (
    AnsatzSpecifier,
    MixerType,
    PhaseSeparatorType,
    QubitMappingType,
)
from quapopt.optimization.QAOA.circuits import MappedAnsatzCircuit
from quapopt.optimization.QAOA.circuits.time_block_ansatz import (
    build_fractional_time_block_ansatz_qiskit,
)


class FullyConnectedQAOACircuit(MappedAnsatzCircuit):
    """
    QAOA ansatz circuit optimized for fully-connected qubit topologies.

    This class constructs parameterized Quantum Approximate Optimization Algorithm (QAOA)
    circuits assuming all-to-all qubit connectivity. Unlike hardware-constrained circuits,
    this implementation can directly apply two-qubit gates between any pair of qubits without
    requiring routing or SWAP gates. It supports both standard QAOA and fractional time
    blocking approaches.

    The circuit construction is multi-SDK compatible (Qiskit, PyQuil, Cirq) and provides
    flexible control over circuit structure including phase separator types, mixer types,
    and initial state preparation.

    :param sdk_name: Quantum SDK to use for circuit construction ('qiskit', 'pyquil', 'cirq')
    :type sdk_name: str
    :param depth: Number of QAOA layers (p parameter)
    :type depth: int
    :param hamiltonian_phase: Phase Hamiltonian defining the optimization problem
    :type hamiltonian_phase: ClassicalHamiltonian
    :param program_gate_builder: Gate builder for SDK-specific gate implementations
    :type program_gate_builder: AbstractProgramGateBuilder
    :param time_block_size: Fraction of Hamiltonian interactions per layer (0.0-1.0) or None for standard QAOA
    :type time_block_size: float, optional
    :param phase_separator_type: Type of phase separator gates (QAOA or QAMPA)
    :type phase_separator_type: PhaseSeparatorType
    :param mixer_type: Type of mixer gates (QAOA or QAMPA)
    :type mixer_type: MixerType
    :param every_gate_has_its_own_parameter: Whether each gate gets independent parameters (not yet supported)
    :type every_gate_has_its_own_parameter: bool
    :param qubit_indices_physical: Optional mapping to specific physical qubits
    :type qubit_indices_physical: List[int], optional
    :param add_barriers: Whether to add quantum barriers between layers (Qiskit only)
    :type add_barriers: bool
    :param initial_state: Initial quantum state ('|+>', '|0>', or AbstractCircuit)
    :type initial_state: str or AbstractCircuit, optional
    :param number_of_qubits_circuit: Override for total circuit qubits
    :type number_of_qubits_circuit: int, optional

    Example:
        >>> from quapopt.hamiltonians import ClassicalHamiltonian
        >>> from quapopt.circuits.gates.logical import LogicalGateBuilderQiskit
        >>> # Create MaxCut Hamiltonian
        >>> ham = ClassicalHamiltonian([(1.0, (0, 1)), (1.0, (1, 2))], number_of_qubits=3)
        >>> gate_builder = LogicalGateBuilderQiskit()
        >>> # Build standard QAOA circuit
        >>> circuit = FullyConnectedQAOACircuit(
        ...     sdk_name='qiskit',
        ...     depth=2,
        ...     hamiltonian_phase=ham,
        ...     program_gate_builder=gate_builder
        ... )
        >>> # Build fractional time-blocked circuit
        >>> fractional_circuit = FullyConnectedQAOACircuit(
        ...     sdk_name='qiskit',
        ...     depth=1,
        ...     hamiltonian_phase=ham,
        ...     program_gate_builder=gate_builder,
        ...     time_block_size=0.5
        ... )

    .. note::
        For fully-connected QAOA, `time_block_size` represents the fraction of
        Hamiltonian interactions included per layer, distinct from LinearSwapNetwork
        where it specifies the number of linear chains.

    .. note::
        When `time_block_size < 1.0`, the circuit automatically switches to fractional
        time blocking mode, creating multiple sub-layers with subsets of interactions.
    """

    def __init__(
        self,
        sdk_name: str,
        depth: conint(ge=0),
        hamiltonian_phase: ClassicalHamiltonian,
        program_gate_builder: AbstractProgramGateBuilder,
        time_block_size: Optional[float] = None,
        phase_separator_type=PhaseSeparatorType.QAOA,
        mixer_type=MixerType.QAOA,
        every_gate_has_its_own_parameter: bool = False,
        qubit_indices_physical=None,
        add_barriers=False,
        initial_state: Optional[str] = None,
        number_of_qubits_circuit: int = None,
    ):
        """

        :param sdk_name:
        :param depth:
        :param hamiltonian_phase:
        :param program_gate_builder:
        :param time_block_size:
        NOTE: for fully-connected QAOA, time_block_size is the number of interactions in a layer.
        Note that this is disticnt from LinearSwapNetwork implementation, where time_block_size specifies
        number of LINEAR CHAINS in the layer.
        For fully connected topology, we can allow more freedom, hence this parametrization

        :param phase_separator_type:
        :param mixer_type:
        :param every_gate_has_its_own_parameter:
        """

        assert sdk_name.lower() in _SUPPORTED_SDKs, (
            f"Unsupported SDK: {sdk_name}. "
            f"Please choose one of the following: {_SUPPORTED_SDKs}"
        )
        if initial_state is None:
            initial_state = "|+>"

        ansatz_specifier = AnsatzSpecifier(
            PhaseHamiltonianClass=hamiltonian_phase.hamiltonian_class_specifier,
            PhaseHamiltonianInstance=hamiltonian_phase.hamiltonian_instance_specifier,
            Depth=depth,
            PhaseSeparatorType=phase_separator_type,
            MixerType=mixer_type,
            QubitMappingType=QubitMappingType.fully_connected,
            TimeBlockSize=time_block_size,
        )

        number_of_qubits = hamiltonian_phase.number_of_qubits

        qubit_ids_device = qubit_indices_physical

        if qubit_ids_device is None:
            qubit_ids_device = tuple(range(number_of_qubits))

        logical_qubit_indices = tuple(range(number_of_qubits))

        hamiltonian_abstract_interaction_edges_all = [
            tup for tup in hamiltonian_phase.hamiltonian if len(tup[1]) == 2
        ]
        if time_block_size is None:
            time_block_size = 1.0

        assert not every_gate_has_its_own_parameter, (
            "every_gate_has_its_own_parameter=True "
            "is not supported for FullyConnectedQAOACircuit yet"
        )

        param_name_phase = "AngPS"
        param_name_mixer = "AngMIX"

        if sdk_name.lower() in ["qiskit"]:
            if qiskit is None:
                raise ModuleNotFoundError(
                    "Qiskit is not installed. Please install Qiskit to use this feature."
                )

            if number_of_qubits_circuit is None:
                number_of_qubits_circuit = max(
                    [max(qubit_ids_device) + 1, number_of_qubits]
                )

            quantum_circuit = qiskit.QuantumCircuit(
                number_of_qubits_circuit, number_of_qubits
            )

            if not every_gate_has_its_own_parameter:
                angle_phase = (
                    qiskit.circuit.ParameterVector(name=param_name_phase, length=depth)
                    if depth > 0
                    else None
                )
                angle_mixer = (
                    qiskit.circuit.ParameterVector(name=param_name_mixer, length=depth)
                    if depth > 0
                    else None
                )

        elif sdk_name.lower() in ["pyquil"]:
            if pyquil is None:
                raise ModuleNotFoundError(
                    "PyQuil is not installed. Please install PyQuil to use this feature."
                )
            quantum_circuit = pyquil.Program()
            angle_phase = (
                quantum_circuit.declare(param_name_phase, "REAL", depth)
                if depth > 0
                else None
            )
            angle_mixer = (
                quantum_circuit.declare(param_name_mixer, "REAL", depth)
                if depth > 0
                else None
            )

        elif sdk_name.lower() in ["cirq"]:
            if cirq is None:
                raise ModuleNotFoundError(
                    "Cirq is not installed. Please install Cirq to use this feature."
                )

            import sympy

            quantum_circuit = cirq.Circuit()
            angle_phase = [
                sympy.Symbol(name=f"{param_name_phase}-{i}") for i in range(depth)
            ]
            angle_mixer = [
                sympy.Symbol(name=f"{param_name_mixer}-{i}") for i in range(depth)
            ]

        else:
            raise ValueError(
                (
                    f"Unsupported SDK: {sdk_name}. "
                    f"Please choose one of the following: {_SUPPORTED_SDKs}"
                )
            )

        # TODO(FBM): add support for general input states
        # We start from |+>^n
        if initial_state == "|+>":
            quantum_circuit = program_gate_builder.H(
                quantum_circuit=quantum_circuit, qubits_tuple=qubit_ids_device
            )
        elif initial_state == "|0>":
            pass

        elif isinstance(initial_state, AbstractCircuit):
            quantum_circuit = program_gate_builder.combine_circuits(
                left_circuit=quantum_circuit, right_circuit=initial_state
            )

        else:
            raise ValueError(
                f"Unsupported input state: {initial_state} of type: {type(quantum_circuit)}. Supported: |+>, |0>, {AbstractCircuit}"
            )

        if time_block_size == 1.0:
            hamiltonian_phase_dict = hamiltonian_phase.get_hamiltonian_dictionary()

            for layer_index in range(depth):
                if not every_gate_has_its_own_parameter:
                    beta = angle_mixer[layer_index] if layer_index < depth else 0.0
                    gamma = angle_phase[layer_index] if layer_index < depth else 0.0

                # We implement a layer of single-qubit gates if present in the Hamiltonian
                single_qubit_indices_abstract_PS = [
                    xi
                    for xi in logical_qubit_indices
                    if (xi,) in hamiltonian_phase_dict
                ]

                # (1) phase for all one-qubit interactions
                if len(single_qubit_indices_abstract_PS) > 0:
                    single_qubit_coefficients = [
                        hamiltonian_phase_dict[(i,)]
                        for i in single_qubit_indices_abstract_PS
                    ]
                    single_qubit_indices_PS_device = [
                        qubit_ids_device[i] for i in single_qubit_indices_abstract_PS
                    ]

                    # In given layer, we add the phase separator for single qubit interactions at the end of the layer
                    # TODO(FBM): in theory it doesn't matter if it's at the beginning or the end, in practice it might.
                    quantum_circuit = program_gate_builder.exp_Z(
                        quantum_circuit=quantum_circuit,
                        angles_tuple=tuple(
                            [coeff * gamma for coeff in single_qubit_coefficients]
                        ),
                        qubits_tuple=single_qubit_indices_PS_device,
                    )

                if every_gate_has_its_own_parameter:
                    pass

                current_coefficients = [
                    tup[0] for tup in hamiltonian_abstract_interaction_edges_all
                ]
                current_edges_logical = [
                    tup[1] for tup in hamiltonian_abstract_interaction_edges_all
                ]
                current_edges_device = [
                    (qubit_ids_device[qi], qubit_ids_device[qj])
                    for qi, qj in current_edges_logical
                ]

                if len(current_edges_device) != len(current_coefficients):
                    raise ValueError(
                        f"Number of edges and coefficients don't match: {len(current_edges_device)} vs {len(current_coefficients)}"
                    )

                for coeff, edge_device in zip(
                    current_coefficients, current_edges_device
                ):
                    if phase_separator_type in [PhaseSeparatorType.QAOA]:
                        if not every_gate_has_its_own_parameter:
                            quantum_circuit = program_gate_builder.exp_ZZ(
                                quantum_circuit=quantum_circuit,
                                angles_tuple=(gamma * coeff,),
                                qubits_pairs_tuple=[edge_device],
                            )

                    elif phase_separator_type in [PhaseSeparatorType.QAMPA]:
                        if not every_gate_has_its_own_parameter:
                            quantum_circuit = program_gate_builder.exp_ZZXXYY(
                                quantum_circuit=quantum_circuit,
                                angles_tuple=((gamma * coeff, beta)),
                                qubits_pairs_tuple=(edge_device,),
                            )

                    else:
                        raise ValueError(
                            f"Unsupported Phase Separator Type: {phase_separator_type}"
                        )

                if mixer_type in [MixerType.QAOA]:
                    # (3) one-qubit mixing operators
                    # In given layer, we add the phase separator for single qubit interactions at the end of the layer
                    if not every_gate_has_its_own_parameter:
                        quantum_circuit = program_gate_builder.exp_X(
                            quantum_circuit=quantum_circuit,
                            angles_tuple=beta,
                            qubits_tuple=qubit_ids_device,
                        )
                    else:
                        quantum_circuit = program_gate_builder.exp_X(
                            quantum_circuit=quantum_circuit,
                            angles_tuple=tuple([b for b in beta]),
                            qubits_tuple=qubit_ids_device,
                        )
                elif mixer_type in [MixerType.QAMPA]:
                    pass
                else:
                    raise ValueError(f"Unsupported Mixer Type: {mixer_type}")

                if add_barriers:
                    if sdk_name != "qiskit":
                        raise ValueError("Barriers are only supported for Qiskit SDK")
                    quantum_circuit.barrier(qubit_ids_device)

        elif time_block_size < 1.0:
            ansatz_builder_callable = partial(
                FullyConnectedQAOACircuit,
                sdk_name=sdk_name,
                program_gate_builder=program_gate_builder,
                phase_separator_type=phase_separator_type,
                mixer_type=mixer_type,
                every_gate_has_its_own_parameter=every_gate_has_its_own_parameter,
                number_of_qubits_circuit=number_of_qubits_circuit,
            )

            if sdk_name.lower() == "qiskit":
                quantum_circuit, (angle_phase, angle_mixer) = (
                    build_fractional_time_block_ansatz_qiskit(
                        hamiltonian_phase=hamiltonian_phase,
                        depth=depth,
                        time_block_size=time_block_size,
                        ansatz_builder_callable=ansatz_builder_callable,
                        ansatz_builder_kwargs={
                            "qubit_indices_physical": qubit_indices_physical
                        },
                        initial_state=quantum_circuit,
                        add_barriers=add_barriers,
                        parameter_names=(param_name_phase, param_name_mixer),
                    )
                )
            else:
                raise NotImplementedError(
                    "Fractional time block size is only implemented for Qiskit SDK"
                )
        else:
            raise ValueError(
                "time_block_size must be in (0, 1] for circuit not based on SWAP networks"
            )

        super().__init__(
            quantum_circuit=quantum_circuit,
            logical_to_physical_qubits_map=qubit_ids_device,
            parameters=[angle_phase, angle_mixer],
            qubit_mapping_permutation=None,
            ansatz_specifier=ansatz_specifier,
        )

        self._depth = depth
        self._phase_separator_type = phase_separator_type
        self._mixer_type = mixer_type
        self._time_block_size = time_block_size
        self._gate_builder = program_gate_builder

    @property
    def depth(self):
        """
        Get the QAOA circuit depth (number of layers).

        :returns: Number of QAOA layers (p parameter)
        :rtype: int
        """
        return self._depth

    @property
    def phase_separator_type(self):
        """
        Get the type of phase separator gates used in the circuit.

        Phase separators implement the cost Hamiltonian evolution in each QAOA layer.
        Standard QAOA uses exp(-i*gamma*C) gates, while QAMPA uses more complex
        parametrized gates.

        :returns: Phase separator gate type configuration
        :rtype: PhaseSeparatorType
        """
        return self._phase_separator_type

    @property
    def mixer_type(self):
        """
        Get the type of mixer gates used in the circuit.

        Mixers implement the driver Hamiltonian evolution that explores the solution
        space. Standard QAOA uses X-rotation gates, while QAMPA uses alternative
        mixer strategies.

        :returns: Mixer gate type configuration
        :rtype: MixerType
        """
        return self._mixer_type

    @property
    def time_block_size(self):
        """
        Get the time block size for fractional QAOA layers.

        For fully-connected QAOA, this represents the fraction of Hamiltonian
        interactions included in each layer. When < 1.0, the circuit uses
        fractional time blocking to split interactions across sub-layers.

        For time_block_size = 0.5 with n interactions, each layer contains
        approximately n*0.5 interactions.

        :returns: Fraction of Hamiltonian interactions per layer (0.0-1.0)
        :rtype: float
        """
        return self._time_block_size

    @property
    def gate_builder(self):
        """
        Get the gate builder used for SDK-specific gate implementations.

        The gate builder provides an abstraction layer for different quantum SDKs,
        allowing the same circuit logic to generate Qiskit, PyQuil, or Cirq circuits.

        :returns: Gate builder instance for this circuit
        :rtype: AbstractProgramGateBuilder
        """
        return self._gate_builder
