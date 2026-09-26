# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import copy
from typing import List, Tuple, Optional, Dict

from quapopt.circuits.gates import AbstractCircuit, AbstractAngle
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType, InitialStateType
from quapopt.circuits.gates import AbstractProgramGateBuilder


class MappedAnsatzCircuit:
    """Build a parameterized quantum approximate optimization circuit program for a device."""

    def __init__(
            self,
            quantum_circuit: AbstractCircuit,
            parameters: List[AbstractAngle],
            logical_to_physical_qubits_map: Tuple[int, ...],
            qubit_mapping_permutation: Optional[Tuple[int, ...]],
            ansatz_specifier: Optional[AnsatzSpecifier] = None,

    ):
        parameters_formatted = []
        for param in parameters_formatted:
            parameters_formatted.append(param)

        self._quantum_circuit = quantum_circuit
        self._parameters = parameters

        self._logical_to_physical_qubits_map = logical_to_physical_qubits_map
        self._physical_to_logical_qubits_map = {device_qubit: qubit for qubit, device_qubit in
                                                enumerate(logical_to_physical_qubits_map)}
        self._qubit_mapping_permutation = qubit_mapping_permutation

        self.AnsatzSpecifier = ansatz_specifier

    @property
    def quantum_circuit(self) -> AbstractCircuit:
        return self._quantum_circuit

    @quantum_circuit.setter
    def quantum_circuit(self, value: AbstractCircuit):
        if not isinstance(value, AbstractCircuit):
            raise TypeError("quantum_circuit must be an instance of AbstractCircuit.")
        # print("WARNING:", "Setting quantum_circuit directly is not recommended. Use the constructor instead.")

        self._quantum_circuit = value

    @property
    def parameters(self) -> List[AbstractAngle]:
        return self._parameters

    @property
    def qubit_mapping_permutation(self):
        return self._qubit_mapping_permutation

    @property
    def logical_to_physical_qubits_map(self) -> Tuple[int, ...]:
        return self._logical_to_physical_qubits_map

    @property
    def physical_to_logical_qubits_map(self) -> Dict[int, int]:
        return self._physical_to_logical_qubits_map

    def copy(self):
        return copy.deepcopy(self)



class MappedQAOACircuit(MappedAnsatzCircuit):


    def __init__(
            self,
            quantum_circuit: AbstractCircuit,
            parameters: List[AbstractAngle],
            logical_to_physical_qubits_map: Tuple[int, ...],
            qubit_mapping_permutation: Optional[Tuple[int, ...]],
            depth:int,
            program_gate_builder: AbstractProgramGateBuilder,
            phase_separator_type=PhaseSeparatorType.QAOA,
            mixer_type=MixerType.QAOA,
            initial_state: InitialStateType = InitialStateType.QAOA,
            time_block_size: Optional[float|int] = None,
            ansatz_specifier: Optional[AnsatzSpecifier] = None,



        ):

        super().__init__(
            quantum_circuit=quantum_circuit,
            parameters=parameters,
            logical_to_physical_qubits_map=logical_to_physical_qubits_map,
            qubit_mapping_permutation=qubit_mapping_permutation,
            ansatz_specifier=ansatz_specifier,
        )


        self._depth = depth
        self._phase_separator_type = phase_separator_type
        self._mixer_type = mixer_type
        self._initial_state = initial_state

        self._time_block_size = time_block_size
        self._gate_builder = program_gate_builder

    @property
    def depth(self)->int:
        """
        Get the QAOA circuit depth (number of pairs of parametrized layers).

        :returns: Number of QAOA layers (p parameter)
        :rtype: int
        """
        return self._depth

    @property
    def phase_separator_type(self)->PhaseSeparatorType:
        """
        Phase separators implement the cost Hamiltonian evolution in each QAOA layer.
        Standard QAOA uses exp(-i*gamma*C) gates.

        :returns: Phase separator gate type configuration
        :rtype: PhaseSeparatorType
        """
        return self._phase_separator_type

    @property
    def mixer_type(self)->MixerType:
        """
        Mixers implement the driver Hamiltonian evolution that explores the solution
        space. Standard QAOA uses X-rotation gates.

        :returns: Mixer gate type configuration
        :rtype: MixerType
        """
        return self._mixer_type

    @property
    def initial_state(self)->InitialStateType:
        """
        Get the initial state used in the circuit.

        :return:
        """


        return self._initial_state

    @property
    def time_block_size(self)->Optional[int|float]:
        """
        For non-SWAP-network implementation, this represents the fraction of Hamiltonian
        interactions included in each layer. When < 1.0, the circuit uses
        fractional time blocking to split interactions across sub-layers.

        For time_block_size = 0.5 with n interactions, each layer contains
        approximately n*0.5 interactions.

        For SWAP-network implementation, this represents the number of SWAP layers parametrized as a single layer.
        For example, time_block_size = 5 means that 5 SWAP layers of interactions are parametrized as a single layer.

        None means that the circuit uses full Hamiltonian interactions per layer (standard QAOA).


        :returns: Fraction of Hamiltonian interactions per layer (0.0-1.0); or number of SWAP layers for SWAP-network implementation
        :rtype: float|int
        """
        return self._time_block_size

    @property
    def gate_builder(self)->AbstractProgramGateBuilder:
        """
        Get the gate builder used for SDK-specific gate implementations.

        The gate builder provides an abstraction layer for different quantum SDKs,
        allowing the same circuit logic to generate Qiskit, PyQuil, or Cirq circuits.

        :returns: Gate builder instance for this circuit
        :rtype: AbstractProgramGateBuilder
        """
        return self._gate_builder
