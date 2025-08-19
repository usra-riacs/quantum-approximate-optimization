# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 
import copy
from typing import List, Tuple, Optional, Dict
from pydantic import conint
from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.circuits.gates import AbstractCircuit, AbstractAngle
from quapopt.hamiltonians.representation import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.circuits.gates import _SUPPORTED_SDKs
import copy

class MappedAnsatzCircuit:
    """Build a parameterized quantum approximate optimization circuit program for a device."""

    def __init__(
            self,
            quantum_circuit: AbstractCircuit,
            mapped_hamiltonian: ClassicalHamiltonian,
            parameters: List[AbstractAngle],
            logical_to_physical_qubits_map: Tuple[int, ...],
            permutation_circuit_network=None,
            ansatz_specifier: Optional[AnsatzSpecifier] = None,

    ):
        parameters_formatted = []
        for param in parameters_formatted:
            parameters_formatted.append(param)

        self._quantum_circuit = quantum_circuit
        self._mapped_observable = mapped_hamiltonian
        self._parameters = parameters

        self._number_of_qubits = mapped_hamiltonian.number_of_qubits
        self._logical_to_physical_qubits_map = logical_to_physical_qubits_map
        self._physical_to_logical_qubits_map = {device_qubit: qubit for qubit, device_qubit in enumerate(logical_to_physical_qubits_map)}
        self._permutation_circuit_network = permutation_circuit_network

        self.AnsatzSpecifier = ansatz_specifier

    @property
    def quantum_circuit(self) -> AbstractCircuit:
        return self._quantum_circuit

    @quantum_circuit.setter
    def quantum_circuit(self, value: AbstractCircuit):
        if not isinstance(value, AbstractCircuit):
            raise TypeError("quantum_circuit must be an instance of AbstractCircuit.")
        #print("WARNING:", "Setting quantum_circuit directly is not recommended. Use the constructor instead.")

        self._quantum_circuit = value

    @property
    def mapped_observable(self) -> ClassicalHamiltonian:
        """Get the circuit's optimization cost function hamiltonian."""
        return self._mapped_observable

    @property
    def parameters(self) -> List[AbstractAngle]:
        return self._parameters

    @property
    def permutation_circuit_network(self):
        return self._permutation_circuit_network

    @property
    def logical_to_physical_qubits_map(self) -> Tuple[int, ...]:
        return self._logical_to_physical_qubits_map

    @property
    def physical_to_logical_qubits_map(self) -> Dict[int,int]:
        return self._physical_to_logical_qubits_map


    def copy(self):
        return copy.deepcopy(self)

