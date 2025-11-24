# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import copy
from typing import Dict, List, Optional, Tuple

from quapopt.circuits.gates import AbstractAngle, AbstractCircuit
from quapopt.optimization.QAOA import AnsatzSpecifier


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
        self._physical_to_logical_qubits_map = {
            device_qubit: qubit
            for qubit, device_qubit in enumerate(logical_to_physical_qubits_map)
        }
        self._qubit_mapping_permutation = qubit_mapping_permutation

        self.AnsatzSpecifier = ansatz_specifier

    @property
    def quantum_circuit(self) -> AbstractCircuit:
        return self._quantum_circuit

    @quantum_circuit.setter
    def quantum_circuit(self, value: AbstractCircuit):
        if not isinstance(value, AbstractCircuit):
            raise TypeError("quantum_circuit must be an instance of AbstractCircuit.")

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
