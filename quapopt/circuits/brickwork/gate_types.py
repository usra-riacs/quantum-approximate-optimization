# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


"""
Gate types and descriptors for Pauli Correlators Encoding quantum circuits.
"""

from enum import Enum
from typing import Dict

# Gate parameter counts
_GATE_PARAMS_COUNTS_1Q: Dict[str, int] = {
    "rz": 1,
    "rx": 1,
    "ry": 1,
    "gpi": 1,
    "gpi2": 1,
    "u1q_quantinuum": 2,
    "h": 0,
    "sdagh": 0,
}

_GATE_PARAMS_COUNTS_2Q: Dict[str, int] = {"rzz": 1, "ms_pe": 3}

# Diagonal gates (can be applied efficiently using broadcasting)
_DIAGONAL_GATES = {"rz", "rzz"}


class GateType1q(Enum):
    """Single-qubit gate types supported in PCE circuits."""

    RZ = "rz"
    RX = "rx"
    RY = "ry"
    GPI = "gpi"
    GPI2 = "gpi2"
    U1Q_QUANTINUUM = "u1q_quantinuum"
    H = "H"
    SdagH = "SdagH"  # Custom gate combining S† and H for Y-basis measurement


class GateType2q(Enum):
    """Two-qubit gate types supported in PCE circuits."""

    RZZ = "rzz"
    MS_PE = "MS_pe"  # Mølmer-Sørensen gate with phase encoding


class GateDescriptor:
    """
    Descriptor for a quantum gate containing its type and properties.

    Attributes:
        gate_type: The gate type (1-qubit or 2-qubit)
        number_of_qubits: Number of qubits the gate acts on
        param_count: Number of parameters required for the gate
        diagonal: Whether the gate is diagonal in computational basis
    """

    def __init__(self, gate_type: GateType1q | GateType2q):
        self.gate_type: GateType1q | GateType2q = gate_type
        self.diagonal = self.gate_type.value.lower() in _DIAGONAL_GATES

        if isinstance(gate_type, GateType1q):
            self.number_of_qubits = 1
            self.param_count = _GATE_PARAMS_COUNTS_1Q[self.gate_type.value.lower()]
        elif isinstance(gate_type, GateType2q):
            self.number_of_qubits = 2
            self.param_count = _GATE_PARAMS_COUNTS_2Q[self.gate_type.value.lower()]
        else:
            raise ValueError(
                "gate_type must be an instance of GateType1q or GateType2q enum."
            )
