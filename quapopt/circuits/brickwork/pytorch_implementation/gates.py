# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


"""
Quantum gate implementations for Pauli Correlators Encoding circuits.
"""

import numpy as np
import torch
from typing import Optional

from quapopt.circuits.brickwork.gate_types import GateDescriptor, GateType1q, GateType2q


# Constants for gate implementations
_SQRT2 = 1 / np.sqrt(2)
_SQRT2I = -1j * _SQRT2


def get_precomputed_gates(device: torch.device) -> dict:
    """Get precomputed gates that don't require parameters."""
    _H_gate = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=device) / np.sqrt(2)
    _Sdag_gate = torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex64, device=device)
    _SH_gate = _H_gate @ _Sdag_gate
    
    return {
        'H': _H_gate, #changes measurement basis to |+⟩, |-⟩ if applied to state
        'SdagH': _SH_gate # changes measurement basis to |+i⟩, |-i⟩ if applied to state
    }


def get_1q_unitary(angles: torch.Tensor,
                   gate_descriptor: GateDescriptor,
                   device: torch.device,
                   precomputed_gates: Optional[dict] = None) -> torch.Tensor:
    """
    Returns a 1-qubit unitary gate based on angles and gate descriptor.
    
    Args:
        angles: Parameter tensor for the gate
        gate_descriptor: Descriptor specifying gate type and properties
        device: PyTorch device for tensor operations
        precomputed_gates: Optional dict of precomputed parameter-free gates
    
    Returns:
        Unitary tensor representing the gate
    """
    if precomputed_gates is None:
        precomputed_gates = get_precomputed_gates(device)
    
    if gate_descriptor.diagonal:
        gate_1q = torch.zeros(2, dtype=torch.complex64, device=device)
    else:
        gate_1q = torch.zeros(2, 2, dtype=torch.complex64, device=device)

    if gate_descriptor.gate_type == GateType1q.RZ:
        gate_1q[0] = 1.
        gate_1q[1] = torch.exp(1j * angles[0])

    elif gate_descriptor.gate_type == GateType1q.RX:
        el_diagonal = torch.cos(angles[0] / 2)
        el_01 = -1j * torch.sin(angles[0] / 2)
        gate_1q[0, 0], gate_1q[0, 1] = el_diagonal, el_01
        gate_1q[1, 0], gate_1q[1, 1] = el_01, el_diagonal

    elif gate_descriptor.gate_type == GateType1q.RY:
        el_diagonal = torch.cos(angles[0] / 2)
        el_01 = torch.sin(angles[0] / 2)
        gate_1q[0, 0], gate_1q[0, 1] = el_diagonal, -el_01
        gate_1q[1, 0], gate_1q[1, 1] = el_01, el_diagonal

    elif gate_descriptor.gate_type == GateType1q.GPI:
        # Native gate in IonQ: https://docs.ionq.com/guides/getting-started-with-native-gates#gpi
        gate_1q[0, 1], gate_1q[1, 0] = torch.exp(-1j * angles[0]), torch.exp(1j * angles[0])
        
    elif gate_descriptor.gate_type == GateType1q.GPI2:
        # Native gate in IonQ: https://docs.ionq.com/guides/getting-started-with-native-gates#gpi2
        el_diagonal = _SQRT2
        el_01 = _SQRT2I * torch.exp(-1j * angles[0])
        el_10 = _SQRT2I * torch.exp(1j * angles[0])
        gate_1q[0, 1], gate_1q[1, 0] = el_01, el_10
        gate_1q[0, 0], gate_1q[1, 1] = el_diagonal, el_diagonal

    elif gate_descriptor.gate_type == GateType1q.U1Q_QUANTINUUM:
        # Native gate in Quantinuum
        el_diagonal = torch.cos(angles[0] / 2)
        sine_i = -1j * torch.sin(angles[0] / 2)
        el_01 = sine_i * torch.exp(-1j * angles[1])
        el_10 = sine_i * torch.exp(1j * angles[1])
        gate_1q[0, 0], gate_1q[0, 1] = el_diagonal, el_01
        gate_1q[1, 0], gate_1q[1, 1] = el_10, el_diagonal

    elif gate_descriptor.gate_type == GateType1q.H:
        return precomputed_gates['H']
        
    elif gate_descriptor.gate_type == GateType1q.SdagH:
        return precomputed_gates['SdagH']

    else:
        raise ValueError(f"Unsupported gate_type: {gate_descriptor.gate_type}")

    return gate_1q


def get_2q_unitary(angles: torch.Tensor,
                   gate_descriptor: GateDescriptor,
                   device: torch.device) -> torch.Tensor:
    """
    Returns a 2-qubit unitary gate based on angles and gate descriptor.
    
    Args:
        angles: Parameter tensor for the gate
        gate_descriptor: Descriptor specifying gate type and properties
        device: PyTorch device for tensor operations
    
    Returns:
        Unitary tensor representing the 2-qubit gate
    """
    if gate_descriptor.diagonal:
        gate_2q = torch.zeros(4, dtype=torch.complex64, device=device)
    else:
        gate_2q = torch.zeros(4, 4, dtype=torch.complex64, device=device)

    if gate_descriptor.gate_type == GateType2q.RZZ:
        el_diagonal = torch.exp(1j * angles[0])
        gate_2q[0] = 1.
        gate_2q[1] = el_diagonal
        gate_2q[2] = el_diagonal
        gate_2q[3] = 1.

    elif gate_descriptor.gate_type == GateType2q.MS_PE:
        # Mølmer-Sørensen gate with phase encoding (IonQ native)
        # https://docs.ionq.com/guides/getting-started-with-native-gates#ms-gates
        el_diagonal = torch.cos(angles[0])
        sine_i = -1j * torch.sin(angles[0])
        sum_phi = angles[1] + angles[2]
        diff_phi = angles[1] - angles[2]

        el_03 = sine_i * torch.exp(-1j * sum_phi)
        el_12 = sine_i * torch.exp(-1j * diff_phi)
        el_21 = sine_i * torch.exp(1j * diff_phi)
        el_30 = sine_i * torch.exp(1j * sum_phi)

        gate_2q[0, 0], gate_2q[1, 1], gate_2q[2, 2], gate_2q[3, 3] = el_diagonal, el_diagonal, el_diagonal, el_diagonal
        gate_2q[0, 3], gate_2q[1, 2], gate_2q[2, 1], gate_2q[3, 0] = el_03, el_12, el_21, el_30

    else:
        raise ValueError(f"Unsupported gate_type: {gate_descriptor.gate_type}")

    return gate_2q