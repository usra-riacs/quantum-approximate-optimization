# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


"""
Circuit layer operations for Pauli Correlators Encoding ansatz.
"""

from typing import Optional, List, Dict, Tuple

import torch

from quapopt.circuits.brickwork.gate_types import GateDescriptor, GateType1q
from quapopt.circuits.brickwork.pytorch_implementation.gates import get_1q_unitary, get_2q_unitary, \
    get_precomputed_gates


def get_brickwork_2q_gate_count(number_of_qubits: int) -> Tuple[int, int]:
    """
    Computes the number of gates in a brickwork ansatz layer.

    Args:
        number_of_qubits: Number of qubits in the ansatz
    Returns:
        Tuple containing:
        (even_gates_count, odd_gates_count)
    """

    if number_of_qubits % 2 == 0:
        return number_of_qubits // 2, max(0, number_of_qubits // 2 - 1)
    else:
        return (number_of_qubits - 1) // 2, (number_of_qubits - 1) // 2


def apply_1q_layer(quantum_state: torch.Tensor,
                   angles: torch.Tensor,
                   gate_descriptors: GateDescriptor | List[GateDescriptor],
                   device: torch.device,
                   precomputed_gates: Optional[dict] = None) -> torch.Tensor:
    """
    Applies a layer of 1-qubit gates to the quantum state sequentially.
    
    Args:
        quantum_state: Quantum state tensor with shape (2, 2, ..., 2)
        angles: Parameter tensor for all gates in the layer
        gate_descriptors: Single GateDescriptor (applied to all qubits) or 
                         list of GateDescriptors (one per qubit)
        device: PyTorch device for tensor operations
        precomputed_gates: Optional dict of precomputed parameter-free gates
    
    Returns:
        Updated quantum state after applying the layer
    """
    if precomputed_gates is None:
        precomputed_gates = get_precomputed_gates(device)

    number_of_qubits = len(quantum_state.shape)

    # Handle both single descriptor and list of descriptors
    if isinstance(gate_descriptors, GateDescriptor):
        # Single descriptor: apply same gate to all qubits (backward compatibility)
        gate_descriptors_list = [gate_descriptors] * number_of_qubits
    else:
        # List of descriptors: one per qubit
        gate_descriptors_list = gate_descriptors
        if len(gate_descriptors_list) != number_of_qubits:
            raise ValueError(f"Number of gate descriptors ({len(gate_descriptors_list)}) "
                             f"must match number of qubits ({number_of_qubits})")

    # Create a view for diagonal gates
    _view_diagonal = [1] * number_of_qubits

    # Track parameter offset for varying gate types
    angle_offset = 0

    for qubit_index in range(number_of_qubits):
        gate_descriptor = gate_descriptors_list[qubit_index]
        if gate_descriptor is None:
            continue
        param_count = gate_descriptor.param_count

        # Extract parameters for this specific gate
        gate_angles = angles[angle_offset:angle_offset + param_count]

        gate_1q_i = get_1q_unitary(
            angles=gate_angles,
            gate_descriptor=gate_descriptor,
            device=device,
            precomputed_gates=precomputed_gates
        )

        # Exploit diagonal structure for computational efficiency
        if gate_descriptor.diagonal:
            view_diagonal_i = _view_diagonal.copy()
            view_diagonal_i[qubit_index] = 2
            quantum_state = quantum_state * gate_1q_i.view(view_diagonal_i)
        else:
            # Standard tensor contraction for non-diagonal gates
            quantum_state = torch.tensordot(gate_1q_i, quantum_state, dims=([1], [qubit_index]))
            quantum_state = torch.moveaxis(quantum_state, 0, qubit_index)

        # Advance parameter offset
        angle_offset += param_count

    return quantum_state


def apply_2q_layer(quantum_state: torch.Tensor,
                   angles: torch.Tensor,
                   gate_descriptor: GateDescriptor,
                   device: torch.device,
                   even_layer: bool) -> torch.Tensor:
    """
    Applies a layer of 2-qubit gates to the quantum state using brickwork architecture.
    
    Args:
        quantum_state: Quantum state tensor with shape (2, 2, ..., 2)
        angles: Parameter tensor for all gates in the layer
        gate_descriptor: Descriptor specifying gate type and properties
        device: PyTorch device for tensor operations
        even_layer: If True, starts from 0th qubit; if False, starts from 1st qubit
    
    Returns:
        Updated quantum state after applying the layer
    """
    number_of_qubits = len(quantum_state.shape)
    param_count = gate_descriptor.param_count

    # Create views for 2-qubit gates
    _view_diagonal_2q = [1] * number_of_qubits
    _view_2q = [2] * 4

    if even_layer:
        start_index = 0
    else:
        start_index = 1

    for ord_index, qubit_index in enumerate(range(start_index, number_of_qubits, 2)):
        qi, qj = qubit_index, qubit_index + 1

        if qj > number_of_qubits - 1:
            continue

        # Get the 2-qubit gate tensor with relevant angles
        gate_2q_i = get_2q_unitary(
            angles=angles[param_count * ord_index:param_count * (ord_index + 1)],
            gate_descriptor=gate_descriptor,
            device=device
        )

        # Exploit diagonal structure for computational efficiency
        if gate_descriptor.diagonal:
            view_diagonal_i = _view_diagonal_2q.copy()
            view_diagonal_i[qi] = 2
            view_diagonal_i[qj] = 2
            quantum_state = quantum_state * gate_2q_i.view(view_diagonal_i)
        else:
            # Standard tensor contraction for non-diagonal gates
            quantum_state = torch.tensordot(gate_2q_i.view(_view_2q), quantum_state, dims=([2, 3], [qi, qj]))
            quantum_state = torch.moveaxis(quantum_state, (0, 1), (qi, qj))

    return quantum_state


def apply_brickwork_ansatz(number_of_qubits: int,
                           angles_1q: torch.Tensor,
                           angles_2q: torch.Tensor,
                           gate_descriptors_1q: List[GateDescriptor] | GateDescriptor,
                           gate_descriptor_2q: GateDescriptor,
                           device: torch.device,
                           number_of_layers: int,
                           add_additional_rz_rotations: bool,
                           quantum_state: Optional[torch.Tensor] = None,
                           precomputed_gates: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """
    Applies a brickwork ansatz to the quantum state.
    
    A single layer consists of:
    1. Single-qubit gates: (U_1q)^⊗n
    2. [Optional] Additional RZ rotations: (RZ)^⊗n
    3. Two-qubit gates: (U_2q)^⊗K with brickwork connectivity
    
    Args:
        number_of_qubits: Number of qubits in the ansatz
        angles_1q: Parameters for all 1-qubit gates
        angles_2q: Parameters for all 2-qubit gates
        gate_descriptors_1q: Descriptor for 1-qubit gates
        gate_descriptor_2q: Descriptor for 2-qubit gates
        device: PyTorch device for tensor operations
        number_of_layers: Number of ansatz layers
        add_additional_rz_rotations: Whether to add RZ layer after 1-qubit gates
        quantum_state: Initial state (defaults to |0...0⟩)
    
    Returns:
        Final quantum state after applying the ansatz
    """
    if quantum_state is None:
        # Initialize to |0...0⟩ state
        quantum_state = torch.zeros([2] * number_of_qubits, dtype=torch.complex64, device=device)
        quantum_state[tuple([0] * number_of_qubits)] = 1.0

    if isinstance(gate_descriptors_1q, GateDescriptor):
        gate_descriptors_1q = [gate_descriptors_1q] * number_of_qubits

    # Parameter counting for layer structure
    angles_per_layer_1q_main = sum([gd.param_count for gd in gate_descriptors_1q])

    angles_per_layer_RZ = number_of_qubits if add_additional_rz_rotations else 0

    angles_per_layer_1q = angles_per_layer_1q_main + angles_per_layer_RZ

    # Precompute gates for efficiency
    if precomputed_gates is None:
        precomputed_gates = get_precomputed_gates(device)
    _gate_descriptor_RZ = GateDescriptor(gate_type=GateType1q.RZ)

    offset_2q = 0

    even_gates_count, odd_gates_count = get_brickwork_2q_gate_count(number_of_qubits=number_of_qubits)

    for layer_index in range(number_of_layers):
        # 1-qubit gates layer
        angles_1q_i = angles_1q[layer_index * angles_per_layer_1q:(layer_index + 1) * angles_per_layer_1q]

        quantum_state = apply_1q_layer(
            quantum_state=quantum_state,
            angles=angles_1q_i[0:angles_per_layer_1q_main],
            gate_descriptors=gate_descriptors_1q,
            device=device,
            precomputed_gates=precomputed_gates
        )
        if add_additional_rz_rotations:
            # Split angles: main 1q gates + additional RZ rotations
            quantum_state = apply_1q_layer(
                quantum_state=quantum_state,
                angles=angles_1q_i[angles_per_layer_1q_main:],
                gate_descriptors=_gate_descriptor_RZ,
                device=device,
                precomputed_gates=precomputed_gates
            )

        # 2-qubit gates layer with brickwork connectivity
        even_layer = (layer_index % 2 == 0)

        # Parameter counting for brickwork architecture
        if even_layer:
            number_of_angles_2q_i = gate_descriptor_2q.param_count * even_gates_count
        else:
            number_of_angles_2q_i = gate_descriptor_2q.param_count * odd_gates_count

        angles_2q_i = angles_2q[offset_2q:offset_2q + number_of_angles_2q_i]

        # Only apply 2q layer if there are gates to apply
        if number_of_angles_2q_i > 0:
            quantum_state = apply_2q_layer(
                quantum_state=quantum_state,
                angles=angles_2q_i,
                gate_descriptor=gate_descriptor_2q,
                device=device,
                even_layer=even_layer
            )

        offset_2q += number_of_angles_2q_i

    return quantum_state
