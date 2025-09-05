# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import copy
from typing import List, Tuple, Optional, Dict, Callable
from pydantic import conint
from quapopt import ancillary_functions as anf

from quapopt.circuits.gates import AbstractCircuit, AbstractAngle
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian



from quapopt.optimization.QAOA import AnsatzSpecifier
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.circuits.gates import _SUPPORTED_SDKs
import copy
import numpy as np
class MappedAnsatzCircuit:
    """Build a parameterized quantum approximate optimization circuit program for a device."""

    def __init__(
            self,
            quantum_circuit: AbstractCircuit,
            parameters: List[AbstractAngle],
            logical_to_physical_qubits_map: Tuple[int, ...],
            qubit_mapping_permutation:Optional[Tuple[int,...]],
            ansatz_specifier: Optional[AnsatzSpecifier] = None,

    ):
        parameters_formatted = []
        for param in parameters_formatted:
            parameters_formatted.append(param)

        self._quantum_circuit = quantum_circuit
        self._parameters = parameters

        self._logical_to_physical_qubits_map = logical_to_physical_qubits_map
        self._physical_to_logical_qubits_map = {device_qubit: qubit for qubit, device_qubit in enumerate(logical_to_physical_qubits_map)}
        self._qubit_mapping_permutation = qubit_mapping_permutation

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
    def parameters(self) -> List[AbstractAngle]:
        return self._parameters

    @property
    def qubit_mapping_permutation(self):
        return self._qubit_mapping_permutation

    @property
    def logical_to_physical_qubits_map(self) -> Tuple[int, ...]:
        return self._logical_to_physical_qubits_map

    @property
    def physical_to_logical_qubits_map(self) -> Dict[int,int]:
        return self._physical_to_logical_qubits_map


    def copy(self):
        return copy.deepcopy(self)





def build_fractional_time_block_ansatz_qiskit(hamiltonian_phase: ClassicalHamiltonian,
                                              depth: int,
                                              time_block_size: float,
                                              ansatz_builder_callable: Callable,
                                              ansatz_builder_kwargs: Optional[dict]=None,
                                              initial_state: str|AbstractCircuit = '|+>',
                                              add_barriers: bool = False,
                                              parameter_names: Tuple[str, str] = ("AngPS", "AngMIX"),
                                              shuffling_seed= 42
                                              ):
    """
    Build a fractional time-block QAOA ansatz using an abstract ansatz builder.
    
    This function splits the Hamiltonian into time blocks and sequentially applies
    the ansatz builder to each block, building up the circuit layer by layer.
    
    Args:
        hamiltonian_phase: The phase Hamiltonian to optimize
        depth: Number of QAOA layers  
        time_block_size: Fraction of interactions per time block (0 < time_block_size <= 1.0)
        ansatz_builder_callable: Function that builds ansatz circuits - should accept
                                hamiltonian_phase and other kwargs and return a quantum circuit
        ansatz_builder_kwargs: Additional keyword arguments for the ansatz builder
        initial_state: Initial quantum state ('|+>' or '|0>')
        add_barriers: Whether to add barriers between time blocks
        parameter_names: Tuple of (phase_param_name, mixer_param_name)
    
    Returns:
        Quantum circuit with fractional time-block structure
    """
    from qiskit import QuantumCircuit, ClassicalRegister
    from qiskit.circuit import ParameterVector

    assert 0 < time_block_size <= 1.0, f"time_block_size must be in (0, 1], got {time_block_size}"
    
    number_of_qubits = hamiltonian_phase.number_of_qubits
    param_name_phase, param_name_mixer = parameter_names

    param_name_phase_temp = 'TBPhase'
    param_name_mixer_temp = 'TBMixer'
    
    # Create parameter vectors for the full depth
    angle_phase = ParameterVector(name=param_name_phase_temp, length=depth) if depth > 0 else None
    angle_mixer = ParameterVector(name=param_name_mixer_temp, length=depth) if depth > 0 else None

    if ansatz_builder_kwargs is None:
        ansatz_builder_kwargs = {}

    if time_block_size == 1.0:
        # Standard case - build full ansatz with original Hamiltonian
        circuit = ansatz_builder_callable(
            hamiltonian_phase=hamiltonian_phase,
            depth=depth,
            add_barriers=add_barriers,
            initial_state=initial_state,
            **ansatz_builder_kwargs
        )
        

        return circuit
    
    # Fractional time blocking case
    hamiltonian_phase_list = hamiltonian_phase.hamiltonian.copy()

    rng_shuffling = np.random.default_rng(seed=shuffling_seed)
    rng_shuffling.shuffle(hamiltonian_phase_list)

    number_of_interactions = len(hamiltonian_phase_list)

    # Calculate batching parameters
    number_of_batches = int(np.ceil(1 / time_block_size))
    number_of_interactions_per_batch = int(np.ceil(number_of_interactions / number_of_batches))
    
    # Initialize circuit with initial state
    ansatz_init = ansatz_builder_callable(hamiltonian_phase=hamiltonian_phase,
                                            depth=0,  # no layers to just get initial state
                                            initial_state=initial_state,
                                            add_barriers=False,
                                            time_block_size=1.0,
                                            **ansatz_builder_kwargs)

    ansatz_circuit_qiskit = ansatz_init.quantum_circuit

    # Build fractional time blocks
    for param_index in range(depth):
        batch_index = param_index % number_of_batches
        
        # Extract Hamiltonian batch
        start_idx = batch_index * number_of_interactions_per_batch
        end_idx = (batch_index + 1) * number_of_interactions_per_batch
        hamiltonian_batch = hamiltonian_phase_list[start_idx:end_idx]
        
        if len(hamiltonian_batch) == 0:
            continue

        # Create sub-Hamiltonian for this batch
        hamiltonian_batch_obj = ClassicalHamiltonian(
            hamiltonian_list_representation=hamiltonian_batch,
            number_of_qubits=hamiltonian_phase.number_of_qubits,
            default_backend='numpy'

        )
        
        # Build single-layer ansatz for this batch
        batch_kwargs = ansatz_builder_kwargs.copy()
        batch_ansatz = ansatz_builder_callable(hamiltonian_phase=hamiltonian_batch_obj,
                                                depth=1,  # Single layer for each batch
                                                initial_state=ansatz_circuit_qiskit,
                                                add_barriers=False,
                                                time_block_size=1.0,
                                                **batch_kwargs)

        parameters_ansatz_batch = batch_ansatz.parameters
        phase_name_to_look_for = parameters_ansatz_batch[0].name
        mixer_name_to_look_for = parameters_ansatz_batch[1].name

        ansatz_circuit_qiskit = batch_ansatz.quantum_circuit
        if add_barriers:
            ansatz_circuit_qiskit.barrier()

        parameters_default = list(ansatz_circuit_qiskit.parameters)
        
        # Detect which parameter is phase and which is mixer based on name
        phase_param_obj = None
        mixer_param_obj = None
        for param in parameters_default:
            param_name = param.name
            if param_name_phase_temp in param_name or param_name_mixer_temp in param_name:
                continue
            if phase_name_to_look_for in param_name:
                phase_param_obj = param
            elif mixer_name_to_look_for in param_name:
                mixer_param_obj = param
        
        # Fallback to positional if name detection fails
        if phase_param_obj is None or mixer_param_obj is None:
            raise ValueError("Parameters not found!")
        
        # Directly assign parameters using the specific parameter objects
        params_dict = {}
        params_dict[phase_param_obj] = angle_phase[param_index]
        params_dict[mixer_param_obj] = angle_mixer[param_index]

        ansatz_circuit_qiskit.assign_parameters(params_dict, inplace=True)

    # Rename parameters to original names
    final_angle_phase = ParameterVector(name=param_name_phase, length=depth) if depth > 0 else None
    final_angle_mixer = ParameterVector(name=param_name_mixer, length=depth) if depth > 0 else None
    
    # Create mapping from temporary parameters to final parameters
    final_params_dict = {}
    for i in range(depth):
        if angle_phase is not None and final_angle_phase is not None:
            final_params_dict[angle_phase[i]] = final_angle_phase[i]
        if angle_mixer is not None and final_angle_mixer is not None:
            final_params_dict[angle_mixer[i]] = final_angle_mixer[i]
    
    # Apply final parameter renaming
    if final_params_dict:
        ansatz_circuit_qiskit.assign_parameters(final_params_dict, inplace=True)

    return ansatz_circuit_qiskit, (final_angle_phase, final_angle_mixer)

def _assign_batch_parameters(circuit,
                             gamma_param,
                             beta_param):
    """Assign parameters to a single batch circuit."""
    params_dict = {}
    for param in circuit.parameters:
        param_name = param.name
        if "β" in param_name or "AngMIX" in param_name or "mixer" in param_name.lower():
            if beta_param is not None:
                params_dict[param] = beta_param
        elif "γ" in param_name or "AngPS" in param_name or "phase" in param_name.lower():
            if gamma_param is not None:
                params_dict[param] = gamma_param
    
    if params_dict:
        circuit.assign_parameters(parameters=params_dict, inplace=True)
    
    return circuit


def _extract_parameter_index(param_name: str) -> Optional[int]:
    """Extract parameter index from parameter name."""
    import re
    
    # Try to extract index from patterns like β[0], γ[1], AngPHS-2, etc.
    patterns = [
        r'\[(\d+)\]',  # β[0], γ[1]
        r'-(\d+)$',    # AngPHS-0, AngMIX-1
        r'_(\d+)$',    # AngPHS_0, AngMIX_1
    ]
    
    for pattern in patterns:
        match = re.search(pattern, param_name)
        if match:
            return int(match.group(1))
    
    return None

