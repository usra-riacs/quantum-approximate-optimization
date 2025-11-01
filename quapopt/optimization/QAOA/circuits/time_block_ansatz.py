# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without auth
#
# ors' permission is strictly prohibited.

from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from typing import Optional, Tuple, Callable, Dict
import numpy as np

from quapopt.circuits.gates import AbstractCircuit, AbstractAngle




def _divide_hamiltonian_into_batches_fractional(hamiltonian:ClassicalHamiltonian,
                                                time_block_size:Optional[float],
                                                seed:int=0):

    if time_block_size == 1.0 or time_block_size is None:
        return {0:hamiltonian}

    rng = np.random.default_rng(seed=seed)
    shuffled_hamiltonian = hamiltonian.hamiltonian.copy()
    rng.shuffle(shuffled_hamiltonian)

    number_of_terms = len(shuffled_hamiltonian)
    number_of_batches = int(np.ceil(1 / time_block_size))
    number_of_terms_per_batch = int(np.ceil(number_of_terms / number_of_batches))

    hamiltonian_batches = {}
    for batch_index in range(number_of_batches):
        ham_i = shuffled_hamiltonian[batch_index*number_of_terms_per_batch:(batch_index+1)*number_of_terms_per_batch]
        hamiltonian_batches[batch_index] = ClassicalHamiltonian(hamiltonian_list_representation=ham_i,
                                                                number_of_qubits=hamiltonian.number_of_qubits)
    return hamiltonian_batches

def _divide_hamiltonian_into_batches_swap_network(hamiltonian:ClassicalHamiltonian,
                                                time_block_size:Optional[int]):
    """
    Imitate interactions batching that would correspond to an optimal linear SWAP network implementation.

    :param hamiltonian:
    :param time_block_size:
    number of linear chains treated as a single layer ("time block")
    each linear chain implements at most number_of_qubits interactions


    :return:
    """

    if time_block_size == hamiltonian.number_of_qubits or time_block_size is None:
        return {0:hamiltonian}

    #TODO(FBM): implement this.

    raise NotImplementedError("Swap network batching not implemented yet.")

def divide_hamiltonian_into_batches(hamiltonian:ClassicalHamiltonian,
                                   time_block_size:Optional[int|float],
                                    batching_type:str='fractional',
                                   seed:int=0)->Dict[int,ClassicalHamiltonian]:

    if batching_type.lower() in ['fractional']:
        return _divide_hamiltonian_into_batches_fractional(hamiltonian=hamiltonian,
                                                         time_block_size=time_block_size,
                                                         seed=seed)
    elif batching_type.lower() in ['swap_network']:
        return _divide_hamiltonian_into_batches_swap_network(hamiltonian=hamiltonian,
                                                             time_block_size=time_block_size)
    else:
        raise ValueError(f"Batching type {batching_type} not recognised. Choose from 'fractional' or 'swap_network'.")




def build_fractional_time_block_ansatz_qiskit(hamiltonian_phase: ClassicalHamiltonian,
                                              depth: int,
                                              time_block_size: float,
                                              ansatz_builder_callable: Callable,
                                              ansatz_builder_kwargs: Optional[dict] = None,
                                              initial_state: str | AbstractCircuit = '|+>',
                                              add_barriers: bool = False,
                                              parameter_names: Tuple[str, str] = ("AngPS", "AngMIX"),
                                              shuffling_seed=0,
                                              time_block_partition: Optional[Dict[int, ClassicalHamiltonian]] = None,
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


    if time_block_partition is None:
        time_block_partition = divide_hamiltonian_into_batches(hamiltonian=hamiltonian_phase,
                                                                time_block_size=time_block_size,
                                                                seed=shuffling_seed,
                                                               batching_type='fractional')


    # Calculate batching parameters
    number_of_batches = len(time_block_partition)

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

        hamiltonian_batch = time_block_partition[batch_index]

        if len(hamiltonian_batch.hamiltonian) == 0:
            continue



        # Build single-layer ansatz for this batch
        batch_kwargs = ansatz_builder_kwargs.copy()
        batch_ansatz = ansatz_builder_callable(hamiltonian_phase=hamiltonian_batch,
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
        r'-(\d+)$',  # AngPHS-0, AngMIX-1
        r'_(\d+)$',  # AngPHS_0, AngMIX_1
    ]

    for pattern in patterns:
        match = re.search(pattern, param_name)
        if match:
            return int(match.group(1))

    return None

