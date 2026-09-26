# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from typing import Optional, Tuple, Callable, Dict
import numpy as np

from quapopt.circuits.gates import AbstractCircuit, AbstractAngle
from quapopt.optimization.QAOA import InitialStateType
from quapopt.optimization.QAOA.circuits.swap_networks import (get_linear_chain_permutation,
                                                              get_swap_network_permutation,
                                                              get_hamiltonian_partition_equivalent_to_time_block_ansatz_with_linear_swap_network)

from enum import Enum

class TimeBlockBatchingType(Enum):
    FRACTIONAL = 'Fractional'
    SWAP_NETWORK = 'LinearSwapNetwork'


def _extract_circular_slice(arr: list, start_idx: int, length: int) -> list:
    """
    Extract a slice from list with circular wrapping, supporting length > len(arr).

    Args:
        arr: Source list (of Hamiltonian terms)
        start_idx: Starting index (will be wrapped if >= len(arr))
        length: Number of elements to extract (can exceed len(arr) for repetition)

    Returns:
        List of `length` elements extracted with circular wrapping
    """
    n = len(arr)
    if length <= 0:
        return []

    # Handle case where we need multiple full loops + partial
    full_loops = length // n
    remaining = length % n

    result = []

    # Add full copies of the list
    for _ in range(full_loops):
        result.extend(arr)

    # Add remaining partial slice with wrapping
    if remaining > 0:
        start_idx = start_idx % n
        end_idx = start_idx + remaining
        if end_idx <= n:
            result.extend(arr[start_idx:end_idx])
        else:
            # Wrap around
            result.extend(arr[start_idx:])
            result.extend(arr[:end_idx - n])

    return result


def _divide_hamiltonian_into_batches_fractional(hamiltonian:ClassicalHamiltonian,
                                                time_block_size:Optional[float],
                                                time_block_seed:Optional[int]=-1,
                                                max_depth:Optional[int]=None,
                                                assert_1q_terms_in_first_layer:bool=True, ):

    if time_block_size is None or np.isclose(time_block_size, 1.0,atol=1/len(hamiltonian.hamiltonian)):
        return {0:hamiltonian}


    shuffled_hamiltonian = hamiltonian.hamiltonian.copy()

    if assert_1q_terms_in_first_layer and 1 in hamiltonian.localities:
        terms_1q = [tup for tup in shuffled_hamiltonian if len(tup[1])==1]
        terms_other = [tup for tup in shuffled_hamiltonian if len(tup[1])!=1]
        shuffled_hamiltonian = terms_other



    if time_block_seed is not None:
        if time_block_seed == -1:
            _interactions_abs = [abs(_c) for _c, _ in shuffled_hamiltonian]
            if len(list(set(_interactions_abs))) == 1:
                #if this is +-1 Hamiltonian, we don't really want to shuffle anything and sorting might cluster +1s together
                pass
            else:
                #sort in descending order by absolute value of interaction
                shuffled_hamiltonian = sorted(shuffled_hamiltonian,
                                              key=lambda x: abs(x[0]),
                                              reverse=True)
        else:
            rng = np.random.default_rng(seed=time_block_seed)
            rng.shuffle(shuffled_hamiltonian)

    number_of_terms = len(shuffled_hamiltonian)

    # Each batch contains time_block_size fraction of terms (can be > 1.0 for repetition)
    terms_per_batch = int(round(time_block_size * number_of_terms))
    terms_per_batch = max(1, terms_per_batch)  # at least 1 term

    # Stride is where the next batch starts in the circular buffer
    stride = terms_per_batch % number_of_terms

    # Number of batches needed to cycle back to start
    if stride == 0:
        number_of_batches = 1
    else:
        number_of_batches = number_of_terms // np.gcd(stride, number_of_terms)


    if max_depth is not None:
        number_of_batches = min(number_of_batches, max_depth)

    original_class_description = hamiltonian.hamiltonian_class_description
    original_instance_description = hamiltonian.hamiltonian_instance_description

    tb_class_description = f"FractionalTimeBlockAnsatz;{original_class_description}"

    hamiltonian_batches = {}
    for batch_index in range(number_of_batches):
        tb_instance_description = f"Batch={batch_index};{original_instance_description}"

        # Extract terms with circular wrapping (supports repetition when terms_per_batch > number_of_terms)
        start_idx = (batch_index * stride) % number_of_terms
        ham_i = _extract_circular_slice(shuffled_hamiltonian, start_idx, terms_per_batch)

        if assert_1q_terms_in_first_layer and 1 in hamiltonian.localities:
            ham_i+=terms_1q

        hamiltonian_batches[batch_index] = ClassicalHamiltonian(hamiltonian_list_representation=ham_i,
                                                                number_of_qubits=hamiltonian.number_of_qubits,
                                                                hamiltonian_class_specifier=tb_class_description,
                                                                hamiltonian_instance_specifier=tb_instance_description
                                                                )



    return hamiltonian_batches



def _divide_hamiltonian_into_batches_swap_network(hamiltonian:ClassicalHamiltonian,
                                                  time_block_size:Optional[int],
                                                  max_depth:Optional[int]=None):
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

    depth = int(np.ceil(hamiltonian.number_of_qubits / time_block_size))

    return get_hamiltonian_partition_equivalent_to_time_block_ansatz_with_linear_swap_network(hamiltonian_phase=hamiltonian,
                                                                                              depth=depth,
                                                                                              time_block_size=time_block_size,
                                                                                              max_depth=max_depth)

def divide_hamiltonian_into_batches(hamiltonian:ClassicalHamiltonian,
                                    time_block_size:Optional[int|float],
                                    batching_type:TimeBlockBatchingType=TimeBlockBatchingType.FRACTIONAL,
                                    time_block_seed:Optional[int]=-1,
                                    max_depth:Optional[int]=None
                                    )->Dict[int,ClassicalHamiltonian]:

    if batching_type in [TimeBlockBatchingType.FRACTIONAL]:
        return _divide_hamiltonian_into_batches_fractional(hamiltonian=hamiltonian,
                                                           time_block_size=time_block_size,
                                                           time_block_seed=time_block_seed,
                                                           max_depth=max_depth)
    elif batching_type in [TimeBlockBatchingType.SWAP_NETWORK]:
        return _divide_hamiltonian_into_batches_swap_network(hamiltonian=hamiltonian,
                                                             time_block_size=time_block_size,
                                                             max_depth=max_depth)
    else:
        raise ValueError(f"Batching type {batching_type} not recognised.")




def build_fractional_time_block_ansatz_qiskit(hamiltonian_phase: ClassicalHamiltonian,
                                              depth: int,
                                              time_block_size: float,
                                              ansatz_builder_callable: Callable,
                                              ansatz_builder_kwargs: Optional[dict] = None,
                                              initial_state: InitialStateType | AbstractCircuit = InitialStateType.QAOA,
                                              add_barriers: bool = False,
                                              parameter_names: Tuple[str, str] = ("AngPS", "AngMIX"),
                                              shuffling_seed:Optional[int]=-1,
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
    from qiskit.circuit import ParameterVector, Parameter

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

    if time_block_size is None or np.isclose(time_block_size, 1.0,atol=1/len(hamiltonian_phase.hamiltonian)):
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
                                                               time_block_seed=shuffling_seed,
                                                               batching_type=TimeBlockBatchingType.FRACTIONAL)


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
        ws_param_obj = None
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

