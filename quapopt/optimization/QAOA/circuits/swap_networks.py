# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from typing import Optional, Tuple, List, Dict

import numpy as np
from pydantic import conint

from quapopt import ancillary_functions as anf

from quapopt.circuits.gates import _SUPPORTED_SDKs
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.optimization.QAOA.circuits import MappedAnsatzCircuit

from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType, InitialStateType
from quapopt.circuits.gates import _SUPPORTED_SDKs, pyquil, qiskit, cirq, AbstractCircuit


def get_linear_chain_permutation(swap_chain, number_of_qubits):
    """
    Convert a SWAP chain specification to a standard permutation representation.

    This function transforms a sequence of SWAP operations into a permutation tuple
    following the standard format where π_i represents the image of element i under
    the permutation. For example, if π_0 = 1, element 0 is mapped to element 1.

    :param swap_chain: List of SWAP pairs, e.g., [(0,1), (2,3)] for swapping (0↔1) and (2↔3)
    :type swap_chain: List[Tuple[int, int]]
    :param number_of_qubits: Total number of qubits in the system
    :type number_of_qubits: int

    :returns: Permutation tuple (π_0, π_1, ..., π_{n-1}) representing qubit mapping
    :rtype: Tuple[int, ...]

    :raises AssertionError: If swap_chain contains duplicate qubit indices

    Example:
        get_linear_chain_permutation([(0, 1), (2, 3)], 4)
        # Returns: (1, 0, 3, 2)  # 0↔1, 2↔3, others unchanged
    """
    # permutation here is defined by SWAP chain. We want to convert it to standard format of permutations
    # In our standard format, permuation is tuple (\pi_0, \pi_1, \pi_2, ..., \pi_{n-1}),where \pi_i is image of the
    # permutation on i-th element. So, if \pi_0 = 1, it means that 0-th element is mapped to 1st element

    # verify swap_chain contains only unique elements
    # this requires flattening
    flat_swap_chain = [qubit for pair in swap_chain for qubit in pair]
    assert len(flat_swap_chain) == len(set(flat_swap_chain)), "Swap chain contains duplicate elements"

    # We will do this by going through the SWAP chain and constructing the permutation
    # the initial permutation is just identity, we update it using linear chain specification
    permutation = list(range(number_of_qubits))
    for swap_pair in swap_chain:
        qubit_0, qubit_1 = swap_pair
        permutation[qubit_0] = qubit_1
        permutation[qubit_1] = qubit_0

    return tuple(permutation)


def get_swap_network_permutation(number_of_qubits: int,
                                 depth: int,
                                 time_block_size: Optional[int] = None,
                                 ):
    """
    Construct the overall permutation for a complete Linear Swap Network circuit.

    This function computes the cumulative effect of alternating linear chains across
    all QAOA layers and time blocks. The swap network alternates between two chain
    types to enable all pairwise interactions over the circuit execution.

    The two alternating chain patterns are:
    - Chain 1: (0,1), (2,3), (4,5), ... (even-indexed pairs)
    - Chain 2: (1,2), (3,4), (5,6), ... (odd-indexed pairs)

    :param number_of_qubits: Total number of qubits in the circuit
    :type number_of_qubits: int
    :param depth: Number of QAOA layers (p parameter)
    :type depth: int
    :param time_block_size: Number of linear chains per layer, defaults to number_of_qubits
    :type time_block_size: int, optional

    :returns: Overall permutation representing cumulative qubit mapping
    :rtype: Tuple[int, ...]

    Example:
        get_swap_network_permutation(4, depth=1, time_block_size=2)
        # Returns permutation for 2 alternating chains in 1 layer

    .. note::
        The total number of chains is depth × time_block_size. Chains alternate
        between the two patterns, with Chain 1 used for odd total counts.
    """
    # We will construct the permutation for the SWAP network
    # We will do this by constructing the permutation for each layer and then multiplying them

    if time_block_size is None:
        time_block_size = number_of_qubits

    total_number_of_linear_chains = depth * time_block_size
    # there are two linear chains, so we will have to construct the permutation for each of them
    swap_chain_1 = [(i, i + 1) for i in range(0, number_of_qubits - 1, 2)]
    swap_chain_2 = [(i, i + 1) for i in range(1, number_of_qubits - 1, 2)]

    permutation_1 = get_linear_chain_permutation(swap_chain=swap_chain_1,
                                                 number_of_qubits=number_of_qubits)
    permutation_2 = get_linear_chain_permutation(swap_chain=swap_chain_2,
                                                 number_of_qubits=number_of_qubits)

    all_permutations = [permutation_1, permutation_2] * int(total_number_of_linear_chains / 2)
    if total_number_of_linear_chains % 2 == 1:
        # if we have odd number of linear chains, we need to add one more permutation
        all_permutations.append(permutation_1)

    return anf.concatenate_permutations(permutations=all_permutations,
                                        number_of_qubits=number_of_qubits)


def get_hamiltonian_partition_equivalent_to_time_block_ansatz_with_linear_swap_network(
        hamiltonian_phase: ClassicalHamiltonian,
        depth: conint(ge=0),
        time_block_size: conint(ge=0),
        max_depth:Optional[int]=None
) -> Optional[Dict[int, ClassicalHamiltonian]]:
    """
    Returns what Hamiltonian terms are implemented by time block ansatz with given parameters implemented via
    linear swap network.
    This function simply contains copy of lines of code from the class LinearSwapNetworkQAOACircuit that are responsible
    for tracking abstract qubit indices

    :param hamiltonian_phase:
    Hamiltonian to be implemented
    :param depth:
    :param time_block_size:
    :return:
    """

    number_of_qubits = hamiltonian_phase.number_of_qubits

    if time_block_size is None:
        time_block_size = number_of_qubits

    assert isinstance(time_block_size, (int, np.int32, np.int64)), (
        f"time_block_size must be an integer, not {type(time_block_size)}: {time_block_size}")

    if time_block_size == number_of_qubits:
        return {i: hamiltonian_phase for i in range(depth)}



    tuple_0_abstract = tuple([(i, i + 1) for i in range(0, number_of_qubits - 1, 2)])
    tuple_1_abstract = tuple([(i, i + 1) for i in range(1, number_of_qubits - 1, 2)])
    linear_chains_pair_abstract = [tuple_0_abstract, tuple_1_abstract]

    linear_chain_permutations_abstract = [get_linear_chain_permutation(swap_chain=chain,
                                                                       number_of_qubits=number_of_qubits)
                                          for chain in linear_chains_pair_abstract]
    hamiltonian_phase_dict = hamiltonian_phase.get_hamiltonian_dictionary()

    for qi in range(number_of_qubits):
        for qj in range(qi + 1, number_of_qubits):
            if (qi, qj) not in hamiltonian_phase_dict:
                hamiltonian_phase_dict[(qi, qj)] = 0.0

    current_indices_1q_abstract = list(range(number_of_qubits))
    current_edges_abstract = linear_chains_pair_abstract[0]

    current_permutation = list(range(number_of_qubits))
    batches_hamiltonian = {}

    original_class_description = hamiltonian_phase.hamiltonian_class_description
    original_instance_description = hamiltonian_phase.hamiltonian_instance_description
    tb_class_description = f"LinearSwapNetworkTimeBlockAnsatz;{original_class_description}"

    # Determine actual depth to use (handle None case)
    actual_depth = depth if max_depth is None else min(max_depth, depth)

    for layer_index in range(actual_depth):
        tb_instance_description = f"Batch={layer_index};{original_instance_description}"

        total_number_of_cycles_so_far = layer_index * time_block_size

        implemented_2q_set = set()
        implemented_1q_set = None
        if total_number_of_cycles_so_far % number_of_qubits == 0:
            single_qubit_indices_abstract_PS = [xi for xi in current_indices_1q_abstract if
                                                (xi,) in hamiltonian_phase_dict]
            if len(single_qubit_indices_abstract_PS) > 0:
                implemented_1q_set = {(hamiltonian_phase_dict[(i,)], (i,))
                                      for i in single_qubit_indices_abstract_PS
                                      if hamiltonian_phase_dict[(i,)] != 0
                                      }

        offset_gates = layer_index * time_block_size
        for gates_cycle_index in range(time_block_size):
            current_coefficients = [hamiltonian_phase_dict[tup] for tup in current_edges_abstract]
            linear_chain_permutation_here = linear_chain_permutations_abstract[
                (offset_gates + gates_cycle_index) % 2]

            current_permutation = anf.concatenate_permutations(permutations=[current_permutation,
                                                                             linear_chain_permutation_here,
                                                                             ],
                                                               number_of_qubits=number_of_qubits)
            implemented_2q_set = implemented_2q_set.union({(_c, _pair)
                                                           for _c, _pair in
                                                           zip(current_coefficients, current_edges_abstract)
                                                           if _c != 0.0
                                                           })
            current_edges_abstract = linear_chains_pair_abstract[(offset_gates + gates_cycle_index + 1) % 2]
            current_edges_abstract = [tuple(sorted([current_permutation[xi] for xi in tup])) for tup in
                                      current_edges_abstract]

            current_indices_1q_abstract = [current_permutation[xi] for xi in current_indices_1q_abstract]

        sorted_interactions = sorted(list(implemented_2q_set), key=lambda x: x[1])

        batch_i = []
        if implemented_1q_set is not None:
            sorted_1q_terms = sorted(list(implemented_1q_set), key=lambda x: x[1])
            batch_i += sorted_1q_terms
        batch_i += sorted_interactions

        CH_batch = ClassicalHamiltonian(hamiltonian_list_representation=batch_i,
                                        number_of_qubits=number_of_qubits,
                                        hamiltonian_class_specifier=tb_class_description,
                                        hamiltonian_instance_specifier=tb_instance_description
                                        )



        batches_hamiltonian[layer_index] = CH_batch

    return batches_hamiltonian
