# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)



from typing import Optional, Tuple

import numpy as np
from pydantic import conint

from quapopt import ancillary_functions as anf

from quapopt.circuits.gates import _SUPPORTED_SDKs
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.optimization.QAOA.circuits import MappedAnsatzCircuit

from quapopt.circuits.gates import AbstractProgramGateBuilder



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
        >>> get_linear_chain_permutation([(0, 1), (2, 3)], 4)
        (1, 0, 3, 2)  # 0↔1, 2↔3, others unchanged
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


def apply_permutation_to_edges(edges_list, permutation):
    """
    Apply a qubit permutation to a list of interaction edges.
    
    This function transforms edge specifications according to a given permutation,
    ensuring edges remain in sorted form. Used to track how Hamiltonian interactions
    are mapped when qubits are permuted by SWAP operations.
    
    :param edges_list: List of edge tuples representing qubit interactions
    :type edges_list: List[Tuple[int, int]]
    :param permutation: Permutation tuple mapping qubit indices
    :type permutation: Tuple[int, ...]
    
    :returns: List of permuted edges in sorted form
    :rtype: List[Tuple[int, int]]
    
    Example:
        >>> edges = [(0, 1), (1, 2)]
        >>> perm = (1, 0, 2)  # swap qubits 0 and 1
        >>> apply_permutation_to_edges(edges, perm)
        [(0, 1), (0, 2)]  # (0,1) → (1,0) → (0,1), (1,2) → (0,2)
    """
    # edges_list is a list of sorted tuples, where each tuple is an edge (sorted)
    # permutation is a tuple, where each element is the image of the permutation on the corresponding element

    return [tuple(sorted((permutation[edge[0]], permutation[edge[1]]))) for edge in edges_list]


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
        >>> get_swap_network_permutation(4, depth=1, time_block_size=2)
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

    return anf.concatenate_permutations(permutations=all_permutations)


class LinearSwapNetworkQAOACircuit(MappedAnsatzCircuit):
    """
    QAOA ansatz circuit using Linear Swap Network topology for constrained qubit connectivity.
    
    This class constructs parameterized QAOA circuits optimized for linear qubit topologies
    where interactions are implemented through alternating SWAP networks. The approach uses
    two alternating linear chains of SWAP gates to enable interactions between all qubit pairs
    over time, making it ideal for fully-connected optimization problems on linear hardware.
    
    The Linear Swap Network realizes all pairwise interactions by alternating between:
    - Chain 1: (0,1), (2,3), (4,5), ... (even-indexed adjacent pairs)
    - Chain 2: (1,2), (3,4), (5,6), ... (odd-indexed adjacent pairs)
    
    Over multiple time blocks, these alternating chains enable all qubits to interact,
    effectively simulating fully-connected topology on linearly-connected hardware.
    
    :param sdk_name: Quantum SDK to use for circuit construction ('qiskit', 'pyquil', 'cirq')
    :type sdk_name: str
    :param depth: Number of QAOA layers (p parameter)
    :type depth: int
    :param hamiltonian_phase: Phase Hamiltonian defining the optimization problem
    :type hamiltonian_phase: ClassicalHamiltonian
    :param program_gate_builder: Gate builder for SDK-specific gate implementations
    :type program_gate_builder: AbstractProgramGateBuilder
    :param time_block_size: Number of linear chains per QAOA layer
    :type time_block_size: int, optional
    :param phase_separator_type: Type of phase separator gates (QAOA or QAMPA)
    :type phase_separator_type: PhaseSeparatorType
    :param mixer_type: Type of mixer gates (QAOA or QAMPA)
    :type mixer_type: MixerType
    :param linear_chains_pair_device: Custom device qubit mapping for the two linear chains
    :type linear_chains_pair_device: Tuple[Tuple[int, ...], Tuple[int, ...]], optional
    :param every_gate_has_its_own_parameter: Whether each gate gets independent parameters (Qiskit only)
    :type every_gate_has_its_own_parameter: bool
    :param input_state: Initial quantum state ('|+>' or '|0>')
    :type input_state: str, optional
    :param add_barriers: Whether to add quantum barriers between layers (Qiskit only)
    :type add_barriers: bool
    
    Example:
        >>> from quapopt.hamiltonians import ClassicalHamiltonian
        >>> from quapopt.circuits.gates.logical import LogicalGateBuilderQiskit
        >>> # Create MaxCut Hamiltonian
        >>> ham = ClassicalHamiltonian([(1.0, (0, 1)), (1.0, (1, 2))], number_of_qubits=3)
        >>> gate_builder = LogicalGateBuilderQiskit()
        >>> # Build Linear Swap Network circuit
        >>> circuit = LinearSwapNetworkQAOACircuit(
        ...     sdk_name='qiskit',
        ...     depth=2,
        ...     hamiltonian_phase=ham,
        ...     program_gate_builder=gate_builder,
        ...     time_block_size=4
        ... )
    
    .. note::
        This approach is optimal for fully-connected graphs but inefficient for sparse graphs
        where many SWAP operations implement unused interactions.
    
    .. note::
        For Linear Swap Networks, `time_block_size` specifies the number of linear chains
        per layer, distinct from FullyConnected QAOA where it represents interaction fraction.
    
    .. todo::
        Add "mirror trick" optimization for depth > 1 ansatz to reduce circuit depth.
    
    .. warning::
        The `every_gate_has_its_own_parameter` mode only supports Qiskit SDK with specific
        constraints: 2-local Hamiltonians, depth=1, and time_block_size=number_of_qubits.
    """

    def __init__(
            self,
            sdk_name: str,
            depth: conint(ge=0),
            hamiltonian_phase: ClassicalHamiltonian,
            program_gate_builder:AbstractProgramGateBuilder,
            time_block_size: Optional[conint(ge=0)] = None,
            phase_separator_type=PhaseSeparatorType.QAOA,
            mixer_type=MixerType.QAOA,
            linear_chains_pair_device: Tuple[Tuple[int, ...], Tuple[int, ...]] = None,
            every_gate_has_its_own_parameter: bool = False,
            input_state:Optional[str]=None,
            add_barriers:bool=False


    ):

        assert sdk_name.lower() in _SUPPORTED_SDKs, (f"Unsupported SDK: {sdk_name}. "
                                                     f"Please choose one of the following: {_SUPPORTED_SDKs}")

        if input_state is None:
            input_state = '|+>'

        ansatz_specifier = AnsatzSpecifier(
                                    PhaseHamiltonianClass=hamiltonian_phase.hamiltonian_class_specifier,
                                    PhaseHamiltonianInstance=hamiltonian_phase.hamiltonian_instance_specifier,
                                    Depth=depth,
                                    PhaseSeparatorType=phase_separator_type,
                                    MixerType=mixer_type,
                                    QubitMappingType=QubitMappingType.linear_swap_network,
                                    TimeBlockSize=time_block_size
                                    )

        # We will need two types of indices. One is abstract qubit indexing so from 0 to n-1 for construction of SWAP network
        # The other is device qubit indexing, which is the actual qubit indices on the device. We need to map between them

        number_of_qubits = hamiltonian_phase.number_of_qubits
        if time_block_size is None:
            time_block_size = number_of_qubits

        assert int(time_block_size)==time_block_size, "Time block size must be an integer"
        time_block_size = int(time_block_size)

        if every_gate_has_its_own_parameter:
            assert sdk_name.lower() in ['qiskit'], "Only Qiskit supports every gate has its own parameter"
            assert time_block_size==number_of_qubits, "Every gate has its own parameter only works for time_block_size = number_of_qubits"
            assert set(hamiltonian_phase.localities) == {2}, "Every gate has its own parameter only works for 2-local Hamiltonians"
            assert depth == 1, "Every gate has its own parameter only works for depth = 1"


            # if number_of_qubits%2==0:
        tuple_0_abstract = tuple([(i, i + 1) for i in range(0, number_of_qubits - 1, 2)])
        tuple_1_abstract = tuple([(i, i + 1) for i in range(1, number_of_qubits - 1, 2)])
        linear_chains_pair_abstract = [tuple_0_abstract, tuple_1_abstract]

        if linear_chains_pair_device is None:
            # If nothing provided, we assume that the mapping abstract_qubit->device_qubit is trivial q_i -> q_i
            linear_chains_pair_device = [tuple_0_abstract, tuple_1_abstract]

        # Otherwise, the mapping needs to be constructed. First set is the first chain, second set is the second chain
        tuple_0_device, tuple_1_device = linear_chains_pair_device

        # we flatten tuple_0_device
        qubit_ids_abstract = tuple(range(number_of_qubits))
        # first linear chain should contain all of the qubits of the device in case that number_of_qubits is even
        qubit_ids_device = [qubit for pair in tuple_0_device for qubit in pair]
        # If it's odd, then we need to add the last qubit
        if number_of_qubits % 2 == 1:
            qubit_ids_device += [tuple_1_device[-1][1]]
        qubit_ids_device = tuple(qubit_ids_device)

        # print(qubit_ids_abstract)
        # print(qubit_ids_device)
        # map_qubits_to_device = {qubit_ids_abstract[i]: qubit_ids_device[i] for i in range(number_of_qubits)}
        map_logical_qubits_to_physical_qubits = tuple(qubit_ids_device)

        param_name_phase = "AngPS"
        param_name_mixer = "AngMIX"

        if sdk_name.lower() in ['qiskit']:
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector

            number_of_qubits_physical = max(qubit_ids_device) + 1

            quantum_circuit = QuantumCircuit(number_of_qubits_physical, number_of_qubits)


            if not every_gate_has_its_own_parameter:
                angle_phase = ParameterVector(name=param_name_phase, length=depth) if depth > 0 else None
                angle_mixer = ParameterVector(name=param_name_mixer, length=depth) if depth > 0 else None
            else:
                #in that case, number of gates per layer is number of interactions in the Hamiltonian
                number_of_phase_gates_per_layer = int(number_of_qubits*(number_of_qubits-1)/2)
                number_of_mixer_gates_per_layer = number_of_qubits
                #number_of_gates = depth*number_of_gates_per_layer
                angle_phase = ParameterVector(name=param_name_phase, length=number_of_phase_gates_per_layer*depth) if depth > 0 else None
                angle_mixer = ParameterVector(name=param_name_mixer, length=int(depth*number_of_mixer_gates_per_layer)) if depth > 0 else None





        elif sdk_name.lower() in ['pyquil']:
            from pyquil import Program
            quantum_circuit = Program()
            angle_phase = quantum_circuit.declare(param_name_phase, "REAL", depth) if depth > 0 else None
            angle_mixer = quantum_circuit.declare(param_name_mixer, "REAL", depth) if depth > 0 else None

        elif sdk_name.lower() in ['cirq']:

            from cirq import Circuit
            import sympy
            quantum_circuit = Circuit()
            angle_phase = [sympy.Symbol(name=f"{param_name_phase}-{i}") for i in range(depth)]
            angle_mixer = [sympy.Symbol(name=f"{param_name_mixer}-{i}") for i in range(depth)]

        else:
            raise AssertionError((f"Unsupported SDK: {sdk_name}. "
                                  f"Please choose one of the following: {_SUPPORTED_SDKs}"))

        hamiltonian_phase_dict = hamiltonian_phase.get_hamiltonian_dictionary()

        for qi in range(number_of_qubits):
            for qj in range(qi + 1, number_of_qubits):
                if (qi, qj) not in hamiltonian_phase_dict:
                    # this is necessary because even without PS edge we want to implement the SWAP network
                    hamiltonian_phase_dict[(qi, qj)] = 0.0

        # TODO(FBM): add support for general input states
        # We start from |+>^n
        #print(program_gate_builder)
        if input_state == '|+>':
            quantum_circuit = program_gate_builder.H(quantum_circuit=quantum_circuit,
                                                     qubits_tuple=qubit_ids_device)
        elif input_state == '|0>':
            pass
        else:
            raise ValueError(f"Unsupported input state: {input_state}. Supported states are: |+>, |0>")

        # We need to create an object that stores what coefficients we need to implement
        current_edges_abstract = linear_chains_pair_abstract[0]
        # print(current_edges_abstract)

        linear_chain_permutations_abstract = [get_linear_chain_permutation(swap_chain=chain,
                                                                           number_of_qubits=number_of_qubits)
                                              for chain in linear_chains_pair_abstract]
        current_indices_abstract = list(range(number_of_qubits))
        current_permutation = list(range(number_of_qubits))

        for layer_index in range(depth):
            if not every_gate_has_its_own_parameter:
                beta = angle_mixer[layer_index] if layer_index < depth else 0.0
                gamma = angle_phase[layer_index] if layer_index < depth else 0.0
            else:
                #in that case, number of gates per layer is number of interactions in the Hamiltonian
                gamma = angle_phase[layer_index*number_of_phase_gates_per_layer:(layer_index+1)*number_of_phase_gates_per_layer]
                beta = angle_mixer[layer_index*number_of_mixer_gates_per_layer:(layer_index+1)*number_of_mixer_gates_per_layer]

            total_number_of_cycles_so_far = layer_index * time_block_size

            if total_number_of_cycles_so_far % number_of_qubits == 0:
                single_qubit_indices_abstract_PS = [xi for xi in current_indices_abstract if
                                                    (xi,) in hamiltonian_phase_dict]

                # (1) phase for all one-qubit interactions
                if len(single_qubit_indices_abstract_PS) > 0:
                    single_qubit_coefficients = [hamiltonian_phase_dict[(i,)] for i in
                                                 single_qubit_indices_abstract_PS]
                    single_qubit_indices_PS_device = [qubit_ids_device[i] for i in single_qubit_indices_abstract_PS]

                    # In given layer, we add the phase separator for single qubit interactions at the end of the layer
                    # TODO(FBM): in theory it doesn't matter if it's at the beginning or the end, in practice it might.
                    quantum_circuit = program_gate_builder.exp_Z(quantum_circuit=quantum_circuit,
                                                                 angles_tuple=tuple([coeff * gamma for coeff in
                                                                                     single_qubit_coefficients]),
                                                                 qubits_tuple=single_qubit_indices_PS_device)

            # we are gonna write the code for SWAP network implementation of QAOA
            # what we want is we want to go through interactions one by one and implement them together with SWAPs
            # we will have to implement SWAPs for each interaction
            # The relevant edges are ordered in linear chains, so we can just go through them one by one

            # this is how many linear chains are implement in a single layer
            offset_gates = layer_index * time_block_size
            # Here we will implement Phase Separator Hamiltonian (or PS+mixer for ansatze such as QAMPA)
            if every_gate_has_its_own_parameter:
                counter_gates = 0
            for gates_cycle_index in range(time_block_size):
                # We take the coefficients from currently implemented part of the Hamiltonian
                current_coefficients = [hamiltonian_phase_dict[tup] for tup in current_edges_abstract]
                # e.g., this implementz Z0Z1 and Z2Z3 for the first cycle
                # edges on the device are fixed: it is always either first or second linear chain
                current_edges_device = linear_chains_pair_device[(offset_gates + gates_cycle_index) % 2]

                if len(current_edges_device) != len(current_coefficients):
                    raise ValueError(
                        f"Number of edges and coefficients don't match: {len(current_edges_device)} vs {len(current_coefficients)}")

                for coeff, edge_device in zip(current_coefficients, current_edges_device):
                    # We implement the Phase Separator gate with the coefficient and SWAP gate
                    # We combine it PS+SWAP as single method, because sometimes there are efficient ways to compile them
                    # both together

                    if phase_separator_type in [PhaseSeparatorType.QAOA]:
                        if coeff == 0.0:
                            quantum_circuit = program_gate_builder.SWAP(quantum_circuit=quantum_circuit,
                                                                        qubits_pairs_tuple=[edge_device])
                        else:
                            if not every_gate_has_its_own_parameter:
                                quantum_circuit = program_gate_builder.exp_ZZ_SWAP(quantum_circuit=quantum_circuit,
                                                                                   angles_tuple=(gamma * coeff,),
                                                                                   qubits_pairs_tuple=[edge_device]
                                                                                   )
                            else:
                                quantum_circuit = program_gate_builder.exp_ZZ_SWAP(quantum_circuit=quantum_circuit,
                                                                                   angles_tuple=(gamma[counter_gates] * coeff,),
                                                                                   qubits_pairs_tuple=[edge_device]
                                                                                   )
                    elif phase_separator_type in [PhaseSeparatorType.QAMPA]:
                        if not every_gate_has_its_own_parameter:
                            quantum_circuit = program_gate_builder.exp_ZZXXYY_SWAP(quantum_circuit=quantum_circuit,
                                                                               angles_tuple=((gamma * coeff, beta)),
                                                                               qubits_pairs_tuple=(edge_device,)
                                                                               )
                        else:
                            quantum_circuit = program_gate_builder.exp_ZZXXYY_SWAP(quantum_circuit=quantum_circuit,
                                                                                   angles_tuple=((gamma[counter_gates] * coeff, beta)),
                                                                                   qubits_pairs_tuple=(edge_device,)
                                                                                   )



                    else:
                        raise ValueError(f"Unsupported Phase Separator Type: {phase_separator_type}")

                    if every_gate_has_its_own_parameter:
                        counter_gates += 1
                # Here we go through each gates cycle in the current layer
                linear_chain_permutation_here = linear_chain_permutations_abstract[
                    (offset_gates + gates_cycle_index) % 2]

                current_permutation = anf.concatenate_permutations(permutations=[linear_chain_permutation_here,
                                                                                 current_permutation,
                                                                                 ])
                # print('permutation:', current_permutation)
                # +1 because we plan for next cycle
                current_edges_abstract = linear_chains_pair_abstract[(offset_gates + gates_cycle_index + 1) % 2]
                current_edges_abstract = [tuple(sorted([current_permutation[xi] for xi in tup])) for tup in
                                          current_edges_abstract]

                current_indices_abstract = current_permutation

            if mixer_type in [MixerType.QAOA]:
                # (3) one-qubit mixing operators
                # In given layer, we add the phase separator for single qubit interactions at the end of the layer
                # TODO(FBM): in theory it doesn't matter if it's at the beginning or the end, in practice it might.
                if not every_gate_has_its_own_parameter:
                    quantum_circuit = program_gate_builder.exp_X(quantum_circuit=quantum_circuit,
                                                                 angles_tuple=beta,
                                                                 qubits_tuple=qubit_ids_device)
                else:
                    quantum_circuit = program_gate_builder.exp_X(quantum_circuit=quantum_circuit,
                                                                 angles_tuple=tuple([b for b in beta]),
                                                                 qubits_tuple=qubit_ids_device)
            elif mixer_type in [MixerType.QAMPA]:
                pass
            else:
                raise ValueError(f"Unsupported Mixer Type: {mixer_type}")


            if add_barriers:
                if sdk_name!='qiskit':
                    raise ValueError("Barriers are only supported for Qiskit SDK")
                quantum_circuit.barrier(qubit_ids_device)

        #print(number_of_qubits, depth,time_block_size)
        swap_network_permutation = get_swap_network_permutation(number_of_qubits=number_of_qubits,
                                                                depth=depth,
                                                                time_block_size=time_block_size)
        # Now we want hamiltonian for which we don't have to do anything with the bitstrings when calculating energies
        # We apply swap network permutation to the hamiltonian on the level of abstract indices.

        super().__init__(
            quantum_circuit=quantum_circuit,
            logical_to_physical_qubits_map=map_logical_qubits_to_physical_qubits,
            parameters=[angle_phase, angle_mixer],
            qubit_mapping_permutation=swap_network_permutation,
            ansatz_specifier=ansatz_specifier

        )

        self._depth = depth
        self._phase_separator_type = phase_separator_type
        self._mixer_type = mixer_type
        self._linear_chains_pair_device = linear_chains_pair_device
        self._time_block_size = time_block_size
        self._gate_builder = program_gate_builder

    @property
    def depth(self):
        return self._depth

    @property
    def phase_separator_type(self):
        return self._phase_separator_type

    @property
    def mixer_type(self):
        return self._mixer_type

    @property
    def linear_chains_pair_device(self):
        return self._linear_chains_pair_device

    @property
    def time_block_size(self):
        return self._time_block_size

    @property
    def gate_builder(self):
        return self._gate_builder








if __name__ == '__main__':
    # example usage
    # define the Hamiltonian
    number_of_qubits_test = 4
    swap_chain_1_test = [(i, i + 1) for i in range(0, number_of_qubits_test - 1, 2)]
    swap_chain_2_test = [(i, i + 1) for i in range(1, number_of_qubits_test - 1, 2)]

    permutation_1_test = get_linear_chain_permutation(swap_chain=swap_chain_1_test,
                                                      number_of_qubits=number_of_qubits_test)
    permutation_2_test = get_linear_chain_permutation(swap_chain=swap_chain_2_test,
                                                      number_of_qubits=number_of_qubits_test)
    print("TESTING SWAP NETWORK PERMUTATIONS:")
    print("linear chain 1:", swap_chain_1_test)
    print("permutation 1:", permutation_1_test)
    print("linear chain 2:", swap_chain_2_test)
    print("permutation 2:", permutation_2_test)

    time_block_size_test = None
    depth_test = 1
    swap_network_permutation_test = get_swap_network_permutation(number_of_qubits=number_of_qubits_test,
                                                                 depth=depth_test,
                                                                 time_block_size=time_block_size_test)
    print("swap network permutation:", swap_network_permutation_test)
    print("______________________________________________________________")

    pairs_test = [(i, j) for i in range(number_of_qubits_test) for j in range(i + 1, number_of_qubits_test)]

    # "name" coeffs via pair indices to make it easier to read in plots
    coeffs = list([int(''.join([str(x) for x in sorted([i, j], reverse=True)])) for (i, j) in pairs_test])
    hamiltonian_list_representation = [(coeff / 2, pair) for coeff, pair in zip(coeffs, pairs_test)]
    hamiltonian_list_representation += [((i + 1) / 2, (i,)) for i in range(number_of_qubits_test)]

    print({pair: coeff * 2 for coeff, pair in hamiltonian_list_representation})

    hamiltonian_test = ClassicalHamiltonian(hamiltonian_list_representation=hamiltonian_list_representation,
                                            number_of_qubits=number_of_qubits_test)

    from quapopt.circuits.gates.logical.LogicalGateBuilderQiskit import LogicalGateBuilderQiskit

    program_gate_builder_test = LogicalGateBuilderQiskit()
    circuit_builder_LSN = LinearSwapNetworkQAOACircuit(sdk_name='qiskit',
                                                       depth=depth_test,
                                                       hamiltonian_phase=hamiltonian_test,
                                                       program_gate_builder=program_gate_builder_test,
                                                       time_block_size=time_block_size_test,
                                                       phase_separator_type=PhaseSeparatorType.QAOA,
                                                       mixer_type=MixerType.QAOA,
                                                       linear_chains_pair_device=None)

    quantum_circuit_qiskit = circuit_builder_LSN.quantum_circuit
    # quantum_circuit_qiskit.draw(output='mpl').show()

    # raise KeyboardInterrupt
    from qiskit.quantum_info import Operator

    from quapopt.optimization.QAOA.simulation.wrapped_functions import get_QAOA_ket
    from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em

    # gamma_values = [0.1]*depth_test
    # betas_values = [0.2]*depth_test

    random_angles = np.random.uniform(-np.pi / 2, np.pi / 2, (2 * depth_test, 1000))
    from tqdm.notebook import tqdm

    gammas_list = circuit_builder_LSN.parameters[0]
    betas_list = circuit_builder_LSN.parameters[1]
    hamiltonian_matrix_diag = em.get_matrix_representation_of_classical_hamiltonian_diag(
        hamiltonian_test.hamiltonian_list_representation,
        number_of_qubits=number_of_qubits_test)

    for angles_index in tqdm(list(range(random_angles.shape[1]))):
        gamma_values = random_angles[:depth_test, angles_index]
        betas_values = random_angles[depth_test:, angles_index]

        memory_map = {}
        for gamma_index, gamma_value in enumerate(gamma_values):
            memory_map[gammas_list[gamma_index]] = gamma_value

        for beta_index, beta_value in enumerate(betas_values):
            memory_map[betas_list[beta_index]] = beta_value

        # So qiskit labels qubits in reversed order, therefore for p = 1 the kets should be equal (SWAP network is
        # complete reverse of the qiskit order)
        executable_qiskit = program_gate_builder_test.resolve_parameters(quantum_circuit=quantum_circuit_qiskit,
                                                                         memory_map=memory_map)

        circOp = Operator.from_circuit(executable_qiskit)
        circuit_unitary_qiskit = circOp.to_matrix()
        ket_qiskit = circuit_unitary_qiskit[:, 0].reshape(-1, 1)

        ket_direct = get_QAOA_ket(gammas=gamma_values,
                                  betas=betas_values,
                                  hamiltonian_matrix_diag=hamiltonian_matrix_diag)

        dot_product = np.vdot(ket_qiskit, ket_direct)
        overlap = abs(dot_product) ** 2

        assert np.isclose(overlap, 1.0), f"Doverlap is not 1.0: {overlap}"
