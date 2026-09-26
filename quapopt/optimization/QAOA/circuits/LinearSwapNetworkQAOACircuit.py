# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import Optional, Tuple, List, Dict

import numpy as np
from pydantic import conint

from quapopt import ancillary_functions as anf

from quapopt.circuits.gates import _SUPPORTED_SDKs
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.optimization.QAOA.circuits import MappedQAOACircuit

from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType, InitialStateType
from quapopt.circuits.gates import _SUPPORTED_SDKs, pyquil, qiskit, cirq, AbstractCircuit

from quapopt.optimization.QAOA.circuits.swap_networks import (get_linear_chain_permutation,
                                                              get_swap_network_permutation,
                                                              get_hamiltonian_partition_equivalent_to_time_block_ansatz_with_linear_swap_network)
from quapopt.optimization.QAOA.circuits.circuit_build_utilities import (build_initial_state_QAOA,
                                                                        build_mixer_layer_QAOA,
                                                                        build_phase_separator_layer_QAOA_2q,
                                                                        prepare_parametrized_circuit)
class LinearSwapNetworkQAOACircuit(MappedQAOACircuit):
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
    :param initial_state: Initial quantum state ('|+>' or '|0>')
    :type initial_state: str, optional
    :param add_barriers: Whether to add quantum barriers between layers (Qiskit only)
    :type add_barriers: bool
    
    Example:
        from quapopt.hamiltonians import ClassicalHamiltonian
        from quapopt.circuits.gates.logical import LogicalGateBuilderQiskit
        
        # Create MaxCut Hamiltonian
        ham = ClassicalHamiltonian([(1.0, (0, 1)), (1.0, (1, 2))], number_of_qubits=3)
        gate_builder = LogicalGateBuilderQiskit()
        
        # Build Linear Swap Network circuit
        circuit = LinearSwapNetworkQAOACircuit(
            sdk_name='qiskit',
            depth=2,
            hamiltonian_phase=ham,
            program_gate_builder=gate_builder,
            time_block_size=4
        )
    
    .. note::
        This approach is optimal for fully-connected graphs but inefficient for sparse graphs
        where many SWAP operations implement unused interactions.
    
    .. note::
        For Linear Swap Networks, `time_block_size` specifies the number of linear chains
        per layer, distinct from FullyConnected/SabreMapped QAOA where it represents interaction fraction.
    
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
            program_gate_builder: AbstractProgramGateBuilder,
            time_block_size: Optional[conint(ge=0)] = None,
            phase_separator_type=PhaseSeparatorType.QAOA,
            mixer_type=MixerType.QAOA,
            linear_chains_pair_device: Tuple[Tuple[int, ...], Tuple[int, ...]] = None,
            every_gate_has_its_own_parameter: bool = False,
            initial_state: InitialStateType = InitialStateType.QAOA,
            add_barriers: bool = False

    ):

        assert sdk_name.lower() in _SUPPORTED_SDKs, (f"Unsupported SDK: {sdk_name}. "
                                                     f"Please choose one of the following: {_SUPPORTED_SDKs}")



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

        assert isinstance(time_block_size, (int, np.int32, np.int64)), (
            f"time_block_size must be an integer, not {type(time_block_size)}: {time_block_size}")


        if every_gate_has_its_own_parameter:
            raise NotImplementedError("every_gate_has_its_own_parameter is not implemented yet")


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


        quantum_circuit, (angle_phase, angle_mixer, angle_bias_WS) = prepare_parametrized_circuit(sdk_name=sdk_name,
                                                                                                  number_of_qubits=number_of_qubits,
                                                                                                  qubit_ids_device=qubit_ids_device,
                                                                                                  depth=depth,
                                                                                                  mixer_type=mixer_type,
                                                                                                  initial_state=initial_state
                                                                                                  )

        hamiltonian_phase_dict = hamiltonian_phase.get_hamiltonian_dictionary()

        for qi in range(number_of_qubits):
            for qj in range(qi + 1, number_of_qubits):
                if (qi, qj) not in hamiltonian_phase_dict:
                    # this is necessary because even without PS edge we want to implement the SWAP network
                    hamiltonian_phase_dict[(qi, qj)] = 0.0

        quantum_circuit = build_initial_state_QAOA(quantum_circuit=quantum_circuit,
                                                   program_gate_builder=program_gate_builder,
                                                   qubit_ids_device=qubit_ids_device,
                                                   initial_state=initial_state,
                                                   bias_angles_WS=angle_bias_WS)


        # We need to create an object that stores what coefficients we need to implement
        current_edges_abstract = linear_chains_pair_abstract[0]
        # print(current_edges_abstract)

        linear_chain_permutations_abstract = [get_linear_chain_permutation(swap_chain=chain,
                                                                           number_of_qubits=number_of_qubits)
                                              for chain in linear_chains_pair_abstract]
        #current_indices_1q_abstract = list(range(number_of_qubits))
        current_permutation = list(range(number_of_qubits))

        batches_hamiltonian = []
        for layer_index in range(depth):
            beta = angle_mixer[layer_index] if layer_index < depth else 0.0
            gamma = angle_phase[layer_index] if layer_index < depth else 0.0

            # each layer implements "time_block_size" linear chains
            # full cycle contains "number_of_qubits" linear chains
            # the problem is that if time_block_size doesn't divide number_of_qubits,
            # then layer_index*time_block_size might give a little bit more than number_of_qubits at the borders

            total_number_of_cycles_so_far = layer_index * time_block_size
            implemented_2q_set = set()
            implemented_1q_set = None
            if total_number_of_cycles_so_far % number_of_qubits == 0:
                # single_qubit_indices_abstract_PS = [xi for xi in current_indices_1q_abstract if
                #                                     (xi,) in hamiltonian_phase_dict]
                
                # single_qubit_coefficients = [hamiltonian_phase_dict[(i,)] for i in
                #                              current_permutation if (i,) in hamiltonian_phase_dict]
                # single_qubit_indices_PS_device = qubit_ids_device

                # (1) phase for all one-qubit interactions
                # Extract coefficients and corresponding physical qubit positions together
                # to maintain alignment after filtering
                filtered_data = [(qubit_ids_device[idx], hamiltonian_phase_dict[(i,)])
                                 for idx, i in enumerate(current_permutation)
                                 if (i,) in hamiltonian_phase_dict]
                single_qubit_indices_PS_device = tuple([qubit for qubit, _ in filtered_data])
                single_qubit_coefficients = [coeff for _, coeff in filtered_data]
                # print('coeffs:',single_qubit_coefficients)
                # print('indices:',single_qubit_indices_PS_device)

                # In given layer, we add the phase separator for single qubit interactions at the end of the layer
                # TODO(FBM): in theory it doesn't matter if it's at the beginning or the end, in practice it might.
                quantum_circuit = program_gate_builder.exp_Z(quantum_circuit=quantum_circuit,
                                                             angles_tuple=tuple([coeff * gamma for coeff in
                                                                                 single_qubit_coefficients]),
                                                             qubits_tuple=single_qubit_indices_PS_device)
                implemented_1q_set = set()
                for i in current_permutation:
                    if (i,) in hamiltonian_phase_dict.keys():
                        if hamiltonian_phase_dict[(i,)]!=0:
                            implemented_1q_set.add((hamiltonian_phase_dict[(i,)], (i,)))

            # we are gonna write the code for SWAP network implementation of QAOA
            # what we want is we want to go through interactions one by one and implement them together with SWAPs
            # we will have to implement SWAPs for each interaction
            # The relevant edges are ordered in linear chains, so we can just go through them one by one

            # this is how many linear chains are implement in a single layer
            offset_gates = layer_index * time_block_size
            # Here we will implement Phase Separator Hamiltonian (or PS+mixer for ansatze such as QAMPA)
            for gates_cycle_index in range(time_block_size):
                # We take the coefficients from currently implemented part of the Hamiltonian
                current_coefficients = [hamiltonian_phase_dict[tup] for tup in current_edges_abstract]
                # e.g., this implementz Z0Z1 and Z2Z3 for the first cycle
                # edges on the device are fixed: it is always either first or second linear chain
                current_edges_device = linear_chains_pair_device[(offset_gates + gates_cycle_index) % 2]

                if len(current_edges_device) != len(current_coefficients):
                    raise ValueError(
                        f"Number of edges and coefficients don't match: {len(current_edges_device)} vs {len(current_coefficients)}")

                quantum_circuit = build_phase_separator_layer_QAOA_2q(program_gate_builder=program_gate_builder,
                                                                      quantum_circuit=quantum_circuit,
                                                                      gamma=gamma,
                                                                      phase_separator_type=phase_separator_type,
                                                                      coefficients_list=current_coefficients,
                                                                      edges_list=current_edges_device,
                                                                      with_swap_network=True,
                                                                      beta=beta)
                # Here we go through each gates cycle in the current layer
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
                # +1 because we plan for next cycle
                current_edges_abstract = linear_chains_pair_abstract[(offset_gates + gates_cycle_index + 1) % 2]
                current_edges_abstract = [tuple(sorted([current_permutation[xi] for xi in tup])) for tup in
                                          current_edges_abstract]

            #This is relevant for non-identical mixers! (similarly to RZ rotations of PS operator)

            if add_barriers:
                if sdk_name != 'qiskit':
                    raise ValueError("Barriers are only supported for Qiskit SDK")
                quantum_circuit.barrier(qubit_ids_device)

            #TODO(FBM): I think this should work but test this in WS setting!
            qubit_ids_device_permuted = [qubit_ids_device[current_permutation[i]] for i, _ in enumerate(qubit_ids_device)]
            quantum_circuit = build_mixer_layer_QAOA(program_gate_builder=program_gate_builder,
                                                     quantum_circuit=quantum_circuit,
                                                     list_of_qubits=qubit_ids_device_permuted,
                                                     beta=beta,
                                                     mixer_type=mixer_type,
                                                     bias_angles_WS=angle_bias_WS
                                                     )


            if add_barriers:
                if sdk_name != 'qiskit':
                    raise ValueError("Barriers are only supported for Qiskit SDK")
                quantum_circuit.barrier(qubit_ids_device)

            batch_i = []

            _sorted_interactions = sorted(list(implemented_2q_set), key=lambda x: x[1])
            #print(implemented_1q_set)
            if implemented_1q_set is not None and implemented_1q_set!={}:
                _sorted_1q_terms = sorted(list(implemented_1q_set), key=lambda x: x[1])
                batch_i += _sorted_1q_terms
            batch_i += _sorted_interactions
            batches_hamiltonian.append(batch_i)

        # print(number_of_qubits, depth,time_block_size)
        swap_network_permutation = get_swap_network_permutation(number_of_qubits=number_of_qubits,
                                                                depth=depth,
                                                                time_block_size=time_block_size)
        assert swap_network_permutation == current_permutation, "Something went wrong with permutations"
        # print('swap newtork:',swap_network_permutation)
        # print('ours:',current_permutation)

        # print("hejka:",swap_network_permutation)
        # raise KeyboardInterrupt
        # Now we want hamiltonian for which we don't have to do anything with the bitstrings when calculating energies
        # We apply swap network permutation to the hamiltonian on the level of abstract indices.

        test_list = [set(x) for x in batches_hamiltonian]
        test_list2 = [set(x.hamiltonian) for x in
                      get_hamiltonian_partition_equivalent_to_time_block_ansatz_with_linear_swap_network(
                          hamiltonian_phase=hamiltonian_phase,
                          depth=depth,
                          time_block_size=time_block_size).values()]

        #print('hejka5', test_list2)


        # print(test_list)
        # for x in test_list:
        #     print(x)
        # print('swap network perm:', swap_network_permutation)
        # print(test_list2)
        assert test_list == test_list2, "Something went wrong with the implementation of the SWAP network"


        _parameters = [angle_phase, angle_mixer]

        if angle_bias_WS is not None:
            _parameters += [angle_bias_WS]


        super().__init__(
            quantum_circuit=quantum_circuit,
            logical_to_physical_qubits_map=map_logical_qubits_to_physical_qubits,
            parameters=_parameters,
            qubit_mapping_permutation=swap_network_permutation,
            ansatz_specifier=ansatz_specifier,
            depth=depth,
            program_gate_builder=program_gate_builder,
            phase_separator_type=phase_separator_type,
            mixer_type=mixer_type,
            initial_state=initial_state,
            time_block_size=time_block_size,

        )

        self._linear_chains_pair_device = linear_chains_pair_device


    @property
    def linear_chains_pair_device(self):
        return self._linear_chains_pair_device
