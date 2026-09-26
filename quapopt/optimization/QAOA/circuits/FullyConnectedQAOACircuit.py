# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from functools import partial
from typing import Optional, List, Tuple, Dict

from pydantic import conint
import numpy as np

from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.circuits.gates import _SUPPORTED_SDKs, pyquil, qiskit, cirq, AbstractCircuit
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType, InitialStateType
from quapopt.optimization.QAOA.circuits import MappedQAOACircuit
from quapopt.optimization.QAOA.circuits.time_block_ansatz import build_fractional_time_block_ansatz_qiskit

from quapopt.optimization.QAOA.circuits.circuit_build_utilities import (build_initial_state_QAOA,
                                                                        build_mixer_layer_QAOA,
                                                                        build_phase_separator_layer_QAOA_2q,
                                                                        prepare_parametrized_circuit)

class FullyConnectedQAOACircuit(MappedQAOACircuit):
    """
    QAOA ansatz circuit optimized for fully-connected qubit topologies.
    
    This class constructs parameterized Quantum Approximate Optimization Algorithm (QAOA)
    circuits assuming all-to-all qubit connectivity. Unlike hardware-constrained circuits,
    this implementation can directly apply two-qubit gates between any pair of qubits without
    requiring routing or SWAP gates. It supports both standard QAOA and fractional time
    blocking approaches.
    
    The circuit construction is multi-SDK compatible (Qiskit, PyQuil, Cirq) and provides
    flexible control over circuit structure including phase separator types, mixer types,
    and initial state preparation.
    
    :param sdk_name: Quantum SDK to use for circuit construction ('qiskit', 'pyquil', 'cirq')
    :type sdk_name: str
    :param depth: Number of QAOA layers (p parameter)
    :type depth: int
    :param hamiltonian_phase: Phase Hamiltonian defining the optimization problem
    :type hamiltonian_phase: ClassicalHamiltonian
    :param program_gate_builder: Gate builder for SDK-specific gate implementations
    :type program_gate_builder: AbstractProgramGateBuilder
    :param time_block_size: Fraction of Hamiltonian interactions per layer (0.0-1.0) or None for standard QAOA
    :type time_block_size: float, optional
    :param phase_separator_type: Type of phase separator gates (QAOA or QAMPA)
    :type phase_separator_type: PhaseSeparatorType
    :param mixer_type: Type of mixer gates (QAOA or QAMPA)
    :type mixer_type: MixerType
    :param every_gate_has_its_own_parameter: Whether each gate gets independent parameters (not yet supported)
    :type every_gate_has_its_own_parameter: bool
    :param qubit_indices_physical: Optional mapping to specific physical qubits
    :type qubit_indices_physical: List[int], optional
    :param add_barriers: Whether to add quantum barriers between layers (Qiskit only)
    :type add_barriers: bool
    :param initial_state: Initial quantum state ('|+>', '|0>', or AbstractCircuit)
    :type initial_state: str or AbstractCircuit, optional
    :param number_of_qubits_circuit: Override for total circuit qubits
    :type number_of_qubits_circuit: int, optional
    
    Example:
        from quapopt.hamiltonians import ClassicalHamiltonian
        from quapopt.circuits.gates.logical import LogicalGateBuilderQiskit
        
        # Create MaxCut Hamiltonian
        ham = ClassicalHamiltonian([(1.0, (0, 1)), (1.0, (1, 2))], number_of_qubits=3)
        gate_builder = LogicalGateBuilderQiskit()
        
        # Build standard QAOA circuit
        circuit = FullyConnectedQAOACircuit(
            sdk_name='qiskit',
            depth=2,
            hamiltonian_phase=ham,
            program_gate_builder=gate_builder
        )
        
        # Build fractional time-blocked circuit
        fractional_circuit = FullyConnectedQAOACircuit(
            sdk_name='qiskit',
            depth=1,
            hamiltonian_phase=ham,
            program_gate_builder=gate_builder,
            time_block_size=0.5
        )
    
    .. note::
        For fully-connected QAOA, `time_block_size` represents the fraction of 
        Hamiltonian interactions included per layer, distinct from LinearSwapNetwork
        where it specifies the number of linear chains.
    
    .. note::
        When `time_block_size < 1.0`, the circuit automatically switches to fractional
        time blocking mode, creating multiple sub-layers with subsets of interactions.
    """

    def __init__(
            self,
            sdk_name: str,
            depth: conint(ge=0),
            hamiltonian_phase: ClassicalHamiltonian,
            program_gate_builder: AbstractProgramGateBuilder,
            time_block_size: Optional[float] = None,
            time_block_seed: Optional[int] = -1,
            phase_separator_type=PhaseSeparatorType.QAOA,
            mixer_type=MixerType.QAOA,
            every_gate_has_its_own_parameter: bool = False,
            qubit_indices_physical=None,
            add_barriers=False,
            initial_state: InitialStateType = InitialStateType.QAOA,
            number_of_qubits_circuit: int = None,
            time_block_partition: Optional[Dict[int, ClassicalHamiltonian]] = None,
            pre_created_bias_parameter=None,

    ):
        """

        :param sdk_name:
        :param depth:
        :param hamiltonian_phase:
        :param program_gate_builder:
        :param time_block_size:
        NOTE: for fully-connected QAOA, time_block_size is the FRACTION of Hamiltonian interactions
        per layer, in (0, 1] -- each batch then contains round(time_block_size * number_of_terms) terms.
        Note that this is distinct from LinearSwapNetwork implementation, where time_block_size specifies
        the integer number of LINEAR CHAINS in the layer.
        For fully connected topology, we can allow more freedom, hence this parametrization

        :param phase_separator_type:
        :param mixer_type:
        :param every_gate_has_its_own_parameter:
        """

        assert sdk_name.lower() in _SUPPORTED_SDKs, (f"Unsupported SDK: {sdk_name}. "
                                                     f"Please choose one of the following: {_SUPPORTED_SDKs}")

        if every_gate_has_its_own_parameter:
            raise NotImplementedError("every_gate_has_its_own_parameter is not implemented yet")

        ansatz_specifier = AnsatzSpecifier(
            PhaseHamiltonianClass=hamiltonian_phase.hamiltonian_class_specifier,
            PhaseHamiltonianInstance=hamiltonian_phase.hamiltonian_instance_specifier,
            Depth=depth,
            PhaseSeparatorType=phase_separator_type,
            MixerType=mixer_type,
            QubitMappingType=QubitMappingType.fully_connected,
            TimeBlockSize=time_block_size
        )

        number_of_qubits = hamiltonian_phase.number_of_qubits

        qubit_ids_device = qubit_indices_physical

        if qubit_ids_device is None:
            qubit_ids_device = tuple(range(number_of_qubits))

        logical_qubit_indices = tuple(range(number_of_qubits))

        hamiltonian_abstract_interaction_edges_all = [tup for tup in hamiltonian_phase.hamiltonian if len(tup[1]) == 2]
        if time_block_size is None:
            time_block_size = 1.0

        assert not every_gate_has_its_own_parameter, ("every_gate_has_its_own_parameter=True "
                                                      "is not supported for FullyConnectedQAOACircuit yet")

        quantum_circuit, (angle_phase, angle_mixer, angle_bias_WS) = prepare_parametrized_circuit(sdk_name=sdk_name,
                                                                                                  number_of_qubits=number_of_qubits,
                                                                                                  qubit_ids_device=qubit_ids_device,
                                                                                                  depth=depth,
                                                                                                  mixer_type=mixer_type,
                                                                                                  initial_state=initial_state,
                                                                                                  pre_created_bias_parameter=pre_created_bias_parameter)

       # print(depth, number_of_qubits, angle_phase, angle_mixer, angle_bias_WS)

        param_name_phase = angle_phase.name
        param_name_mixer = angle_mixer.name


        quantum_circuit = build_initial_state_QAOA(quantum_circuit=quantum_circuit,
                                                   program_gate_builder=program_gate_builder,
                                                   qubit_ids_device=qubit_ids_device,
                                                   initial_state=initial_state,
                                                   bias_angles_WS=angle_bias_WS)

        if time_block_size is None or np.isclose(time_block_size, 1.0, atol=1 / len(hamiltonian_phase.hamiltonian)):
            hamiltonian_phase_dict = hamiltonian_phase.get_hamiltonian_dictionary()

            for layer_index in range(depth):
                beta = angle_mixer[layer_index] if layer_index < depth else 0.0
                gamma = angle_phase[layer_index] if layer_index < depth else 0.0

                # We implement a layer of single-qubit gates if present in the Hamiltonian
                single_qubit_indices_abstract_PS = [xi for xi in logical_qubit_indices if
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


                current_coefficients = [tup[0] for tup in hamiltonian_abstract_interaction_edges_all]
                current_edges_logical = [tup[1] for tup in hamiltonian_abstract_interaction_edges_all]
                current_edges_device = [(qubit_ids_device[qi], qubit_ids_device[qj]) for qi, qj in
                                        current_edges_logical]

                if len(current_edges_device) != len(current_coefficients):
                    raise ValueError(
                        f"Number of edges and coefficients don't match: {len(current_edges_device)} vs {len(current_coefficients)}")


                quantum_circuit = build_phase_separator_layer_QAOA_2q(program_gate_builder=program_gate_builder,
                                                                      quantum_circuit=quantum_circuit,
                                                                      gamma=gamma,
                                                                      phase_separator_type=phase_separator_type,
                                                                      coefficients_list=current_coefficients,
                                                                      edges_list=current_edges_device,
                                                                      with_swap_network=False,
                                                                      beta=beta)

                if add_barriers:
                    if sdk_name != 'qiskit':
                        raise ValueError("Barriers are only supported for Qiskit SDK")
                    quantum_circuit.barrier(qubit_ids_device)


                quantum_circuit = build_mixer_layer_QAOA(program_gate_builder=program_gate_builder,
                                                         quantum_circuit=quantum_circuit,
                                                         list_of_qubits=qubit_ids_device,
                                                         beta=beta,
                                                         mixer_type=mixer_type,
                                                         bias_angles_WS=angle_bias_WS
                                                         )

                if add_barriers:
                    if sdk_name != 'qiskit':
                        raise ValueError("Barriers are only supported for Qiskit SDK")
                    quantum_circuit.barrier(qubit_ids_device)

        elif time_block_size < 1.0:
            ansatz_builder_callable = partial(FullyConnectedQAOACircuit,
                                              sdk_name=sdk_name,
                                              program_gate_builder=program_gate_builder,
                                              phase_separator_type=phase_separator_type,
                                              mixer_type=mixer_type,
                                              every_gate_has_its_own_parameter=every_gate_has_its_own_parameter,
                                              number_of_qubits_circuit=number_of_qubits_circuit,
                                              pre_created_bias_parameter=angle_bias_WS
                                              )

            if sdk_name.lower() == 'qiskit':
                quantum_circuit, (angle_phase, angle_mixer) = build_fractional_time_block_ansatz_qiskit(
                    hamiltonian_phase=hamiltonian_phase,
                    depth=depth,
                    time_block_size=time_block_size,
                    ansatz_builder_callable=ansatz_builder_callable,
                    ansatz_builder_kwargs={'qubit_indices_physical': qubit_indices_physical},
                    initial_state=quantum_circuit,
                    add_barriers=add_barriers,
                    parameter_names=(param_name_phase, param_name_mixer),
                    shuffling_seed=time_block_seed,
                    time_block_partition=time_block_partition
                )
            else:
                raise NotImplementedError("Fractional time block size is only implemented for Qiskit SDK")
        else:
            raise ValueError("time_block_size must be in (0, 1] for circuit not based on SWAP networks")


        _parameters = [angle_phase, angle_mixer]

        if angle_bias_WS is not None:
            _parameters.append(angle_bias_WS)


        super().__init__(
            quantum_circuit=quantum_circuit,
            logical_to_physical_qubits_map=qubit_ids_device,
            parameters=_parameters,
            qubit_mapping_permutation=None,
            ansatz_specifier=ansatz_specifier,
            depth=depth,
            program_gate_builder=program_gate_builder,
            phase_separator_type=phase_separator_type,
            mixer_type=mixer_type,
            initial_state=initial_state,
            time_block_size=time_block_size,


        )
