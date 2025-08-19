# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import Optional

from pydantic import conint
import numpy as np

from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.circuits.gates import _SUPPORTED_SDKs
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.optimization.QAOA.circuits import MappedAnsatzCircuit


class FullyConnectedQAOACircuit(MappedAnsatzCircuit):
    """Build a parameterized quantum approximate optimization circuit for any optimization problem."""

    def __init__(
            self,
            sdk_name: str,
            depth: conint(ge=0),
            hamiltonian_phase: ClassicalHamiltonian,
            program_gate_builder: AbstractProgramGateBuilder,
            time_block_size: Optional[conint(ge=0)] = None,
            phase_separator_type=PhaseSeparatorType.QAOA,
            mixer_type=MixerType.QAOA,
            every_gate_has_its_own_parameter: bool = False,
            qubit_indices_physical=None,
            add_barriers=False,
            input_state: Optional[str] = None,
            enforce_no_ancilla_qubits: bool = True,

    ):
        """

        :param sdk_name:
        :param depth:
        :param hamiltonian_phase:
        :param program_gate_builder:
        :param time_block_size:
        NOTE: for fully-connected QAOA, time_block_size is the number of interactions in a layer.
        Note that this is disticnt from LinearSwapNetwork implementation, where time_block_size specifies
        number of LINEAR CHAINS in the layer.
        For fully connected topology, we can allow more freedom, hence this parametrization

        :param phase_separator_type:
        :param mixer_type:
        :param every_gate_has_its_own_parameter:
        """



        assert sdk_name.lower() in _SUPPORTED_SDKs, (f"Unsupported SDK: {sdk_name}. "
                                                     f"Please choose one of the following: {_SUPPORTED_SDKs}")
        if input_state is None:
            input_state = '|+>'

        ansatz_specifier = AnsatzSpecifier(
            phase_hamiltonian_class_specifier=hamiltonian_phase.hamiltonian_class_specifier,
            phase_hamiltonian_instance_specifier=hamiltonian_phase.hamiltonian_instance_specifier,
            depth=depth,
            phase_separator_type=phase_separator_type,
            mixer_type=mixer_type,
            qubit_mapping_type=QubitMappingType.fully_connected,
            time_block_size=time_block_size
        )

        number_of_qubits = hamiltonian_phase.number_of_qubits

        qubit_ids_device = qubit_indices_physical


        hamiltonian_abstract_interaction_edges_all = [tup for tup in hamiltonian_phase.hamiltonian if len(tup[1])==2]
        number_of_interactions = len(hamiltonian_abstract_interaction_edges_all)


        #
        if qubit_ids_device is None:
            qubit_ids_device = tuple(range(number_of_qubits))

        logical_qubit_indices = tuple(range(number_of_qubits))

        # We will need two types of indices. One is abstract qubit indexing so from 0 to n-1 for construction of SWAP network
        # The other is device qubit indexing, which is the actual qubit indices on the device. We need to map between them

        if time_block_size is None:
            time_block_size = number_of_interactions

        if every_gate_has_its_own_parameter:
            assert sdk_name.lower() in ['qiskit'], "Only Qiskit supports every gate has its own parameter"
            assert time_block_size == number_of_interactions, ("Every gate has its own parameter "
                                                         "only works for time_block_size = number_of_interactions")
            assert set(hamiltonian_phase.localities) == {
                2}, "Every gate has its own parameter only works for 2-local Hamiltonians"
            assert depth == 1, "Every gate has its own parameter only works for depth = 1"

        param_name_phase = "AngPHS"
        param_name_mixer = "AngMIX"

        if sdk_name.lower() in ['qiskit']:
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector

            quantum_circuit = QuantumCircuit(number_of_qubits, number_of_qubits)

            if not every_gate_has_its_own_parameter:
                angle_phase = ParameterVector(name=param_name_phase, length=depth) if depth > 0 else None
                angle_mixer = ParameterVector(name=param_name_mixer, length=depth) if depth > 0 else None
            else:
                # in that case, number of gates per layer is number of interactions in the Hamiltonian
                number_of_phase_gates_per_layer = time_block_size
                number_of_mixer_gates_per_layer = number_of_qubits
                # number_of_gates = depth*number_of_gates_per_layer
                angle_phase = ParameterVector(name=param_name_phase,
                                               length=number_of_phase_gates_per_layer*depth) if depth > 0 else None
                angle_mixer = ParameterVector(name=param_name_mixer, length=int(
                    depth * number_of_mixer_gates_per_layer)) if depth > 0 else None
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
        # TODO(FBM): add support for general input states
        # We start from |+>^n
        # print(program_gate_builder)
        if input_state == '|+>':
            quantum_circuit = program_gate_builder.H(quantum_circuit=quantum_circuit,
                                                 qubits_tuple=qubit_ids_device)
        elif input_state == '|0>':
            pass
        else:
            raise ValueError(f"Unsupported input state: {input_state}. Supported: |+>, |0>")

        number_of_interactions = len(hamiltonian_abstract_interaction_edges_all)
        #We will want to divide hamiltonian indices into batches of size time_block_size
        # We will implement them sequentially
        number_of_batches = int(np.ceil(number_of_interactions/time_block_size))

        divided_interactions_logical = [hamiltonian_abstract_interaction_edges_all[i*time_block_size:(i+1)*time_block_size]
                                         for i in range(0, number_of_batches)]

        print(divided_interactions_logical)

        divided_interactions_physical = [[(qubit_ids_device[pair[0]], qubit_ids_device[pair[1]]) for coeff, pair in edges] for
                                         edges in divided_interactions_logical]




        for layer_index in range(depth):
            if not every_gate_has_its_own_parameter:
                beta = angle_mixer[layer_index] if layer_index < depth else 0.0
                gamma = angle_phase[layer_index] if layer_index < depth else 0.0
            else:
                # in that case, number of gates per layer is number of interactions in the Hamiltonian
                gamma = angle_phase[layer_index * number_of_phase_gates_per_layer:(layer_index + 1) * number_of_phase_gates_per_layer]
                beta = angle_mixer[layer_index * number_of_mixer_gates_per_layer:(
                                                                                         layer_index + 1) * number_of_mixer_gates_per_layer]

            total_number_of_interactions_so_far = time_block_size * layer_index
            #We want to check if all interactions have already been implemented.
            #Sometimes the number of interactions is not divisible by time_block_size, so we need to account for that
            #We want to see when we already implemented all interactions. This means that layer_index is divisible by number_of_batches

            if layer_index%number_of_batches == 0:
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


            if every_gate_has_its_own_parameter:
                counter_gates = 0

            current_interactions_logical = divided_interactions_logical[layer_index%number_of_batches]
            current_coefficients = [tup[0] for tup in current_interactions_logical]
            current_edges_logical = [tup[1] for tup in current_interactions_logical]
            current_edges_device = divided_interactions_physical[layer_index%number_of_batches]
            if len(current_edges_device) != len(current_coefficients):
                raise ValueError(
                    f"Number of edges and coefficients don't match: {len(current_edges_device)} vs {len(current_coefficients)}")

            for coeff, edge_device in zip(current_coefficients, current_edges_device):
                if phase_separator_type in [PhaseSeparatorType.QAOA]:
                    if not every_gate_has_its_own_parameter:
                        quantum_circuit = program_gate_builder.exp_ZZ(quantum_circuit=quantum_circuit,
                                                                       angles_tuple=(gamma * coeff,),
                                                                       qubits_pairs_tuple=[edge_device]
                                                                       )
                    else:
                        quantum_circuit = program_gate_builder.exp_ZZ(quantum_circuit=quantum_circuit,
                                                                           angles_tuple=(
                                                                               gamma[counter_gates] * coeff,),
                                                                           qubits_pairs_tuple=[edge_device]
                                                                           )
                elif phase_separator_type in [PhaseSeparatorType.QAMPA]:
                    if not every_gate_has_its_own_parameter:
                        quantum_circuit = program_gate_builder.exp_ZZXXYY(quantum_circuit=quantum_circuit,
                                                                               angles_tuple=(
                                                                                   (gamma * coeff, beta)),
                                                                               qubits_pairs_tuple=(edge_device,)
                                                                               )
                    else:
                        quantum_circuit = program_gate_builder.exp_ZZXXYY(quantum_circuit=quantum_circuit,
                                                                               angles_tuple=((
                                                                                   gamma[counter_gates] * coeff,
                                                                                   beta)),
                                                                               qubits_pairs_tuple=(edge_device,)
                                                                               )



                else:
                    raise ValueError(f"Unsupported Phase Separator Type: {phase_separator_type}")

                if every_gate_has_its_own_parameter:
                    counter_gates += 1

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





        super().__init__(
            quantum_circuit=quantum_circuit,
            mapped_hamiltonian=hamiltonian_phase,
            logical_to_physical_qubits_map=qubit_ids_device,
            parameters=[angle_phase, angle_mixer],
            permutation_circuit_network=None,
            ansatz_specifier=ansatz_specifier

        )

        self._depth = depth
        self._phase_separator_type = phase_separator_type
        self._mixer_type = mixer_type
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
    def time_block_size(self):
        return self._time_block_size

    @property
    def gate_builder(self):
        return self._gate_builder
