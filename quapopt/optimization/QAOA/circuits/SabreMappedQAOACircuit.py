# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.


from typing import Optional

import numpy as np
import qiskit
from pydantic import conint, confloat
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz as QiskitQAOAAnsatz
from qiskit.transpiler.passmanager import StagedPassManager

from quapopt.circuits import backend_utilities as bck_utils
from quapopt.circuits.gates import _SUPPORTED_SDKs
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.optimization.QAOA.circuits import MappedAnsatzCircuit
from quapopt.optimization.QAOA.circuits.qiskit_ansatze import build_qiskit_qaoa_ansatz


class SabreMappedQAOACircuit(MappedAnsatzCircuit):
    """Build a parameterized quantum approximate optimization circuit for any optimization problem.

    This currently just wraps the qiskit's pass manager and the QAOA ansatz builder.
    In the future, we might extend its functionalities by time block ansatz addition and possibly others.

    """

    def __init__(
            self,
            # sdk_name: str,
            depth: conint(ge=0),
            hamiltonian_phase: ClassicalHamiltonian,
            qiskit_pass_manager: StagedPassManager,
            time_block_size: Optional[confloat(ge=0, le=1)] = None,
            phase_separator_type=PhaseSeparatorType.QAOA,
            mixer_type=MixerType.QAOA,
            # linear_chains_pair_device: Tuple[Tuple[int, ...], Tuple[int, ...]] = None,
            # every_gate_has_its_own_parameter: bool = False,
            input_state: Optional[str] = None,
            # add_barriers:bool=False,
            enforce_no_ancilla_qubits:bool= True,

    ):
        """
        :param depth:
        :param hamiltonian_phase:
        :param qiskit_pass_manager:
        :param time_block_size:

        Float that specifies what percentage of edges should be used in each time block.

        :param phase_separator_type:
        :param mixer_type:
        :param input_state:
        """

        sdk_name = 'qiskit'
        assert sdk_name.lower() in _SUPPORTED_SDKs, "qiskit not detected!"

        ansatz_specifier = AnsatzSpecifier(
            phase_hamiltonian_class_specifier=hamiltonian_phase.hamiltonian_class_specifier,
            phase_hamiltonian_instance_specifier=hamiltonian_phase.hamiltonian_instance_specifier,
            depth=depth,
            phase_separator_type=phase_separator_type,
            mixer_type=mixer_type,
            qubit_mapping_type=QubitMappingType.linear_swap_network,
            time_block_size=time_block_size
        )

        number_of_qubits = hamiltonian_phase.number_of_qubits
        hamiltonian_phase = hamiltonian_phase.copy()

        ansatz_qiskit = build_qiskit_qaoa_ansatz(depth=depth,
                                                 hamiltonian_phase=hamiltonian_phase,
                                                 time_block_size=time_block_size,
                                                 phase_separator_type=phase_separator_type,
                                                 mixer_type=mixer_type,
                                                 input_state=input_state)


        circuit_qiskit_isa: QuantumCircuit = bck_utils.recompile_until_no_ancilla_qubits(quantum_circuit=ansatz_qiskit,
                                                                                         pass_manager=qiskit_pass_manager,
                                                                                         expected_number_of_qubits=number_of_qubits,
                                                                                         enforce_no_ancilla_qubits=enforce_no_ancilla_qubits,
                                                                                         max_trials=20
                                                                                         )


        # TODO(FBM): verify that the mappings are correct; I might've misunderstood the Qiskit API
        logical_to_physical_qubits_map = bck_utils.get_logical_to_physical_qubits_map_from_circuit(
            quantum_circuit=circuit_qiskit_isa)

        logical_to_physical_qubits_map_tuple = np.zeros(number_of_qubits, dtype=int)
        for logical_bit_index, physical_bit_index in logical_to_physical_qubits_map.items():
            logical_to_physical_qubits_map_tuple[logical_bit_index] = physical_bit_index

        logical_to_physical_qubits_map_tuple = tuple([int(x) for x in logical_to_physical_qubits_map_tuple])

        #TODO(FBM): temporarily disabled 2025.08.13
        #swap_network_permutation = bck_utils.get_logical_swap_network_from_circuit(quantum_circuit=circuit_qiskit_isa)
        swap_network_permutation = tuple(range(number_of_qubits))
        # Now we want hamiltonian for which we don't have to do anything with the bitstrings when calculating energies
        # We apply swap network permutation to the hamiltonian on the level of abstract indices.
        # this is only for storage purposes
        hamiltonian_phase = hamiltonian_phase.copy()
        hamiltonian_phase = hamiltonian_phase.apply_permutation(permutation_tuple=swap_network_permutation)

        #print("hejka:",ansatz_qiskit.parameters)

        _beta_done = False
        _gamma_done = False
        for params_vector in ansatz_qiskit.parameters:
            if params_vector.name[0:6] == 'AngMIX':
                if not _beta_done:
                    angle_mixer = params_vector._vector
                    _beta_done = True

            elif params_vector.name[0:5] == 'AngPS':
                if not _gamma_done:
                    angle_phase = params_vector._vector
                    _gamma_done = True
            else:
                raise ValueError(f"Unknown parameter name: {params_vector.name}. ")

        #add classical register to circuit_qiskit_isa
        #circuit_qiskit_isa.add_register(qiskit.ClassicalRegister(number_of_qubits))

        super().__init__(
            quantum_circuit=circuit_qiskit_isa,
            mapped_hamiltonian=hamiltonian_phase,
            logical_to_physical_qubits_map=logical_to_physical_qubits_map_tuple,
            parameters=[angle_phase, angle_mixer],
            permutation_circuit_network=swap_network_permutation,
            ansatz_specifier=ansatz_specifier)

        self._depth = depth
        self._phase_separator_type = phase_separator_type
        self._mixer_type = mixer_type
        # self._linear_chains_pair_device = linear_chains_pair_device
        self._time_block_size = time_block_size
        # self._gate_builder = program_gate_builder
        self._ansatz_qiskit = ansatz_qiskit

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
    def ansatz_qiskit(self) -> QiskitQAOAAnsatz:
        """Get the Qiskit QAOA ansatz."""
        return self._ansatz_qiskit


def get_qaoa_ansatz_qiskit_router(
        phase_hamiltonian: ClassicalHamiltonian,
        qiskit_pass_manager: StagedPassManager,
        ansatz_kwargs: Optional[dict] = None,
        assert_no_ancilla_qubits: bool = True
) -> MappedAnsatzCircuit:
    # TODO(FBM): just generating it this way is likely to cause indexing errors. Should either solve this or remove

    from quapopt.optimization.QAOA.circuits.SabreMappedQAOACircuit import SabreMappedQAOACircuit

    if ansatz_kwargs is None:
        ansatz_kwargs = {'depth': 1,
                         'time_block_size': None,
                         'phase_separator_type': None,
                         'mixer_type': None,
                         }

    if not assert_no_ancilla_qubits:
        raise NotImplementedError("THis function is not implemented for circuits with ancilla qubits yet. ")

    for _ in range(10):
        if assert_no_ancilla_qubits:
            try:
                ansatz_qaoa = SabreMappedQAOACircuit(qiskit_pass_manager=qiskit_pass_manager,
                                                     # depth=depth,
                                                     hamiltonian_phase=phase_hamiltonian,
                                                     # time_block_size=time_block_size,
                                                     # phase_separator_type=phase_separator_type,
                                                     # mixer_type=mixer_type,
                                                     **ansatz_kwargs
                                                     )
                number_of_qubits_circuit = len(bck_utils.get_nontrivial_physical_indices_from_circuit(
                    quantum_circuit=ansatz_qaoa.quantum_circuit))
                if number_of_qubits_circuit == phase_hamiltonian.number_of_qubits:
                    break
            except(AssertionError) as e:
                continue
        else:
            break

    if assert_no_ancilla_qubits:
        assert phase_hamiltonian.number_of_qubits == number_of_qubits_circuit, (f"Exceeded maximal number of "
                                                                                f"tries for flag "
                                                                                f"assert_no_ancilla_qubits = True;"
                                                                                f" and didn't find the no-ancilla routing")

    return ansatz_qaoa



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
    from tqdm import tqdm

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


