# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)



from typing import Optional

import numpy as np
from pydantic import conint, confloat
from qiskit import QuantumCircuit
from qiskit.transpiler.passmanager import StagedPassManager

from quapopt.circuits import backend_utilities as bck_utils
from quapopt.circuits.gates import _SUPPORTED_SDKs
from quapopt.circuits.gates.native.NativeGateBuilderHeron import NativeGateBuilderHeronCustomizable
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.optimization.QAOA.circuits import MappedAnsatzCircuit
from quapopt.optimization.QAOA.circuits.FullyConnectedQAOACircuit import FullyConnectedQAOACircuit
from quapopt.optimization.QAOA.circuits.qiskit_ansatze import build_qiskit_qaoa_ansatz

class SabreMappedQAOACircuit(MappedAnsatzCircuit):
    """
    QAOA ansatz circuit with Sabre routing for hardware-aware qubit mapping.
    
    This class builds a parameterized Quantum Approximate Optimization Algorithm (QAOA) 
    circuit that uses Qiskit's Sabre routing algorithm to map logical qubits to physical 
    qubits based on hardware connectivity constraints. The circuit construction automatically
    handles both 2-local and higher-order Hamiltonians using appropriate ansatz builders.
    
    The Sabre algorithm performs routing and gate scheduling to minimize the number of 
    SWAP gates required while respecting the quantum hardware's limited connectivity graph.
    This makes the circuits executable on real quantum devices with specific qubit topologies.
    
    :param depth: Number of QAOA layers (p parameter)
    :type depth: int
    :param hamiltonian_phase: Phase Hamiltonian for the QAOA circuit
    :type hamiltonian_phase: ClassicalHamiltonian
    :param qiskit_pass_manager: Qiskit pass manager containing Sabre routing and optimization passes
    :type qiskit_pass_manager: StagedPassManager
    :param time_block_size: Fraction of Hamiltonian terms to include per time block (0.0-1.0)
    :type time_block_size: float, optional
    :param phase_separator_type: Type of phase separator gates to use
    :type phase_separator_type: PhaseSeparatorType
    :param mixer_type: Type of mixer gates to use
    :type mixer_type: MixerType
    :param program_gate_builder: Gate builder for hardware-specific gate compilation
    :type program_gate_builder: NativeGateBuilderHeronCustomizable
    :param every_gate_has_its_own_parameter: Whether each gate gets independent parameters
    :type every_gate_has_its_own_parameter: bool
    :param initial_state: Initial quantum state preparation
    :type initial_state: str, optional
    :param add_barriers: Whether to add quantum barriers between circuit layers
    :type add_barriers: bool
    :param number_of_qubits_circuit: Override for the number of qubits in circuit
    :type number_of_qubits_circuit: int, optional
    
    Example:
        >>> from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        >>> from quapopt.hamiltonians import ClassicalHamiltonian
        >>> # Create Hamiltonian for MaxCut problem
        >>> ham = ClassicalHamiltonian([(1.0, (0, 1)), (1.0, (1, 2))], number_of_qubits=3)
        >>> # Generate Sabre pass manager for specific backend
        >>> pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
        >>> # Build Sabre-mapped QAOA circuit
        >>> circuit = SabreMappedQAOACircuit(
        ...     depth=2,
        ...     hamiltonian_phase=ham,
        ...     qiskit_pass_manager=pass_manager
        ... )
    
    .. note::
        The circuit automatically detects whether the Hamiltonian is 2-local or higher-order
        and selects the appropriate ansatz builder. For 2-local Hamiltonians, it uses the
        custom FullyConnectedQAOACircuit for greater flexibility. For higher-order terms,
        it uses Qiskit's standard QAOA ansatz.
    
    .. warning::
        Sabre routing is non-deterministic and may produce different qubit mappings
        across multiple runs. Use the `qubit_mapping_permutation` property to track
        how logical qubits map to physical qubits after compilation.
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
            program_gate_builder=NativeGateBuilderHeronCustomizable(use_fractional_gates=False),
            # linear_chains_pair_device: Tuple[Tuple[int, ...], Tuple[int, ...]] = None,
            every_gate_has_its_own_parameter: bool = False,
            initial_state: Optional[str] = None,
            add_barriers: bool = False,
            number_of_qubits_circuit: int = None

    ):
        """
        :param depth:
        :param hamiltonian_phase:
        :param qiskit_pass_manager:
        :param time_block_size:

        Float that specifies what percentage of edges should be used in each time block.

        :param phase_separator_type:
        :param mixer_type:
        :param initial_state:
        """

        sdk_name = 'qiskit'
        assert sdk_name.lower() in _SUPPORTED_SDKs, "qiskit not detected!"

        ansatz_specifier = AnsatzSpecifier(
            PhaseHamiltonianClass=hamiltonian_phase.hamiltonian_class_specifier,
            PhaseHamiltonianInstance=hamiltonian_phase.hamiltonian_instance_specifier,
            Depth=depth,
            PhaseSeparatorType=phase_separator_type,
            MixerType=mixer_type,
            QubitMappingType=QubitMappingType.linear_swap_network,
            TimeBlockSize=time_block_size
        )

        number_of_qubits = hamiltonian_phase.number_of_qubits
        hamiltonian_phase = hamiltonian_phase.copy()

        if hamiltonian_phase.is_two_local:
            #For 2-local Hamiltonians, we use our custom ansatz builder that allows some more customization
            ansatz_qiskit = FullyConnectedQAOACircuit(sdk_name='qiskit',
                                                      depth=depth,
                                                      initial_state=initial_state,
                                                      hamiltonian_phase=hamiltonian_phase,
                                                      program_gate_builder=program_gate_builder,
                                                      time_block_size=time_block_size,
                                                      phase_separator_type=phase_separator_type,
                                                      mixer_type=mixer_type,
                                                      every_gate_has_its_own_parameter=every_gate_has_its_own_parameter,
                                                      add_barriers=add_barriers,
                                                      number_of_qubits_circuit=number_of_qubits_circuit)
            original_circuit = ansatz_qiskit.quantum_circuit
            parameters = ansatz_qiskit.parameters

        else:
            #for higher-locality Hamiltonians, we use the standard qiskit QAOA ansatz builder
            ansatz_qiskit, parameters = build_qiskit_qaoa_ansatz(depth=depth,
                                                     hamiltonian_phase=hamiltonian_phase,
                                                     time_block_size=time_block_size,
                                                     phase_separator_type=phase_separator_type,
                                                     mixer_type=mixer_type,
                                                     input_state=initial_state,
                                                     number_of_qubits_circuit=number_of_qubits_circuit)
            original_circuit = ansatz_qiskit

            #parameters = ansatz_qiskit.parameters



        # #TODO(FBM): finish refactoring this

       # original_circuit = bck_utils.remove_idle_qubits_from_circuit(quantum_circuit=original_circuit)

        qubit_indices_original = bck_utils.get_nontrivial_physical_indices_from_circuit(
            quantum_circuit=original_circuit)

        circuit_qiskit_isa = qiskit_pass_manager.run(original_circuit.copy())


        # circuit_qiskit_isa: QuantumCircuit = bck_utils.recompile_until_no_ancilla_qubits(
        #     quantum_circuit=original_circuit,
        #     pass_manager=qiskit_pass_manager,
        #     expected_number_of_qubits=number_of_qubits,
        #     #enforce_no_ancilla_qubits=enforce_no_ancilla_qubits,
        #     max_trials=20
        #     )

        qubits_physical_indices = bck_utils.get_nontrivial_physical_indices_from_circuit(
            quantum_circuit=circuit_qiskit_isa)

        mapping_original_to_final = bck_utils.get_physical_qubits_mapping_from_circuit(quantum_circuit=circuit_qiskit_isa)

        qubit_mapping_permutation = tuple([mapping_original_to_final[i] for i in qubit_indices_original])

        self._depth = depth
        self._phase_separator_type = phase_separator_type
        self._mixer_type = mixer_type
        self._time_block_size = time_block_size
        self._gate_builder = program_gate_builder
        self._original_circuit = original_circuit
        self._qubits_physical_indices = qubits_physical_indices

        super().__init__(
            quantum_circuit=circuit_qiskit_isa,
            logical_to_physical_qubits_map=qubits_physical_indices,
            parameters=parameters,
            qubit_mapping_permutation=qubit_mapping_permutation,
            ansatz_specifier=ansatz_specifier)

    @property
    def depth(self):
        """
        Get the QAOA circuit depth (number of layers).
        
        :returns: Number of QAOA layers (p parameter)
        :rtype: int
        """
        return self._depth

    @property
    def phase_separator_type(self):
        """
        Get the type of phase separator gates used in the circuit.
        
        :returns: Phase separator gate type configuration
        :rtype: PhaseSeparatorType
        """
        return self._phase_separator_type

    @property
    def mixer_type(self):
        """
        Get the type of mixer gates used in the circuit.
        
        :returns: Mixer gate type configuration
        :rtype: MixerType
        """
        return self._mixer_type

    @property
    def time_block_size(self):
        """
        Get the time block size for fractional QAOA.
        
        When time_block_size < 1.0, the Hamiltonian terms are split into 
        "time blocks".
        In particular, we now treat SUBSET of interactions as our phase separator Hamiltonian.
        If Hamiltonian has "n_terms", then a single parametrized layer of Time-Block ansatz consists of
        "n_terms*time_block_size" interactions.     
        
        
        :returns: Fraction of Hamiltonian terms per time block (0.0-1.0)
        :rtype: float or None
        """
        return self._time_block_size

    @property
    def circuit_before_compilation(self) -> QuantumCircuit:
        """
        Get the original QAOA circuit before Sabre routing compilation.
        
        This returns the logical circuit representation before any hardware-specific
        routing or optimization passes have been applied. Useful for debugging
        and understanding the original circuit structure.
        
        :returns: Original uncompiled QAOA quantum circuit
        :rtype: QuantumCircuit
        """
        return self._original_circuit

    @property
    def qubits_physical_indices(self):
        """
        Get the physical qubit indices used in the compiled circuit.
        
        After Sabre routing, the circuit uses specific physical qubits on the target
        hardware. This property returns the list of physical qubit indices that
        are actually utilized in the final compiled circuit.
        
        :returns: List of physical qubit indices used in the circuit
        :rtype: List[int]
        """
        return self._qubits_physical_indices


def get_qaoa_ansatz_qiskit_router(
        phase_hamiltonian: ClassicalHamiltonian,
        qiskit_pass_manager: StagedPassManager,
        ansatz_kwargs: Optional[dict] = None,
        assert_no_ancilla_qubits: bool = True
) -> MappedAnsatzCircuit:
    """
    Generate a Sabre-routed QAOA ansatz circuit with retry logic for ancilla-free compilation.
    
    This function creates a SabreMappedQAOACircuit and repeatedly attempts compilation
    until a circuit without ancilla qubits is found, or the maximum number of attempts
    is reached. The Sabre routing algorithm is non-deterministic, so multiple attempts
    may yield different qubit mappings.
    
    :param phase_hamiltonian: Phase Hamiltonian for the QAOA circuit
    :type phase_hamiltonian: ClassicalHamiltonian
    :param qiskit_pass_manager: Pass manager containing Sabre routing passes
    :type qiskit_pass_manager: StagedPassManager
    :param ansatz_kwargs: Additional keyword arguments for circuit construction
    :type ansatz_kwargs: dict, optional
    :param assert_no_ancilla_qubits: Whether to ensure the final circuit has no ancilla qubits
    :type assert_no_ancilla_qubits: bool
    
    :returns: Successfully compiled Sabre-mapped QAOA circuit
    :rtype: MappedAnsatzCircuit
    
    :raises AssertionError: If no ancilla-free circuit is found within maximum attempts
    :raises NotImplementedError: If assert_no_ancilla_qubits=False (not yet supported)
    
    .. warning::
        This function attempts compilation up to 10 times. For complex circuits or 
        restrictive hardware topologies, this may not be sufficient to find a 
        suitable mapping.
    
    .. todo::
        Address potential indexing errors mentioned in the original TODO comment.
    """
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
