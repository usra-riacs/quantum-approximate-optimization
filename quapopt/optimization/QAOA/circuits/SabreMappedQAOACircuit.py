# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import Optional, List, Tuple, Dict

import numpy as np
from pydantic import conint, confloat
from qiskit import QuantumCircuit
from qiskit.transpiler.passmanager import StagedPassManager

from quapopt.circuits import backend_utilities as bck_utils
from quapopt.circuits.gates import _SUPPORTED_SDKs
from quapopt.circuits.gates.native.NativeGateBuilderHeron import NativeGateBuilderHeronCustomizable
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.optimization.QAOA.circuits import MappedQAOACircuit
from quapopt.optimization.QAOA.circuits.FullyConnectedQAOACircuit import FullyConnectedQAOACircuit
from quapopt.optimization.QAOA.circuits.qiskit_ansatze import build_qiskit_qaoa_ansatz
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType, InitialStateType
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_aer.backends.aer_simulator import AerSimulator

def _depth_filter_function(x):
    _test_1 = not getattr(x.operation, '_directive', False)
    _test_2 = x.operation.name.lower()!='rz'


    return _test_1 and _test_2


class SabreMappedQAOACircuit(MappedQAOACircuit):
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
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from quapopt.hamiltonians import ClassicalHamiltonian
        
        # Create Hamiltonian for MaxCut problem
        ham = ClassicalHamiltonian([(1.0, (0, 1)), (1.0, (1, 2))], number_of_qubits=3)
        
        # Generate Sabre pass manager for specific backend
        pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
        
        # Build Sabre-mapped QAOA circuit
        circuit = SabreMappedQAOACircuit(
            depth=2,
            hamiltonian_phase=ham,
            qiskit_pass_manager=pass_manager
        )
    
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
            qiskit_pass_manager: StagedPassManager = None,
            pass_manager_kwargs: Optional[dict] = None,
            pass_manager_seeds_list: Optional[List[int]] = None,
            qiskit_backend: Optional[IBMBackend | AerSimulator]=None,

            time_block_size: Optional[confloat(ge=0, le=1)] = None,
            time_block_seed:Optional[int]=-1,
            phase_separator_type=PhaseSeparatorType.QAOA,
            mixer_type=MixerType.QAOA,
            program_gate_builder=NativeGateBuilderHeronCustomizable(use_fractional_gates=False),
            # linear_chains_pair_device: Tuple[Tuple[int, ...], Tuple[int, ...]] = None,
            every_gate_has_its_own_parameter: bool = False,
            initial_state: Optional[InitialStateType] = InitialStateType.QAOA,
            add_barriers: bool = False,
            number_of_qubits_circuit: int = None,
            time_block_partition: Optional[Dict[int, ClassicalHamiltonian]] = None

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
            QubitMappingType=QubitMappingType.sabre,
            TimeBlockSize=time_block_size
        )

        #number_of_qubits = hamiltonian_phase.number_of_qubits
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
                                                      number_of_qubits_circuit=number_of_qubits_circuit,
                                                      time_block_seed=time_block_seed,
                                                      time_block_partition=time_block_partition)
            original_circuit = ansatz_qiskit.quantum_circuit
            parameters = ansatz_qiskit.parameters

        else:
            assert mixer_type == MixerType.QAOA, "Only QAOA mixer is supported for non-2-local Hamiltonians"
            assert phase_separator_type == PhaseSeparatorType.QAOA, "Only QAOA phase separator is supported for non-2-local Hamiltonians"
            assert initial_state == InitialStateType.QAOA, "Only QAOA initial state is supported for non-2-local Hamiltonians"

            #for higher-locality Hamiltonians, we use the standard qiskit QAOA ansatz builder
            ansatz_qiskit, parameters = build_qiskit_qaoa_ansatz(depth=depth,
                                                                 hamiltonian_phase=hamiltonian_phase,
                                                                 time_block_size=time_block_size,
                                                                 phase_separator_type=phase_separator_type,
                                                                 mixer_type=mixer_type,
                                                                 input_state=initial_state,
                                                                 number_of_qubits_circuit=number_of_qubits_circuit,
                                                                 time_block_seed=time_block_seed,
                                                                 time_block_partition=time_block_partition)
            original_circuit = ansatz_qiskit

            #parameters = ansatz_qiskit.parameters



        # #TODO(FBM): finish refactoring this

       # original_circuit = bck_utils.remove_idle_qubits_from_circuit(quantum_circuit=original_circuit)

        # qubit_indices_original = bck_utils.get_nontrivial_physical_indices_from_circuit(
        #     quantum_circuit=original_circuit)

        qubit_indices_original = list(range(hamiltonian_phase.number_of_qubits))

        if qiskit_pass_manager is None:

            assert qiskit_backend is not None, "If you don't provide a pass manager, you must provide a backend."

            if pass_manager_kwargs is None:
                pass_manager_kwargs = {}

            optimization_level = pass_manager_kwargs.get('optimization_level', 3)
            scheduling_method = pass_manager_kwargs.get('scheduling_method', None)

            if pass_manager_seeds_list is None:
                pass_manager_seeds_list = [0]

            coupling_map = pass_manager_kwargs.get('coupling_map', 'auto')

            if coupling_map == 'auto':
                # print('Auto-detecting coupling map from backend using heuristic filtering...')
                coupling_map = bck_utils.filter_couplings_map_heuristic(backend=qiskit_backend,
                                                                        std_multiplier=2)



            pass_manager_kwargs = pass_manager_kwargs.copy()

            pass_manager_kwargs.update({'optimization_level': optimization_level,
                                        'scheduling_method': scheduling_method,
                                        'coupling_map':coupling_map})


            _fom_best = np.inf
            _circuit_best = None

            for seed_transpiler in pass_manager_seeds_list:
                pass_manager_kwargs_i = pass_manager_kwargs.copy()
                pass_manager_kwargs_i['seed_transpiler'] = seed_transpiler

                qiskit_pass_manager_i, _ = bck_utils.get_qiskit_pass_manager(qiskit_backend=qiskit_backend,
                                                                            qubit_mapping_type=QubitMappingType.sabre,
                                                                            pass_manager_kwargs=pass_manager_kwargs_i)

                _circuit_isa_i = qiskit_pass_manager_i.run(original_circuit.copy())

                _depth_i = _circuit_isa_i.depth(_depth_filter_function)
                _gates_occurances_dict_i = bck_utils.count_gates_occurences_in_circuit(quantum_circuit=_circuit_isa_i)
                _cz_occurances_i = _gates_occurances_dict_i.get('cz', 0)

                _fom_i = 2*_depth_i+_cz_occurances_i

                if _fom_i < _fom_best:
                    _circuit_best = _circuit_isa_i
                    _fom_best = _fom_i


            circuit_qiskit_isa = _circuit_best






        else:
            assert pass_manager_kwargs is None, "If you provide a pass manager, you cannot provide pass manager kwargs."
            assert pass_manager_seeds_list is None, "If you provide a pass manager, you cannot provide pass manager seeds list."
            circuit_qiskit_isa = qiskit_pass_manager.run(original_circuit.copy())



        qubits_physical_indices = bck_utils.get_nontrivial_physical_indices_from_circuit(
            quantum_circuit=circuit_qiskit_isa)

        mapping_original_to_final = bck_utils.get_physical_qubits_mapping_from_circuit(quantum_circuit=circuit_qiskit_isa)


        qubit_mapping_permutation = tuple([mapping_original_to_final[i] for i in qubit_indices_original])



        super().__init__(
            quantum_circuit=circuit_qiskit_isa,
            logical_to_physical_qubits_map=qubits_physical_indices,
            parameters=parameters,
            qubit_mapping_permutation=qubit_mapping_permutation,
            ansatz_specifier=ansatz_specifier,
            depth=depth,
            program_gate_builder=program_gate_builder,
            phase_separator_type=phase_separator_type,
            mixer_type=mixer_type,
            initial_state=initial_state,
            time_block_size=time_block_size,)

        self._original_circuit = original_circuit
        self._qubits_physical_indices = qubits_physical_indices



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
