# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import Optional

import numpy as np
from qiskit.transpiler import StagedPassManager
from qiskit_ibm_runtime.fake_provider import FakeAthensV2
import time
from quapopt import ancillary_functions as anf

from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.circuits.gates.gate_delays import DelaySchedulerBase, add_delays_to_circuit_layers
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import PhaseSeparatorType, MixerType, QubitMappingType
from quapopt.optimization.QAOA.circuits.FullyConnectedQAOACircuit import FullyConnectedQAOACircuit
from quapopt.optimization.QAOA.circuits.LinearSwapNetworkQAOACircuit import LinearSwapNetworkQAOACircuit
from quapopt.optimization.QAOA.circuits.SabreMappedQAOACircuit import SabreMappedQAOACircuit
from quapopt.circuits.backend_utilities import (attempt_to_run_qiskit_circuits,
                                                get_counts_from_bit_array)
from quapopt.circuits.backend_utilities.qiskit import QiskitSessionManagerMixin
from qiskit.primitives.containers import SamplerPubResult
from quapopt.circuits import backend_utilities as bck_utils

class QAOARunnerQiskit(QiskitSessionManagerMixin):
    """
    Qiskit-based QAOA circuit runner with comprehensive qubit mapping and measurement handling.
    
    This class provides a complete QAOA execution interface using Qiskit as the quantum computing
    backend. It supports multiple qubit mapping strategies (linear swap network, fully connected, 
    Sabre routing), handles both simulation and hardware execution, and manages quantum sessions
    for IBM Quantum backends.
    
    The class automatically handles complex index transformations required for different circuit
    topologies, ensuring that measurement results are correctly interpreted regardless of the
    underlying qubit mapping strategy used.
    
    Key Features:
    - Multiple circuit topologies with automatic ansatz selection
    - Intelligent measurement mapping that eliminates post-processing complexity
    - IBM Quantum session management with context switching
    - Delay scheduling for realistic hardware modeling
    - Flexible backend support (simulators, real hardware)
    
    :param hamiltonian_phase: Phase Hamiltonian defining the optimization problem
    :type hamiltonian_phase: ClassicalHamiltonian
    :param qiskit_pass_manager: Pass manager for circuit compilation and optimization
    :type qiskit_pass_manager: StagedPassManager
    :param qiskit_backend: Qiskit backend (simulator or hardware), defaults to FakeAthensV2
    :type qiskit_backend: Backend, optional
    :param program_gate_builder: Gate builder for hardware-specific implementations
    :type program_gate_builder: AbstractProgramGateBuilder, optional
    :param number_of_qubits_device_qiskit: Total qubits available on the device
    :type number_of_qubits_device_qiskit: int, optional
    :param qubit_indices_physical: Physical qubit indices to use for the circuit
    :type qubit_indices_physical: Tuple[int, ...], optional
    :param classical_indices: Classical bit indices for measurement mapping
    :type classical_indices: List[int], optional
    :param qaoa_depth: Number of QAOA layers (p parameter)
    :type qaoa_depth: int
    :param time_block_size: Time blocking parameter (meaning depends on circuit type)
    :type time_block_size: int or float, optional
    :param qubit_mapping_type: Circuit topology strategy to use
    :type qubit_mapping_type: QubitMappingType
    :param phase_separator_type: Type of phase separator gates
    :type phase_separator_type: PhaseSeparatorType
    :param mixer_type: Type of mixer gates
    :type mixer_type: MixerType
    :param every_gate_has_its_own_parameter: Whether each gate gets independent parameters
    :type every_gate_has_its_own_parameter: bool
    :param add_barriers: Whether to add quantum barriers between layers
    :type add_barriers: bool
    :param simulation: Whether to run in simulation mode or use real hardware
    :type simulation: bool
    :param qiskit_sampler_options: Additional options for the Qiskit Sampler
    :type qiskit_sampler_options: Dict[str, Any], optional
    :param mock_context_manager_if_simulated: Whether to mock session context for simulators
    :type mock_context_manager_if_simulated: bool
    :param session_ibm: External IBM Quantum session to reuse
    :type session_ibm: Session, optional
    :param delay_scheduler: Scheduler for adding realistic gate delays
    :type delay_scheduler: DelaySchedulerBase, optional
    :param noiseless_simulation: Whether to disable noise models for simulation
    :type noiseless_simulation: bool
    
    Example:
        >>> from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
        >>> from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        >>> # Create Hamiltonian
        >>> ham = ClassicalHamiltonian([(1.0, (0, 1))], number_of_qubits=2)
        >>> pass_manager = generate_preset_pass_manager(backend=backend)
        >>> # Create QAOA runner
        >>> qaoa_runner = QAOARunnerQiskit(
        ...     hamiltonian_phase=ham,
        ...     qiskit_pass_manager=pass_manager,
        ...     qaoa_depth=2,
        ...     qubit_mapping_type=QubitMappingType.sabre
        ... )
    
    .. note::
        The class automatically configures measurement operations to handle index transformations,
        eliminating the need for complex bitstring post-processing in most cases.
    
    .. note::
        Different circuit topologies have different optimization levels and compilation requirements.
        The class automatically selects appropriate defaults based on the chosen mapping type.
    """
    def __init__(self,
                 hamiltonian_phase: ClassicalHamiltonian,
                 # Compilation kwargs specific to Qiskit
                 qiskit_pass_manager: StagedPassManager,
                 qiskit_backend=None,
                 program_gate_builder: AbstractProgramGateBuilder = None,
                 number_of_qubits_device_qiskit: Optional[int] = None,
                 qubit_indices_physical: tuple = None,
                 classical_indices=None,
                 #enforce_no_ancilla_qubits: bool = True,
                 # Ansatz kwargs
                 qaoa_depth=1,
                 time_block_size=None,
                 qubit_mapping_type=QubitMappingType.linear_swap_network,
                 phase_separator_type=PhaseSeparatorType.QAOA,
                 mixer_type=MixerType.QAOA,
                 every_gate_has_its_own_parameter=False,
                 add_barriers=False,
                 # Session management kwargs
                 simulation: bool = True,
                 qiskit_sampler_options: Optional[dict] = None,
                 mock_context_manager_if_simulated: bool = True,
                 session_ibm=None,
                 delay_scheduler:DelaySchedulerBase = None,
                 noiseless_simulation: bool = False

                 ):

        input_state = None

        if qiskit_backend is None:
            qiskit_backend = FakeAthensV2()

        self._qiskit_backend = qiskit_backend




        self._program_gate_builder = program_gate_builder
        self._number_of_qubits = hamiltonian_phase.number_of_qubits

        if qubit_indices_physical is None:
            qubit_indices_physical = list(range(self._number_of_qubits))

        if number_of_qubits_device_qiskit is None:
            number_of_qubits_device_qiskit = self._qiskit_backend.num_qubits

        self._number_of_qubits_device_qiskit = number_of_qubits_device_qiskit

        self._qubit_indices_physical = qubit_indices_physical

        if classical_indices is None:
            #inverse mapping so that '01010000' means qubit 1 and 3 are in state |1>
            #(this is due to qiskit's reverse ordering of quantum bits)
            classical_indices = list(range(self._number_of_qubits))[::-1]

        self._delay_scheduler = delay_scheduler
        self._noiseless_simulation = noiseless_simulation

        if qubit_mapping_type == QubitMappingType.linear_swap_network:
            tuple_0_device = tuple(
                [(qubit_indices_physical[i], qubit_indices_physical[i + 1]) for i in
                 range(0, self._number_of_qubits - 1, 2)])
            tuple_1_device = tuple(
                [(qubit_indices_physical[i], qubit_indices_physical[i + 1]) for i in
                 range(1, self._number_of_qubits - 1, 2)])

            linear_chains_pair_device = (tuple_0_device, tuple_1_device)

            ansatz = LinearSwapNetworkQAOACircuit(sdk_name='qiskit',
                                                  depth=qaoa_depth,
                                                  hamiltonian_phase=hamiltonian_phase,
                                                  program_gate_builder=program_gate_builder,
                                                  time_block_size=time_block_size,
                                                  phase_separator_type=phase_separator_type,
                                                  mixer_type=mixer_type,
                                                  linear_chains_pair_device=linear_chains_pair_device,
                                                  every_gate_has_its_own_parameter=every_gate_has_its_own_parameter,
                                                  input_state=input_state)
            _default_opt_level_pm = 0
        elif qubit_mapping_type == QubitMappingType.fully_connected:
            ansatz = FullyConnectedQAOACircuit(sdk_name='qiskit',
                                               depth=qaoa_depth,
                                               hamiltonian_phase=hamiltonian_phase,
                                               program_gate_builder=program_gate_builder,
                                               time_block_size=time_block_size,
                                               phase_separator_type=phase_separator_type,
                                               mixer_type=mixer_type,
                                               every_gate_has_its_own_parameter=every_gate_has_its_own_parameter,
                                               add_barriers=add_barriers,
                                               initial_state=input_state)
            _default_opt_level_pm = 2

        elif qubit_mapping_type == QubitMappingType.sabre:
            ansatz = SabreMappedQAOACircuit(qiskit_pass_manager=qiskit_pass_manager,
                                            hamiltonian_phase=hamiltonian_phase,
                                            depth=qaoa_depth,
                                            time_block_size=time_block_size,
                                            phase_separator_type=phase_separator_type,
                                            mixer_type=mixer_type,
                                            initial_state=input_state,
                                            #enforce_no_ancilla_qubits=enforce_no_ancilla_qubits
                                            )

            _default_opt_level_pm = 2
        else:

            raise ValueError(f"Unsupported ansatz type: {qubit_mapping_type}")


        circuit_base_delay = ansatz.quantum_circuit
        circuit_delayed = add_delays_to_circuit_layers(quantum_circuit=circuit_base_delay,
                                                       number_of_qubits=self._number_of_qubits,
                                                       delay_scheduler=self._delay_scheduler,
                                                       for_visualization=False,
                                                       ignore_delay_at_the_end=False,
                                                       ignore_add_barriers_flag=False
                                                       )

        ansatz.quantum_circuit = circuit_delayed

        self.ansatz = ansatz

        # Store circuit before transpilation (with exception of SABRE, that transpiles circuit inside the class)
        ansatz_circuit = ansatz.quantum_circuit.copy()

        # self.ansatz_circuit_abstract = ansatz_circuit_abstract
        self.parameters_PHASE = self.ansatz.parameters[0]
        self.parameters_MIXER = self.ansatz.parameters[1]

        # Here we aim to create qubit mappings on the level of measurement operations, so we don't need to relabel bitstrings much
        if qubit_mapping_type == QubitMappingType.sabre:
            # original_indices_sabre = ansatz.logical_to_physical_qubits_map
            qubits_permutation = ansatz.qubit_mapping_permutation

            # We fully reverse the routing of qubits when creating measurement map
            bitstrings_permutation = None
            for classical_index, qubit_index in zip(classical_indices, qubits_permutation):
                ansatz_circuit.measure(qubit=qubit_index, cbit=classical_index)

        else:
            qubit_indices_physical = ansatz.logical_to_physical_qubits_map

            if qubit_mapping_type == QubitMappingType.linear_swap_network:
                swap_network_permutation = ansatz.qubit_mapping_permutation
                #print('swap_network_permutation:',swap_network_permutation)
                # we want to reverse swap network permutation:
                qubits_permutation = anf.reverse_permutation(permutation=swap_network_permutation)
                qubit_indices_measurement = [qubit_indices_physical[qubits_permutation[i]] for i in
                                             range(len(qubits_permutation))]
            else:
                qubit_indices_measurement = tuple(qubit_indices_physical)

            for classical_index, qubit_index in zip(classical_indices,
                                                    qubit_indices_measurement):
                ansatz_circuit.measure(qubit=qubit_index, cbit=classical_index)


            bitstrings_permutation = None

        self.ansatz.quantum_circuit = ansatz_circuit
        self.bitstrings_permutation = bitstrings_permutation

        # Initialize session management via mixin
        self._init_session_management(
            qiskit_backend=qiskit_backend,
            simulation=simulation,
            mock_context_manager_if_simulated=mock_context_manager_if_simulated,
            session_ibm=session_ibm,
            qiskit_sampler_options=qiskit_sampler_options,
            noiseless_simulation=self._noiseless_simulation)

        if not self._simulation:
            anf.cool_print("WARNING:", 'Running QAOA on real hardware.\n'
                                       'Proceed with caution :-)', 'red')

    @property
    def AnsatzSpecifier(self):
        return self.ansatz.AnsatzSpecifier
    @property
    def ansatz_circuit(self):
        return self.ansatz.quantum_circuit

    def remap_bitstrings(self,
                         bitstrings_array: np.ndarray, ):

        if self.bitstrings_permutation is None:
            return bitstrings_array


        # swap_network_permutation = self.ansatz.qubit_mapping_permutation
        # if swap_network_permutation is None:
        #     return bitstrings_array
        #
        # swap_network_permutation_reversed = anf.reverse_permutation(permutation=swap_network_permutation)

        return anf.apply_permutation_to_array(array=bitstrings_array,
                                              permutation=self.bitstrings_permutation)

    def run_qaoa(self,
                 angles_PHASE,
                 angles_MIXER,
                 number_of_samples: int):

        # TODO(FBM): ADD MEASUREMENT NOISE!

        if isinstance(angles_PHASE, float):
            angles_PHASE = np.array([angles_PHASE])
        if isinstance(angles_MIXER, float):
            angles_MIXER = np.array([angles_MIXER])

        angles_dict = {self.parameters_PHASE: angles_PHASE,
                       self.parameters_MIXER: angles_MIXER}

        isa_circuit_resolved = self._program_gate_builder.parameters_resolver(quantum_circuit=self.ansatz_circuit,
                                                                              memory_map=angles_dict)

        # Get or create cached sampler
        t0 = time.perf_counter()
        sampler = self._ensure_sampler()
        t1 = time.perf_counter()

        _success, job, results, df_job_metadata = attempt_to_run_qiskit_circuits(
            circuits_isa=isa_circuit_resolved,
            sampler_ibm=sampler,
            number_of_shots=number_of_samples,
            max_attempts_run=5)
        t2 = time.perf_counter()

        results: SamplerPubResult = results[0]

        if not _success:
            raise RuntimeError(f"Failed to run QAOA circuit after 5 attempts")

        if len(results.data.values()) > 1:
            print(type(isa_circuit_resolved))
            print(results.data)
            raise ValueError("Multiple data keys found in results. ")

        data_c = list(results.data.values())[0]
        bitstrings_res, counts_res = get_counts_from_bit_array(bit_array=data_c)
        mapped_array = self.remap_bitstrings(bitstrings_array=bitstrings_res)
        bitstrings_array = np.repeat(mapped_array, counts_res, axis=0)
        t3 = time.perf_counter()
        # print("Time to create sampler:", t1 - t0)
        # print("Time to run circuit:", t2 - t1)
        # print("Time to post-process results:", t3 - t2)
        return (job, df_job_metadata), bitstrings_array
