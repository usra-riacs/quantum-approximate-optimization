# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

from typing import Optional

import numpy as np
from qiskit.transpiler import StagedPassManager
from qiskit_ibm_runtime.fake_provider import FakeAthensV2
import time
from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.circuits.gates.gate_delays import DelaySchedulerBase, add_delays_to_circuit_layers
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import PhaseSeparatorType, MixerType, QubitMappingType
from quapopt.optimization.QAOA.circuits.FullyConnectedQAOACircuit import FullyConnectedQAOACircuit
from quapopt.optimization.QAOA.circuits.LinearSwapNetworkQAOACircuit import LinearSwapNetworkQAOACircuit
from quapopt.optimization.QAOA.circuits.SabreMappedQAOACircuit import SabreMappedQAOACircuit
from quapopt.circuits.backend_utilities import (attempt_to_run_qiskit_circuit,
                                                get_counts_from_bit_array,
                                                create_qiskit_sampler_with_session,
                                                create_qiskit_session,
                                                create_qiskit_sampler)
from quapopt.circuits.backend_utilities.qiskit import QiskitSessionManagerMixin
from qiskit.primitives.containers import SamplerPubResult

class QAOARunnerQiskit(QiskitSessionManagerMixin):
    def __init__(self,
                 hamiltonian_phase: ClassicalHamiltonian,
                 # Compilation kwargs specific to Qiskit
                 qiskit_pass_manager: StagedPassManager,
                 qiskit_backend=None,
                 program_gate_builder: AbstractProgramGateBuilder = None,
                 number_of_qubits_device_qiskit: Optional[int] = None,
                 qubit_indices_physical: tuple = None,
                 classical_indices=None,
                 enforce_no_ancilla_qubits: bool = True,
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
                 delay_scheduler:DelaySchedulerBase = None,):

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
            classical_indices = list(range(self._number_of_qubits))[::-1]

        self._delay_scheduler = delay_scheduler


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
                                               input_state=input_state)
            _default_opt_level_pm = 2

        elif qubit_mapping_type == QubitMappingType.sabre:
            ansatz = SabreMappedQAOACircuit(qiskit_pass_manager=qiskit_pass_manager,
                                            hamiltonian_phase=hamiltonian_phase,
                                            depth=qaoa_depth,
                                            time_block_size=time_block_size,
                                            phase_separator_type=phase_separator_type,
                                            mixer_type=mixer_type,
                                            input_state=input_state,
                                            enforce_no_ancilla_qubits=enforce_no_ancilla_qubits)

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
        ansatz_circuit_abstract = ansatz.quantum_circuit.copy()
        for classical_index, qubit_index in zip(classical_indices, qubit_indices_physical[::-1]):
            ansatz_circuit_abstract.measure(qubit=qubit_index, cbit=classical_index)

        self.ansatz_circuit_abstract = ansatz_circuit_abstract
        self.parameters_PHASE = self.ansatz.parameters[0]
        self.parameters_MIXER = self.ansatz.parameters[1]

        # get isa circuit
        # ansatz_circuit = bck_utils.recompile_until_no_ancilla_qubits(quantum_circuit=ansatz.quantum_circuit.copy(),
        #                                                                   pass_manager=qiskit_pass_manager,
        #                                                                   expected_number_of_qubits=self._number_of_qubits,
        #                                                                   max_trials=20,
        #                                                                   enforce_no_ancilla_qubits = enforce_no_ancilla_qubits
        #                                                              )
        ansatz_circuit = ansatz.quantum_circuit.copy()

        for classical_index, qubit_index in zip(classical_indices, qubit_indices_physical[::-1]):
            ansatz_circuit.measure(qubit=qubit_index, cbit=classical_index)

        ansatz_circuit = qiskit_pass_manager.run(ansatz_circuit)


        self.ansatz_circuit = ansatz_circuit
        self._inverse_permutation = tuple(list(range(self._number_of_qubits))[::-1])
        
        # Initialize session management via mixin
        self._init_session_management(
            qiskit_backend=qiskit_backend,
            simulation=simulation,
            mock_context_manager_if_simulated=mock_context_manager_if_simulated,
            session_ibm=session_ibm
        )
        
        # QAOA-specific attributes
        self._qiskit_sampler_options = qiskit_sampler_options
        self._sampler = None  # Cached sampler instance

        if not self._simulation:
            anf.cool_print("WARNING:",'Running QAOA on real hardware.\n'
                                      'Proceed with caution :-)','red')











    @property
    def AnsatzSpecifier(self):
        return self.ansatz.AnsatzSpecifier
    
    def _ensure_sampler(self, number_of_shots: int):
        """Create or update cached sampler if needed."""
        # If we don't have a session, start one
        if self._current_session is None:
            self.start_session()
        
        # If we don't have a sampler, create a new one
        #TODO(FBM): if the sampler exists, we don't update number of shots
        if self._sampler is None:
            self._sampler = create_qiskit_sampler(
                qiskit_backend=self._qiskit_backend,
                simulation=self._simulation,
                number_of_shots=number_of_shots,
                qiskit_sampler_options=self._qiskit_sampler_options,
                session_ibm=self._current_session
            )
        return self._sampler

    def end_session(self):
        """End the current session (only if we created it). Override to clear sampler."""
        super().end_session()  # Call mixin's end_session
        # Clear sampler when session ends (QAOA-specific behavior)
        self._sampler = None

    def remap_bitstrings(self,
                         bitstrings_array: np.ndarray, ):

        swap_network_permutation = self.ansatz.permutation_circuit_network
        if swap_network_permutation is None:
            return bitstrings_array

        swap_network_permutation_reversed = anf.reverse_permutation(permutation=swap_network_permutation)
        # TODO(FBM): verify whether for qiskit it should be reversed or not.
        # I think we also want to first reverse the bitstrings, because qiskit's convention is weird
        permutation_to_apply = anf.concatenate_permutations(permutations=[self._inverse_permutation,
                                                                          swap_network_permutation_reversed])

        return anf.apply_permutation_to_array(array=bitstrings_array,
                                              permutation=permutation_to_apply)

    def run_qaoa(self,
                 angles_PHASE,
                 angles_MIXER,
                 number_of_samples: int):

        if isinstance(angles_PHASE,float):
            angles_PHASE = np.array([angles_PHASE])
        if isinstance(angles_MIXER,float):
            angles_MIXER = np.array([angles_MIXER])

        angles_dict = {self.parameters_PHASE: angles_PHASE,
                       self.parameters_MIXER: angles_MIXER}

        isa_circuit_resolved = self._program_gate_builder.parameters_resolver(quantum_circuit=self.ansatz_circuit,
                                                                              memory_map=angles_dict)
        
        # Get or create cached sampler
        t0 = time.perf_counter()
        sampler = self._ensure_sampler(number_of_shots=number_of_samples)
        t1 = time.perf_counter()
        
        _success, job, results, df_job_metadata = attempt_to_run_qiskit_circuit(
            circuit_isa=isa_circuit_resolved,
            sampler_ibm=sampler,
            number_of_shots=number_of_samples,
            max_attempts_run=5)
        t2 = time.perf_counter()
        results:SamplerPubResult = results

        if not _success:
            raise RuntimeError(f"Failed to run QAOA circuit after 5 attempts")
        

        if len(results.data.values())>1:
            print(type(isa_circuit_resolved))
            print(results.data)
            raise ValueError("Multiple data keys found in results. ")


        data_c = list(results.data.values())[0]



        bitstrings_res, counts_res = get_counts_from_bit_array(bit_array=data_c)
        mapped_array = self.remap_bitstrings(bitstrings_array=bitstrings_res)
        bitstrings_array = np.repeat(mapped_array, counts_res, axis=0)
        t3 = time.perf_counter()
        #print("Time to create sampler:", t1 - t0)
        #print("Time to run circuit:", t2 - t1)
        #print("Time to post-process results:", t3 - t2)
        return job, bitstrings_array
