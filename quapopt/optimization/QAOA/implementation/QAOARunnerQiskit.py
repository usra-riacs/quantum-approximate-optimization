# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
from qiskit import QuantumCircuit
from qiskit.primitives.containers.bit_array import BitArray as QiskitBitArray

from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from qiskit.transpiler import StagedPassManager
from qiskit_ibm_runtime.fake_provider import FakeAthensV2
import time
from quapopt import ancillary_functions as anf
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV, STANDARD_NAMES_DATA_TYPES as SNDT, \
    ResultsLogger
import pandas as pd
from quapopt.circuits.gates import AbstractProgramGateBuilder
from quapopt.circuits.gates.gate_delays import DelaySchedulerBase, add_delays_to_circuit_layers
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import PhaseSeparatorType, MixerType, QubitMappingType, InitialStateType
from quapopt.optimization.QAOA.circuits.FullyConnectedQAOACircuit import FullyConnectedQAOACircuit
from quapopt.optimization.QAOA.circuits.LinearSwapNetworkQAOACircuit import LinearSwapNetworkQAOACircuit
from quapopt.optimization.QAOA.circuits.SabreMappedQAOACircuit import SabreMappedQAOACircuit
from quapopt.circuits.backend_utilities import (attempt_to_run_qiskit_circuits,
                                                get_counts_from_bit_array)
from quapopt.circuits.backend_utilities.qiskit import QiskitSessionManagerMixin
from qiskit.primitives.containers import SamplerPubResult, DataBin, BitArray
from quapopt.circuits import backend_utilities as bck_utils
from qiskit_ibm_runtime import (
    Batch,
    SamplerV2 as Sampler,
    EstimatorV2 as Estimator,
)
from qiskit_aer.primitives import SamplerV2 as SamplerAer

from quapopt.data_analysis.data_handling import LoggingLevel
from qiskit_ibm_runtime import (Session as SessionRuntime,
                                SamplerV2 as SamplerRuntime,
                                RuntimeJobV2 as QiskitJobHardware)
from quapopt.optimization import EnergyResultMain
from quapopt.optimization.QAOA import QAOAResult

from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_aer.backends.aer_simulator import AerSimulator



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
    - Measurement mapping that attempts to eliminate post-processing complexity
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
        from quapopt.hamiltonians import ClassicalHamiltonian
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        
        # Create Hamiltonian
        ham = ClassicalHamiltonian([(1.0, (0, 1))], number_of_qubits=2)
        pass_manager = generate_preset_pass_manager(backend=backend)
        
        # Create QAOA runner
        qaoa_runner = QAOARunnerQiskit(
            hamiltonian_phase=ham,
            qiskit_pass_manager=pass_manager,
            qaoa_depth=2,
            qubit_mapping_type=QubitMappingType.sabre
        )
    
    .. note::
        The class automatically configures measurement operations to handle index transformations,
        eliminating the need for complex bitstring post-processing in most cases.
    
    .. note::
        Different circuit topologies have different optimization levels and compilation requirements.
        The class automatically selects appropriate defaults based on the chosen mapping type.
    """

    def __init__(self,
                 hamiltonian_phase: ClassicalHamiltonian,
                 qiskit_backend: IBMBackend | AerSimulator,
                 # Compilation kwargs specific to Qiskit
                 qiskit_pass_manager: StagedPassManager = None,
                 pass_manager_kwargs: Optional[dict] = None,
                 pass_manager_seeds_list: Optional[List[int]] = None,

                 program_gate_builder: AbstractProgramGateBuilder = None,
                 number_of_qubits_device_qiskit: Optional[int] = None,
                 qubit_indices_physical: tuple = None,
                 classical_indices=None,
                 # enforce_no_ancilla_qubits: bool = True,
                 # Ansatz kwargs
                 qaoa_depth=1,
                 time_block_size=None,
                 time_block_seed=-1,
                 time_block_partition: Optional[Dict[int, ClassicalHamiltonian]] = None,
                 qubit_mapping_type=QubitMappingType.sabre,
                 phase_separator_type=PhaseSeparatorType.QAOA,
                 mixer_type=MixerType.QAOA,
                 initial_state=InitialStateType.QAOA,
                 every_gate_has_its_own_parameter=False,
                 add_barriers=False,
                 # Session management kwargs
                 simulation: bool = True,
                 qiskit_sampler_options: Optional[dict] = None,
                 mock_context_manager_if_simulated: bool = True,
                 session_ibm=None,
                 delay_scheduler: DelaySchedulerBase = None,
                 noiseless_simulation: bool = False,
                 REPLACE_WITH_RANDOM_SAMPLING:bool=False,
                 ):


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
            # inverse mapping so that '01010000' means qubit 1 and 3 are in state |1>
            # (this is due to qiskit's reverse ordering of quantum bits)
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
                                                  initial_state=initial_state,
                                                  add_barriers=add_barriers, )
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
                                               time_block_seed=time_block_seed,
                                               initial_state=initial_state,
                                               time_block_partition=time_block_partition,
                                               )
            _default_opt_level_pm = 2

        elif qubit_mapping_type == QubitMappingType.sabre:
            ansatz = SabreMappedQAOACircuit(qiskit_pass_manager=qiskit_pass_manager,
                                            qiskit_backend=self._qiskit_backend,
                                            pass_manager_kwargs=pass_manager_kwargs,
                                            pass_manager_seeds_list=pass_manager_seeds_list,
                                            hamiltonian_phase=hamiltonian_phase,
                                            depth=qaoa_depth,
                                            time_block_size=time_block_size,
                                            phase_separator_type=phase_separator_type,
                                            mixer_type=mixer_type,
                                            initial_state=initial_state,
                                            time_block_seed=time_block_seed,
                                            program_gate_builder=program_gate_builder,
                                            add_barriers=add_barriers,
                                            time_block_partition=time_block_partition,
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
        # For parametric WS-QAOA, there's a third parameter for the bias angle
        self.parameters_BIAS_WS = self.ansatz.parameters[2] if len(self.ansatz.parameters) > 2 else None

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
                # print('hejka0', qubit_indices_physical)
                # print('hejka1',swap_network_permutation)
                # print('hejka2',qubits_permutation)
                # print('hejka3',qubit_indices_measurement)


            else:
                qubit_indices_measurement = tuple(qubit_indices_physical)

            for classical_index, qubit_index in zip(classical_indices,
                                                    qubit_indices_measurement):
                ansatz_circuit.measure(qubit=qubit_index, cbit=classical_index)

            # print(qiskit_pass_manager.__dict__)
            #display(circuit_base_delay.draw('mpl', idle_wires=False))
            #print(qiskit_pass_manager.__dict__)
            #ansatz_circuit = qiskit_pass_manager.run(ansatz_circuit)
            #display(ansatz_circuit.draw('mpl', idle_wires=False))

            bitstrings_permutation = None

        # self.ansatz_circuit = ansatz_circuit
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

        self._REPLACE_WITH_RANDOM_SAMPLING = REPLACE_WITH_RANDOM_SAMPLING


        self._numpy_rng = np.random.default_rng(seed=0)


        # if not self._simulation:
        #     anf.cool_print("WARNING:", 'Running QAOA on real hardware.\n'
        #                                'Proceed with caution :-)', 'red')

    @property
    def AnsatzSpecifier(self):
        return self.ansatz.AnsatzSpecifier

    @property
    def ansatz_circuit(self)->QuantumCircuit:
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

    @staticmethod
    def bias_to_theta(bias: float|np.ndarray) -> float:
        """
        Convert bias parameter to rotation angle.

        For WS-QAOA with bias parameter c (probability of |1> in input state),
        the RY rotation angle is theta = 2 * arcsin(sqrt(c)).

        Parameters
        ----------
        bias : float
            Bias parameter in [0, 0.5]

        Returns
        -------
        float
            Rotation angle theta
        """
        return 2 * np.arcsin(np.sqrt(bias))

    def run_qaoa(self,
                 angles_PHASE,
                 angles_MIXER,
                 number_of_samples: int,
                 bias_parameters_WS: Optional[List[float] | Tuple[float,...] | np.ndarray | float] = None):




        # TODO(FBM): ADD MEASUREMENT NOISE!

        if isinstance(angles_PHASE, float):
            angles_PHASE = np.array([angles_PHASE])
        if isinstance(angles_MIXER, float):
            angles_MIXER = np.array([angles_MIXER])

        angles_dict = {self.parameters_PHASE: angles_PHASE.reshape(-1),
                       self.parameters_MIXER: angles_MIXER.reshape(-1)}

        # For parametric WS-QAOA, bind the bias angle parameter
        if self.parameters_BIAS_WS is not None:
            if bias_parameters_WS is None:
                raise ValueError("angles_BIAS_WS must be provided for parametric WS-QAOA circuits")
            if isinstance(bias_parameters_WS, float):
                bias_parameters_WS = [bias_parameters_WS]
            bias_parameters_WS = np.array(bias_parameters_WS)
            
            angles_dict[self.parameters_BIAS_WS] = self.bias_to_theta(bias=bias_parameters_WS)


        all_pubs_isa = [(self.ansatz_circuit, angles_dict)]
        # Get or create cached sampler
        t0 = time.perf_counter()
        sampler:SamplerRuntime|SamplerAer = self._ensure_sampler()
        t1 = time.perf_counter()
        if self._REPLACE_WITH_RANDOM_SAMPLING:
            job = None

            t_start = time.perf_counter()
            bitstrings_array0 = self._numpy_rng.binomial(n=1, p=0.5, size=(number_of_samples, self._number_of_qubits))
            t_end = time.perf_counter()
            actual_runtime_wallclock = t_end - t_start

            df_job_metadata = pd.DataFrame(data={SNV.SessionId.id_long: ["RandomSampling"],
                                                 SNV.JobId.id_long: ["RandomSampling"],
                                                 'EstimatedRuntime': [None],
                                                 'ActualRuntimeQPU': [None],
                                                 'ActualRuntimeWallclock': [actual_runtime_wallclock],
                                                 })
            t2 = time.perf_counter()

            bitstrings_res, counts_res = np.unique(bitstrings_array0,
                                                  axis=0,
                                                  return_counts=True)
            mapped_array = self.remap_bitstrings(bitstrings_array=bitstrings_res)
            bitstrings_array = np.repeat(mapped_array, counts_res, axis=0)
            t3 = time.perf_counter()



        else:
            _success, job, results, df_job_metadata = attempt_to_run_qiskit_circuits(
                circuits_isa=all_pubs_isa,
                sampler_ibm=sampler,
                number_of_shots=number_of_samples,
                max_attempts_run=5)
            t2 = time.perf_counter()

            results: SamplerPubResult = results[0]

            if not _success:
                raise RuntimeError(f"Failed to run QAOA circuit after 5 attempts")

            if len(results.data.values()) > 1:
                print(type(all_pubs_isa))
                print(results.data)
                raise ValueError("Multiple data keys found in results. ")

            data_c:QiskitBitArray = list(results.data.values())[0]
            bitstrings_array = data_c.to_bool_array().astype(int)



            bitstrings_array = self.remap_bitstrings(bitstrings_array=bitstrings_array)


            #TODO(FBM): is there a reason we do the "flat_array -> counts -> flat_array" conversion here?
            # It's not large overhead for small system sizes, but this is generally not needed I think.
            # The only difference is that the array of bitstrings is somewhat sorted.
            # bitstrings_res, counts_res = get_counts_from_bit_array(bit_array=data_c)
            # mapped_array = self.remap_bitstrings(bitstrings_array=bitstrings_res)
            # bitstrings_array = np.repeat(mapped_array, counts_res, axis=0)
            t3 = time.perf_counter()

        return (job, df_job_metadata), bitstrings_array

    def run_qaoa_batch(self,
                        angles_PHASE_batch:np.ndarray,
                        angles_MIXER_batch:np.ndarray,
                        number_of_samples: int,
                        job_id_for_download: Optional[str] = None,
                        bias_parameters_ws:Optional[np.ndarray]=None
                       ):

        # TODO(FBM): ADD MEASUREMENT NOISE!

        assert len(angles_PHASE_batch.shape)==2, "angles_PHASE_batch must be a 2D array"
        assert len(angles_MIXER_batch.shape)==2, "angles_MIXER_batch must be a 2D array"


        assert angles_PHASE_batch.shape[0]==angles_MIXER_batch.shape[0], "angles_PHASE_batch and angles_MIXER_batch must have the same number of rows"
        assert angles_PHASE_batch.shape[1] == len(self.parameters_PHASE), "angles_PHASE_batch must have the same number of columns as the number of parameters in the ansatz"
        assert angles_MIXER_batch.shape[1] == len(self.parameters_MIXER), "angles_MIXER_batch must have the same number of columns as the number of parameters in the ansatz"

        if bias_parameters_ws is not None:
            assert bias_parameters_ws.shape[0]==angles_PHASE_batch.shape[0], "bias_parameters_ws must have the same number of rows as angles_PHASE_batch"
            # assert bias_parameters_ws.shape[1]==1, "bias_parameters_ws must have a single column"


        #TODO(FBM): I think we can merge this with just run_qaoa method


        batch_binding = {self.parameters_PHASE: angles_PHASE_batch,
                         self.parameters_MIXER: angles_MIXER_batch}

        if bias_parameters_ws is not None:
            batch_binding[self.parameters_BIAS_WS] = self.bias_to_theta(bias=bias_parameters_ws)



        # Get or create cached sampler
        t0 = time.perf_counter()
        sampler:SamplerRuntime|SamplerAer = self._ensure_sampler()
        t1 = time.perf_counter()



        circuit_run = self.ansatz_circuit

        all_pubs_isa = [(circuit_run, batch_binding)]
        shots_run = number_of_samples



        #
        # if job_id_for_download is None and not self._simulation:
        #     assert isinstance(self.current_session,Batch) or self.current_session is None, "For batched execution, the current session must be a Batch object."


        if self._REPLACE_WITH_RANDOM_SAMPLING:
            job = None

            t_start = time.perf_counter()
            bitstrings_arrays_all= self._numpy_rng.binomial(n=1, p=0.5, size=(angles_PHASE_batch.shape[0],number_of_samples, self._number_of_qubits))
            t_end = time.perf_counter()
            actual_runtime_wallclock = t_end - t_start

            df_job_metadata = pd.DataFrame(data={SNV.SessionId.id_long: ["RandomSampling"],
                                                 SNV.JobId.id_long: ["RandomSampling"],
                                                 'EstimatedRuntime': [None],
                                                 'ActualRuntimeQPU': [None],
                                                 'ActualRuntimeWallclock': [actual_runtime_wallclock],
                                                 })
            t2 = time.perf_counter()

            bitstrings_arrays_list = []
            for array_ind in range(angles_PHASE_batch.shape[1]):
                bitstrings_res, counts_res = np.unique(bitstrings_arrays_all[array_ind,:,:],
                                                       axis=0,
                                                       return_counts=True)
                mapped_array = self.remap_bitstrings(bitstrings_array=bitstrings_res)
                bitstrings_array = np.repeat(mapped_array, counts_res, axis=0)
                bitstrings_arrays_list.append(bitstrings_array)
            t3 = time.perf_counter()



        else:

            _service_job = None
            if job_id_for_download is not None:
                _service_job = self.current_session.service


            _success, job, results, df_job_metadata = attempt_to_run_qiskit_circuits(
                circuits_isa=all_pubs_isa,
                sampler_ibm=sampler,
                number_of_shots=shots_run,
                max_attempts_run=5,
                job_id_for_download=job_id_for_download,
                service_for_job_download=_service_job)
            t2 = time.perf_counter()

            if not _success:
                raise RuntimeError(f"Failed to run QAOA circuit after 5 attempts")


            bitstrings_arrays_list = []

            if len(results)>1:
                raise ValueError("Multiple results found in results. ")

            res_all:SamplerPubResult = results[0]

            data_all:DataBin = res_all.data
            data_all_c:BitArray = data_all.c
            #in this case, each "data_c" in BitArray corresponds to different parameters values
            for data_c in data_all_c:
                # TODO(FBM): is there a reason we do the "flat_array -> counts -> flat_array" conversion here?
                # It's not large overhead for small system sizes, but this is generally not needed I think.
                # The only difference is that the array of bitstrings is somewhat sorted.
                data_c:QiskitBitArray = data_c
                bitstrings_array = data_c.to_bool_array().astype(int)
                bitstrings_array = self.remap_bitstrings(bitstrings_array=bitstrings_array)

                # bitstrings_res, counts_res = get_counts_from_bit_array(bit_array=data_c)
                # mapped_array = self.remap_bitstrings(bitstrings_array=bitstrings_res)
                # bitstrings_array = np.repeat(mapped_array, counts_res, axis=0)
                bitstrings_arrays_list.append(bitstrings_array)



            t3 = time.perf_counter()



        return (job, df_job_metadata), bitstrings_arrays_list




    def get_ansatz_get_counts(self,
                              depth_filter_function:Optional[callable]=None):

        if depth_filter_function is None:
            def depth_filter_function(x):
                _test_1 = not getattr(x.operation, '_directive', False)
                _test_2 = x.operation.name.lower() != 'rz'
                return _test_1 and _test_2

        ansatz_circuit = self.ansatz_circuit

        gates_occurances_dict = bck_utils.count_gates_occurences_in_circuit(quantum_circuit=ansatz_circuit)

        filtered_depth = ansatz_circuit.depth(depth_filter_function)

        x_occurances = gates_occurances_dict.get('x', 0)
        sx_occurances = gates_occurances_dict.get('sx', 0) + x_occurances * 2
        cz_occurances = gates_occurances_dict.get('cz', 0)

        df_metadata_i = {}
        df_metadata_i['CircuitDepth'] = [filtered_depth]
        df_metadata_i['CZCount'] = [cz_occurances]
        df_metadata_i['SXCount'] = [sx_occurances]

        return pd.DataFrame(data=df_metadata_i)
