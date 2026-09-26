# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from quapopt.circuits.backend_utilities.qiskit.check_backends import AVAILABLE_SIMULATORS_QISKIT

if 'aer-gpu' in AVAILABLE_SIMULATORS_QISKIT:
    DEFAULT_DEVICE_SIMULATOR_QISKIT = 'GPU'
else:
    DEFAULT_DEVICE_SIMULATOR_QISKIT = 'CPU'


DEFAULT_SIMULATED_SAMPLER_KWARGS =  {'device': DEFAULT_DEVICE_SIMULATOR_QISKIT,
                                     'method':'statevector',
                                      'precision': 'double',
                                      'seed_simulator': 42,
                                      'cuStateVec_enable': True,
                                      'max_job_size': None,
                                      'max_shot_size': None,
                                      'enable_truncation': True,
                                      'zero_threshold': 10 ** (-10),
                                      'validation_threshold': 10 ** (-8),
                                      'max_parallel_threads': 15,
                                      'max_parallel_experiments': 1,
                                     'max_parallel_shots': 0,
                                     'max_memory_mb': 0,
                                     'blocking_enable': False,
                                     'blocking_qubits': 0,
                                     'chunk_swap_buffer_qubits': 15,
                                     'batched_shots_gpu': False,
                                     'batched_shots_gpu_max_qubits': 16,
                                     'num_threads_per_device': 1,
                                     'shot_branching_enable': False,
                                     'shot_branching_sampling_enable': False,
                                     'accept_distributed_results': None,
                                     'runtime_parameter_bind_enable': False,
                                     }

DEFAULT_MPS_OPTIONS = {
                  'matrix_product_state_max_bond_dimension': None,
                  'matrix_product_state_truncation_threshold': 1e-16,
                  'mps_sample_measure_algorithm': 'mps_apply_measure',
                  'mps_log_data': True,
                  'mps_swap_direction': 'mps_swap_left',
                  'chop_threshold': 1e-8,
                  'mps_parallel_threshold': 14,
                  'mps_omp_threads': 1,
                  'mps_lapack': False,
}

DEFAULT_SIMULATED_SAMPLER_KWARGS = {**DEFAULT_SIMULATED_SAMPLER_KWARGS,
                                    **DEFAULT_MPS_OPTIONS}


DEFAULT_QPU_EM_KWARGS= {'dynamical_decoupling': dict(enable=False),
                        'twirling': dict(enable_gates=False,
                                         enable_measure=False),
                        'execution': dict(init_qubits=True,
                                          rep_delay=0.00005
                                          )}

DEFAULT_QPU_JOB_KWARGS = {'max_execution_time': 10**15,}

DEFAULT_QPU_SAMPLER_KWARGS = {**DEFAULT_QPU_EM_KWARGS,
                              **DEFAULT_QPU_JOB_KWARGS}


DEFAULT_QPU_BACKEND_KWARGS = {'use_fractional_gates':False}
DEFAULT_SIMULATOR_BACKEND_KWARGS = {**{'device': DEFAULT_DEVICE_SIMULATOR_QISKIT,
                                     'method':'statevector'},
                                    **DEFAULT_SIMULATED_SAMPLER_KWARGS.copy()}


REAL_DEVICES_IBM = [
                #Heron r1
                'ibm_torino',
                # Heron r2
                'ibm_kingston',
                'ibm_fez',
                'ibm_marrakesh',
                #Heron r3
                'ibm_pittsburgh',
                'ibm_boston',
]

HERON_DEVICES_IBM = ['ibm_torino',
                     'ibm_kingston', 'ibm_fez', 'ibm_marrakesh',
                     'ibm_pittsburgh', 'ibm_boston']


SIMULATOR_NAMES_IBM = ['aer']
