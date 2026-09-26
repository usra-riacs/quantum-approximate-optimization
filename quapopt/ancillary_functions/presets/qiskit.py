# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
from quapopt.circuits.backend_utilities.qiskit.qiskit_config import DEFAULT_QPU_SAMPLER_KWARGS, DEFAULT_SIMULATED_SAMPLER_KWARGS, DEFAULT_SIMULATOR_BACKEND_KWARGS, REAL_DEVICES_IBM


from qiskit_aer.noise import (
            NoiseModel,
            QuantumError,
            ReadoutError,
            depolarizing_error,
            pauli_error,
            thermal_relaxation_error,
        amplitude_damping_error,
)
def create_uncorrelated_noise_model_amplitude_damping(number_of_qubits,
                                                     amplitude_damping_probability:float,
                                                     basis_gates_1q=('x','sx','rz','id'),
                                                     basis_gates_2q=('cz',),
                                                     both_qubit_orders:bool=True,
                                                     two_qubit_damping:bool=True):
    """Uncorrelated amplitude-damping noise model with the same damping probability
    on every gate.

    :param both_qubit_orders: register the two-qubit error on both (i, j) and (j, i).
        Aer keys a two-qubit error by the ORDERED qubit tuple of the instruction, so an
        error registered only on [i, j] leaves every cz(j, i) noiseless.
    :param two_qubit_damping: register the two-qubit error at all. Set False for a model
        where only the one-qubit gates damp.
    """
    noise_model = NoiseModel(basis_gates=list(basis_gates_1q) + list(basis_gates_2q))

    amplitude_damping_error_1q = amplitude_damping_error(param_amp=amplitude_damping_probability)

    noise_model.add_all_qubit_quantum_error(amplitude_damping_error_1q,
                                            instructions=basis_gates_1q)

    if two_qubit_damping:
        amplitude_damping_error_2q = amplitude_damping_error_1q.tensor(amplitude_damping_error_1q)

        for i in range(number_of_qubits):
            for j in range(i+1,number_of_qubits):
                qubit_orders = [[i,j],[j,i]] if both_qubit_orders else [[i,j]]
                for qubits in qubit_orders:
                    noise_model.add_quantum_error(error=amplitude_damping_error_2q,
                                                  instructions=basis_gates_2q,
                                                  qubits=qubits
                    )

    return noise_model
