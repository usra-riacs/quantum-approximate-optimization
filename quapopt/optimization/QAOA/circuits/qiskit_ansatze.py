# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from typing import Optional, Tuple
from pydantic import conint, confloat

import numpy as np

from quapopt import ancillary_functions as anf

from quapopt.circuits.gates import _SUPPORTED_SDKs
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType
from quapopt.optimization.QAOA.circuits import MappedAnsatzCircuit

from quapopt.circuits.gates import AbstractProgramGateBuilder
from qiskit.circuit.library import QAOAAnsatz as QiskitQAOAAnsatz
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit import QuantumCircuit, ClassicalRegister

#from quapopt.circuits.backend_utilities import repeat_circuit_qiskit

from quapopt.circuits import backend_utilities as bck_utils
from qiskit.circuit import ParameterVector


def build_qiskit_qaoa_ansatz(            #sdk_name: str,
            depth: conint(ge=0),
            hamiltonian_phase: ClassicalHamiltonian,
            time_block_size: Optional[confloat(ge=0,le=1)] = None,
            phase_separator_type=PhaseSeparatorType.QAOA,
            mixer_type=MixerType.QAOA,
            input_state:Optional[str]=None,
            add_barriers=False,
            number_of_qubits_circuit: Optional[int]=None,
            shuffling_seed: int = 42
            ):

    #TODO(FBM): incorporate the build_fractional_time_block_ansatz_qiskit function here


    # We will need two types of indices. One is abstract qubit indexing so from 0 to n-1
    # The other is device qubit indexing, which is the actual qubit indices on the device. We need to map between them

    number_of_qubits = hamiltonian_phase.number_of_qubits
    if time_block_size is None:
        time_block_size = 1.0

    initial_state = input_state
    if initial_state is not None:
        if initial_state == '|+>':
            initial_state = None
        else:
            raise NotImplementedError("Only |+> initial state is supported for now. ")

    if mixer_type == MixerType.QAOA:
        mixer_operator = None
    else:
        raise NotImplementedError("Only QAOA mixer is supported for now. ")

    assert phase_separator_type == PhaseSeparatorType.QAOA, "Only QAOA phase separator is supported for now. "

    param_name_phase = "AngPS"
    param_name_mixer = "AngMIX"

    angle_phase = ParameterVector(name=param_name_phase, length=depth) if depth > 0 else None
    angle_mixer = ParameterVector(name=param_name_mixer, length=depth) if depth > 0 else None

    if time_block_size != 1.0:
        #if depth>1:
           # raise NotImplementedError("Time block size != 1.0 is not supported for depth > 1 yet. ")

        #we split the hamiltonian interactions into sub-blocks

        # Fractional time blocking case
        hamiltonian_phase_list = hamiltonian_phase.hamiltonian.copy()

        rng_shuffling = np.random.default_rng(seed=shuffling_seed)
        rng_shuffling.shuffle(hamiltonian_phase_list)

        number_of_interactions = len(hamiltonian_phase_list)


        number_of_batches = int(np.ceil(1/time_block_size))
        number_of_interactions_per_batch = int(np.ceil(number_of_interactions/number_of_batches))

        if number_of_qubits_circuit is None:
            number_of_qubits_circuit = number_of_qubits


        ansatz_qiskit = QuantumCircuit(number_of_qubits_circuit, number_of_qubits)
        for qi in range(number_of_qubits):
            if initial_state == '|+>' or initial_state is None:
                ansatz_qiskit.h(qi)
            else:
                raise NotImplementedError("Only |+> initial state is supported for now. ")


        for param_index in range(depth):
            batch_index = param_index % number_of_batches

            hamiltonian_batch = hamiltonian_phase_list[batch_index*number_of_interactions_per_batch:
                                                            (batch_index+1)*number_of_interactions_per_batch]
            if len(hamiltonian_batch) == 0:
                continue

            #print(hamiltonian_batch, hamiltonian_phase.number_of_qubits)

            hamiltonian_batch_qiskit = bck_utils.convert_hamiltonian_list_representation_to_qiskit_observable(
                hamiltonian_list_representation=hamiltonian_batch,
                number_of_qubits=hamiltonian_phase.number_of_qubits)



            ansatz_qiskit = QiskitQAOAAnsatz(cost_operator=hamiltonian_batch_qiskit,
                                            reps=1,
                                            initial_state=ansatz_qiskit,
                                            mixer_operator=mixer_operator,
                                            name='QAOA',
                                            flatten=True)

            beta_batch = None
            gamma_batch = None
            for param in ansatz_qiskit.parameters:
                if param.name == "β[0]":
                    beta_batch = param
                if param.name == "γ[0]":
                    gamma_batch = param

            ansatz_qiskit.assign_parameters(parameters={beta_batch:angle_mixer[param_index],
                                                        gamma_batch:angle_phase[param_index]},
                                            inplace=True)



            if add_barriers:
                ansatz_qiskit.barrier()

    else:
        if add_barriers:
            raise NotImplementedError("Add barriers is not supported for now. ")
        qiskit_hamiltonian = bck_utils.convert_hamiltonian_list_representation_to_qiskit_observable(
            hamiltonian_list_representation=hamiltonian_phase.hamiltonian,
            number_of_qubits=hamiltonian_phase.number_of_qubits)

        ansatz_qiskit = QiskitQAOAAnsatz(cost_operator=qiskit_hamiltonian,
                                         reps=depth,
                                         initial_state=initial_state,
                                         mixer_operator=mixer_operator,
                                         name='QAOA',
                                         flatten=True)
        params_dict = {}
        for param in ansatz_qiskit.parameters:
            index_param = int(param.name[2:-1])
            if param.name[0:1] == "β":
                params_dict[param] = angle_mixer[index_param]
            if param.name[0:1] == "γ":
                params_dict[param] = angle_phase[index_param]
        ansatz_qiskit.assign_parameters(parameters=params_dict,
                                        inplace=True)


    if len(ansatz_qiskit.cregs)==0:
        #I don't know why, but sometimes Qiskit does not add classical registers, so we add them manually
        ansatz_qiskit.add_register(ClassicalRegister(number_of_qubits))






    return ansatz_qiskit, [angle_phase, angle_mixer]


if __name__ == '__main__':
    noq_test = 5
    ham_simple = [(ind+1, (i,j)) for ind, (i,j) in enumerate([(k,l) for k in range(noq_test) for l in range(k+1, noq_test)])]
    print(ham_simple)
    test_ansatz = build_qiskit_qaoa_ansatz(hamiltonian_phase=ClassicalHamiltonian(hamiltonian_list_representation=ham_simple,
                                                                                  number_of_qubits=noq_test),
                                            depth=1,
                                            time_block_size=1.0,
                                            phase_separator_type=PhaseSeparatorType.QAOA,
                                            mixer_type=MixerType.QAOA)


    print(test_ansatz)


    test_ansatz = build_qiskit_qaoa_ansatz(hamiltonian_phase=ClassicalHamiltonian(hamiltonian_list_representation=ham_simple,
                                                                                  number_of_qubits=noq_test),
                                            depth=2,
                                            time_block_size=0.5,
                                            phase_separator_type=PhaseSeparatorType.QAOA,
                                            mixer_type=MixerType.QAOA,
                                           add_barriers=True)
    #test_ansatz.draw('mpl')

    print(test_ansatz)


