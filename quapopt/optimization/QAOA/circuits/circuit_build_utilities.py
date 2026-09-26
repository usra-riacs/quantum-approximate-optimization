# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType, PhaseSeparatorType, MixerType, InitialStateType
from quapopt.circuits.gates import AbstractProgramGateBuilder, AbstractAngle
from quapopt.circuits.gates import _SUPPORTED_SDKs, AbstractCircuit, AbstractAngle
from typing import List, Optional, Tuple
import numpy as np



def prepare_parametrized_circuit(
        sdk_name:str,
        number_of_qubits:int,
        qubit_ids_device:List[int]|Tuple[int,...],
        depth:int,
        mixer_type:MixerType,
        initial_state:InitialStateType,
        pre_created_bias_parameter=None,
                                 ):
    """
    Create a parametrized quantum circuit with phase, mixer, and optional WS bias parameters.

    Parameters
    ----------
    pre_created_bias_parameter : Parameter, optional
        Pre-created bias parameter to use for WS-QAOA. If provided, this parameter
        will be used instead of creating a new one. This is essential for time-block
        ansatz where all batches must share the same bias parameter object to avoid
        Qiskit parameter name conflicts during circuit composition.
    """

    param_name_phase = "AngPS"
    param_name_mixer = "AngMIX"
    param_name_bias = "AngBiasWS"

    angle_bias_WS = None

    if sdk_name.lower() in ['qiskit']:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector, Parameter

        number_of_qubits_physical = max(qubit_ids_device) + 1

        quantum_circuit = QuantumCircuit(number_of_qubits_physical, number_of_qubits)

        angle_phase = ParameterVector(name=param_name_phase, length=depth)
        angle_mixer = ParameterVector(name=param_name_mixer, length=depth)

        if mixer_type == MixerType.ws_qaoa_general or initial_state == InitialStateType.ws_qaoa_general:
            # Use pre-created parameter if provided (for time-block ansatz to avoid name conflicts)
            if pre_created_bias_parameter is not None:
                angle_bias_WS = pre_created_bias_parameter
            else:
                angle_bias_WS = ParameterVector(name=param_name_bias,
                                                 length=len(qubit_ids_device))

        elif mixer_type == MixerType.ws_qaoa_identical or initial_state == InitialStateType.ws_qaoa_identical:
            # Use pre-created parameter if provided (for time-block ansatz to avoid name conflicts)
            if pre_created_bias_parameter is not None:
                angle_bias_WS = pre_created_bias_parameter
            else:
                angle_bias_WS = Parameter(name=param_name_bias)




    elif sdk_name.lower() in ['pyquil']:
        from pyquil import Program
        quantum_circuit = Program()
        angle_phase = quantum_circuit.declare(param_name_phase, "REAL", depth)
        angle_mixer = quantum_circuit.declare(param_name_mixer, "REAL", depth)

    elif sdk_name.lower() in ['cirq']:

        from cirq import Circuit
        import sympy
        quantum_circuit = Circuit()
        angle_phase = [sympy.Symbol(name=f"{param_name_phase}-{i}") for i in range(depth)]
        angle_mixer = [sympy.Symbol(name=f"{param_name_mixer}-{i}") for i in range(depth)]

    else:
        raise AssertionError((f"Unsupported SDK: {sdk_name}. "
                              f"Please choose one of the following: {_SUPPORTED_SDKs}"))


    return quantum_circuit, (angle_phase, angle_mixer, angle_bias_WS)


def build_initial_state_QAOA(program_gate_builder:AbstractProgramGateBuilder,
                             quantum_circuit:AbstractCircuit,
                             qubit_ids_device:List[int]|Tuple[int,...],
                             initial_state:InitialStateType|AbstractCircuit,
                             bias_angles_WS:Optional[AbstractAngle|List[AbstractAngle]]=None,
                             ):


    from qiskit.circuit import Parameter as ParameterQiskit, ParameterExpression, ParameterVector


    # TODO(FBM): abstract this away
    if initial_state == InitialStateType.QAOA:
        quantum_circuit = program_gate_builder.H(quantum_circuit=quantum_circuit,
                                                 qubits_tuple=qubit_ids_device)
        # print('hejka:','adding H')
    elif initial_state == InitialStateType.zero:
        pass
    elif initial_state == InitialStateType.one:
        quantum_circuit = program_gate_builder.X(quantum_circuit=quantum_circuit,
                                                 qubits_tuple=qubit_ids_device)


    elif initial_state in [InitialStateType.ws_qaoa_identical,
                           InitialStateType.ws_qaoa_identical_opposite,
                           InitialStateType.ws_qaoa_general]:

        if isinstance(bias_angles_WS, (ParameterQiskit, float, int)):
            bias_angles_WS = [bias_angles_WS] * (len(qubit_ids_device))
        # else:

        assert len(bias_angles_WS) == len(qubit_ids_device), (
            f"Number of bias angles must match number of qubits. "
            f"Got {len(bias_angles_WS)} bias angles for {len(qubit_ids_device)} qubits."
        )

        quantum_circuit = program_gate_builder.RY(quantum_circuit=quantum_circuit,
                                                  angles_tuple=bias_angles_WS,
                                                  qubits_tuple=qubit_ids_device
                                                  )



    elif isinstance(initial_state, AbstractCircuit):
        quantum_circuit = program_gate_builder.combine_circuits(left_circuit=quantum_circuit,
                                                                right_circuit=initial_state)

    else:
        raise ValueError(
            f"Unsupported input state: {initial_state} of type: {type(quantum_circuit)}. ",
            f"Supported types: {InitialStateType} or AbstractCircuit")

    return quantum_circuit

def build_mixer_layer_QAOA(program_gate_builder:AbstractProgramGateBuilder,
                           quantum_circuit:AbstractCircuit,
                           list_of_qubits: List[int] | Tuple[int,...],
                           beta:AbstractAngle,
                           mixer_type:MixerType,
                           bias_angles_WS: Optional[AbstractAngle | List[AbstractAngle]] = None,
                           ):

    if mixer_type in [MixerType.QAOA]:
        # (3) one-qubit mixing operators
        quantum_circuit = program_gate_builder.exp_X(quantum_circuit=quantum_circuit,
                                                     angles_tuple=beta,
                                                     qubits_tuple=list_of_qubits)
    elif mixer_type in [MixerType.QAMPA]:
        pass


    elif mixer_type in [MixerType.ws_qaoa_identical,
                        MixerType.ws_qaoa_identical_opposite,
                        MixerType.ws_qaoa_general
                        ]:
        if isinstance(bias_angles_WS, (AbstractAngle, float, int)):
            bias_angles_WS = [bias_angles_WS] * (len(list_of_qubits))


        assert len(bias_angles_WS) == len(list_of_qubits), (
            f"Number of bias angles must match number of qubits. "
            f"Got {len(bias_angles_WS)} bias angles for {len(list_of_qubits)} qubits."
        )
        if mixer_type in [MixerType.ws_qaoa_identical_opposite]:
            angles_tuple = [(beta, np.pi-_c) for _c in bias_angles_WS]
        else:
            angles_tuple = [(beta, _c) for _c in bias_angles_WS]

        # print("hejka", angles_tuple)
        # print(quantum_circuit)
        quantum_circuit = program_gate_builder.WS_QAOA_MIXER(quantum_circuit=quantum_circuit,
                                                              angles_tuple=angles_tuple,
                                                              qubits_tuple=list_of_qubits)



    else:
        raise ValueError(f"Unsupported Mixer Type: {mixer_type}")


    return quantum_circuit


def build_phase_separator_layer_QAOA_2q(program_gate_builder:AbstractProgramGateBuilder,
                                        quantum_circuit:AbstractCircuit,
                                        # qubit_ids_device:List[int]|Tuple[int,...],
                                        gamma:AbstractAngle,
                                        phase_separator_type:PhaseSeparatorType,
                                        coefficients_list:List[float],
                                        edges_list: List[Tuple[int,int]] | Tuple[Tuple[int,int],...],
                                        with_swap_network:bool,
                                        beta:Optional[AbstractAngle]=None,
                                        ):

    if phase_separator_type == PhaseSeparatorType.QAMPA:
        assert beta is not None, "beta must be specified for QAMPA"

    for coeff, edge_device in zip(coefficients_list, edges_list):
        if len(edge_device)!=2:
            raise ValueError(f"edge_device must be a pair of qubits, got: {edge_device}")

        if with_swap_network:
            if phase_separator_type in [PhaseSeparatorType.QAOA]:
                # TODO(FBM): for time-block ansatz with non-fully-connected graphs,
                # 0-valued coeffs can cause some parameters gamma to be irrelevant,
                # we should be exploiting this
                quantum_circuit = program_gate_builder.exp_ZZ_SWAP(quantum_circuit=quantum_circuit,
                                                                   angles_tuple=(gamma * coeff,),
                                                                   qubits_pairs_tuple=[edge_device]
                                                                   )
            elif phase_separator_type in [PhaseSeparatorType.QAMPA]:
                quantum_circuit = program_gate_builder.exp_ZZXXYY_SWAP(quantum_circuit=quantum_circuit,
                                                                       angles_tuple=((gamma * coeff, beta)),
                                                                       qubits_pairs_tuple=(edge_device,)
                                                                       )
            else:
                raise ValueError(f"Unsupported Phase Separator Type: {phase_separator_type}")


        else:
            if phase_separator_type in [PhaseSeparatorType.QAOA]:
                quantum_circuit = program_gate_builder.exp_ZZ(quantum_circuit=quantum_circuit,
                                                              angles_tuple=(gamma * coeff,),
                                                              qubits_pairs_tuple=[edge_device]
                                                              )

            elif phase_separator_type in [PhaseSeparatorType.QAMPA]:
                quantum_circuit = program_gate_builder.exp_ZZXXYY(quantum_circuit=quantum_circuit,
                                                                  angles_tuple=(
                                                                      (gamma * coeff, beta)),
                                                                  qubits_pairs_tuple=(edge_device,)
                                                                  )

            else:
                raise ValueError(f"Unsupported Phase Separator Type: {phase_separator_type}")




    return quantum_circuit

