# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 

from typing import Union, List, Tuple, Dict, Optional, Callable, Any

from pydantic import conint
# cirq uses sympy for symbolic computation
from sympy import Number, Symbol, Mul

AngleCirq = Union[Symbol, Number, Mul, int, float]

_SUPPORTED_SDKs = []
try:
    import cirq
    from cirq import Circuit as CircuitCirq

    _SUPPORTED_SDKs.append('cirq')
except (ImportError, ModuleNotFoundError):
    CircuitCirq = None
try:
    import qiskit
    _SUPPORTED_SDKs.append('qiskit')
    from qiskit.circuit import Parameter as ParameterQiskit, ParameterExpression

    AngleQiskit = Union[ParameterQiskit, ParameterExpression, float, int]

    from qiskit import QuantumCircuit as CircuitQiskit

    from qiskit.transpiler.passmanager import StagedPassManager as PassManagerQiskit



except (ImportError, ModuleNotFoundError):
    AngleQiskit = None
    CircuitQiskit = None
    PassManagerQiskit = None
try:
    import pyquil
    from pyquil.quilatom import MemoryReference as ParameterPyquil, Mul
    from pyquil import Program as CircuitPyquil

    AnglePyquil = Union[ParameterPyquil, int, float, Mul]

    _SUPPORTED_SDKs.append('pyquil')
except (ImportError, ModuleNotFoundError):
    AnglePyquil = None

    CircuitPyquil = None

if len(_SUPPORTED_SDKs) == 0:
    raise NotImplementedError("No supported SDKs found. "
                  "Please install qiskit, pyquil, or cirq to use this submodule.")

#None is added as a workaround so my IDE doesn't mess up type hinting
AbstractCircuit = Union[CircuitQiskit, CircuitPyquil, CircuitCirq, type(None)]
AbstractAngle = Union[AngleQiskit, AngleCirq, AnglePyquil, int, float]
AbstractAngleTuple = Union[Tuple[AbstractAngle, ...], List[AbstractAngle]]

EdgeId = Tuple[conint(ge=0), conint(ge=0)]
NodeId = conint(ge=0)
QubitsTuple = Union[Tuple[NodeId, ...], List[NodeId]]
QubitsPairsTuple = Union[Tuple[Tuple[conint(ge=0), conint(ge=0)], ...], List[Tuple[conint(ge=0), conint(ge=0)]]]

DecompositionDict = Dict[str, Dict[str, List[Union[NodeId, EdgeId]]]]

GateBuilderTypeQiskit = Union[Callable[[], CircuitQiskit], Callable[[AngleQiskit, ...], CircuitQiskit]]

GateBuilderTypePyquil = Union[
    Callable[[QubitsTuple], CircuitPyquil], Callable[[QubitsTuple, AnglePyquil, ...], CircuitPyquil]]
GateBuilderTypeCirq = Union[
    Callable[[QubitsTuple], CircuitCirq], Callable[[QubitsTuple, AngleCirq, ...], CircuitCirq]]

AbstractGateBuilderType = Union[GateBuilderTypeQiskit, GateBuilderTypePyquil, GateBuilderTypeCirq]


def add_gate_qiskit(quantum_circuit: CircuitQiskit,
                    gate_builder: GateBuilderTypeQiskit,
                    qubits: QubitsTuple,
                    parameters: Optional[Tuple[AngleQiskit, ...]] = None,
                    ):
    arguments = []
    if parameters is not None:
        if isinstance(parameters, AngleQiskit):
            parameters = (parameters,)
        for angle in parameters:
            arguments.append(angle)



    return quantum_circuit.compose(gate_builder(*arguments),
                                   qubits=qubits)


def add_gate_pyquil(quantum_circuit: CircuitPyquil,
                    gate_builder: GateBuilderTypePyquil,
                    qubits: QubitsTuple,
                    parameters: Optional[Tuple[AnglePyquil, ...]] = None,):
    arguments = [qubits]
    if parameters is not None:
        if isinstance(parameters, AnglePyquil):
            parameters = (parameters,)

        for angle in parameters:
            arguments.append(angle)

    arguments = tuple(arguments)
    quantum_circuit += gate_builder(*arguments)

    return quantum_circuit


def add_gate_cirq(quantum_circuit: CircuitCirq,
                  gate_builder: GateBuilderTypeCirq,
                  qubits: QubitsTuple,
                  parameters: Optional[Tuple[AngleCirq, ...]] = None,
                  probability:Optional[float]=None):

    if probability is None:
        probability = 1.0

    arguments = [qubits]
    if parameters is not None:
        if isinstance(parameters, AngleCirq):
            parameters = (parameters,)

        for angle in parameters:
            arguments.append(angle)

    if probability is not None:
        arguments.append(probability)

    arguments = tuple(arguments)

    quantum_circuit.append([gate_builder(*arguments)])

    return quantum_circuit


def add_measurements_pyquil(quantum_circuit: CircuitPyquil,
                            qubit_indices: Tuple[int],
                            classical_indices: Tuple[int] = None) -> CircuitPyquil:
    number_of_qubits = len(qubit_indices)

    if classical_indices is None:
        classical_indices = list(range(number_of_qubits))

    from pyquil.quilatom import MemoryReference
    from pyquil.gates import MEASURE
    program = quantum_circuit.copy()
    try:
        ro_declaration = program.declarations['ro']
        ro = MemoryReference(name=ro_declaration.name,
                             declared_size=ro_declaration.memory_size)
    except(KeyError):
        ro = program.declare(name='ro', memory_type='BIT', memory_size=number_of_qubits)

    for classical_index, physical_index in zip(classical_indices, qubit_indices):
        program += MEASURE(qubit=physical_index,
                           classical_reg=ro[classical_index])
    return program


def add_measurements_qiskit(quantum_circuit: CircuitQiskit,
                            qubit_indices: Tuple[int],
                            classical_indices: Tuple[int] = None) -> CircuitQiskit:

    number_of_qubits = len(qubit_indices)
    if classical_indices is None:
        classical_indices = list(range(number_of_qubits))
    for classical_index, logical_index in zip(classical_indices, qubit_indices[::-1]):
        quantum_circuit.measure(qubit=logical_index, cbit=classical_index)

    return quantum_circuit


def add_measurements_cirq(quantum_circuit: CircuitCirq,
                          qubit_indices: Tuple[int],
                          quantum_register,
                          classical_indices: Tuple[int] = None) -> CircuitCirq:
    number_of_qubits = len(qubit_indices)

    if classical_indices is None:
        classical_indices = list(range(number_of_qubits))
    for classical_index, physical_index in zip(classical_indices, qubit_indices):
        # TODO FBM: check if cirq allows to relabel qubits?
        quantum_circuit += cirq.measure(quantum_register[physical_index])

    return quantum_circuit


def resolve_parameters_cirq(quantum_circuit: CircuitCirq,
                            memory_map: Dict[AngleCirq, float]):
    return cirq.resolve_parameters(quantum_circuit, memory_map)


def resolve_parameters_qiskit(quantum_circuit: CircuitQiskit,
                              memory_map: Dict[AngleQiskit, float]):
    # memory_map_names = {angle.name:value for angle, value in memory_map.items()}
    # print(memory_map_names)
    return quantum_circuit.assign_parameters(parameters=memory_map)


def resolve_parameters_pyquil(quantum_circuit: CircuitPyquil,
                              memory_map: Dict[AnglePyquil, float]):
    # quantum_circuit.
    # for angle, value in memory_map.items():
    #     quantum_circuit = quantum_circuit.write_memory(region_name=angle.name,
    #                                  value=value)
    # In pyquil 4.16, the memory maps are passed at runtime
    return quantum_circuit


def combine_circuits_qiskit(left_circuit: CircuitQiskit,
                            right_circuit: CircuitQiskit,
                            pass_manager:PassManagerQiskit=None) -> CircuitQiskit:

    if pass_manager is not None:
        left_circuit = pass_manager.run(left_circuit)
        right_circuit = pass_manager.run(right_circuit)


    return left_circuit.compose(other=right_circuit)


def combine_circuits_pyquil(left_circuit: CircuitPyquil,
                            right_circuit: CircuitPyquil) -> CircuitPyquil:
    return left_circuit + right_circuit


def combine_circuits_cirq(left_circuit: CircuitCirq,
                          right_circuit: CircuitCirq) -> CircuitCirq:
    return left_circuit + right_circuit


def copy_circuit_qiskit(quantum_circuit: CircuitQiskit):
    return quantum_circuit.copy()


def copy_circuit_pyquil(quantum_circuit: CircuitPyquil):
    return quantum_circuit.copy()


def copy_circuit_cirq(quantum_circuit: CircuitCirq):
    return quantum_circuit.copy()


class AbstractProgramGateBuilder:
    def __init__(
            self,
            sdk_name: str,
            native_gate_set:Optional[List[str]]=None):
        assert sdk_name.lower() in _SUPPORTED_SDKs, f"The only supported SDKs are: {_SUPPORTED_SDKs}"
        self._sdk_name = sdk_name
        self._native_gate_set = native_gate_set

        if sdk_name.lower() in ['qiskit']:
            self._circuit_composer = add_gate_qiskit
            self._measurements_handler = add_measurements_qiskit
            self._parameters_resolver = resolve_parameters_qiskit
            self._circuits_combiner = combine_circuits_qiskit
            self._circuit_copier = copy_circuit_qiskit

        elif sdk_name.lower() in ['pyquil']:
            self._circuit_composer = add_gate_pyquil
            self._measurements_handler = add_measurements_pyquil
            self._parameters_resolver = resolve_parameters_pyquil
            self._circuits_combiner = combine_circuits_pyquil
            self._circuit_copier = copy_circuit_pyquil

        elif sdk_name.lower() in ['cirq']:
            self._circuit_composer = add_gate_cirq
            self._measurements_handler = add_measurements_cirq
            self._parameters_resolver = resolve_parameters_cirq
            self._circuits_combiner = combine_circuits_cirq
            self._circuit_copier = copy_circuit_cirq
        else:
            raise NotImplementedError(f"{sdk_name} is not supported.")

    @property
    def sdk_name(self):
        return self._sdk_name
    @property
    def native_gate_set(self):
        return self._native_gate_set

    @property
    def circuit_composer(self):
        return self._circuit_composer

    @property
    def circuits_combiner(self):
        return self._circuits_combiner

    @property
    def circuit_copier(self):
        return self._circuit_copier

    @property
    def measurements_handler(self) -> Callable:
        return self._measurements_handler

    @property
    def parameters_resolver(self) -> Callable:
        return self._parameters_resolver


    def repeat_circuit(self,
                       quantum_circuit:AbstractCircuit,
                       repeats:int)->AbstractCircuit:

        repeated_circuit = self.copy_circuit(quantum_circuit=quantum_circuit)
        for _ in range(repeats-1):
            repeated_circuit = self.combine_circuits(left_circuit=repeated_circuit,
                                                     right_circuit=quantum_circuit)
        return repeated_circuit



    def get_circuit_adjoint(self,
                            quantum_circuit: AbstractCircuit):
        if self.sdk_name == 'qiskit':
            inverse_circuit:CircuitQiskit = quantum_circuit.copy()
            return inverse_circuit.inverse()
        else:
            #TODO(FBM): implement this for cirq and pyquil
            raise NotImplementedError("Adjoint is not implemented for this SDK yet.")


    def add_measurements(self,
                         quantum_circuit: AbstractCircuit,
                         qubit_indices: Tuple[int, ...],
                         classical_indices: Tuple[int, ...] = None,
                         quantum_register=None):

        kwargs = {"quantum_circuit": quantum_circuit,
                  "qubit_indices": qubit_indices,
                  "classical_indices": classical_indices}
        if self.sdk_name in ['cirq']:
            kwargs['quantum_register'] = quantum_register

        return self.measurements_handler(**kwargs)

    def resolve_parameters(self,
                           quantum_circuit: AbstractCircuit,
                           memory_map: Dict):

        quantum_circuit = quantum_circuit.copy()

        return self.parameters_resolver(quantum_circuit=quantum_circuit,
                                        memory_map=memory_map)

    def combine_circuits(self,
                         left_circuit: AbstractCircuit,
                         right_circuit: AbstractCircuit):
        return self.circuits_combiner(left_circuit=left_circuit,
                                      right_circuit=right_circuit)

    def copy_circuit(self,
                     quantum_circuit: AbstractCircuit):
        return self.circuit_copier(quantum_circuit=quantum_circuit)

    def compose_multiple_gates(self,
                               gate_builder: AbstractGateBuilderType,
                               quantum_circuit: AbstractCircuit,
                               targets_tuple: Tuple[Union[int, Tuple[int, ...]], ...],
                               angles_tuple: Optional[Tuple[AbstractAngleTuple, ...]] = None,
                               other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None,
                               #probabilities_tuple: Optional[Tuple[float, ...]] = None
                               )->AbstractCircuit:



        if angles_tuple is None:
            angles_tuple = [None for _ in range(len(targets_tuple))]


        if other_kwargs_tuple is None:
            other_kwargs_tuple = [{} for _ in range(len(targets_tuple))]

        if len(other_kwargs_tuple) == 1 and len(targets_tuple) > 1:
            other_kwargs_tuple = list(other_kwargs_tuple) * len(targets_tuple)


        if isinstance(angles_tuple, AbstractAngle):
            angles_tuple = [angles_tuple for _ in range(len(targets_tuple))]

        if len(angles_tuple)==1 and len(targets_tuple)>1:
            angles_tuple = list(angles_tuple)* len(targets_tuple)

        if isinstance(angles_tuple, list):
            angles_tuple = tuple(angles_tuple)

        # print(targets_tuple)

        # print(angles_tuple)
        # print(probabilities_tuple)
        for node_or_edge, angles, some_kwargs in zip(targets_tuple,
                                                     angles_tuple,
                                                     other_kwargs_tuple):

            if angles is not None:
                if isinstance(angles, AbstractAngle):
                    angles = (angles,)

            quantum_circuit = self.circuit_composer(quantum_circuit=quantum_circuit,
                                                    gate_builder=gate_builder,
                                                    qubits=node_or_edge,
                                                    parameters=angles,
                                                    **some_kwargs
                                                    )


        return quantum_circuit

    def H(self,
          quantum_circuit: AbstractCircuit,
          qubits_tuple: QubitsTuple,
          other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a Hadamard gate cycle."""
        return self.compose_multiple_gates(gate_builder=self._H,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def S(self,
          quantum_circuit: AbstractCircuit,
          qubits_tuple: QubitsTuple,
          other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a phase gate cycle."""

        return self.compose_multiple_gates(gate_builder=self._S,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def Sdag(self,
             quantum_circuit: AbstractCircuit,
             qubits_tuple: QubitsTuple,
             other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a phase gate cycle."""

        return self.compose_multiple_gates(gate_builder=self._Sdag,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def T(self,
          quantum_circuit: AbstractCircuit,
          qubits_tuple: QubitsTuple,
          other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a T gate cycle."""

        return self.compose_multiple_gates(gate_builder=self._T,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def exp_X(self,
              quantum_circuit: AbstractCircuit,
              angles_tuple: Tuple[Tuple[AbstractAngle], ...],
              qubits_tuple: QubitsTuple,
              other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated X operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._exp_X,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def exp_Y(self,
              quantum_circuit: AbstractCircuit,
              angles_tuple: Tuple[Tuple[AbstractAngle], ...],
              qubits_tuple: QubitsTuple,
              other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated Y operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._exp_Y,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def exp_Z(self,
              quantum_circuit: AbstractCircuit,
              angles_tuple: Tuple[Tuple[AbstractAngle], ...],
              qubits_tuple: QubitsTuple,
              other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated Z operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._exp_Z,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)



    def X(self,
          quantum_circuit: AbstractCircuit,
          qubits_tuple: QubitsTuple,
          other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a X gate cycle."""

        return self.compose_multiple_gates(gate_builder=self._X,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def Y(self,
          quantum_circuit: AbstractCircuit,
          qubits_tuple: QubitsTuple,
          other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a Y gate cycle."""

        return self.compose_multiple_gates(gate_builder=self._Y,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def Z(self,
          quantum_circuit: AbstractCircuit,
          qubits_tuple: QubitsTuple,
          other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a Y gate cycle."""

        return self.compose_multiple_gates(gate_builder=self._Z,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def I(self,
          quantum_circuit: AbstractCircuit,
          qubits_tuple: QubitsTuple,
          other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return identity gate cycle"""
        return self.compose_multiple_gates(gate_builder=self._I,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def exp_XXYY(self,
                 quantum_circuit: AbstractCircuit,
                 angles_tuple: Tuple[Tuple[AbstractAngle], ...],
                 qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
                 other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated XX+YY operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._exp_XXYY,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def exp_XX(self,
               quantum_circuit: AbstractCircuit,
               angles_tuple: Tuple[Tuple[AbstractAngle], ...],
               qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
               other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated XX operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._exp_XX,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def exp_YY(self,
               quantum_circuit: AbstractCircuit,
               angles_tuple: Tuple[Tuple[AbstractAngle], ...],
               qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
               other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated YY operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._exp_YY,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def exp_ZZ(self,
               quantum_circuit: AbstractCircuit,
               angles_tuple: Tuple[Tuple[AbstractAngle], ...],
               qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
               other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated ZZ operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._exp_ZZ,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def XX(self,
           quantum_circuit: AbstractCircuit,
           qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
           other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return XX operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._XX,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def YY(self,
           quantum_circuit: AbstractCircuit,
           qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
           other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated YY operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._YY,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def ZZ(self,
           quantum_circuit: AbstractCircuit,
           qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
           other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated ZZ operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._ZZ,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def exp_ZZ_SWAP(self,
                    quantum_circuit: AbstractCircuit,
                    angles_tuple: Tuple[Tuple[AbstractAngle], ...],
                    qubits_pairs_tuple: Union[Tuple[Tuple[int, int], ...], List[Tuple[int, int]]],
                    other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated ZZ operator cycle with SWAPs attached"""
        return self.compose_multiple_gates(gate_builder=self._exp_ZZ_SWAP,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)


    def exp_ZZXXYY(self,
                        quantum_circuit: AbstractCircuit,
                        angles_tuple: Tuple[Tuple[AbstractAngle, AbstractAngle], ...],
                        qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
                        other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        return self.compose_multiple_gates(gate_builder=self._exp_ZZXXYY,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)
    def exp_ZZXXYY_SWAP(self,
                        quantum_circuit: AbstractCircuit,
                        angles_tuple: Tuple[Tuple[AbstractAngle, AbstractAngle], ...],
                        qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
                        other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        return self.compose_multiple_gates(gate_builder=self._exp_ZZXXYY_SWAP,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def SWAP(self,
             quantum_circuit: AbstractCircuit,
             qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
             other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a SWAP operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._SWAP,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def CNOT(self,
             quantum_circuit: AbstractCircuit,
             qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
             other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a CNOT operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._CNOT,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)

    def CZ(self,
             quantum_circuit: AbstractCircuit,
             qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
             other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return a CZ operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._CZ,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)




    def RZZ(self,
               quantum_circuit: AbstractCircuit,
               angles_tuple: Tuple[Tuple[AbstractAngle], ...],
               qubits_pairs_tuple: Tuple[Tuple[int, int], ...],
               other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return RZZ operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._RZZ,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_pairs_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)








    def RZ(self,
           quantum_circuit: AbstractCircuit,
           angles_tuple: Tuple[Tuple[AbstractAngle], ...],
           qubits_tuple: QubitsTuple,
           other_kwargs_tuple: Optional[Tuple[Dict[str, Any], ...]] = None) -> AbstractCircuit:
        """Return an RZ gate cycle.
        :rtype: AbstractCircuit
        """
        return self.compose_multiple_gates(gate_builder=self._RZ,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)
    def RX(self,
              quantum_circuit: AbstractCircuit,
              angles_tuple: Tuple[Tuple[AbstractAngle], ...],
              qubits_tuple: QubitsTuple,
              other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an exponentiated X operator cycle."""

        return self.compose_multiple_gates(gate_builder=self._RX,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           angles_tuple=angles_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)


    def SX(self,
           quantum_circuit: AbstractCircuit,
           qubits_tuple: QubitsTuple,
           other_kwargs_tuple:Optional[Tuple[Dict[str, Any], ...]]=None) -> AbstractCircuit:
        """Return an SX gate cycle."""
        return self.compose_multiple_gates(gate_builder=self._SX,
                                           quantum_circuit=quantum_circuit,
                                           targets_tuple=qubits_tuple,
                                           other_kwargs_tuple=other_kwargs_tuple)




    def _H(self,
           *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _H().")

    def _X(self,
           *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _X().")

    def _Y(self,
           *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _Y().")

    def _Z(self,
           *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _Z().")

    def _I(self,
           *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _I().")

    def _S(self, *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _S().")

    def _Sdag(self, *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _Sdag().")

    def _T(self, *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _T().")

    def _exp_X(self, *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _exp_X().")

    def _exp_Y(self, *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _exp_Y().")

    def _exp_Z(self,
               *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _exp_Z().")

    def _exp_XXYY(self,
                  *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _exp_XXYY().")

    def _exp_XX(self,
                *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _exp_XX().")

    def _exp_YY(self,
                *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _exp_YY().")

    def _exp_ZZ(self,
                *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _exp_ZZ().")

    def _exp_ZZ_SWAP(self,
                     *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _exp_ZZ_SWAP().")

    def _exp_ZZXXYY(self,
                    *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _exp_ZZXXYY().")

    def _exp_ZZXXYY_SWAP(self,
                         *arguments) -> AbstractCircuit:
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented _exp_ZZXXYY_SWAP().")

    def _XX(self,
            *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _XX().")

    def _YY(self,
            *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _YY().")

    def _ZZ(self,
            *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _ZZ().")

    def _SWAP(self,
              *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _SWAP().")

    def _CNOT(self,
              *arguments) -> AbstractCircuit:
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented _SWAP().")

    #optional gates for some specific experiments
    def _CZ(self,
            *arguments) -> AbstractCircuit:
        pass

    def _RZZ(self,
            *arguments) -> AbstractCircuit:
        pass

    def _RZ(self,
            *arguments) -> AbstractCircuit:
        pass

    def _RX(self,
            *arguments) -> AbstractCircuit:
        pass

    def _SX(self,
            *arguments) -> AbstractCircuit:
        pass










            # def _u3(self,
    #         angles:Tuple[AbstractAngle,AbstractAngle,AbstractAngle],
    #         # theta: AbstractAngle,
    #         # phi: AbstractAngle,
    #         # lam: AbstractAngle,
    #         qubit: Optional[NodeId]
    #         ) -> AbstractCircuit:
    #     raise NotImplementedError(
    #         f"{self.__class__.__name__} has not implemented _u3().")
