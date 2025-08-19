# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 

from typing import Tuple
from quapopt.circuits.gates import (
    AnglePyquil,
    AbstractProgramGateBuilder, AbstractCircuit,
)
from pyquil import Program
from pyquil.gates import H, X, Y, Z, SWAP, S, T, ISWAP,CNOT
from pyquil.paulis import exponential_map, sZ, sX, sY


class LogicalGateBuilderPyquil(AbstractProgramGateBuilder):
    """In this case, using logical gates that require compilation via Quil-C before they can be run on QPU."""

    def __init__(self):
        super().__init__(sdk_name='pyquil')

    def _H(self,
           qubit) -> Program:
        circuit = Program()
        circuit += H(qubit)
        return circuit
    def _X(self,
           qubit) -> Program:
        circuit = Program()
        circuit += X(qubit)
        return circuit
    def _Y(self,
           qubit) -> Program:
        circuit = Program()
        circuit += Y(qubit)
        return circuit
    def _Z(self,
           qubit) -> Program:
        circuit = Program()
        circuit += Z(qubit)
        return circuit

    def _S(self,
           qubit) -> Program:
        circuit = Program()
        circuit += S(qubit)
        return circuit
    def _Sdag(self,
              qubit) -> Program:
          circuit = Program()
          circuit += S(qubit).dagger()
          return circuit

    def _T(self,
           qubit) -> Program:
        circuit = Program()
        circuit += T(qubit)
        return circuit
    def _exp_X(self,
               qubit: int,
               angle: AnglePyquil,
               ) -> Program:

        circuit = Program()
        pauli_term = sX(q=qubit)
        circuit += exponential_map(term=pauli_term)(angle)
        return circuit

    def _exp_Y(self,
               qubit: int,
               angle: AnglePyquil,
               ) -> Program:
        circuit = Program()
        pauli_term = sY(q=qubit)
        circuit += exponential_map(term=pauli_term)(angle)
        return circuit
    def _exp_Z(self,
               qubit: int,
               angle: AnglePyquil,
               ) -> Program:

        circuit = Program()
        pauli_term = sZ(q=qubit)
        circuit += exponential_map(term=pauli_term)(angle)
        return circuit

    def _exp_XX(self,
                qubits_pair: Tuple[int, int],
                angle: AnglePyquil,
                ) -> Program:
        qi, qj = qubits_pair

        circuit = Program()
        pauli_term = sX(q=qi) * sX(q=qj)
        circuit += exponential_map(term=pauli_term)(angle)
        return circuit

    def _exp_YY(self,
                qubits_pair: Tuple[int, int],
                angle: AnglePyquil) -> Program:
        qi, qj = qubits_pair

        circuit = Program()
        pauli_term = sY(q=qi) * sY(q=qj)
        circuit += exponential_map(term=pauli_term)(angle)
        return circuit
    def _exp_ZZ(self,
                qubits_pair: Tuple[int, int],
                angle: AnglePyquil) -> Program:
        qi, qj = qubits_pair

        circuit = Program()
        pauli_term = sZ(q=qi)*sZ(q=qj)
        circuit += exponential_map(term=pauli_term)(angle)
        return circuit

    def _exp_ZZ_SWAP(self,
                     qubits_pair,
                     angle: AnglePyquil,
                     ) -> Program:
        circuit_0 = self._exp_ZZ(qubits_pair=qubits_pair,
                                 angle=angle)
        circuit_1 = self._SWAP(qubits_pair=qubits_pair)
        return self.combine_circuits(left_circuit=circuit_0,
                                     right_circuit=circuit_1)

    def _exp_XXYY(self,
                  qubits_pair: Tuple[int, int],
                  angle: AnglePyquil,
                  ) -> Program:
        qi, qj = qubits_pair

        circuit = Program()
        pauli_term1 = sX(q=qi) * sX(q=qj)
        circuit += exponential_map(term=pauli_term1)(angle)

        pauli_term2 = sY(q=qi) * sY(q=qj)
        circuit += exponential_map(term=pauli_term2)(angle)
        return circuit


    def _exp_ZZXXYY(self,
                    qubits_pair:Tuple[int,int],
                    angle_ZZ:AnglePyquil,
                    angle_XY:AnglePyquil) -> AbstractCircuit:

        circuit_0 = self._exp_ZZ(qubits_pair=qubits_pair,
                                 angle=angle_ZZ)
        circuit_1 = self._exp_XXYY(qubits_pair=qubits_pair,
                                   angle=angle_XY)
        return self.combine_circuits(left_circuit=circuit_0,
                                     right_circuit=circuit_1
                                        )

    def _exp_ZZXXYY_SWAP(self,
                         qubits_pair: Tuple[int, int],
                         angle_ZZ: AnglePyquil,
                         angle_XY: AnglePyquil) -> AbstractCircuit:
        circuit_0 = self._exp_ZZXXYY(qubits_pair=qubits_pair,
                                     angle_ZZ=angle_ZZ,
                                     angle_XY=angle_XY)
        circuit_1 = self._SWAP(qubits_pair=qubits_pair)
        return self.combine_circuits(left_circuit=circuit_0,
                                     right_circuit=circuit_1)


    def _SWAP(self,
              qubits_pair: Tuple[int, int]) -> Program:
        qi, qj = qubits_pair
        circuit = Program()
        circuit += SWAP(q1=qi, q2=qj)

        return circuit
    def _ISWAP(self,
              qubits_pair: Tuple[int, int]) -> Program:
        qi, qj = qubits_pair
        circuit = Program()
        circuit += ISWAP(q1=qi, q2=qj)

        return circuit
    def _CNOT(self,
              qubits_pair: Tuple[int, int]) -> Program:
        qi, qj = qubits_pair
        circuit = Program()
        circuit += CNOT(control=qi, target=qj)

        return circuit
