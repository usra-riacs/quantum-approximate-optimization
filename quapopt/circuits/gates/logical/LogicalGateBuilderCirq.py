# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import numpy as np
import sympy
from cirq import (
    CNOT,
    ISWAP,
    SWAP,
    XX,
    YY,
    ZZ,
    Circuit,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
    ZPowGate,
    protocols,
)
from pydantic import confloat

from quapopt.circuits.gates import AbstractProgramGateBuilder, AngleCirq


def _pi(rads):
    return sympy.pi if protocols.is_parameterized(rads) else np.pi


def _angle_to_exponent_function_1q(angle):
    return 2 * angle / _pi(rads=angle)


def _angle_to_exponent_function_2q(angle):
    return 2 * angle / _pi(rads=angle)


# def _probability_handler()


class LogicalGateBuilderCirq(AbstractProgramGateBuilder):

    def __init__(self, quantum_register):
        self._quantum_register = quantum_register
        super().__init__(sdk_name="cirq")

    @property
    def quantum_register(self):
        return self._quantum_register

    def _add_probabilistic_gate(
        self, gate, qubit_or_qubits, gate_probability=None, angle=None
    ):
        if gate_probability is None:
            gate_probability = 1.0

        if isinstance(qubit_or_qubits, int):
            qubit_or_qubits = (self._quantum_register[qubit_or_qubits],)

        if isinstance(qubit_or_qubits, list):
            qubit_or_qubits = tuple(qubit_or_qubits)

        if isinstance(qubit_or_qubits, tuple):
            if isinstance(qubit_or_qubits[0], int):
                len(self._quantum_register)
                qubit_or_qubits = tuple(
                    [self._quantum_register[qubit] for qubit in qubit_or_qubits]
                )

        circuit = Circuit()

        if gate_probability == 0:
            return circuit

        gate_to_add = gate.on(*qubit_or_qubits)
        if angle is not None:
            gate_to_add = gate_to_add**angle

        if gate_probability != 1:
            gate_to_add = gate_to_add.with_probability(probability=gate_probability)

        circuit += gate_to_add
        return circuit

    def _H(self, qubit, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        return self._add_probabilistic_gate(
            gate=H, qubit_or_qubits=qubit, gate_probability=gate_probability
        )

    def _T(self, qubit, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        return self._add_probabilistic_gate(
            gate=T, qubit_or_qubits=qubit, gate_probability=gate_probability
        )

    def _S(self, qubit, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        return self._add_probabilistic_gate(
            gate=S, qubit_or_qubits=qubit, gate_probability=gate_probability
        )

    def _Sdag(self, qubit, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:
        return self._add_probabilistic_gate(
            gate=ZPowGate(exponent=-0.5),
            qubit_or_qubits=qubit,
            gate_probability=gate_probability,
        )

    def _exp_X(
        self, qubit, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        if angle is not None:
            angle = _angle_to_exponent_function_1q(angle=angle)

        #

        return self._add_probabilistic_gate(
            gate=X,
            qubit_or_qubits=qubit,
            gate_probability=gate_probability,
            angle=angle,
        )

    def _exp_Y(
        self, qubit, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:
        if angle is not None:
            angle = _angle_to_exponent_function_1q(angle=angle)

        return self._add_probabilistic_gate(
            gate=Y,
            qubit_or_qubits=qubit,
            gate_probability=gate_probability,
            angle=angle,
        )

    def _exp_Z(
        self, qubit, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:
        if angle is not None:
            angle = _angle_to_exponent_function_1q(angle=angle)

        return self._add_probabilistic_gate(
            gate=Z,
            qubit_or_qubits=qubit,
            gate_probability=gate_probability,
            angle=angle,
        )

    def _I(self, qubit, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:
        return self._add_probabilistic_gate(
            gate=I, qubit_or_qubits=qubit, gate_probability=gate_probability
        )

    def _X(
        self,
        qubit,
        gate_probability: confloat(ge=0.0, le=1.0),
    ) -> Circuit:

        return self._exp_X(qubit=qubit, gate_probability=gate_probability, angle=None)

    def _Y(
        self,
        qubit,
        gate_probability: confloat(ge=0.0, le=1.0),
    ) -> Circuit:

        return self._exp_Y(qubit=qubit, gate_probability=gate_probability, angle=None)

    def _Z(
        self,
        qubit,
        gate_probability: confloat(ge=0.0, le=1.0),
    ) -> Circuit:

        return self._exp_Z(qubit=qubit, gate_probability=gate_probability, angle=None)

    def _exp_XX(
        self, qubits_pair, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:
        if angle is not None:
            angle = _angle_to_exponent_function_2q(angle=angle)

        return self._add_probabilistic_gate(
            gate=XX,
            qubit_or_qubits=qubits_pair,
            gate_probability=gate_probability,
            angle=angle,
        )

    def _exp_YY(
        self, qubits_pair, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        if angle is not None:
            angle = _angle_to_exponent_function_2q(angle=angle)

        return self._add_probabilistic_gate(
            gate=YY,
            qubit_or_qubits=qubits_pair,
            gate_probability=gate_probability,
            angle=angle,
        )

    def _exp_ZZ(
        self, qubits_pair, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        if angle is not None:
            angle = _angle_to_exponent_function_2q(angle=angle)

        return self._add_probabilistic_gate(
            gate=ZZ,
            qubit_or_qubits=qubits_pair,
            gate_probability=gate_probability,
            angle=angle,
        )

    def _exp_ZZ_SWAP(
        self, qubits_pair, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        if angle is not None:
            angle = _angle_to_exponent_function_2q(angle=angle)

        # TODO(FBM): Possibly work out a better compilation strategy for simulations
        # (in hardware, this depends on native gateset).

        circuit_0 = self._add_probabilistic_gate(
            gate=ZZ,
            qubit_or_qubits=qubits_pair,
            gate_probability=gate_probability,
            angle=angle,
        )
        circuit_1 = self._add_probabilistic_gate(
            gate=SWAP, qubit_or_qubits=qubits_pair, gate_probability=gate_probability
        )
        return self.combine_circuits(circuit_0, circuit_1)

    def _exp_ZZXXYY(
        self,
        qubits_pair,
        angle_ZZ: AngleCirq,
        angle_XY: AngleCirq,
        gate_probability: confloat(ge=0.0, le=1.0),
    ) -> Circuit:
        circuit0 = self._exp_ZZ(
            qubits_pair=qubits_pair, angle=angle_ZZ, gate_probability=gate_probability
        )
        circuit1 = self._exp_XXYY(
            qubits_pair=qubits_pair, angle=angle_XY, gate_probability=gate_probability
        )

        return self.combine_circuits(circuit0, circuit1)

    def _exp_ZZXXYY_SWAP(
        self,
        qubits_pair,
        angle_ZZ: AngleCirq,
        angle_XY: AngleCirq,
        gate_probability: confloat(ge=0.0, le=1.0),
    ) -> Circuit:

        # TODO(FBM): replace with phase shift

        circuit0 = self._exp_ZZXXYY(
            qubits_pair=qubits_pair,
            angle_ZZ=angle_ZZ,
            angle_XY=angle_XY,
            gate_probability=gate_probability,
        )
        circuit1 = self._SWAP(
            qubits_pair=qubits_pair, gate_probability=gate_probability
        )

        return self.combine_circuits(left_circuit=circuit0, right_circuit=circuit1)

    def _exp_XXYY(
        self,
        qubits_pair,
        angle: AngleCirq,
        gate_probability: confloat(ge=0.0, le=1.0) = 1.0,
    ) -> Circuit:

        if angle is not None:
            angle = _angle_to_exponent_function_2q(angle=angle)
            # additional factor for ISWAP
            angle *= -2

        return self._add_probabilistic_gate(
            gate=ISWAP,
            qubit_or_qubits=qubits_pair,
            gate_probability=gate_probability,
            angle=angle,
        )

    def _XX(self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        return self._add_probabilistic_gate(
            gate=XX,
            qubit_or_qubits=qubits_pair,
            gate_probability=gate_probability,
            angle=None,
        )

    def _YY(self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        return self._add_probabilistic_gate(
            gate=YY,
            qubit_or_qubits=qubits_pair,
            gate_probability=gate_probability,
            angle=None,
        )

    def _ZZ(self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        return self._add_probabilistic_gate(
            gate=ZZ,
            qubit_or_qubits=qubits_pair,
            gate_probability=gate_probability,
            angle=None,
        )

    def _SWAP(self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        return self._add_probabilistic_gate(
            gate=SWAP, qubit_or_qubits=qubits_pair, gate_probability=gate_probability
        )

    def _ISWAP(
        self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        return self._add_probabilistic_gate(
            gate=ISWAP, qubit_or_qubits=qubits_pair, gate_probability=gate_probability
        )

    def _CNOT(self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:
        return self._add_probabilistic_gate(
            gate=CNOT, qubit_or_qubits=qubits_pair, gate_probability=gate_probability
        )
