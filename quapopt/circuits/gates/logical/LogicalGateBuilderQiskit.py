# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import Tuple

from qiskit import QuantumCircuit

from quapopt.circuits.gates import (
    AbstractCircuit,
    AbstractProgramGateBuilder,
    AngleQiskit,
)


class LogicalGateBuilderQiskit(AbstractProgramGateBuilder):
    def __init__(self):
        super().__init__(sdk_name="qiskit")

    def _H(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.h(qubit=0)
        return circuit

    def _X(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.x(qubit=0)
        return circuit

    def _Y(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.y(qubit=0)
        return circuit

    def _Z(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.z(qubit=0)
        return circuit

    def _I(self):
        circuit = QuantumCircuit(1, 1)
        circuit.id(qubit=0)
        return circuit

    def _S(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.s(qubit=0)
        return circuit

    def _Sdag(self) -> AbstractCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.sdg(qubit=0)
        return circuit

    def _T(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.t(qubit=0)
        return circuit

    def _exp_X(self, angle: AngleQiskit) -> QuantumCircuit:

        angle *= 2

        circuit = QuantumCircuit(1, 1)
        circuit.rx(theta=angle, qubit=0)
        return circuit

    def _exp_Y(self, angle: AngleQiskit) -> QuantumCircuit:
        angle *= 2
        circuit = QuantumCircuit(1, 1)
        circuit.ry(theta=angle, qubit=0)
        return circuit

    def _exp_Z(self, angle: AngleQiskit) -> QuantumCircuit:
        angle *= 2
        circuit = QuantumCircuit(1, 1)
        circuit.rz(phi=angle, qubit=0)
        return circuit

    def _exp_XX(self, angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        angle *= 2
        circuit.rxx(theta=angle, qubit1=0, qubit2=1)

        return circuit

    def _exp_YY(self, angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        angle *= 2
        circuit.ryy(theta=angle, qubit1=0, qubit2=1)

        return circuit

    def _exp_ZZ(self, angle: AngleQiskit) -> QuantumCircuit:

        circuit = QuantumCircuit(2, 2)
        angle *= 2

        circuit.rzz(theta=angle, qubit1=0, qubit2=1)

        return circuit

    def _exp_XXYY(
        self,
        angle: AngleQiskit,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        angle *= 2

        circuit.rxx(theta=angle, qubit1=0, qubit2=1)
        circuit.ryy(theta=angle, qubit1=0, qubit2=1)

        return circuit

    def _u3(self, angles_tuple: Tuple[AngleQiskit]) -> QuantumCircuit:
        theta, phi, lam = angles_tuple
        circuit = QuantumCircuit(1, 1)
        circuit.u(theta=theta, phi=phi, lam=lam, qubit=0)
        return circuit

    def _SWAP(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        circuit.swap(qubit1=0, qubit2=1)

        return circuit

    def _CNOT(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        circuit.cx(control_qubit=0, target_qubit=1)

        return circuit

    def _exp_ZZ_SWAP(self, angle: AngleQiskit) -> QuantumCircuit:
        # TODO(FBM): Possibly work out a better compilation strategy for simulations (in hardware, this depends on native gateset).
        circuit1 = self._exp_ZZ(angle=angle)
        circuit2 = self._SWAP()
        return self.combine_circuits(left_circuit=circuit1, right_circuit=circuit2)

    def _exp_ZZXXYY(
        self, angle_ZZ: AngleQiskit, angle_XY: AngleQiskit
    ) -> QuantumCircuit:
        circuit0 = self._exp_ZZ(angle=angle_ZZ)
        circuit1 = self._exp_XXYY(angle=angle_XY)
        return self.combine_circuits(left_circuit=circuit0, right_circuit=circuit1)

    def _exp_ZZXXYY_SWAP(
        self, angle_ZZ: AngleQiskit, angle_XY: AngleQiskit
    ) -> QuantumCircuit:

        # TODO(FBM): replace with phase shift

        circuit0 = self._exp_ZZXXYY(angle_ZZ=angle_ZZ, angle_XY=angle_XY)
        circuit1 = self._SWAP()
        return self.combine_circuits(left_circuit=circuit0, right_circuit=circuit1)
