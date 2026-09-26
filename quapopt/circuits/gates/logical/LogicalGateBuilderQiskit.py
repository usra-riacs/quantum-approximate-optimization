# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import Tuple, Callable

import numpy as np

from quapopt.circuits.gates import (
    AngleQiskit,
    AbstractProgramGateBuilder, AbstractCircuit,
)

from qiskit import QuantumCircuit

class LogicalGateBuilderQiskit(AbstractProgramGateBuilder):
    def __init__(self):
        super().__init__(sdk_name='qiskit')


    def _H(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.h(qubit=0)
        #circuit.delay()
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


    def _exp_X(self,
               angle: AngleQiskit) -> QuantumCircuit:

        angle *= 2

        circuit = QuantumCircuit(1, 1)
        circuit.rx(theta=angle, qubit=0)
        return circuit

    def _exp_Y(self,
               angle: AngleQiskit) -> QuantumCircuit:
        angle *= 2
        circuit = QuantumCircuit(1, 1)
        circuit.ry(theta=angle, qubit=0)
        return circuit

    def _exp_Z(self,
               angle: AngleQiskit) -> QuantumCircuit:
        angle *= 2
        circuit = QuantumCircuit(1, 1)
        circuit.rz(phi=angle, qubit=0)
        return circuit

    def _RX(self,
            angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.rx(theta=angle, qubit=0)
        return circuit


    def _RY(self,
            angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.ry(theta=angle, qubit=0)
        return circuit

    def _RZ(self,
            angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        circuit.rz(phi=angle, qubit=0)
        return circuit

    def _RZZ(self,
             angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        circuit.rzz(theta=angle, qubit1=0, qubit2=1)
        return circuit

    def _SX(self):
        circuit = QuantumCircuit(1, 1)
        circuit.sx(qubit=0)
        return circuit

    def _exp_XX(self,
                angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        angle *= 2
        circuit.rxx(theta=angle, qubit1=0, qubit2=1)

        return circuit

    def _exp_YY(self,
                angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        angle *= 2
        circuit.ryy(theta=angle, qubit1=0, qubit2=1)

        return circuit

    def _exp_ZZ(self,
                angle: AngleQiskit) -> QuantumCircuit:

        circuit = QuantumCircuit(2, 2)
        angle *= 2
        #print("HEJKA:",angle)

        circuit.rzz(theta=angle, qubit1=0, qubit2=1)

        return circuit

    def _exp_XXYY(self,
                  angle: AngleQiskit,
                  ) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        angle *= 2

        circuit.rxx(theta=angle, qubit1=0, qubit2=1)
        circuit.ryy(theta=angle, qubit1=0, qubit2=1)

        return circuit

    def _u3(self,
            angles_tuple: Tuple[AngleQiskit])->QuantumCircuit:
        theta, phi, lam = angles_tuple
        circuit = QuantumCircuit(1, 1)
        circuit.u(theta=theta,
                  phi=phi,
                  lam=lam,
                  qubit=0)
        return circuit

    def _SWAP(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        circuit.swap(qubit1=0, qubit2=1)

        return circuit
    def _CNOT(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        circuit.cx(control_qubit=0, target_qubit=1)

        return circuit
    def _CZ(self)->QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        circuit.cz(control_qubit=0, target_qubit=1)
        return circuit


    def _exp_ZZ_SWAP(self,
                     angle: AngleQiskit) -> QuantumCircuit:
        #circuit = QuantumCircuit(2, 2)
        #TODO(FBM): Possibly work out a better compilation strategy for simulations (in hardware, this depends on native gateset).
        circuit1 = self._exp_ZZ(angle=angle)
        circuit2 = self._SWAP()
        return self.combine_circuits(left_circuit=circuit1,
                                     right_circuit=circuit2)


    def _exp_ZZXXYY(self,
                         angle_ZZ: AngleQiskit,
                         angle_XY:AngleQiskit) -> QuantumCircuit:
        circuit0 = self._exp_ZZ(angle=angle_ZZ)
        circuit1 = self._exp_XXYY(angle=angle_XY)
        return self.combine_circuits(left_circuit=circuit0,
                                     right_circuit=circuit1)




    def _exp_ZZXXYY_SWAP(self,
                         angle_ZZ: AngleQiskit,
                         angle_XY:AngleQiskit) -> QuantumCircuit:

        #TODO(FBM): replace with phase shift

        circuit0 = self._exp_ZZXXYY(angle_ZZ=angle_ZZ,
                                    angle_XY=angle_XY)
        circuit1 = self._SWAP()
        return self.combine_circuits(left_circuit=circuit0,
                                     right_circuit=circuit1)

    def _WS_QAOA_MIXER(self,
                       angle_mixer: AngleQiskit | float | int,
                       angle_bias: AngleQiskit | float | int) -> QuantumCircuit:
        # U_M(beta) = exp(-i*beta*H_M), where H_M = sin(theta)*X + cos(theta)*Z
        # and theta = angle_bias. Since RY(theta)*Z*RY(-theta) = cos(theta)*Z + sin(theta)*X,
        # we have U_M(beta) = RY(theta) * RZ(2*beta) * RY(-theta).
        circuit = QuantumCircuit(1, 1)
        circuit.ry(theta=-angle_bias, qubit=0)
        circuit.rz(phi=2 * angle_mixer, qubit=0)
        circuit.ry(theta=angle_bias, qubit=0)
        return circuit

    def _SPECIAL_GATES_1Q(self,
                          gate_name: str) -> Callable[[AngleQiskit], QuantumCircuit]:

        if gate_name.lower() in ['wsqaoazerobiased',
                                 'wsqaoaonebiased']:
            def _local_gate_builder(angle: AngleQiskit,
                                    bias_parameter_WS: float):

                if bias_parameter_WS is None:
                    raise ValueError("bias_parameters_WS must be set to a value")

                if bias_parameter_WS < 0 or bias_parameter_WS > 0.5:
                    raise ValueError(f"c_value must be between 0 and 0.5, but is {bias_parameter_WS}")
                if gate_name.lower() == 'wsqaoazerobiased':
                    pass
                elif gate_name.lower() == 'wsqaoaonebiased':
                    bias_parameter_WS = 1 - bias_parameter_WS
                else:
                    raise ValueError(f"Unknown gate name: {gate_name}")

                theta_value = 2 * np.arcsin(np.sqrt(bias_parameter_WS))

                circuit = self._WS_QAOA_MIXER(angle_mixer=angle,
                                              angle_bias=theta_value)

                return circuit

        else:
            raise ValueError(f"Unknown gate name: {gate_name}")
        return _local_gate_builder
