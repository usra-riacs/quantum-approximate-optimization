# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.


from typing import Tuple, Optional, List

import numpy as np
from qiskit import QuantumCircuit

from quapopt.circuits.gates import (
    AngleQiskit,
    AbstractProgramGateBuilder, AbstractCircuit,
)


class NativeGateBuilderHeronCustomizable(AbstractProgramGateBuilder):
    """
    Native Heron gate set:
    [ID, RZ, SX, X, CZ, RX, RZZ] (with fractional gates)
    [ID, RZ, SX, X, CZ, ]  (without fractional gates)

    """

    def __init__(self,
                 use_fractional_gates: bool = False,
                 replace_rz_with_barriers: bool = False,
                 random_delays_kwargs: Optional[dict] = None,
                 seed: Optional[int] = None):
        #TODO(FBM): we switched to adding delays after creating a circuit, but this can still be worth having. Should refactor though.


        if use_fractional_gates:
            native_gate_set = ['CZ', 'ID', 'RZ', 'SX', 'X', 'RX', 'RZZ']
        else:
            native_gate_set = ['CZ', 'ID', 'RZ', 'SX', 'X']

        super().__init__(sdk_name='qiskit',
                         native_gate_set=native_gate_set)

        self._use_fractional_gates = use_fractional_gates
        self._replace_rz_with_barriers = replace_rz_with_barriers

        _add_random_delays = False
        _delays_distribution_properties = None
        #TODO(FBM): maybe should add more options for random delays
        if random_delays_kwargs is not None:
            # If kwargs are provided, but there's no add_random_delays key, we assume that we want to add random delays
            _add_random_delays = random_delays_kwargs.get('add_random_delays', True)
            # If kwargs are provided, but there's no distribution_properties key, we assume that we want to use a default distribution
            _delays_distribution_properties = random_delays_kwargs.get('distribution_properties', None)
            if _delays_distribution_properties is None:
                # T1 time of Heron devices is around 200 microseconds, so we want to add delays that are much #
                # shorter than that.
                # Typical time of 2-qubit gate is around 50-200ns; single-qubit is around 20-60ns;
                # So we set here default delays to 10ns which is half of extremely fast single-qubit gate.
                _delays_distribution_properties = {'mean': 0.01,
                                                   'std': 0.01}

            else:
                assert 'mean' in _delays_distribution_properties, "Delays distribution properties must contain 'mean' key."
                assert 'std' in _delays_distribution_properties, "Delays distribution properties must contain 'std' key."

            # anticipating that for metadata saving it might be more handy to set mean to 0.0 than have additional flag
            # note: if you want distribution that really has mean zero and sample to the right, please just add very small mean
            if _delays_distribution_properties['mean'] == 0.0:
                _add_random_delays = False
                _delays_distribution_properties = None

        self._add_random_delays = _add_random_delays
        self._delays_distribution_properties = _delays_distribution_properties

        self._numpy_rng = np.random.default_rng(seed)

    @property
    def use_fractional_gates(self):
        return self._use_fractional_gates

    @property
    def replace_rz_with_barriers(self):
        return self._replace_rz_with_barriers

    @replace_rz_with_barriers.setter
    def replace_rz_with_barriers(self, value: bool):
        self._replace_rz_with_barriers = value

    def _maybe_add_random_delay(self,
                                circuit: QuantumCircuit,
                                qubits: List[int]) -> QuantumCircuit:
        # TODO(FBM): perhaps this function should be promoted to parent class at some point?
        if not self._add_random_delays:
            return circuit

        distro_mean = self._delays_distribution_properties['mean']
        distro_std = self._delays_distribution_properties['std']

        delays = self._numpy_rng.normal(loc=distro_mean,
                                        scale=distro_std,
                                        size=len(qubits))
        delays = np.clip(delays, 0, None)  # Ensure non-negative delays
        for qubit, delay in zip(qubits, delays):
            if delay > 0:
                circuit.delay(duration=delay, qarg=qubit, unit='us')
        return circuit

    ########### NATIVE GATE SET STARTS HERE
    def _I(self):
        circuit = QuantumCircuit(1, 1)
        circuit.id(qubit=0)

        circuit = self._maybe_add_random_delay(circuit=circuit,
                                               qubits=[0])

        return circuit

    def _RZ(self,
            angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        if self.replace_rz_with_barriers:
            circuit.barrier(0)
        else:
            circuit.rz(phi=angle, qubit=0)

        circuit = self._maybe_add_random_delay(circuit=circuit,
                                               qubits=[0])

        return circuit

    def _SX(self):
        circuit = QuantumCircuit(1, 1)
        circuit.sx(qubit=0)

        circuit = self._maybe_add_random_delay(circuit=circuit,
                                               qubits=[0])

        return circuit

    def _X(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        # this decomposition is the same for fractional and non-fractional gates
        circuit.x(qubit=0)

        circuit = self._maybe_add_random_delay(circuit=circuit,
                                               qubits=[0])

        return circuit

    def _CZ(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        circuit.cz(control_qubit=0, target_qubit=1)

        circuit = self._maybe_add_random_delay(circuit=circuit,
                                               qubits=[0, 1])

        return circuit

    ########### NATIVE GATE SET FOR FRACTIONAL GATES CONTINUES HERE
    def _RX(self,
            angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)

        if self.use_fractional_gates:
            # In this setting, RX is native
            circuit.rx(theta=angle, qubit=0)
            circuit = self._maybe_add_random_delay(circuit=circuit,
                                                   qubits=[0])
        else:
            # Otherwise, it needs to be decomposed into # RZ and SX gates, which is done in # _exp_X
            circuit = self.exp_X(quantum_circuit=circuit,
                                 angles_tuple=[angle / 2],
                                 qubits_tuple=[0])

        return circuit

    def _RZZ(self,
             angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        if self.use_fractional_gates:
            # In this setting, RZZ is native
            circuit.rzz(theta=angle, qubit1=0, qubit2=1)
            circuit = self._maybe_add_random_delay(circuit=circuit,
                                                   qubits=[0, 1])
        else:
            # Otherwise, it needs to be decomposed into # CZ and RZ gates, which is done in # _exp_ZZ
            circuit = self.exp_ZZ(quantum_circuit=circuit,
                                  angles_tuple=[angle / 2],
                                  qubits_pairs_tuple=[(0, 1)])
        return circuit

    ########### NATIVE GATE SET ENDS HERE
    def _H(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        # this decomposition is the same for fractional and non-fractional gates
        circuit = self.RZ(quantum_circuit=circuit,
                          angles_tuple=[np.pi / 2],
                          qubits_tuple=[0])
        circuit = self.SX(quantum_circuit=circuit,
                          qubits_tuple=[0])
        circuit = self.RZ(quantum_circuit=circuit,
                          angles_tuple=[np.pi / 2],
                          qubits_tuple=[0])
        return circuit

    def _Y(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        # this decomposition is the same for fractional and non-fractional gates
        # circuit.rz(phi=np.pi, qubit=0)
        # circuit.x(qubit=0)

        circuit = self.RZ(quantum_circuit=circuit,
                          angles_tuple=[np.pi],
                          qubits_tuple=[0])
        circuit = self.X(quantum_circuit=circuit,
                         qubits_tuple=[0])

        return circuit

    def _Z(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        # this decomposition is the same for fractional and non-fractional gates
        # circuit.rz(phi=np.pi, qubit=0)
        circuit = self.RZ(quantum_circuit=circuit,
                          angles_tuple=[np.pi],
                          qubits_tuple=[0])

        return circuit



    def _S(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        # this decomposition is the same for fractional and non-fractional gates
        # circuit.rz(phi=np.pi / 2, qubit=0)
        circuit = self.RZ(quantum_circuit=circuit,
                          angles_tuple=[np.pi / 2],
                          qubits_tuple=[0])

        return circuit

    def _Sdag(self) -> AbstractCircuit:
        circuit = QuantumCircuit(1, 1)
        # this decomposition is the same for fractional and non-fractional gates
        # circuit.rz(phi=-np.pi / 2, qubit=0)
        circuit = self.RZ(quantum_circuit=circuit,
                          angles_tuple=[-np.pi / 2],
                          qubits_tuple=[0])

        return circuit

    def _T(self) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        # this decomposition is the same for fractional and non-fractional gates
        # circuit.rz(phi=np.pi / 4, qubit=0)

        circuit = self.RZ(quantum_circuit=circuit,
                          angles_tuple=[np.pi / 4],
                          qubits_tuple=[0])

        return circuit

    def _exp_X(self,
               angle: AngleQiskit) -> QuantumCircuit:

        angle *= 2
        circuit = QuantumCircuit(1, 1)

        if self.use_fractional_gates:
            # circuit.rx(theta=angle, qubit=0)
            circuit = self.RX(quantum_circuit=circuit,
                              angles_tuple=[angle],
                              qubits_tuple=[0])
        else:
            # circuit.rz(phi=np.pi / 2, qubit=0)
            # circuit.sx(qubit=0)
            # circuit.rz(phi=angle + np.pi, qubit=0)
            # circuit.sx(qubit=0)
            # circuit.rz(phi=5 * np.pi / 2, qubit=0)
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi / 2],
                              qubits_tuple=[0])
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[0])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[angle + np.pi],
                              qubits_tuple=[0])
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[0])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[5 * np.pi / 2],
                              qubits_tuple=[0])

        return circuit

    def _exp_Y(self,
               angle: AngleQiskit) -> QuantumCircuit:
        angle *= 2
        circuit = QuantumCircuit(1, 1)

        if self.use_fractional_gates:
            # circuit.rz(phi=-np.pi / 2, qubit=0)
            # circuit.rx(theta=angle, qubit=0)
            # circuit.rz(phi=np.pi / 2, qubit=0)
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[-np.pi / 2],
                              qubits_tuple=[0])
            circuit = self.RX(quantum_circuit=circuit,
                              angles_tuple=[angle],
                              qubits_tuple=[0])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi / 2],
                              qubits_tuple=[0])


        else:
            # circuit.sx(qubit=0)
            # circuit.rz(phi=angle + np.pi, qubit=0)
            # circuit.sx(qubit=0)
            # circuit.rz(phi=3 * np.pi, qubit=0)
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[0])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[angle + np.pi],
                              qubits_tuple=[0])
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[0])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[3 * np.pi],
                              qubits_tuple=[0])

        return circuit

    def _exp_Z(self,
               angle: AngleQiskit) -> QuantumCircuit:
        angle *= 2
        circuit = QuantumCircuit(1, 1)
        # this decomposition is the same for fractional and non-fractional gates
        # circuit.rz(phi=angle, qubit=0)
        circuit = self.RZ(quantum_circuit=circuit,
                          angles_tuple=[angle],
                          qubits_tuple=[0])

        return circuit

    def _exp_XX(self,
                angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        angle *= 2

        if self.use_fractional_gates:
            # # q0
            # circuit.rz(phi=np.pi / 2, qubit=0)
            # circuit.sx(qubit=0)
            # circuit.rz(phi=np.pi, qubit=0)
            # # q1
            # circuit.rz(phi=np.pi / 2, qubit=1)
            # circuit.sx(qubit=1)
            # circuit.rz(phi=np.pi, qubit=1)
            # rzz
            # circuit.rzz(theta=angle, qubit1=0, qubit2=1)
            # # q0
            # circuit.sx(qubit=0)
            # circuit.rz(phi=np.pi / 2, qubit=0)
            # # q1
            # circuit.sx(qubit=1)
            # circuit.rz(phi=np.pi / 2, qubit=1)

            # q0 and q1
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi / 2],
                              qubits_tuple=[0, 1])
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[0, 1])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi],
                              qubits_tuple=[0, 1])

            # rzz
            circuit = self.RZZ(quantum_circuit=circuit,
                               angles_tuple=[angle],
                               qubits_pairs_tuple=[(0, 1)])

            # q0 and q1
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[0, 1])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi / 2],
                              qubits_tuple=[0, 1])

        else:
            raise NotImplementedError("This gate is not implemented yet.")

        return circuit

    def _exp_YY(self,
                angle: AngleQiskit) -> QuantumCircuit:

        circuit = QuantumCircuit(2, 2)
        angle *= 2

        if self.use_fractional_gates:
            # # q0
            # circuit.sx(qubit=0)
            # circuit.rz(phi=np.pi, qubit=0)
            # # q1
            # circuit.sx(qubit=1)
            # circuit.rz(phi=np.pi, qubit=1)
            # # rzz
            # circuit.rzz(theta=angle, qubit1=0, qubit2=1)
            # # q0
            # circuit.sx(qubit=0)
            # # q1
            # circuit.sx(qubit=1)

            # q0 and q1
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[0, 1])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi],
                              qubits_tuple=[0, 1])

            # rzz
            circuit = self.RZZ(quantum_circuit=circuit,
                               angles_tuple=[angle],
                               qubits_pairs_tuple=[(0, 1)])
            # q0 and q1
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[0, 1])

        else:
            raise NotImplementedError("This gate is not implemented yet.")

        return circuit

    def _exp_ZZ(self,
                angle: AngleQiskit) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)
        angle *= 2
        # circuit.rzz(theta=angle, qubit1=0, qubit2=1)

        if self.use_fractional_gates:
            # this gate is native in this setting
            # circuit.rzz(theta=angle, qubit1=0, qubit2=1)
            circuit = self.RZZ(quantum_circuit=circuit,
                               angles_tuple=[angle],
                               qubits_pairs_tuple=[(0, 1)])

        else:
            # Gate count: 2 CZ, and 9 single-qubit gates
            # 3 initial rotations
            # # q1
            # circuit.rz(phi=np.pi / 2, qubit=1)
            # circuit.sx(qubit=1)
            # # CZ
            # circuit.cz(control_qubit=0, target_qubit=1)
            # # q1
            # circuit.rz(phi=np.pi, qubit=1)
            # circuit.sx(qubit=1)
            # circuit.rz(phi=np.pi + angle, qubit=1)
            # circuit.sx(qubit=1)
            # circuit.rz(phi=np.pi, qubit=1)
            # # CZ
            # circuit.cz(control_qubit=0, target_qubit=1)
            # # q1
            # circuit.sx(qubit=1)
            # circuit.rz(phi=np.pi / 2, qubit=1)

            # q1
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi / 2],
                              qubits_tuple=[1])
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[1])
            # CZ
            circuit = self.CZ(quantum_circuit=circuit,
                              qubits_pairs_tuple=[(0, 1)])
            # q1
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi],
                              qubits_tuple=[1])

            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[1])

            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi + angle],
                              qubits_tuple=[1])

            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[1])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi],
                              qubits_tuple=[1])
            # CZ
            circuit = self.CZ(quantum_circuit=circuit,
                              qubits_pairs_tuple=[(0, 1)])
            # q1
            circuit = self.SX(quantum_circuit=circuit,
                              qubits_tuple=[1])
            circuit = self.RZ(quantum_circuit=circuit,
                              angles_tuple=[np.pi / 2],
                              qubits_tuple=[1])

        return circuit

    def _exp_XXYY(self,
                  angle: AngleQiskit,
                  ) -> QuantumCircuit:
        raise NotImplementedError("This gate is not implemented yet.")

        circuit = QuantumCircuit(2, 2)
        angle *= 2

        circuit.rxx(theta=angle, qubit1=0, qubit2=1)
        circuit.ryy(theta=angle, qubit1=0, qubit2=1)

        return circuit

    def _u3(self,
            angles_tuple: Tuple[AngleQiskit]) -> QuantumCircuit:
        raise NotImplementedError("This gate is not implemented yet.")

        theta, phi, lam = angles_tuple
        circuit = QuantumCircuit(1, 1)
        circuit.u(theta=theta,
                  phi=phi,
                  lam=lam,
                  qubit=0)
        return circuit

    def _SWAP(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2, 2)

        # TODO(FBM): I think we can get just two 2-qubit gates if we use fractional gates (add phase to XX+YY decomposition)
        # this decomposition is the same for fractional and non-fractional gates

        # Gate count: # 3 CZ, and 6 SX

        # circuit.sx(qubit=0)
        # circuit.sx(qubit=1)
        # circuit.cz(control_qubit=0, target_qubit=1)
        # circuit.sx(qubit=0)
        # circuit.sx(qubit=1)
        # circuit.cz(control_qubit=0, target_qubit=1)
        # circuit.sx(qubit=0)
        # circuit.sx(qubit=1)
        # circuit.cz(control_qubit=0, target_qubit=1)

        circuit = self.SX(quantum_circuit=circuit,
                          qubits_tuple=[0, 1])
        circuit = self.CZ(quantum_circuit=circuit,
                          qubits_pairs_tuple=[(0, 1)])
        circuit = self.SX(quantum_circuit=circuit,
                          qubits_tuple=[0, 1])
        circuit = self.CZ(quantum_circuit=circuit,
                          qubits_pairs_tuple=[(0, 1)])
        circuit = self.SX(quantum_circuit=circuit,
                          qubits_tuple=[0, 1])
        circuit = self.CZ(quantum_circuit=circuit,
                          qubits_pairs_tuple=[(0, 1)])

        return circuit

    def _CNOT(self) -> QuantumCircuit:
        raise NotImplementedError("This gate is not implemented yet.")

        circuit = QuantumCircuit(2, 2)
        circuit.cx(control_qubit=0, target_qubit=1)

        return circuit

    def _exp_ZZ_SWAP(self,
                     angle: AngleQiskit,
                     compilation_variant=None) -> QuantumCircuit:

        circuit = QuantumCircuit(2, 2)
        angle *= 2

        if self.use_fractional_gates:
            if compilation_variant is None:
                compilation_variant = 0

            if compilation_variant == 0:
                # Gate count: 2CZ + 1RZZ, and 6 SX + 2 RZ
                # circuit.sx(qubit=0)
                # circuit.sx(qubit=1)
                # circuit.cz(control_qubit=0, target_qubit=1)
                # circuit.sx(qubit=0)
                # circuit.sx(qubit=1)
                # circuit.cz(control_qubit=0, target_qubit=1)
                # circuit.sx(qubit=0)
                # circuit.sx(qubit=1)
                # circuit.rzz(theta=angle - np.pi / 2, qubit1=0, qubit2=1)
                # circuit.rz(phi=np.pi / 2, qubit=0)
                # circuit.rz(phi=np.pi / 2, qubit=1)

                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[0, 1])
                circuit = self.CZ(quantum_circuit=circuit,
                                  qubits_pairs_tuple=[(0, 1)])
                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[0, 1])
                circuit = self.CZ(quantum_circuit=circuit,
                                  qubits_pairs_tuple=[(0, 1)])
                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[0, 1])
                circuit = self.RZZ(quantum_circuit=circuit,
                                   angles_tuple=[angle - np.pi / 2],
                                   qubits_pairs_tuple=[(0, 1)])
                circuit = self.RZ(quantum_circuit=circuit,
                                  angles_tuple=[np.pi / 2],
                                  qubits_tuple=[0, 1])



            elif compilation_variant == 1:
                # # Gate count: 3 RZZ, and 4 RX + 2SX + 6RZ
                # circuit.rzz(theta=angle + np.pi / 2, qubit1=0, qubit2=1)
                # # q0
                # circuit.rz(phi=np.pi / 2, qubit=0)
                # circuit.rx(theta=-np.pi / 2, qubit=0)
                # circuit.rz(phi=np.pi / 2, qubit=0)
                # # q1
                # circuit.rz(phi=np.pi / 2, qubit=1)
                # circuit.rx(theta=-np.pi / 2, qubit=1)
                # circuit.rz(phi=np.pi / 2, qubit=1)
                # # rzz
                # circuit.rzz(theta=np.pi / 2, qubit1=0, qubit2=1)
                # # q0
                # circuit.rx(theta=np.pi / 2, qubit=0)
                # circuit.rz(phi=-np.pi / 2, qubit=0)
                # # q1
                # circuit.rx(theta=np.pi / 2, qubit=1)
                # circuit.rz(phi=-np.pi / 2, qubit=1)
                # # rzz
                # circuit.rzz(theta=np.pi / 2, qubit1=0, qubit2=1)
                # # q0
                # circuit.sx(qubit=0)
                # # q1
                # circuit.sx(qubit=1)
                # RZZ
                circuit = self.RZZ(quantum_circuit=circuit,
                                   angles_tuple=[angle + np.pi / 2],
                                   qubits_pairs_tuple=[(0, 1)])
                # q0 and q1
                circuit = self.RZ(quantum_circuit=circuit,
                                  angles_tuple=[np.pi / 2],
                                  qubits_tuple=[0, 1])
                circuit = self.RX(quantum_circuit=circuit,
                                  angles_tuple=[-np.pi / 2],
                                  qubits_tuple=[0, 1])
                circuit = self.RZ(quantum_circuit=circuit,
                                  angles_tuple=[np.pi / 2],
                                  qubits_tuple=[0, 1])
                # RZZ
                circuit = self.RZZ(quantum_circuit=circuit,
                                   angles_tuple=[np.pi / 2],
                                   qubits_pairs_tuple=[(0, 1)])
                # q0 and q1
                circuit = self.RX(quantum_circuit=circuit,
                                  angles_tuple=[np.pi / 2],
                                  qubits_tuple=[0, 1])
                circuit = self.RZ(quantum_circuit=circuit,
                                  angles_tuple=[-np.pi / 2],
                                  qubits_tuple=[0, 1])
                # RZZ
                circuit = self.RZZ(quantum_circuit=circuit,
                                   angles_tuple=[np.pi / 2],
                                   qubits_pairs_tuple=[(0, 1)])
                # q0 and q1
                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[0, 1])





            else:
                raise ValueError(f"Unknown compilation variant: {compilation_variant}")

        else:
            if compilation_variant is None:
                compilation_variant = 0

            if compilation_variant == 0:
                # Gate count: 3 CZ, and 8 SX + 5 RZ
                # circuit.sx(qubit=0)
                # circuit.sx(qubit=1)
                # ##############CZ
                # circuit.cz(control_qubit=0, target_qubit=1)
                # # q0
                # circuit.sx(qubit=0)
                # circuit.rz(phi=-np.pi / 2, qubit=0)
                # # q1
                # circuit.rz(phi=-np.pi / 2, qubit=1)
                # circuit.sx(qubit=1)
                # ##############CZ
                # circuit.cz(control_qubit=0, target_qubit=1)
                # # q0
                # circuit.sx(qubit=0)
                # circuit.rz(phi=angle + np.pi / 2, qubit=0)
                # circuit.sx(qubit=0)
                # circuit.rz(phi=-np.pi, qubit=0)
                # # q1
                # circuit.sx(qubit=1)
                # ##############CZ
                # circuit.cz(control_qubit=0, target_qubit=1)
                # # q0
                # circuit.sx(qubit=0)
                # circuit.rz(phi=np.pi / 2, qubit=0)
                #

                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[0, 1])
                # CZ
                circuit = self.CZ(quantum_circuit=circuit,
                                  qubits_pairs_tuple=[(0, 1)])
                # q0
                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[0])
                circuit = self.RZ(quantum_circuit=circuit,
                                  angles_tuple=[-np.pi / 2],
                                  qubits_tuple=[0])
                # q1
                circuit = self.RZ(quantum_circuit=circuit,
                                  angles_tuple=[-np.pi / 2],
                                  qubits_tuple=[1])
                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[1])
                # CZ
                circuit = self.CZ(quantum_circuit=circuit,
                                  qubits_pairs_tuple=[(0, 1)])
                # q0 and q1
                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[0, 1])
                # q0
                circuit = self.RZ(quantum_circuit=circuit,
                                  angles_tuple=[angle + np.pi / 2],
                                  qubits_tuple=[0])
                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[0])
                circuit = self.RZ(quantum_circuit=circuit,
                                  angles_tuple=[-np.pi],
                                  qubits_tuple=[0])
                # CZ
                circuit = self.CZ(quantum_circuit=circuit,
                                  qubits_pairs_tuple=[(0, 1)])
                # q0
                circuit = self.SX(quantum_circuit=circuit,
                                  qubits_tuple=[0])
                circuit = self.RZ(quantum_circuit=circuit,
                                  angles_tuple=[np.pi / 2],
                                  qubits_tuple=[0])





            else:
                raise NotImplementedError("This compilation variant is not implemented yet.")

        return circuit

    def _exp_ZZXXYY(self,
                    angle_ZZ: AngleQiskit,
                    angle_XY: AngleQiskit) -> QuantumCircuit:
        raise NotImplementedError("This gate is not implemented yet.")
        circuit0 = self._exp_ZZ(angle=angle_ZZ)
        circuit1 = self._exp_XXYY(angle=angle_XY)
        return self.combine_circuits(left_circuit=circuit0,
                                     right_circuit=circuit1)

    def _exp_ZZXXYY_SWAP(self,
                         angle_ZZ: AngleQiskit,
                         angle_XY: AngleQiskit) -> QuantumCircuit:

        raise NotImplementedError("This gate is not implemented yet.")
        circuit0 = self._exp_ZZXXYY(angle_ZZ=angle_ZZ,
                                    angle_XY=angle_XY)
        circuit1 = self._SWAP()
        return self.combine_circuits(left_circuit=circuit0,
                                     right_circuit=circuit1)


class NativeGateBuilderHeron(NativeGateBuilderHeronCustomizable):
    """
    Native Heron gate set:
    [CZ, ID, RZ, SX, X, RX, RZZ] (with fractional gates)
    [CZ, ID, RZ, SX, X]  (without fractional gates)

    Heron compiler with default settings hard-coded, so that it can be used without any additional arguments, and
    we don't risk accidentally using wrong settings.
    """

    def __init__(self,
                 use_fractional_gates: bool = False):
        super().__init__(use_fractional_gates=use_fractional_gates,
                         replace_rz_with_barriers=False,
                         random_delays_kwargs=None,
                         seed=None
                         )
