# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


from typing import Tuple

import cirq
import numpy as np
import pandas as pd
from cirq import (
    Circuit,
    H,
    S,
    T,
    ZPowGate,
    amplitude_damp,
    asymmetric_depolarize,
    depolarize,
)
from pydantic import confloat

from quapopt import ancillary_functions as anf
from quapopt.circuits.gates import AngleCirq
from quapopt.circuits.gates.logical.LogicalGateBuilderCirq import LogicalGateBuilderCirq
from quapopt.circuits.noise.simulation import GateNoiseType

_SUPPORTED_GATES = {
    "X",
    "Y",
    "Z",
    "S",
    "T",
    "H",
    "Sdag",
    "SWAP",
    "ISWAP",
    "XX",
    "YY",
    "ZZ",
    "RX",
    "RY",
    "RZ",
    "RXX",
    "RYY",
    "RZZ",
    "RXY",
}

_SUPPORTED_NOISE_CHANNELS = [
    GateNoiseType.pauli,
    GateNoiseType.amplitude_damping,
    GateNoiseType.depolarizing,
]


class LogicalNoisyGateBuilderCirq(LogicalGateBuilderCirq):

    def __init__(self, quantum_register, noise_description_dataframe: pd.DataFrame):
        super().__init__(quantum_register=quantum_register)

        noise_description_dataframe["gate_type"] = noise_description_dataframe[
            "gate_type"
        ].apply(lambda string: string.upper())

        self._noise_description_dataframe = noise_description_dataframe

        noisy_gates_set = set(noise_description_dataframe["gate_type"].values)

        if len(noisy_gates_set.difference(_SUPPORTED_GATES)) > 1:
            raise ValueError(
                f"Provided noise model contains gates outside supported gateset. "
                f"Please choose gates only from set: {_SUPPORTED_GATES}"
            )

        self._noisy_gates_set = noisy_gates_set

    @property
    def noise_description_dataframe(self):
        return self._noise_description_dataframe

    @property
    def noisy_gates_set(self):
        return self._noisy_gates_set

    def _add_noise_to_gate(
        self,
        quantum_circuit: Circuit,
        qubit_or_qubits: Tuple[int],
        gate_name: str,
        gate_probability: float = 1.0,
    ):
        if isinstance(qubit_or_qubits, int):
            qubit_or_qubits = (qubit_or_qubits,)

        # TODO(FBM): refactor this to new format
        raise NotImplementedError("This method is not yet implemented.")

        # TODO(FBM): add this to the constructor
        local_noise_df = anf.find_dataframe_subset(
            df=self.noise_description_dataframe,
            variable_values_pairs=[
                ("gate_type", gate_name),
                ("qubits", qubit_or_qubits),
            ],
        )

        if gate_name not in self.noisy_gates_set:
            return quantum_circuit

        if gate_probability == 0:
            return quantum_circuit

        noise_type = local_noise_df["noise_type"].values[0]

        assert noise_type in _SUPPORTED_NOISE_CHANNELS, (
            f"Noise channel type '{noise_type}' not supported."
            f" Please choose one of the following: {_SUPPORTED_NOISE_CHANNELS}"
        )

        if noise_type in [GateNoiseType.pauli]:
            local_pauli_errors = local_noise_df["noise_model"].values[0]
            if gate_probability < 1.0:
                I_identifier = "".join(["I" for _ in qubit_or_qubits])
                local_pauli_errors = {
                    key: val * gate_probability
                    for key, val in local_pauli_errors.items()
                    if key != I_identifier
                }
                local_pauli_errors[I_identifier] = 1 - sum(local_pauli_errors.values())

            local_noise_channel = asymmetric_depolarize(
                error_probabilities=local_pauli_errors
            )

        elif noise_type in [GateNoiseType.amplitude_damping]:
            amplitude_damping_probability = local_noise_df["noise_model"].values[0]
            if gate_probability < 1.0:
                amplitude_damping_probability *= gate_probability

            local_noise_channel = amplitude_damp(
                gamma=np.sqrt(amplitude_damping_probability)
            )

        elif noise_type in [GateNoiseType.depolarizing]:
            depolarizing_probability = local_noise_df["noise_model"].values[0]
            if gate_probability < 1.0:
                depolarizing_probability *= gate_probability

            local_noise_channel = depolarize(p=depolarizing_probability)

        quantum_register_elements = [
            self.quantum_register[qi] for qi in qubit_or_qubits
        ]

        quantum_circuit += cirq.Circuit(
            local_noise_channel.on(*quantum_register_elements)
        )

        return quantum_circuit

    def _H(self, qubit, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        quantum_circuit = self._add_probabilistic_gate(
            gate=H, qubit_or_qubits=qubit, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="H",
            gate_probability=gate_probability,
        )

    def _T(self, qubit, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:
        quantum_circuit = self._add_probabilistic_gate(
            gate=T, qubit_or_qubits=qubit, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="T",
            gate_probability=gate_probability,
        )

    def _S(self, qubit, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        quantum_circuit = self._add_probabilistic_gate(
            gate=S, qubit_or_qubits=qubit, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="S",
            gate_probability=gate_probability,
        )

    def _Sdag(self, qubit, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:
        quantum_circuit = self._add_probabilistic_gate(
            gate=ZPowGate(exponent=-0.5),
            qubit_or_qubits=qubit,
            gate_probability=gate_probability,
        )
        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="Sdag",
            gate_probability=gate_probability,
        )

    def _exp_X(
        self, qubit, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        quantum_circuit = super()._exp_X(
            qubit=qubit, angle=angle, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="RX",
            gate_probability=gate_probability,
        )

    def _exp_Y(
        self, qubit, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        quantum_circuit = super()._exp_Y(
            qubit=qubit, angle=angle, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="RY",
            gate_probability=gate_probability,
        )

    def _exp_Z(
        self, qubit, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        quantum_circuit = super()._exp_Z(
            qubit=qubit, angle=angle, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="RZ",
            gate_probability=gate_probability,
        )

    def _X(
        self,
        qubit,
        gate_probability: confloat(ge=0.0, le=1.0),
    ) -> Circuit:

        quantum_circuit = super()._X(qubit=qubit, gate_probability=gate_probability)

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="X",
            gate_probability=gate_probability,
        )

    def _Y(
        self,
        qubit,
        gate_probability: confloat(ge=0.0, le=1.0),
    ) -> Circuit:

        quantum_circuit = super()._Y(qubit=qubit, gate_probability=gate_probability)
        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="Y",
            gate_probability=gate_probability,
        )

    def _Z(
        self,
        qubit,
        gate_probability: confloat(ge=0.0, le=1.0),
    ) -> Circuit:
        quantum_circuit = super()._Z(qubit=qubit, gate_probability=gate_probability)
        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubit,
            gate_name="Z",
            gate_probability=gate_probability,
        )

    def _exp_XX(
        self, qubits_pair, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        quantum_circuit = super()._exp_XX(
            qubits_pair=qubits_pair, gate_probability=gate_probability, angle=angle
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubits_pair,
            gate_name="RXX",
            gate_probability=gate_probability,
        )

    def _exp_YY(
        self, qubits_pair, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        quantum_circuit = super()._exp_YY(
            qubits_pair=qubits_pair, gate_probability=gate_probability, angle=angle
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubits_pair,
            gate_name="RYY",
            gate_probability=gate_probability,
        )

    def _exp_ZZ(
        self, qubits_pair, angle: AngleCirq, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        if angle is not None:
            angle *= 2

        quantum_circuit = super()._exp_ZZ(
            qubits_pair=qubits_pair, gate_probability=gate_probability, angle=angle
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubits_pair,
            gate_name="RZZ",
            gate_probability=gate_probability,
        )

    def _XX(self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        quantum_circuit = super()._XX(
            qubits_pair=qubits_pair, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubits_pair,
            gate_name="XX",
            gate_probability=gate_probability,
        )

    def _YY(self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        quantum_circuit = super()._YY(
            qubits_pair=qubits_pair, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubits_pair,
            gate_name="YY",
            gate_probability=gate_probability,
        )

    def _ZZ(self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        quantum_circuit = super()._ZZ(
            qubits_pair=qubits_pair, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubits_pair,
            gate_name="ZZ",
            gate_probability=gate_probability,
        )

    def _exp_XXYY(
        self,
        qubits_pair,
        angle: AngleCirq,
        gate_probability: confloat(ge=0.0, le=1.0) = 1.0,
    ) -> Circuit:

        quantum_circuit = super()._exp_XXYY(
            qubits_pair=qubits_pair, gate_probability=gate_probability, angle=angle
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubits_pair,
            gate_name="RXY",
            gate_probability=gate_probability,
        )

    def _SWAP(self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)) -> Circuit:

        quantum_circuit = super()._SWAP(
            qubits_pair=qubits_pair, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubits_pair,
            gate_name="SWAP",
            gate_probability=gate_probability,
        )

    def _ISWAP(
        self, qubits_pair, gate_probability: confloat(ge=0.0, le=1.0)
    ) -> Circuit:

        quantum_circuit = super()._ISWAP(
            qubits_pair=qubits_pair, gate_probability=gate_probability
        )

        return self._add_noise_to_gate(
            quantum_circuit=quantum_circuit,
            qubit_or_qubits=qubits_pair,
            gate_name="ISWAP",
            gate_probability=gate_probability,
        )
