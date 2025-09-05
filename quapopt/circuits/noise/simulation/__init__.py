# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

from enum import Enum
class GateNoiseType(Enum):
    pauli = 'Pauli'
    amplitude_damping = 'AmplitudeDamping'
    depolarizing = 'Depolarizing'


class MeasurementNoiseType(Enum):
    TP_1q_identical = 'TensorProduct1QIdentical'
    TP_1q_general = 'TensorProduct1QGeneral'
    TP_general = 'TensorProductGeneral'
