# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import copy
from enum import Enum
from typing import List, Optional

import numpy as np

from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)

try:
    pass
except (ImportError, ModuleNotFoundError):
    pass

from qiskit.transpiler.passmanager import StagedPassManager

from quapopt.circuits.gates.gate_delays import DelayScheduleType
from quapopt.data_analysis.data_handling import MAIN_KEY_SEPARATOR as MKS
from quapopt.data_analysis.data_handling import MAIN_KEY_VALUE_SEPARATOR as MKVS
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.optimization.QAOA import AnsatzSpecifier, QubitMappingType
from quapopt.optimization.QAOA.circuits import MappedAnsatzCircuit


class MirrorCircuitType(Enum):
    """
    This class is used to define the types of mirror circuits that can be generated for our experiments.
    """

    QAOA_LINEAR_SWAP_NETWORK = "QAOA-LNS"
    QAOA_FULLY_CONNECTED = "QAOA-FC"
    QAOA_SABRE = "QAOA-SABRE"
    NONE = "NONE"


class MirrorCircuitSpecification:
    """
    This class is used to define the specifications of the mirror circuits that can be generated for our experiments.
    """

    def __init__(
        self,
        mirror_circuit_type: MirrorCircuitType,
        mirror_circuit_ansatz_kwargs: Optional[dict] = None,
        mirror_circuit_parameters_values: Optional[np.ndarray] = None,
        phase_hamiltonian_qaoa: Optional[ClassicalHamiltonian] = None,
        fixed_ansatz: Optional[MappedAnsatzCircuit] = None,
        pass_manager_mirror_circuit: Optional[StagedPassManager] = None,
        qubit_mapping_type: Optional[QubitMappingType] = None,
    ):
        if mirror_circuit_type in [
            MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK,
            MirrorCircuitType.QAOA_FULLY_CONNECTED,
            MirrorCircuitType.QAOA_SABRE,
        ]:
            assert phase_hamiltonian_qaoa is not None, (
                "If the mirror circuit is of QAOA type, "
                "we need to provide the phase hamiltonian."
            )

        self.mirror_circuit_type = mirror_circuit_type
        self.mirror_circuit_ansatz_kwargs = mirror_circuit_ansatz_kwargs
        self.mirror_circuit_parameters_values = mirror_circuit_parameters_values
        self.phase_hamiltonian_qaoa = phase_hamiltonian_qaoa
        self._fixed_ansatz = fixed_ansatz
        self.pass_manager_mirror_circuit = pass_manager_mirror_circuit
        self.qubit_mapping_type = qubit_mapping_type

    @property
    def fixed_ansatz(self):
        return self._fixed_ansatz

    @fixed_ansatz.setter
    def fixed_ansatz(self, value: MappedAnsatzCircuit):
        if not isinstance(value, MappedAnsatzCircuit):
            raise TypeError(
                "The fixed ansatz must be an instance of MappedAnsatzCircuit."
            )
        self._fixed_ansatz = value

    def copy(self):
        return copy.deepcopy(self)

    def get_ansatz_description_string(self):

        circuit_type = self.mirror_circuit_type

        if circuit_type in [
            MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK,
            MirrorCircuitType.QAOA_FULLY_CONNECTED,
            MirrorCircuitType.QAOA_SABRE,
        ]:

            dhq = self.phase_hamiltonian_qaoa
            mcak = self.mirror_circuit_ansatz_kwargs
            hamiltonian_class_spec = dhq.hamiltonian_class_specifier
            hamiltonian_instance_spec = dhq.hamiltonian_instance_specifier

            depth = mcak["depth"]
            time_block_size = mcak["time_block_size"]
            phase_separator_type = mcak["phase_separator_type"]
            mixer_type = mcak["mixer_type"]

            if circuit_type == MirrorCircuitType.QAOA_LINEAR_SWAP_NETWORK:
                qubit_mapping_type = QubitMappingType.linear_swap_network
            elif circuit_type == MirrorCircuitType.QAOA_FULLY_CONNECTED:
                qubit_mapping_type = QubitMappingType.fully_connected
            elif circuit_type == MirrorCircuitType.QAOA_SABRE:
                qubit_mapping_type = QubitMappingType.sabre
            else:
                raise NotImplementedError(
                    "This mirror circuit type is not implemented yet."
                )

            specifier = AnsatzSpecifier(
                PhaseHamiltonianClass=hamiltonian_class_spec,
                PhaseHamiltonianInstance=hamiltonian_instance_spec,
                Depth=depth,
                TimeBlockSize=time_block_size,
                PhaseSeparatorType=phase_separator_type,
                MixerType=mixer_type,
                QubitMappingType=qubit_mapping_type,
            )

            return specifier.get_description_string()
        else:
            raise ValueError(f"Unsupported mirror circuit type: {circuit_type}")


def get_CAF_standardized_folder_hierarchy_depreciated(
    backend_name: str,
    simulated: bool,
    use_fractional_gates: bool,
    enable_DD: bool,
    enable_twirled_gates: bool,
    enable_twirled_measurements: bool,
    enable_active_reset: bool,
    number_of_qubits: int,
    mirror_circuit_type: Optional[MirrorCircuitType] = None,
    replace_rz_with_barriers: Optional[bool] = None,
    optimization_level: Optional[int] = None,
    routing_method: Optional[str] = None,
    delay_schedule_type: Optional[DelayScheduleType] = None,
    add_barriers_to_layers: Optional[bool] = None,
    optimized_parametrs: Optional[bool] = None,
) -> List[str]:
    if mirror_circuit_type is None:
        mirror_circuit_type = MirrorCircuitType.NONE

    em_string = (
        f"WFG{MKVS}{use_fractional_gates}{MKS}"
        f"DD{MKVS}{enable_DD}{MKS}"
        f"RCG{MKVS}{enable_twirled_gates}{MKS}"
        f"RCM{MKVS}{enable_twirled_measurements}{MKS}"
        f"AR{MKVS}{enable_active_reset}"
        f"{MKS}MCT{MKVS}{mirror_circuit_type.value}"
    )

    if replace_rz_with_barriers is not None:
        em_string += f"{MKS}RZB{MKVS}{replace_rz_with_barriers}"

    if optimization_level is not None:
        em_string += f"{MKS}OPL{MKVS}{optimization_level}"

    if isinstance(routing_method, str):
        if routing_method.lower() == "none":
            routing_method = None
    if routing_method is not None:
        em_string += f"{MKS}ROM{MKVS}{routing_method}"

    if delay_schedule_type is not None:
        em_string += f"{MKS}DST{MKVS}{delay_schedule_type.value}"

    if add_barriers_to_layers is not None:
        em_string += f"{MKS}ABL{MKVS}{add_barriers_to_layers}"

    if optimized_parametrs is not None:
        em_string += f"{MKS}OPA{MKVS}{optimized_parametrs}"

    experiment_folders_hierarchy = [
        "Results",
        "NoiseCharacterization",
        "CAFExperiments",
        f"{SNV.Backend.id}{MKVS}{backend_name}{MKS}{SNV.Simulated.id}{MKVS}{simulated}",
        em_string,
        f"{SNV.NumberOfQubits.id}={number_of_qubits}",
    ]

    return experiment_folders_hierarchy


def _standardized_table_name_CAF(
    delay_schedule_description: str,
    mirror_circuit_repeats: int,
):
    return f"{delay_schedule_description};REP={mirror_circuit_repeats}"
