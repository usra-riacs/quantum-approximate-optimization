# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import datetime
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
import rustworkx as rx
from qiskit.circuit import Qubit as QubitQiskit
from qiskit.converters import circuit_to_dag
from qiskit.primitives import PrimitiveJob
from qiskit.primitives.containers import SamplerPubResult
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.primitives.containers.data_bin import DataBin
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.backends.aer_simulator import AerSimulator
from qiskit_aer.noise.noise_model import NoiseModel
from qiskit_aer.primitives import SamplerV2 as SamplerAer
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJob
from qiskit_ibm_runtime import RuntimeJobV2 as QiskitJobHardware
from qiskit_ibm_runtime import SamplerV2 as SamplerRuntime
from qiskit_ibm_runtime import Session as SessionRuntime
from qiskit_ibm_runtime.fake_provider.local_runtime_job import LocalRuntimeJob
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_ibm_runtime.models.backend_properties import BackendProperties

from quapopt import ancillary_functions as anf
from quapopt.circuits.backend_utilities.qiskit.qiskit_config import *
from quapopt.circuits.gates import CircuitQiskit
from quapopt.circuits.gates.native.NativeGateBuilderHeron import (
    AbstractProgramGateBuilder,
    NativeGateBuilderHeron,
    NativeGateBuilderHeronCustomizable,
)
from quapopt.data_analysis.data_handling import MAIN_KEY_VALUE_SEPARATOR
from quapopt.data_analysis.data_handling import STANDARD_NAMES_DATA_TYPES as SNDT
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling.io_utilities.results_logging import (
    ResultsLogger,
)
from quapopt.optimization.QAOA import QubitMappingType


@contextmanager
def _ibm_runtime_context(mocked: bool, qiskit_backend=None):
    if mocked:
        # Simulated (mock) context manager does nothing.
        yield
    else:
        assert (
            qiskit_backend is not None
        ), "qiskit_backend must be provided when context manager is not mocked"
        # Real context manager: connect to the actual session.
        session = SessionRuntime(backend=qiskit_backend)
        try:
            yield session
        finally:
            session.close()


def get_ibm_runtime_context_manager(qiskit_backend, mocked=False):
    return _ibm_runtime_context(mocked=mocked, qiskit_backend=qiskit_backend)


@contextmanager
def create_qiskit_session(qiskit_backend, mocked: bool = False, session_ibm=None):
    """
    Create just the IBM Runtime session context manager.

    Args:
        qiskit_backend: Qiskit backend for session creation
        mocked: Whether to mock the session (useful for simulation testing)
        session_ibm: Optional existing session to reuse instead of creating new one

        NOTE: if session_ibm is provided, it will be used directly without creating a new session and
        the mocked flag and backend will be ignored.

    Yields:
        session_ibm: IBM Runtime session or mocked equivalent (existing or newly created)

    Example:
        # Create new session
        with create_qiskit_session(backend, mocked=False) as session:
            sampler = create_qiskit_sampler(backend, False, 1000, session_ibm=session)

        # Reuse existing session
        with existing_session_manager:
            with create_qiskit_session(backend, session_ibm=existing_session) as session:
                # session == existing_session, no new session created
    """

    if session_ibm is not None:
        # Use existing session, don't create a new one
        yield session_ibm
    else:
        # Create new session
        runtime_context_manager = get_ibm_runtime_context_manager(
            qiskit_backend=qiskit_backend, mocked=mocked
        )

        with runtime_context_manager as new_session_ibm:
            yield new_session_ibm


def create_qiskit_sampler(
    qiskit_backend: IBMBackend | AerSimulator,
    simulation: bool,
    qiskit_sampler_options: Optional[dict] = None,
    session_ibm=None,
    override_to_noiseless_simulation=False,
    noise_model: Optional[NoiseModel] = None,
):
    """

    :param qiskit_backend:
    :param simulation:
    :param qiskit_sampler_options:
    :param session_ibm:
    :param override_to_noiseless_simulation:
    :param noise_model:
    :return:
    """

    if isinstance(qiskit_backend, AerSimulator):
        simulation = True

    if qiskit_sampler_options is None:
        if simulation:
            if qiskit_backend.name in REAL_DEVICES_IBM:
                qiskit_sampler_options = DEFAULT_SIMULATED_SAMPLER_KWARGS.copy()
            elif qiskit_backend.name[0:3] == "aer":
                qiskit_sampler_options = DEFAULT_SIMULATOR_BACKEND_KWARGS.copy()
            else:
                raise ValueError(
                    f"Backend {qiskit_backend.name} not recognized for simulation. "
                )

            if override_to_noiseless_simulation:
                qiskit_sampler_options.update({"noise_model": None})
        else:
            qiskit_sampler_options = DEFAULT_QPU_SAMPLER_KWARGS.copy()

    qiskit_sampler_options = qiskit_sampler_options.copy()
    if simulation:
        if qiskit_backend.name in REAL_DEVICES_IBM:
            if not override_to_noiseless_simulation:
                if noise_model is None:
                    noise_model = NoiseModel.from_backend(
                        backend=qiskit_backend,
                        gate_error=True,
                        readout_error=True,
                        thermal_relaxation=True,
                        temperature=0.0,
                    )
                if not noise_model.is_ideal():
                    qiskit_sampler_options.update({"noise_model": noise_model})

            aer_simulator = AerSimulator.from_backend(
                backend=qiskit_backend, **qiskit_sampler_options
            )

            sampler_ibm = SamplerAer.from_backend(backend=aer_simulator)
        elif qiskit_backend.name[0:3] == "aer":

            qiskit_backend.set_options(**qiskit_sampler_options)

            if noise_model is not None:
                if not noise_model.is_ideal():
                    qiskit_backend.set_options(noise_model=noise_model)

            sampler_ibm = SamplerAer.from_backend(backend=qiskit_backend)
        else:
            raise ValueError(
                f"Backend {qiskit_backend.name} not recognized for simulation. "
            )
    else:
        # Import here to avoid circular imports
        from qiskit_ibm_runtime import SamplerV2 as SamplerRuntime

        sampler_ibm = SamplerRuntime(mode=session_ibm, options=qiskit_sampler_options)

    return sampler_ibm


def get_default_qiskit_backend_and_pass_manager(
    backend_name: str,
    qubit_mapping_type: Optional[QubitMappingType] = None,
    provider: Optional[QiskitRuntimeService] = None,
    backend_kwargs: Optional[dict] = None,
    pass_manager_kwargs: Optional[dict] = None,
):
    """

    :param provider:
    :param backend_name:
    :param qubit_mapping_type:
    :param backend_kwargs:
    :param pass_manager_kwargs:
    :return:
    """
    if provider is None:
        provider = get_qiskit_provider()

    if qubit_mapping_type is None:
        qubit_mapping_type = QubitMappingType.sabre

    qiskit_backend = get_qiskit_backend(
        backend_name=backend_name,
        backend_kwargs=backend_kwargs,
        qiskit_provider=provider,
    )

    # This pass manager is used to generate the main circuit in mirror circuits experiments
    pass_manager, pass_manager_kwargs = get_qiskit_pass_manager(
        qiskit_backend=qiskit_backend,
        qubit_mapping_type=qubit_mapping_type,
        pass_manager_kwargs=pass_manager_kwargs,
    )

    return qiskit_backend, pass_manager, pass_manager_kwargs


def attempt_to_run_qiskit_circuits(
    circuits_isa: List[CircuitQiskit],
    sampler_ibm: SamplerAer | SamplerRuntime,
    number_of_shots: int,
    max_attempts_run=5,
    metadata_for_error_printing: Optional[Any] = None,
) -> Tuple[
    bool,
    Optional[RuntimeJob | LocalRuntimeJob | PrimitiveJob | QiskitJobHardware],
    Optional[PrimitiveResult],
    Optional[pd.DataFrame],
]:
    """
    Attempt to run a list of circuits on a given sampler, with retries on failure.
    :param circuits_isa:
    :param sampler_ibm:
    :param number_of_shots:
    :param max_attempts_run:
    :param metadata_for_error_printing:
    :return:
    """

    if isinstance(circuits_isa, CircuitQiskit):
        circuits_isa = [circuits_isa]

    _success = False
    job_circuit, results_circuit, df_job_metadata = None, None, None
    for _ in range(0, max_attempts_run):
        try:
            job_circuit = sampler_ibm.run(circuits_isa, shots=number_of_shots)
            t0 = time.perf_counter()
            results_circuit = job_circuit.result()
            t1 = time.perf_counter()
            actual_runtime_wallclock = t1 - t0

            if isinstance(job_circuit, QiskitJobHardware):
                # This is type returned by qiskit_ibm_runtime when run on real device
                session_id = job_circuit.session_id
                estimated_runtime = job_circuit.usage_estimation["quantum_seconds"]
                actual_runtime_QPU = job_circuit.usage()

            elif isinstance(job_circuit, PrimitiveJob):
                # This is type returned by qiskit simulators
                session_id = str(None)
                estimated_runtime = 0.0
                actual_runtime_QPU = actual_runtime_wallclock

            else:
                raise TypeError(
                    f"job_circuit is not of type PrimitiveJob, or RuntimeJobV2. "
                    f"It is of type {type(job_circuit)}"
                )

            job_id = job_circuit.job_id()
            df_job_metadata = pd.DataFrame(
                data={
                    SNV.SessionId.id_long: [session_id],
                    SNV.JobId.id_long: [job_id],
                    "EstimatedRuntime": [estimated_runtime],
                    "ActualRuntimeQPU": [actual_runtime_QPU],
                    "ActualRuntimeWallclock": [actual_runtime_wallclock],
                }
            )

            _success = True
            break
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            raise KeyboardInterrupt("KeyboardInterrupt")
        except Exception as e:

            print("ERROR running circuit:", metadata_for_error_printing)
            print("error message:", e)
            print("Retrying...")

    if not _success:
        print("FAILED TO RUN CIRCUIT:", metadata_for_error_printing)
        print("After ", max_attempts_run, "attempts")

    return _success, job_circuit, results_circuit, df_job_metadata


def get_counts_from_bit_array(
    bit_array: BitArray, return_dict=False
) -> Tuple[np.ndarray, np.ndarray] | Dict[Tuple[int, ...], int]:
    """
    Convert a Qiskit BitArray to a dictionary of counts in a tuple format
    :param bit_array:
    :return:
    (unique_bitstrings, their counts)
    """
    unique_bitstrings, counts = np.unique(
        bit_array.to_bool_array(), axis=0, return_counts=True
    )
    unique_bitstrings = unique_bitstrings.astype(np.int32)

    if not return_dict:
        return unique_bitstrings, counts

    return {
        tuple(bitstring): count
        for bitstring, count in zip(unique_bitstrings.tolist(), counts.tolist())
    }


def get_counts_from_sampler_result(
    sampler_results: SamplerPubResult, return_dict: bool = False
):
    data_bin: DataBin = sampler_results.data
    values = list(data_bin.values())
    assert len(values) == 1, "Only single classical register results are supported"

    return get_counts_from_bit_array(values[0], return_dict=return_dict)


REAL_DEVICES_IBM = [
    "ibm_marrakesh",
    "ibm_torino",
    "ibm_fez",
    "ibm_pittsburgh",
    "ibm_kingston",
]
_SIMULATOR_NAMES = ["aer"]


def get_qiskit_backend(
    backend_name: str,
    backend_kwargs: Optional[dict] = None,
    qiskit_provider=None,
) -> IBMBackend | AerSimulator:
    if backend_name.lower() in REAL_DEVICES_IBM:
        assert (
            qiskit_provider is not None
        ), f"qiskit_provider must be provided when backend_name is {backend_name}"
        if backend_kwargs is None:
            backend_kwargs = DEFAULT_QPU_BACKEND_KWARGS.copy()

        qiskit_backend: IBMBackend = qiskit_provider.backend(
            name=backend_name, **backend_kwargs
        )

    elif backend_name.lower() in _SIMULATOR_NAMES or backend_name.lower()[0:3] == "aer":
        if backend_kwargs is None:
            backend_kwargs = DEFAULT_SIMULATOR_BACKEND_KWARGS.copy()

        qiskit_backend: AerSimulator = AerSimulator(**backend_kwargs)

    else:
        raise ValueError(
            f'Backend {backend_name} not recognised. Available backends: {", ".join(REAL_DEVICES_IBM + _SIMULATOR_NAMES)}'
        )

    return qiskit_backend


def get_qiskit_simulated_backend(
    qiskit_backend: IBMBackend | AerSimulator,
    noiseless_simulation: bool = False,
    simulator_kwargs: Optional[dict] = None,
):
    if simulator_kwargs is None:
        simulator_kwargs = DEFAULT_SIMULATOR_BACKEND_KWARGS.copy()

    if noiseless_simulation:
        simulator_kwargs.update({"noise_model": None})
    else:
        noise_model = simulator_kwargs.get(
            "noise_model", NoiseModel.from_backend(qiskit_backend)
        )
        simulator_kwargs.update({"noise_model": noise_model})

    simulated_backend = AerSimulator.from_backend(qiskit_backend, **simulator_kwargs)

    return simulated_backend


def get_qiskit_pass_manager(
    qiskit_backend: IBMBackend | AerSimulator,
    qubit_mapping_type: QubitMappingType,
    pass_manager_kwargs: Optional[dict] = None,
):
    if pass_manager_kwargs is None:
        pass_manager_kwargs = {}

    pass_manager_kwargs = pass_manager_kwargs.copy()

    if qubit_mapping_type in [QubitMappingType.linear_swap_network]:
        routing_method = pass_manager_kwargs.get("routing_method", "none")
        optimization_level = pass_manager_kwargs.get("optimization_level", 0)
    elif qubit_mapping_type in [QubitMappingType.sabre]:
        routing_method = pass_manager_kwargs.get("routing_method", "sabre")
        optimization_level = pass_manager_kwargs.get("optimization_level", 3)
    elif qubit_mapping_type in [QubitMappingType.fully_connected]:
        routing_method = pass_manager_kwargs.get("routing_method", "sabre")
        optimization_level = pass_manager_kwargs.get("optimization_level", 3)
    else:
        raise NotImplementedError("")

    scheduling_method = pass_manager_kwargs.get("scheduling_method", None)
    seed_transpiler = pass_manager_kwargs.get("seed_transpiler", 42)

    pass_manager_kwargs.update(
        {
            "scheduling_method": scheduling_method,
            "seed_transpiler": seed_transpiler,
            "routing_method": routing_method,
            "optimization_level": optimization_level,
        }
    )

    # This pass manager is used to generate the main circuit in mirror circuits experiments
    pass_manager = generate_preset_pass_manager(
        backend=qiskit_backend, **pass_manager_kwargs
    )

    return pass_manager, pass_manager_kwargs


def convert_hamiltonian_list_representation_to_qiskit_observable(
    hamiltonian_list_representation: List[Tuple[float | int, Tuple[int, ...]]],
    number_of_qubits: Optional[int] = None,
) -> SparsePauliOp:
    """
    A simple function to convert a list of tuples representing a Hamiltonian
    to qiskit's SparsePauliOp object.
    :param hamiltonian_list_representation:
    :param number_of_qubits:
    :return:
    """

    if number_of_qubits is None:
        number_of_qubits = (
            max([max(tup) for _, tup in hamiltonian_list_representation]) + 1
        )

    number_of_terms = len(hamiltonian_list_representation)
    paulis_z = np.zeros((number_of_terms, number_of_qubits), dtype=np.int32)
    paulis_x = np.zeros((number_of_terms, number_of_qubits), dtype=np.int32)
    coeffs = []
    for idx_coeff, (coeff, tup) in enumerate(hamiltonian_list_representation):
        coeffs.append(coeff)
        for idx_qubit in tup:
            paulis_z[idx_coeff, idx_qubit] = 1
            paulis_x[idx_coeff, idx_qubit] = 0

    paulis: PauliList = PauliList.from_symplectic(z=paulis_z, x=paulis_x)
    sparse_pauli = SparsePauliOp(
        data=paulis, coeffs=np.array(coeffs, dtype=np.float32), ignore_pauli_phase=True
    )

    return sparse_pauli


def convert_qiskit_observable_to_hamiltonian_list_representation(
    sparse_observable: SparsePauliOp,
) -> List[Tuple[float, Tuple[int, ...]]]:
    """
    Convert a Qiskit SparsePauliOp hamiltonian to a list representation of Hamiltonian.
    :param sparse_observable:
    :return:
    List of tuples, where each tuple is (coefficient, tuple of qubit indices)

    NOTE: this function ASSUMES that the Hamiltonian is CLASSICAL. (i.e., no X and Y terms)
    """

    hamiltonian_list_representation = []

    coeffs = sparse_observable.coeffs.real

    for idx_term, (coeff, pauli) in enumerate(zip(coeffs, sparse_observable.paulis)):
        z_terms = pauli.z

        nonzero_indices = np.nonzero(z_terms)[0]

        term = (coeff, tuple(nonzero_indices))
        hamiltonian_list_representation.append(term)
    return hamiltonian_list_representation


# def count_gates_per_qubit_in_circuit(quantum_circuit: CircuitQiskit,)->Dict[int,int]:
#     def _filter_function(instr: CircuitInstructionQiskit,
#                          qubit_index: int):
#         if getattr(instr.operation, "_directive", False):
#         for q in instr.qubits:
#             if q._index == qubit_index:
#     for qindex in range(quantum_circuit.num_qubits):
#         if depth_qubit != 0:


def count_gates_per_qubit_in_circuit(
    quantum_circuit: CircuitQiskit, skip_rz_gates: bool = False
) -> Dict[int, int]:
    """
    Count the number of gates per qubit in a quantum circuit.
    :param quantum_circuit: CircuitQiskit object
    :return: Dictionary with qubit indices as keys and number of gates as values
    """
    qubit_depths = {}
    for instr in quantum_circuit.data:
        if getattr(instr.operation, "_directive", False):
            continue
        for q in instr.qubits:
            q: QubitQiskit = q
            if q._index not in qubit_depths:
                qubit_depths[q._index] = 0

            if skip_rz_gates:
                if instr.name.startswith("rz"):
                    continue

            qubit_depths[q._index] += 1
    return qubit_depths



def get_qiskit_provider(account_name: Optional[str] = None,
                        instance_ibm: Optional[str] = None,
                        credentials_path: Optional[str] = None,
                        token:Optional[str]=None,
                        channel:Optional[str]=None
                        ):
    """
    Get the Qiskit Runtime Service provider.
    :param account_name:
    :param instance_ibm:
    :param credentials_path:
    :return:
    """
    from dotenv import load_dotenv
    load_dotenv()
    if credentials_path is None:
        credentials_path = os.getenv('IBM_CREDENTIALS_PATH')
    if account_name is None:
        account_name = os.getenv('IBM_ACCOUNT_NAME')
    if instance_ibm is None:
        instance_ibm = os.getenv('IBM_INSTANCE_NAME')
    if token is None:
        token = os.getenv('IBM_TOKEN')
    if channel is None:
        channel = os.getenv('IBM_CHANNEL')

    return QiskitRuntimeService(name=account_name,
                                filename=credentials_path,
                                instance=instance_ibm,
                                token=token,
                                channel=channel
                                )



def get_physical_qubit_to_classical_bit_mapping_from_circuit(
    quantum_circuit: CircuitQiskit,
):
    from qopt_best_practices.swap_strategies.build_circuit import make_meas_map

    return make_meas_map(circuit=quantum_circuit)


def get_idle_qubit_indices_from_circuit(quantum_circuit: CircuitQiskit):
    dag = circuit_to_dag(quantum_circuit.copy())
    qbits = quantum_circuit.qubits
    idle_qubits = set()
    for wire in dag.idle_wires(ignore=["delay", "barrier"]):
        if wire in qbits:
            idle_qubits.add(wire._index)

    return idle_qubits


def get_gate_counts_dict(qc: CircuitQiskit, integer_indices: bool = True):

    if integer_indices:
        _handler = lambda x: x._index
    else:
        _handler = lambda x: x

    gate_counts_dict = {_handler(qubit): 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_counts_dict[_handler(qubit)] += 1
    return gate_counts_dict


def get_all_qubit_indices_from_circuit(
    quantum_circuit: CircuitQiskit, integer_indices=True
):
    """
    Get all qubits from a quantum circuit.
    :param quantum_circuit: CircuitQiskit object
    :return: List of all qubit indices
    """

    if integer_indices:
        _handler = lambda x: x._index
    else:
        _handler = lambda x: x

    return sorted([_handler(q) for q in quantum_circuit.qubits])


def _get_nontrivial_qubit_indices_from_circuit(
    quantum_circuit: CircuitQiskit, filter_ancillas: bool = True
):
    """
    Get the indices of non-trivial qubits from a quantum circuit.
    A non-trivial qubit is one that is involved in at least one operation in the circuit.
    :param quantum_circuit: CircuitQiskit object
    :return: List of non-trivial qubit indices
    """
    idle_qubit = get_idle_qubit_indices_from_circuit(quantum_circuit=quantum_circuit)
    all_qubits = set(
        get_all_qubit_indices_from_circuit(quantum_circuit=quantum_circuit)
    )
    nontrivial_qubits = sorted(list(all_qubits - idle_qubit))

    ancillas = [qi._index for qi in quantum_circuit.ancillas]

    nontrivial_qubits = (
        [q for q in nontrivial_qubits if q not in ancillas]
        if filter_ancillas
        else nontrivial_qubits
    )

    return nontrivial_qubits


def get_nontrivial_physical_indices_from_circuit(
    quantum_circuit: CircuitQiskit, filter_ancillas=True
):
    """
    Get the indices of non-trivial logical qubits from a quantum circuit.
    A non-trivial logical qubit is one that is involved in at least one operation in the circuit.
    :param quantum_circuit: CircuitQiskit object
    :return: List of non-trivial logical qubit indices

    Note: for circuits with layout, this should return the same as circuit.layout.initial_index_layout(filter_ancillas=True);
    however, as of now (2025.07), that method is bugged and returns IndexError when with flag "filter_ancillas=True".
    """

    # nontrivial_qubits = _get_nontrivial_qubit_indices_from_circuit(quantum_circuit=quantum_circuit,
    #                                                                filter_ancillas=filter_ancillas)

    return list(
        get_physical_qubits_mapping_from_circuit(
            quantum_circuit=quantum_circuit, filter_ancillas=filter_ancillas
        ).values()
    )


def get_physical_qubits_mapping_from_circuit(
    quantum_circuit: CircuitQiskit, filter_ancillas: bool = True
):
    """
    This function maps initial qubits to final qubits, in terms of physical qubit indices.

    :param quantum_circuit:
    :return:
    """

    # physical mapping should be given by the final layout
    nontrivial_physical = _get_nontrivial_qubit_indices_from_circuit(
        quantum_circuit=quantum_circuit, filter_ancillas=filter_ancillas
    )
    # if no layout is present, we return trivial mapping
    if quantum_circuit.layout is None:
        return {i: i for i in nontrivial_physical}

    # TODO(FBM): currently, this assumes that indices in input circuit were just from 0 to n.
    # If we're using transpiler to generate layout, this assumption is usually correct, but not always.

    # this combines two things:
    # 1. Layout mapping = original qubits are changed to some other qubits.
    #   Assumption: original qubits are from 0 to n (where n is total number of qubits in the circuit)
    # 2. Routing mapping = qubits after layout are swapped around to introduce interactions between qubits that are not
    #   physically connected.
    # The following function maps qubits from 0 to n (original), to qubits after Layout AND Routing.
    final_layout = quantum_circuit.layout.final_index_layout(
        filter_ancillas=filter_ancillas
    )

    return {i: final_layout[i] for i in range(len(final_layout))}


##################################
# COUPLING MAPS#

# Those are linear chains on Heron devices that I found by looking at the coupling maps.
_NODES_EXCLUDED_BY_HAND = {
    "ibm_marrakesh": {
        16,
        17,
        18,
        39,
        38,
        37,
        20,
        40,
        56,
        57,
        58,
        79,
        78,
        77,
        60,
        80,
        96,
        97,
        98,
        119,
        118,
        117,
        100,
        120,
        136,
        137,
        138,
    },
    "ibm_torino": {
        18,
        17,
        16,
        34,
        35,
        36,
        56,
        55,
        54,
        72,
        73,
        74,
        94,
        93,
        92,
        110,
        111,
        112,
        132,
        131,
        130,
    },
    "ibm_fez": {
        16,
        17,
        18,
        37,
        38,
        39,
        56,
        57,
        58,
        77,
        78,
        79,
        96,
        97,
        98,
        117,
        118,
        119,
        136,
        137,
        138,
        #
        20,
        40,
        60,
        80,
        100,
        120,
    },
    "ibm_pittsburgh": {
        16,
        17,
        18,
        37,
        38,
        39,
        56,
        57,
        58,
        77,
        78,
        79,
        96,
        97,
        98,
        117,
        118,
        119,
        136,
        137,
        138,
        #
        20,
        40,
        60,
        80,
        100,
        120,
    },
    "ibm_kingston": {
        16,
        17,
        18,
        37,
        38,
        39,
        56,
        57,
        58,
        77,
        78,
        79,
        96,
        97,
        98,
        117,
        118,
        119,
        136,
        137,
        138,
        #
        20,
        40,
        60,
        80,
        100,
        120,
    },
}
_DEVICE_SIZES = {
    "ibm_brisbane": 127,
    "ibm_torino": 133,
    "ibm_fez": 156,
    "ibm_marrakesh": 156,
    "ibm_pittsburgh": 156,
    "ibm_kingston": 156,
}

_LINEAR_CHAIN_SIZES_BY_HAND = {
    _bck_name: _DEVICE_SIZES[_bck_name] - len(_NODES_EXCLUDED_BY_HAND[_bck_name])
    for _bck_name in _NODES_EXCLUDED_BY_HAND.keys()
}
_LINEAR_CHAIN_SIZES_BY_HAND = {
    "ibm_marrakesh": 129,
    "ibm_torino": 112,
    "ibm_fez": 129,
    "ibm_pittsburgh": 129,
    "ibm_kingston": 129,
}


def filter_coupling_map(
    coupling_map: CouplingMap,
    nodes_to_exclude: Optional[Set[int]] = None,
):
    if nodes_to_exclude is None:
        return coupling_map

    # Create a new CouplingMap object
    edges = coupling_map.get_edges()
    edges_filtered = [
        [i, j]
        for (i, j) in edges
        if i not in nodes_to_exclude and j not in nodes_to_exclude
    ]

    filtered_coupling_map = CouplingMap(couplinglist=edges_filtered)
    filtered_coupling_map.make_symmetric()

    return filtered_coupling_map


def get_nodes_from_coupling_map(coupling_map: CouplingMap):
    edges = coupling_map.get_edges()
    return set([edge[0] for edge in edges] + [edge[1] for edge in edges])


def get_gate_count_from_logical_gates_qiskit(
    gate_builder: AbstractProgramGateBuilder, gate_name: str
):
    if gate_name[0] != "_":
        gate_name = "_" + gate_name

    try:

        quantum_circuit: CircuitQiskit = getattr(gate_builder, gate_name)()
    except TypeError:
        quantum_circuit: CircuitQiskit = getattr(gate_builder, gate_name)(np.pi / 4)

    gates_dict = {"q0": [], "q1": [], "q0q1": []}

    for instr in quantum_circuit.data:
        if len(instr.qubits) == 2:
            gates_dict["q0q1"].append(instr.name)
        elif len(instr.qubits) == 1:
            if instr.qubits[0]._index == 0:
                gates_dict["q0"].append(instr.name)
            elif instr.qubits[0]._index == 1:
                gates_dict["q1"].append(instr.name)
            else:
                raise ValueError(f"Qubit index {instr.qubits[0]._index} is not 0 or 1")
        else:
            raise ValueError(f"Instruction {instr} has more than 2 qubits")
    return gates_dict


def get_qaoa_qaoa_fidelities(
    gate_builder: AbstractProgramGateBuilder,
    linear_chain: List[int],
    qiskit_backend: Union[IBMBackend, AerSimulator],
    phase_gate="exp_ZZ_SWAP",
    mixer_gate="exp_X",
    qaoa_depth=1,
    time_block_size=None,
    verbosity=1,
):
    gate_count_ZZ_SWAP = get_gate_count_from_logical_gates_qiskit(
        gate_builder=gate_builder, gate_name=phase_gate
    )
    gate_count_RX = get_gate_count_from_logical_gates_qiskit(
        gate_builder=gate_builder, gate_name=mixer_gate
    )

    bck_properties: BackendProperties = qiskit_backend.properties()

    lc = linear_chain

    edge_chains = [(lc[i], lc[i + 1]) for i in range(0, len(lc) - 1, 2)] + [
        (lc[i], lc[i + 1]) for i in range(1, len(lc) - 1, 2)
    ]

    fidelity_phase = 1.0
    for edge in edge_chains:
        q0, q1 = edge
        q0_gates = gate_count_ZZ_SWAP["q0"]
        q1_gates = gate_count_ZZ_SWAP["q1"]
        q0q1_gates = gate_count_ZZ_SWAP["q0q1"]
        for gate in q0_gates:
            err = bck_properties.gate_error(gate=gate, qubits=[q0])
            if err == 1.0:
                if verbosity > 0:
                    print(f"DATA FOR: {gate} on {q0} NOT AVAILABLE")

                continue
            fidelity_phase *= 1 - err
        for gate in q1_gates:
            err = bck_properties.gate_error(gate=gate, qubits=[q1])
            if err == 1.0:
                if verbosity > 0:
                    print(f"DATA FOR: {gate} on {q1} NOT AVAILABLE")
                continue
            fidelity_phase *= 1 - err
        for gate in q0q1_gates:
            err = bck_properties.gate_error(gate=gate, qubits=[q0, q1])
            if err == 1.0:
                if verbosity > 0:
                    print(f"DATA FOR: {gate} on {(q0, q1)} NOT AVAILABLE")
                continue
            fidelity_phase *= 1 - err
    fidelity_mixer = 1.0
    for qi in lc:
        q0_gates = gate_count_RX["q0"]
        for gate in q0_gates:
            err = bck_properties.gate_error(gate=gate, qubits=[qi])
            if err == 1.0:
                continue
            fidelity_mixer *= 1 - err

    number_of_qubits = len(lc)
    if time_block_size is None:
        time_block_size = number_of_qubits

    double_linear_chains_per_single_layer = time_block_size / 2
    fidelity_phase = fidelity_phase ** (
        qaoa_depth * double_linear_chains_per_single_layer
    )
    fidelity_mixer = fidelity_mixer ** (qaoa_depth)

    return fidelity_phase, fidelity_mixer


def find_longest_qubits_chain(qiskit_backend: IBMBackend | AerSimulator):
    cm = qiskit_backend.coupling_map
    backend_name = qiskit_backend.name

    if backend_name in _NODES_EXCLUDED_BY_HAND:
        cm = filter_coupling_map(
            cm, nodes_to_exclude=_NODES_EXCLUDED_BY_HAND[backend_name]
        )

    return rx.longest_simple_path(cm.graph)


def find_all_qubits_chains(
    qiskit_backend: IBMBackend | AerSimulator,
    number_of_qubits: int,
    coupling_map: Optional[CouplingMap] = None,
) -> List[List[int]]:
    backend_name = qiskit_backend.name

    if coupling_map is None:
        coupling_map = qiskit_backend.coupling_map
        if coupling_map is None:
            best_qubits = list(range(number_of_qubits))
            # best_chains = ([(best_qubits[i], best_qubits[i+1]) for i in range(len(best_qubits)-1)],
            #                  [(best_qubits[i], best_qubits[i+1]) for i in range(1, len(best_qubits)-2)])
            return [best_qubits]

        if backend_name in _NODES_EXCLUDED_BY_HAND:
            coupling_map = filter_coupling_map(
                coupling_map, nodes_to_exclude=_NODES_EXCLUDED_BY_HAND[backend_name]
            )

    max_nodes = len(get_nodes_from_coupling_map(coupling_map))
    if number_of_qubits > max_nodes:
        raise ValueError(
            f"Number of qubits {number_of_qubits} "
            f"exceeds the maximum number of nodes {max_nodes} in the coupling map."
        )
    if not coupling_map.is_symmetric:
        coupling_map.make_symmetric()

    # BELOW CODE IS COPIED FROM "qopt_best_practices", see: https://github.com/qiskit-community/qopt-best-practices

    all_paths = rx.all_pairs_all_simple_paths(
        coupling_map.graph,
        min_depth=number_of_qubits,
        cutoff=number_of_qubits,
    ).values()

    paths = np.asarray(
        [
            (list(c), list(sorted(list(c))))
            for a in iter(all_paths)
            for b in iter(a)
            for c in iter(a[b])
        ]
    )

    # filter out duplicated paths
    _, unique_indices = np.unique(paths[:, 1], return_index=True, axis=0)
    all_lines = paths[:, 0][unique_indices].tolist()

    return all_lines


def _get_backend_data_path_standardized(backend_name: str):
    experiment_folders_hierarchy = [
        "BackendInformation",
        f"{SNV.Backend.id}{MAIN_KEY_VALUE_SEPARATOR}{backend_name}",
    ]

    return experiment_folders_hierarchy


def find_and_save_best_linear_chains_heron(
    qiskit_backend: IBMBackend | AerSimulator,
    number_of_qubits: int,
    backend_name: Optional[str] = None,
    gate_builder_class_qiskit: Type[
        NativeGateBuilderHeronCustomizable
    ] = NativeGateBuilderHeronCustomizable,
    verbosity=0,
):

    if backend_name in REAL_DEVICES_IBM:

        def _fidelity_function(chain, gate_builder: AbstractProgramGateBuilder):
            fidelity_phase, fidelity_mixer = get_qaoa_qaoa_fidelities(
                qiskit_backend=qiskit_backend,
                gate_builder=gate_builder,
                linear_chain=chain,
                phase_gate="exp_ZZ_SWAP",
                mixer_gate="exp_X",
                qaoa_depth=1,
                time_block_size=number_of_qubits,
                verbosity=verbosity,
            )

            return fidelity_phase * fidelity_mixer

    elif backend_name[0:3] in ["aer"]:

        def _fidelity_function(x, y):
            return 1.0

    else:
        raise NotImplementedError(f"Backend {backend_name} is not implemented. ")

    qubits_chains = find_all_qubits_chains(
        qiskit_backend=qiskit_backend, number_of_qubits=number_of_qubits
    )

    # TODO(FBM): This is specific to heron devices, but rest of the function is not. Improve this
    gate_builder_wfg = gate_builder_class_qiskit(use_fractional_gates=True)
    qubits_chains_sorted_wfg = sorted(
        qubits_chains,
        key=lambda x: _fidelity_function(x, gate_builder_wfg),
        reverse=True,
    )

    gate_builder_nwfg = gate_builder_class_qiskit(use_fractional_gates=False)
    qubits_chains_sorted_nwfg = sorted(
        qubits_chains,
        key=lambda x: _fidelity_function(x, gate_builder_nwfg),
        reverse=True,
    )

    best_chain_wfg = qubits_chains_sorted_wfg[0]
    best_chain_nwfg = qubits_chains_sorted_nwfg[0]
    best_fidelity_wfg = _fidelity_function(best_chain_wfg, gate_builder_wfg)
    best_fidelity_nwfg = _fidelity_function(best_chain_nwfg, gate_builder_nwfg)

    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    hour = now.strftime("%H:%M:%S")

    if backend_name is None:
        backend_name = qiskit_backend.name

    df_chain_wfg = pd.DataFrame(
        data={
            "backend_name": [backend_name],
            "gate_builder_name": [gate_builder_class_qiskit.__name__],
            SNV.NumberOfQubits.id_long: [number_of_qubits],
            "with_fractional_gates": [True],
            "predicted_QAOA_fidelity": [best_fidelity_wfg],
            "qubits_chain": [list(best_chain_wfg)],
            "date": [today],
            "time": [hour],
        }
    )
    df_chain_nwfg = pd.DataFrame(
        data={
            "backend_name": [backend_name],
            "gate_builder_name": [gate_builder_class_qiskit.__name__],
            SNV.NumberOfQubits.id_long: [number_of_qubits],
            "with_fractional_gates": [False],
            "predicted_QAOA_fidelity": [best_fidelity_nwfg],
            "qubits_chain": [list(best_chain_nwfg)],
            "date": [today],
            "time": [hour],
        }
    )

    df_chains = pd.concat([df_chain_wfg, df_chain_nwfg], axis=0)

    experiment_folders_hierarchy = _get_backend_data_path_standardized(
        backend_name=backend_name
    )
    res_writer = ResultsLogger(
        experiment_folders_hierarchy=experiment_folders_hierarchy,
        table_name_prefix="BestLinearChains",
        experiment_set_name=f"BackendData-{backend_name}",
        experiment_set_id=f"BackendData-{backend_name}",
        experiment_instance_id="BestLinearChains",
    )
    res_writer.write_metadata(
        metadata=df_chains,
        data_type=SNDT.BackendData,
        shared_across_experiment_set=False,
        annotate_with_experiment_metadata=False,
        ignore_logging_level=True,
    )


def read_best_linear_chains(
    number_of_qubits: int,
    backend_name: Optional[str] = None,
    qiskit_backend: Optional[IBMBackend | AerSimulator] = None,
    date: Optional[str] = None,
    time_cutoff: Optional[datetime.datetime] = None,
    gate_builder_class_qiskit: Type[
        NativeGateBuilderHeronCustomizable
    ] = NativeGateBuilderHeron,
    verbosity=0,
):
    if backend_name is None:
        assert (
            qiskit_backend is not None
        ), "backend_name or qiskit_backend must be provided"
        backend_name = qiskit_backend.name

    experiment_folders_hierarchy = _get_backend_data_path_standardized(
        backend_name=backend_name
    )
    res_reader = ResultsLogger(
        experiment_folders_hierarchy=experiment_folders_hierarchy,
        table_name_prefix="BestLinearChains",
        experiment_set_name=f"BackendData-{backend_name}",
        experiment_set_id=f"BackendData-{backend_name}",
        experiment_instance_id="BestLinearChains",
    )

    if date is None:
        if backend_name in REAL_DEVICES_IBM:
            assert (
                qiskit_backend is not None
            ), "time_cutoff must be provided if qiskit_backend is not provided"
            date = qiskit_backend.properties(refresh=True).last_update_date.date()

            if time_cutoff is None:
                # if date from backend is not today, we should set time cutoff to midnight
                today = datetime.datetime.now().date()
                if date < today:
                    time_cutoff = datetime.time(0, 0, 0)
                else:
                    # otherwise, we take calibration time
                    time_cutoff = qiskit_backend.properties(
                        refresh=True
                    ).last_update_date.time()

        else:
            date = datetime.datetime.now().strftime("%Y-%m-%d")

    # if time_cutoff is None:
    #     if backend_name in _REAL_DEVICES:
    #         assert qiskit_backend is not None, 'time_cutoff must be provided if qiskit_backend is not provided'
    #
    #
    #
    #

    def _try_to_find_and_save_best_linear_chains():
        assert (
            qiskit_backend is not None
        ), "NO DATA FOUND. qiskit_backend must be provided when no data was gathered for given day"

        print("Trying to find and save best linear chains")

        find_and_save_best_linear_chains_heron(
            qiskit_backend=qiskit_backend,
            number_of_qubits=number_of_qubits,
            backend_name=backend_name,
            gate_builder_class_qiskit=gate_builder_class_qiskit,
            verbosity=verbosity,
        )
        return read_best_linear_chains(
            number_of_qubits=number_of_qubits,
            backend_name=backend_name,
            qiskit_backend=qiskit_backend,
            date=date,
            time_cutoff=time_cutoff,
            gate_builder_class_qiskit=gate_builder_class_qiskit,
            verbosity=verbosity,
        )

    try:
        df_chains = res_reader.read_metadata(
            data_type=SNDT.BackendData, shared_across_experiment_set=False
        )
    except FileNotFoundError:
        print("DIDNT FIND FILE!")
        return _try_to_find_and_save_best_linear_chains()

    df_chains = df_chains[df_chains[SNV.NumberOfQubits.id_long] == number_of_qubits]
    if df_chains.empty:
        print("NO DATA FOR NUMBER OF QUBITS:", number_of_qubits)
        return _try_to_find_and_save_best_linear_chains()

    df_chains["date"] = pd.to_datetime(df_chains["date"], format="%Y-%m-%d").dt.date
    df_chains = df_chains[df_chains["date"] >= date]

    if df_chains.empty:
        print("NO DATA FOR DATE:", date)
        return _try_to_find_and_save_best_linear_chains()

    df_chains = df_chains[df_chains["backend_name"] == backend_name]
    df_chains = df_chains[
        df_chains["gate_builder_name"] == gate_builder_class_qiskit.__name__
    ]
    if df_chains.empty:
        print(
            "NO DATA FOR BACKEND NAME:",
            backend_name,
            "and gate_builder_name:",
            gate_builder_class_qiskit.__name__,
        )
        return _try_to_find_and_save_best_linear_chains()

    if time_cutoff is not None:

        df_chains["time"] = pd.to_datetime(df_chains["time"], format="%H:%M:%S").dt.time

        # We want only newest data
        df_chains = df_chains[df_chains["time"] >= time_cutoff]
        if df_chains.empty:
            print("NO DATA AFTER CUTOFF:", time_cutoff)
            return _try_to_find_and_save_best_linear_chains()

    # sort w.r.t. predicted fidelity
    df_chains = df_chains.sort_values(
        by="predicted_QAOA_fidelity", ascending=False
    ).reset_index(drop=True)

    df_chains["qubits_chain"] = df_chains["qubits_chain"].apply(eval)

    return df_chains

    #
    # for i in tqdm(list(range(1, max_nodes + 1, 1))[::-1], position=0, desc='Finding longest chains', colour='green'):
    #         return get_qubits_chains(qiskit_backend=qiskit_backend,
    #                                  coupling_map=cm)
    #     except(IndexError):
    # continue

    #
    # for sub in all_lines:
    #     fid = qopt_qubit_selection.evaluate_fidelity(path=sub,
    #                                                  edges=lin_1 + lin_2)


def find_best_linear_chain_qiskit(
    qiskit_backend,
    number_of_qubits: int,
    # backend_name: str,
    use_fractional_gates: bool = False,
    recommend_whether_to_use_fractional_gates: bool = False,
    gate_builder_class_qiskit=NativeGateBuilderHeronCustomizable,
):
    # READING BACKEND DATA
    df_chains = read_best_linear_chains(
        qiskit_backend=qiskit_backend,
        number_of_qubits=number_of_qubits,
        backend_name=qiskit_backend.name,
        date=None,
        time_cutoff=None,
        gate_builder_class_qiskit=gate_builder_class_qiskit,
    )

    df_best_chain = df_chains.iloc[0]
    use_fractional_gates_recommended = df_best_chain["with_fractional_gates"]
    qubit_indices_physical = df_best_chain["qubits_chain"]

    # Our linear chain finder tests both "use_fractional_gates=True" and "use_fractional_gates=False" and provides both values
    # Here we can choose to reinitialize the backend_computation with the recommended value of use_fractional_gates
    use_fractional_gates = use_fractional_gates
    if recommend_whether_to_use_fractional_gates:
        if anf.query_yes_no(
            f"Backend data suggest to change use_fractional_gates parameter {use_fractional_gates}-->{use_fractional_gates_recommended}. Should we change?"
        ):
            print(
                "OK, reinitailizing the backend_computation with the new value of use_fractional_gates."
            )
            use_fractional_gates = use_fractional_gates_recommended
        else:
            print(
                "Keeping the original value of use_fractional_gates:",
                use_fractional_gates,
            )

    return qubit_indices_physical, use_fractional_gates


#
# def recompile_until_no_ancilla_qubits(quantum_circuit: CircuitQiskit,
#                                       expected_number_of_qubits: int,
#                                       pass_manager: StagedPassManager,
#                                       ):
#     """
#     Recompile a quantum circuit until it does not use ancillas (i.e., it has the expected number of qubits).
#     :param quantum_circuit:
#     :param expected_number_of_qubits:
#     :param pass_manager:
#     :param max_trials:
#     :param enforce_no_ancilla_qubits:
#     #helper flag, if False, the function will return the circuit as is, without recompiling it.
#
#     :return:
#     """
#
#     for trial_index in range(max_trials):
#         if not enforce_no_ancilla_qubits:
#
#             quantum_circuit=recompiled_circuit))
#
#         if number_of_qubits_circuit == expected_number_of_qubits:
#
