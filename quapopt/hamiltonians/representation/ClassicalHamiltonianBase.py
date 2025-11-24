# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import copy
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pydantic as pyd

from quapopt import ancillary_functions as anf
from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.data_analysis.data_handling import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling import (
    HamiltonianClassSpecifierGeneral,
    HamiltonianInstanceSpecifierGeneral,
)
from quapopt.hamiltonians.representation import (
    HamiltonianListRepresentation,
    convert_list_representation_to_adjacency_matrix,
)
from quapopt.hamiltonians.representation.transformations import (
    HamiltonianTransformation,
    apply_bitflip_to_hamiltonian,
    apply_permutation_to_hamiltonian,
    concatenate_hamiltonian_transformations,
)

_REPRESENTATION_DESCRIPTION_LIST = """Classical Hamiltonian represented as a list of interactions.
                                     Each interaction is a tuple of the form (c, (i,j,k,...)) 
                                     where i, j, k, ... are qubit indices
                                     The length of the tuple determines the locality of the interaction
                                     c is the coefficient of the interaction term in the Hamiltonian
                                     """

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp


class ClassicalHamiltonianBase:
    # TODO(FBM): Perhaps should create separate child class for 2-local Hamiltonians
    """
    Base class for classical Hamiltonians.
    The class handles different Hamiltonian representations and transformations.
    """

    _hamiltonian: list[tuple[float, tuple[int, ...]]]
    _representation_description: Optional[str]
    _localities: List[int]
    _number_of_qubits: int
    _applied_transformations: list[HamiltonianTransformation]
    _evaluate_energy_function: Callable
    _couplings: Optional[np.ndarray]
    _local_fields: Optional[np.ndarray]

    def __init__(
        self,
        hamiltonian: HamiltonianListRepresentation,
        number_of_qubits: int,
        representation_description: Optional[str] = None,
        solve_at_initialization=False,
        hamiltonian_class_specifier: Optional[HamiltonianClassSpecifierGeneral] = None,
        hamiltonian_instance_specifier: Optional[
            HamiltonianInstanceSpecifierGeneral
        ] = None,
        known_energies_dict=None,
        class_specific_data: Optional[dict] = None,
        default_backend: str = None,
    ):
        """
        Initializes the Hamiltonian object. The Hamiltonian can be provided as a networkit graph or as a list of interactions.
        The properties of the Hamiltonian are inferred from the input.

        :param hamiltonian List[Tuple[Union[float,int], Tuple[int, ...]]]: Description of the Hamiltonian.
        :param representation_description (Optional[str]): Description of the Hamiltonian representation.
        """

        try:
            hamiltonian = hamiltonian.hamiltonian_list_representation
        except AttributeError:
            try:
                hamiltonian = hamiltonian[0].hamiltonian_list_representation
            except (AttributeError, IndexError):
                pass

        assert isinstance(
            hamiltonian, list
        ), "Hamiltonian must be a list of interactions"

        if isinstance(hamiltonian, list):
            hamiltonian = sorted(hamiltonian, key=lambda x: x[1])

        self._hamiltonian = hamiltonian
        if representation_description is None:
            if isinstance(self._hamiltonian, list):
                self._representation_description = _REPRESENTATION_DESCRIPTION_LIST
            else:
                raise ValueError(
                    "Hamiltonian must be a networkit graph or a list of interactions"
                    "or description must be provided."
                )

        self._representation_description = representation_description
        self._hamiltonian_class_specifier = hamiltonian_class_specifier
        self._hamiltonian_instance_specifier = hamiltonian_instance_specifier

        self._localities = sorted(
            list(set(([len(interaction[1]) for interaction in self._hamiltonian])))
        )

        if set(self.localities) in [{2}, {1, 2}, {1}]:
            self._is_two_local = True
        else:
            self._is_two_local = False

        self._number_of_qubits = int(number_of_qubits)
        self._applied_transformations = []

        _spectrum = None
        _lowest_energy = None
        _lowest_energy_state = None
        _highest_energy = None
        _highest_energy_state = None

        if known_energies_dict is not None:
            if "spectrum" in known_energies_dict:
                _spectrum = known_energies_dict["spectrum"]
            if "lowest_energy" in known_energies_dict:
                _lowest_energy = known_energies_dict["lowest_energy"]
            if "lowest_energy_state" in known_energies_dict:
                _lowest_energy_state = known_energies_dict["lowest_energy_state"]

            if "highest_energy" in known_energies_dict:
                _highest_energy = known_energies_dict["highest_energy"]
            if "highest_energy_state" in known_energies_dict:
                _highest_energy_state = known_energies_dict["highest_energy_state"]

        self._spectrum = _spectrum
        self._lowest_energy = _lowest_energy
        self._lowest_energy_state = _lowest_energy_state
        self._highest_energy = _highest_energy
        self._highest_energy_state = _highest_energy_state

        self._class_specific_data = class_specific_data

        if default_backend is None:
            from quapopt import AVAILABLE_SIMULATORS

            default_backend = "numpy"
            if "cupy" in AVAILABLE_SIMULATORS and self._number_of_qubits > 50:
                default_backend = "cupy"

        self._default_backend = default_backend

        self._evaluate_energy_function = None
        self._couplings = None
        self._local_fields = None

        self._update_hamiltonian_representation(new_representation=self._hamiltonian)

        if solve_at_initialization:
            self.solve_hamiltonian()

        _two_local_properties = None
        if self._is_two_local:
            _two_local_properties = {}
            _two_local_properties["number_of_edges"] = len(
                [s for s in self._hamiltonian if len(s[1]) == 2]
            )
            _two_local_properties["average_degree"] = (
                _two_local_properties["number_of_edges"] / self.number_of_qubits
            )
            _two_local_properties["density"] = _two_local_properties[
                "number_of_edges"
            ] / (self.number_of_qubits * (self.number_of_qubits - 1) / 2)

        self._two_local_properties = _two_local_properties

    def __repr__(self):
        return (
            f"Classical Hamiltonian with {self.number_of_qubits} qubits\n"
            f"Localities: {self.localities}\n"
            f"Class: {self._hamiltonian_class_specifier.get_description_string()}\n"
            f"Instance: {self._hamiltonian_instance_specifier.get_description_string()}\n"
            f"Known spectral data: {self.get_known_energies_dict()}"
            f"Number of applied transformations: {len(self.applied_transformations)}"
        )

    def get_known_energies_dict(self):
        known_energies_dict = {
            "spectrum": self._spectrum,
            "lowest_energy": self._lowest_energy,
            "lowest_energy_state": self._lowest_energy_state,
            "highest_energy": self._highest_energy,
            "highest_energy_state": self._highest_energy_state,
        }
        return known_energies_dict

    @property
    def hamiltonian(self) -> List[Tuple[Union[float, int], Tuple[int, ...]]]:
        return self._hamiltonian

    @property
    def hamiltonian_class_specifier(self) -> Optional[HamiltonianClassSpecifierGeneral]:
        return self._hamiltonian_class_specifier

    @property
    def hamiltonian_class_description(self, long_strings=False) -> Optional[str]:
        hcs = self._hamiltonian_class_specifier

        if hcs is None:
            return None
        return hcs.get_description_string(long_strings=long_strings)

    @property
    def hamiltonian_instance_specifier(
        self,
    ) -> Optional[HamiltonianInstanceSpecifierGeneral]:
        return self._hamiltonian_instance_specifier

    @property
    def hamiltonian_instance_description(self) -> Optional[str]:
        his = self._hamiltonian_instance_specifier
        if his is None:
            return None
        return his.get_description_string()

    @property
    def class_specific_information(self) -> Optional[Any]:
        return self._class_specific_data

    @property
    def spectrum(self):
        return self._spectrum

    @property
    def localities(self) -> List[int]:
        return self._localities

    @property
    def number_of_qubits(self) -> int:
        return self._number_of_qubits

    @property
    def representation_description(self) -> Optional[str]:
        return self._representation_description

    @property
    def applied_transformations(self) -> List[HamiltonianTransformation]:
        return self._applied_transformations

    @property
    def lowest_energy(self):
        return self._lowest_energy

    @property
    def ground_state_energy(self):
        return self._lowest_energy

    @property
    def highest_energy(self):
        return self._highest_energy

    @property
    def local_fields(self):
        # if 1 in self.localities:
        return self._local_fields

    @property
    def couplings(self):
        return self._couplings

    @property
    def default_backend(self):
        return self._default_backend

    # @default_backend.setter
    # def default_backend(self,
    #                     backend_computation:str):

    @property
    def is_two_local(self):
        return self._is_two_local

    @property
    def number_of_edges(self):
        if self.is_two_local:
            return self._two_local_properties["number_of_edges"]
        return None

    @property
    def average_degree(self):
        if self.is_two_local:
            return self._two_local_properties["average_degree"]
        return None

    @property
    def density(self):
        if self.is_two_local:
            return self._two_local_properties["density"]
        return None

    def update_hamiltonian_representation(
        self, new_representation: HamiltonianListRepresentation
    ):
        """
        WARNING: this assumes we DO NOT CHANGE the localities of the Hamiltonian!
        This class is useful when applying gauge transformations to the same Hamiltonian.
        :param new_representation:
        :return:
        """
        self._update_hamiltonian_representation(new_representation=new_representation)

    def _reinitialize_evaluate_function_energy(self):
        localities_set = set(self._localities)

        if {1, 2} == localities_set or {1} == localities_set:

            def _wrapped_eval(
                bitstrings_array,
                pm_input: bool = False,
                backend_computation=self._default_backend,
                backend_output=self._default_backend,
            ):
                return em.calculate_energies_from_bitstrings_2_local(
                    couplings_array=self._couplings,
                    local_fields=self._local_fields,
                    bitstrings_array=bitstrings_array,
                    local_fields_present=True,
                    pm_input=pm_input,
                    computation_backend=backend_computation,
                    output_backend=backend_output,
                )

            self._evaluate_energy_function = _wrapped_eval

        elif {2} == localities_set:

            def _wrapped_eval(
                bitstrings_array,
                pm_input: bool = False,
                backend_computation=self._default_backend,
                backend_output=self._default_backend,
            ):
                return em.calculate_energies_from_bitstrings_2_local(
                    couplings_array=self._couplings,
                    local_fields=None,
                    bitstrings_array=bitstrings_array,
                    local_fields_present=False,
                    pm_input=pm_input,
                    computation_backend=backend_computation,
                    output_backend=backend_output,
                )

            self._evaluate_energy_function = _wrapped_eval
        else:
            # TODO(FBM): Refactor this for higher-locality hamiltonians
            def _wrapped_eval(
                bitstrings_array,
                pm_input=False,
                backend_computation=self._default_backend,
                backend_output=self._default_backend,
            ):
                if pm_input:
                    raise NotImplementedError(
                        "pm_input is not supported for K-local Hamiltonians"
                    )
                return em.calculate_energies_from_bitstrings(
                    hamiltonian=self._hamiltonian,
                    bitstrings_array=bitstrings_array,
                    backend_output=backend_output,
                    backend_computation=backend_computation,
                )

            self._evaluate_energy_function = _wrapped_eval

    def reinitialize_backend(self, backend: str):

        if self._default_backend == backend:
            return

        if self._couplings is not None:
            couplings, local_fields = self.get_couplings_and_local_fields(
                matrix_type="SYM", backend=backend
            )
            self._couplings = couplings
            self._local_fields = local_fields
        self._default_backend = backend
        self._reinitialize_evaluate_function_energy()

    def _update_hamiltonian_representation(
        self,
        new_representation: HamiltonianListRepresentation,
        bitflip_for_cost_function_initialization: Tuple[int, ...] = None,
    ):
        """
        WARNING: this assumes we DO NOT CHANGE the localities of the Hamiltonian!
        This class is useful when applying gauge transformations to the same Hamiltonian.
        :param new_representation:
        :param bitflip_for_cost_function_initialization:
        :return:
        """

        if bitflip_for_cost_function_initialization is None:
            if self.is_two_local:
                couplings, local_fields = self.get_couplings_and_local_fields(
                    matrix_type="SYM", backend=self._default_backend
                )
                self._couplings = couplings
                self._local_fields = local_fields

            self._hamiltonian = new_representation
            self._reinitialize_evaluate_function_energy()

        elif not self.is_two_local:
            self._hamiltonian = new_representation
            self._reinitialize_evaluate_function_energy()

        else:
            # TODO(FBM): add more efficient version for 2-local Hamiltonians!
            # This is a more efficient version special for two-local Hamiltonians
            if self.default_backend == "cupy":
                bck = cp
            elif self.default_backend == "numpy":
                bck = np
            else:
                raise ValueError("Backend not recognized")
            bitflip_pm = 1 - 2 * bck.array(bitflip_for_cost_function_initialization)
            bitflip_outer = bck.outer(bitflip_pm, bitflip_pm)
            if self.couplings is not None:
                new_couplings = self.couplings * bitflip_outer
                self._couplings = new_couplings
            if self.local_fields is not None:
                new_fields = self.local_fields * bitflip_pm
                self._local_fields = new_fields

            self._reinitialize_evaluate_function_energy()
            self._hamiltonian = new_representation

        if self._spectrum is not None:
            self.solve_hamiltonian(both_directions=True)

    def get_hamiltonian_dictionary(self):
        return {qubits: weight for weight, qubits in self.hamiltonian}

    def get_adjacency_matrix(
        self,
        matrix_type: str = "SYM",
        backend: Optional[str] = None,
        precision=np.float32,
    ) -> Union[np.ndarray, cp.ndarray]:
        assert set(self.localities) in [
            {1},
            {2},
            {1, 2},
        ], "Adjacency matrix can only be obtained for 2-local Hamiltonians"
        if backend is None:
            backend = self._default_backend

        couplings = self.couplings
        local_fields = self.local_fields

        if couplings is None or (local_fields is None and 1 in self.localities):
            return convert_list_representation_to_adjacency_matrix(
                hamiltonian_list_representation=self.hamiltonian,
                matrix_type=matrix_type,
                backend=backend,
                number_of_qubits=self.number_of_qubits,
                precision=precision,
            )

        if backend == "numpy":
            import numpy as bck
        elif backend == "cupy":
            import cupy as bck
        else:
            raise ValueError("Backend not recognized")

        couplings = anf.convert_cupy_numpy_array(
            array=couplings, output_backend=backend
        )

        adjacency_matrix = bck.array(couplings.copy())
        if 1 in self.localities:
            local_fields = anf.convert_cupy_numpy_array(
                array=local_fields, output_backend=backend
            )
            bck.fill_diagonal(adjacency_matrix, local_fields)

        return adjacency_matrix

    def get_couplings_and_local_fields(
        self,
        matrix_type: str = "SYM",
        backend: Optional[str] = None,
        precision: type = np.float32,
    ):
        assert set(self.localities) in [
            {1},
            {2},
            {1, 2},
        ], "Adjacency matrix can only be obtained for 2-local Hamiltonians"

        if backend is None:
            backend = self._default_backend
        couplings = self.get_adjacency_matrix(
            backend=backend, matrix_type=matrix_type, precision=precision
        )
        local_fields = np.diag(couplings).copy()
        np.fill_diagonal(couplings, 0)

        if np.all(local_fields == 0):
            local_fields = None

        return couplings, local_fields

    def copy(self):
        return copy.deepcopy(self)

    def solve_hamiltonian(
        self, both_directions=True, solver_kwargs=None, verbose=False
    ):
        from quapopt import AVAILABLE_SIMULATORS

        if self.is_two_local:
            # TODO(FBM) DO NOT IGNORE SOLVER KWARGS
            if self.number_of_qubits <= 23:
                if "cuda" in AVAILABLE_SIMULATORS:
                    self._spectrum = em.cuda_solve_hamiltonian(self._hamiltonian)
                else:
                    self._spectrum = em.solve_hamiltonian_python(self)

                argmin_spectrum = np.argmin(self._spectrum)

                self._lowest_energy = self._spectrum[argmin_spectrum]
                self._lowest_energy_state = anf.convert_int_to_binary_tuple(
                    integer=argmin_spectrum, number_of_bits=self.number_of_qubits
                )
                argmax_spectrum = np.argmax(self._spectrum)
                self._highest_energy = self._spectrum[argmax_spectrum]
                self._highest_energy_state = anf.convert_int_to_binary_tuple(
                    integer=argmax_spectrum, number_of_bits=self.number_of_qubits
                )
            else:
                if solver_kwargs is None:
                    solver_kwargs = {"solver_name": "BURER2002", "solver_timeout": 1}
                self._spectrum = None

                from quapopt.optimization.classical_solvers.mqlib_solvers import (
                    solve_ising_hamiltonian_mqlib,
                )

                _adj_matrix = self.get_adjacency_matrix(
                    matrix_type="SYM", backend="numpy"
                )
                if verbose:
                    print("SOLVING HAMILTONIAN with:", solver_kwargs)

                (lowest_energy_state, lowest_energy_value), opt_res_low = (
                    solve_ising_hamiltonian_mqlib(
                        hamiltonian=_adj_matrix,
                        solver_kwargs=solver_kwargs,
                        number_of_qubits=self.number_of_qubits,
                    )
                )
                if verbose:
                    print("GOT THE LOWEST ENERGY!")
                self._lowest_energy = lowest_energy_value
                self._lowest_energy_state = lowest_energy_state

                if both_directions:
                    if verbose:
                        print("FINDING HIGHEST ENERGY")
                    (highest_energy_state, highest_energy_value), opt_res_high = (
                        solve_ising_hamiltonian_mqlib(
                            hamiltonian=_adj_matrix,
                            solver_kwargs=solver_kwargs,
                            number_of_qubits=self.number_of_qubits,
                            maximization=True,
                        )
                    )
                    if verbose:
                        print("GOT HIGHEST ENERGY")
                    self._highest_energy = highest_energy_value
                    self._highest_energy_state = highest_energy_state
        else:
            raise NotImplementedError(
                "Hamiltonian solving is only implemented for 2-local Hamiltonians"
            )

    @classmethod
    def initialize_from_file(
        cls,
        hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral,
        hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral,
    ):
        raise NotImplementedError

    def write_to_file(self, hamiltonian):
        raise NotImplementedError

    def apply_bitflip(self, bitflip_tuple: Tuple[pyd.conint(ge=0, le=1), ...]):
        """
        Applies a bitflip transformation to the Hamiltonian graph.
        :param bitflip_tuple:
        :return:
        """
        # TODO FBM: add unit tests
        time.perf_counter()
        transformed_graph = apply_bitflip_to_hamiltonian(
            hamiltonian=self._hamiltonian, bitflip_tuple=bitflip_tuple
        )
        time.perf_counter()

        self._update_hamiltonian_representation(
            new_representation=transformed_graph,
            bitflip_for_cost_function_initialization=bitflip_tuple,
        )

        time.perf_counter()

        transformation = HamiltonianTransformation(
            transformation=SNV.Bitflip, value=bitflip_tuple
        )

        self._applied_transformations.append(transformation)

        return self

    def apply_permutation(self, permutation_tuple: Tuple[pyd.conint(ge=0), ...]):
        """
        Applies a permutation transformation to the Hamiltonian graph.
        :param permutation_tuple:
        :return:
        """
        # TODO FBM: add unit tests
        if permutation_tuple is None:
            return self

        transformed_graph = apply_permutation_to_hamiltonian(
            hamiltonian=self._hamiltonian, permutation_tuple=permutation_tuple
        )
        self._update_hamiltonian_representation(new_representation=transformed_graph)

        transformation = HamiltonianTransformation(
            transformation=SNV.Permutation, value=permutation_tuple
        )
        self._applied_transformations.append(transformation)
        return self

    def apply_transformations(
        self, transformations_tuple: Tuple[HamiltonianTransformation]
    ):
        """
        :param transformations_tuple:
        :return:
        """
        # TODO FBM: add unit tests

        if isinstance(transformations_tuple[0], HamiltonianTransformation):
            transformations_tuple = [transformations_tuple]
        # TODO(FBM): this shouldn't reinitialize backend_computation for each transformation, only for the concatenated one.
        for transformation, value in transformations_tuple:
            if transformation == SNV.Bitflip:
                self.apply_bitflip(bitflip_tuple=value)
            elif transformation == SNV.Permutation:
                self.apply_permutation(permutation_tuple=value)
            else:
                raise ValueError("Transformation type not recognized")
        return self

    def evaluate_energy(
        self,
        bitstrings_array: Union[
            np.ndarray, cp.ndarray, List[List[int]], List[np.ndarray]
        ],
        pm_input: bool = False,
        backend_computation: Optional[str] = None,
        backend_output: Optional[str] = None,
    ):
        if backend_computation is None:
            backend_computation = self._default_backend

        if backend_computation == "cupy":
            import cupy

            if not isinstance(bitstrings_array, cupy.ndarray):
                bitstrings_array = cupy.asarray(bitstrings_array)

        elif backend_computation == "numpy":
            import numpy

            if not isinstance(bitstrings_array, numpy.ndarray):
                bitstrings_array = numpy.asarray(bitstrings_array)

        if backend_output is None:
            backend_output = self._default_backend

        return self._evaluate_energy_function(
            bitstrings_array=bitstrings_array,
            pm_input=pm_input,
            backend_computation=backend_computation,
            backend_output=backend_output,
        )

    def get_concatenated_transformations(self):
        """
        :return:
        """
        return concatenate_hamiltonian_transformations(self._applied_transformations)

    def get_reversed_list_representations(
        self,
        hamiltonian: Optional[List[Tuple[Union[float, int], Tuple[int, ...]]]] = None,
    ):
        """
        :param hamiltonian:
        :return:
        """
        if hamiltonian is None:
            hamiltonian = self._hamiltonian

        list_reversed = []
        for weight, qubits in hamiltonian:
            qubits_rev = tuple([self._number_of_qubits - 1 - qi for qi in qubits])
            list_reversed.append((weight, qubits_rev))

        return list_reversed

    def get_fields_and_couplings(self, precision: Optional[type] = np.float32):
        return get_fields_and_couplings_from_hamiltonian(self, precision=precision)


def get_fields_and_couplings_from_hamiltonian_list(
    hamiltonian: List[Tuple[Union[float, int], Tuple[int, ...]]],
    number_of_qubits=None,
    precision: Optional[type] = np.float32,
):
    if number_of_qubits is None:
        number_of_qubits = max([max(interaction[1]) for interaction in hamiltonian]) + 1

    couplings = np.zeros((number_of_qubits, number_of_qubits), dtype=precision)
    fields = np.zeros(number_of_qubits, dtype=precision)

    for weight, qubits in hamiltonian:
        if len(qubits) == 1:
            i = qubits[0]
            fields[i] = weight
        elif len(qubits) == 2:
            i, j = qubits
            couplings[i, j] = weight
            couplings[j, i] = weight
        else:
            raise ValueError("Only 1-local and 2-local Hamiltonians are supported")
    return fields, couplings


def get_fields_and_couplings_from_hamiltonian(
    hamiltonian: ClassicalHamiltonianBase, precision: Optional[type] = np.float32
):

    assert set(hamiltonian.localities) in [
        {1},
        {2},
        {1, 2},
    ], "Only 1-local and 2-local have fields and correlations"

    fields = None
    if hamiltonian._local_fields is not None:
        fields = hamiltonian._local_fields

    correlations = None
    if hamiltonian._couplings is not None:
        correlations = hamiltonian._couplings

    update_correlations = False
    update_fields = False
    if correlations is None:
        update_correlations = True
    if fields is None:
        update_fields = True

    if update_correlations:
        correlations = np.zeros(
            (hamiltonian._number_of_qubits, hamiltonian._number_of_qubits),
            dtype=precision,
        )
    if update_fields:
        fields = np.zeros(hamiltonian._number_of_qubits, dtype=precision)

    if update_correlations or update_fields:
        for weight, qubits in hamiltonian._hamiltonian:
            if len(qubits) == 1:
                if update_fields:
                    fields[qubits[0]] += weight
            elif len(qubits) == 2:
                if update_correlations:
                    correlations[qubits[0], qubits[1]] += weight
                    correlations[qubits[1], qubits[0]] += weight
            else:
                raise ValueError("Only 1-local and 2-local Hamiltonians are supported")

    if hamiltonian._default_backend == "cupy":
        fields = cp.asnumpy(fields)
        correlations = cp.asnumpy(correlations)

    fields = np.array(fields, dtype=precision)
    correlations = np.array(correlations, dtype=precision)
    return fields, correlations
