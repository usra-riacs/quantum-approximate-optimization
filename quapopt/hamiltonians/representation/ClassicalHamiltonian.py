# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import copy
import os
import time
from typing import Tuple, Union, Optional, List, Callable, Any, Dict

import numpy as np
import pydantic as pyd

from quapopt import ancillary_functions as anf
from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em
from quapopt.data_analysis.data_handling import (STANDARD_NAMES_VARIABLES as SNV,
                                                 HamiltonianInstanceSpecifierGeneral,
                                                 HamiltonianClassSpecifierGeneral)
from quapopt.data_analysis.data_handling.io_utilities.standardized_io import IOMixin, IOHamiltonianMixin
from quapopt.data_analysis.data_handling.schemas.naming import MAIN_KEY_SEPARATOR
from quapopt.hamiltonians.representation import HamiltonianListRepresentation, \
    convert_list_representation_to_adjacency_matrix
from quapopt.hamiltonians.representation.transformations import (apply_bitflip_to_hamiltonian,
                                                                 apply_bitflip_to_list,
                                                                 apply_permutation_to_hamiltonian,
                                                                 apply_permutation_to_list,
invert_permutation,
                                                                 HamiltonianTransformation,
                                                                 concatenate_hamiltonian_transformations)
import scipy
# Lazy monkey-patching of cupy
from quapopt import AVAILABLE_SIMULATORS
if 'cupy' in AVAILABLE_SIMULATORS:
    import cupy as cp
else:
    import numpy as cp


class IOClassicalHamiltonianMixin(IOHamiltonianMixin):

    @classmethod
    def _get_file_path_main(cls,
                            hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral,
                            hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral):

        hamiltonian_class_directory = IOMixin.get_hamiltonian_class_base_path(
            hamiltonian_class_specifier=hamiltonian_class_specifier)
        hamiltonian_instance_filename = IOMixin.get_hamiltonian_instance_filename(
            hamiltonian_instance_specifier=hamiltonian_instance_specifier)
        file_path_main = f"{hamiltonian_class_directory}/{hamiltonian_instance_filename}"

        return file_path_main

    @classmethod
    def write_hamiltonian_to_file(cls,
                                  hamiltonian: Any,  # should be ClassicalHamiltonian
                                  known_energies_dict: Optional[dict] = None,
                                  class_specific_data: Optional[dict] = None,
                                  overwrite_if_exists=False,
                                  ignore_if_exists=True

                                  ):

        file_path_main = cls._get_file_path_main(hamiltonian_class_specifier=hamiltonian.hamiltonian_class_specifier,
                                                 hamiltonian_instance_specifier=hamiltonian.hamiltonian_instance_specifier)
        cls._write_hamiltonian_to_text_file(hamiltonian=hamiltonian.hamiltonian_list_representation,
                                            file_path=file_path_main,
                                            overwrite_if_exists=overwrite_if_exists,
                                            ignore_if_exists=ignore_if_exists
                                            )
        # if known_energies_dict is not None:
        #     cls._write_hamiltonian_solutions(file_path_main=file_path_main,
        #                                      known_energies_dict=known_energies_dict)
        if class_specific_data is not None:
            file_path_class_specific_information = f"{file_path_main}{MAIN_KEY_SEPARATOR}ClassSpecificData"
            IOMixin.write_results(data=class_specific_data,
                                  full_path=file_path_class_specific_information,
                                  format_type='pickle')

    @classmethod
    def load_hamiltonian_from_file(cls,
                                   hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral,
                                   hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral,
                                   default_backend: Optional[str] = None
                                   ):
        file_path_main = cls._get_file_path_main(hamiltonian_class_specifier=hamiltonian_class_specifier,
                                                 hamiltonian_instance_specifier=hamiltonian_instance_specifier)

        hamiltonian_list_representation = cls._load_hamiltonian_from_text_file(file_path=file_path_main)

        file_path_known_solutions = f"{file_path_main}{MAIN_KEY_SEPARATOR}KnownSolutions"

        known_solutions_df = cls.read_results(full_path=file_path_known_solutions,
                                              return_none_if_not_found=True,
                                              format_type='dataframe')
        known_energies_dict = {}
        if known_solutions_df is not None:


            known_solutions_df = known_solutions_df.sort_values(by=SNV.Energy.id_long)
            lowest_energy_state = known_solutions_df[SNV.Bitstring.id_long].values[0]
            lowest_energy = known_solutions_df[SNV.Energy.id_long].values[0]
            if len(known_solutions_df) > 1:
                highest_energy_state = known_solutions_df[SNV.Bitstring.id_long].values[-1]
                highest_energy = known_solutions_df[SNV.Energy.id_long].values[-1]
            else:
                # A single stored solution (e.g. a planted ground state) says nothing about
                # the top of the spectrum; fabricating highest==lowest here would make the
                # spectral spread 0 and poison renormalized approximation ratios downstream.
                highest_energy_state = None
                highest_energy = None

            if highest_energy_state is not None and not isinstance(highest_energy_state, tuple):
                highest_energy_state = tuple(highest_energy_state)
            if lowest_energy_state is not None and not isinstance(lowest_energy_state, tuple):
                lowest_energy_state = tuple(lowest_energy_state)

            known_energies_dict = {'lowest_energy_state': lowest_energy_state,
                                   'lowest_energy': lowest_energy,
                                   'highest_energy_state': highest_energy_state,
                                   'highest_energy': highest_energy}

        spectrum = None
        file_path_spectral_information = f"{file_path_main}{MAIN_KEY_SEPARATOR}Spectrum"
        if os.path.exists(f"{file_path_spectral_information}.txt"):
            with open(f"{file_path_spectral_information}.txt", 'r') as file:
                spectrum = []
                for line in file:
                    spectrum.append(float(line))

        known_energies_dict['spectrum'] = spectrum

        file_path_class_specific_data = f"{file_path_main}{MAIN_KEY_SEPARATOR}ClassSpecificData"
        class_specific_data = cls.read_results(full_path=file_path_class_specific_data,
                                               return_none_if_not_found=True,
                                               format_type='pickle')

        number_of_qubits = hamiltonian_instance_specifier.NumberOfQubits
        if number_of_qubits is None:
            # Such instances are written with a contiguous 0..n-1 index set, which makes the
            # terms determine n exactly. The gap check is deliberately confined to this
            # branch: instances that DO record a count as part of their identity may
            # legitimately leave qubits untouched (edge_density < 1).
            stored_indices = {index
                              for _, subset in hamiltonian_list_representation
                              for index in subset}
            if not stored_indices:
                raise ValueError(
                    f"Cannot infer the number of qubits for '{file_path_main}': the instance "
                    f"specifier records none and the stored term list is empty.")
            number_of_qubits = max(stored_indices) + 1
            missing_indices = sorted(set(range(number_of_qubits)) - stored_indices)
            if missing_indices:
                shown = missing_indices[:10]
                raise ValueError(
                    f"Cannot infer the number of qubits for '{file_path_main}': the instance "
                    f"specifier records none, and the stored terms skip qubit indices {shown}"
                    f"{' ...' if len(missing_indices) > len(shown) else ''} below their maximum "
                    f"{max(stored_indices)}, so the count is not recoverable from them.")
            # Stamp it back, so a reloaded specifier is indistinguishable from the one the
            # generator handed out.
            hamiltonian_instance_specifier.NumberOfQubits = number_of_qubits

        cost = ClassicalHamiltonian(hamiltonian_list_representation=hamiltonian_list_representation,
                                    known_energies_dict=known_energies_dict,
                                    hamiltonian_class_specifier=hamiltonian_class_specifier,
                                    hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                    number_of_qubits=number_of_qubits,
                                    class_specific_data=class_specific_data,
                                    default_backend=default_backend)
        cost.was_read_from_drive = True

        return cost


_REPRESENTATION_DESCRIPTION_LIST = """Classical Hamiltonian represented as a list of interactions.
                                     Each interaction is a tuple of the form (c, (i,j,k,...)) 
                                     where i, j, k, ... are qubit indices
                                     The length of the tuple determines the locality of the interaction
                                     c is the coefficient of the interaction term in the Hamiltonian
                                     """


class ClassicalHamiltonianBase(IOClassicalHamiltonianMixin):
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
    _has_planted_solution:Optional[bool]

    def __init__(self,
                 hamiltonian: HamiltonianListRepresentation,
                 number_of_qubits: int,
                 representation_description: Optional[str] = None,
                 solve_at_initialization=False,
                 hamiltonian_class_specifier: Optional[HamiltonianClassSpecifierGeneral | str] = None,
                 hamiltonian_instance_specifier: Optional[HamiltonianInstanceSpecifierGeneral | str] = None,
                 known_energies_dict=None,
                 class_specific_data: Optional[dict] = None,
                 default_backend: str = None):
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
            except(AttributeError, IndexError):
                pass

        assert isinstance(hamiltonian, list), "Hamiltonian must be a list of interactions"

        if isinstance(hamiltonian, list):
            hamiltonian = sorted(hamiltonian, key=lambda x: x[1])
            hamiltonian = sorted(hamiltonian, key=lambda x: len(x[1]))

        self._hamiltonian = hamiltonian
        if representation_description is None:
            if isinstance(self._hamiltonian, list):
                self._representation_description = _REPRESENTATION_DESCRIPTION_LIST
            else:
                raise ValueError("Hamiltonian must be a networkit graph or a list of interactions"
                                 "or description must be provided.")

        self._representation_description = representation_description

        _hamiltonian_class_description = None
        _hamiltonian_class_specifier = None
        # in this case, _hamiltonian_instance_specifier is really a _hamiltonian_class_description
        if isinstance(hamiltonian_class_specifier, str):
            _hamiltonian_class_description = hamiltonian_class_specifier
        elif hamiltonian_class_specifier is not None:
            _hamiltonian_class_description = hamiltonian_class_specifier.get_description_string()
            _hamiltonian_class_specifier = hamiltonian_class_specifier

        _hamiltonian_instance_description = None
        _hamiltonian_instance_specifier = None
        # in this case, _hamiltonian_instance_specifier is really a _hamiltonian_instance_description
        if isinstance(hamiltonian_instance_specifier, str):
            _hamiltonian_instance_description = hamiltonian_instance_specifier
        elif hamiltonian_instance_specifier is not None:
            _hamiltonian_instance_description = hamiltonian_instance_specifier.get_description_string()
            _hamiltonian_instance_specifier = hamiltonian_instance_specifier

        self._hamiltonian_class_description: Optional[str] = _hamiltonian_class_description
        self._hamiltonian_instance_description: Optional[str] = _hamiltonian_instance_description

        self._hamiltonian_class_specifier: Optional[HamiltonianClassSpecifierGeneral] = _hamiltonian_class_specifier
        self._hamiltonian_instance_specifier: Optional[
            HamiltonianInstanceSpecifierGeneral] = _hamiltonian_instance_specifier

        self._localities = sorted(list(set(([len(interaction[1]) for interaction in self._hamiltonian]))))

        if set(self.localities) in [{2}, {1, 2}, {1}]:
            self._is_two_local = True
        else:
            self._is_two_local = False

        self._has_local_fields = 1 in self.localities

        self._number_of_qubits = int(number_of_qubits)
        self._applied_transformations = []

        _spectrum = None
        _lowest_energy = None
        _lowest_energy_state = None
        _lowest_energy_state_solver = None
        _lowest_energy_state_runtime = None

        _highest_energy = None
        _highest_energy_state = None
        _highest_energy_state_solver = None
        _highest_energy_state_runtime = None
        _has_planted_solution = False

        if known_energies_dict is not None:
            _spectrum = known_energies_dict.get('spectrum', None)
            _lowest_energy = known_energies_dict.get('lowest_energy', None)
            _lowest_energy_state = known_energies_dict.get('lowest_energy_state', None)
            _lowest_energy_state_solver = known_energies_dict.get('lowest_energy_state_solver', "Unknown")
            _lowest_energy_state_runtime = known_energies_dict.get('lowest_energy_state_runtime', None)
            _highest_energy = known_energies_dict.get('highest_energy', None)
            _highest_energy_state = known_energies_dict.get('highest_energy_state', None)
            _highest_energy_state_solver = known_energies_dict.get('highest_energy_state_solver', "Unknown")
            _highest_energy_state_runtime = known_energies_dict.get('highest_energy_state_runtime', None)

            _has_planted_solution = known_energies_dict.get('has_planted_solution', False)


        self._spectrum = _spectrum
        self._lowest_energy = _lowest_energy
        self._lowest_energy_state = _lowest_energy_state
        self._lowest_energy_state_solver = _lowest_energy_state_solver
        self._lowest_energy_state_runtime = _lowest_energy_state_runtime

        self._highest_energy = _highest_energy
        self._highest_energy_state = _highest_energy_state
        self._highest_energy_state_solver = _highest_energy_state_solver
        self._highest_energy_state_runtime = _highest_energy_state_runtime

        self._spectral_spread = None

        self._class_specific_data = class_specific_data
        self._known_energies_dict = known_energies_dict

        self._has_planted_solution = _has_planted_solution

        if default_backend is None:
            from quapopt import AVAILABLE_SIMULATORS
            default_backend = 'numpy'
            if 'cupy' in AVAILABLE_SIMULATORS and self._number_of_qubits > 50:
                default_backend = 'cupy'

        self._default_backend = default_backend

        # print("WHAT",self._default_backend, default_backend)
        self._evaluate_energy_function = None
        self._couplings = None
        self._local_fields = None

        self._update_hamiltonian_representation(new_representation=self._hamiltonian)

        if solve_at_initialization:
            self.solve_hamiltonian()

        _two_local_properties = None
        if self._is_two_local:
            _two_local_properties = {}
            _two_local_properties['number_of_edges'] = len([s for s in self._hamiltonian if len(s[1]) == 2])

            # average degree is the number of connections per qubit
            _two_local_properties['average_degree'] = 2 * _two_local_properties[
                'number_of_edges'] / self.number_of_qubits

            if self.number_of_qubits == 1:
                _density = 1.0
            else:
                # density is the number of edges divided by the number of possible edges
                _density = _two_local_properties['number_of_edges'] / (
                        self.number_of_qubits * (self.number_of_qubits - 1) / 2)

            _two_local_properties['density'] = _density
        self._two_local_properties = _two_local_properties

        self._read_from_drive = None

    def __repr__(self):

        _known_lowest_energy = self.lowest_energy if self.lowest_energy is not None else "unknown"
        _known_highest_energy = self.highest_energy if self.highest_energy is not None else "unknown"
        _known_spectral_data = {'Lowest Energy': _known_lowest_energy, 'Highest Energy': _known_highest_energy}

        _main_part = f"Classical Hamiltonian with {self.number_of_qubits} qubits\n" \
                     f"Localities: {self.localities}\n" \
                     f"Class: {self.hamiltonian_class_description}\n" \
                     f"Instance: {self.hamiltonian_instance_description}\n" \
                     f"Known spectral data: {_known_spectral_data}\n" \
                     f"Number of applied transformations: {len(self.applied_transformations)}\n"

        _two_local_part = ""
        if self.is_two_local:
            _two_local_part = f"Number of Edges: {self.number_of_edges}\n" \
                              f"Average Degree: {self.average_degree}\n" \
                              f"Density: {self.density}\n"

        _main_part += _two_local_part

        _zero_energy = self.compute_zero_energy()
        _main_part += f"Energy of |0...0>: {_zero_energy}\n"
        _zero_ar = self.calculate_approximation_ratio(energy=_zero_energy)
        if _zero_ar is not None:
            _main_part += f"Approximation Ratio of |0...0>: {_zero_ar}\n"

        _main_part += f"Default Backend: {self.default_backend}\n"

        return _main_part

    def get_known_energies_dict(self):
        known_energies_dict = {'spectrum': self._spectrum,
                               'lowest_energy': self._lowest_energy,
                               'lowest_energy_state': self._lowest_energy_state,
                               'lowest_energy_state_solver': self._lowest_energy_state_solver,
                               'lowest_energy_state_runtime': self._lowest_energy_state_runtime,

                               'highest_energy': self._highest_energy,
                               'highest_energy_state': self._highest_energy_state,
                               'highest_energy_state_solver': self._highest_energy_state_solver,
                               'highest_energy_state_runtime': self._highest_energy_state_runtime,

                               'has_planted_solution': self._has_planted_solution

                               }
        return known_energies_dict


    @property
    def hamiltonian(self) -> List[Tuple[Union[float, int], Tuple[int, ...]]]:
        return self._hamiltonian

    @property
    def hamiltonian_class_specifier(self) -> Optional[HamiltonianClassSpecifierGeneral]:
        return self._hamiltonian_class_specifier

    @property
    def hamiltonian_class_description(self) -> Optional[str]:

        if self.hamiltonian_class_specifier is not None:
            return self.hamiltonian_class_specifier.get_description_string()

        return self._hamiltonian_class_description

    @property
    def hamiltonian_instance_specifier(self) -> Optional[HamiltonianInstanceSpecifierGeneral]:
        return self._hamiltonian_instance_specifier

    @property
    def hamiltonian_instance_description(self) -> Optional[str]:
        if self.hamiltonian_instance_specifier is not None:
            return self.hamiltonian_instance_specifier.get_description_string()

        return self._hamiltonian_instance_description

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
    def lowest_energy_state(self):
        return self._lowest_energy_state

    @property
    def ground_state_energy(self):
        return self._lowest_energy

    @property
    def ground_state(self):
        return self._lowest_energy_state

    @property
    def highest_energy(self):
        return self._highest_energy

    @property
    def highest_energy_state(self):
        return self._highest_energy_state
    @property
    def has_planted_solution(self):
        return self._has_planted_solution

    @property
    def spectral_spread(self):

        if self._spectral_spread is None:
            if self.lowest_energy is None or self.highest_energy is None:
                return None
            self._spectral_spread = self.highest_energy - self.lowest_energy
        return self._spectral_spread

    @property
    def default_backend(self):
        return self._default_backend

    # @default_backend.setter
    # def default_backend(self,
    #                     backend_computation:str):
    #     assert backend_computation in ['cupy', 'numpy'], "Please add either 'cupy' or 'numpy'"
    #     self._default_backend = backend_computation

    @property
    def is_two_local(self):
        return self._is_two_local

    @property
    def has_local_fields(self):
        return self._has_local_fields

    @property
    def was_read_from_drive(self):
        return self._read_from_drive

    @was_read_from_drive.setter
    def was_read_from_drive(self, read_from_drive: bool):
        self._read_from_drive = read_from_drive

    @property
    def hamiltonian_metadata(self):
        return {f"{SNV.HamiltonianClassDescription.id_long}": self.hamiltonian_class_description,
                f"{SNV.HamiltonianInstanceDescription.id_long}": self.hamiltonian_instance_description

                }

    def calculate_approximation_ratio(self,
                                      energy: Optional[float | np.ndarray],
                                      spectrum_renormalized: bool = True,
                                      minimization: bool = True
                                      ) -> Optional[float | np.ndarray]:
        """
        Helper function to calculate the approximation ratio for a given energy.
        :param energy:
        :param spectrum_renormalized:
        If True, it renormalizes the AR calculation to the spectral spread of the Hamiltonian.
        This means it's always between 0 and 1, even for Hamiltonians with both positive and negative eigenvalues.
        :param minimization:
        If True, it assumes that the lowest energy is the optimal one (AR = 1.0).
        :return:
        """

        if energy is None:
            return None

        if (isinstance(energy, np.ndarray) and len(energy) == 1) or isinstance(energy, (float, int)):
            _reformat_fun = float

        else:
            _reformat_fun = lambda x: x

        if spectrum_renormalized:
            if self.lowest_energy is None or self.highest_energy is None:
                return None

            delta = self.spectral_spread
            if minimization:
                return _reformat_fun((self.highest_energy - energy) / delta)
            else:
                return _reformat_fun((energy - self.lowest_energy) / delta)
        else:
            if minimization:
                if self.lowest_energy is None:
                    return None
                return _reformat_fun(energy / self.lowest_energy)
            else:
                if self.highest_energy is None:
                    return None
                return _reformat_fun(energy / self.highest_energy)

    def update_hamiltonian_representation(self,
                                          new_representation: HamiltonianListRepresentation):
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
            def _wrapped_eval(bitstrings_array,
                              pm_input: bool = False,
                              backend_computation=self._default_backend,
                              backend_output=self._default_backend):
                return em.calculate_energies_from_bitstrings_2_local(couplings_array=self._couplings,
                                                                     local_fields=self._local_fields,
                                                                     bitstrings_array=bitstrings_array,
                                                                     local_fields_present=True,
                                                                     pm_input=pm_input,
                                                                     computation_backend=backend_computation,
                                                                     output_backend=backend_output)

            self._evaluate_energy_function = _wrapped_eval

        elif {2} == localities_set:
            def _wrapped_eval(bitstrings_array,
                              pm_input: bool = False,
                              backend_computation=self._default_backend,
                              backend_output=self._default_backend):
                return em.calculate_energies_from_bitstrings_2_local(couplings_array=self._couplings,
                                                                     local_fields=None,
                                                                     bitstrings_array=bitstrings_array,
                                                                     local_fields_present=False,
                                                                     pm_input=pm_input,
                                                                     computation_backend=backend_computation,
                                                                     output_backend=backend_output)

            self._evaluate_energy_function = _wrapped_eval
        else:
            # TODO(FBM): Refactor this for higher-locality hamiltonians
            def _wrapped_eval(bitstrings_array,
                              pm_input=False,
                              backend_computation=self._default_backend,
                              backend_output=self._default_backend):
                if pm_input:
                    raise NotImplementedError("pm_input is not supported for K-local Hamiltonians")
                return em.calculate_energies_from_bitstrings(hamiltonian=self._hamiltonian,
                                                             bitstrings_array=bitstrings_array,
                                                             backend_output=backend_output,
                                                             backend_computation=backend_computation)

            self._evaluate_energy_function = _wrapped_eval

    def reinitialize_backend(self,
                             backend: str):

        if self._default_backend == backend:
            return

        if self._couplings is not None:
            couplings, local_fields = self.get_couplings_and_local_fields(matrix_type='SYM',
                                                                          backend=backend,
                                                                          precision=np.float64)
            self._couplings = couplings
            self._local_fields = local_fields
        self._default_backend = backend
        self._reinitialize_evaluate_function_energy()

    def _update_hamiltonian_representation(self,
                                           new_representation: HamiltonianListRepresentation,
                                           bitflip_for_cost_function_initialization: Tuple[int, ...] = None):
        """
        WARNING: this assumes we DO NOT CHANGE the localities of the Hamiltonian!
        This class is useful when applying gauge transformations to the same Hamiltonian.
        :param new_representation:
        :param bitflip_for_cost_function_initialization:
        :return:
        """

        if bitflip_for_cost_function_initialization is None:
            self._hamiltonian = new_representation

            if self.is_two_local:
                # The matrices must come from the NEW representation. They are cached, and
                # get_couplings_and_local_fields returns the cache untouched when it is
                # there, so clear it first - otherwise the evaluator keeps describing the
                # representation this call is replacing. The cache is float64: energies
                # evaluated from it, and the python solver's spectrum, are float64 sums.
                self._couplings = None
                self._local_fields = None
                couplings, local_fields = self.get_couplings_and_local_fields(matrix_type='SYM',
                                                                              backend=self._default_backend,
                                                                              precision=np.float64)
                self._couplings = couplings
                self._local_fields = local_fields

            self._reinitialize_evaluate_function_energy()


        elif not self.is_two_local:
            self._hamiltonian = new_representation
            self._reinitialize_evaluate_function_energy()

        else:
            # TODO(FBM): add more efficient version for 2-local Hamiltonians!
            # This is a more efficient version special for two-local Hamiltonians
            if self.default_backend == 'cupy':
                bck = cp
            elif self.default_backend == 'numpy':
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

    def copy(self):
        return copy.deepcopy(self)

    def _bck(self):
        if self._default_backend == 'cupy':
            return cp
        else:
            return np

    #@classmethod
    def solve_hamiltonian(cls,
                          both_directions=True,
                          solver_kwargs=None,
                          verbose=False):
        """
        Find extremal energies/states and append them to the instance's solutions file.

        n<=25: exact brute-force spectrum; always overwrites stored extremal data.
        n>25: heuristic solver (MQLib, default BURER2002). A heuristic result never
        replaces a stored extremum that is at least as good - e.g. an exactly known
        planted ground state survives a solver that fails to beat it - and no
        KnownSolutions row is written for a discarded result. To force an overwrite,
        clear the stored values first.

        Every solver result, accepted or discarded, is additionally recorded in the
        instance's SolutionsArchive file (deduplicated on (state, energy)) - a pool of
        good solutions independent of the extremal certificates.
        """
        from quapopt import AVAILABLE_SIMULATORS

        if cls.is_two_local:
            # TODO(FBM) DO NOT IGNORE SOLVER KWARGS
            if cls.number_of_qubits <= 25:
                t0 = time.perf_counter()
                if 'cuda' in AVAILABLE_SIMULATORS:
                    cls._spectrum = em.cuda_solve_hamiltonian(cls)
                else:
                    cls._spectrum = em.solve_hamiltonian_python(cls)
                t1 = time.perf_counter()
                runtime = t1 - t0
                argmin_spectrum = np.argmin(cls._spectrum)

                cls._lowest_energy = cls._spectrum[argmin_spectrum]
                cls._lowest_energy_state = anf.convert_int_to_binary_tuple(integer=argmin_spectrum,
                                                                            number_of_bits=cls.number_of_qubits)
                cls._lowest_energy_state_solver = 'bruteforce'
                cls._lowest_energy_state_runtime = runtime
                cls.write_solutions_to_file(which='lowest')
                cls.append_solution_to_archive(bitstring=cls._lowest_energy_state,
                                               energy=cls._lowest_energy,
                                               solver_name='bruteforce',
                                               solver_runtime=runtime)

                argmax_spectrum = np.argmax(cls._spectrum)
                cls._highest_energy = cls._spectrum[argmax_spectrum]
                cls._highest_energy_state = anf.convert_int_to_binary_tuple(integer=argmax_spectrum,
                                                                             number_of_bits=cls.number_of_qubits)
                cls._highest_energy_state_solver = 'bruteforce'
                cls._highest_energy_state_runtime = runtime
                cls.write_solutions_to_file(which='highest')
                cls.append_solution_to_archive(bitstring=cls._highest_energy_state,
                                               energy=cls._highest_energy,
                                               solver_name='bruteforce',
                                               solver_runtime=runtime)


            else:
                if solver_kwargs is None:
                    solver_kwargs = {'solver_name': "BURER2002",
                                     'solver_timeout': 1}
                cls._spectrum = None

                from quapopt.optimization.classical_solvers.mqlib_solvers import solve_ising_hamiltonian_mqlib

                _adj_matrix = cls.get_adjacency_matrix(matrix_type='SYM',
                                                        backend='numpy')
                if verbose:
                    print("SOLVING HAMILTONIAN with:", solver_kwargs)

                def _energy_on_the_cache(bitstring):
                    # MQLib evaluates its result on the float32 adjacency matrix; the stored
                    # extremum is compared with float64 energies, so it is re-evaluated here.
                    energy = cls.evaluate_energy(bitstrings_array=np.array([bitstring], dtype=np.int64),
                                                 backend_output='numpy')
                    return np.float64(np.asarray(energy).ravel()[0])

                (lowest_energy_state, _), opt_res_low = solve_ising_hamiltonian_mqlib(
                    hamiltonian=_adj_matrix,
                    solver_kwargs=solver_kwargs,
                    number_of_qubits=cls.number_of_qubits)
                lowest_energy_value = _energy_on_the_cache(lowest_energy_state)
                if verbose:
                    print("GOT THE LOWEST ENERGY!")
                cls.append_solution_to_archive(bitstring=lowest_energy_state,
                                               energy=lowest_energy_value,
                                               solver_name=solver_kwargs.get('solver_name', 'Unknown'),
                                               solver_runtime=opt_res_low['runtime'].values[0])
                if cls._lowest_energy is not None and cls._lowest_energy <= lowest_energy_value:
                    # A heuristic must never displace a stored optimum it failed to beat
                    # (e.g. an exactly known planted ground state); keeping ties preserves
                    # the stored provenance and avoids duplicate solution rows on disk.
                    if verbose:
                        print(f"KEEPING stored lowest energy {cls._lowest_energy} "
                              f"(solver: {cls._lowest_energy_state_solver}); discarding "
                              f"{solver_kwargs.get('solver_name', 'Unknown')} result {lowest_energy_value}")
                else:
                    cls._lowest_energy = lowest_energy_value
                    cls._lowest_energy_state = lowest_energy_state

                    cls._lowest_energy_state_solver = solver_kwargs.get('solver_name', 'Unknown')
                    cls._lowest_energy_state_runtime = opt_res_low['runtime'].values[0]

                    cls.write_solutions_to_file(which='lowest')

                if both_directions:
                    if verbose:
                        print("FINDING HIGHEST ENERGY")
                    (highest_energy_state, _), opt_res_high = solve_ising_hamiltonian_mqlib(
                        hamiltonian=_adj_matrix,
                        solver_kwargs=solver_kwargs,
                        number_of_qubits=cls.number_of_qubits,
                        maximization=True)
                    highest_energy_value = _energy_on_the_cache(highest_energy_state)
                    if verbose:
                        print("GOT HIGHEST ENERGY")
                    cls.append_solution_to_archive(bitstring=highest_energy_state,
                                                   energy=highest_energy_value,
                                                   solver_name=solver_kwargs.get('solver_name', 'Unknown'),
                                                   solver_runtime=opt_res_high['runtime'].values[0])
                    if cls._highest_energy is not None and cls._highest_energy >= highest_energy_value:
                        # Same guard, opposite direction.
                        if verbose:
                            print(f"KEEPING stored highest energy {cls._highest_energy} "
                                  f"(solver: {cls._highest_energy_state_solver}); discarding "
                                  f"{solver_kwargs.get('solver_name', 'Unknown')} result {highest_energy_value}")
                    else:
                        cls._highest_energy = highest_energy_value
                        cls._highest_energy_state = highest_energy_state

                        cls._highest_energy_state_solver = solver_kwargs.get('solver_name', 'Unknown')
                        cls._highest_energy_state_runtime = opt_res_high['runtime'].values[0]

                        cls.write_solutions_to_file(which='highest')



        else:
            raise NotImplementedError("Hamiltonian solving is only implemented for 2-local Hamiltonians")

    @classmethod
    def initialize_from_file(cls,
                             hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral,
                             hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral):
        raise NotImplementedError

    # def write_to_file(self,
    #                   hamiltonian):
    #     raise NotImplementedError

    def _carry_extrema_to_new_frame(self,
                                    forward_map: Callable[[Tuple[int, ...]], Tuple[int, ...]]) -> None:
        """
        Move the stored extremal states into the frame a gauge call is about to create.

        The extremal ENERGIES are invariant under a gauge, the extremal STATES are not, so
        an instance that keeps reporting the states it had would be describing the
        representation it just left. An instance holding extrema and no spectrum - the
        shape a file load produces - has no other way of learning them.

        :param forward_map: sends a state of the current frame to its image in the new one
        """
        for attribute_name in ('_lowest_energy_state', '_highest_energy_state'):
            state = getattr(self, attribute_name)
            if state is None:
                continue
            setattr(self, attribute_name, forward_map(tuple(int(bit) for bit in state)))

    def apply_bitflip(self,
                      bitflip_tuple: Optional[Tuple[pyd.conint(ge=0, le=1), ...]]) -> 'ClassicalHamiltonianBase':
        """
        Applies a bitflip transformation to the Hamiltonian graph.
        :param bitflip_tuple:
        :return:
        """
        if bitflip_tuple is None:
            return self

        transformed_graph = apply_bitflip_to_hamiltonian(hamiltonian=self._hamiltonian,
                                                         bitflip_tuple=bitflip_tuple)

        # The record and the extrema are brought up to date BEFORE the representation is
        # replaced: replacing it re-solves a solved instance, and that solve writes
        # solutions through the mapping, which reads this list.
        transformation = HamiltonianTransformation(transformation=SNV.Bitflip,
                                                   value=bitflip_tuple)
        self._applied_transformations.append(transformation)
        self._carry_extrema_to_new_frame(
            forward_map=lambda state: apply_bitflip_to_list(bitflip_tuple=bitflip_tuple,
                                                            list_of_bits=state))

        self._update_hamiltonian_representation(new_representation=transformed_graph,
                                                bitflip_for_cost_function_initialization=bitflip_tuple)

        return self

    def apply_permutation(self,
                          permutation_tuple: Tuple[pyd.conint(ge=0), ...]):
        """
        Applies a permutation transformation to the Hamiltonian graph.
        :param permutation_tuple:
        :return:
        """
        if permutation_tuple is None:
            return self

        transformed_graph = apply_permutation_to_hamiltonian(hamiltonian=self._hamiltonian,
                                                             permutation_tuple=permutation_tuple)

        # Same ordering as apply_bitflip, and for the same reason.
        transformation = HamiltonianTransformation(transformation=SNV.Permutation,
                                                   value=permutation_tuple)
        self._applied_transformations.append(transformation)
        # E_{P_p(H)}(s) = E_H(s o p), so the state carrying a given energy moves the other
        # way: the image of u is u o p^-1.
        permutation_inverse = invert_permutation(permutation_tuple=permutation_tuple)
        self._carry_extrema_to_new_frame(
            forward_map=lambda state: apply_permutation_to_list(permutation=permutation_inverse,
                                                                list_to_permute=state))

        self._update_hamiltonian_representation(new_representation=transformed_graph)

        return self

    def apply_transformations(self,
                              transformations_tuple: Union[HamiltonianTransformation,
                                                           List[HamiltonianTransformation]]):
        """
        Apply gauge transformations in the order given.

        Accepts a single HamiltonianTransformation or any iterable of them - the shape
        `applied_transformations` returns - so a recorded sequence replays on a fresh
        instance.

        :param transformations_tuple:
        :return:
        """
        if isinstance(transformations_tuple, HamiltonianTransformation):
            transformations_tuple = [transformations_tuple]
        # TODO(FBM): this shouldn't reinitialize backend_computation for each transformation, only for the concatenated one.
        for transformation_full in transformations_tuple:
            transformation, value = transformation_full.transformation, transformation_full.value
            if transformation == SNV.Bitflip:
                self.apply_bitflip(bitflip_tuple=value)
            elif transformation == SNV.Permutation:
                self.apply_permutation(permutation_tuple=value)
            else:
                raise ValueError("Transformation type not recognized")
        return self

    def evaluate_energy(self,
                        bitstrings_array: Union[np.ndarray, cp.ndarray, List[List[int]], List[np.ndarray]],
                        pm_input: bool = False,
                        backend_computation: Optional[str] = None,
                        backend_output: Optional[str] = None,
                        ):
        if backend_computation is None:
            backend_computation = self._default_backend

        bitstrings_array = anf.convert_cupy_numpy_array(array=bitstrings_array,
                                                        output_backend=backend_computation)

        if backend_output is None:
            backend_output = self._default_backend

        return self._evaluate_energy_function(bitstrings_array=bitstrings_array,
                                              pm_input=pm_input,
                                              backend_computation=backend_computation,
                                              backend_output=backend_output)

    def compute_zero_energy(self):

        if self.default_backend == 'cupy':
            import cupy as bck
        else:
            import numpy as bck

        zeros = bck.zeros((1, self.number_of_qubits))

        return float(self.evaluate_energy(bitstrings_array=zeros)[0])

    def compute_zero_approximation_ratio(self):
        return self.calculate_approximation_ratio(energy=self.compute_zero_energy())

    def get_concatenated_transformations(self):
        """
        Fold the applied transformations into one bitflip and one permutation.

        The pair (b, p) means H_current = P_p(B_b(H_original)) - bitflip first, permutation
        second, whatever order the calls came in. (None, None) when nothing is applied.

        :return: (bitflip, permutation)
        """

        if len(self._applied_transformations) == 0:
            return None, None

        return concatenate_hamiltonian_transformations(self._applied_transformations)

    def map_bitstring_to_original_representation(self,
                                                 bitstring: Union[Tuple[int, ...], List[int], np.ndarray]
                                                 ) -> Tuple[int, ...]:
        """
        Map a state of the CURRENT representation to the state of the original one that
        carries the same energy.

        Gauge transformations act in place and leave the class and instance specifiers
        untouched, so a transformed instance keeps the storage identity of the original.
        A state found for the transformed representation therefore belongs to a frame the
        stored file does not describe, and it must be mapped before it is written.

        Each transformation is undone at the state level, newest first: a bitflip B_v
        gives E_{B_v(H)}(s) = E_H(s XOR v), and a permutation P_v gives
        E_{P_v(H)}(s) = E_H(v applied to s). The concatenated pair reads
        H_current = P_p(B_b(H_original)) (see concatenate_hamiltonian_transformations) and
        would give the same answer; the walk is kept as the reference implementation, the
        pair as the fast path for arrays, and the two are tested against each other.

        :param bitstring: state of the current representation
        :return: the state of the original representation with the same energy
        """
        bitstring = tuple(int(bit) for bit in bitstring)

        for transformation_full in reversed(self._applied_transformations):
            transformation, value = transformation_full.transformation, transformation_full.value
            if transformation == SNV.Bitflip:
                bitstring = apply_bitflip_to_list(bitflip_tuple=value,
                                                  list_of_bits=bitstring)
            elif transformation == SNV.Permutation:
                bitstring = apply_permutation_to_list(permutation=value,
                                                      list_to_permute=bitstring)
            else:
                raise ValueError(f"Unknown transformation: {transformation}")

        return bitstring

    def _prepare_solution_for_storage(self,
                                      bitstring: Union[Tuple[int, ...], List[int], np.ndarray],
                                      energy: float,
                                      relative_energy_tolerance: float = 1e-3) -> Tuple[int, ...]:
        """
        Move a solution of the current representation into the frame its file is keyed by.

        Without transformations the two frames are the same and the state is returned as
        it came. With transformations, the state is checked against the current
        representation first: the energy being written must be the energy the state
        carries here, otherwise the pair comes from another frame or another instance and
        mapping it would store a state that looks certified and is not.

        The energy check catches a pair a caller supplies from another frame or another
        instance. It cannot fire on a solver's own row: a solver writes the state it found
        with the energy that state carries where it found it, which is self-consistent
        whatever the frame. The mapping is what protects that case.

        :param bitstring: state of the current representation
        :param energy: the energy stored beside it, invariant under the transformations
        :param relative_energy_tolerance: tolerance of the check, relative to the larger
            of the two energies - the evaluation backends differ in precision
        :return: the state to write
        """
        bitstring = tuple(int(bit) for bit in bitstring)

        if len(self._applied_transformations) == 0:
            return bitstring

        energy_of_bitstring = float(self.evaluate_energy(bitstrings_array=[list(bitstring)])[0])
        energy = float(energy)
        scale = max(abs(energy_of_bitstring), abs(energy), 1.0)
        if abs(energy_of_bitstring - energy) > relative_energy_tolerance * scale:
            raise ValueError(f"Refusing to store energy {energy} for state {bitstring}: under the "
                             f"current representation that state has energy {energy_of_bitstring}. "
                             f"{len(self._applied_transformations)} transformation(s) are applied, "
                             f"so the pair belongs to neither this frame nor the stored one.")

        return self.map_bitstring_to_original_representation(bitstring=bitstring)

    def recover_original_hamiltonian_representation(self):
        """
        Return a fresh instance holding the ORIGINAL representation.

        The concatenated pair reads H_current = P_p(B_b(H_original)), so the original is
        B_b(P_{p^-1}(H_current)): the permutation is undone first, then the bitflip. Both
        steps run as pure functions on the list representation. Building an instance per
        step instead would re-solve it - constructing or re-representing a Hamiltonian
        that carries a spectrum triggers a solve - and append its rows to the files of the
        instance this one was derived from, which already hold them.

        The extremal states belong to the current frame, so they are mapped back. The
        spectrum is indexed by basis states of the current frame and means nothing here,
        so the returned instance carries none.

        :return: an untransformed instance with the original representation
        """

        if len(self._applied_transformations) == 0:
            return self.copy()

        if not self.is_two_local:
            raise NotImplementedError("This function is only implemented for 2-local Hamiltonians")

        bitflip_combined, permutation_combined = self.get_concatenated_transformations()

        recovered_representation = apply_permutation_to_hamiltonian(
            hamiltonian=self.hamiltonian_list_representation,
            permutation_tuple=invert_permutation(permutation_tuple=permutation_combined))
        recovered_representation = apply_bitflip_to_hamiltonian(hamiltonian=recovered_representation,
                                                                bitflip_tuple=bitflip_combined)

        known_energies_dict = dict(self.get_known_energies_dict())
        known_energies_dict['spectrum'] = None
        for _name in ('lowest', 'highest'):
            state = known_energies_dict.get(f'{_name}_energy_state', None)
            if state is not None:
                known_energies_dict[f'{_name}_energy_state'] = \
                    self.map_bitstring_to_original_representation(bitstring=state)

        return ClassicalHamiltonian(
            hamiltonian_list_representation=recovered_representation,
            number_of_qubits=self.number_of_qubits,
            hamiltonian_class_specifier=self.hamiltonian_class_specifier,
            hamiltonian_instance_specifier=self.hamiltonian_instance_specifier,
            known_energies_dict=known_energies_dict,
            class_specific_data=self.class_specific_information,
            default_backend=self.default_backend)

    def _array_module_of(self,
                         solutions_array):
        """
        The array module of the INPUT array, never of the instance.

        cupy refuses mixed operands, and a numpy sample array on a cupy-backed instance is
        the common case, so both array maps stay on whatever backend the caller handed in.
        """
        if cp is not np and isinstance(solutions_array, cp.ndarray):
            return cp
        return np

    def recover_original_solutions_representation(self,
                                                  solutions_array: np.ndarray | cp.ndarray):
        """
        Map an array of CURRENT-frame states to the original frame, row by row.

        With H_current = P_p(B_b(H_original)), a state s of the current representation
        carries the energy the state (s o p) XOR b carries under the original, where
        (s o p)[i] = s[p[i]] - so column i of the result is column p[i] of the input,
        followed by the bitflip. The result stays on the input array's backend.

        :param solutions_array: rows are states of the current representation
        :return: the same rows as states of the original representation
        """

        if isinstance(solutions_array, list):
            solutions_array = self._bck().array(solutions_array)
        solutions_array = solutions_array.copy()
        if len(self._applied_transformations) == 0:
            return solutions_array

        bitflip_combined, permutation_combined = self.get_concatenated_transformations()
        _bck = self._array_module_of(solutions_array=solutions_array)

        recovered_solutions = solutions_array[:, list(permutation_combined)]
        recovered_solutions = recovered_solutions ^ _bck.asarray(bitflip_combined,
                                                                 dtype=recovered_solutions.dtype)

        return recovered_solutions

    def map_solutions_to_current_representation(self,
                                                solutions_array: np.ndarray | cp.ndarray):
        """
        Map an array of ORIGINAL-frame states to the current frame, row by row.

        The inverse of recover_original_solutions_representation: the bitflip first, then
        the permutation's forward map u -> u o p^-1. The identity while no transformation
        is applied. The result stays on the input array's backend.

        :param solutions_array: rows are states of the original representation
        :return: the same rows as states of the current representation
        """

        if isinstance(solutions_array, list):
            solutions_array = self._bck().array(solutions_array)
        solutions_array = solutions_array.copy()
        if len(self._applied_transformations) == 0:
            return solutions_array

        bitflip_combined, permutation_combined = self.get_concatenated_transformations()
        _bck = self._array_module_of(solutions_array=solutions_array)

        mapped_solutions = solutions_array ^ _bck.asarray(bitflip_combined,
                                                          dtype=solutions_array.dtype)

        return mapped_solutions[:, list(invert_permutation(permutation_tuple=permutation_combined))]

    def get_reversed_list_representations(self,
                                          hamiltonian: Optional[
                                              List[Tuple[Union[float, int], Tuple[int, ...]]]] = None):
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


    @property
    def hamiltonian_list_representation(self) -> List[Tuple[Union[float, int], Tuple[int, ...]]]:
        return self._hamiltonian

    def reverse_list_representation_indices(self):
        # reversed_representation =
        self._hamiltonian = self.get_reversed_list_representations()


    @classmethod
    def initialize_from_file(cls,
                             hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral,
                             hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral,
                             default_backend: Optional[str] = None):
        """
        Initialize a ClassicalHamiltonian instance from a file.
        :param hamiltonian_class_specifier: Specifier for the Hamiltonian class.
        :param hamiltonian_instance_specifier: Specifier for the Hamiltonian instance.
        :return: An instance of ClassicalHamiltonian.
        """

        return cls.load_hamiltonian_from_file(hamiltonian_class_specifier=hamiltonian_class_specifier,
                                              hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                              default_backend=default_backend)

    def write_to_file(self,
                      hamiltonian=None,
                      overwrite_if_exists=False,
                      ignore_if_exists=True
                      ):

        if hamiltonian is None:
            hamiltonian = self

        known_energies_dict = hamiltonian.get_known_energies_dict()
        class_specific_data = hamiltonian._class_specific_data
        return self.write_hamiltonian_to_file(hamiltonian=hamiltonian,
                                              known_energies_dict=known_energies_dict,
                                              class_specific_data=class_specific_data,
                                              overwrite_if_exists=overwrite_if_exists,
                                              ignore_if_exists=ignore_if_exists)

    def get_file_path_main(self):
        return self.construct_base_path()

    def write_solutions_to_file(self,
                                which: str = 'all'):
        known_energies_dict = self.get_known_energies_dict()
        if known_energies_dict is not None:
            # Only the extrema this call writes are mapped. Both are states of the current
            # representation - every gauge call carries them forward - so mapping the one
            # this call does not write would be work nobody reads.
            known_energies_dict = dict(known_energies_dict)
            _names = ['lowest', 'highest'] if which.lower() == 'all' else [which.lower()]
            for _name in _names:
                state = known_energies_dict.get(f'{_name}_energy_state', None)
                energy = known_energies_dict.get(f'{_name}_energy', None)
                if state is not None and energy is not None:
                    known_energies_dict[f'{_name}_energy_state'] = self._prepare_solution_for_storage(
                        bitstring=state,
                        energy=energy)

            file_path_main = self._get_file_path_main(
                hamiltonian_class_specifier=self.hamiltonian_class_specifier,
                hamiltonian_instance_specifier=self.hamiltonian_instance_specifier)

            self._write_hamiltonian_solutions(file_path_main=file_path_main,
                                              known_energies_dict=known_energies_dict,
                                              which=which)

    def append_solution_to_archive(self,
                                   bitstring,
                                   energy: float,
                                   solver_name: str = "Unknown",
                                   solver_runtime: float = 0.0):
        """
        Record one solver-produced solution in the instance's SolutionsArchive file.

        The archive is a sibling of KnownSolutions holding every solution any solver
        produced for this instance, deduplicated on (state, energy) - including results
        that did not improve on the stored extrema. KnownSolutions keeps only extremal
        certificates; the archive provides a pool of good states (warm-start seeds,
        degenerate ground states, solver comparisons). Instances without both specifiers
        have no storage identity, so the call is a no-op for them.
        """
        if self.hamiltonian_class_specifier is None or self.hamiltonian_instance_specifier is None:
            return
        bitstring = self._prepare_solution_for_storage(bitstring=bitstring,
                                                       energy=energy)
        file_path_main = self._get_file_path_main(
            hamiltonian_class_specifier=self.hamiltonian_class_specifier,
            hamiltonian_instance_specifier=self.hamiltonian_instance_specifier)
        self._write_single_solution(full_path=f"{file_path_main}{MAIN_KEY_SEPARATOR}SolutionsArchive",
                                    bitstring=bitstring,
                                    energy=energy,
                                    solver_name=solver_name,
                                    solver_runtime=solver_runtime)

    def read_solutions_archive(self):
        """Return the SolutionsArchive dataframe, or None if absent (or no specifiers)."""
        if self.hamiltonian_class_specifier is None or self.hamiltonian_instance_specifier is None:
            return None
        file_path_main = self._get_file_path_main(
            hamiltonian_class_specifier=self.hamiltonian_class_specifier,
            hamiltonian_instance_specifier=self.hamiltonian_instance_specifier)
        return self.read_results(full_path=f"{file_path_main}{MAIN_KEY_SEPARATOR}SolutionsArchive",
                                 return_none_if_not_found=True,
                                 format_type='dataframe')

    def compute_sums_of_weights(self,
                                localities: Optional[List[int]] = None) -> Dict[int, float]:
        if localities is None:
            localities = self.localities

        if self.default_backend == 'cupy':
            import cupy as _bck
        else:
            import numpy as _bck

        _sums_dict = {}

        if self.is_two_local:
            single_body_terms = self.local_fields
            two_body_terms = self.couplings

            if single_body_terms is None:
                sum_1q = 0.0
            else:
                sum_1q = _bck.sum(single_body_terms)

            if two_body_terms is None:
                sum_2q = 0.0
            else:
                sum_2q = _bck.sum(_bck.triu(two_body_terms, k=1))
            _sums_dict[1] = float(sum_1q)
            _sums_dict[2] = float(sum_2q)
        else:
            for locality in localities:
                sum_kq = 0.0
                for weight, qubits in self.hamiltonian:
                    if len(qubits) == locality:
                        sum_kq += weight
                _sums_dict[locality] = float(sum_kq)

        return _sums_dict


class ClassicalHamiltonian(ClassicalHamiltonianBase):
    def __init__(self,
                 hamiltonian_list_representation: List[Tuple[float | int, Tuple[int, ...]]],
                 number_of_qubits: int,
                 hamiltonian_class_specifier: Optional[HamiltonianClassSpecifierGeneral | str] = None,
                 hamiltonian_instance_specifier: Optional[HamiltonianInstanceSpecifierGeneral | str] = None,
                 known_energies_dict: Optional[dict] = None,
                 class_specific_data: Optional[dict] = None,
                 default_backend: str = None):
        #TODO(FBM): this is really two-local Hamiltonian subclass

        super().__init__(hamiltonian=hamiltonian_list_representation,
                         number_of_qubits=number_of_qubits,
                         hamiltonian_class_specifier=hamiltonian_class_specifier,
                         hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                         known_energies_dict=known_energies_dict,
                         class_specific_data=class_specific_data,
                         default_backend=default_backend)



        self._csr_matrix = None
        #
        # if self.is_sparse:
        #     print('Initializing sparse representation')
        #     self._csr_matrix = self.get_csr_matrix(backend='cupy' if self.default_backend == 'cupy' else 'scipy')
        #     print('done.')


    @property
    def is_sparse(self):
        density = self.density
        if density is not None:
            return density<0.5
        return False


    @property
    def local_fields(self):
        # if 1 in self.localities:
        return self._local_fields

    @property
    def couplings(self):
        return self._couplings

    @property
    def number_of_edges(self):
        if self.is_two_local:
            return self._two_local_properties['number_of_edges']
        return None

    @property
    def average_degree(self):
        if self.is_two_local:
            return self._two_local_properties['average_degree']
        return None

    @property
    def density(self):
        if self.is_two_local:
            return self._two_local_properties['density']
        return None

    def get_csr_arrays(self,
                      backend='numpy') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (indptr, indices, data) for CSR representation."""

        if not self.is_two_local:
            raise ValueError("Can only return CSR arrays for two-local Hamiltonians.")

        indptr = [0]
        indices = []
        data = []
        if self.has_local_fields:
            local_fields = anf.convert_cupy_numpy_array(array=self.local_fields,
                                                        output_backend=backend)

        couplings = anf.convert_cupy_numpy_array(array=self.couplings,
                                                 output_backend=backend)

        for qi in range(self.number_of_qubits):
            if self.has_local_fields:
                if local_fields[qi] != 0.0:
                    indices.append(qi)
                    data.append(local_fields[qi])

            for qj in range(self.number_of_qubits):
                if couplings[qi, qj] == 0.0 or qi == qj:
                    continue
                indices.append(qj)
                data.append(couplings[qi, qj])

            indptr.append(len(indices))

        if backend == 'numpy':
            return np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32), np.array(data, dtype=np.float64)
        elif backend == 'cupy':
            return cp.array(indptr, dtype=cp.int32), cp.array(indices, dtype=cp.int32), cp.array(data, dtype=cp.float64)
        else:
            raise ValueError(f"Unknown backend {backend}")


    def get_csr_matrix(self,
                       backend:Optional[str]=None):

        if backend is None:
            backend = self.default_backend

        if (self.default_backend == backend) or (self.default_backend == 'numpy' and backend == 'scipy'):
            if self._csr_matrix is not None:
                return self._csr_matrix


        if backend in ['scipy','numpy']:
            arrays = self.get_csr_arrays(backend='numpy')

            csr_matrix = scipy.sparse.csr_matrix((arrays[2], arrays[1], arrays[0]), shape=(self.number_of_qubits, self.number_of_qubits))

        elif backend == 'cupy':
            arrays = self.get_csr_arrays(backend=backend)
            csr_matrix = cp.sparse.csr_matrix((arrays[2], arrays[1], arrays[0]), shape=(self.number_of_qubits, self.number_of_qubits))
        else:
            raise ValueError(f"Unknown backend {backend}")

        if backend == self.default_backend:
            self._csr_matrix = csr_matrix
        return csr_matrix


       # print('Initializing sparse representation')

    def get_edges(self,
                  include_weights: bool = False):
        if self.is_two_local:
            if self.has_local_fields:
                _verifier = lambda x: len(x[1]) == 2
            else:
                _verifier = lambda x: True

            if include_weights:
                _indexer = lambda x: x
            else:
                _indexer = lambda x: x[1]

            for tup in self.hamiltonian:
                if _verifier(tup):
                    yield _indexer(tup)

        else:
            return None


    def get_fields_and_couplings(self, precision: Optional[type] = np.float32):
        return get_fields_and_couplings_from_hamiltonian(self,
                                                         precision=precision)

    def get_adjacency_matrix(self,
                             matrix_type: str = 'SYM',
                             backend: Optional[str] = None,
                             precision=np.float32) -> Union[np.ndarray, cp.ndarray]:

        assert set(self.localities) in [{1}, {2},
                                        {1, 2}], "Adjacency matrix can only be obtained for 2-local Hamiltonians"

        if backend is None:
            backend = self._default_backend

        couplings = self.couplings
        local_fields = self.local_fields

        if couplings is None or (local_fields is None and self.has_local_fields):
            return convert_list_representation_to_adjacency_matrix(hamiltonian_list_representation=self.hamiltonian,
                                                                   matrix_type=matrix_type,
                                                                   backend=backend,
                                                                   number_of_qubits=self.number_of_qubits,
                                                                   precision=precision)

        if backend == 'numpy':
            import numpy as bck
        elif backend == 'cupy':
            import cupy as bck
        else:
            raise ValueError("Backend not recognized")

        couplings = anf.convert_cupy_numpy_array(array=couplings,
                                                 output_backend=backend)

        # One copy, at the requested precision: the float64 cache is never modified, and a
        # float32 request never holds a second float64 matrix.
        adjacency_matrix = couplings.astype(precision, copy=True)
        if self.has_local_fields:
            local_fields = anf.convert_cupy_numpy_array(array=local_fields,
                                                        output_backend=backend)
            bck.fill_diagonal(adjacency_matrix, local_fields)

        return adjacency_matrix

    def get_couplings_and_local_fields(self,
                                       matrix_type: str = 'SYM',
                                       backend: Optional[str] = None,
                                       precision: type = np.float32):
        assert set(self.localities) in [{1}, {2},
                                        {1, 2}], "Adjacency matrix can only be obtained for 2-local Hamiltonians"

        if backend is None:
            backend = self._default_backend
        couplings = self.get_adjacency_matrix(backend=backend,
                                              matrix_type=matrix_type,
                                              precision=precision)
        local_fields = np.diag(couplings).copy()
        np.fill_diagonal(couplings, 0)

        if np.all(local_fields == 0):
            local_fields = None

        return couplings, local_fields


def get_fields_and_couplings_from_hamiltonian_list(hamiltonian: List[Tuple[Union[float, int], Tuple[int, ...]]],
                                                   number_of_qubits=None,
                                                   precision: Optional[type] = np.float32):
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


def get_fields_and_couplings_from_hamiltonian(hamiltonian: ClassicalHamiltonian,
                                              precision: Optional[type] = np.float32):
    assert set(hamiltonian.localities) in [{1}, {2}, {1, 2}], "Only 1-local and 2-local have fields and correlations"

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
        correlations = np.zeros((hamiltonian._number_of_qubits, hamiltonian._number_of_qubits), dtype=precision)
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

    if hamiltonian._default_backend == 'cupy':
        fields = cp.asnumpy(fields)
        correlations = cp.asnumpy(correlations)

    fields = np.array(fields, dtype=precision)
    correlations = np.array(correlations, dtype=precision)
    return fields, correlations
