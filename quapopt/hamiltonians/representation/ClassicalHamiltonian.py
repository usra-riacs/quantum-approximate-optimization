# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import os
from typing import Any, List, Optional, Tuple, Union

from quapopt.data_analysis.data_handling.io_utilities import IOHamiltonianMixin, IOMixin
from quapopt.data_analysis.data_handling.schemas import STANDARD_NAMES_VARIABLES as SNV
from quapopt.data_analysis.data_handling.schemas import (
    HamiltonianClassSpecifierGeneral,
    HamiltonianInstanceSpecifierGeneral,
)
from quapopt.data_analysis.data_handling.schemas.naming import MAIN_KEY_SEPARATOR
from quapopt.hamiltonians.representation.ClassicalHamiltonianBase import (
    ClassicalHamiltonianBase,
)


class IOClassicalHamiltonianMixin(IOHamiltonianMixin):

    @classmethod
    def _get_file_path_main(
        cls,
        hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral,
        hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral,
    ):

        hamiltonian_class_directory = IOMixin.get_hamiltonian_class_base_path(
            hamiltonian_class_specifier=hamiltonian_class_specifier
        )
        hamiltonian_instance_filename = IOMixin.get_hamiltonian_instance_filename(
            hamiltonian_instance_specifier=hamiltonian_instance_specifier
        )
        file_path_main = (
            f"{hamiltonian_class_directory}/{hamiltonian_instance_filename}"
        )
        return file_path_main

    @classmethod
    def write_hamiltonian_to_file(
        cls,
        hamiltonian: Any,  # should be ClassicalHamiltonian
        known_energies_dict: Optional[dict] = None,
        class_specific_data: Optional[dict] = None,
        overwrite_if_exists=False,
        ignore_if_exists=True,
    ):

        file_path_main = cls._get_file_path_main(
            hamiltonian_class_specifier=hamiltonian.hamiltonian_class_specifier,
            hamiltonian_instance_specifier=hamiltonian.hamiltonian_instance_specifier,
        )
        cls._write_hamiltonian_to_text_file(
            hamiltonian=hamiltonian.hamiltonian_list_representation,
            file_path=file_path_main,
            overwrite_if_exists=overwrite_if_exists,
            ignore_if_exists=ignore_if_exists,
        )
        if known_energies_dict is not None:
            cls._write_hamiltonian_solutions(
                file_path_main=file_path_main, known_energies_dict=known_energies_dict
            )
        if class_specific_data is not None:
            file_path_class_specific_information = (
                f"{file_path_main}{MAIN_KEY_SEPARATOR}ClassSpecificData"
            )
            IOMixin.write_results(
                data=class_specific_data,
                full_path=file_path_class_specific_information,
                format_type="pickle",
            )

    @classmethod
    def load_hamiltonian_from_file(
        cls,
        hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral,
        hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral,
    ):
        file_path_main = cls._get_file_path_main(
            hamiltonian_class_specifier=hamiltonian_class_specifier,
            hamiltonian_instance_specifier=hamiltonian_instance_specifier,
        )

        hamiltonian_list_representation = cls._load_hamiltonian_from_text_file(
            file_path=file_path_main
        )

        file_path_known_solutions = (
            f"{file_path_main}{MAIN_KEY_SEPARATOR}KnownSolutions"
        )

        known_solutions_df = cls.read_results(
            full_path=file_path_known_solutions,
            return_none_if_not_found=True,
            format_type="dataframe",
        )
        known_solutions_df = known_solutions_df.sort_values(by=SNV.Energy.id_long)
        lowest_energy_state = known_solutions_df[SNV.Bitstring.id_long].values[0]
        lowest_energy = known_solutions_df[SNV.Energy.id_long].values[0]
        highest_energy_state = known_solutions_df[SNV.Bitstring.id_long].values[-1]
        highest_energy = known_solutions_df[SNV.Energy.id_long].values[-1]
        known_energies_dict = {
            "lowest_energy_state": lowest_energy_state,
            "lowest_energy": lowest_energy,
            "highest_energy_state": highest_energy_state,
            "highest_energy": highest_energy,
        }

        spectrum = None
        file_path_spectral_information = f"{file_path_main}{MAIN_KEY_SEPARATOR}Spectrum"
        if os.path.exists(f"{file_path_spectral_information}.txt"):
            with open(f"{file_path_spectral_information}.txt", "r") as file:
                spectrum = []
                for line in file:
                    spectrum.append(float(line))

        known_energies_dict["spectrum"] = spectrum

        file_path_class_specific_data = (
            f"{file_path_main}{MAIN_KEY_SEPARATOR}ClassSpecificData"
        )
        class_specific_data = cls.read_results(
            full_path=file_path_class_specific_data,
            return_none_if_not_found=True,
            format_type="pickle",
        )

        number_of_qubits = hamiltonian_instance_specifier.NumberOfQubits

        return ClassicalHamiltonian(
            hamiltonian_list_representation=hamiltonian_list_representation,
            known_energies_dict=known_energies_dict,
            hamiltonian_class_specifier=hamiltonian_class_specifier,
            hamiltonian_instance_specifier=hamiltonian_instance_specifier,
            number_of_qubits=number_of_qubits,
            class_specific_data=class_specific_data,
        )


class ClassicalHamiltonian(ClassicalHamiltonianBase, IOClassicalHamiltonianMixin):
    def __init__(
        self,
        hamiltonian_list_representation: List[
            Tuple[Union[float, int], Tuple[int, ...]]
        ],
        number_of_qubits: int,
        hamiltonian_class_specifier: Optional[HamiltonianClassSpecifierGeneral] = None,
        hamiltonian_instance_specifier: Optional[
            HamiltonianInstanceSpecifierGeneral
        ] = None,
        known_energies_dict: Optional[dict] = None,
        class_specific_data: Optional[dict] = None,
        default_backend: str = None,
    ):

        super().__init__(
            hamiltonian=hamiltonian_list_representation,
            number_of_qubits=number_of_qubits,
            hamiltonian_class_specifier=hamiltonian_class_specifier,
            hamiltonian_instance_specifier=hamiltonian_instance_specifier,
            known_energies_dict=known_energies_dict,
            class_specific_data=class_specific_data,
            default_backend=default_backend,
        )

    @property
    def hamiltonian_list_representation(
        self,
    ) -> List[Tuple[Union[float, int], Tuple[int, ...]]]:
        return self._hamiltonian

    def reverse_list_representation_indices(self):
        # reversed_representation =
        self._hamiltonian = self.get_reversed_list_representations()

    @classmethod
    def initialize_from_file(
        cls,
        hamiltonian_class_specifier: HamiltonianClassSpecifierGeneral,
        hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral,
    ):
        """
        Initialize a ClassicalHamiltonian instance from a file.
        :param hamiltonian_class_specifier: Specifier for the Hamiltonian class.
        :param hamiltonian_instance_specifier: Specifier for the Hamiltonian instance.
        :return: An instance of ClassicalHamiltonian.
        """
        return cls.load_hamiltonian_from_file(
            hamiltonian_class_specifier=hamiltonian_class_specifier,
            hamiltonian_instance_specifier=hamiltonian_instance_specifier,
        )

    def write_to_file(
        self, hamiltonian=None, overwrite_if_exists=False, ignore_if_exists=True
    ):

        if hamiltonian is None:
            hamiltonian = self

        known_energies_dict = hamiltonian.get_known_energies_dict()
        class_specific_data = hamiltonian._class_specific_data
        return self.write_hamiltonian_to_file(
            hamiltonian=hamiltonian,
            known_energies_dict=known_energies_dict,
            class_specific_data=class_specific_data,
            overwrite_if_exists=overwrite_if_exists,
            ignore_if_exists=ignore_if_exists,
        )

    def get_file_path_main(self):
        return self.construct_base_path()

    def write_solutions_to_file(
        self,
    ):
        known_energies_dict = self.get_known_energies_dict()
        if known_energies_dict is not None:
            file_path_main = self._get_file_path_main(
                hamiltonian_class_specifier=self.hamiltonian_class_specifier,
                hamiltonian_instance_specifier=self.hamiltonian_instance_specifier,
            )

            self._write_hamiltonian_solutions(
                file_path_main=file_path_main, known_energies_dict=known_energies_dict
            )
