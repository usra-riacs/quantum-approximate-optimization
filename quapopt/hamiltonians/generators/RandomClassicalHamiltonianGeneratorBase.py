# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import itertools
from typing import Callable, List, Optional, Union

import numpy as np

from quapopt.data_analysis.data_handling import (
    BaseName,
    CoefficientsDistribution,
    CoefficientsDistributionSpecifier,
    CoefficientsType,
    HamiltonianClassSpecifierGeneral,
    HamiltonianInstanceSpecifierGeneral,
    HamiltonianModels,
)
from quapopt.hamiltonians.representation.ClassicalHamiltonian import (
    ClassicalHamiltonian,
)


def _get_default_coefficient_sampling_function(
    coefficients_distribution: BaseName,
    coefficients_type: BaseName,
    coefficients_distribution_properties: dict,
) -> Callable[[np.random.Generator, int], Union[float, int, List[int], List[float]]]:

    if coefficients_distribution in [CoefficientsDistribution.Custom]:
        return None

    if coefficients_type == CoefficientsType.CONSTANT:

        def _coefficient_sampling_function(
            numpy_rng: Optional[np.random.Generator] = None, size: Optional[int] = None
        ) -> Union[float, int, List[float], List[int]]:

            if isinstance(coefficients_distribution_properties, dict):
                value_coeff = coefficients_distribution_properties["value"]
            elif isinstance(coefficients_distribution_properties, (int, float)):
                value_coeff = coefficients_distribution_properties

            if size is None:
                return value_coeff
            return [value_coeff] * size

    elif coefficients_type in [CoefficientsType.CONTINUOUS]:
        if coefficients_distribution in [CoefficientsDistribution.Uniform]:
            _low = coefficients_distribution_properties["low"]
            _high = coefficients_distribution_properties["high"]

            def _coefficient_sampling_function(
                numpy_rng: np.random.Generator, size: Optional[int] = None
            ) -> Union[float, int, List[float], List[int]]:

                numbers = numpy_rng.uniform(low=_low, high=_high, size=size)
                return numbers.tolist() if size is not None else numbers

        elif coefficients_distribution in [CoefficientsDistribution.Normal]:
            _loc = coefficients_distribution_properties["loc"]
            _scale = coefficients_distribution_properties["scale"]

            def _coefficient_sampling_function(
                numpy_rng: np.random.Generator, size: Optional[int] = None
            ) -> Union[float, int, List[float], List[int]]:

                numbers = numpy_rng.normal(loc=_loc, scale=_scale, size=size)
                return numbers.tolist() if size is not None else numbers

    elif coefficients_type in [CoefficientsType.DISCRETE]:
        if coefficients_distribution in [CoefficientsDistribution.Normal]:
            _loc = coefficients_distribution_properties["loc"]
            _scale = coefficients_distribution_properties["scale"]

            def _coefficient_sampling_function(
                numpy_rng: np.random.Generator, size: Optional[int] = None
            ) -> Union[float, int, List[float], List[int]]:

                numbers = np.int32(numpy_rng.normal(loc=_loc, scale=_scale, size=size))

                return numbers.tolist() if size is not None else numbers

        elif coefficients_distribution in [CoefficientsDistribution.Uniform]:

            if "values" in coefficients_distribution_properties:
                _values = coefficients_distribution_properties["values"]
            else:
                assertion_message = "If 'values' are not provided, 'low', 'high', and 'step' must be provided."

                cdp = coefficients_distribution_properties

                assert (
                    "low" in cdp and "high" in cdp and "step" in cdp
                ), assertion_message

                _low = coefficients_distribution_properties["low"]
                _high = coefficients_distribution_properties["high"]
                _step = coefficients_distribution_properties["step"]
                _values = np.arange(_low, _high + 1, _step, dtype=np.int32)

            coeffs_range = sorted(list(set(_values) - {0}))

            def _coefficient_sampling_function(
                numpy_rng: np.random.Generator, size: Optional[int] = None
            ) -> Union[float, int, List[float], List[int]]:
                numbers = numpy_rng.choice(a=coeffs_range, size=size)
                return numbers.tolist() if size is not None else numbers

    else:
        raise ValueError(
            "Invalid coefficients type. Choose from: CONSTANT, CONTINUOUS, DISCRETE."
        )

    return _coefficient_sampling_function


class RandomClassicalHamiltonianGeneratorBase:
    def __init__(
        self,
        # number_of_qubits: int,
        localities: Union[int, List[int]] = None,
        hamiltonian_model_name: BaseName = HamiltonianModels.Unspecified,
        coefficients_distribution_specifier: [CoefficientsDistributionSpecifier] = None,
        hamiltonian_class_specifier: Optional[HamiltonianClassSpecifierGeneral] = None,
    ):
        # TODO(FBM): refactor this so different localities support different distributions

        """

        :param number_of_qubits:
        :param localities:
        All possible localities of the interactions.
        For example, localities=[1,3] means that the Hamiltonian is composed of 1, and 3-local interactions.
        :param average_degree:
        We define "average_degree" as the average number of interactions (of any locality) per qubit.
        Thus max degree of a qubit is given by Newton binomial coefficient if locality is fixed.
        If smaller localities are allowed, the max degree is the sum of all possible coeffs.
        For example, if locality is [1,2,3]
        then the max degree is the sum of all possible 1, 2, and 3-localities
        E.g., qubit 0 could participate in interactions Z_0, Z_01, Z_02, Z_012 for 3-body Hamiltonian.
        This would yield a max degree of 4, as opposed to 1 if smaller localities ([1,2] in this example) are not allowed.
        :param CoefficientsType:
        Whether coefficients are integers or floats.
        :param _coefficients_distribution:
        :param allow_smaller_locality:
        """
        # assert number_of_qubits > 0, "Number of qubits must be positive."

        if hamiltonian_class_specifier is None:
            assert (
                coefficients_distribution_specifier is not None
            ), "If hamiltonian_class_specifier is None, coefficients_distribution_specifier must be provided."
            assert (
                localities is not None
            ), "If hamiltonian_class_specifier is None, localities must be provided."

            hamiltonian_class_specifier = HamiltonianClassSpecifierGeneral(
                hamiltonian_model_name=hamiltonian_model_name,
                localities=localities,
                coefficients_distribution_specifier=coefficients_distribution_specifier,
            )

        else:

            if coefficients_distribution_specifier is not None:
                print(
                    "OVERWRITING coefficients_distribution_specifier with hamiltonian_class_specifier.CoefficientsDistributionSpecifier"
                )
            coefficients_distribution_specifier = (
                hamiltonian_class_specifier.CoefficientsDistributionSpecifier
            )

        hamiltonian_model_name = hamiltonian_class_specifier.HamiltonianModelName

        localities = hamiltonian_class_specifier.Localities

        if isinstance(localities, int):
            localities = [localities]

        #
        # #TODO FBM: think whether this is needed
        # if isinstance(average_degree, float):
        #     assert average_degree > 0, "Average degree must be positive."
        # if isinstance(average_degree, float):
        #     assert average_degree <= max_degree, (
        #         "Average degree must be less than or equal to the maximum degree of a qubit.")

        self._localities = localities
        self._CDS = coefficients_distribution_specifier

        self._hamiltonian_model_name = hamiltonian_model_name
        self._hamiltonian_class_specifier = hamiltonian_class_specifier

    @property
    def hamiltonian_class_specifier(self) -> HamiltonianClassSpecifierGeneral:
        return self._hamiltonian_class_specifier

    @property
    def hamiltonian_class_description(self, long_strings: bool = False) -> str:
        return self.hamiltonian_class_specifier.get_description_string(
            long_strings=long_strings
        )

    @property
    def hamiltonian_model_name(self) -> BaseName:
        return self._hamiltonian_model_name

    @property
    def localities(self) -> List[int]:
        return self._localities

    @property
    def coefficients_type(self) -> BaseName:
        return self._CDS.CoefficientsType

    @property
    def coefficients_distribution(self) -> BaseName:
        return self._CDS.CoefficientsDistributionName

    @property
    def coefficients_distribution_properties(self) -> dict:
        return self._CDS.CoefficientsDistributionProperties

    @property
    def coefficients_distribution_specifier(self) -> CoefficientsDistributionSpecifier:
        return self._CDS

    def _read_from_drive(
        self,
        hamiltonian_instance_specifier: HamiltonianInstanceSpecifierGeneral,
        hamiltonian_class_specifier=None,
    ):

        if hamiltonian_instance_specifier is None:
            print(
                "Hamiltonian instance specifier must be provided for reading from drive. Skipping."
            )
            return None

        if hamiltonian_class_specifier is None:
            hamiltonian_class_specifier = self.hamiltonian_class_specifier
        try:
            return ClassicalHamiltonian.initialize_from_file(
                hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                hamiltonian_class_specifier=hamiltonian_class_specifier,
            )
        except FileNotFoundError:
            #print("File not found!")
            return None

    # def get_class_and_instance_descriptions(self,
    #                                         hamiltonian_instance_index:int):
    #     if self.hamiltonian_class_specifier is None:
    #     hamiltonian_specifier = HamiltonianInstanceSpecififerGeneral(number_of_qubits=self.number_of_qubits,
    #                                                                  hamiltonian_instance_index=hamiltonian_instance_index)
    #

    def _generate_instance(
        self,
        number_of_qubits: int,
        subsets_generator: Union[
            list, tuple, iter, Callable[[np.random.Generator], Union[list, tuple, iter]]
        ] = None,
        term_addition_function: Optional[
            Callable[[np.random.Generator, int], np.ndarray[bool]]
        ] = None,
        coefficient_sampling_function: Optional[
            Callable[[np.random.Generator], Union[float, int]]
        ] = None,
        seed: Optional[int] = None,
        read_from_drive_if_present=True,
        hamiltonian_instance_specifier=None,
        class_specific_data=None,
        random_instance: ClassicalHamiltonian = None,
        default_backend: Optional[str] = None,
    ) -> ClassicalHamiltonian:
        """
        Generates a random classical Hamiltonian instance.
        :param seed:
        :return:
        """

        # Holder for random instance
        if random_instance is not None:
            return random_instance

        hamiltonian_class_specifier = self.hamiltonian_class_specifier

        if hamiltonian_instance_specifier is None:
            hamiltonian_instance_specifier = (
                hamiltonian_class_specifier.instance_specifier_constructor(
                    NumberOfQubits=number_of_qubits, HamiltonianInstanceIndex=seed
                )
            )

        if read_from_drive_if_present:
            hamiltonian = self._read_from_drive(
                hamiltonian_instance_specifier=hamiltonian_instance_specifier
            )
            if hamiltonian is not None:
                return hamiltonian

        if number_of_qubits > 500:
            print("Generating new instance.")

        if self.coefficients_distribution == CoefficientsDistribution.Custom:
            raise NotImplementedError(
                f"This method must be implemented in a subclass for 'custom' coefficients distribution."
            )

        numpy_rng = np.random.default_rng(seed)
        if subsets_generator is None:
            subsets_generator = itertools.combinations(
                range(number_of_qubits), self._localities[0]
            )
            for locality_i in self._localities[1:]:
                subsets_generator = itertools.chain(
                    subsets_generator,
                    itertools.combinations(range(number_of_qubits), locality_i),
                )
        # if term_addition_function is None:
        #     # def term_addition_function(numpy_rng) -> bool:
        #
        #     def term_addition_function(numpy_rng) -> bool:

        if coefficient_sampling_function is None:
            coefficient_sampling_function = _get_default_coefficient_sampling_function(
                coefficients_distribution=self.coefficients_distribution,
                coefficients_type=self.coefficients_type,
                coefficients_distribution_properties=self.coefficients_distribution_properties,
            )

        if isinstance(subsets_generator, Callable):
            subsets_generator = subsets_generator(numpy_rng)

        all_potential_terms = list(subsets_generator)

        if term_addition_function is None:
            terms_mask = np.full(len(all_potential_terms), True)
        else:
            terms_mask = term_addition_function(numpy_rng, len(all_potential_terms))

        all_terms = [
            term for term, mask in zip(all_potential_terms, terms_mask) if mask
        ]
        all_coeffs = coefficient_sampling_function(numpy_rng, len(all_terms))

        hamiltonian = [
            (coeff, tuple(sorted(set(term))))
            for coeff, term in zip(all_coeffs, all_terms)
        ]

        return ClassicalHamiltonian(
            number_of_qubits=number_of_qubits,
            hamiltonian_list_representation=hamiltonian,
            hamiltonian_class_specifier=hamiltonian_class_specifier,
            hamiltonian_instance_specifier=hamiltonian_instance_specifier,
            class_specific_data=class_specific_data,
            default_backend=default_backend,
        )

    def generate_instance(self, **params) -> ClassicalHamiltonian:
        """
        Generates a random classical Hamiltonian instance.
        :param params:
        :return:
        """
        return self._generate_instance(**params)


if __name__ == "__main__":
    import time

    import numba.cuda as nb_cuda
    import numpy as np

    from quapopt.additional_packages.ancillary_functions_usra import (
        efficient_math as em,
    )

    cdp = CoefficientsDistributionSpecifier(
        CoefficientsType=CoefficientsType.DISCRETE,
        CoefficientsDistributionName=CoefficientsDistribution.Uniform,
        CoefficientsDistributionProperties={"low": -10, "high": 10, "step": 1},
    )
    hamiltonian_spec = HamiltonianClassSpecifierGeneral(
        hamiltonian_model_name=HamiltonianModels.GeneralRandom,
        localities=[1, 2],
        coefficients_distribution_specifier=cdp,
    )

    RCHGB = RandomClassicalHamiltonianGeneratorBase(
        hamiltonian_class_specifier=hamiltonian_spec
    )

    number_of_qubits = 10000
    random_hamiltonian = RCHGB._generate_instance(
        seed=42,
        number_of_qubits=number_of_qubits,
    )

    print("got hamiltonian")
    n_bitstrings = 10**5
    numpy_rng = np.random.default_rng(seed=42)
    random_bitstrings = numpy_rng.integers(
        low=0, high=2, size=(n_bitstrings, number_of_qubits), dtype=np.int32
    )
    d_energies = nb_cuda.device_array(n_bitstrings, dtype=np.float32)

    t1 = time.perf_counter()
    energies_2 = em.calculate_energies_from_bitstrings(
        observable=random_hamiltonian.hamiltonian,
        measurement_values=random_bitstrings,
    )
    print("done with numpy")
    t2 = time.perf_counter()
    energies_3 = em.cuda_calculate_energies_from_bitstrings(
        hamiltonian=random_hamiltonian.hamiltonian,
        bitstrings=random_bitstrings,
        d_energies=d_energies,
    )
    print("done with cuda")
    t3 = time.perf_counter()
    energies_4 = em.cython_calculate_energies_from_bitstrings(
        hamiltonian=random_hamiltonian.hamiltonian,
        bitstrings=random_bitstrings,
    )
    print("done with cython")

    t4 = time.perf_counter()

    # test = precompute_gpu(rank=0,
    #                       first_qubit_first_bit=True)

    print("CUDA:", np.allclose(energies_2, energies_3))
    print("CYTHON", np.allclose(energies_2, energies_4))

    print("Numpy reduce: ", t2 - t1)
    print("Cuda: ", t3 - t2)
    print("Cython: ", t4 - t3)
