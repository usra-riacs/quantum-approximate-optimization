# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import itertools
import dataclasses
from typing import Optional, Union, Callable, List, Tuple, ClassVar, Dict
import numpy as np
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian

from quapopt.data_analysis.data_handling import BaseName, \
    HamiltonianInstanceSpecifierGeneral
from quapopt.data_analysis.data_handling import (CoefficientsType,
                                                 CoefficientsDistribution,
                                                 CoefficientsDistributionSpecifier,
                                                 HamiltonianClassSpecifierGeneral,
                                                 HamiltonianModels)

def _get_default_coefficient_sampling_function(coefficients_distribution: BaseName,
                                               coefficients_type: BaseName,
                                               coefficients_distribution_properties: dict)->Callable[[np.random.Generator, int], Union[float, int, List[int], List[float]]]:

    if coefficients_distribution in [CoefficientsDistribution.Custom]:
        return None

    if coefficients_type == CoefficientsType.CONSTANT:
        def _coefficient_sampling_function(numpy_rng: Optional[np.random.Generator]=None,
                                           size:Optional[int]=None) -> Union[float, int, List[float], List[int]]:

            if isinstance(coefficients_distribution_properties,dict):
                value_coeff = coefficients_distribution_properties['value']
            elif isinstance(coefficients_distribution_properties, (int, float)):
                value_coeff = coefficients_distribution_properties

            if size is None:
                return value_coeff
            return [value_coeff]*size

    elif coefficients_type in [CoefficientsType.CONTINUOUS]:
        if coefficients_distribution in [CoefficientsDistribution.Uniform]:
            _low = coefficients_distribution_properties['low']
            _high = coefficients_distribution_properties['high']
            def _coefficient_sampling_function(numpy_rng: np.random.Generator,
                                               size:Optional[int]=None) -> Union[float, int, List[float], List[int]]:

                    numbers = numpy_rng.uniform(low=_low, high=_high,size=size)
                    return numbers.tolist() if size is not None else numbers


        elif coefficients_distribution in [CoefficientsDistribution.Normal]:
            _loc = coefficients_distribution_properties['loc']
            _scale = coefficients_distribution_properties['scale']

            def _coefficient_sampling_function(numpy_rng: np.random.Generator,
                                               size:Optional[int]=None) -> Union[float, int, List[float], List[int]]:

                numbers = numpy_rng.normal(loc=_loc, scale=_scale, size=size)
                return numbers.tolist() if size is not None else numbers
        else:
            raise ValueError(f"Coefficients distribution '{coefficients_distribution}' not supported.")

    elif coefficients_type in [CoefficientsType.DISCRETE]:
        if coefficients_distribution in [CoefficientsDistribution.Normal]:
            # Unreachable via the specifier (which rejects this pair at construction); kept as
            # a second gate because the failure mode is silent -- int32 truncation zeroed most
            # coefficients while leaving them in the term list, so bad data looked healthy.
            raise NotImplementedError(
                "Discrete + Normal coefficient sampling is not supported: truncating the "
                "sampled normals to int32 collapses most coefficients to exactly 0.")

        elif coefficients_distribution in [CoefficientsDistribution.Uniform]:

            if 'values' in coefficients_distribution_properties:
                _values = coefficients_distribution_properties['values']
            else:
                assertion_message = "If 'values' are not provided, 'low', 'high', and 'step' must be provided."

                cdp = coefficients_distribution_properties

                assert 'low' in cdp and 'high' in cdp and 'step' in cdp, assertion_message

                _low = coefficients_distribution_properties['low']
                _high = coefficients_distribution_properties['high']
                _step = coefficients_distribution_properties['step']
                _values = np.arange(_low, _high + 1, _step, dtype=np.int32)


            coeffs_range = sorted(list(set(_values) - {0}))

            def _coefficient_sampling_function(numpy_rng: np.random.Generator,
                                               size:Optional[int]=None) -> Union[float, int, List[float], List[int]]:
                numbers = numpy_rng.choice(a=coeffs_range, size=size)
                return numbers.tolist() if size is not None else numbers

        else:
            raise ValueError(f"Coefficients distribution '{coefficients_distribution}' not supported.")

    else:
        raise ValueError("Invalid coefficients type. Choose from: CONSTANT, CONTINUOUS, DISCRETE.")

    return _coefficient_sampling_function


class RandomClassicalHamiltonianGeneratorBase:
    _SPECIFIER_FIELD_TO_KWARG: ClassVar[Dict[str, Optional[str]]] = {
        'HamiltonianInstanceIndex': 'seed',
        'NumberOfQubits': 'number_of_qubits',
    }


    def __init__(self,
                 localities: Union[int,List[int], Tuple[int,...]] = None,
                 hamiltonian_model_name: BaseName = HamiltonianModels.Unspecified,
                 coefficients_distribution_specifier: CoefficientsDistributionSpecifier = None,
                 hamiltonian_class_specifier:Optional[HamiltonianClassSpecifierGeneral] = None,
                 ):
        #TODO(FBM): refactor this so different localities support different distributions

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
            #assert coefficients_distribution_specifier is not None, "If hamiltonian_class_specifier is None, coefficients_distribution_specifier must be provided."
            assert localities is not None, "If hamiltonian_class_specifier is None, localities must be provided."

            hamiltonian_class_specifier = HamiltonianClassSpecifierGeneral(HamiltonianModelName=hamiltonian_model_name,
                                                                            Localities=localities,
                                                                            CoefficientsDistributionSpecifier=coefficients_distribution_specifier)

        else:

            if coefficients_distribution_specifier is not None:
                print("OVERWRITING coefficients_distribution_specifier with hamiltonian_class_specifier.CoefficientsDistributionSpecifier")
            coefficients_distribution_specifier = hamiltonian_class_specifier.CoefficientsDistributionSpecifier


        hamiltonian_model_name = hamiltonian_class_specifier.HamiltonianModelName

        localities = hamiltonian_class_specifier.Localities

        if isinstance(localities, int):
            localities = [localities]

        # max_degree = int(sum([sc.special.binom(number_of_qubits, i) for i in localities]))
        #
        # #TODO FBM: think whether this is needed
        # if isinstance(average_degree, float):
        #     assert average_degree > 0, "Average degree must be positive."
        # elif isinstance(average_degree, str):
        #     assert average_degree.lower() == 'max', "Invalid average degree. Choose a positive number or 'max'."
        #     average_degree = float(max_degree)
        # if isinstance(average_degree, float):
        #     assert average_degree <= max_degree, (
        #         "Average degree must be less than or equal to the maximum degree of a qubit.")


        #self._number_of_qubits = number_of_qubits
        self._localities = localities
        self._CDS = coefficients_distribution_specifier
        # self._average_degree = average_degree
        # self._max_degree = max_degree

        self._hamiltonian_model_name = hamiltonian_model_name
        self._hamiltonian_class_specifier = hamiltonian_class_specifier

    @property
    def hamiltonian_class_specifier(self) -> HamiltonianClassSpecifierGeneral:
        return self._hamiltonian_class_specifier

    @property
    def hamiltonian_class_description(self,
                                      long_strings:bool=False) -> str:
        return self.hamiltonian_class_specifier.get_description_string(long_strings=long_strings)

    @property
    def hamiltonian_model_name(self) -> BaseName:
        return self._hamiltonian_model_name



    @property
    def localities(self) -> List[int]:
        return self._localities

    @property
    def coefficients_type(self) -> Optional[BaseName]:

        if self._CDS is None:
            return None

        return self._CDS.CoefficientsType

    @property
    def coefficients_distribution(self) -> Optional[BaseName]:
        if self._CDS is None:
            return None
        return self._CDS.CoefficientsDistributionName

    @property
    def coefficients_distribution_properties(self) -> Optional[dict]:
        if self._CDS is None:
            return None

        return self._CDS.CoefficientsDistributionProperties
    @property
    def coefficients_distribution_specifier(self) -> CoefficientsDistributionSpecifier:
        return self._CDS


    def _read_from_drive(self,
                         hamiltonian_instance_specifier:HamiltonianInstanceSpecifierGeneral,
                         hamiltonian_class_specifier = None,
                         default_backend: Optional[str] = None
                         ):

        if hamiltonian_instance_specifier is None:
            print("Hamiltonian instance specifier must be provided for reading from drive. Skipping.")
            return None

        if hamiltonian_class_specifier is None:
            hamiltonian_class_specifier = self.hamiltonian_class_specifier
        try:
            cost = ClassicalHamiltonian.initialize_from_file(
                hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                hamiltonian_class_specifier=hamiltonian_class_specifier,
                default_backend=default_backend)
            cost.was_read_from_drive = True

            return cost

        except FileNotFoundError:
            # print("File not found!")
            return None




    # def get_class_and_instance_descriptions(self,
    #                                         hamiltonian_instance_indices:int):
    #     if self.hamiltonian_class_specifier is None:
    #         class_description = None
    #     else:
    #         class_description = self.hamiltonian_class_specifier.get_description_string(long_strings=False)
    #     hamiltonian_specifier = HamiltonianInstanceSpecififerGeneral(number_of_qubits=self.number_of_qubits,
    #                                                                  hamiltonian_instance_indices=hamiltonian_instance_indices)
    #
    #     instance_description = hamiltonian_specifier.get_description_string(long_strings=False)
    #     return class_description, instance_description




    def _generate_instance(self,
                           number_of_qubits: int,
                           subsets_generator: Union[list, tuple, iter, Callable[[np.random.Generator],Union[list, tuple, iter]]] = None,
                           term_addition_function: Optional[Callable[[np.random.Generator,int], np.ndarray[bool]]] = None,
                           coefficient_sampling_function: Optional[
                                             Callable[[np.random.Generator], Union[float, int]]] = None,
                           seed: Optional[int] = None,
                           read_from_drive_if_present=True,
                           hamiltonian_instance_specifier=None,
                           class_specific_data=None,
                           already_generated_instance: ClassicalHamiltonian = None,
                           default_backend:Optional[str]= None
                           ) -> ClassicalHamiltonian:
        """
        Generates a random classical Hamiltonian instance.
        :param seed:
        :return:
        """

        #Holder for a Hamiltonian instance that is provided by the child class
        if already_generated_instance is not None:
            return already_generated_instance

        hamiltonian_class_specifier = self.hamiltonian_class_specifier

        if hamiltonian_instance_specifier is None:
            hamiltonian_instance_specifier = hamiltonian_class_specifier.instance_specifier_constructor(
                                                                    NumberOfQubits=number_of_qubits,
                                                                    HamiltonianInstanceIndex=seed)

        if read_from_drive_if_present:
            hamiltonian = self._read_from_drive(hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                                default_backend=default_backend)
            if hamiltonian is not None:
                return hamiltonian
            else:
                # print("FILE NOT FOUND!")
                pass

        if self.coefficients_distribution == CoefficientsDistribution.Custom:
            raise NotImplementedError(
                f"This method must be implemented in a subclass for 'custom' coefficients distribution.")

        numpy_rng = np.random.default_rng(seed)
        if subsets_generator is None:
            subsets_generator = itertools.combinations(range(number_of_qubits), self._localities[0])
            for locality_i in self._localities[1:]:
                subsets_generator = itertools.chain(subsets_generator,
                                                    itertools.combinations(range(number_of_qubits), locality_i))

        if coefficient_sampling_function is None:
            coefficient_sampling_function = _get_default_coefficient_sampling_function(
                coefficients_distribution=self.coefficients_distribution,
                coefficients_type=self.coefficients_type,
                coefficients_distribution_properties=self.coefficients_distribution_properties)

        if isinstance(subsets_generator, Callable):
            subsets_generator = subsets_generator(numpy_rng)


        all_potential_terms = list(subsets_generator)

        if term_addition_function is None:
            #TODO(FBM): if we mutate this later, it should be copy, but currently we don't
            all_terms = all_potential_terms
        else:
            terms_mask = term_addition_function(numpy_rng, len(all_potential_terms))
            all_terms = [term for term, mask in zip(all_potential_terms, terms_mask) if mask]

        all_coeffs = coefficient_sampling_function(numpy_rng, len(all_terms))

        hamiltonian = [(coeff, tuple(sorted(set(term)))) for coeff, term in zip(all_coeffs, all_terms)]



        return ClassicalHamiltonian(number_of_qubits=number_of_qubits,
                                    hamiltonian_list_representation=hamiltonian,
                                    hamiltonian_class_specifier=hamiltonian_class_specifier,
                                    hamiltonian_instance_specifier=hamiltonian_instance_specifier,
                                    class_specific_data=class_specific_data,
                                    default_backend=default_backend
                                    )

    def generate_instance(self,
                          **params) -> ClassicalHamiltonian:
        """
        Generates a random classical Hamiltonian instance.
        :param params:
        :return:
        """
        return self._generate_instance(**params)

    @staticmethod
    def _camel_to_snake(text):
        import re
        # Insert an underscore before any capital letter followed by a lowercase letter
        str1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        # Insert an underscore between lowercase letters/numbers and a capital letter
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str1).lower()

    def generate_instance_from_specifier(self,
                                         instance_specifier,
                                         read_from_drive_if_present=True,
                                         default_backend=None,
                                         **additional_kwargs):
        aliases = {}
        #go through subclasses and update kwarg by kwarg
        for klass in reversed(type(self).__mro__):
            aliases.update(getattr(klass, '_SPECIFIER_FIELD_TO_KWARG', {}))
        kwargs = {}
        for f in dataclasses.fields(instance_specifier):
            if not f.init:  # skip computed fields
                continue
            arg = aliases.get(f.name, self._camel_to_snake(f.name))
            value = getattr(instance_specifier, f.name)
            if arg is not None and value is not None:
                kwargs[arg] = value
        kwargs.update(additional_kwargs)
        return self.generate_instance(**kwargs,
                                      read_from_drive_if_present=read_from_drive_if_present,
                                      default_backend=default_backend)
