# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


"""
Description string parsing and reconstruction functionality.

This module provides functions to parse standardized description strings back into
their corresponding specifier objects, enabling automatic reconstruction of
Hamiltonian class and instance specifiers from their string representations.
"""

import ast

from quapopt.data_analysis.data_handling.schemas.naming import (
    ERDOS_RENYI_TYPES,
    MAIN_KEY_SEPARATOR,
    MAIN_KEY_VALUE_SEPARATOR,
    STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS,
    STANDARD_NAMES_VARIABLES,
    SUB_KEY_SEPARATOR,
    SUB_KEY_VALUE_SEPARATOR,
    CoefficientsDistribution,
    CoefficientsDistributionSpecifier,
    CoefficientsType,
    HamiltonianClassSpecifierErdosRenyi,
    HamiltonianClassSpecifierLABS,
    HamiltonianClassSpecifierMaxCut,
    HamiltonianClassSpecifierMAXkSAT,
    HamiltonianClassSpecifierRegular,
    HamiltonianClassSpecifierSK,
    HamiltonianClassSpecifierWishartPlantedEnsemble,
    HamiltonianInstanceSpecifierGeneral,
    HamiltonianModels,
)


def _find_matching_base_name_value(value_string: str, standard_names_collection):
    """
    Find a matching BaseName value from a standard names collection using standardized lookup.

    Args:
        value_string: String to match (could be id or id_long)
        standard_names_collection: StandardNamesBase collection to search in

    Returns:
        Matching BaseName instance or None if not found
    """
    # Use the existing detect_base_name functionality
    detected = standard_names_collection.detect_base_name(value_string)
    return detected


def parse_description_string(description: str) -> dict:
    """
    Parse a description string back into a dictionary of field names and values.

    Args:
        description: Description string like "HMN=ER;LOC=(2,);CFD=CT~DIS_CDN~UNI"

    Returns:
        Dict mapping field names to their parsed values
    """
    # Initialize standard names for detection
    _SN = STANDARD_NAMES_VARIABLES
    _SNH = STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS

    result = {}

    # Split by main separator (;)
    main_pairs = description.split(MAIN_KEY_SEPARATOR)

    for pair in main_pairs:
        if MAIN_KEY_VALUE_SEPARATOR not in pair:
            continue

        key, value = pair.split(MAIN_KEY_VALUE_SEPARATOR, 1)

        # Try to detect the field name using existing functionality
        detected_name = _SN.detect_base_name(name=key)
        if detected_name is None:
            detected_name = _SNH.detect_base_name(name=key)

        field_name = detected_name.id_long if detected_name else key

        # Parse the value based on its format
        parsed_value = _parse_value(value)
        result[field_name] = parsed_value

    return result


def _parse_value(value: str):
    """Parse a value string into its appropriate Python type."""
    # Handle sub-key-value pairs (e.g., "CT~DIS_CDN~UNI_CDP~values~[-1, 1]")
    if SUB_KEY_VALUE_SEPARATOR in value:
        sub_dict = {}
        sub_pairs = value.split(SUB_KEY_SEPARATOR)

        # Handle cases where we have consecutive ~ separators like "CDP~values~[-1, 1]"
        i = 0
        while i < len(sub_pairs):
            sub_pair = sub_pairs[i]
            if SUB_KEY_VALUE_SEPARATOR in sub_pair:
                sub_key, sub_value = sub_pair.split(SUB_KEY_VALUE_SEPARATOR, 1)

                # Check if the next element should be part of this value
                # This handles cases like "CDP~values~[-1, 1]" where we split by _ but
                # "values~[-1, 1]" should be parsed as "values": [-1, 1]
                if (
                    i + 1 < len(sub_pairs)
                    and SUB_KEY_VALUE_SEPARATOR not in sub_pairs[i + 1]
                ):
                    # The next part is likely a continuation of this value
                    sub_value = sub_value + SUB_KEY_SEPARATOR + sub_pairs[i + 1]
                    i += 1  # Skip the next element since we consumed it

                # Now parse the sub_value, which might itself contain ~
                if SUB_KEY_VALUE_SEPARATOR in sub_value:
                    # This is a nested sub-key-value pair like "values~[-1, 1]"
                    nested_parts = sub_value.split(SUB_KEY_VALUE_SEPARATOR)
                    if len(nested_parts) == 2:
                        nested_key, nested_value = nested_parts
                        sub_dict[sub_key] = {
                            nested_key: _parse_simple_value(nested_value)
                        }
                    else:
                        sub_dict[sub_key] = _parse_simple_value(sub_value)
                else:
                    sub_dict[sub_key] = _parse_simple_value(sub_value)
            i += 1
        return sub_dict

    return _parse_simple_value(value)


def _parse_simple_value(value: str):
    """Parse a simple value string into its Python type."""
    # Handle tuples like "(2,)" or "(1, 2)"
    if value.startswith("(") and value.endswith(")"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

    # Handle lists like "[-1, 1]"
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

    # Handle numbers
    try:
        if "." in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass

    # Return as string if nothing else worked
    return value


def reconstruct_hamiltonian_class_specifier(class_description: str):
    """
    Reconstruct a HamiltonianClassSpecifier from its description string.

    Args:
        class_description: Class description string (e.g., "HMN=ER;LOC=(2,);CFD=CT~DIS_CDN~UNI...")

    Returns:
        Appropriate HamiltonianClassSpecifier instance
    """
    # Parse the description string
    class_data = parse_description_string(class_description)

    _SNH = STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS

    # Extract Hamiltonian model name using standardized field names
    hamiltonian_model_name = class_data.get(
        _SNH.HamiltonianModelName.id_long, class_data.get(_SNH.HamiltonianModelName.id)
    )

    # Parse coefficients distribution if present
    coefficients_distribution_specifier = None
    cfd_data = class_data.get(
        _SNH.CoefficientsDistributionSpecifier.id_long,
        class_data.get(_SNH.CoefficientsDistributionSpecifier.id),
    )
    if cfd_data and isinstance(cfd_data, dict):
        coefficients_distribution_specifier = (
            _reconstruct_coefficients_distribution_specifier(cfd_data)
        )

    # Parse localities using standardized field names
    localities = class_data.get(
        _SNH.Localities.id_long, class_data.get(_SNH.Localities.id)
    )

    # Parse Erdos-Renyi type if present using standardized field names
    erdos_renyi_type = None
    ert_data = class_data.get(
        _SNH.ErdosRenyiType.id_long, class_data.get(_SNH.ErdosRenyiType.id)
    )
    if ert_data:
        # Use standardized name system to map string to enum
        erdos_renyi_type = _find_matching_base_name_value(ert_data, ERDOS_RENYI_TYPES)

    # Use standardized name system to determine Hamiltonian model
    hamiltonian_model = _find_matching_base_name_value(
        hamiltonian_model_name, HamiltonianModels
    )

    # Create appropriate specifier based on model using standardized comparisons
    if hamiltonian_model == HamiltonianModels.ErdosRenyi:
        return HamiltonianClassSpecifierErdosRenyi(
            Localities=localities,
            CoefficientsDistributionSpecifier=coefficients_distribution_specifier,
            ErdosRenyiType=erdos_renyi_type,
        )
    elif hamiltonian_model == HamiltonianModels.MaxCut:
        return HamiltonianClassSpecifierMaxCut(
            CoefficientsDistributionSpecifier=coefficients_distribution_specifier,
            ErdosRenyiType=erdos_renyi_type,
        )
    elif hamiltonian_model == HamiltonianModels.SherringtonKirkpatrick:
        return HamiltonianClassSpecifierSK(
            Localities=localities,
            CoefficientsDistributionSpecifier=coefficients_distribution_specifier,
        )
    elif hamiltonian_model == HamiltonianModels.RegularGraph:
        return HamiltonianClassSpecifierRegular(
            Localities=localities,
            CoefficientsDistributionSpecifier=coefficients_distribution_specifier,
        )
    elif hamiltonian_model == HamiltonianModels.MAXkSAT:
        return HamiltonianClassSpecifierMAXkSAT()
    elif hamiltonian_model == HamiltonianModels.WishartPlantedEnsemble:
        return HamiltonianClassSpecifierWishartPlantedEnsemble()
    elif hamiltonian_model == HamiltonianModels.LABS:
        return HamiltonianClassSpecifierLABS()
    else:
        raise ValueError(f"Unknown hamiltonian model: {hamiltonian_model_name}")


def _reconstruct_coefficients_distribution_specifier(cfd_data: dict):
    """Reconstruct CoefficientsDistributionSpecifier from parsed data."""
    # Parse coefficients type
    coefficients_type = None
    _SN = STANDARD_NAMES_VARIABLES

    if STANDARD_NAMES_VARIABLES.CoefficientsType.id in cfd_data:
        ct_value = cfd_data[_SN.CoefficientsType.id]
        coefficients_type = _find_matching_base_name_value(ct_value, CoefficientsType)

    # Parse coefficients distribution name
    coefficients_distribution_name = None
    if _SN.CoefficientsDistributionName.id in cfd_data:
        cdn_value = cfd_data[_SN.CoefficientsDistributionName.id]
        coefficients_distribution_name = _find_matching_base_name_value(
            cdn_value, CoefficientsDistribution
        )

    # Parse coefficients distribution properties
    coefficients_distribution_properties = {}

    # Check if CDP (CoefficientsDistributionProperties) field exists
    if _SN.CoefficientsDistributionProperties.id in cfd_data:
        cdp_data = cfd_data[_SN.CoefficientsDistributionProperties.id]
        if isinstance(cdp_data, dict):
            coefficients_distribution_properties = cdp_data
        else:
            # Handle case where CDP is a single value
            coefficients_distribution_properties = {"value": cdp_data}

    # Also check for any other properties not in the standard fields
    for key, val in cfd_data.items():
        if key not in [
            _SN.CoefficientsType.id,
            _SN.CoefficientsDistributionName.id,
            _SN.CoefficientsDistributionProperties.id,
        ]:
            coefficients_distribution_properties[key] = val

    if coefficients_type and coefficients_distribution_name:
        return CoefficientsDistributionSpecifier(
            CoefficientsType=coefficients_type,
            CoefficientsDistributionName=coefficients_distribution_name,
            CoefficientsDistributionProperties=coefficients_distribution_properties,
        )

    return None


def reconstruct_hamiltonian_instance_specifier(
    instance_description: str, class_specifier
):
    """
    Reconstruct a HamiltonianInstanceSpecifier from its description string.

    Args:
        instance_description: Instance description string (e.g., "NOQ=8;HII=3;EPA=0.5")
        class_specifier: The corresponding HamiltonianClassSpecifier

    Returns:
        Appropriate HamiltonianInstanceSpecifier instance
    """
    # Parse the description string
    instance_data = parse_description_string(instance_description)

    # Extract common parameters using standardized names
    _SN = STANDARD_NAMES_VARIABLES
    _SNH = STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS
    number_of_qubits = instance_data.get(
        _SN.NumberOfQubits.id_long, instance_data.get(_SN.NumberOfQubits.id)
    )
    hamiltonian_instance_index = instance_data.get(
        _SNH.HamiltonianInstanceIndex.id_long,
        instance_data.get(_SNH.HamiltonianInstanceIndex.id),
    )

    # Check if we need EdgeProbabilityOrAmount
    edge_probability_or_amount = instance_data.get(
        _SNH.EdgeProbabilityOrAmount.id_long,
        instance_data.get(_SNH.EdgeProbabilityOrAmount.id),
    )

    # Check if we need WishartDensity
    wishart_density = instance_data.get(
        _SNH.WishartDensity.id_long, instance_data.get(_SNH.WishartDensity.id)
    )

    # Use the class specifier's instance_specifier_constructor if available
    if hasattr(class_specifier, "instance_specifier_constructor"):
        kwargs = {
            _SN.NumberOfQubits.id_long: number_of_qubits,
            _SNH.HamiltonianInstanceIndex.id_long: hamiltonian_instance_index,
        }
        if edge_probability_or_amount is not None:
            kwargs[_SNH.EdgeProbabilityOrAmount.id_long] = edge_probability_or_amount
        if wishart_density is not None:
            kwargs[_SNH.WishartDensity.id_long] = wishart_density

        return class_specifier.instance_specifier_constructor(**kwargs)
    else:
        # Fallback to general instance specifier
        return HamiltonianInstanceSpecifierGeneral(
            NumberOfQubits=number_of_qubits,
            HamiltonianInstanceIndex=hamiltonian_instance_index,
        )
