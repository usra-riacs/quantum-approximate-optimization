# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import enum
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Callable, ClassVar, Dict, Final, List, Optional, Tuple, Union

import numpy as np

##########################################
#############BASIC RULES##################
##########################################


# In strings, will separate the key-value parts,
# e.g. "key1=value1;key2=value2"
MAIN_KEY_SEPARATOR = ";"
MAIN_KEY_VALUE_SEPARATOR = "="

# In strings, sometimes "value" will be also a key-value pair,
# e.g. "key1=subkey1~subvalue1_subkey2~subvalue2;key2=value2"
SUB_KEY_SEPARATOR = "_"
SUB_KEY_VALUE_SEPARATOR = "~"

# In table names, will separate the parts of the table name, e.g. "TablePrefix,TableName,TableSuffix,DataTypeSuffix"
DEFAULT_TABLE_NAME_PARTS_SEPARATOR = ","

# In dataframes, will separate the column name from its type, e.g., "Energy|float64"
DEFAULT_DATAFRAME_NAME_TYPE_SEPARATOR = "|"


def create_sub_name(
    key_value_pairs: Union[Dict[str, Any], Tuple[Tuple[str, Any], ...]],
    major_separator: bool,
    skip_null_values: bool = False,
    skip_attributes: List[str] = (),
) -> str:
    """
    Create a sub name string from key-value pairs.
    :param key_value_pairs:
    :param major_separator:
    :param skip_null_values:
    :param skip_attributes:
    :return:
    """
    if isinstance(key_value_pairs, dict):
        key_value_pairs = key_value_pairs.items()

    if major_separator:
        separator = MAIN_KEY_SEPARATOR
        value_separator = MAIN_KEY_VALUE_SEPARATOR
    else:
        separator = SUB_KEY_SEPARATOR
        value_separator = SUB_KEY_VALUE_SEPARATOR
    if skip_null_values:
        key_value_pairs = [
            (key, value)
            for key, value in key_value_pairs
            if value is not None and key not in skip_attributes
        ]

    return separator.join(
        [f"{key}{value_separator}{value}" for key, value in key_value_pairs]
    )


@dataclass(frozen=True)
class BaseName:
    """Base class for standardized variable names with validation."""

    id: str
    id_long: str
    value_format: Optional[type] = None

    def __post_init__(self):
        if len(self.id) > 3:
            raise ValueError(
                f"'id' must be no longer than 3 characters. Got '{self.id}'"
            )
        # Convert to uppercase
        object.__setattr__(self, "id", self.id.upper())


@dataclass(frozen=True)
class BaseNameDataType:
    """Base class for standardized data type names with categorization."""

    id: str
    id_long: str
    data_category: Optional[str] = None
    data_subcategory: Optional[str] = None
    value_format: Optional[type] = None

    def __post_init__(self):
        if len(self.id) > 3:
            raise ValueError(
                f"'id' must be no longer than 3 characters. Got '{self.id}'"
            )
        # Convert to lowercase
        object.__setattr__(self, "id", self.id.lower())


class StandardNamesBaseVariables:
    """Base class for managing collections of BaseName instances in dataclasses."""

    def __post_init__(self):
        """Validate ID uniqueness across all attributes."""
        all_attrs = self.get_all_attributes()
        all_ids = [attr.id for attr in all_attrs.values()]
        all_long_ids = [attr.id_long for attr in all_attrs.values()]

        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                if all_ids[i] == all_ids[j]:
                    raise ValueError(
                        f"Duplicate ID {all_ids[i]} found in {self.__class__.__name__}"
                    )
                if all_long_ids[i] == all_long_ids[j]:
                    raise ValueError(
                        f"Duplicate long ID {all_long_ids[i]} found in {self.__class__.__name__}"
                    )

    def get_all_attributes(self) -> Dict[str, Union[BaseName, BaseNameDataType]]:
        """
        Returns a dictionary mapping field names to their BaseName instances.
        Optimized for dataclass usage.
        """
        result = {}

        # Scan all non-private attributes for BaseName/BaseNameDataType instances
        for name in dir(self):
            if not name.startswith("_") and not callable(getattr(self, name, None)):
                try:
                    value = getattr(self, name)
                    if isinstance(value, (BaseName, BaseNameDataType)):
                        result[name] = value
                except (AttributeError, TypeError):
                    continue

        return result

    def detect_base_name(
        self, name: str
    ) -> Optional[Union[BaseName, BaseNameDataType]]:
        """Detect BaseName instance by id or id_long."""
        for base in self.get_all_attributes().values():
            if name in [base.id, base.id_long]:
                return base
        return None

    def detect_id(self, name: str) -> Optional[str]:
        """Detect short id from name."""
        base = self.detect_base_name(name)
        return base.id if base is not None else None

    def detect_id_long(self, name: str) -> Optional[str]:
        """Detect long id from name."""
        base = self.detect_base_name(name)
        return base.id_long if base is not None else None

    def detect_format(self, name: str) -> Optional[type]:
        """Detect value format from name."""
        base = self.detect_base_name(name)
        return base.value_format if base is not None else None


class StandardNamesBaseDataTypes(StandardNamesBaseVariables):
    """Extended base class specifically for BaseNameDataType instances."""

    def detect_data_category(self, name: str) -> Optional[str]:
        """Detect data category from name."""
        base = self.detect_base_name(name)
        if base is not None and hasattr(base, "data_category"):
            return base.data_category
        return None

    def detect_data_subcategory(self, name: str) -> Optional[str]:
        """Detect data subcategory from name."""
        base = self.detect_base_name(name)
        if base is not None and hasattr(base, "data_subcategory"):
            return base.data_subcategory
        return None


@dataclass
class StandardizedSpecifier:
    """Modern dataclass-based replacement for StandardizedSpecifier with metadata control."""

    def __post_init__(self):
        """Optional validation hook for subclasses."""

    def _get_annotation_fields(self, long_names: bool = True) -> Dict[str, Any]:
        """Get fields that should be included in annotations based on metadata."""
        annotation_fields = {}

        _SN = STANDARD_NAMES_VARIABLES
        _SNH = STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS

        for field_obj in fields(self):
            # Check if field should be included in annotations (default: True)
            include_in_annotation = field_obj.metadata.get(
                "include_in_annotation", True
            )

            if include_in_annotation:
                field_name = field_obj.name
                field_value = getattr(self, field_name)

                # check if we can detect field_name to be in standard set
                detected_name = _SN.detect_base_name(name=field_name)
                if detected_name is None:
                    detected_name = _SNH.detect_base_name(name=field_name)

                if detected_name is not None:
                    field_name = (
                        detected_name.id_long if long_names else detected_name.id
                    )

                annotation_fields[field_name] = field_value

        return annotation_fields

    def _get_dataframe_annotation(self, long_names: bool = True) -> Dict[str, Any]:
        """Generate dataframe annotations from fields marked for inclusion."""
        annotation = {}
        annotation_fields = self._get_annotation_fields(long_names=long_names)

        for field_name, field_value in annotation_fields.items():
            proper_name = field_name
            if isinstance(field_value, BaseName):
                proper_value = field_value.id_long if long_names else field_value.id
            elif isinstance(field_value, enum.Enum):
                proper_value = field_value.value
            else:
                proper_value = field_value

            # Use the field name as key (could be enhanced to use BaseName.id/id_long)
            annotation[proper_name] = proper_value

        return annotation

    def get_dataframe_annotation(self, *args) -> Dict[str, Any]:
        """Public interface for dataframe annotations."""
        return self._get_dataframe_annotation(*args)

    def _process_field_value(
        self,
        field_value: Any,
        long_strings: bool = False,
        skip_null_values: bool = True,
        skip_attributes: List[str] = None,
    ) -> Any:
        """Process individual field values for description string generation."""
        if skip_attributes is None:
            skip_attributes = []

        if isinstance(field_value, dict):
            processed_dict = {}
            for k, v in field_value.items():
                if isinstance(v, BaseName):
                    processed_dict[k] = v.id_long if long_strings else v.id
                else:
                    processed_dict[k] = v
            return create_sub_name(
                key_value_pairs=processed_dict,
                major_separator=False,
                skip_null_values=skip_null_values,
                skip_attributes=skip_attributes,
            )
        elif isinstance(field_value, StandardizedSpecifier):
            return field_value._get_description_string_general(
                skip_null_values=skip_null_values,
                long_strings=long_strings,
                major_separator=False,
                skip_attributes=skip_attributes,
            )
        elif isinstance(field_value, BaseName):
            return field_value.id_long if long_strings else field_value.id
        elif isinstance(field_value, Callable):
            return None  # Skip callable attributes
        else:
            return field_value

    def _get_description_string_general(
        self,
        skip_null_values: bool = True,
        long_strings: bool = False,
        major_separator: bool = True,
        skip_attributes: List[str] = None,
    ) -> str:
        """Generate description string from annotated fields."""
        if skip_attributes is None:
            skip_attributes = []

        annotation_fields = self._get_annotation_fields(long_names=long_strings)
        key_value_pairs = {}

        for field_name, field_value in annotation_fields.items():
            if field_name in skip_attributes:
                continue

            processed_value = self._process_field_value(
                field_value, long_strings, skip_null_values, skip_attributes
            )

            if processed_value is not None:
                key_value_pairs[field_name] = processed_value

        return create_sub_name(
            key_value_pairs=key_value_pairs,
            major_separator=major_separator,
            skip_null_values=skip_null_values,
            skip_attributes=skip_attributes,
        )

    def get_description_string(self, *args, **kwargs) -> str:
        """Public interface for description string generation."""
        return self._get_description_string_general(*args, **kwargs)

    def merge_with(
        self,
        other: "StandardizedSpecifier",
        allow_conflicts: bool = False,
        in_place: bool = False,
    ) -> Optional["StandardizedSpecifier"]:
        """
        Merge this specifier with another, checking for attribute conflicts.

        Args:
            other: Another StandardizedSpecifier to merge with
            allow_conflicts: If False, raises ValueError when same field has different values
            in_place: If True, modify self instead of returning new instance

        Returns:
            If in_place=False: New StandardizedSpecifier instance with merged attributes
            If in_place=True: None (self is modified)

        Raises:
            ValueError: If conflicting values found and allow_conflicts=False
            TypeError: If other is not a StandardizedSpecifier
        """
        if not isinstance(other, StandardizedSpecifier):
            raise TypeError(
                f"Can only merge with StandardizedSpecifier, got {type(other)}"
            )

        # Get field dictionaries for both specifiers
        self_fields = {f.name: getattr(self, f.name) for f in fields(self)}
        other_fields = {f.name: getattr(other, f.name) for f in fields(other)}

        # Check for conflicts
        conflicts = []
        for field_name in set(self_fields.keys()) & set(other_fields.keys()):
            self_value = self_fields[field_name]
            other_value = other_fields[field_name]

            # Deep comparison handling different types
            if not self._values_equal(self_value, other_value):
                conflicts.append((field_name, self_value, other_value))

        if conflicts and not allow_conflicts:
            conflict_details = []
            for field_name, self_val, other_val in conflicts:
                conflict_details.append(f"  {field_name}: {self_val} != {other_val}")
            raise ValueError(
                f"Conflicting field values found:\n" + "\n".join(conflict_details)
            )

        # Create merged field dictionary (other takes precedence)
        merged_fields = {**self_fields, **other_fields}

        # Handle in-place merge (merge all compatible fields)
        if in_place:
            for field_name, field_value in other_fields.items():
                # Try to set the attribute - if it fails, skip it
                try:
                    setattr(self, field_name, field_value)
                except (AttributeError, TypeError):
                    # Skip fields that can't be set (e.g., init=False computed fields)
                    continue
            return None

        # Create new instance (original behavior)
        # Helper function to filter init-only fields
        def get_init_fields(obj_type):
            return {f.name: f for f in fields(obj_type) if f.init}

        # Check if we can use an existing type (self's type can accommodate all fields)
        self_init_fields = get_init_fields(type(self))
        self_init_field_names = set(self_init_fields.keys())
        if set(merged_fields.keys()).issubset({f.name for f in fields(self)}):
            # All merged fields exist in self's type - filter to init-only fields
            init_only_fields = {
                k: v for k, v in merged_fields.items() if k in self_init_field_names
            }
            return type(self)(**init_only_fields)

        # Check if we can use other's type
        other_init_fields = get_init_fields(type(other))
        other_init_field_names = set(other_init_fields.keys())
        if set(merged_fields.keys()).issubset({f.name for f in fields(other)}):
            # All merged fields exist in other's type - filter to init-only fields
            init_only_fields = {
                k: v for k, v in merged_fields.items() if k in other_init_field_names
            }
            return type(other)(**init_only_fields)

        # Need to create a dynamic type that combines both
        return self._create_merged_type(other, merged_fields)

    def _create_merged_type(
        self, other: "StandardizedSpecifier", merged_fields: Dict[str, Any]
    ) -> "StandardizedSpecifier":
        """
        Create a new dynamic dataclass type that combines fields from both specifiers.

        Args:
            other: The other specifier being merged
            merged_fields: Dictionary of field names to values for the merged instance

        Returns:
            Instance of dynamically created merged type
        """
        # Create field definitions for the new type
        field_definitions = {}

        # Collect field information from both types
        self_field_info = {f.name: f for f in fields(self)}
        other_field_info = {f.name: f for f in fields(other)}

        # Process each field in merged_fields
        for field_name, field_value in merged_fields.items():
            # Determine field type and default from original field definitions
            if field_name in self_field_info:
                original_field = self_field_info[field_name]
                field_type = original_field.type
                field_default = (
                    original_field.default
                    if original_field.default is not MISSING
                    else field_value
                )
            elif field_name in other_field_info:
                original_field = other_field_info[field_name]
                field_type = original_field.type
                field_default = (
                    original_field.default
                    if original_field.default is not MISSING
                    else field_value
                )
            else:
                # This shouldn't happen, but handle it gracefully
                field_type = type(field_value) if field_value is not None else Any
                field_default = field_value

            field_definitions[field_name] = (field_type, field_default)

        # Generate a unique class name
        self_name = type(self).__name__
        other_name = type(other).__name__
        merged_class_name = f"Merged;{self_name},{other_name}"

        # Create the new dataclass type
        merged_class = self._make_dataclass(merged_class_name, field_definitions)

        # Filter merged_fields to only include init-able fields
        merged_class_init_fields = {f.name for f in fields(merged_class) if f.init}
        init_only_fields = {
            k: v for k, v in merged_fields.items() if k in merged_class_init_fields
        }

        # Create and return instance
        return merged_class(**init_only_fields)

    def _make_dataclass(
        self, class_name: str, field_definitions: Dict[str, tuple]
    ) -> type:
        """
        Create a dataclass dynamically with proper field ordering.

        Args:
            class_name: Name for the new class
            field_definitions: Dict mapping field names to (type, default_value) tuples

        Returns:
            New dataclass type inheriting from StandardizedSpecifier
        """
        # Separate fields with and without defaults to ensure proper ordering
        # Non-default fields must come before default fields in dataclass
        non_default_fields = []
        default_fields = []

        for field_name, (field_type, field_default) in field_definitions.items():
            if field_default is None:
                non_default_fields.append((field_name, field_type))
            else:
                # Check if default is mutable (like a specifier object)
                if hasattr(field_default, "__dict__") and not isinstance(
                    field_default, (str, int, float, bool, tuple)
                ):
                    # For mutable objects, we'll handle them as non-default and set after instantiation
                    non_default_fields.append((field_name, field_type))
                else:
                    default_fields.append((field_name, field_type, field_default))

        # Create ordered annotations and defaults
        annotations = {}
        defaults = {}

        # Add non-default fields first
        for field_name, field_type in non_default_fields:
            annotations[field_name] = field_type

        # Add default fields after
        for field_name, field_type, field_default in default_fields:
            annotations[field_name] = field_type
            defaults[field_name] = field_default

        # Create class attributes
        class_attrs = {"__annotations__": annotations, **defaults}

        # Create the class
        new_class = type(class_name, (StandardizedSpecifier,), class_attrs)

        return dataclass(new_class)

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """
        Compare two values for equality, handling special cases.

        Args:
            val1, val2: Values to compare

        Returns:
            True if values are considered equal
        """
        # Handle None values
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False

        # Handle numpy arrays
        if hasattr(val1, "__array__") and hasattr(val2, "__array__"):
            try:
                return np.array_equal(val1, val2)
            except:
                return False

        # Handle tuples/lists
        if isinstance(val1, (tuple, list)) and isinstance(val2, (tuple, list)):
            if len(val1) != len(val2):
                return False
            return all(self._values_equal(v1, v2) for v1, v2 in zip(val1, val2))

        # Handle dictionaries
        if isinstance(val1, dict) and isinstance(val2, dict):
            if set(val1.keys()) != set(val2.keys()):
                return False
            return all(self._values_equal(val1[k], val2[k]) for k in val1.keys())

        # Standard equality
        try:
            return val1 == val2
        except:
            return False


##########################################
#############NAMING FOR POPULAR DATA##################
##########################################


@dataclass(frozen=True)
class _StandardNamesDataCategories(StandardNamesBaseDataTypes):
    Results: Final[BaseNameDataType] = BaseNameDataType("Res", "Results", "Results")
    Metadata: Final[BaseNameDataType] = BaseNameDataType("Met", "Metadata", "Metadata")
    LowLevelData: Final[BaseNameDataType] = BaseNameDataType(
        "LLD", "LowLevelData", "LowLevelData"
    )

    UnspecifiedCategory: Final[BaseNameDataType] = BaseNameDataType(
        "Unc", "UnspecifiedCategory", "UnspecifiedCategory"
    )


DATA_CATEGORIES: _StandardNamesDataCategories = _StandardNamesDataCategories()

if __name__ == "__main__":
    SNDC = DATA_CATEGORIES
    print(SNDC)
    print(SNDC.Results)
    print(SNDC.Metadata)
    print(SNDC.LowLevelData)
    print(SNDC.get_all_attributes())


@dataclass(frozen=True)
class _StandardNamesDataSubCategories(StandardNamesBaseDataTypes):
    SNDC = DATA_CATEGORIES
    __res_id = SNDC.Results.data_category
    __met_id = SNDC.Metadata.data_category
    __lld_id = SNDC.LowLevelData.data_category
    __un_id = SNDC.UnspecifiedCategory.data_category

    ExperimentSetMetadata = BaseNameDataType(
        "ESM", "ExperimentSetMetadata", __met_id, "ExperimentSetMetadata"
    )
    ExperimentInstanceMetadata = BaseNameDataType(
        "EIM", "ExperimentInstanceMetadata", __met_id, "ExperimentInstanceMetadata"
    )

    DatasetsInfo = BaseNameDataType("DI", "DatasetsInfo", __met_id, "DatasetsInfo")

    Overviews = BaseNameDataType("Ov", "Overviews", __res_id, "Overviews")
    ResultsRaw = BaseNameDataType("RR", "ResultsRaw", __res_id, "ResultsRaw")
    ResultsProcessed = BaseNameDataType(
        "RP", "ResultsProcessed", __res_id, "ResultsProcessed"
    )

    # Unspecified data subcategory
    UnspecifiedSubCategory = BaseNameDataType(
        "Uns", "UnspecifiedSubCategory", __un_id, "UnspecifiedSubCategory"
    )


DATA_SUBCATEGORIES: _StandardNamesDataSubCategories = _StandardNamesDataSubCategories()


@dataclass(frozen=True)
class _StandardNamesDataTypes(StandardNamesBaseDataTypes):
    SNDSC = DATA_SUBCATEGORIES

    __OV = (SNDSC.Overviews.data_category, SNDSC.Overviews.data_subcategory)
    __RR = (SNDSC.ResultsRaw.data_category, SNDSC.ResultsRaw.data_subcategory)
    __RP = (
        SNDSC.ResultsProcessed.data_category,
        SNDSC.ResultsProcessed.data_subcategory,
    )
    __EIS = (
        SNDSC.ExperimentInstanceMetadata.data_category,
        SNDSC.ExperimentInstanceMetadata.data_subcategory,
    )

    __DI = (SNDSC.DatasetsInfo.data_category, SNDSC.DatasetsInfo.data_subcategory)
    # unspecifiecd data type
    __UN = (
        SNDSC.UnspecifiedSubCategory.data_category,
        SNDSC.UnspecifiedSubCategory.data_subcategory,
    )

    # ____________ Overview data types ____________
    VariationalParameters = BaseNameDataType(
        "VAP", "VariationalParameters", __OV[0], __OV[1]
    )
    ExpectedValues = BaseNameDataType("EXV", "ExpectedValues", __OV[0], __OV[1])
    OptimizationOverview = BaseNameDataType(
        "OOV", "OptimizationOverview", __OV[0], __OV[1]
    )
    OptimizationOverviewAbstract = BaseNameDataType('OOA', 'OptimizationOverviewAbstract', __OV[0], __OV[1])

    NDAROverview = BaseNameDataType("NDO", "NDAROverview", __OV[0], __OV[1])

    # ____________ Raw results data types ____________
    Bitstrings = BaseNameDataType("BTS", "Bitstrings", __RR[0], __RR[1])
    Energies = BaseNameDataType("ENG", "Energies", __RR[0], __RR[1])
    VariableValues = BaseNameDataType("VAV", "VariableValues", __RR[0], __RR[1])

    TestResults = BaseNameDataType("000", "TestResults", __RR[0], __RR[1])

    # ____________ Processed results data types ____________
    BitstringsHistograms = BaseNameDataType(
        "BTH", "BitstringsHistograms", __RP[0], __RP[1]
    )
    EnergiesHistograms = BaseNameDataType("ENH", "EnergiesHistograms", __RP[0], __RP[1])
    Correlators = BaseNameDataType("COR", "Correlators", __RP[0], __RP[1])
    StateVectors = BaseNameDataType("STV", "StateVectors", __RP[0], __RP[1])

    # ____________ Jobs metadata types ____________
    JobMetadata = BaseNameDataType("JMD", "JobMetadata", __EIS[0], __EIS[1])

    ExperimentSetTracking = BaseNameDataType(
        "EST", "ExperimentSetTracking", __EIS[0], __EIS[1]
    )
    BackendData = BaseNameDataType("BCK", "BackendData", __EIS[0], __EIS[1])
    MiscMetadata = BaseNameDataType("MIM", "MiscMetadata", __EIS[0], __EIS[1])
    ExperimentSetSpecificationMetadata = BaseNameDataType(
        "ESM", "ExperimentSetSpecificationMetadata", __EIS[0], __EIS[1]
    )

    CostHamiltonianMetadata = BaseNameDataType(
        "CHM", "CostHamiltonianMetadata", __EIS[0], __EIS[1]
    )
    PhaseHamiltonianMetadata = BaseNameDataType(
        "PHM", "PhaseHamiltonianMetadata", __EIS[0], __EIS[1]
    )
    QAOAOptimizationMetadata = BaseNameDataType(
        "QOM", "QAOAOptimizationMetadata", __EIS[0], __EIS[1]
    )

    HamiltonianTransformations = BaseNameDataType(
        "HT", "HamiltonianTransformations", __EIS[0], __EIS[1]
    )
    QuantumCircuits = BaseNameDataType("QC", "QuantumCircuits", __EIS[0], __EIS[1])
    CircuitsMetadata = BaseNameDataType("CIM", "CircuitsMetadata", __EIS[0], __EIS[1])

    # ____________ Misc data types info ____________
    Unspecified = BaseNameDataType("UNS", "UnspecifiedDataType", __UN[0], __UN[1])
    DataType = BaseNameDataType("DAT", "DataType", __DI[0], __DI[1])


STANDARD_NAMES_DATA_TYPES: _StandardNamesDataTypes = _StandardNamesDataTypes()


@dataclass(frozen=True)
class _StandardNamesDataTypesExperimentSets(_StandardNamesDataTypes):
    """Experiment set level versions of standard data types.

    This class automatically converts all metadata-category data types to use
    experiment set metadata categories instead of instance metadata.
    Only affects data types where data_category == 'Metadata'.
    """

    def __post_init__(self):
        """Override metadata types to use ExperimentSetMetadata subcategory."""
        super().__post_init__()

        # Get the target subcategory for experiment set metadata
        esm_category = DATA_SUBCATEGORIES.ExperimentSetMetadata.data_category
        esm_subcategory = DATA_SUBCATEGORIES.ExperimentSetMetadata.data_subcategory
        metadata_category = DATA_CATEGORIES.Metadata.data_category

        # Override only metadata types
        for attr_name in dir(self):
            if not attr_name.startswith("_") and not callable(
                getattr(self, attr_name, None)
            ):
                try:
                    attr = getattr(self, attr_name)
                    if isinstance(attr, BaseNameDataType):
                        # Only override if it's a metadata type
                        if attr.data_category == metadata_category:
                            # Create new BaseNameDataType with experiment set metadata
                            new_attr = BaseNameDataType(
                                id=attr.id,
                                id_long=attr.id_long,
                                data_category=esm_category,
                                data_subcategory=esm_subcategory,
                                value_format=attr.value_format,
                            )
                            object.__setattr__(self, attr_name, new_attr)
                except (AttributeError, TypeError):
                    continue


STANDARD_NAMES_DATA_TYPES_EXPERIMENT_SETS: _StandardNamesDataTypesExperimentSets = (
    _StandardNamesDataTypesExperimentSets()
)


@dataclass(frozen=True)
class _StandardNamesGeneralVariables(StandardNamesBaseVariables):
    # ____________ General Variables ____________
    UUID: BaseName = BaseName("UID", "UUID", str)
    ID: BaseName = BaseName("ID", "Identifier", str)

    NumberOfQubits: BaseName = BaseName("NoQ", "NumberOfQubits", int)
    Repetition: BaseName = BaseName("REP", "Repetition", int)
    Index: BaseName = BaseName("IND", "Index", int)
    Timestamp: BaseName = BaseName("TIS", "Timestamp", str)
    Runtime: BaseName = BaseName("RNT", "Runtime", float)
    IterationIndex: BaseName = BaseName("ITI", "IterationIndex", int)
    Seed = BaseName("SEE", "RNGSeed", int)
    TimeStepSize = BaseName("TSS", "TimeStepSize", float)

    SessionId = BaseName("SID", "SessionId", str)
    JobId = BaseName("JID", "JobId", str)

    ExperimentSetID = BaseName("ESI", "ExperimentSetID", str)
    ExperimentInstanceID = BaseName("EII", "ExperimentInstanceID", str)

    ExperimentSetName = BaseName("ESN", "ExperimentSetName", str)
    ExperimentInstanceName = BaseName("EIN", "ExperimentInstanceName", str)

    CircuitLabel = BaseName("CIL", "CircuitLabel", str)

    CircuitIndex = BaseName("CIN", "CircuitIndex", int)

    Backend = BaseName("BCK", "Backend", str)
    Simulated = BaseName("SIM", "Simulated", bool)

    Bitstring: BaseName = BaseName("BTS", "Bitstring", np.ndarray)
    Energy: BaseName = BaseName("ENG", "Energy", float)
    ThresholdValue: BaseName = BaseName("TRV", "ThresholdValue", float)

    Count: BaseName = BaseName("CNT", "Count", int)

    QubitIndices = BaseName("QI", "QubitIndices", List[int])

    CoefficientsType: BaseName = BaseName("CT", "CoefficientsType", type)
    CoefficientsDistributionName: BaseName = BaseName(
        "CDN", "CoefficientsDistributionName", str
    )
    CoefficientsDistributionProperties: BaseName = BaseName(
        "CDP", "CoefficientsDistributionProperties", dict
    )

    # ____________ General Optimization Variables ____________
    NumberOfTrials: BaseName = BaseName("NoT", "NumberOfTrials", int)
    NumberOfSamples: BaseName = BaseName("NoS", "NumberOfSamples", int)
    CostHamiltonianClass: BaseName = BaseName("CHC", "CostHamiltonianClass", str)
    CostHamiltonianInstance: BaseName = BaseName("CHI", "CostHamiltonianInstance", int)

    TrialIndex: BaseName = BaseName("TI", "TrialIndex", int)

    EnergyMean: BaseName = BaseName("ENM", "EnergyMean", float)
    EnergySTD: BaseName = BaseName("ENS", "EnergySTD", float)
    EnergyBest: BaseName = BaseName("ENB", "EnergyBest", float)
    BitstringBest: BaseName = BaseName("BTB", "BitstringBest", str)

    ApproximationRatioMean: BaseName = BaseName("ARM", "ApproximationRatioMean", float)
    ApproximationRatioSTD: BaseName = BaseName("ARS", "ApproximationRatioSTD", float)
    ApproximationRatioBest: BaseName = BaseName("ARB", "ApproximationRatioBest", float)
    ApproximationRatio: BaseName = BaseName("AR", "ApproximationRatio", float)

    HamiltonianTransformationType: BaseName = BaseName(
        "HTT", "HamiltonianTransformationType", str
    )
    HamiltonianTransformationValue: BaseName = BaseName(
        "HTV", "HamiltonianTransformationValue", float
    )

    Bitflip = BaseName("BFL", "Bitflip", bool)
    Permutation = BaseName("PER", "Permutation", bool)

    # ____________ Quantum Optimization Variables ____________
    AlgorithmicDepth: BaseName = BaseName("p", "AlgorithmicDepth", int)
    AnsatzSpecifier: BaseName = BaseName("AS", "AnsatzSpecifier", str)

    Angles: BaseName = BaseName("Ang", "Angles", np.ndarray)
    HamiltonianRepresentationIndex: BaseName = BaseName(
        "HRI", "HamiltonianRepresentationIndex", int
    )

    # ____________ QAOA Variables ____________

    QubitMappingType: BaseName = BaseName("SNT", "QubitMappingType", str)
    PhaseSeparatorType: BaseName = BaseName("DT", "PhaseSeparatorType", str)
    MixerType: BaseName = BaseName("MT", "MixerType", str)
    TimeBlockSize: BaseName = BaseName("k", "TimeBlockSize", int)

    PhaseHamiltonianClass: BaseName = BaseName("DHC", "PhaseHamiltonianClass", str)
    PhaseHamiltonianInstance: BaseName = BaseName(
        "DHI", "PhaseHamiltonianInstance", int
    )

    # ____________ NDAR Variables ____________

    NDARIteration: BaseName = BaseName("NDI", "NDARIteration", int)

    ConvergenceCriterion: BaseName = BaseName("COC", "ConvergenceCriterion", str)
    ConvergenceValue: BaseName = BaseName("COV", "ConvergenceValue", float)
    MaximalIterations: BaseName = BaseName("MIT", "MaximalIterations", int)

    AttractorModel: BaseName = BaseName("ATM", "AttractorModel", str)
    AttractorStateType: BaseName = BaseName("AST", "AttractorStateType")
    AttractorState: BaseName = BaseName("ATS", "AttractorState", str)

    TrialChoiceRule: BaseName = BaseName("TCR", "TrialChoiceRule")
    BitstringChoiceRule: BaseName = BaseName("BCR", "BitstringChoiceRule")
    PropertyFunctionRule: BaseName = BaseName("PFR", "PropertyFunctionRule")

    # ____________ Simulation Variables ____________
    Noiseless: BaseName = BaseName("NLS", "Noiseless", bool)
    Noisy: BaseName = BaseName("NOY", "Noisy", bool)

    # Monte Carlo variables
    Temperature: BaseName = BaseName("TMP", "Temperature", float)
    InverseTemperature: BaseName = BaseName("ITP", "InverseTemperature", float)
    FlipProbability = BaseName("FPR", "FlipProbability", float)
    ReplicaIndex: BaseName = BaseName("RI", "ReplicaIndex", int)

    # ____________ test variables ____________
    # Below should not be used in production, only for testing purposes
    # this should return "duplicate id" error
    # #this should return "duplicate id_long" error


STANDARD_NAMES_VARIABLES: _StandardNamesGeneralVariables = (
    _StandardNamesGeneralVariables()
)

ALL_IDS_VARIABLES_LONG = [
    x.id_long for x in STANDARD_NAMES_VARIABLES.get_all_attributes().values()
]
ALL_IDS_VARIABLES_SHORT = [
    x.id for x in STANDARD_NAMES_VARIABLES.get_all_attributes().values()
]

if __name__ == "__main__":
    VA = STANDARD_NAMES_VARIABLES
    all_attrs = VA.get_all_attributes()
    for key, value in all_attrs.items():
        print(f"{key}: {value}")


##########################################
#############NAMING FOR HAMILTONIANS##################
##########################################


@dataclass(frozen=True)
class _StandardNamesHamiltonianModels(StandardNamesBaseVariables):
    ErdosRenyi = BaseName("ER", "ErdosRenyi")

    SherringtonKirkpatrick = BaseName("SK", "SherringtonKirkpatrick")

    MaxCut = BaseName("MC", "MaxCut")

    MAX2SAT = BaseName("M2S", "MAX2SAT")

    # TODO(FBM): add
    MAXkSAT = BaseName("MKS", "MAXkSAT")

    RegularGraph = BaseName("RG", "RegularGraph")
    WishartPlantedEnsemble = BaseName("WPE", "WishartPlantedEnsemble")

    # TODO FBM: Maximum Independent Set

    Unspecified = BaseName("UNS", "Unspecified")

    # TODO(FBM):
    # Add reading standardized databases

    # TODO(FBM):
    # add Karloff graph

    LABS = BaseName("LAB", "LowAutocorrelationBinarySequences")


HamiltonianModels = _StandardNamesHamiltonianModels()


@dataclass(frozen=True)
class _StandardNamesErdosRenyiTypes(StandardNamesBaseVariables):
    GnM = BaseName("GnM", "GnM", str)
    Gnp = BaseName("GnP", "Gnp", str)


ERDOS_RENYI_TYPES = _StandardNamesErdosRenyiTypes()


@dataclass(frozen=True)
class _StandardNamesHamiltonianDescriptions(StandardNamesBaseVariables):
    HamiltonianModelName = BaseName("HMN", "HamiltonianModelName", str)
    Localities = BaseName("LOC", "Localities", List[int])
    HamiltonianInstanceIndex = BaseName("HII", "HamiltonianInstanceIndex", int)

    CoefficientsDistributionSpecifier = BaseName(
        "CFD", "CoefficientsDistributionSpecifier", str
    )
    HamiltonianDescription = BaseName("HAD", "HamiltonianDescription", str)
    AverageDegree = BaseName("AVD", "AverageDegree", float)

    ClassSpecificAttributes = BaseName("CSA", "ClassSpecificAttributes", dict)
    ClassInstanceSpecificAttributes = BaseName(
        "CIS", "ClassInstanceSpecificAttributes", dict
    )

    ErdosRenyiType = BaseName("ERT", "ErdosRenyiType", str)
    EdgeProbabilityOrAmount = BaseName("EPA", "EdgeProbabilityOrAmount", float)

    kSAT = BaseName("SAT", "kSAT", int)
    ClauseDensity = BaseName("CLD", "ClauseDensity", float)

    WishartDensity = BaseName("WID", "WishartDensity", float)


STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS = _StandardNamesHamiltonianDescriptions()


@dataclass(frozen=True)
class _StandardNamesCoefficientsType(StandardNamesBaseVariables):
    CONTINUOUS = BaseName("CON", "Continuous", str)
    DISCRETE = BaseName("DIS", "Discrete", str)
    CONSTANT = BaseName("FIX", "Fixed", str)


CoefficientsType = _StandardNamesCoefficientsType()


@dataclass(frozen=True)
class _StandardNamesCoefficientsDistribution(StandardNamesBaseVariables):
    Normal = BaseName("NOR", "Normal", str)
    Uniform = BaseName("UNI", "Uniform", str)
    Constant = BaseName("CON", "Constant", str)
    Custom = BaseName("CUS", "Custom", str)


CoefficientsDistribution = _StandardNamesCoefficientsDistribution()

if __name__ == "__main__":
    print("type check:", CoefficientsType.DISCRETE in [CoefficientsType.DISCRETE])


@dataclass
class CoefficientsDistributionSpecifier(StandardizedSpecifier):
    CoefficientsType: BaseName
    CoefficientsDistributionName: BaseName
    CoefficientsDistributionProperties: Optional[dict] = field(default=None)

    def __post_init__(self):
        """Set default properties based on type and distribution if none provided."""
        if self.CoefficientsDistributionProperties is None:
            if self.CoefficientsType == CoefficientsType.CONSTANT:
                raise ValueError(
                    "Constant coefficients must have 'value' property specified"
                )
            elif self.CoefficientsType in [CoefficientsType.CONTINUOUS]:
                if self.CoefficientsDistributionName in [
                    CoefficientsDistribution.Uniform
                ]:
                    default_props = {"low": -1, "high": 1}
                elif self.CoefficientsDistributionName in [
                    CoefficientsDistribution.Normal
                ]:
                    default_props = {"loc": 0, "scale": 1}
                else:
                    default_props = {}
                self.CoefficientsDistributionProperties = default_props
            elif self.CoefficientsType in [CoefficientsType.DISCRETE]:
                if self.CoefficientsDistributionName in [
                    CoefficientsDistribution.Uniform
                ]:
                    default_props = {"values": [-1, 1]}
                elif self.CoefficientsDistributionName in [
                    CoefficientsDistribution.Normal
                ]:
                    default_props = {"loc": 0, "scale": 1}
                else:
                    default_props = {}
                self.CoefficientsDistributionProperties = default_props

        super().__post_init__()


@dataclass
class HamiltonianInstanceSpecifierGeneral(StandardizedSpecifier):
    NumberOfQubits: int
    HamiltonianInstanceIndex: int

    def get_description_string(self, long_strings=False):
        return self._get_description_string_general()


@dataclass
class HamiltonianInstanceSpecifierErdosRenyi(HamiltonianInstanceSpecifierGeneral):
    EdgeProbabilityOrAmount: Union[int, float]


@dataclass
class HamiltonianInstanceSpecifierRegular(HamiltonianInstanceSpecifierGeneral):
    AverageDegree: float


@dataclass
class HamiltonianInstanceSpecifierMAXkSAT(HamiltonianInstanceSpecifierGeneral):
    ClauseDensity: float


@dataclass
class HamiltonianInstanceSpecifierLABS(HamiltonianInstanceSpecifierGeneral):
    def __post_init__(self):
        # LABS always has instance index 0
        super().__post_init__()
        self.HamiltonianInstanceIndex = 0


@dataclass
class HamiltonianInstanceSpecifierWishartPlantedEnsemble(
    HamiltonianInstanceSpecifierGeneral
):
    WishartDensity: float


HamiltonianInstanceSpecifierMaxCut = HamiltonianInstanceSpecifierErdosRenyi
HamiltonianInstanceSpecifierSK = HamiltonianInstanceSpecifierGeneral


@dataclass
class HamiltonianClassSpecifierGeneral(StandardizedSpecifier):
    HamiltonianModelName: BaseName
    Localities: Tuple[int, ...]
    CoefficientsDistributionSpecifier: Optional[CoefficientsDistributionSpecifier] = (
        None
    )
    ClassSpecificAttributes: Optional[Dict[str, Any]] = field(default=None, init=False)

    # Class-level attribute - each subclass overrides this
    instance_specifier_constructor: ClassVar[Callable] = (
        HamiltonianInstanceSpecifierGeneral
    )

    def __post_init__(self):
        """Validate localities are positive."""
        assert np.all(np.array(self.Localities) > 0), "Locality must be positive."
        # Convert to tuple to ensure immutability
        self.Localities = tuple(self.Localities)
        super().__post_init__()

    def get_description_string(self, long_strings=False):
        return self._get_description_string_general(long_strings=long_strings)


@dataclass
class HamiltonianClassSpecifierErdosRenyi(HamiltonianClassSpecifierGeneral):
    # Class-level attribute override
    HamiltonianModelName: BaseName = field(
        default=HamiltonianModels.ErdosRenyi, init=False
    )
    instance_specifier_constructor: ClassVar[Callable] = field(
        default=HamiltonianInstanceSpecifierErdosRenyi, init=False
    )

    # User-configurable
    ErdosRenyiType: BaseName = ERDOS_RENYI_TYPES.Gnp

    # Computed from class-specific fields - not user-settable
    ClassSpecificAttributes: Optional[Dict[str, Any]] = field(default=None, init=False)

    def __post_init__(self):
        # Set default values if not provided
        if self.CoefficientsDistributionSpecifier is None:
            default_coeffs = CoefficientsDistributionSpecifier(
                CoefficientsType=CoefficientsType.DISCRETE,
                CoefficientsDistributionName=CoefficientsDistribution.Uniform,
                CoefficientsDistributionProperties={"values": [-1, 1]},
            )
            self.CoefficientsDistributionSpecifier = default_coeffs

        # Set class-specific attributes
        if self.ClassSpecificAttributes is None:
            class_specific_attributes = {
                STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS.ErdosRenyiType.id: self.ErdosRenyiType
            }
            self.ClassSpecificAttributes = class_specific_attributes

        super().__post_init__()


@dataclass
class HamiltonianClassSpecifierMaxCut(HamiltonianClassSpecifierErdosRenyi):
    # Fixed values - not user-settable
    HamiltonianModelName: BaseName = field(default=HamiltonianModels.MaxCut, init=False)
    Localities: Tuple[int, ...] = field(default=(2,), init=False)  # MaxCut is 2-local
    instance_specifier_constructor: ClassVar[Callable] = field(
        default=HamiltonianInstanceSpecifierMaxCut, init=False
    )

    def __post_init__(self):
        # Set default coefficients if not provided
        if self.CoefficientsDistributionSpecifier is None:
            default_coeffs = CoefficientsDistributionSpecifier(
                CoefficientsType=CoefficientsType.CONSTANT,
                CoefficientsDistributionName=CoefficientsDistribution.Constant,
                CoefficientsDistributionProperties={"value": 1},
            )
            self.CoefficientsDistributionSpecifier = default_coeffs

        super().__post_init__()


@dataclass
class HamiltonianClassSpecifierSK(HamiltonianClassSpecifierGeneral):
    # Class-level attribute override
    HamiltonianModelName: BaseName = field(
        default=HamiltonianModels.SherringtonKirkpatrick, init=False
    )
    instance_specifier_constructor: ClassVar[Callable] = field(
        default=HamiltonianInstanceSpecifierSK, init=False
    )

    def __post_init__(self):
        # Validate localities for SK model
        assert (
            max(self.Localities) == 2
        ), "Sherrington-Kirkpatrick model is only defined for 2-local interactions"

        # Set default coefficients if not provided
        if self.CoefficientsDistributionSpecifier is None:
            default_coeffs = CoefficientsDistributionSpecifier(
                CoefficientsType=CoefficientsType.DISCRETE,
                CoefficientsDistributionName=CoefficientsDistribution.Uniform,
                CoefficientsDistributionProperties={"values": [-1, 1]},
            )
            self.CoefficientsDistributionSpecifier = default_coeffs

        # Validate symmetric coefficients for SK model
        distribution_properties = (
            self.CoefficientsDistributionSpecifier.CoefficientsDistributionProperties
        )
        if "values" in distribution_properties:
            min_val = min(distribution_properties["values"])
            max_val = max(distribution_properties["values"])
        elif "low" in distribution_properties and "high" in distribution_properties:
            min_val = distribution_properties["low"]
            max_val = distribution_properties["high"]
        elif "value" in distribution_properties:
            min_val = distribution_properties["value"]
            max_val = distribution_properties["value"]
        else:
            raise ValueError("Invalid distribution properties")
        assert abs(min_val) == abs(
            max_val
        ), f"SK model requires symmetric coefficients, now we have: {min_val}, {max_val}"

        super().__post_init__()


@dataclass
class HamiltonianClassSpecifierRegular(HamiltonianClassSpecifierGeneral):
    # Fixed values - not user-settable
    Localities: Tuple[int, ...] = field(
        default=(2,), init=True
    )  # Regular graphs are 2-local
    HamiltonianModelName: BaseName = field(
        default=HamiltonianModels.RegularGraph, init=False
    )
    instance_specifier_constructor: ClassVar[Callable] = field(
        default=HamiltonianInstanceSpecifierRegular, init=False
    )

    def __post_init__(self):
        # Validate localities for Regular graph model
        assert (
            max(self.Localities) == 2
        ), "Regular graph model is only defined for 2-local interactions"

        # Set default coefficients if not provided
        if self.CoefficientsDistributionSpecifier is None:
            default_coeffs = CoefficientsDistributionSpecifier(
                CoefficientsType=CoefficientsType.CONSTANT,
                CoefficientsDistributionName=CoefficientsDistribution.Constant,
                CoefficientsDistributionProperties={"value": 1},
            )
            self.CoefficientsDistributionSpecifier = default_coeffs

        super().__post_init__()


@dataclass
class HamiltonianClassSpecifierMAXkSAT(HamiltonianClassSpecifierGeneral):
    # Computed from class-specific fields - not user-settable
    HamiltonianModelName: BaseName = field(
        default=HamiltonianModels.MAXkSAT, init=False
    )
    Localities: Tuple[int, ...] = field(default=None, init=False)  # Computed from kSAT
    ClassSpecificAttributes: Optional[Dict[str, Any]] = field(
        default=None, init=False
    )  # Computed from kSAT
    instance_specifier_constructor: ClassVar[Callable] = field(
        default=HamiltonianInstanceSpecifierMAXkSAT, init=False
    )

    # User-configurable
    kSAT: int = 2  # Default to 2-SAT since only 2-SAT is implemented

    def __post_init__(self):
        # Validate kSAT
        assert self.kSAT > 0, "k-SAT must be positive"
        if self.kSAT != 2:
            raise NotImplementedError("Only 2-SAT is implemented")

        # Set localities based on kSAT
        localities = tuple(range(1, self.kSAT + 1))
        self.Localities = localities

        # Set class-specific attributes
        if self.ClassSpecificAttributes is None:
            class_specific_attributes = {
                STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS.kSAT.id: self.kSAT
            }
            self.ClassSpecificAttributes = class_specific_attributes

        super().__post_init__()


@dataclass
class HamiltonianClassSpecifierWishartPlantedEnsemble(HamiltonianClassSpecifierGeneral):
    # Fixed values - not user-settable
    Localities: Tuple[int, ...] = field(default=(2,), init=False)  # Wishart is 2-local
    HamiltonianModelName: BaseName = field(
        default=HamiltonianModels.WishartPlantedEnsemble, init=False
    )
    instance_specifier_constructor: ClassVar[Callable] = field(
        default=HamiltonianInstanceSpecifierWishartPlantedEnsemble, init=False
    )


@dataclass
class HamiltonianClassSpecifierLABS(HamiltonianClassSpecifierGeneral):
    # Fixed values - not user-settable
    Localities: Tuple[int, ...] = field(
        default=(2, 4), init=False
    )  # LABS has 2-local and 4-local terms
    HamiltonianModelName: BaseName = field(default=HamiltonianModels.LABS, init=False)

    # Class-level attribute override
    instance_specifier_constructor: ClassVar[Callable] = field(
        default=HamiltonianInstanceSpecifierLABS, init=False
    )


if __name__ == "__main__":
    example_coefficients_distribution = CoefficientsDistributionSpecifier(
        CoefficientsType=CoefficientsType.DISCRETE,
        CoefficientsDistributionName=CoefficientsDistribution.Uniform,
        CoefficientsDistributionProperties={"values": [-1, 1]},
    )
    example_class_specifier = HamiltonianClassSpecifierErdosRenyi(
        Localities=(1, 2),
        CoefficientsDistributionSpecifier=example_coefficients_distribution,
    )
    print(example_class_specifier.get_description_string())

    example_instance_specifier = HamiltonianInstanceSpecifierGeneral(
        NumberOfQubits=10, HamiltonianInstanceIndex=0
    )


##########################################
#############QUANTUM OPTIMIZATION##################
##########################################


@dataclass
class HamiltonianOptimizationSpecifier(StandardizedSpecifier):
    NumberOfQubits: int = field(init=False)
    HamiltonianInstanceIndex: int = field(init=False)
    CostHamiltonianClass: HamiltonianClassSpecifierGeneral = field(init=True)
    CostHamiltonianInstance: HamiltonianInstanceSpecifierGeneral = field(init=True)

    def __post_init__(self):
        # Extract computed fields from the input specifiers
        number_of_qubits = self.CostHamiltonianInstance.NumberOfQubits
        hamiltonian_instance_cost = (
            self.CostHamiltonianInstance.HamiltonianInstanceIndex
        )
        self.NumberOfQubits = number_of_qubits
        self.HamiltonianInstanceIndex = hamiltonian_instance_cost

        super().__post_init__()

    def _get_dataframe_annotation(self, long_names=True) -> dict:

        SNV = STANDARD_NAMES_VARIABLES
        SNHD = STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS

        if long_names:
            df_annotation = {
                SNV.NumberOfQubits.id_long: self.NumberOfQubits,
                SNHD.HamiltonianInstanceIndex.id_long: self.HamiltonianInstanceIndex,
                SNV.CostHamiltonianClass.id_long: self.CostHamiltonianClass.get_description_string(),
                SNV.CostHamiltonianInstance.id_long: self.CostHamiltonianInstance.get_description_string(),
            }
        else:
            df_annotation = {
                SNV.NumberOfQubits.id: self.NumberOfQubits,
                SNHD.HamiltonianInstanceIndex.id: self.HamiltonianInstanceIndex,
                SNV.CostHamiltonianClass.id: self.CostHamiltonianClass.get_description_string(),
                SNV.CostHamiltonianInstance.id: self.CostHamiltonianInstance.get_description_string(),
            }

        return df_annotation
