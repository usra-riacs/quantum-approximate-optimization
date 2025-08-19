# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

import pandas as pd

from quapopt.data_analysis.data_handling.standard_names.data_hierarchy import (MAIN_KEY_VALUE_SEPARATOR,
                                                             MAIN_KEY_SEPARATOR,
                                                             SUB_KEY_SEPARATOR,
                                                             BaseName)
from quapopt.data_analysis.data_handling.standard_names import (STANDARD_NAMES_VARIABLES,
                                                                STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS,
                                                                STANDARD_NAMES_DATA_TYPES)


def search_for_standard_variable_value_in_string(string: str,
                                                 base_name: BaseName):
    split_main = string.split(MAIN_KEY_SEPARATOR)

    id_short,id_long = base_name.id, base_name.id_long

    for substring in split_main:
        split_sub = substring.split(MAIN_KEY_VALUE_SEPARATOR)
        if len(split_sub) != 2:
            continue
        variable_name = split_sub[0]
        if variable_name == id_short or variable_name == id_long:
            variable_value_sub = split_sub[1].split(SUB_KEY_SEPARATOR)

            if len(variable_value_sub)==1:
                variable_value = split_sub[1]
                try:
                    proper_value = base_name.value_format(variable_value)
                except Exception as e:
                    proper_value = variable_value

                return proper_value
            #In other case, we go through SUB_KEY_SEPARATOR
            for sub_key, sub_value in variable_value_sub:
                if sub_key == id_short or sub_key == id_long:
                    try:
                        proper_value = base_name.value_format(sub_value)
                    except Exception as e:
                        proper_value = sub_value
                    return proper_value

    return None

def decode_standard_variables_from_string(string: str):
    # First we check the "variables names"
    # Then we check the "data names"
    # Let's split the string by the separator
    # We will first check the variables
    split_main = string.split(MAIN_KEY_SEPARATOR)
    undetected = []
    detected_dict = {}
    for substring in split_main:
        split_sub = substring.split(MAIN_KEY_VALUE_SEPARATOR)
        #print(substring,split_sub)
        if len(split_sub) != 2:
            #print("WHOOPS")
            undetected.append(substring)
            continue
        variable_name = split_sub[0]
        variable_value = split_sub[1]

        detected_base_name = STANDARD_NAMES_VARIABLES.detect_base_name(variable_name)
        if detected_base_name is None:
            detected_base_name = STANDARD_NAMES_HAMILTONIAN_DESCRIPTIONS.detect_base_name(variable_name)
            if detected_base_name is None:
                detected_base_name = STANDARD_NAMES_DATA_TYPES.detect_base_name(variable_name)

        if detected_base_name is None:
            undetected.append(substring)
            continue

        try:
            proper_value = detected_base_name.value_format(variable_value)
        except Exception as e:
            #print(e)
            proper_value = variable_value

        detected_dict[detected_base_name.id_long] = proper_value

    detected_dict['Unrecognized'] = undetected
    return detected_dict




def unroll_properties_from_string_description_df(df:pd.DataFrame,
                                                 column_name:str,
                                                 inplace=True):

    if not inplace:
        df = df.copy()

    example_value = df[column_name].iloc[0]
    print(example_value)
    detected_dict = decode_standard_variables_from_string(example_value)
    print(detected_dict)
    raise KeyboardInterrupt

    for key, value in detected_dict.items():
        if key!='Unrecognized':
            df[key] = df[column_name].apply(lambda x: search_for_standard_variable_value_in_string(string=x,
                                                                                                   base_name=value))
    return df








