# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import copy
import time
import uuid
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


def create_random_uuid() -> str:
    return "".join(str(uuid.uuid4()).split("-"))


def get_current_date_time() -> str:
    """
    Returns the current date and time as a string in the format YYYY-MM-DD_HH-MM-SS.
    """
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def get_current_date() -> str:
    """
    Returns the current date as a string in the format YYYY-MM-DD.
    """
    return time.strftime("%Y-%m-%d", time.localtime())


def get_current_time() -> str:
    """
    Returns the current time as a string in the format HH-MM-SS.
    """
    return time.strftime("%H-%M-%S", time.localtime())


def convert_int_to_binary_tuple(
    integer: int,
    number_of_bits: Optional[int] = None,
    return_as_list: Optional[bool] = False,
) -> Tuple[int, ...]:
    """
    This function takes an integer and returns its binary representation as a tuple.
    :param integer:
    :param number_of_bits:
    :param return_as_list:
    :return:
    """

    tuple_representation = tuple(
        int((integer >> i) & 1) for i in range(number_of_bits - 1, -1, -1)
    )

    if not return_as_list:
        tuple_representation = tuple(tuple_representation)

    return tuple_representation


def convert_binary_tuple_to_integer(
    binary_tuple: Union[Tuple[int, ...], List[int]],
) -> int:
    """
    This function takes a binary tuple and returns its integer representation.
    :param binary_tuple:
    :return:
    """
    return sum([x << i for i, x in enumerate(reversed(binary_tuple))])


def convert_binary_string_to_integer(binary_string: str) -> int:
    """
    This function takes a binary string and returns its integer representation.
    :param binary_string:
    :return:
    """
    return int(binary_string, 2)


def convert_binary_string_to_tuple(binary_string: str) -> Tuple[int, ...]:
    """
    :param binary_string: e.g., '00110'
    :return: (0, 0, 1, 1, 0)
    """
    return tuple(map(int, binary_string))


# By far this is the fastest method to do that
def eval_string_tuple_to_tuple(s: str) -> Tuple[int, ...]:
    return tuple(map(int, s[1:-1].split(",")))


def eval_string_tuple_to_tuple_float(s: str) -> Tuple[float, ...]:
    return tuple(map(float, s[1:-1].split(",")))


def eval_string_float_tuple_to_int_tuple(s: str) -> Tuple[int, ...]:
    return tuple(map(lambda x: int(float(x)), s[1:-1].split(",")))


def concatenate_permutations(
    permutations: List[Tuple[int, ...]],
):
    """
    This function takes a list of permutations and concatenates them.
    :param permutations:
    :return:
    """

    combined = list(range(np.max(permutations[0]) + 1))
    for perm in permutations:
        if len(list(set(perm))) != len(perm):
            raise ValueError(f"Permutation is invalid: {perm}")
        if set(combined) != set(perm):
            raise ValueError(f"Permutation is invalid: {perm}")
        combined = [perm[pi] for pi in combined]
    return tuple(combined)


def reverse_permutation(permutation: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    This function takes a permutation and returns its reverse.

    :param permutation:
    :return:
    """
    return tuple(permutation.index(i) for i in range(len(permutation)))


def apply_permutation_to_array(array: np.ndarray, permutation: Tuple[int, ...]):
    if permutation is None:
        return array
    # WARGNING: If we use the same convention as in rest of the repo, we should first REVERSE the permutation.
    permutation_rev = reverse_permutation(permutation=permutation)

    return array[:, permutation_rev]


try:
    # Colorama for colorful printing -- that's luxiourious
    from colorama import Fore, Style

    def cool_print(
        colored_string: str,
        stuff_to_print_without_color: Optional[Any] = None,
        color_name: Optional[str] = "cyan",
        print_floors=False,
    ) -> None:
        """

        :param colored_string:  is printed with color
        :param stuff_to_print_without_color: is printed without color
        :param color_name:
        :return:
        """
        color = Fore.__dict__[color_name.upper()] + Style.BRIGHT
        if isinstance(stuff_to_print_without_color, str):
            stuff_to_really_print_without_color = stuff_to_print_without_color
        else:
            stuff_to_really_print_without_color = repr(stuff_to_print_without_color)

        if print_floors:
            print("_________________________")

        if stuff_to_print_without_color is None:
            print(color + str(colored_string) + Style.RESET_ALL)
        else:
            print(
                color + str(colored_string),
                Style.RESET_ALL + stuff_to_really_print_without_color,
            )
        if print_floors:
            print("_________________________")

except ImportError:

    def cool_print(
        colored_string: str,
        stuff_to_print_without_color: Optional[Any] = None,
        color_name: Optional[str] = "cyan",
        print_floors=False,
    ) -> None:
        """

        :param colored_string:  is printed with color
        :param stuff_to_print_without_color: is printed without color
        :param color_name:
        :return:
        """

        if isinstance(stuff_to_print_without_color, str):
            stuff_to_really_print_without_color = stuff_to_print_without_color
        else:
            stuff_to_really_print_without_color = repr(stuff_to_print_without_color)

        if print_floors:
            print("_________________________")

        if stuff_to_print_without_color is None:
            print(str(colored_string))
        else:
            print(str(colored_string), stuff_to_really_print_without_color)
        if print_floors:
            print("_________________________")


def _embed_1q_operator(
    number_of_qubits: int, local_operator: np.ndarray, global_qubit_index: int
):
    if global_qubit_index == 0:
        embed_operator = np.kron(
            local_operator, np.eye(int(2 ** (number_of_qubits - 1)))
        )
        return embed_operator
    else:
        first_eye = np.eye(2 ** (global_qubit_index))
        second_eye = np.eye(2 ** (number_of_qubits - global_qubit_index - 1))

        embeded_operator = np.kron(np.kron(first_eye, local_operator), second_eye)

        return embeded_operator


def embed_operator_in_bigger_hilbert_space(
    number_of_qubits: int,
    local_operator: np.ndarray,
    global_indices: Optional[Union[List[int], Tuple[int]]] = [0, 1],
    vector=False,
):
    import qutip

    N_small = int(np.log2(local_operator.shape[0]))

    if N_small == 1:
        return _embed_1q_operator(number_of_qubits, local_operator, global_indices[0])
    if vector:
        raise NotImplementedError("Vector not implemented yet")
        qutip_object = qutip.Qobj(
            local_operator,
            dims=[[2 for _ in range(N_small)], [1 for _ in range(N_small)]],
        )
    else:
        qutip_object = qutip.Qobj(
            local_operator,
            dims=[[2 for _ in range(N_small)], [2 for _ in range(N_small)]],
        )

    return qutip.core.expand_operator(
        oper=qutip_object,
        dims=[2 for _ in range(number_of_qubits)],
        targets=global_indices,
    ).full()


def get_permutation_operator(
    permutation: Union[Tuple[int, ...], List[int]], dtype=np.float32
):

    import qutip
    from sympy.combinatorics.permutations import Permutation

    perm_rev = Permutation(permutation)
    perm_rev_transpositions = perm_rev.transpositions()
    swap_01 = qutip.swap(2, 2)

    number_of_qubits = len(permutation)

    full_permutation = np.eye(2**number_of_qubits, dtype=dtype)
    for transposition in perm_rev_transpositions:
        swap_ij = embed_operator_in_bigger_hilbert_space(
            number_of_qubits=number_of_qubits,
            local_operator=swap_01.full(),
            global_indices=transposition,
        )

        full_permutation = swap_ij @ full_permutation
    return full_permutation


def transform_histogram_to_bitstrings_array(
    bitstrings_array_histogram: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    This function transforms the histogram of bitstrings to an array of bitstrings.
    :param bitstrings_array_histogram:
    :return:
    """
    # TODO(FBM): this is not efficient for large objects
    all_arrays = []

    for bitstrings_array, counts_array in bitstrings_array_histogram:
        all_arrays.append(
            np.repeat(
                bitstrings_array.astype(np.int32), counts_array.astype(int), axis=0
            )
        )

    return np.array(all_arrays, dtype=np.int32)


def query_yes_no(question: str) -> bool:
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    The "answer" return value is True for "yes" or False for "no".

    The function will continue to ask until a valid answer is given.
    """
    _yes_answers = {"yes", "y", "ye", "tak", "sure", "of course", "Yes", "yeah"}
    _no_answers = {"no", "n", "nope", "nah", "nie", "noo", "nooo", "noooo", "No"}

    _existential_answers = {
        "I am never sure about anything",
        "What is certain in this world?",
    }

    cool_print(question, "[y/n]", "red")
    prompt = f"{question}; [y/n]"

    choice = input(prompt).lower()
    if choice in _yes_answers:
        cool_print("ANSWER:", choice, "green")
        return True
    elif choice in _no_answers:
        cool_print("ANSWER:", choice, "red")
        return False
    else:
        cool_print("ANSWER:", choice, "blue")
        if choice in _existential_answers:
            cool_print("I feel you. However:", "")
        cool_print("Please:", "respond with 'yes' or 'no'")
        return query_yes_no(question)


def wait_unless_interrupted(wait_time: float, progress_bar_in_notebook: bool = True):
    if progress_bar_in_notebook:
        _tqdm = tqdm_notebook
    else:
        _tqdm = tqdm
    for sleepy in _tqdm(range(wait_time), position=0, colour="green"):
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            if query_yes_no("\nBreak the loop and run?"):
                break
            else:
                if query_yes_no("\nAbort?"):
                    raise KeyboardInterrupt("OK, aborting")
                else:
                    cool_print("OK, waiting...")
                    wait_unless_interrupted(wait_time=wait_time - sleepy)


def contract_dataframe_with_minmax_values(
    df: pd.DataFrame,
    variable_name: str,
    find_maximal_value: bool,
    columns_to_skip: Optional[List[str]] = None,
    grouping_columns: Optional[List[str]] = None,
    allow_degeneracy: bool = False,
) -> pd.DataFrame:
    """
    Contract a DataFrame to only include rows with the minimum or maximum value of a specified variable.
    :param df:
    :param variable_name:
    The name of the variable to find the min/max value for.
    :param columns_to_skip:
    Optional list of columns to skip when finding the min/max value.
    :param find_maximal_value:
    Whether to find the maximum value (True) or minimum value (False).
    :param grouping_columns:
    Optional list of columns to group by before finding the min/max value.
    :param allow_degeneracy
    If True, allows for multiple rows with the same min/max value to be returned.


    :return:
    """

    if isinstance(variable_name, list):
        assert (
            len(variable_name) == 1
        ), "If variable_name is a list, it should contain exactly one element."
        variable_name = variable_name[0]

    if isinstance(grouping_columns, str):
        grouping_columns = [grouping_columns]

    if columns_to_skip is None:
        columns_to_skip = []

    # To avoid errors
    df = df.copy().drop(columns=columns_to_skip)
    if grouping_columns is not None:
        df_grouped = df.groupby(grouping_columns)

        if allow_degeneracy:
            df_minmax = df_grouped[variable_name].transform(
                "max" if find_maximal_value else "min"
            )
            return df[df[variable_name] == df_minmax].reset_index(drop=True)

    else:
        if allow_degeneracy:
            minmax_val = (
                df[variable_name].max()
                if find_maximal_value
                else df[variable_name].min()
            )
            return df[df[variable_name] == minmax_val].reset_index(drop=True)

        df_grouped = df

    if find_maximal_value:
        idx = df_grouped[variable_name].idxmax()
    else:
        idx = df_grouped[variable_name].idxmin()

    if grouping_columns is None:
        idx = [idx]

    return df.loc[idx].reset_index(drop=True)


def contract_dataframe_with_aggregating_functions(
    df: pd.DataFrame,
    functions_to_apply: Union[List[str], Dict[str, str]],
    columns_to_skip: Optional[Union[str, List[str]]] = None,
    grouping_columns: Optional[List[str]] = None,
    record_min_max_index_for_variables: Optional[str] = None,
    flatten_column_names=True,
) -> pd.DataFrame:
    """
    Contract a DataFrame by applying specified aggregation functions to its columns,
    optionally grouping by specified columns.

    It applies "functions_to_apply" to all columns besides "columns_to_skip" and "grouping_columns".
    If "grouping_columns" is specified, then "functions_to_apply" are applied to each group separately.

    :param df:
    :param functions_to_apply:
    This can be a list of functions to apply to all columns or a dictionary mapping column names to lists of functions.
    :param columns_to_skip
    If specified, these columns will not be returned in the resulting DataFrame.
    :param grouping_columns:
    Optional list of columns to group by before applying the functions.
    :param record_min_max_index_for_variables:
    Optional list of column names for which to record the index of the min/max values
    if 'min' or 'max' functions are included in functions_to_apply.
    :param flatten_column_names:
    If True, flattens the multi-level column names resulting from aggregation.
    :return:
    """

    if columns_to_skip is None:
        columns_to_skip = []

    df = df.copy().drop(columns=columns_to_skip)

    if isinstance(grouping_columns, str):
        grouping_columns = [grouping_columns]

    columns_to_apply_to = list(set(df.columns))
    if grouping_columns is not None:
        if columns_to_skip is not None:
            grouping_columns = list(
                set(grouping_columns).difference(set(columns_to_skip))
            )

        df_grouped = df.groupby(grouping_columns)
        columns_to_apply_to = list(
            set(columns_to_apply_to).difference(set(grouping_columns))
        )

    else:
        df_grouped = df

    if columns_to_skip is not None:
        columns_to_apply_to = list(
            set(columns_to_apply_to).difference(set(columns_to_skip))
        )

    if isinstance(columns_to_apply_to, str):
        columns_to_apply_to = [columns_to_apply_to]

    if isinstance(functions_to_apply, str):
        functions_to_apply = [functions_to_apply]

    if isinstance(functions_to_apply, list):
        functions_to_apply = {
            col_name: functions_to_apply.copy() for col_name in columns_to_apply_to
        }

    if record_min_max_index_for_variables is not None:
        if isinstance(record_min_max_index_for_variables, str):
            record_min_max_index_for_variables = [record_min_max_index_for_variables]

        for col_name in record_min_max_index_for_variables:
            if col_name in columns_to_apply_to:
                if "max" in functions_to_apply[col_name]:
                    functions_to_apply[col_name].append("idxmax")
                if "min" in functions_to_apply[col_name]:
                    functions_to_apply[col_name].append("idxmin")

    contracted_dataframe = df_grouped.agg(functions_to_apply)

    if grouping_columns is None:
        contracted_dataframe = contracted_dataframe.unstack().to_frame().T

    if flatten_column_names:
        contracted_dataframe.columns = [
            "_".join(col).rstrip("_") for col in contracted_dataframe.columns
        ]

    return contracted_dataframe.reset_index(drop=grouping_columns is None)


def contract_dataframe_with_functions(
    df: pd.DataFrame,
    unique_variables_columns_names: List[str],
    functions_to_apply: Union[str, List[str]],
    contraction_column: Optional[Union[str, List[str]]] = None,
    record_min_max_for: Optional[str] = None,
):

    # TODO(FBM): depreciated, please use "contract_dataframe_with_aggregating_functions" instead
    """

    The function will take dataframe "df" and perform contraction along "contraction_column" by applying functions
    "functions_to_apply" w.r.t. variables that ARE NOT in "unique_variables_columns_names".

    In other words -- function creates "subdataframes" specified by unique tuples of values of columns from
    "unique_column_names", and within each subdataframe it applies "functions_to_apply" to values of other columns,
    contracting over all rows in that subdataframe ("contraction_column" values are then removed).

    If contraction column is set to None (or ""), then the function does not remove any column.

    WARNING: For this to work properly, for fixed tuple of values of variables from "unique_variables_columns_names"
    (plus "contraction_column"), there must be just one value for each other variable.
    In other words, "unique_variables_columns_names+[contraction_column]" must uniquely specify a row.

    Since this was developed to analyze very specific data for QAOA experiments, I will give example related to that:
    Say we have dataframe with columns ["i","k, "p", "energy", "approximation_ratio"]
    (here "i" = hamiltonian index, "k" = layer cycle count, "p" = depth)
    Say we want to, for each unique pair ("p", "k") calculate an average energy and AR over hamiltonians.
    Then we pass:
    - contraction_column = "i", to contract over Hamiltonians
    - unique_columns_names = ["p", "k"], to specify what is averaged
    - functions_to_apply = ["mean"], to calculate mean ove hamiltonian indices for each unique (p,k) pair;

    Additionally passing functions_to_apply = ["mean", "min", "max"], will also give the minimal and maximal value of
    "variables_names" for each ("p", "k").
    If we also pass record_min_max_for = "approximation_ratio", it will return hamiltonian indices corresponding to
    the lowest and the highest approximation ratios for each ("p", "k").


    :param df: input dataframe
    :param contraction_column: column over which we want to contract
    :param unique_variables_columns_names:
    :param functions_to_apply:
    :param record_min_max_for:
    :return:
    """

    print("THIS function is depreciated, please use 'contract_dataframe_with_aggregating_functions' instead")
    if contraction_column is None:
        contraction_column = [""]
    if isinstance(contraction_column, str):
        contraction_column = [contraction_column]
    if isinstance(unique_variables_columns_names, str):
        unique_variables_columns_names = [unique_variables_columns_names]
    if isinstance(functions_to_apply, str):
        functions_to_apply = [functions_to_apply]

    if contraction_column in unique_variables_columns_names:
        unique_variables_columns_names = list(
            set(unique_variables_columns_names).difference(set(contraction_column))
        )

    variables_names = list(
        set(df.columns).difference(
            set(unique_variables_columns_names + contraction_column)
        )
    )

    df = df.fillna("None", inplace=False).copy()
    functions_dictionary = {
        key: copy.deepcopy(functions_to_apply) for key in variables_names
    }
    if record_min_max_for is not None:
        if not isinstance(record_min_max_for, list):
            record_min_max_for = [record_min_max_for]

        for rmmf in record_min_max_for:
            if "max" in functions_to_apply:
                functions_dictionary[rmmf].insert(0, "idxmax")
            if "min" in functions_to_apply:
                functions_dictionary[rmmf].insert(0, "idxmin")

    if len(unique_variables_columns_names) == 0:
        grouped_initial = df
        # TODO(FBM): implement this
        raise NotImplementedError("This is not implemented yet")
    else:
        grouped_initial = df.groupby(unique_variables_columns_names)

    contracted_dataframe = grouped_initial.agg(functions_dictionary).reset_index()
    contracted_dataframe.columns = [
        "_".join(col).rstrip("_") for col in contracted_dataframe.columns
    ]
    df_con = contracted_dataframe

    if record_min_max_for is not None:
        for rmmf in record_min_max_for:
            for minmax in ["min", "max"]:
                if minmax in functions_to_apply:
                    # TODO FBM: make sure degeneracy does not break this (comment: I think it shouldn't, it will just choose
                    #           the first occurance of min/max value)
                    minmax_indices = df_con[f"{rmmf}_idx{minmax}"].values

                    for c_c in contraction_column:
                        try:
                            df_con[f"{contraction_column}_{rmmf}_{minmax}"] = df[
                                f"{contraction_column}"
                            ][minmax_indices].values
                        except Exception as e:
                            print(df_con)
                            print("Error!", e)
                            print(contraction_column, rmmf)
                            print(minmax_indices)

                            raise Exception

                    df_con.drop(columns=[f"{rmmf}_idx{minmax}"], inplace=True)

    return df_con


def _apply_func_to_series(data, func):
    data = data.apply(func)
    return data


def df_column_apply_function_parallelized(
    series: pd.Series, function_to_apply: Callable, number_of_threads=1
):
    """
    Apply some function to pandas series in a parallelized way.
    :param series:
    :param function_to_apply:
    :param number_of_threads:
    :return:
    """
    if number_of_threads == 1:
        return series.apply(function_to_apply)
    else:
        pool = Pool(number_of_threads)
        series_split = np.array_split(series, number_of_threads)
        results = pool.starmap(
            _apply_func_to_series, [(data, function_to_apply) for data in series_split]
        )
        series = pd.concat(results)
        pool.close()
        pool.join()
        return series


def apply_permutation_operator_to_statevector(
    statevector: np.ndarray,
    permutation: (
        Tuple[int, ...] | List[int] | Tuple[Tuple[int, int]] | List[Tuple[int, int]]
    ),
):
    """
    Apply a qubit permutation to a statevector without materializing the full permutation matrix.

    This function efficiently permutes qubits in a statevector by directly manipulating indices
    rather than constructing and applying a full permutation matrix, which is memory-efficient
    for large systems.

    :param statevector: The input statevector as a 1D numpy array of shape (2**n_qubits,)
    :param permutation: Tuple or list specifying the qubit permutation.
                       permutation[i] = j means qubit i goes to position j.
                       OR
                       list/tuple of equivalent transpositions, e.g. [(0, 1), (2, 3)]

    :return: Permuted statevector as a numpy array

    Example:
        # 3-qubit system with permutation (2, 1, 0) - reverses qubit order
        statevector = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # |000> state
        permuted = apply_permutation_operator_to_statevector(statevector, (2, 1, 0))
    """
    from sympy.combinatorics.permutations import Permutation

    # Determine number of qubits from statevector size
    number_of_qubits = int(np.log2(len(statevector)))

    if isinstance(permutation[0], (tuple, list)):
        assert (
            len(permutation[0]) == 2
        ), "If permutation is a list of tuples, it should contain exactly two elements."
        transpositions = permutation
    else:
        permutation = list(permutation)
        # Get transpositions that implement this permutation
        perm_obj = Permutation(permutation)
        transpositions = perm_obj.transpositions()

    # Reshape statevector to n-qubit tensor
    state_tensor = statevector.reshape([2] * number_of_qubits)

    # Apply each transposition by swapping tensor axes
    for i, j in transpositions:
        state_tensor = np.swapaxes(state_tensor, i, j)

    # Flatten back to statevector
    return state_tensor.reshape(-1)
