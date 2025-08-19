import time
from typing import Tuple, Dict, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from quapopt.additional_packages.ancillary_functions_usra import ancillary_functions as anf
from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em

# Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError, ModuleNotFoundError):
    import numpy as cp


def get_stochastic_matrix_1q(p_10: float,
                             p_01: float,
                             ) -> np.ndarray:
    """

    Args:
        p_10: Probability of incorrectly measuring 1 when the true state is 0
        p_01: Probability of incorrectly measuring 0 when the true state is 1
    Returns: Left-stochastic matrix that represents the noise model of a single qubit
    """

    return np.array([[1 - p_10, p_01],
                     [p_10, 1 - p_01]])


def create_tensor_product_matrix(matrices_dictionary: Dict[Tuple[int], np.ndarray]) -> np.ndarray:
    """
    NOTE: indices of qubits shouldn't overlap for this to work properly
    :param matrices_dictionary: Key -- indices of qubits, value -- operator that acts on those qubits.
    :return: global matrix
    """
    number_of_qubits = sum([len(x) for x in matrices_dictionary.keys()])
    dtype = type(list(matrices_dictionary.values())[0][0, 0])

    global_matrix = np.eye(2 ** number_of_qubits, dtype=dtype)

    for node_ids, local_matrix in matrices_dictionary.items():
        embedded_local_matrix = em.embed_operator_in_bigger_hilbert_space(number_of_qubits=number_of_qubits,
                                                                          global_indices=node_ids,
                                                                          local_operator=local_matrix)
        global_matrix = global_matrix @ embedded_local_matrix

    return global_matrix


def add_tensor_product_stochastic_noise_to_samples(ideal_samples_list: Union[List[Tuple[int, ...]], np.ndarray],
                                                   stochastic_matrices_dictionary: Dict[Tuple[int, ...], np.ndarray],
                                                   number_of_threads=1,
                                                   show_progress_bar=None,
                                                   rng=None):
    """
    # TODO FBM: This function is super bruteforce, probably should be optimized.
    #TODO FBM: add unit tests

    :param ideal_samples_list:
    :param stochastic_matrices_dictionary:
    :return:
    """
    if isinstance(ideal_samples_list, np.ndarray):
        ideal_samples_list = [tuple(x) for x in ideal_samples_list]

    if rng is None:
        rng = np.random.default_rng(seed=None)

    ideal_samples = pd.DataFrame(data={'experiment_id': ideal_samples_list,
                                       'bitstrings_array': ideal_samples_list})

    number_of_qubits = len(ideal_samples_list[0])

    if show_progress_bar is None:
        show_progress_bar = number_of_qubits > 10 and number_of_samples > 1000

    clustering = sorted(list(stochastic_matrices_dictionary.keys()))
    unique_sizes = set([len(x) for x in clustering])

    classical_registers_local = {size: anf.get_full_classical_register(number_of_bits=size,
                                                                       register_format=tuple) for size in unique_sizes}
    t0 = time.perf_counter()
    # we check what are the "marginal input states" for all subsets. This will determine columns
    # of the stochastic matrices we will sample from.
    from ancillary_functions_usra.data_analysis import marginals as marg_fun

    marginals_df = marg_fun.get_marginals_from_bitstrings(results_dataframe=ideal_samples,
                                                          subsets_of_qubits=clustering,
                                                          number_of_threads=number_of_threads,
                                                          print_progress_bar=False,
                                                          normalize=False,
                                                          store_number_of_shots=False)
    t1 = time.perf_counter()
    marginals_df['state_int'] = marginals_df['marginal'].apply(np.argmax)
    marginals_df.sort_values(by=['state_int'], axis=0, inplace=True)
    marginals_df['state_int_count'] = marginals_df.apply(lambda row: row['marginal'][row['state_int']][0], axis=1)

    t2 = time.perf_counter()

    all_values_sets_exp = {exp_id: {'qi': [], 'si': []} for exp_id in ideal_samples_list}
    for subset, noise_matrix in tqdm(list(stochastic_matrices_dictionary.items()), disable=not show_progress_bar):
        local_register = classical_registers_local[len(subset)]
        marginals_subset = anf.find_dataframe_subset(df=marginals_df,
                                                     variable_values_pairs=[('subset', subset)])
        marginals_subset.drop(columns=['marginal'], inplace=True)
        marginals_subset_ints_counted = anf.contract_dataframe_with_functions(df=marginals_subset,
                                                                              contraction_column=None,
                                                                              unique_variables_columns_names=[
                                                                                  'experiment_id',
                                                                                  'subset',
                                                                                  'state_int',
                                                                              ],
                                                                              functions_to_apply=['sum'])
        # numpy_rng = np.random.default_rng(seed=None)

        for state_int, state_int_count, exp_id in zip(marginals_subset_ints_counted['state_int'].values,
                                                      marginals_subset_ints_counted['state_int_count_sum'].values,
                                                      marginals_subset_ints_counted['experiment_id']):
            # TODO FBM: is that possible?
            if state_int_count == 0:
                continue
            random_samples_int = rng.multinomial(n=state_int_count,
                                                 pvals=noise_matrix[:, state_int])
            sequence = []
            for index, counts in enumerate(random_samples_int):
                sequence += [local_register[index]] * counts
            rng.shuffle(sequence)
            all_values_sets_exp[exp_id]['qi'] += [subset]
            all_values_sets_exp[exp_id]['si'] += [sequence]

    offset = 0
    samples_array = np.zeros((len(ideal_samples_list), number_of_qubits), dtype=int)
    for key, samples_separated in all_values_sets_exp.items():
        qubit_indices = samples_separated['qi']
        sample_indices = samples_separated['si']
        number_of_samples_here = len(sample_indices[0])

        for ind_enum, (q_inds, s_values) in enumerate(zip(qubit_indices, sample_indices)):
            samples_array[offset:offset + number_of_samples_here, q_inds] = s_values
        offset += number_of_samples_here
    return samples_array



def _add_identical_1q_tensor_product_noise_to_samples(ideal_samples_array: Union[np.ndarray,cp.ndarray],
                                                      p_01: float = None,
                                                      p_10: float = None,
                                                      rng=None)->Union[np.ndarray,cp.ndarray]:
    """
    Adds identical 1-qubit noise to the samples. This is done by flipping bits with given probabilities.

    :param ideal_samples_array: samples to which noise is added
    WARNING: the array is changed in place, so if you want to keep the original samples, make a copy first!

    :param p_01: probability of measuring 0 if input was |1>
    :param p_10: probability of measuring 1 if input was |0>
    :param rng: numpy or cupy rng object
    :return:
    """

    if p_01 is None:
        p_01 = 0.0
    if p_10 is None:
        p_10 = 0.0

    if p_01 != 0.0:
        ones_mask = ideal_samples_array == 1
    if p_10 != 0.0:
        zeros_mask = ideal_samples_array == 0

    if isinstance(ideal_samples_array,np.ndarray):
        bck = np
    elif isinstance(ideal_samples_array,cp.ndarray):
        bck = cp
    else:
        raise ValueError("ideal_samples_array should be either numpy or cupy array")


    if p_01 != 0.0:
        ones_size = int(ones_mask.sum())
        ideal_samples_array[ones_mask] = ideal_samples_array[ones_mask] ^ rng.binomial(n=1, p=p_01, size=ones_size).astype(bck.uint8)

    if p_10 != 0.0:
        zeros_size = int(zeros_mask.sum())
        ideal_samples_array[zeros_mask] = ideal_samples_array[zeros_mask] ^ rng.binomial(n=1, p=p_10, size=zeros_size).astype(bck.uint8)

    return ideal_samples_array


def _add_nonidentical_1q_tensor_product_noise_to_samples(ideal_samples_array: Union[np.ndarray, cp.ndarray],
                                                         p_01_list: List[float] = None,
                                                         p_10_list: List[float] = None,
                                                         rng=None):
    """
    Adds non-identical 1-qubit noise to the samples. This is done by flipping bits with given probabilities.
    Same as "_add_identical_1q_tensor_product_noise_to_samples", but since probabilities are different for each qubit,
    we need to sample each qubit separately. This is done by creating a mask for each qubit and then sampling
    the bits separately.

    :param ideal_samples_array:
    :param p_01_list: probabilities of measuring 0 if input was |1>
    :param p_10_list: probabilities of measuring 1 if input was |0>
    :param rng: numpy or cupy rng object
    :return:
    """


    if isinstance(ideal_samples_array, np.ndarray):
        bck = np
    elif isinstance(ideal_samples_array, cp.ndarray):
        bck = cp
    else:
        raise ValueError("ideal_samples_array should be either numpy or cupy array")

    if p_01_list is not None:
        ones_mask = ideal_samples_array == 1
    if p_10_list is not None:
        zeros_mask = ideal_samples_array == 0

    number_of_samples = ideal_samples_array.shape[0]
    if p_01_list is not None:
        # since probabilities are different, we need to sample each qubit separately
        # Here we have overhead that effectively doubles number of samples, and we use mask below to only use some of them.
        # TODO FBM: this is not optimal, but I think actually might be faster than alternative solutions
        bits_flipped_or_not_ones = bck.array(
            [rng.binomial(n=1, p=p_01, size=number_of_samples) for qi, p_01 in enumerate(p_01_list)],
            dtype=bck.uint8).T

        ideal_samples_array = ideal_samples_array ^ (bits_flipped_or_not_ones * ones_mask)

    if p_10_list is not None:
        # since probabilities are different, we need to sample each qubit separately
        bits_flipped_or_not_zeros = bck.array(
            [rng.binomial(n=1, p=p_10, size=number_of_samples) for qi, p_10 in enumerate(p_10_list)],
            dtype=bck.uint8).T
        ideal_samples_array = ideal_samples_array ^ (bits_flipped_or_not_zeros * zeros_mask)

    return ideal_samples_array




def add_1q_tensor_product_noise_to_samples(ideal_samples_array: Union[np.ndarray, cp.ndarray],
                                           p_01_errors: Union[List[float], float] = None,
                                           p_10_errors: Union[List[float], float] = None,
                                           rng=None):

    if p_01_errors is None and p_10_errors is None:
        raise ValueError("At least one of the probabilities should be provided")

    if rng is None:
        if isinstance(ideal_samples_array, np.ndarray):
            rng = np.random.default_rng(seed=None)
        elif isinstance(ideal_samples_array, cp.ndarray):
            rng = cp.random.default_rng(seed=None)
        else:
            raise ValueError("ideal_samples_array should be either numpy or cupy array")
    elif isinstance(rng,int):
        if isinstance(ideal_samples_array, np.ndarray):
            rng = np.random.default_rng(seed=rng)
        elif isinstance(ideal_samples_array, cp.ndarray):
            rng = cp.random.default_rng(seed=rng)
        else:
            raise ValueError("ideal_samples_array should be either numpy or cupy array")

    number_of_qubits = ideal_samples_array.shape[1]
    ideal_samples_array = ideal_samples_array.copy()

    if p_01_errors is None:
        p_01_errors = 0.0
    if p_10_errors is None:
        p_10_errors = 0.0

    if isinstance(p_01_errors, float) and isinstance(p_10_errors, float):
        return _add_identical_1q_tensor_product_noise_to_samples(ideal_samples_array=ideal_samples_array,
                                                                 p_01=p_01_errors,
                                                                 p_10=p_10_errors,
                                                                 rng=rng)
    else:
        if isinstance(p_01_errors, float):
            p_01_errors = [p_01_errors] * number_of_qubits
        if isinstance(p_10_errors, float):
            p_10_errors = [p_10_errors] * number_of_qubits
        return _add_nonidentical_1q_tensor_product_noise_to_samples(ideal_samples_array=ideal_samples_array,
                                                                    p_01_list=p_01_errors,
                                                                    p_10_list=p_10_errors,
                                                                    rng=rng)












if __name__ == '__main__':

    raise KeyboardInterrupt

    number_of_qubits_test = 10
    dim_test = 2 ** number_of_qubits_test

    numpy_rng_test = np.random.default_rng(seed=0)
    min_p10_test = 0.005
    max_p10_test = 0.1
    min_p01_test = 0.02
    max_p01_test = 0.5
    noise_matrices_dict = {}

    for i in range(number_of_qubits_test):
        p_10_test = numpy_rng_test.uniform(low=min_p10_test, high=max_p10_test)
        p_01_test = numpy_rng_test.uniform(low=min_p01_test, high=max_p01_test)
        stochastic_matrix_test = get_stochastic_matrix_1q(p_10=p_10_test, p_01=p_01_test)
        noise_matrices_dict[(i,)] = stochastic_matrix_test

    global_noise_matrix = create_tensor_product_matrix(matrices_dictionary=noise_matrices_dict)

    full_register_test = anf.get_full_classical_register(number_of_bits=number_of_qubits_test,
                                                         register_format=tuple)
    ideal_samples_test = []
    ideal_distro_test = []

    total_number_of_samples_max = 10 ** 6

    for bts_test in full_register_test:
        number_of_samples = numpy_rng_test.integers(low=0, high=int(total_number_of_samples_max / dim_test))
        ideal_samples_test += [bts_test] * number_of_samples
        ideal_distro_test.append(number_of_samples * 1.0)
    # ideal_samples_test = np.array(ideal_samples_test)
    ideal_distro_test = np.array(ideal_distro_test)
    norm_test = np.sum(ideal_distro_test)
    ideal_distro_test *= 1 / norm_test
    ideal_distro_test = ideal_distro_test.reshape(-1, 1)

    noisy_distro_global_test = global_noise_matrix @ ideal_distro_test

    noisy_samples_test = add_tensor_product_stochastic_noise_to_samples(ideal_samples_list=ideal_samples_test,
                                                                        stochastic_matrices_dictionary=noise_matrices_dict, )
    # import Counter from collections
    from collections import Counter

    noisy_distro_test = dict(Counter([tuple(x) for x in noisy_samples_test]))

    noisy_distro_local_test = []
    for bts_test in full_register_test:
        noisy_distro_local_test.append(noisy_distro_test[bts_test] * 1.0)
    noisy_distro_local_test = np.array(noisy_distro_local_test)
    norm_local_test = np.sum(noisy_distro_local_test)
    noisy_distro_local_test *= 1 / norm_local_test
    noisy_distro_local_test = noisy_distro_local_test.reshape(-1, 1)
    if norm_local_test != norm_test:
        raise ValueError("Normalization is wrong")

    diff = np.linalg.norm(noisy_distro_local_test - noisy_distro_global_test, ord=1)
    anf.cool_print("Difference norm is:", diff)
    anf.cool_print("Conservative sample error bound (samples):",
                   f"{dim_test / np.sqrt(norm_test)} ({norm_test})")
    if diff > dim_test / np.sqrt(norm_test):
        # print("Difference norm is: ", diff)
        # print(noisy_distro_global_test)
        raise ValueError("Distributions are not the same")

    # print(noisy_samples_test)

    # print(ideal_samples_test)
