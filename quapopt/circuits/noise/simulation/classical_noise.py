from typing import Tuple, Dict, List, Union

import numpy as np


from quapopt.additional_packages.ancillary_functions_usra import efficient_math as em

# Lazy monkey-patching of cupy
from quapopt import AVAILABLE_SIMULATORS
if 'cupy' in AVAILABLE_SIMULATORS:
    import cupy as cp
else:
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

    if isinstance(ideal_samples_array, cp.ndarray):
        bck = cp
    elif isinstance(ideal_samples_array,np.ndarray):
        bck = np
    else:
        raise ValueError("ideal_samples_array should be either numpy or cupy array")


    if p_01 != 0.0:
        ones_size = int(ones_mask.sum())
        ideal_samples_array[ones_mask] = ideal_samples_array[ones_mask] ^ rng.binomial(n=1, p=p_01, size=ones_size).astype(bck.int32)

    if p_10 != 0.0:
        zeros_size = int(zeros_mask.sum())
        ideal_samples_array[zeros_mask] = ideal_samples_array[zeros_mask] ^ rng.binomial(n=1, p=p_10, size=zeros_size).astype(bck.int32)

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
            dtype=int).T

        ideal_samples_array = ideal_samples_array ^ (bits_flipped_or_not_ones * ones_mask)

    if p_10_list is not None:
        # since probabilities are different, we need to sample each qubit separately
        bits_flipped_or_not_zeros = bck.array(
            [rng.binomial(n=1, p=p_10, size=number_of_samples) for qi, p_10 in enumerate(p_10_list)],
            dtype=int).T
        ideal_samples_array = ideal_samples_array ^ (bits_flipped_or_not_zeros * zeros_mask)

    return ideal_samples_array




def add_1q_tensor_product_noise_to_samples(ideal_samples_array: Union[np.ndarray, cp.ndarray],
                                           p_01_errors: Union[List[float], float] = None,
                                           p_10_errors: Union[List[float], float] = None,
                                           rng=None):

    if p_01_errors is None and p_10_errors is None:
        raise ValueError("At least one of the probabilities should be provided")

    if rng is None:
        if isinstance(ideal_samples_array, cp.ndarray):
            rng = cp.random.default_rng(seed=None)
        elif isinstance(ideal_samples_array, np.ndarray):
            rng = np.random.default_rng(seed=None)

        else:
            raise ValueError("ideal_samples_array should be either numpy or cupy array")
    elif isinstance(rng,int):
        if isinstance(ideal_samples_array, cp.ndarray):
            rng = cp.random.default_rng(seed=rng)
        elif isinstance(ideal_samples_array, np.ndarray):
            rng = np.random.default_rng(seed=rng)
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
