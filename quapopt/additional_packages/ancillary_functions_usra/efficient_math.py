# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import time
from typing import Optional, Union, List, Tuple, Dict, Any
import itertools
#Lazy monkey-patching of numba
try:
    import numba
    from quapopt.additional_packages.ancillary_functions_usra._efficient_math_cuda import *
except(ImportError,ModuleNotFoundError):
    pass


from quapopt.additional_packages.ancillary_functions_usra._efficient_math_cython import *

#Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
    import numpy as cp



def _sample_from_probability_distribution_numpy_multinomial(probabilities: np.ndarray,
                                                            number_of_samples: int,
                                                            numpy_rng: np.random.Generator = None) -> np.ndarray:
    number_of_qubits = int(np.log2(len(probabilities)))

    samples_integers = numpy_rng.multinomial(n=number_of_samples,
                                             pvals=probabilities)

    # Get the indices where count > 0
    non_zero_indices = np.nonzero(samples_integers)[0]

    # Create an array of the non-zero counts
    counts = samples_integers[non_zero_indices]

    # Convert integers to binary representations
    binary_array = ((non_zero_indices[:, np.newaxis] & (1 << np.arange(number_of_qubits - 1, -1, -1)))) > 0

    # Create the final array by repeating each row according to its count
    samples_binary = np.repeat(binary_array, counts, axis=0)
    return samples_binary.astype(int)


def _sample_from_probability_distribution_numpy_searchsorted(probabilities: np.ndarray,
                                                             number_of_samples: int,
                                                             numpy_rng: np.random.Generator = None) -> np.ndarray:
    """
    This function samples from a probability distribution using numpy's searchsorted function.
    :param probabilities: The probabilities of the different outcomes.
    :param number_of_samples: The number of samples to draw.
    :param numpy_rng: The numpy random generator to use.
    :return: An array of samples.

    """

    t0 = time.perf_counter()
    number_of_qubits = int(np.log2(len(probabilities)))
    t1 = time.perf_counter()
    # cumprobs = np.cumsum(probabilities)
    cumprobs = cython_cumsum(probabilities)
    t2 = time.perf_counter()
    # this returns the indices
    samples_integers = np.searchsorted(cumprobs, numpy_rng.random(number_of_samples))
    t3 = time.perf_counter()

    binary_array = ((samples_integers[:, np.newaxis] & (1 << np.arange(number_of_qubits - 1, -1, -1)))) > 0
    t4 = time.perf_counter()
    binary_array = binary_array.astype(int)
    t5 = time.perf_counter()
    ts = [t0, t1, t2, t3, t4, t5]
    dts = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
    dts_names = ['Logarithm', 'Cumsum', 'Searchsorted', 'Binary conversion', 'Final conversion']
    #
    # for dt, name in zip(dts, dts_names):
    #     print(f'{name}: {dt}')
    #

    return binary_array


def sample_from_probability_distribution(probabilities: np.ndarray,
                                         number_of_samples: int,
                                         numpy_rng: np.random.Generator = None,
                                         sampling_method='numpy_multinomial') -> np.ndarray:
    if numpy_rng is None:
        numpy_rng = np.random.default_rng()

    if sampling_method == 'auto':
        # This is based on empirical observations
        if number_of_samples <= 100000:
            sampling_method = 'numpy_searchsorted'
        else:
            sampling_method = 'numpy_multinomial'

    if sampling_method == 'numpy_multinomial':
        return _sample_from_probability_distribution_numpy_multinomial(probabilities=probabilities,
                                                                       number_of_samples=number_of_samples,
                                                                       numpy_rng=numpy_rng)
    elif sampling_method == 'numpy_searchsorted':
        return _sample_from_probability_distribution_numpy_searchsorted(probabilities=probabilities,
                                                                        number_of_samples=number_of_samples,
                                                                        numpy_rng=numpy_rng)
    else:
        raise ValueError('sampling_method should be either "numpy_multinomial" or "numpy_searchsorted"')


def sample_from_statevector(statevector: np.ndarray,
                            number_of_samples: int,
                            numpy_rng: np.random.Generator = None,
                            sampling_method='auto') -> np.ndarray:
    probabilities = cython_abs_squared(statevector)
    return sample_from_probability_distribution(probabilities=probabilities,
                                                number_of_samples=number_of_samples,
                                                numpy_rng=numpy_rng,
                                                sampling_method=sampling_method)


def _calculate_energies_from_bitstrings_2_local(bitstrings_array: Union[cp.ndarray, np.ndarray],
                                                couplings_array: Union[cp.ndarray, np.ndarray],
                                                local_fields: Optional[Union[cp.ndarray, np.ndarray]] = None,
                                                backend='numpy') -> Union[cp.ndarray, np.ndarray]:
    """
    This function calculates the energies of the bitstrings given the Hamiltonian.
    :param bitstrings_array:
    An array of bitstrings, where each row is a bitstring.
    :param couplings_array:
    A real symmetric matrix of couplings.
    :param local_fields:
    A real vector of local fields.
    :param backend:
    'cupy' or 'numpy'; recommend 'cupy' for large problems.
    :return:
    A 1-dimensional array, where each element is the energy of the corresponding bitstring.
    """

    if backend == 'numpy':
        import numpy as bck
    elif backend == 'cupy':
        import cupy as bck



    else:
        raise ValueError(f'backend_computation should be either "numpy" or "cupy"')

    # TODO(FBM): this is the most memory intensive part
    products = bck.einsum('ij,ij->i',
                          bck.dot(bitstrings_array, couplings_array) / 2,
                          bitstrings_array)

    if local_fields is not None:
        products += bck.dot(bitstrings_array,
                            local_fields)
    return products


def calculate_energies_from_bitstrings_2_local(bitstrings_array: Union[cp.ndarray, np.ndarray],
                                               pm_input: bool = False,
                                               # Those arguments are used together.
                                               adjacency_matrix: Optional[Union[cp.ndarray, np.ndarray, list]] = None,
                                               local_fields_present: Optional[bool] = None,
                                               # Those arguments are used together.
                                               couplings_array: Optional[Union[cp.ndarray, np.ndarray, list]] = None,
                                               local_fields: Optional[Union[cp.ndarray, np.ndarray]] = None,
                                               # 'cupy' or 'numpy'
                                               computation_backend='numpy',
                                               output_backend='numpy') -> Union[cp.ndarray, np.ndarray]:
    """
    This function calculates the energies of the bitstrings given the Hamiltonian.
    It is a bit complicated to account for various types of input I used in the past.

    :param bitstrings_array:
    an array of bitstrings, where each row is a bitstring.
    it can be either 0s and 1s, or +1s and 1s which is indicated by "pm_input".
    WARNING: we do not check if the bitstrings are of the correct form, so the user should make sure that the input is correct.
    The default assumption is that it's 0s and 1s.
    :param pm_input: if True, we ASSUME that bitstrings are in the form of +1s and -1s.


    :param adjacency_matrix:
    If adjacency_matrix is provided, then couplings_array and local_fields are set to None and we infer
    the couplings_array and local_fields from the adjacency_matrix.

    Adjacency_matrix should be a real symmetric matrix.
    The off-diagonal terms are the couplings, and the diagonal terms are the local fields.

    Adjacency_matrix can also be a list of tuples of the form [(coeff, (i,j)), ...]
    where the tuples represent the non-zero elements of the adjacency_matrix.



    :param local_fields_present:
    If local_fields_present is True, then we assume that the diagonal terms of the adjacency_matrix are the local fields.
    If False, we assume that the diagonal terms are zero.
    If None, we infer it from the adjacency_matrix.

    :param couplings_array:
    If couplings_array is provided, then adjacency_matrix is not used
    The couplings_array should be a real symmetric matrix.

    :param local_fields:
    If local_fields is provided, then adjacency_matrix is not used
    The local_fields should be a real vector.

    :param computation_backend:
    'cupy' or 'numpy'
    Recommend 'cupy' for large problems.

    :return:
    A 1-dimensional array, where each element is the energy of the corresponding bitstring.

    """

    if adjacency_matrix is None and couplings_array is None:
        raise ValueError('Either adjacency_matrix or couplings_array should be provided')

    if computation_backend == 'cupy':
        import cupy as bck
        if isinstance(bitstrings_array, np.ndarray):
            bitstrings_array = bck.asarray(bitstrings_array)

    elif computation_backend == 'numpy':
        import numpy as bck
    else:
        raise ValueError(f"Unknown backend_computation: {computation_backend}; should be either 'numpy' or 'cupy'")

    if adjacency_matrix is not None:
        if computation_backend == 'cupy':
            if isinstance(adjacency_matrix, np.ndarray):
                adjacency_matrix = bck.asarray(adjacency_matrix)

        if isinstance(adjacency_matrix, list):
            if isinstance(adjacency_matrix[0][1], tuple):
                if computation_backend == 'cupy':
                    import cupy as bck
                elif computation_backend == 'numpy':
                    import numpy as bck
                else:
                    raise ValueError('backend_computation should be either "numpy" or "cupy')
                # This is a situation when the Hamiltonian is of a form [(coeff, (i,j)), ...]
                couplings_array = bck.zeros(shape=(len(bitstrings_array[0]),
                                                   len(bitstrings_array[0]),),
                                            dtype=float)

                if local_fields_present is None or local_fields_present:
                    local_fields = bck.zeros(shape=(len(bitstrings_array[0]),), dtype=float)
                else:
                    local_fields = None

                for coeff, tup in adjacency_matrix:
                    if len(tup) == 1:
                        qi = tup[0]
                        if coeff != 0:
                            local_fields[qi] = coeff
                    elif len(tup) == 2:
                        qi, qj = tup
                        couplings_array[qi, qj] = coeff
                        couplings_array[qj, qi] = coeff
                    else:
                        raise ValueError('keys of weights_matrix should be tuples of length 1 or 2')
            else:
                raise ValueError('keys of weights_matrix should be tuples of length 1 or 2')

        else:
            if local_fields_present is None:
                local_fields_present = bck.any(bck.diag(adjacency_matrix) != 0)
            if local_fields_present:
                local_fields = bck.diag(adjacency_matrix).copy()
                couplings_array = adjacency_matrix.copy()
                bck.fill_diagonal(a=couplings_array, val=0)
            else:
                local_fields = None
                # We do not make in-place operations later, so without local fields we can save on copying
                couplings_array = adjacency_matrix

    if not pm_input:
        # 0 -> 1, 1 -> -1
        # This way the |0> state is the GROUND STATE of the "-sigma_z" Hamiltonian, which is the typical physics Hamiltonian
        bitstrings_array = 1 - 2 * bitstrings_array


    if computation_backend == 'cupy':
        if isinstance(couplings_array, np.ndarray):
            couplings_array = bck.asarray(couplings_array)
        if isinstance(local_fields, np.ndarray):
            local_fields = bck.asarray(local_fields)

    products = _calculate_energies_from_bitstrings_2_local(couplings_array=couplings_array,
                                                           bitstrings_array=bitstrings_array,
                                                           local_fields=local_fields,
                                                           backend=computation_backend)

    if output_backend == computation_backend:
        return products
    else:
        if output_backend == 'numpy':
            return bck.asnumpy(products)
        else:
            return products



def calculate_energies_from_bitstrings(bitstrings_array:Union[np.ndarray, cp.ndarray],
                                       hamiltonian:List[Tuple[float,Tuple[int,...]]],
                                       backend_computation:str='numpy',
                                       backend_output:str='numpy'):
    """
    Calculate the energies of the bitstrings given the Hamiltonian.

    :param bitstrings_array: array of 0s and 1s, each row corresponds to distinct bitstring
    :param hamiltonian:
    :param backend_computation: 'numpy' or 'cupy'
    :param backend_output: 'numpy' or 'cupy'

    #NOTE: Currently "bitwise_xor.reduce" is not implemented in cupy, so the computation is done using numpy regardless
    of the backend_computation.
    #TODO(FBM): this should be refactored. If cupy is desired backend, alternative computation method should be implemented

    :return: Array of energies of bitstrings
    """



    if isinstance(bitstrings_array, np.ndarray):
        comp_array = bitstrings_array
    elif isinstance(bitstrings_array, cp.ndarray):
        comp_array = bitstrings_array.get()
    else:
        raise ValueError('bitstrings_array should be either numpy or cupy array')

    arr = np.array(sum((1 - 2 * np.bitwise_xor.reduce(comp_array[:, node_ids], axis=1)) * coefficient
                for coefficient, node_ids in hamiltonian), dtype=float)

    if backend_output == 'numpy':
        return arr
    elif backend_output == 'cupy':
        return cp.asarray(arr)
    else:
        raise ValueError(f'backend_output should be either "numpy" or "cupy", not {backend_output}')










try:
    from numba import njit, prange
    from numba.typed import List as numba_list

    # === the Numba‐jitted function ===
    @njit(fastmath=True, cache=True, parallel=True)
    def _calculate_energies_from_bitstrings_numba_kernel(coefficients,
                                             subsets_list,
                                             bitstrings_array):
        """
        Calculate energies from bitstrings using Numba.
        :param coefficients:
        :param subsets_list:
        :param bitstrings_array:
        :return:
        """
        n_rows = bitstrings_array.shape[0]
        energies = np.zeros(n_rows, dtype=np.float64)
        for term_index in range(coefficients.shape[0]):
            coeff = coefficients[term_index]
            subset = subsets_list[term_index]
            for row_index in prange(n_rows):
                parity = 0
                # XOR‐reduce over the selected qubit‐columns
                row = bitstrings_array[row_index]
                for qubit_index in subset:
                    parity ^= row[qubit_index]
                # map {0→+1,1→−1} via (1 − 2*bit) and accumulate
                energies[row_index] += coeff * (1 - 2 * parity)
        return energies

    def calculate_energies_from_bitstrings_numba(hamiltonian: List[Tuple[float, Tuple[int, ...]]],
                                                bitstrings_array: np.ndarray) -> np.ndarray:
        """
        Calculate energies from bitstrings using Numba. This tends to be faster than the pure Python implementation
        for large bitstrings arrays.
        :param hamiltonian:
        :param bitstrings_array:
        :return:
        """
        coeffs = np.array([tup[0] for tup in hamiltonian], dtype=np.float64)
        idx_lists = numba_list()
        for tup in hamiltonian:
            idx_lists.append(np.array(tup[1], dtype=np.int64))

        return _calculate_energies_from_bitstrings_numba_kernel(coeffs, idx_lists, bitstrings_array)

except(ImportError,ModuleNotFoundError):
    #monkey patching in case numba is not available
    calculate_energies_from_bitstrings_numba = calculate_energies_from_bitstrings




def solve_hamiltonian_python(hamiltonian):
    number_of_qubits = hamiltonian.number_of_qubits
    all_bitstrings = np.array(list(itertools.product([0, 1], repeat=number_of_qubits)), dtype=int)
    all_energies = calculate_energies_from_bitstrings_2_local(bitstrings_array=all_bitstrings,
                                                              couplings_array=hamiltonian.couplings,
    local_fields=hamiltonian.local_fields,
                                                              computation_backend='numpy')

    return all_energies




#####################################################
#MARGINAL CALCULATION FUNCTIONS#
####################################################

def _get_marginal_from_probability_distribution(probability_distribution:Union[cp.ndarray,np.ndarray],
                                               subset:Union[List[int],Tuple[int]],
                                               number_of_qubits:int):

    if len(probability_distribution.shape) != number_of_qubits:
        probability_distribution = probability_distribution.copy().reshape([2]*number_of_qubits)

    # Sum over the unwanted variables to obtain the marginal distribution
    marg = probability_distribution.sum(axis=tuple(i for i in range(number_of_qubits) if i not in subset))
    marg = marg.reshape(-1,1)

    return marg

def get_marginals_from_probability_distribution(probability_distribution:Union[cp.ndarray,np.ndarray],
                                                subsets: List[Union[Tuple[int, ...],List[int]]],
                                                number_of_qubits: int,
                                                )->Dict[Tuple[int, ...],Union[cp.ndarray,np.ndarray]]:

    all_marginals = {}

    for subset in subsets:
        key = subset
        if isinstance(key, list):
            key = tuple(key)
        all_marginals[key] = _get_marginal_from_probability_distribution(probability_distribution=probability_distribution,
                                                                         subset=subset,
                                                                         number_of_qubits=number_of_qubits)
    return all_marginals








def get_1q_marginals_from_bitstrings_array(bitstrings_array: Union[np.ndarray, cp.ndarray],
                                           qubits_list: Optional[Union[List[int], List[Tuple[int]]]] = None,
                                           normalize=True) -> Union[np.ndarray, cp.ndarray]:
    """
    Function to get the 1-qubit marginals from the bitstrings array.
    :param bitstrings_array:
    :param qubits_list:
    :param normalize:
    :return:
    """

    qubits_list = [x[0] if isinstance(x, tuple) else x for x in qubits_list]

    if isinstance(bitstrings_array, np.ndarray):
        bck = np
    elif isinstance(bitstrings_array, cp.ndarray):
        bck = cp

    if qubits_list is None:
        x1 = bitstrings_array.sum(axis=0)
    else:
        qi_idx = bck.array(qubits_list, dtype=int)
        x1 = bitstrings_array[:, qi_idx].sum(axis=0)

    number_of_samples = bitstrings_array.shape[0]
    if normalize:
        x1 = x1 / number_of_samples
        x0 = 1.0 - x1
    else:
        x0 = number_of_samples - x1

    marginals_1q = bck.array([x0, x1])

    return marginals_1q


def get_2q_marginals_from_bitstrings_array(bitstrings_array: Union[np.ndarray, cp.ndarray],
                                           qubits_pairs: Optional[List[Tuple[int, int]]] = None,
                                           normalize=True) -> Union[np.ndarray, cp.ndarray]:
    """
    Function to get the 2-qubit marginals from the bitstrings array.
    :param bitstrings_array:
    :param qubits_pairs:
    :param normalize:
    :return:
    """

    if isinstance(bitstrings_array, np.ndarray):
        bck = np
    elif isinstance(bitstrings_array, cp.ndarray):
        bck = cp

    if qubits_pairs is None:
        number_of_qubits = bitstrings_array.shape[1]
        qubits_pairs = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]

    qi_idx = bck.array([pair[0] for pair in qubits_pairs], dtype=int)
    qj_idx = bck.array([pair[1] for pair in qubits_pairs], dtype=int)

    # Shift bitstrings to map to integers
    # bitstrings_array_shifted = bitstrings_array << 1
    # bi = bitstrings_array_shifted[:,qi_idx]
    # bj = bitstrings_array[:,qj_idx]
    integer_idx = (bitstrings_array << 1)[:, qi_idx] | bitstrings_array[:, qj_idx]
    # x00 = (integer_idx==0).sum(axis=0)
    # x01 = (integer_idx==1).sum(axis=0)
    # x10 = (integer_idx==2).sum(axis=0)
    # x11 = (integer_idx==3).sum(axis=0)
    marginals_2q = bck.array([(integer_idx == i).sum(axis=0) for i in range(4)])
    if normalize:
        number_of_samples = bitstrings_array.shape[0]
        marginals_2q = marginals_2q / number_of_samples

    return marginals_2q


def _get_higher_locality_marginals_from_bitstrings_array_fixed_locality(bitstrings_array: Union[np.ndarray, cp.ndarray],
                                                                        qubits_subsets: Optional[List[Tuple[int, ...]]],
                                                                        normalize=True) -> Union[
    np.ndarray, cp.ndarray]:
    """
    Function to get the marginals from the bitstrings array for higher locality.
    This function ASSUMES that the locality is fixed. If it is not, it will result in inhomogeneous arrays errors.

    :param bitstrings_array:
    :param qubits_subsets:
    :param normalize:
    :return:
    """
    if isinstance(qubits_subsets[0], int):
        # handling special case of single qubit subsets
        qubits_subsets = [[x] for x in qubits_subsets]

    subset_size = len(qubits_subsets[0])
    # Those cases use special tricks that make the computation faster
    if subset_size == 1:
        return get_1q_marginals_from_bitstrings_array(bitstrings_array=bitstrings_array,
                                                      qubits_list=[x[0] for x in qubits_subsets],
                                                      normalize=normalize)
    elif subset_size == 2:
        return get_2q_marginals_from_bitstrings_array(bitstrings_array=bitstrings_array,
                                                      qubits_pairs=qubits_subsets,
                                                      normalize=normalize)

    if isinstance(bitstrings_array, np.ndarray):
        bck = np
    elif isinstance(bitstrings_array, cp.ndarray):
        bck = cp
    local_register = itertools.product(*[range(2)] * subset_size)
    local_register = bck.array([list(bts) for bts in local_register])
    local_arrays = bitstrings_array[:, qubits_subsets]
    marginals_kq = bck.zeros((len(local_register),
                              len(qubits_subsets)),
                             dtype=local_arrays.dtype)
    for i in range(len(local_register)):
        local_bts = local_register[i]
        counting = bck.all(local_arrays == local_bts, axis=2).sum(axis=0)
        marginals_kq[i] = counting

    if normalize:
        number_of_samples = bitstrings_array.shape[0]
        marginals_kq = marginals_kq / number_of_samples
    return marginals_kq


def get_marginals_from_bitstrings_array(bitstrings_array: Union[np.ndarray, cp.ndarray],
                                        qubits_subsets: Optional[List[Union[Tuple[int, ...], int]]] = None,
                                        qubits_subsets_by_locality: Dict[int, List[Union[Tuple[int, ...], int]]] = None,
                                        normalize=True) -> Union[
    Union[np.ndarray, cp.ndarray], List[Tuple[List[Tuple[int, ...]], Union[np.ndarray, cp.ndarray]]]]:
    # TODO(FBM): when doing locality higher than k, perhaps it would be useful to develop
    # framework that exploits the tensor structure of data for marginals. something to think about

    """
    Function to get the marginals from the bitstrings array for arbitrary qubits subsets.
    It wraps around other functions.

    :param bitstrings_array: (s, n) array of 0s and 1s. s is the number of samples, n is the number of qubits
    :param qubits_subsets:
    list of tuples of qubits. Each tuple is a subset of qubits.
    :param qubits_subsets_by_locality:
     dictionary of qubits subsets by locality.
    The keys are the locality, the values are lists of tuples of qubits.

    WARNING: Only one of qubits_subsets or qubits_subsets_by_locality must be provided.

    :param normalize: Whether to normalize the marginals or not.

    :return:

    What the function returns depends on whether the locality of subsets is fixed or not.

    If the locality is fixed, it returns a single array of marginals for qubits subsets ordered as in input "qubits_subsets"
    or qubits_subsets_by_locality[unique_locality] (then it's dict with only single key).

    If the locality is not fixed, it returns a list of tuples, the length of the list equal to number of unique localities.
    Each 2-tuple is (qubits_subsets, marginals_for_those_subsets).
    WARNING: in this case, the ordering of qubits might be different than in input "qubits_subsets" or qubits_subsets_by_locality,
    that's why we return it as well
    """

    assert qubits_subsets is not None or qubits_subsets_by_locality is not None, "At least one of qubits_subsets or qubits_subsets_by_locality must be provided"
    assert not (
            qubits_subsets is not None and qubits_subsets_by_locality is not None), "Only one of qubits_subsets or qubits_subsets_by_locality must be provided"

    if qubits_subsets is not None:
        # we need to organize subsets by locality
        qubits_subsets_by_locality = {}
        for subset in qubits_subsets:
            if isinstance(subset, int):
                # handling special case of single qubit subsets
                subset = [subset]
            locality = len(subset)
            if locality not in qubits_subsets_by_locality:
                qubits_subsets_by_locality[locality] = []
            qubits_subsets_by_locality[locality].append(subset)

    unique_localities = list(qubits_subsets_by_locality.keys())

    if len(unique_localities) == 1:
        if qubits_subsets is not None:
            pass_subsets = qubits_subsets
        else:
            pass_subsets = qubits_subsets_by_locality[unique_localities[0]]

        return _get_higher_locality_marginals_from_bitstrings_array_fixed_locality(bitstrings_array=bitstrings_array,
                                                                                   qubits_subsets=pass_subsets,
                                                                                   normalize=normalize)

    all_results = []
    for locality, subsets_list in qubits_subsets_by_locality.items():
        marginals_l = _get_higher_locality_marginals_from_bitstrings_array_fixed_locality(
            bitstrings_array=bitstrings_array,
            qubits_subsets=subsets_list,
            normalize=normalize)

        all_results.append((subsets_list, marginals_l))
    return all_results



















if __name__ == '__main__':
    # TODO(FBM): should add proper tests
    # from ancillary_functions_usra import efficient_math as em_usra
    # from ancillary_functions_usra import ancillary_functions as anf
    #Lazy monkey-patching of cupy
    try:
        import cupy as cp
    except(ImportError,ModuleNotFoundError):
        import numpy as cp



    noq_test = 8000
    seed_test = 13
    nos_test = 10000

    rng_test_numpy = np.random.default_rng(seed_test)
    hamiltonian_adjacency_test = rng_test_numpy.uniform(-5, 5, (noq_test, noq_test))
    hamiltonian_adjacency_test = (hamiltonian_adjacency_test + hamiltonian_adjacency_test.T) / 2
    #np.fill_diagonal(hamiltonian_adjacency_test, 0)

    couplings_test = hamiltonian_adjacency_test.copy()
    local_fields = np.diag(hamiltonian_adjacency_test).copy()
    np.fill_diagonal(couplings_test, 0)

    hamiltonian_adjacency_test_cupy = cp.asarray(hamiltonian_adjacency_test)
    couplings_test_cupy = cp.asarray(couplings_test)
    local_fields_cupy = cp.asarray(local_fields)

    bitstrings_test_numpy = rng_test_numpy.integers(0, 2, (nos_test, noq_test))
    bitstrings_test_cupy = cp.asarray(bitstrings_test_numpy)

    energies_test = em_usra.calculate_energies_from_bitstrings_2_local(bitstrings_array=bitstrings_test_numpy,
                                                                       weights_matrix=hamiltonian_adjacency_test,
                                                                       computation_method=f"numpy_einsum")
    t0 = time.perf_counter()
    energies_test_here = calculate_energies_from_bitstrings_2_local(bitstrings_array=bitstrings_test_numpy,
                                                                    #adjacency_matrix=hamiltonian_adjacency_test,
                                                                    couplings_array=couplings_test,
                                                                    local_fields=local_fields,
                                                                    computation_backend='numpy')
    t1 = time.perf_counter()
    correct = np.allclose(energies_test, energies_test_here)

    if correct:
        anf.cool_print("TEST (numpy) PASSED FOR:", f"{noq_test} qubits", 'green')
    else:
        anf.cool_print("TEST (numpy) FAILED FOR:", f"{noq_test} qubits", 'red')

    energies_test_cupy = em_usra.calculate_energies_from_bitstrings_2_local(bitstrings_array=bitstrings_test_cupy,
                                                                            weights_matrix=hamiltonian_adjacency_test_cupy,
                                                                            computation_method=f"cupy_einsum")


    t2 = time.perf_counter()
    #bitstrings_test_cupy = cp.asarray(bitstrings_test_numpy)

    energies_test_cupy_here = calculate_energies_from_bitstrings_2_local(bitstrings_array=bitstrings_test_cupy,
                                                                         #adjacency_matrix=hamiltonian_adjacency_test_cupy,
                                                                         couplings_array=couplings_test_cupy,
                                                                         local_fields=local_fields_cupy,
                                                                         computation_backend='cupy')
    t3 = time.perf_counter()


    correct_usra = cp.allclose(energies_test_cupy, energies_test_here)
    correct_here = cp.allclose(energies_test_cupy, energies_test_cupy_here)

    if correct_usra:
        anf.cool_print("TEST (cupy-USRA) PASSED FOR:", f"{noq_test} qubits and {nos_test} samples", 'green')
    else:
        anf.cool_print("TEST (cupy-USRA) FAILED FOR:", f"{noq_test} qubits and {nos_test} samples", 'red')

    if correct_here:
        anf.cool_print("TEST (cupy-HERE) PASSED FOR:", f"{noq_test} qubits and {nos_test} samples", 'green')
    else:
        anf.cool_print("TEST (cupy-HERE) FAILED FOR:", f"{noq_test} qubits and {nos_test} samples", 'red')

    print("Time numpy:", t1 - t0)
    print("Time cupy:", t3 - t2)
