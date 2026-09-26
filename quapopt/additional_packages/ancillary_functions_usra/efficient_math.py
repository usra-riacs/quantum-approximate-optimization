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
from quapopt import AVAILABLE_SIMULATORS
from quapopt.precision import resolve_precision
if 'cupy' in AVAILABLE_SIMULATORS:
    import cupy as cp
else:
    import numpy as cp
import numpy as np

def convert_cupy_numpy_array(array:np.ndarray|cp.ndarray,
                             output_backend:str)->np.ndarray|cp.ndarray:
    from quapopt import AVAILABLE_SIMULATORS
    if 'cupy' not in AVAILABLE_SIMULATORS:
        # Lists and tuples become numpy arrays, as they do through cp.asnumpy when cupy is installed.
        if isinstance(array, (list, tuple)):
            return np.asarray(array)
        return array
    if output_backend == 'numpy':
        if isinstance(array, (cp.ndarray,list,tuple)):
            return cp.asnumpy(array)
        elif isinstance(array,np.ndarray):
            return array
        else:
            raise ValueError(f'array should be np.nddarray/cp.nddarray/list/tuple, not {type(array)}')
    elif output_backend == 'cupy':
        if isinstance(array, (np.ndarray,list,tuple)):
            return cp.asarray(array)
        elif isinstance(array,cp.ndarray):
            return array
        else:
            raise ValueError(f'array should be np.nddarray/cp.nddarray/list/tuple, not {type(array)}')
    else:
        raise ValueError(f'output_backend should be either "numpy" or "cupy", not {output_backend}')


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
    # The running sum must be float64: in float32, near 0.5 its spacing (6e-8) is of the order
    # of a single outcome's probability at n=20, which distorts outcomes by up to 20 % and
    # gives some none at all.
    cumprobs = np.cumsum(probabilities, dtype=np.float64)
    t2 = time.perf_counter()
    # this returns the indices
    samples_integers = np.searchsorted(cumprobs, numpy_rng.random(number_of_samples))
    # A uniform above the last cumulative value (rounding of the total mass) would return
    # 2^n, which the bit expansion below turns into the all-zeros row.
    np.minimum(samples_integers, len(probabilities) - 1, out=samples_integers)
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


if 'cupy' in AVAILABLE_SIMULATORS:
    _ABS_SQUARED_KERNEL = None

    def _abs_squared_kernel():
        """|x|^2 in one pass, float64 out, for complex64 or complex128 input."""
        global _ABS_SQUARED_KERNEL
        if _ABS_SQUARED_KERNEL is None:
            _ABS_SQUARED_KERNEL = cp.ElementwiseKernel(
                'T x', 'float64 p',
                'p = (double)x.real() * (double)x.real() + (double)x.imag() * (double)x.imag();',
                'quapopt_abs_squared_to_f64')
        return _ABS_SQUARED_KERNEL

    def _sample_indices_cupy(statevector, number_of_samples, numpy_rng):
        """Inverse-CDF sampling on the device. One float64 buffer of the state's size; the
        statevector is read once and never copied to the host. The uniforms come from the
        caller's numpy generator, so a seed means the same thing on both paths."""
        flat = statevector.ravel()
        cumulative = cp.empty(flat.size, dtype=cp.float64)
        _abs_squared_kernel()(flat, cumulative)
        cp.cumsum(cumulative, out=cumulative)
        if isinstance(numpy_rng, cp.random.Generator):
            uniforms = numpy_rng.random(number_of_samples, dtype=cp.float64)
        else:
            uniforms = cp.asarray(numpy_rng.random(number_of_samples))
        indices = cp.searchsorted(cumulative, uniforms)
        cp.minimum(indices, flat.size - 1, out=indices)
        return indices


def indices_to_bitstrings(indices, number_of_qubits: int):
    """0/1 rows from flat indices; column j is qubit j, the (n-1-j)-th bit. numpy or cupy in,
    same module out."""
    xp = cp.get_array_module(indices) if 'cupy' in AVAILABLE_SIMULATORS else np
    weights = 1 << xp.arange(number_of_qubits - 1, -1, -1)
    return ((indices[:, None] & weights) > 0).astype(xp.int64)


_DEVICE_SAMPLING_MARGIN = 64 * 2 ** 20


def _device_sampling_fits(number_of_qubits: int, number_of_samples: int, free_bytes: int,
                          state_bytes: int = 0) -> bool:
    """Whether the device route's peak fits in `free_bytes` with a 64 MiB margin.

    The peak beyond a statevector already on the device is the float64 cumulative buffer
    (8 * 2^n bytes) plus, per drawn row, the int64 bitstring row and the energy evaluator's
    temporaries (32 * n + 24 bytes). A host statevector is uploaded first, so its own size
    is passed as `state_bytes`."""
    need = state_bytes + 8 * 2 ** number_of_qubits + number_of_samples * (32 * number_of_qubits + 24)
    return need + _DEVICE_SAMPLING_MARGIN <= free_bytes


def _device_route_is_faster(number_of_qubits: int, number_of_samples: int) -> bool:
    """Whether a host statevector is expected to be sampled faster through the device than on
    the host sampler.

    Measured host time / device time on an RTX 3060 Laptop GPU (6 GB), n = 8..20, 1e2..1e5
    shots, for the sampler runner's draw-and-evaluate step on a dense and a sparse instance,
    complex128 and complex64: up to n = 16 the device route costs about 1.4 ms (1.2-3.0) at
    1e2 shots, so it is faster from 1e4 shots at any n (1.04-1.25 at n = 8), from n = 17 at 3e3
    shots (1.12-1.21; 0.82-0.99 at n = 15-16), from n = 18 at 1e3 shots (1.15-1.52; 0.81-0.85
    at n = 17), and from n = 19 at any number of shots (1.42-1.62 at 1e2; 0.81-1.16 at n = 18).
    Sampling the rows alone crosses about one qubit earlier, by at most 0.2 ms."""
    if number_of_samples >= 10 ** 4 or number_of_qubits >= 19:
        return True
    if number_of_qubits == 18:
        return number_of_samples >= 10 ** 3
    if number_of_qubits == 17:
        return number_of_samples >= 3 * 10 ** 3
    return False


def _device_route(statevector, number_of_samples: int, prefer_device: Optional[bool], free_device_bytes=None):
    """The statevector on the device when sampling there is wanted and fits; None otherwise.
    `prefer_device` None sends a host statevector to the device only where
    `_device_route_is_faster`; True sends it whenever the route fits; False never. A cupy
    statevector takes the device route unless `prefer_device` is False. A host statevector
    is uploaded here, once."""
    if 'cupy' not in AVAILABLE_SIMULATORS or prefer_device is False:
        return None
    on_device = isinstance(statevector, cp.ndarray)
    number_of_qubits = int(np.log2(statevector.size))
    if prefer_device is None and not on_device and not _device_route_is_faster(number_of_qubits, number_of_samples):
        return None
    if free_device_bytes is None:
        # Driver-free plus the pool's cached blocks: the pool serves a repeat call from
        # blocks the driver still counts as taken.
        free_device_bytes = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
    state_bytes = 0 if on_device else statevector.nbytes
    if not _device_sampling_fits(number_of_qubits, number_of_samples, free_device_bytes, state_bytes):
        return None
    return statevector if on_device else cp.asarray(np.ascontiguousarray(statevector).ravel())


def sample_indices_from_statevector(statevector, number_of_samples: int, numpy_rng=None,
                                    free_device_bytes=None, prefer_device: Optional[bool] = None):
    """Flat indices of `number_of_samples` draws from |statevector|^2, computed in float64.

    With cupy present a cupy statevector is sampled on the device, and so is a host
    statevector where the device is measured to be faster (`_device_route_is_faster`), at any
    number of shots; the indices come back as a cupy array, and a host statevector is
    uploaded once for it. `prefer_device=True` sends a host statevector to the device at any
    size, and False keeps every statevector on the host route. When the device route's peak,
    the upload included, does not fit the free device memory (`_device_sampling_fits`), the
    host route runs on a host copy and returns a numpy array. `free_device_bytes` exists for
    tests; left None it is read from the device."""
    if numpy_rng is None:
        numpy_rng = np.random.default_rng()
    device_state = _device_route(statevector, number_of_samples, prefer_device, free_device_bytes)
    if device_state is not None:
        return _sample_indices_cupy(device_state, number_of_samples, numpy_rng)
    host_state = np.asarray(convert_cupy_numpy_array(statevector, output_backend='numpy')).ravel()
    probabilities = cython_abs_squared(host_state, output_precision_if_complex=np.float64)
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    # A cupy generator draws on the device; the host route needs its uniforms on the host.
    uniforms = convert_cupy_numpy_array(numpy_rng.random(number_of_samples), output_backend='numpy')
    indices = np.searchsorted(cumulative, uniforms)
    np.minimum(indices, len(probabilities) - 1, out=indices)
    return indices


def sample_from_statevector(statevector: np.ndarray,
                            number_of_samples: int,
                            numpy_rng: np.random.Generator = None,
                            sampling_method='auto',
                            prefer_device: Optional[bool] = None) -> np.ndarray:
    """Host int64 array of shape (number_of_samples, n) drawn from |statevector|^2.

    The draws are made on the device whenever `sample_indices_from_statevector` would make
    them there (cupy present, `prefer_device` and the size as described there, the route
    fits), by inverse-CDF sampling at any number of shots; the rows are expanded on the
    device and copied down. Otherwise
    the host route runs with `sampling_method` ('auto': searchsorted up to 1e5 shots,
    multinomial above); a cupy generator, which the host multinomial cannot take, draws by
    searchsorted at any number of shots. Up to 1e5 shots both routes draw the same rows from
    the same seed, both working in float64; above 1e5 the host route's multinomial draws
    differ.
    """
    if numpy_rng is None:
        numpy_rng = np.random.default_rng()
    number_of_qubits = int(np.log2(statevector.size))
    device_state = _device_route(statevector, number_of_samples, prefer_device)
    if device_state is not None:
        indices = _sample_indices_cupy(device_state, number_of_samples, numpy_rng)
        return cp.asnumpy(indices_to_bitstrings(indices, number_of_qubits))
    if 'cupy' in AVAILABLE_SIMULATORS and isinstance(numpy_rng, cp.random.Generator):
        indices = sample_indices_from_statevector(statevector, number_of_samples, numpy_rng, prefer_device=False)
        return indices_to_bitstrings(indices, number_of_qubits)

    probabilities = cython_abs_squared(convert_cupy_numpy_array(array=statevector,
                                                                output_backend='numpy'),
                                       output_precision_if_complex=np.float64)
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
        if bitstrings_array.dtype.kind == 'u':
            # An unsigned array wraps under 1 - 2 * bits (a uint8 1 becomes 255); int8 holds -1 and +1.
            bitstrings_array = bitstrings_array.astype(np.int8)
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

    if comp_array.dtype.kind == 'u':
        # An unsigned parity wraps under 1 - 2 * parity (a uint8 1 becomes 255); int8 holds -1 and +1.
        comp_array = comp_array.astype(np.int8)

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




def solve_hamiltonian_python(hamiltonian, precision=None):
    """The spectrum of a 2-local Hamiltonian as a float64 host array.

    The spectrum is float64 under both settings (its consumers are sums and the phase
    argument); the argument is validated and does not change the output."""
    resolve_precision(precision)
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






def get_correlators_from_probability_distribution(probability_distribution:Union[cp.ndarray,np.ndarray],
                                                  cost_hamiltonian):
    #TODO(FBM): write an efficient version of this

    number_of_qubits = cost_hamiltonian.number_of_qubits

    cost_hamiltonian = cost_hamiltonian.hamiltonian

    if isinstance(probability_distribution, np.ndarray):
        _bck = np
    else:
        _bck = cp



    marginals = get_marginals_from_probability_distribution(probability_distribution=probability_distribution,
                                                               subsets=[x for _, x in cost_hamiltonian],
                                                               number_of_qubits=number_of_qubits)

    z_operators = {1: _bck.array([1, -1], dtype=np.float32),
                   2: _bck.array([1, -1, -1, 1], dtype=np.float32)}
    correlators = _bck.zeros((number_of_qubits, number_of_qubits), dtype=float)

    for coeff, subset in cost_hamiltonian:
        marginal_ij = marginals[subset]
        if len(subset) == 1:
            qi = subset[0]
            qj = qi
        elif len(subset) == 2:
            qi, qj = subset
        else:
            raise ValueError('Only 1 or 2 qubits subsets are supported')

        z_operator = z_operators[len(subset)]

        marginal_ij = marginal_ij.reshape(2 ** len(subset))
        correlators[qi, qj] = coeff * _bck.dot(z_operator, marginal_ij)

    if isinstance(correlators, np.ndarray):
        pass
    else:
        correlators = cp.asnumpy(correlators)

    return correlators



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
