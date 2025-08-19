# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 

import numpy as np
cimport numpy as np
cimport cython


ctypedef np.complex128_t COMPLEX128_t
ctypedef np.float32_t FLOAT32_t
ctypedef np.float64_t FLOAT64_t

ctypedef fused floating:
    np.float32_t
    np.float64_t

ctypedef fused complexing:
    np.complex64_t
    np.complex128_t

ctypedef fused arbitrary_type:
    np.float32_t
    np.float64_t
    np.complex64_t
    np.complex128_t


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_cumsum(np.ndarray[arbitrary_type, ndim=1] arr):
    cdef int n = arr.shape[0]
    cdef np.ndarray[arbitrary_type, ndim=1] out = np.zeros(n, dtype=arr.dtype)
    cdef int i
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = out[i-1] + arr[i]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def _cython_abs_squared_complex_float64(np.ndarray[complexing, ndim=1] vector):
    cdef int n = vector.shape[0]
    cdef int i
    cdef np.ndarray[FLOAT64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef FLOAT64_t real_part, imag_part

    for i in range(n):
        real_part = vector[i].real
        imag_part = vector[i].imag
        result[i] = real_part * real_part + imag_part * imag_part

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def _cython_abs_squared_complex_float32(np.ndarray[complexing, ndim=1] vector):
    cdef int n = vector.shape[0]
    cdef int i
    cdef np.ndarray[FLOAT32_t, ndim=1] result = np.zeros(n, dtype=np.float32)
    cdef FLOAT32_t real_part, imag_part

    for i in range(n):
        real_part = vector[i].real
        imag_part = vector[i].imag
        result[i] = real_part * real_part + imag_part * imag_part

    return result



@cython.boundscheck(False)
@cython.wraparound(False)
def _cython_vdot_complex(np.ndarray[complexing, ndim=1] vector1,
                         np.ndarray[complexing, ndim=1] vector2):
    cdef int n = vector1.shape[0]
    cdef int i
    cdef COMPLEX128_t result = 0.0
    cdef FLOAT64_t real_part1, imag_part1, real_part2, imag_part2

    for i in range(n):
        real_part1 = vector1[i].real
        imag_part1 = -vector1[i].imag

        real_part2 = vector2[i].real
        imag_part2 = vector2[i].imag
        result += real_part1 * real_part2 + imag_part1 * imag_part2

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _cython_vdot_real(np.ndarray[floating, ndim=1] vector1,
                      np.ndarray[floating, ndim=1] vector2):
    cdef int n = vector1.shape[0]
    cdef int i
    cdef FLOAT64_t result = 0.0

    for i in range(n):
        result += vector1[i] * vector2[i]

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_abs_squared(vector,
                       output_precision_if_complex=np.float32):
    if vector.dtype in [np.complex64, np.complex128]:
        if output_precision_if_complex == np.float32:
            return _cython_abs_squared_complex_float32(vector)
        elif output_precision_if_complex == np.float64:
            return _cython_abs_squared_complex_float64(vector)
        else:
            raise ValueError("Unsupported dtype for vector.")
    elif vector.dtype in [np.float32, np.float64]:
        return _cython_vdot_real(vector, vector)
    else:
        raise ValueError("Unsupported dtype for vector.")
@cython.boundscheck(False)
@cython.wraparound(False)
def cython_vdot(vector1, vector2):
    if vector1.dtype in [np.complex64, np.complex128] and vector2.dtype in [np.complex64, np.complex128]:
        return _cython_vdot_complex(vector1, vector2)
    elif vector1.dtype in [np.float32, np.float64] and vector2.dtype in [np.float32, np.float64]:
        return _cython_vdot_real(vector1, vector2)
    else:
        raise ValueError("Unsupported dtype for vectors: {} and {}".format(vector1.dtype, vector2.dtype))





#let's replicate the above function in cython

@cython.boundscheck(False)
@cython.wraparound(False)
def get_all_two_1s_bitstrings_cython(int noq,
                                     include_one_1s_bitstrings=False):
    cdef int number_of_pairs = noq*(noq-1)//2

    if include_one_1s_bitstrings:
        number_of_pairs += noq

    cdef np.ndarray[np.int32_t, ndim=2] zeros = np.zeros((number_of_pairs, noq), dtype=np.int32)
    cdef int counter = 0
    cdef int i, j
    for i in range(noq):
        for j in range(i+1,noq):
            zeros[counter, j] = 1
            zeros[counter, i] = 1
            counter += 1
    if include_one_1s_bitstrings:
        for i in range(noq):
            zeros[counter, i] = 1
            counter += 1

    return zeros



#TODO(FBM) make this recursive!
@cython.boundscheck(False)
@cython.wraparound(False)
def get_all_three_1s_bitstrings_cython(int noq,
                                     include_lower_numbers:False):
    cdef int number_of_pairs = noq*(noq-1)*(noq-2)//6
    if include_lower_numbers:
        number_of_pairs += noq*(noq-1)//2 + noq

    cdef np.ndarray[np.int32_t, ndim=2] zeros = np.zeros((number_of_pairs, noq), dtype=np.int32)
    cdef int counter = 0
    cdef int i, j, k
    for i in range(noq):
        for j in range(i+1,noq):
            for k in range(j+1,noq):
                zeros[counter, i] = 1
                zeros[counter, j] = 1
                zeros[counter, k] = 1
                counter += 1
    if include_lower_numbers:
        for i in range(noq):
            for j in range(i+1,noq):
                zeros[counter, i] = 1
                zeros[counter, j] = 1
                counter += 1
        for i in range(noq):
            zeros[counter, i] = 1
            counter += 1

    return zeros
