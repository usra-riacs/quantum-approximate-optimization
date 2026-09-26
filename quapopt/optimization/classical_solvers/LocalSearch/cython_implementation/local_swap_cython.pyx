# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython


cdef void _init_diff_weights(
    const int n,
    const np.int32_t* indptr,
    const np.int32_t* indices,
    const double* data,
    const np.int32_t* assignments,
    double* diff_weights
) noexcept nogil:
    """
    Initialize diff_weights array from CSR adjacency and assignments.

    diff_weights[i] = Σ_{j≠i} w_ij · x_i · x_j  +  h_i · x_i
    where h_i is the self-loop weight (local field), if present.
    """
    cdef int i, j, j_ptr
    cdef double w

    for i in range(n):
        diff_weights[i] = 0.0
        for j_ptr in range(indptr[i], indptr[i + 1]):
            j = indices[j_ptr]
            w = data[j_ptr]
            if j == i:
                # Local field: contributes h_i * x_i
                diff_weights[i] += w * assignments[i]
            else:
                # Coupling: contributes w_ij * x_i * x_j
                diff_weights[i] += w * assignments[i] * assignments[j]


cdef void _update_after_flip(
    const int k,
    const np.int32_t* indptr,
    const np.int32_t* indices,
    const double* data,
    np.int32_t* assignments,
    double* diff_weights
) noexcept nogil:
    """
    Flip node k and incrementally update diff_weights.

    For each neighbor j of k (j ≠ k):
        diff_weights[j] -= 2 · w_kj · x_k · x_j    (using x_k BEFORE flip)

    Then:
        x_k = -x_k
        diff_weights[k] = -diff_weights[k]
    """
    cdef int j, j_ptr
    cdef double w
    cdef np.int32_t old_xk = assignments[k]

    # Update neighbors' diff_weights (before flipping k)
    for j_ptr in range(indptr[k], indptr[k + 1]):
        j = indices[j_ptr]
        w = data[j_ptr]
        if j == k:
            continue  # Self-loop handled by sign flip of diff_weights[k]
        diff_weights[j] -= 2.0 * w * old_xk * assignments[j]

    # Flip assignment and negate diff_weights
    assignments[k] = -old_xk
    diff_weights[k] = -diff_weights[k]


cdef int _all1swap(
    const int n,
    const np.int32_t* indptr,
    const np.int32_t* indices,
    const double* data,
    np.int32_t* assignments,
    double* diff_weights,
    const double tolerance
) noexcept nogil:
    """
    First-improvement 1-swap local search.

    Repeatedly scans nodes; flips the first node with diff_weights[i] > tolerance.
    Returns the total number of flips performed.
    """
    cdef bint move_made = True
    cdef int i
    cdef int total_flips = 0

    while move_made:
        move_made = False
        for i in range(n):
            if diff_weights[i] > tolerance:
                _update_after_flip(i, indptr, indices, data,
                                   assignments, diff_weights)
                total_flips += 1
                move_made = True
                break

    return total_flips


cdef int _all2swap(
    const int n,
    const np.int32_t* indptr,
    const np.int32_t* indices,
    const double* data,
    np.int32_t* assignments,
    double* diff_weights,
    const double tolerance
) noexcept nogil:
    """
    First-improvement 2-swap local search.

    Iterates over edges (i,j) from CSR. For each edge, computes:
        benefit = diff_weights[i] + diff_weights[j] - 2 · x_i · x_j · w_ij

    Flips both i and j if benefit > tolerance.
    Returns the total number of 2-swaps performed.
    """
    cdef bint move_made = True
    cdef int i, j, j_ptr
    cdef double w, benefit
    cdef int total_swaps = 0

    while move_made:
        move_made = False
        for i in range(n):
            for j_ptr in range(indptr[i], indptr[i + 1]):
                j = indices[j_ptr]
                if j <= i:
                    # Skip self-loops and lower triangle (avoid checking each edge twice)
                    continue
                w = data[j_ptr]
                benefit = (diff_weights[i] + diff_weights[j]
                           - 2.0 * assignments[i] * assignments[j] * w)
                if benefit > tolerance:
                    _update_after_flip(i, indptr, indices, data,
                                       assignments, diff_weights)
                    _update_after_flip(j, indptr, indices, data,
                                       assignments, diff_weights)
                    total_swaps += 1
                    move_made = True
                    break
            if move_made:
                break

    return total_swaps


cdef int _single_best_swap(
    const int n,
    const np.int32_t* indptr,
    const np.int32_t* indices,
    const double* data,
    np.int32_t* assignments,
    double* diff_weights,
    const double tolerance_1swap,
    const double tolerance_2swap
) noexcept nogil:
    """
    Best-improvement single-step search over 1-flips and 2-flips.

    Computes the benefit of every 1-flip (when tolerance_1swap >= 0) and every
    2-flip on an edge (when tolerance_2swap >= 0), then applies the single move
    with the largest benefit, provided it exceeds its respective tolerance.

    Returns the number of nodes flipped (0 if no move applied, 1 for a 1-flip,
    or 2 for a 2-flip).
    """
    cdef int i, j, j_ptr
    cdef double w, benefit
    cdef double best_benefit = 0.0
    cdef int best_i = -1
    cdef int best_j = -1  # -1 -> chosen move is a 1-flip; otherwise 2-flip partner

    if tolerance_1swap >= 0:
        for i in range(n):
            benefit = diff_weights[i]
            if benefit > tolerance_1swap and benefit > best_benefit:
                best_benefit = benefit
                best_i = i
                best_j = -1

    if tolerance_2swap >= 0:
        for i in range(n):
            for j_ptr in range(indptr[i], indptr[i + 1]):
                j = indices[j_ptr]
                if j <= i:
                    # Skip self-loops and lower triangle (avoid checking each edge twice)
                    continue
                w = data[j_ptr]
                benefit = (diff_weights[i] + diff_weights[j]
                           - 2.0 * assignments[i] * assignments[j] * w)
                if benefit > tolerance_2swap and benefit > best_benefit:
                    best_benefit = benefit
                    best_i = i
                    best_j = j

    if best_i < 0:
        return 0

    _update_after_flip(best_i, indptr, indices, data, assignments, diff_weights)
    if best_j < 0:
        return 1

    _update_after_flip(best_j, indptr, indices, data, assignments, diff_weights)
    return 2


def run_exhaustive_swap_choose_first_cython(
    np.ndarray[np.int32_t, ndim=1] bitstring_pm,
    np.ndarray[np.int32_t, ndim=1] csr_indptr,
    np.ndarray[np.int32_t, ndim=1] csr_indices,
    np.ndarray[np.float64_t, ndim=1] csr_data,
    double tolerance_1swap = 0.02,
    double tolerance_2swap = 0.2,
):
    """
    Run first-improvement local search (All1Swap + All2Swap) using CSR adjacency.

    Cython implementation of the first-improvement local sweep.

    :param bitstring_pm: ±1 assignment array (will be copied, not modified in-place).
    :param csr_indptr: CSR row pointer array (length n+1).
    :param csr_indices: CSR column indices array.
    :param csr_data: CSR data (edge weights) array.
    :param tolerance_1swap: Minimum improvement for 1-swap (MaxCut diff_weights space).
    :param tolerance_2swap: Minimum improvement for 2-swap (MaxCut diff_weights space).
    :return: Tuple of (improved_assignments, n_1swaps, n_2swaps).
    """
    cdef int n = csr_indptr.shape[0] - 1

    # Work on a copy
    cdef np.ndarray[np.int32_t, ndim=1] assignments = bitstring_pm.copy()
    cdef np.ndarray[np.float64_t, ndim=1] diff_weights = np.empty(n, dtype=np.float64)

    # Raw pointers for nogil access
    cdef np.int32_t* p_indptr = &csr_indptr[0]
    cdef np.int32_t* p_indices = &csr_indices[0]
    cdef double* p_data = &csr_data[0]
    cdef np.int32_t* p_assignments = &assignments[0]
    cdef double* p_diff_weights = &diff_weights[0]

    cdef int n_1swaps, n_2swaps

    # Initialize diff_weights from scratch
    _init_diff_weights(n, p_indptr, p_indices, p_data, p_assignments, p_diff_weights)

    # Run sweeps
    if tolerance_1swap>=0:
        n_1swaps = _all1swap(n, p_indptr, p_indices, p_data,
                             p_assignments, p_diff_weights, tolerance_1swap)
    if tolerance_2swap>=0:
        n_2swaps = _all2swap(n, p_indptr, p_indices, p_data,
                             p_assignments, p_diff_weights, tolerance_2swap)

    return assignments


def run_single_swap_choose_best_cython(
    np.ndarray[np.int32_t, ndim=1] bitstring_pm,
    np.ndarray[np.int32_t, ndim=1] csr_indptr,
    np.ndarray[np.int32_t, ndim=1] csr_indices,
    np.ndarray[np.float64_t, ndim=1] csr_data,
    double tolerance_1swap = 0.0,
    double tolerance_2swap = 0.0,
):
    """
    Run a single best-improvement local search step using CSR adjacency.

    Examines all 1-flips and all 2-flips (over edges) and applies the single
    move with the largest benefit, provided it exceeds its tolerance. At most
    one move is applied per call (no greedy loop).

    :param bitstring_pm: ±1 assignment array (will be copied, not modified in-place).
    :param csr_indptr: CSR row pointer array (length n+1).
    :param csr_indices: CSR column indices array.
    :param csr_data: CSR data (edge weights) array.
    :param tolerance_1swap: Minimum improvement for a 1-flip; set < 0 to disable 1-flips.
    :param tolerance_2swap: Minimum improvement for a 2-flip; set < 0 to disable 2-flips.
    :return: Updated assignment array (with at most one move applied).
    """
    cdef int n = csr_indptr.shape[0] - 1

    # Work on a copy
    cdef np.ndarray[np.int32_t, ndim=1] assignments = bitstring_pm.copy()
    cdef np.ndarray[np.float64_t, ndim=1] diff_weights = np.empty(n, dtype=np.float64)

    # Raw pointers for nogil access
    cdef np.int32_t* p_indptr = &csr_indptr[0]
    cdef np.int32_t* p_indices = &csr_indices[0]
    cdef double* p_data = &csr_data[0]
    cdef np.int32_t* p_assignments = &assignments[0]
    cdef double* p_diff_weights = &diff_weights[0]

    # Initialize diff_weights from scratch
    _init_diff_weights(n, p_indptr, p_indices, p_data, p_assignments, p_diff_weights)

    # Single best-improvement step (over both 1-flips and 2-flips)
    _single_best_swap(n, p_indptr, p_indices, p_data,
                      p_assignments, p_diff_weights,
                      tolerance_1swap, tolerance_2swap)

    return assignments