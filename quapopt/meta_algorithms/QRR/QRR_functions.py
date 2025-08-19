# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
 
# import all types from typing
from typing import Union

import numpy as np

#Lazy monkey-patching of cupy
try:
    import cupy as cp
except(ImportError,ModuleNotFoundError):
    import numpy as cp



def add_weights_to_correlations_matrix(correlations_matrix: np.ndarray,
                                       hamiltonian_list: list) -> np.ndarray:
    for coeff, subset in hamiltonian_list:
        qi, qj = subset
        correlations_matrix[qi, qj] *= coeff
        correlations_matrix[qj, qi] *= coeff

    return correlations_matrix


def solve_correlations_matrix(correlations_matrix: Union[np.ndarray, cp.ndarray],
                              solver: str = 'cupy',
                              UPLO: str = 'U'):
    if solver == 'numpy':
        cupy_input = False
        if not isinstance(correlations_matrix, np.ndarray):
            cupy_input = True
            correlations_matrix = cp.asnumpy(correlations_matrix)

        eigvals, eigvecs = np.linalg.eigh(correlations_matrix,
                                          UPLO=UPLO)
        if cupy_input:
            eigvecs = cp.asarray(eigvecs)

    elif solver == 'cupy':
        cupy_input = True
        if not isinstance(correlations_matrix, cp.ndarray):
            cupy_input = False
            correlations_matrix = cp.asarray(correlations_matrix)

        eigvals, eigvecs = cp.linalg.eigh(correlations_matrix,
                                          UPLO=UPLO)

        if not cupy_input:
            eigvecs = cp.asnumpy(eigvecs)
    else:
        raise ValueError('Solver not recognized. Please use either numpy or cupy.')

    candidate_solutions = eigvecs.T

    return candidate_solutions, eigvals


def find_candidate_solutions_QRR(correlations_matrix: Union[np.ndarray, cp.ndarray],
                                 sign_threshold=0.0,
                                 solver: str = 'cupy',
                                 UPLO: str = 'U',
                                 return_pm_output: bool = False,
                                 return_eigenvectors: bool = False) -> Union[np.ndarray, cp.ndarray]:
    """
    Implements Quantum Relax and Round (QRR) algorithm based on a matrix of 2-local correlations.
    See Ref [1] for more details.
    [1] TODO(FBM): add Maxime's paper

    #TODO(FBM): is eigenvalue propotional to the solution quality?

    """
    candidate_solutions, eigvals = solve_correlations_matrix(correlations_matrix=correlations_matrix,
                                                            solver=solver,
                                                            UPLO=UPLO)

    if return_eigenvectors:
        return candidate_solutions, eigvals

    if return_pm_output:
        state_0 = -1
        state_1 = 1
    else:
        state_0 = 0
        state_1 = 1

    candidate_solutions[candidate_solutions <= sign_threshold] = state_0
    candidate_solutions[candidate_solutions > sign_threshold] = state_1

    return candidate_solutions



def sample_from_eigenvector_qrr_star(eigenvector:Union[np.ndarray,cp.ndarray],
                                     number_of_samples:int):

    # eigenvector_abs = np.abs(eigenvector)
    #negative_values = eigenvector <= 0
    positive_values = eigenvector > 0

    if isinstance(eigenvector, cp.ndarray):
        bck = cp
    else:
        bck = np

    #We will treat value at each position as probability of sampling 0 at that position for negative values, and of sampling
    #1 at that position for positive values
    #we want to keep them all in single vector, so we will take 1-x for positive values
    eigenvector_abs = bck.abs(eigenvector)


    samples = bck.random.binomial(1, eigenvector_abs, size=(number_of_samples, len(eigenvector)))
    return samples

#



























if __name__ == '__main__':

    test_bts = (1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1,)
    noq_test = len(test_bts)
    correlations_matrix = np.zeros((noq_test, noq_test))

    for i in range(noq_test):
        bi = test_bts[i]

        if bi == 0:
            correlations_matrix[i, i] = 1
        else:
            correlations_matrix[i, i] = -1

        for j in range(i + 1, noq_test):
            bj = test_bts[j]
            if bj == bi:
                correlations_matrix[i, j] = 1
                correlations_matrix[j, i] = 1
            else:
                correlations_matrix[i, j] = -1
                correlations_matrix[j, i] = -1

    candidate_solutions = find_candidate_solutions_QRR(correlations_matrix=correlations_matrix,
                                                       sign_threshold=0.0)
    candidate_solutions = [tuple(x) for x in candidate_solutions]
    negation_bts = tuple([1 - x for x in test_bts])

    print(test_bts in candidate_solutions)

    print(candidate_solutions)
