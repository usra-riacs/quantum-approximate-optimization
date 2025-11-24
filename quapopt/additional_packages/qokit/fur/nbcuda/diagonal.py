###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import math

import numba.cuda

from quapopt.additional_packages.qokit.fur.nbcuda.fbm_monkey_patch import (
    _GLOBAL_GRID_SIZE,
)


@numba.cuda.jit
def apply_diagonal_kernel(sv, gamma, diag):
    n = len(sv)
    tid = numba.cuda.grid(_GLOBAL_GRID_SIZE)
    if tid < n:
        x = 0.5 * gamma * diag[tid]

        sv[tid] *= math.cos(x) - 1j * math.sin(x)


def apply_diagonal(sv, gamma, diag):
    apply_diagonal_kernel.forall(len(sv))(sv, gamma, diag)
