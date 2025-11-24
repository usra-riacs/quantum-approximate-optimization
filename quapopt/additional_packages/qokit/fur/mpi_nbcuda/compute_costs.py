###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numba.cuda

from quapopt.additional_packages.qokit.fur.nbcuda.fbm_monkey_patch import (
    _GLOBAL_GRID_SIZE,
)


@numba.cuda.jit
def zero_init_kernel(x):
    tid = numba.cuda.grid(_GLOBAL_GRID_SIZE)

    if tid < len(x):
        x[tid] = 0


def zero_init(x):
    zero_init_kernel.forall(len(x))(x)


@numba.cuda.jit
def compute_costs_kernel(costs, coef: float, pos_mask: int, offset: int):
    tid = numba.cuda.grid(_GLOBAL_GRID_SIZE)

    if tid < len(costs):
        parity = numba.cuda.popc((tid + offset) & pos_mask) & 1
        if parity:
            costs[tid] -= coef
        else:
            costs[tid] += coef


def compute_costs(rank: int, n_local_qubits: int, terms, out):
    offset = rank << n_local_qubits
    n = len(out)

    for coef, pos in terms:
        pos_mask = sum(2**x for x in pos)
        compute_costs_kernel.forall(n)(out, coef, pos_mask, offset)
