# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)

"""In-place CUDA kernels for the cupy branch of QAOASimulatorPython.

The phase separator and the X/Z rotation kernel modify a C-contiguous complex64 or
complex128 cupy array in place and allocate no device memory. The X/Z kernel computes in
the array's own scalar type. The phase separator takes a float64 spectrum: its argument
-angle * E is formed in double and reduced modulo 2 pi before the single- or
double-precision sincos. `abs_squared_float64` returns a new float64 array. The X/Z rotation kernel applies k bits per launch,
each bit with its own (cos b, sin b * term_X, sin b * term_Z); the vanilla mixer is
term_X = 1, term_Z = 0. Kernels are compiled on first use and cached per scalar type and k.
"""
import math

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None

_THREADS_PER_BLOCK = 256
_SCALAR = {'complex64': 'float', 'complex128': 'double'}
_phase_kernels = {}
_xz_kernels = {}
_abs_squared_kernel = None
_weighted_sums_kernel = None
_expectation_kernel = None
# Row count of the first reduction in probability_weighted_sums and expectation_value: of 256, 1024 and 4096 rows,
# 256 was the fastest at n = 20 and within 3 % of the fastest at n = 24 on this laptop.
_WEIGHTED_SUMS_ROWS = 256

_XZ_SRC = r'''
#include <cupy/complex.cuh>
extern "C" __global__
void %(name)s(complex<%(T)s>* __restrict__ x,
              %(params)s,
              const long long n_groups)
{
    const long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= n_groups) return;
    const int bits[%(K)d] = {%(bit_names)s};
    long long stride[%(K)d];
    long long base = tid;
    #pragma unroll
    for (int j = 0; j < %(K)d; ++j) {
        const int b = bits[j];
        const long long low = base & ((1LL << b) - 1);
        base = ((base >> b) << (b + 1)) | low;
        stride[j] = 1LL << b;
    }
    const %(T)s cc[%(K)d] = {%(c_names)s};
    const %(T)s sx[%(K)d] = {%(sx_names)s};
    const %(T)s sz[%(K)d] = {%(sz_names)s};
    %(T)s vr[%(DIM)d], vi[%(DIM)d];
    #pragma unroll
    for (int s = 0; s < %(DIM)d; ++s) {
        long long off = 0;
        #pragma unroll
        for (int j = 0; j < %(K)d; ++j) if ((s >> j) & 1) off += stride[j];
        const complex<%(T)s> z = x[base + off];
        vr[s] = z.real(); vi[s] = z.imag();
    }
    #pragma unroll
    for (int j = 0; j < %(K)d; ++j) {
        const %(T)s c = cc[j], x_ = sx[j], z_ = sz[j];
        #pragma unroll
        for (int s = 0; s < %(DIM)d; ++s) {
            if ((s >> j) & 1) continue;
            const int t = s | (1 << j);
            const %(T)s ar = vr[s], ai = vi[s], br = vr[t], bi = vi[t];
            vr[s] = c * ar + z_ * ai + x_ * bi;
            vi[s] = c * ai - z_ * ar - x_ * br;
            vr[t] = x_ * ai + c * br - z_ * bi;
            vi[t] = -x_ * ar + c * bi + z_ * br;
        }
    }
    #pragma unroll
    for (int s = 0; s < %(DIM)d; ++s) {
        long long off = 0;
        #pragma unroll
        for (int j = 0; j < %(K)d; ++j) if ((s >> j) & 1) off += stride[j];
        x[base + off] = complex<%(T)s>(vr[s], vi[s]);
    }
}
'''


def _require_cupy():
    if cp is None:
        raise ImportError("quapopt.optimization.QAOA.simulation.direct.cupy_kernels needs the 'cupy' "
                          "package; the python simulator's backend='cupy' cannot run its kernels without it. "
                          "Install cupy for this CUDA toolkit, or use backend='cython' / 'numpy'.")


def _scalar_type(state):
    """'float' or 'double' for a C-contiguous complex64 / complex128 cupy array; raises otherwise."""
    name = str(state.dtype)
    if name not in _SCALAR:
        raise TypeError(f"state must be a complex64 or complex128 cupy array, got {state.dtype}; "
                        f"cast with state.astype(cp.complex64) or cp.complex128 first")
    if not state.flags.c_contiguous:
        raise TypeError("state must be C-contiguous (cp.ascontiguousarray)")
    return _SCALAR[name]


def _xz_source(scalar, k):
    bit_params = ', '.join(f'const int b{j}' for j in range(k))
    params = ', '.join(f'const double c{j}, const double x{j}, const double z{j}' for j in range(k))
    return _XZ_SRC % dict(name=f'xz_{scalar}_{k}', T=scalar, K=k, DIM=1 << k,
                          params=params + ', ' + bit_params,
                          bit_names=', '.join(f'b{j}' for j in range(k)),
                          c_names=', '.join(f'({scalar})c{j}' for j in range(k)),
                          sx_names=', '.join(f'({scalar})x{j}' for j in range(k)),
                          sz_names=', '.join(f'({scalar})z{j}' for j in range(k)))


def _get_xz_kernel(scalar, k):
    if (scalar, k) not in _xz_kernels:
        _require_cupy()
        _xz_kernels[(scalar, k)] = cp.RawKernel(_xz_source(scalar, k), f'xz_{scalar}_{k}')
    return _xz_kernels[(scalar, k)]


def _bit_groups(number_of_bits, k):
    """Interleaved groups of at most k bits: bit b goes to group b mod G, G = ceil(n / k)."""
    groups = math.ceil(number_of_bits / k)
    return [sorted(b for b in range(number_of_bits) if b % groups == i) for i in range(groups)]


def apply_xz_rotations_inplace(state, params_per_bit, k=4):
    """Apply, to every bit b of the flat index, the 2x2 matrix
    [[c - i sz, -i sx], [-i sx, c + i sz]] with (c, sx, sz) = params_per_bit[b], in place,
    k bits per launch. `state`: C-contiguous complex64 or complex128 cupy array of size
    2 ** len(params_per_bit); the arithmetic runs in the state's own scalar type."""
    scalar = _scalar_type(state)
    number_of_bits = len(params_per_bit)
    assert state.size == 1 << number_of_bits, (
        f"state has {state.size} amplitudes, expected 2**{number_of_bits}")
    for group in _bit_groups(number_of_bits, k):
        width = len(group)
        n_groups = state.size >> width
        blocks = (n_groups + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        args = [state]
        args += [np.float64(value) for bit in group for value in params_per_bit[bit]]
        args += [np.int32(bit) for bit in group]
        args += [np.int64(n_groups)]
        _get_xz_kernel(scalar, width)((blocks,), (_THREADS_PER_BLOCK,), tuple(args))
    return state


def _get_phase_kernel(scalar):
    if scalar not in _phase_kernels:
        _require_cupy()
        if scalar == 'float':
            body = ('double arg = remainder(-angle_PS * spectrum, 6.283185307179586);'
                    ' float s, c; sincosf((float)arg, &s, &c);'
                    ' state = state * complex<float>(c, s);')
            out_param = 'complex64 state'
        else:
            body = ('double arg = remainder(-angle_PS * spectrum, 6.283185307179586);'
                    ' double s, c; sincos(arg, &s, &c);'
                    ' state = state * complex<double>(c, s);')
            out_param = 'complex128 state'
        _phase_kernels[scalar] = cp.ElementwiseKernel('float64 spectrum, float64 angle_PS', out_param,
                                                      body, f'quapopt_phase_separator_{scalar}')
    return _phase_kernels[scalar]


def apply_phase_separator_inplace(state, angle_PS, spectrum):
    """state[i] *= exp(-1j * angle_PS * spectrum[i]) in place and return it. `spectrum` is a
    float64 cupy array (the policy stores spectra as float64 under both settings); the phase
    argument is formed in double, reduced to [-pi, pi], and evaluated by sincosf on a complex64
    state or sincos on a complex128 one."""
    scalar = _scalar_type(state)
    if spectrum.dtype != cp.float64:
        raise TypeError(f"spectrum must be float64, got {spectrum.dtype}; the simulator casts spectra "
                        f"in update_batches_spectra, so a direct caller casts with spectrum.astype(cp.float64)")
    _get_phase_kernel(scalar)(spectrum, float(angle_PS), state)
    return state


def probability_weighted_sums(state, spectrum):
    """(sum_i |state_i|^2 * spectrum_i, sum_i |state_i|^2) as two Python floats, from one pass
    over a complex64 or complex128 state and a float64 spectrum. Each square is formed in double
    and both sums are float64; the pair is carried as one complex128 accumulator (real part: the
    weighted sum, imaginary part: the norm).

    The pass reduces _WEIGHTED_SUMS_ROWS rows, one block each, and a second small reduction adds
    the row sums: a ReductionKernel reduced to one scalar runs on a single block, 2.9 ms against
    0.20 ms at n = 20 on this laptop. The order of both reductions is fixed by the shape, so the
    result is the same on every call."""
    global _weighted_sums_kernel
    _scalar_type(state)
    if spectrum.dtype != cp.float64:
        raise TypeError(f"spectrum must be float64, got {spectrum.dtype}; cast with spectrum.astype(cp.float64)")
    if _weighted_sums_kernel is None:
        _require_cupy()
        _weighted_sums_kernel = cp.ReductionKernel(
            'T state, float64 spectrum', 'complex128 out',
            'complex<double>(((double)state.real() * (double)state.real()'
            ' + (double)state.imag() * (double)state.imag()) * spectrum,'
            ' (double)state.real() * (double)state.real() + (double)state.imag() * (double)state.imag())',
            'a + b', 'out = a', 'complex<double>(0.0, 0.0)',
            'quapopt_probability_weighted_sums')
    rows = math.gcd(state.size, _WEIGHTED_SUMS_ROWS)
    row_sums = _weighted_sums_kernel(state.reshape(rows, -1), spectrum.reshape(rows, -1), axis=1)
    sums = complex(row_sums.sum())
    return sums.real, sums.imag


def expectation_value(state, spectrum):
    """sum_i |state_i|^2 * spectrum_i as a Python float, from one pass over a complex64 or
    complex128 state and a float64 spectrum: probability_weighted_sums without the norm, for a
    caller that does not divide by it. Each square is formed in double and the sum is float64;
    the row split and its fixed order are those of probability_weighted_sums."""
    global _expectation_kernel
    _scalar_type(state)
    if spectrum.dtype != cp.float64:
        raise TypeError(f"spectrum must be float64, got {spectrum.dtype}; cast with spectrum.astype(cp.float64)")
    if _expectation_kernel is None:
        _require_cupy()
        _expectation_kernel = cp.ReductionKernel(
            'T state, float64 spectrum', 'float64 out',
            '((double)state.real() * (double)state.real()'
            ' + (double)state.imag() * (double)state.imag()) * spectrum',
            'a + b', 'out = a', '0.0',
            'quapopt_expectation_value')
    rows = math.gcd(state.size, _WEIGHTED_SUMS_ROWS)
    row_sums = _expectation_kernel(state.reshape(rows, -1), spectrum.reshape(rows, -1), axis=1)
    return float(row_sums.sum())


def abs_squared_float64(state):
    """|state|^2 as a float64 cupy array from a complex64 or complex128 one, each square formed in double."""
    global _abs_squared_kernel
    _scalar_type(state)
    if _abs_squared_kernel is None:
        _require_cupy()
        _abs_squared_kernel = cp.ElementwiseKernel(
            'T state', 'float64 out',
            'double re = (double)state.real(); double im = (double)state.imag(); out = re * re + im * im;',
            'quapopt_abs_squared_float64')
    return _abs_squared_kernel(state)
