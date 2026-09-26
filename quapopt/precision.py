"""Package-wide precision policy for statevector simulations.

`precision` names the precision of the simulation state an operation stores and returns:
'single' is complex64 statevectors and initial states, 'double' is complex128. Each
operation computes accurately enough to deliver that precision, with whatever internal
arithmetic it needs.

Under both settings:

- problem data is float64: Hamiltonian coefficients, the coupling and local-field caches
  of `ClassicalHamiltonian`, spectra, extrema and stored energies. Rounding them would
  solve a different instance, not the same instance less precisely;
- accumulators are float64: probabilities, sampled energies, expectation sums.

Under 'single' the complex64 mixer's rounding moves the state's norm by up to about 1e-6,
which scales the exact expectation values of `QAOARunnerSampler` and `QAOARunnerStatevector`
by the same factor: a relative bias of up to about 1e-6. Constructed with `renormalize_probabilities=True`, those runners
divide each exact expectation value by the float64 sum of the same probabilities, which
removes the bias; it costs up to about 7 % of an exact-energy GPU `run_qaoa` call and below
1 % on the host. It is off by default under both settings, and the sum is then not computed.

Finite differences: under 'single', a step below 1e-3 gives gradient errors of about 1e-3 to
9e-3 relative, or 1e-4 to 1e-3 with `renormalize_probabilities=True`, and can flip the sign
of small components; `QAOAOptimizationRunner.estimate_gradient_finite_differences` warns
there. Use a step of at least 1e-3, or precision='double'.

A getter that takes its own `precision` dtype argument returns a copy at that dtype for
its caller; the stored data stays float64. Those getters default to float32:
`ClassicalHamiltonian.get_fields_and_couplings`, `get_adjacency_matrix` and
`get_couplings_and_local_fields`, and the module functions
`get_fields_and_couplings_from_hamiltonian_list` and
`get_fields_and_couplings_from_hamiltonian`.

The package default is 'single'. The environment variable QUAPOPT_PRECISION, read once at
import, and `set_default_precision` change it for objects constructed afterwards: every
constructor resolves `precision=None` to the default at construction time and keeps the
resolved value for the object's lifetime.
"""
import os
from dataclasses import dataclass

import numpy as np

_ENV_VAR = 'QUAPOPT_PRECISION'


@dataclass(frozen=True)
class Precision:
    name: str
    complex_dtype: np.dtype

    def __str__(self):
        return self.name


SINGLE = Precision('single', np.dtype(np.complex64))
DOUBLE = Precision('double', np.dtype(np.complex128))
_BY_NAME = {'single': SINGLE, 'double': DOUBLE}


def _parse(value, source):
    if isinstance(value, Precision):
        return value
    if isinstance(value, str) and value.lower() in _BY_NAME:
        return _BY_NAME[value.lower()]
    raise ValueError(
        f"{source} must be 'single' (complex64 statevectors) or 'double' (complex128), got {value!r}. "
        f"Pass precision='single' or precision='double' to the constructor, set the package default "
        f"with quapopt.set_default_precision('double'), or export {_ENV_VAR}=double before importing quapopt.")


_default = _parse(os.environ.get(_ENV_VAR, 'single'), f'the environment variable {_ENV_VAR}')


def get_default_precision() -> Precision:
    """The Precision a constructor resolves `precision=None` to right now."""
    return _default


def set_default_precision(value) -> Precision:
    """Set the package default for objects constructed from now on; returns the Precision set."""
    global _default
    _default = _parse(value, 'precision')
    return _default


def resolve_precision(value=None) -> Precision:
    """The Precision an object constructed now works at: `value` if given, else the package default."""
    if value is None:
        return _default
    return _parse(value, 'precision')
