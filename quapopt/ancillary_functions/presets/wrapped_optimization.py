# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
from typing import Optional, List

import numpy as np
from qiskit import QuantumCircuit

from quapopt.ancillary_functions.presets import create_grid_search_plus_scipy_consecutive_optimizer
from quapopt.circuits import backend_utilities as bck_utils
from quapopt.circuits.backend_utilities import create_qiskit_sampler
from quapopt.hamiltonians.representation.ClassicalHamiltonian import ClassicalHamiltonian
from quapopt.optimization.QAOA.simulation.QAOARunnerExpValuesRDMs import QAOARunnerExpValuesRDMs
from quapopt.optimization.parameter_setting import ParametersBoundType as PBT
from quapopt.optimization.parameter_setting.non_adaptive_optimization.SimpleGridOptimizer import SimpleGridOptimizer
from quapopt.optimization.parameter_setting.variational.QAOAOptimizationRunner import QAOAOptimizationRunner
from quapopt.optimization.parameter_setting.variational.custom_optimizers.DivideAndConquerGridOptimizer import \
    DivideAndConquerGridOptimizer
from quapopt.optimization.parameter_setting.variational.scipy_tools.ScipyOptimizerWrapped import ScipyOptimizerWrapped










def get_classical_optimizer_p1_default(reduced_search_space:bool=False,
                                       smart_starting_point:bool=False,
                                       include_grid_search:bool=False,
                                       grid_search_is_adaptive:bool=False,
                                       divide_and_conquer:bool=False,
                                       calls_split: Optional[List[float]] = None,
                                       shrink_factors: Optional[List[float]] = None,
                                       optimizer_name_scipy: str = 'cobyqa',
                                       basinhopping:bool=True,
                                       optimizer_kwargs:Optional[dict]=None,
                                       basinhopping_kwargs:Optional[dict]=None,
                                       efficient_finding_of_best_beta:bool=False,
                                       fold_gamma_symmetry:bool=False,
                                       max_trials:Optional[int]=None,
                                       number_of_qubits:Optional[int]=None
                                       ):
    r"""
    Build a default classical optimizer for p=1 QAOA angle optimization, together with a JSON-able
    metadata dict describing it. Returns ``(classical_optimizer, optimizer_metadata)``.

    The optimized parameters are ``(gamma, beta)``, or ``(gamma,)`` alone when
    ``efficient_finding_of_best_beta=True`` (beta is then found internally, per gamma, by a
    brute-force scan over a symmetric range). Other flags select the optimizer family
    (scipy/COBYQA, a simple grid, or a divide-and-conquer grid) and the search space.

    fold_gamma_symmetry : bool, default False
        Restrict the gamma range to ``[0, pi]`` instead of the full ``[-pi, pi]``. This is EXACT
        (no optimum is lost), unlike ``reduced_search_space`` which is a lossy truncation: for a real
        cost Hamiltonian the p=1 expectation obeys ``<H>(gamma, beta) = <H>(-gamma, -beta)``, so the
        beta-optimized landscape ``g(gamma) := min_beta <H>(gamma, beta)`` is EVEN in gamma
        (``g(-gamma) = g(gamma)``) provided beta is minimized over a symmetric range. Folding thus
        doubles the effective grid resolution (equivalently, halves the search domain) for free, and
        ``(gamma, beta)`` vs ``(-gamma, -beta)`` yield identical energies and sampling distributions.
        Requires ``efficient_finding_of_best_beta=True`` (the per-gamma beta-min is what makes ``g``
        even); a ``ValueError`` is raised otherwise.

        Provenance note: the ND-AWS paper experiments used the FULL ``[-pi, pi]`` range
        (``fold_gamma_symmetry=False``, the default). Keep it False to reproduce those angles.
    """
    if fold_gamma_symmetry and not efficient_finding_of_best_beta:
        raise ValueError(
            "fold_gamma_symmetry=True requires efficient_finding_of_best_beta=True: the even-in-gamma "
            "symmetry g(gamma)=g(-gamma) holds only when beta is optimized per-gamma over a symmetric range."
        )

    if reduced_search_space:
        parameters_bounds = [(0, np.pi/2),
                             (-np.pi / 2, np.pi / 2)]
    else:
        parameters_bounds = [(-np.pi, np.pi),
                             (-np.pi/2, np.pi/2)
                             ]
    if smart_starting_point:
        if number_of_qubits is not None:
            _gamma = 1/np.sqrt(number_of_qubits)/2
        else:
            _gamma = 0.01

        starting_point = [_gamma, -np.pi / 8]


    else:
        if include_grid_search:
            starting_point = [0.0, 0.0]
        else:
            starting_point = [0.0001, 0.0001]


    if efficient_finding_of_best_beta:
        parameters_bounds = parameters_bounds[0:1]
        starting_point = starting_point[0:1]
        if fold_gamma_symmetry:
            # Even-in-gamma symmetry (see docstring): fold gamma to [0, upper]. Exact, ~2x resolution.
            _gamma_lo, _gamma_hi = parameters_bounds[0]
            parameters_bounds = [(max([0.0, _gamma_lo]), _gamma_hi)]
            starting_point = [abs(starting_point[0])]

    if optimizer_name_scipy is None:
        optimizer_name_scipy = 'cobyqa'

    if basinhopping_kwargs is None:
        basinhopping_kwargs = {'niter': 5,
                               'T': 2.0,
                               'disp': False,
                               'stepsize': 0.5,
                               'seed': 0}



    if include_grid_search and divide_and_conquer:
        parameters_bounds = [(PBT.RANGE, tup) for tup in parameters_bounds]

        if grid_search_is_adaptive:
            if calls_split is None:
                calls_split = [0.9, 0.1]
            if shrink_factors is None:
                shrink_factors = [0.2]

            classical_optimizer = create_grid_search_plus_scipy_consecutive_optimizer(parameter_bounds_global=parameters_bounds,
                                                                                        calls_split=calls_split,
                                                                                        shrink_factors=shrink_factors,
                                                                                        optimizer_name_scipy=optimizer_name_scipy,
                                                                                        specific_points_to_include=starting_point
                                                                                        )
            _description = "Grid optimizer followed by Scipy optimizer",

        else:
            if calls_split is None:
                calls_split = [0.9, 0.1]
            if shrink_factors is None:
                shrink_factors = [0.05]

            _description = 'Divide and conquer Grid search'

            classical_optimizer = DivideAndConquerGridOptimizer(parameter_bounds=parameters_bounds,
                                                                calls_split=calls_split,
                                                                shrink_factors=shrink_factors)


        optimizer_metadata = {'Description': _description,
                              "calls_split": calls_split,
                              "shrink_factors": shrink_factors,
                              "parameters_bounds": parameters_bounds,
                              "parameter_bounds_global": parameters_bounds,
                              "optimizer_name_scipy": optimizer_name_scipy,
                              'starting_point':starting_point,
                              'efficient_finding_of_best_beta':efficient_finding_of_best_beta,
                              'fold_gamma_symmetry':fold_gamma_symmetry
                              }

    elif include_grid_search and not divide_and_conquer:
        parameters_bounds = [(PBT.RANGE, tup) for tup in parameters_bounds]

        _description = 'Simple grid search'

        classical_optimizer = SimpleGridOptimizer(parameter_bounds=parameters_bounds,
                                                    max_trials=max_trials,
                                                    specific_points_to_include=tuple(starting_point))

        optimizer_metadata = {'Description': _description,
                              "parameters_bounds": parameters_bounds,
                               'starting_point': starting_point,
                              'efficient_finding_of_best_beta': efficient_finding_of_best_beta,
                              'fold_gamma_symmetry': fold_gamma_symmetry

                              }


    else:

        if optimizer_kwargs is None:

            optimizer_kwargs = {'options': {'disp': False,
                                            #'maxiter': 100,
                                            #'maxfev': 100,
                                            'initial_tr_radius': 0.01,
                                            'final_tr_radius': 10e-6,
                                            'scale': False,
                                            }, }



        optimizer_metadata = {'Description': "Scipy optimizer",
                              "parameters_bounds": parameters_bounds,
                              "optimizer_kwargs": optimizer_kwargs,
                              "optimizer_name": optimizer_name_scipy,
                              "basinhopping_kwargs": basinhopping_kwargs,
                              'starting_point': starting_point,
                              'efficient_finding_of_best_beta': efficient_finding_of_best_beta,
                              'fold_gamma_symmetry': fold_gamma_symmetry

                              }





        classical_optimizer = ScipyOptimizerWrapped(parameters_bounds=parameters_bounds,
                                                    optimizer_name=optimizer_name_scipy,
                                                    optimizer_kwargs=optimizer_kwargs,
                                                    basinhopping=basinhopping,
                                                    basinhopping_kwargs=basinhopping_kwargs,
                                                    starting_point=starting_point)

    return classical_optimizer, optimizer_metadata
