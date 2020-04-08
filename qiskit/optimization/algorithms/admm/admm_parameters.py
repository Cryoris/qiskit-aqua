# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The module containing the class holding parameters for the ADMM optimization."""

import numpy

UPDATE_RHO_BY_TEN_PERCENT = 0


class ADMMParameters:
    """Defines a set of parameters for ADMM optimizer."""

    def __init__(self, rho_initial: float = 10000, factor_c: float = 100000, beta: float = 1000,
                 max_iter: int = 10, tol: float = 1.e-4, max_time: float = numpy.inf,
                 three_block: bool = True, vary_rho: int = UPDATE_RHO_BY_TEN_PERCENT,
                 tau_incr: float = 2, tau_decr: float = 2, mu_res: float = 10,
                 mu_merit: float = 1000) -> None:
        """Defines parameters for ADMM optimizer and their default values.

        Args:
            rho_initial: Initial value of rho parameter of ADMM.
            factor_c: Penalizing factor for equality constraints, when mapping to QUBO.
            beta: Penalization for y decision variables.
            max_iter: Maximum number of iterations for ADMM.
            tol: Tolerance for the residual convergence.
            max_time: Maximum running time (in seconds) for ADMM.
            three_block: Boolean flag to select the 3-block ADMM implementation.
            vary_rho: Flag to select the rule to update rho.
                If set to 0, then rho increases by 10% at each iteration.
                If set to 1, then rho is modified according to primal and dual residuals.
            tau_incr: Parameter used in the rho update (UPDATE_RHO_BY_RESIDUALS).
                The update rule can be found in:
                Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
                Distributed optimization and statistical learning via the alternating
                direction method of multipliers.
                Foundations and Trends® in Machine learning, 3(1), 1-122.
            tau_decr: Parameter used in the rho update (UPDATE_RHO_BY_RESIDUALS).
            mu_res: Parameter used in the rho update (UPDATE_RHO_BY_RESIDUALS).
            mu_merit: Penalization for constraint residual. Used to compute the merit values.
        """
        super().__init__()
        self.mu_merit = mu_merit
        self.mu_res = mu_res
        self.tau_decr = tau_decr
        self.tau_incr = tau_incr
        self.vary_rho = vary_rho
        self.three_block = three_block
        self.max_time = max_time
        self.tol = tol
        self.max_iter = max_iter
        self.factor_c = factor_c
        self.beta = beta
        self.rho_initial = rho_initial
