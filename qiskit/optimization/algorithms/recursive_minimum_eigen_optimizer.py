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

"""A recursive minimal eigen optimizer in Qiskit Optimization."""

from copy import deepcopy
from typing import Optional
import logging
import numpy as np

from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.utils.validation import validate_min

from .optimization_algorithm import OptimizationAlgorithm, OptimizationResult
from .minimum_eigen_optimizer import MinimumEigenOptimizer
from ..exceptions.qiskit_optimization_error import QiskitOptimizationError
from ..problems.quadratic_program import QuadraticProgram
from ..converters.quadratic_program_to_qubo import QuadraticProgramToQubo

logger = logging.getLogger(__name__)

_HAS_CPLEX = False
try:
    from cplex import SparseTriple
    _HAS_CPLEX = True
except ImportError:
    logger.info('CPLEX is not installed.')


class RecursiveMinimumEigenOptimizer(OptimizationAlgorithm):
    """A meta-algorithm that applies a recursive optimization.

    The recursive minimum eigen optimizer applies a recursive optimization on top of
    :class:`~qiskit.optimization.algorithms.MinimumEigenOptimizer`.
    The algorithm is introduced in [1].

    Examples:
        >>> problem = QuadraticProgram()
        >>> # specify problem here
        >>> # specify minimum eigen solver to be used, e.g., QAOA
        >>> qaoa = QAOA(...)
        >>> optimizer = RecursiveMinEigenOptimizer(qaoa)
        >>> result = optimizer.solve(problem)

    References:
        [1]: Bravyi et al. (2019), Obstacles to State Preparation and Variational Optimization
            from Symmetry Protection. http://arxiv.org/abs/1910.08980.
    """

    def __init__(self, min_eigen_optimizer: MinimumEigenOptimizer, min_num_vars: int = 1,
                 min_num_vars_optimizer: Optional[OptimizationAlgorithm] = None,
                 penalty: Optional[float] = None) -> None:
        """ Initializes the recursive minimum eigen optimizer.

        This initializer takes a ``MinimumEigenOptimizer``, the parameters to specify until when to
        to apply the iterative scheme, and the optimizer to be applied once the threshold number of
        variables is reached.

        Args:
            min_eigen_optimizer: The eigen optimizer to use in every iteration.
            min_num_vars: The minimum number of variables to apply the recursive scheme. If this
                threshold is reached, the min_num_vars_optimizer is used.
            min_num_vars_optimizer: This optimizer is used after the recursive scheme for the
                problem with the remaining variables.
            penalty: The factor that is used to scale the penalty terms corresponding to linear
                equality constraints.

        TODO: add flag to store full history.

        Raises:
            QiskitOptimizationError: In case of invalid parameters (num_min_vars < 1).
            NameError: CPLEX is not installed.
        """

        # TODO: should also allow function that maps problem to <ZZ>-correlators?
        # --> would support efficient classical implementation for QAOA with depth p=1
        # --> add results class for MinimumEigenSolver that contains enough info to do so.
        if not _HAS_CPLEX:
            raise NameError('CPLEX is not installed.')

        validate_min('min_num_vars', min_num_vars, 1)

        self._min_eigen_optimizer = min_eigen_optimizer
        self._min_num_vars = min_num_vars
        if min_num_vars_optimizer:
            self._min_num_vars_optimizer = min_num_vars_optimizer
        else:
            self._min_num_vars_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        self._penalty = penalty

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """
        return QuadraticProgramToQubo.get_compatibility_msg(problem)

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solve the given problem using the recursive optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: Infeasible due to variable substitution
        """
        # convert problem to QUBO, this implicitly checks if the problem is compatible
        qubo_converter = QuadraticProgramToQubo()
        problem_ = qubo_converter.encode(problem)
        problem_ref = deepcopy(problem_)

        # run recursive optimization until the resulting problem is small enough
        replacements = {}
        while problem_.variables.get_num() > self._min_num_vars:

            # solve current problem with optimizer
            result = self._min_eigen_optimizer.solve(problem_)

            # analyze results to get strongest correlation
            correlations = result.get_correlations()
            i, j = self._find_strongest_correlation(correlations)

            x_i = problem_.variables.get_names(i)
            x_j = problem_.variables.get_names(j)
            if correlations[i, j] > 0:
                # set x_i = x_j
                problem_, status = problem_.substitute_variables(
                    variables=SparseTriple([i], [j], [1]))
                if status == problem_.substitution_status.infeasible:
                    raise QiskitOptimizationError('Infeasible due to variable substitution')
                replacements[x_i] = (x_j, 1)
            else:
                # set x_i = 1 - x_j, this is done in two steps:
                # 1. set x_i = 1 + x_i
                # 2. set x_i = -x_j

                # 1a. get additional offset
                offset = problem_.objective.get_offset()
                offset += problem_.objective.get_quadratic_coefficients(i, i) / 2
                offset += problem_.objective.get_linear(i)
                problem_.objective.set_offset(offset)

                # 1b. get additional linear part
                for k in range(problem_.variables.get_num()):
                    coeff = problem_.objective.get_quadratic_coefficients(i, k)
                    if np.abs(coeff) > 1e-10:
                        coeff += problem_.objective.get_linear(k)
                        problem_.objective.set_linear(k, coeff)

                # 2. replace x_i by -x_j
                problem_, status = problem_.substitute_variables(
                    variables=SparseTriple([i], [j], [-1]))
                if status == problem_.substitution_status.infeasible:
                    raise QiskitOptimizationError('Infeasible due to variable substitution')
                replacements[x_i] = (x_j, -1)

        # solve remaining problem
        result = self._min_num_vars_optimizer.solve(problem_)

        # unroll replacements
        var_values = {}
        for i, name in enumerate(problem_.variables.get_names()):
            var_values[name] = result.x[i]

        def find_value(x, replacements, var_values):
            if x in var_values:
                # if value for variable is known, return it
                return var_values[x]
            elif x in replacements:
                # get replacement for variable
                (y, sgn) = replacements[x]
                # find details for replacing variable
                value = find_value(y, replacements, var_values)
                # construct, set, and return new value
                var_values[x] = value if sgn == 1 else 1 - value
                return var_values[x]
            else:
                raise QiskitOptimizationError('Invalid values!')

        # loop over all variables to set their values
        for x_i in problem_ref.variables.get_names():
            if x_i not in var_values:
                find_value(x_i, replacements, var_values)

        # construct result
        x = [var_values[name] for name in problem_ref.variables.get_names()]
        fval = result.fval
        results = OptimizationResult(x, fval, (replacements, qubo_converter))
        results = qubo_converter.decode(results)
        return results

    def _find_strongest_correlation(self, correlations):
        m_max = np.argmax(np.abs(correlations.flatten()))
        i = int(m_max // len(correlations))
        j = int(m_max - i*len(correlations))
        return (i, j)