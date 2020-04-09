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

"""A converter from quadratic program to a QUBO."""

from typing import Optional

from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.results import OptimizationResult
from qiskit.optimization.converters import (PenalizeLinearEqualityConstraints,
                                            IntegerToBinaryConverter)
from qiskit.optimization.utils import QiskitOptimizationError


class QuadraticProgramToQubo:
    """Convert a given optimization problem to a new problem that is a QUBO.

        Examples:
            >>> problem = QuadraticProgram()
            >>> # define a problem
            >>> conv = QuadraticProgramToQubo()
            >>> problem2 = conv.encode(problem)
    """

    def __init__(self, penalty: Optional[float] = 1e5) -> None:
        """
        Args:
            penalty: Penalty factor to scale equality constraints that are added to objective.
        """
        self._int_to_bin = IntegerToBinaryConverter()
        self._penalize_lin_eq_constraints = PenalizeLinearEqualityConstraints()
        self._penalty = penalty

    def encode(self, problem: QuadraticProgram) -> QuadraticProgram:
        """Convert a problem with linear equality constraints into new one with a QUBO form.

        Args:
            problem: The problem with linear equality constraints to be solved.

        Returns:
            The problem converted in QUBO format.

        Raises:
            QiskitOptimizationError: In case of an incompatible problem.
        """

        # analyze compatibility of problem
        msg = self.get_compatibility_msg(problem)
        if len(msg) > 0:
            raise QiskitOptimizationError('Incompatible problem: {}'.format(msg))

        # map integer variables to binary variables
        problem_ = self._int_to_bin.encode(problem)

        # penalize linear equality constraints with only binary variables
        if self._penalty is None:
            # TODO: should be derived from problem
            penalty = 1e5
        else:
            penalty = self._penalty
        problem_ = self._penalize_lin_eq_constraints.encode(problem_, penalty_factor=penalty)

        # return QUBO
        return problem_

    def decode(self, result: OptimizationResult) -> OptimizationResult:
        """ Convert a result of a converted problem into that of the original problem.

            Args:
                result: The result of the converted problem.

            Returns:
                The result of the original problem.
        """
        return self._int_to_bin.decode(result)

    @staticmethod
    def get_compatibility_msg(problem: QuadraticProgram) -> bool:
        """Checks whether a given problem can be cast to a QUBO.

        A quadratic program can be converted to a QUBO (Quadratic Unconstrained Binary
        Optimization) problem, if the problem contains only binary and integer variables as well
        as linear equality constraints.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            True, if the problem can be converted to a QUBO, otherwise a QiskitOptimizationError
            is raised.

        Raises:
            QiskitOptimizationError: If the conversion to QUBO is not possible.
        """

        # initialize message
        msg = ''

        # check whether there are incompatible variable types
        if problem.variables.get_num_continuous() > 0:
            msg += 'Continuous variables are not supported! '
        if problem.variables.get_num_semicontinuous() > 0:
            # TODO: to be removed once semi-continuous to binary + continuous is introduced
            msg += 'Semi-continuous variables are not supported! '
        if problem.variables.get_num_semiinteger() > 0:
            # TODO: to be removed once semi-integer to binary mapping is introduced
            msg += 'Semi-integer variables are not supported! '

        # check whether there are incompatible constraint types
        if not all([sense == 'E' for sense in problem.linear_constraints.get_senses()]):
            msg += 'Only linear equality constraints are supported.'
        if problem.quadratic_constraints.get_num() > 0:
            msg += 'Quadratic constraints are not supported. '

        # if an error occurred, return error message, otherwise, return None
        return msg

    def is_compatible(self, problem: QuadraticProgram) -> bool:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            Returns True if the problem is compatible, False otherwise.
        """
        return len(self.get_compatibility_msg(problem)) == 0
