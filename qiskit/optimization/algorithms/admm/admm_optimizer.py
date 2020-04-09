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

"""An implementation of the ADMM optimization algorithm."""

import logging
import time
from typing import List, Optional, Any, Tuple

import numpy as np
from cplex import SparsePair
from .admm_state import ADMMState
from .admm_parameters import ADMMParameters, UPDATE_RHO_BY_TEN_PERCENT
from ..cplex_optimizer import CplexOptimizer
from ..optimization_algorithm import OptimizationAlgorithm
from ...converters.optimization_problem_to_admm_state import OptimizationProblemToADMMState
from ...problems.optimization_problem import OptimizationProblem
from ...problems.variables import CPX_BINARY, CPX_CONTINUOUS
from ...results.optimization_result import OptimizationResult

UPDATE_RHO_BY_RESIDUALS = 1

logger = logging.getLogger(__name__)


class ADMMOptimizerResult(OptimizationResult):
    """A class representing the result of the ADMMOptimizer."""

    def __init__(self, x: Optional[Any] = None, fval: Optional[Any] = None,
                 state: Optional[ADMMState] = None, results: Optional[Any] = None) -> None:
        """
        Args:
            x: The found solution.
            fval: The function value evaluated at the solution.
            state: A :class:`~qiskit.optimization.algorithms.admm_optimizer.ADMMState` object
                representing the state of the ADMM optimization.
            results: Additional information about the result.
        """
        super().__init__(x, fval, results)
        self._state = state

    @property
    def state(self) -> Optional[ADMMState]:
        """The ADMM optimization state.

        Returns:
            The final ADMM optimization state, if it was set.
        """
        return self._state


class ADMMOptimizer(OptimizationAlgorithm):
    """The ADMM optimization routine.

    This ADMM-based heuristic has been introduced in [1].

    References:
        [1]: Gambella, C., & Simonetto, A. (2020), Multi-block ADMM Heuristics for Mixed-Binary
            Optimization on Classical and Quantum Computers. arXiv preprint arXiv:2001.02069.
    """

    def __init__(self, qubo_optimizer: Optional[OptimizationAlgorithm] = None,
                 continuous_optimizer: Optional[OptimizationAlgorithm] = None,
                 params: Optional[ADMMParameters] = None) -> None:
        """
        Args:
            qubo_optimizer: An instance of OptimizationAlgorithm that can effectively solve
                QUBO problems.
            continuous_optimizer: An instance of OptimizationAlgorithm that can solve
                continuous problems.
            params: An instance of ADMMParameters.
        """

        super().__init__()

        # create default params if not present
        self._params = params or ADMMParameters()

        # create optimizers if not specified
        self._qubo_optimizer = qubo_optimizer or CplexOptimizer()
        self._continuous_optimizer = continuous_optimizer or CplexOptimizer()

        # internal state where we'll keep intermediate solution
        # here, we just declare the class variable, the variable is initialized in kept in
        # the solve method.
        self._state: Optional[ADMMState] = None

    def is_compatible(self, problem: OptimizationProblem) -> Optional[str]:
        """Checks whether a given problem can be solved with the optimizer implementing this method.

        Args:
            problem: The optimization problem to check compatibility for.

        Returns:
            ``None`` if the problem is compatible and else a string with the error message.
        """

        # 1. only binary and continuous variables are supported
        for var_type in problem.variables.get_types():
            if var_type not in (CPX_BINARY, CPX_CONTINUOUS):
                # variable is not binary and not continuous.
                return "Only binary and continuous variables are supported"

        binary_indices = self._get_variable_indices(problem, CPX_BINARY)
        continuous_indices = self._get_variable_indices(problem, CPX_CONTINUOUS)

        # 2. binary and continuous variables are separable in objective
        for binary_index in binary_indices:
            for continuous_index in continuous_indices:
                coeff = problem.objective.get_quadratic_coefficients(binary_index, continuous_index)
                if coeff != 0:
                    # binary and continuous vars are mixed.
                    return "Binary and continuous variables are not separable in the objective"

        # 3. no quadratic constraints are supported.
        quad_constraints = problem.quadratic_constraints.get_num()
        if quad_constraints is not None and quad_constraints > 0:
            # quadratic constraints are not supported.
            return "Quadratic constraints are not supported"

        return None

    def solve(self, problem: OptimizationProblem) -> ADMMOptimizerResult:
        """Tries to solves the given problem using ADMM algorithm.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        # parse problem and convert to an ADMM specific representation
        self._state = OptimizationProblemToADMMState(self._params.rho_initial).encode(problem)

        start_time = time.time()
        # we have not stated our computations yet, so elapsed time initialized as zero.
        elapsed_time = 0
        iteration = 0
        residual = 1.e+2

        while iteration < self._params.max_iter and residual > self._params.tol \
                and elapsed_time < self._params.max_time:
            if self._state.binary_indices:
                op1 = self._create_step1_problem()
                self._state.x0 = self._update_x0(op1)
            # else, no binary variables exist,
            # and no update to be done in this case.
            # debug
            logger.debug("x0=%s", self._state.x0)

            op2 = self._create_step2_problem()
            self._state.u, self._state.z = self._update_x1(op2)
            # debug
            logger.debug("u=%s", self._state.u)
            logger.debug("z=%s", self._state.z)

            if self._params.three_block:
                if self._state.binary_indices:
                    op3 = self._create_step3_problem()
                    self._state.y = self._update_y(op3)
                # debug
                logger.debug("y=%s", self._state.y)

            self._state.lambda_mult = self._update_lambda_mult()

            cost_iterate = self._get_objective_value()
            constraint_residual = self._get_constraint_residual()
            residual, dual_residual = self._get_solution_residuals(iteration)
            merit = self._get_merit(cost_iterate, constraint_residual)
            # debug
            logger.debug("cost_iterate=%s, cr=%s, merit=%s",
                         cost_iterate, constraint_residual, merit)

            # costs and merits are saved with their original sign.
            self._state.cost_iterates.append(self._state.sense * cost_iterate)
            self._state.residuals.append(residual)
            self._state.dual_residuals.append(dual_residual)
            self._state.cons_r.append(constraint_residual)
            self._state.merits.append(merit)
            self._state.lambdas.append(np.linalg.norm(self._state.lambda_mult))

            self._state.x0_saved.append(self._state.x0)
            self._state.u_saved.append(self._state.u)
            self._state.z_saved.append(self._state.z)
            self._state.z_saved.append(self._state.y)

            self._update_rho(residual, dual_residual)

            iteration += 1
            elapsed_time = time.time() - start_time

        solution, objective_value = self._get_best_merit_solution()
        solution = self._revert_solution_indexes(solution)

        # third parameter is our internal state of computations.
        result = ADMMOptimizerResult(solution, objective_value, self._state)
        # debug
        logger.debug("solution=%s, objective=%s at iteration=%s",
                     solution, objective_value, iteration)
        return result

    @staticmethod
    def _get_variable_indices(op: OptimizationProblem, var_type: str) -> List[int]:
        """Returns a list of indices of the variables of the specified type.

        Args:
            op: Optimization problem.
            var_type: Type of variables to look for.

        Returns:
            List of indices.
        """
        indices = []
        for i, variable_type in enumerate(op.variables.get_types()):
            if variable_type == var_type:
                indices.append(i)

        return indices

    def _revert_solution_indexes(self, internal_solution: List[np.ndarray]) -> np.ndarray:
        """Constructs a solution array where variables are stored in the correct order.

        Args:
            internal_solution: a list with two lists: solutions for binary variables and
                for continuous variables.

        Returns:
            A solution array.
        """
        binary_solutions, continuous_solutions = internal_solution
        solution = np.zeros(len(self._state.binary_indices) + len(self._state.continuous_indices))
        # restore solution at the original index location
        for i, binary_index in enumerate(self._state.binary_indices):
            solution[binary_index] = binary_solutions[i]
        for i, continuous_index in enumerate(self._state.continuous_indices):
            solution[continuous_index] = continuous_solutions[i]
        return solution

    def _create_step1_problem(self) -> OptimizationProblem:
        """Creates a step 1 sub-problem.

        Returns:
            A newly created optimization problem.
        """
        op1 = OptimizationProblem()

        binary_size = len(self._state.binary_indices)
        # create the same binary variables.
        op1.variables.add(names=["x0_" + str(i + 1) for i in range(binary_size)],
                          types=["I"] * binary_size,
                          lb=[0.] * binary_size,
                          ub=[1.] * binary_size)

        # prepare and set quadratic objective.
        # NOTE: The multiplication by 2 is needed for the solvers to parse
        # the quadratic coefficients.
        quadratic_objective = self._state.q0 +\
            2 * (
                self._params.factor_c / 2 * np.dot(self._state.a0.transpose(), self._state.a0) +
                self._state.rho / 2 * np.eye(binary_size)
            )
        for i in range(binary_size):
            for j in range(i, binary_size):
                op1.objective.set_quadratic_coefficients(i, j, quadratic_objective[i, j])

        # prepare and set linear objective.
        linear_objective = self._state.c0 - \
            self._params.factor_c * np.dot(self._state.b0, self._state.a0) + \
            self._state.rho * (- self._state.y - self._state.z) + \
            self._state.lambda_mult

        for i in range(binary_size):
            op1.objective.set_linear(i, linear_objective[i])
        return op1

    def _create_step2_problem(self) -> OptimizationProblem:
        """Creates a step 2 sub-problem.

        Returns:
            A newly created optimization problem.
        """
        op2 = OptimizationProblem()

        continuous_size = len(self._state.continuous_indices)
        binary_size = len(self._state.binary_indices)
        lower_bounds = self._state.op.variables.get_lower_bounds(self._state.continuous_indices)
        upper_bounds = self._state.op.variables.get_upper_bounds(self._state.continuous_indices)
        if continuous_size:
            # add u variables.
            op2.variables.add(names=["u0_" + str(i + 1) for i in range(continuous_size)],
                              types=["C"] * continuous_size, lb=lower_bounds, ub=upper_bounds)

        # add z variables.
        op2.variables.add(names=["z0_" + str(i + 1) for i in range(binary_size)],
                          types=["C"] * binary_size,
                          lb=[0.] * binary_size,
                          ub=[1.] * binary_size)

        # set quadratic objective coefficients for u variables.
        if continuous_size:
            q_u = self._state.q1
            for i in range(continuous_size):
                for j in range(i, continuous_size):
                    op2.objective.set_quadratic_coefficients(i, j, q_u[i, j])

        # set quadratic objective coefficients for z variables.
        # NOTE: The multiplication by 2 is needed for the solvers to parse
        # the quadratic coefficients.
        q_z = 2 * (self._state.rho / 2 * np.eye(binary_size))
        for i in range(binary_size):
            for j in range(i, binary_size):
                op2.objective.set_quadratic_coefficients(i + continuous_size, j + continuous_size,
                                                         q_z[i, j])

        # set linear objective for u variables.
        if continuous_size:
            linear_u = self._state.c1
            for i in range(continuous_size):
                op2.objective.set_linear(i, linear_u[i])

        # set linear objective for z variables.
        linear_z = -1 * self._state.lambda_mult - self._state.rho * (self._state.x0 - self._state.y)
        for i in range(binary_size):
            op2.objective.set_linear(i + continuous_size, linear_z[i])

        # constraints for z.
        # A1 z <= b1.
        constraint_count = self._state.a1.shape[0]
        # in SparsePair val="something from numpy" causes an exception
        # when saving a model via cplex method.
        # rhs="something from numpy" is ok.
        # so, we convert every single value to python float
        lin_expr = [SparsePair(ind=list(range(continuous_size, continuous_size + binary_size)),
                               val=self._state.a1[i, :].tolist()) for i in
                    range(constraint_count)]
        op2.linear_constraints.add(lin_expr=lin_expr, senses=["L"] * constraint_count,
                                   rhs=list(self._state.b1))

        if continuous_size:
            # A2 z + A3 u <= b2
            constraint_count = self._state.a2.shape[0]
            lin_expr = [SparsePair(ind=list(range(continuous_size + binary_size)),
                                   val=self._state.a3[i, :].tolist() +
                                   self._state.a2[i, :].tolist())
                        for i in range(constraint_count)]
            op2.linear_constraints.add(lin_expr=lin_expr,
                                       senses=["L"] * constraint_count,
                                       rhs=self._state.b2.tolist())

        if continuous_size:
            # A4 u <= b3
            constraint_count = self._state.a4.shape[0]
            lin_expr = [SparsePair(ind=list(range(continuous_size)),
                                   val=self._state.a4[i, :].tolist()) for i in
                        range(constraint_count)]
            op2.linear_constraints.add(lin_expr=lin_expr,
                                       senses=["L"] * constraint_count,
                                       rhs=self._state.b3.tolist())

        return op2

    def _create_step3_problem(self) -> OptimizationProblem:
        """Creates a step 3 sub-problem.

        Returns:
            A newly created optimization problem.
        """
        op3 = OptimizationProblem()
        # add y variables.
        binary_size = len(self._state.binary_indices)
        op3.variables.add(names=["y_" + str(i + 1) for i in range(binary_size)],
                          types=["C"] * binary_size, lb=[-np.inf] * binary_size,
                          ub=[np.inf] * binary_size)

        # set quadratic objective.
        # NOTE: The multiplication by 2 is needed for the solvers to parse the quadratic coeff-s.
        q_y = 2 * (self._params.beta / 2 * np.eye(binary_size) +
                   self._state.rho / 2 * np.eye(binary_size))
        for i in range(binary_size):
            for j in range(i, binary_size):
                op3.objective.set_quadratic_coefficients(i, j, q_y[i, j])

        linear_y = - self._state.lambda_mult - self._state.rho * (self._state.x0 - self._state.z)
        for i in range(binary_size):
            op3.objective.set_linear(i, linear_y[i])

        return op3

    def _update_x0(self, op1: OptimizationProblem) -> np.ndarray:
        """Solves the Step1 OptimizationProblem via the qubo optimizer.

        Args:
            op1: the Step1 OptimizationProblem.

        Returns:
            A solution of the Step1, as a numpy array.
        """
        return np.asarray(self._qubo_optimizer.solve(op1).x)

    def _update_x1(self, op2: OptimizationProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the Step2 OptimizationProblem via the continuous optimizer.

        Args:
            op2: the Step2 OptimizationProblem

        Returns:
            A solution of the Step2, as a pair of numpy arrays.
            First array contains the values of decision variables u, and
            second array contains the values of decision variables z.
        """
        vars_op2 = self._continuous_optimizer.solve(op2).x
        vars_u = np.asarray(vars_op2[:len(self._state.continuous_indices)])
        vars_z = np.asarray(vars_op2[len(self._state.continuous_indices):])
        return vars_u, vars_z

    def _update_y(self, op3: OptimizationProblem) -> np.ndarray:
        """Solves the Step3 OptimizationProblem via the continuous optimizer.

        Args:
            op3: The Step3 OptimizationProblem.

        Returns:
            A solution of the Step3, as a numpy array.
        """
        return np.asarray(self._continuous_optimizer.solve(op3).x)

    def _get_best_merit_solution(self) -> Tuple[List[np.ndarray], float]:
        """The ADMM solution is that for which the merit value is the best (least for min problems,
        greatest for max problems)
            * sol: Iterate with the best merit value
            * sol_val: Value of sol, according to the original objective

        Returns:
            A tuple of (sol, sol_val), where
                * sol: Solution with the best merit value
                * sol_val: Value of the objective function
        """

        it_best_merits = self._state.merits.index(
            self._state.sense * min(list(map(lambda x: self._state.sense * x, self._state.merits)))
        )
        x_0 = self._state.x0_saved[it_best_merits]
        u_s = self._state.u_saved[it_best_merits]
        sol = [x_0, u_s]
        sol_val = self._state.cost_iterates[it_best_merits]
        return sol, sol_val

    def _update_lambda_mult(self) -> np.ndarray:
        """Updates the values of lambda multiplier, given the updated iterates x0, z, and y.

        Returns:
            The updated array of values of lambda multiplier.
        """
        return self._state.lambda_mult + \
            self._state.rho * (self._state.x0 - self._state.z - self._state.y)

    def _update_rho(self, primal_residual: float, dual_residual: float) -> None:
        """Updating the rho parameter in ADMM.

        Args:
            primal_residual: Primal residual.
            dual_residual: Dual residual.
        """

        if self._params.vary_rho == UPDATE_RHO_BY_TEN_PERCENT:
            # Increase rho, to aid convergence.
            if self._state.rho < 1.e+10:
                self._state.rho *= 1.1
        elif self._params.vary_rho == UPDATE_RHO_BY_RESIDUALS:
            if primal_residual > self._params.mu_res * dual_residual:
                self._state.rho = self._params.tau_incr * self._state.rho
            elif dual_residual > self._params.mu_res * primal_residual:
                self._state.rho = self._params.tau_decr * self._state.rho

    def _get_constraint_residual(self) -> float:
        """Compute violation of the constraints of the original problem.

        The residuals are computed as as:
            * norm 1 of the body-rhs of the constraints A0 x0 - b0
            * -1 * min(body - rhs, 0) for geq constraints
            * max(body - rhs, 0) for leq constraints

        Returns:
            Violation of the constraints as a float value.
        """
        cr0 = sum(np.abs(np.dot(self._state.a0, self._state.x0) - self._state.b0))

        eq1 = np.dot(self._state.a1, self._state.x0) - self._state.b1
        cr1 = sum(max(val, 0) for val in eq1)

        eq2 = np.dot(self._state.a2, self._state.x0) + np.dot(self._state.a3,
                                                              self._state.u) - self._state.b2
        cr2 = sum(max(val, 0) for val in eq2)

        return cr0 + cr1 + cr2

    def _get_merit(self, cost_iterate: float, constraint_residual: float) -> float:
        """Compute merit value associated with the current iterate.

        Args:
            cost_iterate: Cost at the certain iteration.
            constraint_residual: Value of violation of the constraints.

        Returns:
            Merit value as a float
        """
        return cost_iterate + self._params.mu_merit * constraint_residual

    def _get_objective_value(self) -> float:
        """Computes the value of the objective function.

        Returns:
            Value of the objective function as a float.
        """

        def quadratic_form(matrix, x, c):
            return np.dot(x.T, np.dot(matrix / 2, x)) + np.dot(c.T, x)

        obj_val = quadratic_form(self._state.q0, self._state.x0, self._state.c0)
        obj_val += quadratic_form(self._state.q1, self._state.u, self._state.c1)

        obj_val += self._state.op.objective.get_offset()

        return obj_val

    def _get_solution_residuals(self, iteration: int) -> Tuple[float, float]:
        """Compute primal and dual residual.

        Args:
            iteration: Iteration number.

        Returns:
            r, s as primary and dual residuals.
        """
        elements = self._state.x0 - self._state.z - self._state.y
        primal_residual = np.linalg.norm(elements)
        if iteration > 0:
            elements_dual = self._state.z - self._state.z_saved[iteration - 1]
        else:
            elements_dual = self._state.z - self._state.z_init
        dual_residual = self._state.rho * np.linalg.norm(elements_dual)

        return primal_residual, dual_residual
