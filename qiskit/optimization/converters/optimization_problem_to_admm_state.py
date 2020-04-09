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

"""The module containing the converter from the Optimization Problem to an ADMM state."""

from typing import Tuple, List, Optional
import numpy as np

from ..algorithms.admm.admm_state import ADMMState
from ..problems.optimization_problem import OptimizationProblem
from ..problems.variables import CPX_BINARY, CPX_CONTINUOUS


class OptimizationProblemToADMMState:
    """Converts an optimization problem to an ADMM state."""

    def __init__(self, rho_initial: Optional[float] = None) -> None:
        """
        Args:
            rho_initial: The initial rho parameter of the ADMM optimization. If an
                ``ADMMParameters`` object is available, this parameter is accessible as
                ``ADMMParameters.rho_initial``.
        """
        # we need rho_initial only as additional parameter in the ADMM state, the conversion can be
        # done without this -- therefore leave it as optional
        self.rho_initial = rho_initial

    def encode(self, problem: OptimizationProblem) -> ADMMState:
        """Convert the QUBO into an initial ADMM state

        Specifically, the optimization problem is represented as:

        min_{x0, u} x0^T q0 x0 + c0^T x0 + u^T q1 u + c1^T u

        s.t. a0 x0 = b0
            a1 x0 <= b1
            a2 z + a3 u <= b2
            a4 u <= b3

        Args:
            problem: The optimization problem to be converted.

        Returns:
            The initial ADMM state for the ADMM optimization.
        """

        # parse problem and convert to an ADMM specific representation.
        binary_indices = self._get_variable_indices(problem, CPX_BINARY)
        continuous_indices = self._get_variable_indices(problem, CPX_CONTINUOUS)

        # create our computation state.
        state = ADMMState(problem, binary_indices, continuous_indices, self.rho_initial)

        state.q0 = self._get_q(problem, binary_indices)
        state.c0 = self._get_c(problem, binary_indices)
        state.q1 = self._get_q(problem, continuous_indices)
        state.c1 = self._get_c(problem, continuous_indices)
        # constraints
        state.a0, state.b0 = self._get_a0_b0(problem, binary_indices)
        state.a1, state.b1 = self._get_a1_b1(problem, binary_indices)
        state.a2, state.a3, state.b2 = self._get_a2_a3_b2(problem, binary_indices,
                                                          continuous_indices)
        state.a4, state.b3 = self._get_a4_b3(problem, continuous_indices)

        return state

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

    def _get_q(self, problem: OptimizationProblem, variable_indices: List[int]) -> np.ndarray:
        """Constructs a quadratic matrix for the variables with the specified indices
        from the quadratic terms in the objective.

        Args:
            problem: The optimization problem holding the objective.
            variable_indices: Variable indices to look for.

        Returns:
            A matrix as a numpy array of the shape(len(variable_indices), len(variable_indices)).
        """
        size = len(variable_indices)
        q = np.zeros(shape=(size, size))
        # fill in the matrix
        # in fact we use re-indexed variables
        for i, var_index_i in enumerate(variable_indices):
            for j, var_index_j in enumerate(variable_indices):
                q[i, j] = problem.objective.get_quadratic_coefficients(var_index_i,
                                                                       var_index_j)

        # flip the sign, according to the optimization sense, e.g. sense == 1 if minimize,
        # sense == -1 if maximize.
        return q * problem.objective.get_sense()

    def _get_c(self, problem: OptimizationProblem, variable_indices: List[int]) -> np.ndarray:
        """Constructs a vector for the variables with the specified indices from the linear terms
        in the objective.

        Args:
            problem: The optimization problem holding the objective.
            variable_indices: Variable indices to look for.

        Returns:
            A numpy array of the shape(len(variable_indices)).
        """
        c = np.array(problem.objective.get_linear(variable_indices))
        # flip the sign, according to the optimization sense, e.g. sense == 1 if minimize,
        # sense == -1 if maximize.
        return c * problem.objective.get_sense()

    def _get_a0_b0(self, problem: OptimizationProblem, binary_indices: List[int]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Constructs a matrix and a vector from the constraints in a form of Ax = b, where
        x is a vector of binary variables.

        Args:
            problem: The optimization problem holding the objective.
            binary_indices: The binary variable indices to look for.

        Returns:
            Corresponding matrix and vector as numpy arrays.

        Raises:
            ValueError: if the problem is not suitable for this optimizer.
        """
        matrix = []
        vector = []

        senses = problem.linear_constraints.get_senses()
        index_set = set(binary_indices)
        for constraint_index, sense in enumerate(senses):
            # we check only equality constraints here.
            if sense != "E":
                continue
            row = problem.linear_constraints.get_rows(constraint_index)
            if set(row.ind).issubset(index_set):
                self._assign_row_values(problem, matrix, vector, constraint_index, binary_indices)
            else:
                raise ValueError(
                    "Linear constraint with the 'E' sense must contain only binary variables, "
                    "row indices: {}, binary variable indices: {}".format(row, binary_indices)
                )

        return self._create_ndarrays(matrix, vector, len(binary_indices))

    def _get_a1_b1(self, problem: OptimizationProblem, binary_indices: List[int]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Constructs a matrix and a vector from the constraints in a form of Ax <= b, where
        x is a vector of binary variables.

        Returns:
            A numpy based representation of the matrix and the vector.
        """
        matrix, vector = self._get_inequality_matrix_and_vector(problem, binary_indices)
        return self._create_ndarrays(matrix, vector, len(binary_indices))

    def _get_a4_b3(self, problem: OptimizationProblem, continuous_indices: List[int]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Constructs a matrix and a vector from the constraints in a form of Au <= b, where
        u is a vector of continuous variables.

        Returns:
            A numpy based representation of the matrix and the vector.
        """
        matrix, vector = self._get_inequality_matrix_and_vector(problem, continuous_indices)
        return self._create_ndarrays(matrix, vector, len(continuous_indices))

    def _get_a2_a3_b2(self, problem, binary_indices, continuous_indices) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constructs matrices and a vector from the constraints in a form of A_2x + A_3u <= b,
        where x is a vector of binary variables and u is a vector of continuous variables.

        Returns:
            A numpy representation of two matrices and one vector.

        Raises:
            QiskitOptimizationError: If contraint types "E" or "R" are used, which are not yet
                supported.
        """
        matrix = []
        vector = []
        senses = problem.linear_constraints.get_senses()

        binary_index_set = set(binary_indices)
        continuous_index_set = set(continuous_indices)
        all_variables = binary_indices + continuous_indices
        for constraint_index, sense in enumerate(senses):
            if sense in ("E", "R"):
                # TODO: Ranged constraints should be supported as well
                continue

            # sense either G or L.
            row = problem.linear_constraints.get_rows(constraint_index)
            row_indices = set(row.ind)
            # we must have a least one binary and one continuous variable,
            # otherwise it is another type of constraints.
            if len(row_indices & binary_index_set) != 0 and len(
                    row_indices & continuous_index_set) != 0:
                self._assign_row_values(problem, matrix, vector, constraint_index, all_variables)

        matrix, b_2 = self._create_ndarrays(matrix, vector, len(all_variables))
        a_2 = matrix[:, 0:len(binary_indices)]
        a_3 = matrix[:, len(binary_indices):]
        return a_2, a_3, b_2

    def _get_inequality_matrix_and_vector(self, problem: OptimizationProblem,
                                          variable_indices: List[int]) \
            -> Tuple[List[List[float]], List[float]]:
        """Constructs a matrix and a vector from the constraints in a form of Ax <= b, where
        x is a vector of variables specified by the indices.

        Args:
            problem: The optimization problem holding the linear constraints.
            variable_indices: Variable indices to look for.

        Returns:
            A list based representation of the matrix and the vector.

        Raises:
            QiskitOptimizationError: If contraint types "E" or "R" are used, which are not yet
                supported.
        """
        matrix = []
        vector = []
        senses = problem.linear_constraints.get_senses()

        index_set = set(variable_indices)
        for constraint_index, sense in enumerate(senses):
            if sense in ("E", "R"):
                # TODO: Ranged constraints should be supported
                continue

            # sense either G or L.
            row = problem.linear_constraints.get_rows(constraint_index)
            if set(row.ind).issubset(index_set):
                self._assign_row_values(problem, matrix, vector, constraint_index, variable_indices)

        return matrix, vector

    def _assign_row_values(self, problem: OptimizationProblem,
                           matrix: List[List[float]], vector: List[float],
                           constraint_index: int, variable_indices: List[int]) -> None:
        """Appends a row to the specified matrix and vector based on the constraint specified by
        the index using specified variables.

        Args:
            problem: The optimization problem holding the linear constraints.
            matrix: A matrix to extend.
            vector: A vector to expand.
            constraint_index: Constraint index to look for.
            variable_indices: Variables to look for.
        """
        # assign matrix row.
        row = []
        for var_index in variable_indices:
            row.append(problem.linear_constraints.get_coefficients(constraint_index, var_index))
        matrix.append(row)

        # assign vector row.
        vector.append(problem.linear_constraints.get_rhs(constraint_index))

        # flip the sign if constraint is G, we want L constraints.
        if problem.linear_constraints.get_senses(constraint_index) == "G":
            # invert the sign to make constraint "L".
            matrix[-1] = [-1 * el for el in matrix[-1]]
            vector[-1] = -1 * vector[-1]

    def _create_ndarrays(self, matrix: List[List[float]], vector: List[float], size: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Converts representation of a matrix and a vector in form of lists to numpy array.

        Args:
            matrix: matrix to convert.
            vector: vector to convert.
            size: size to create matrix and vector.

        Returns:
            Converted matrix and vector as numpy arrays.
        """
        # if we don't have such constraints, return just dummy arrays.
        if len(matrix) != 0:
            return np.array(matrix), np.array(vector)
        else:
            return np.array([0] * size).reshape((1, -1)), np.zeros(shape=(1,))
