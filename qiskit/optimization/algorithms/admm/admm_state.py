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

"""The module containing the class holding the current state of the ADMM optimization."""

from typing import List
import numpy

from ...problems.optimization_problem import OptimizationProblem


class ADMMState:
    """Internal computation state of the ADMM implementation.

    The state keeps track of various variables are stored that are being updated during problem
    solving. The values are relevant to the problem being solved. The state is recreated for each
    optimization problem. State is returned as the third value.
    """

    def __init__(self,
                 op: OptimizationProblem,
                 binary_indices: List[int],
                 continuous_indices: List[int],
                 rho_initial: float) -> None:
        """Constructs an internal computation state of the ADMM implementation.

        Args:
            op: The optimization problem being solved.
            binary_indices: Indices of the binary decision variables of the original problem.
            continuous_indices: Indices of the continuous decision variables of the original
             problem.
            rho_initial: Initial value of the rho parameter.
        """
        super().__init__()

        # Optimization problem itself
        self.op = op
        # Indices of the variables
        self.binary_indices = binary_indices
        self.continuous_indices = continuous_indices
        self.sense = op.objective.get_sense()

        # define heavily used matrix, they are used at each iteration, so let's cache them,
        # they are numpy.ndarrays
        # pylint:disable=invalid-name
        # objective
        self.q0 = None
        self.c0 = None
        self.q1 = None
        self.c1 = None
        # constraints
        self.a0 = None
        self.b0 = None
        self.a1 = None
        self.b1 = None
        self.a2 = None
        self.a3 = None
        self.b2 = None
        self.a4 = None
        self.b3 = None

        # These are the parameters that are updated in the ADMM iterations.
        self.u: numpy.ndarray = numpy.zeros(len(continuous_indices))
        binary_size = len(binary_indices)
        self.x0: numpy.ndarray = numpy.zeros(binary_size)
        self.z: numpy.ndarray = numpy.zeros(binary_size)
        self.z_init: numpy.ndarray = self.z
        self.y: numpy.ndarray = numpy.zeros(binary_size)
        self.lambda_mult: numpy.ndarray = numpy.zeros(binary_size)

        # The following structures store quantities obtained in each ADMM iteration.
        self.cost_iterates = []
        self.residuals = []
        self.dual_residuals = []
        self.cons_r = []
        self.merits = []
        self.lambdas = []
        self.x0_saved = []
        self.u_saved = []
        self.z_saved = []
        self.y_saved = []
        self.rho = rho_initial
