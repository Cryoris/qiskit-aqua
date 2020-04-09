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

from typing import Optional, Any

from .optimization_result import OptimizationResult
from ..algorithms.admm.admm_state import ADMMState


class ADMMOptimizationResult(OptimizationResult):
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
