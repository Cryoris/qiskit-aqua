# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
========================================================
Optimization stack for Aqua (:mod:`qiskit.optimization.converters`)
========================================================

.. currentmodule:: qiskit.optimization.converters

Structures for converting optimization problems
==========

"""

# no opt problem dependency
from .penalize_linear_equality_constraints import PenalizeLinearEqualityConstraints
from .quadratic_program_to_operator import QuadraticProgramToOperator
from .quadratic_program_to_negative_value_oracle import QuadraticProgramToNegativeValueOracle

# opt problem dependency
from .integer_to_binary import IntegerToBinary
from .inequality_to_equality import InequalityToEquality
from .quadratic_program_to_qubo import QuadraticProgramToQubo

__all__ = [
    "InequalityToEqualityConverter",
    "IntegerToBinaryConverter",
    "QuadraticProgramToNegativeValueOracle",
    "QuadraticProgramToOperator",
    "QuadraticProgramToQubo",
    "PenalizeLinearEqualityConstraints",
]
