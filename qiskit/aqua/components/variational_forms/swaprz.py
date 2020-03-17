# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Layers of Swap+Z rotations followed by entangling gates."""

import warnings
from qiskit.aqua.components.ansatzes import SwapRZ

warnings.warn('This module is deprecated, SwapRZ is now located in '
              'qiskit.aqua.components.ansatzes.swaprz.',
              DeprecationWarning, stacklevel=2)

__all__ = ['SwapRZ']
