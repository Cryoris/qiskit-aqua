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

"""The module for Aqua's gradient."""

from .circuit_gradient_methods import (LinCombGradient, ParamShiftGradient)
from .qfi_methods import (LinCombQFI, OverlapQFI)
from .circuit_gradient_methods.circuit_gradient_method import CircuitGradientMethod
from .qfi_methods.qfi_method import QFIMethod
from .derivatives_base import DerivativeBase
from .gradient import Gradient
from .hessian import Hessian
from .natural_gradient import NaturalGradient
from .qfi import QFI

__all__ = ['DerivativeBase',
           'CircuitGradientMethod',
           'Gradient',
           'NaturalGradient',
           'Hessian',
           'QFI',
           'LinCombGradient',
           'ParamShiftGradient',
           'QFIMethod',
           'LinCombQFI',
           'OverlapQFI']
