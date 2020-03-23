# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Ansatz class."""

import warnings
from typing import Union, Optional, List, Callable
from qiskit.circuit_.library import TwoLocalCircuit

from ..initial_states.initial_state import InitialState


class TwoLocalAnsatz(TwoLocalCircuit):
    """The two-local Ansatz class."""

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 depth: int = 3,
                 rotation_gates: Optional[Union[str, List[str], type, List[type]]] = None,
                 entanglement_gates: Optional[Union[str, List[str], type, List[type]]] = None,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 initial_state: Optional[InitialState] = None,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 ) -> None:
        # insert barriers in between the blocks?
        warnings.warn('Hey we moved to TwoLocalCircuit instead of TwoLocalAnsatz!')
        super().__init__(num_qubits, depth, rotation_gates, entanglement_gates, entanglement,
                         initial_state, skip_unentangled_qubits, skip_final_rotation_layer,
                         parameter_prefix, insert_barriers)
