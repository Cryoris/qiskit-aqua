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
from typing import Union, Optional, List
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Instruction
from qiskit.circuit_library.n_local_circuit import NLocalCircuit
from ..initial_states.initial_state import InitialState


class Ansatz(NLocalCircuit):
    """The Ansatz class."""

    def __init__(self,
                 blocks: Optional[Union[QuantumCircuit, List[QuantumCircuit],
                                        Instruction, List[Instruction]]] = None,
                 entanglement: Optional[Union[List[int], List[List[int]]]] = None,
                 reps: Optional[Union[int, List[int]]] = None,
                 insert_barriers: bool = False,
                 parameter_prefix: str = 'θ',
                 overwrite_block_parameters: Union[bool, List[List[Parameter]]] = True,
                 initial_state: Optional[InitialState] = None) -> None:
        # insert barriers in between the blocks?
        warnings.warn('Hey we moved to NLocalCircuit instead of Ansatz!')
        super().__init__(blocks, entanglement, reps, insert_barriers, parameter_prefix,
                         overwrite_block_parameters, initial_state)
