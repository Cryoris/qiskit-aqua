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

"""Faster Amplitude Estimation."""

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.aqua.aqua_error import AquaError
from qiskit.aqua.utils import CircuitFactory
from .ae_algorithm import AmplitudeEstimationAlgorithm
from .q_factory import QFactory

# pylint:disable=invalid-name


class FasterAmplitudeEstimation(AmplitudeEstimationAlgorithm):
    """

    References:

        [1]: `arXiv:2002.02417 <https://arxiv.org/pdf/2003.02417.pdf>`_
    """

    def __init__(self, N, delta, maxiter, a_factory=None, q_factory=None, x_factory=None,
                 i_objective=None, quantum_instance=None):
        super().__init__(a_factory, q_factory, i_objective, quantum_instance)
        self._N = N
        self._delta = delta
        self._maxiter = maxiter
        self._num_oracle_calls = 0
        self._x_factory = x_factory

    @property
    def x_factory(self):
        """X = A x R"""
        if self._x_factory is not None:
            return self._x_factory

        if self._a_factory is not None:
            return XFactory(self._a_factory)

        return None

    @property
    def q_factory(self):
        """Special Q operator for Faster AE"""
        if self._q_factory is not None:
            return self._q_factory

        if self._a_factory is not None:
            s11 = S11Factory(self._a_factory.num_target_qubits)
            return QFactory(self.x_factory, s_psi_0_factory=s11)

        return None

    @property
    def i_objective(self):
        """Objective qubits"""
        if self._i_objective is not None:
            return self._i_objective

        if self._a_factory is not None:
            return [self._a_factory.num_target_qubits - 1, self._a_factory.num_target_qubits]

        return None

    def _cos_estimate(self, k, shots):
        if self._quantum_instance is None:
            raise AquaError('Quantum instance must be set.')

        if self._quantum_instance.is_statevector:
            raise NotImplementedError
        else:
            circuit = self.construct_circuit(k, measurement=True)
            self._quantum_instance.run_config.shots = shots
            counts = self._quantum_instance.execute(circuit).get_counts()
            self._num_oracle_calls += (2 * k + 1) * shots
            cos_estimate = 1 - 2 * counts.get('11', 0) / shots

        return cos_estimate

    def _churnoff(self, cos, shots):
        width = np.sqrt(np.log(2 / self._delta) * 12 / shots)
        confint = [np.maximum(-1, cos - width), np.minimum(1, cos + width)]
        return confint

    def construct_circuit(self, k: int, measurement: bool = False) -> QuantumCircuit:
        r"""Construct the circuit Q^k X \|0>.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit Q^k X \|0>.
        """
        # set up circuit
        q = QuantumRegister(self.x_factory.num_target_qubits, 'q')
        circuit = QuantumCircuit(q, name='circuit')

        # get number of ancillas and add register if needed
        num_ancillas = np.maximum(self.x_factory.required_ancillas(),
                                  self.q_factory.required_ancillas())

        q_aux = None
        # pylint: disable=comparison-with-callable
        if num_ancillas > 0:
            q_aux = QuantumRegister(num_ancillas, 'aux')
            circuit.add_register(q_aux)

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(2)
            # c = ClassicalRegister(1)
            circuit.add_register(c)

        # add A operator
        self.x_factory.build(circuit, q, q_aux)

        # add Q^k
        if k != 0:
            self.q_factory.build_power(circuit, q, k, q_aux)

        # add optional measurement
        if measurement:
            # real hardware can currently not handle operations after measurements, which might
            # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
            circuit.barrier()
            circuit.measure([q[self.i_objective[0]], q[self.i_objective[1]]], [c[0], c[1]])
            # circuit.measure(q[self.i_objective[0]], c[0])

        return circuit

    def value_to_estimation(self, value):
        return self._a_factory.value_to_estimation(value)

    def _run(self):
        self._num_oracle_calls = 0
        theta_ci = [0, np.arcsin(0.25)]
        first_stage = True

        theta_cis = []
        num_first_stage_steps = 0
        num_steps = 0

        for j in range(1, self._maxiter + 1):
            num_steps += 1
            if first_stage:
                num_first_stage_steps += 1
                c = self._cos_estimate(2**(j - 1), self._N[0])
                ci = self._churnoff(c, self._N[0])
                theta_ci = [np.arccos(x) / (2 ** (j + 1) + 2) for x in ci[::-1]]

                if 2 ** (j + 1) * theta_ci[1] >= 3 * np.pi / 8 and j < self._maxiter:
                    j0 = j
                    v = 2 ** j * np.sum(theta_ci)
                    first_stage = False
            else:
                c = self._cos_estimate(2**(j - 1), self._N[1])
                c2 = self._cos_estimate(2 ** (j - 1) + 2 ** (j0 - 1), self._N[1])
                s = (c * np.cos(v) - c2) / np.sin(v)
                rho = np.arctan2(s, c)
                n = int(((2 ** (j + 1) + 2) * theta_ci[1] - rho) / (2 * np.pi))

                theta_ci = [(2 * np.pi * n + rho + sign * np.pi / 2) / (2 ** (j + 1) + 2)
                            for sign in [-1, 1]]
            theta_cis.append(theta_ci)

        theta = np.mean(theta_ci)
        value = (4 * np.sin(theta)) ** 2
        value_ci = [(4 * np.sin(x)) ** 2 for x in theta_ci]

        results = {
            'theta': theta,
            'theta_ci': theta_ci,
            'value': value,
            'value_ci': value_ci,
            'estimate': self.value_to_estimation(value),
            'confint': [self.value_to_estimation(x) for x in value_ci],
            'num_oracle_calls': self._num_oracle_calls,
            'theta_cis': theta_cis,
            'num_steps': num_steps,
            'num_first_stage_steps': num_first_stage_steps,
        }

        return results


class XFactory(CircuitFactory):
    """X = A x R"""

    def __init__(self, a_factory):
        super().__init__(a_factory.num_target_qubits + 1)
        self.a_factory = a_factory

    # pylint: disable=unused-argument
    def build(self, qc, q, q_ancillas=None, params=None):
        self.a_factory.build(qc, q, q_ancillas)
        qc.ry(2 * np.arcsin(0.25), q[self.num_target_qubits - 1])


class S11Factory(CircuitFactory):
    """Flip if the last two qubits are in |11>"""

    def required_ancillas(self):
        return 0

    def required_ancillas_controlled(self):
        return 0

    # pylint:disable=unused-argument
    def build(self, qc, q, q_ancillas=None, params=None):
        qc.cz(q[self.num_target_qubits - 1], q[self.num_target_qubits])
