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

from typing import Optional, Union, Tuple, List
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.aqua_error import AquaError
from qiskit.aqua.utils import CircuitFactory
from .ae_algorithm import AmplitudeEstimationAlgorithm
from .q_factory import QFactory


class FasterAmplitudeEstimation(AmplitudeEstimationAlgorithm):
    """The Faster Amplitude Estimation algorithm.

    The Faster Amplitude Estimation (FAE) [1] algorithm is a variant of Quantum Amplitude
    Estimation (QAE), where the Quantum Phase Estimation (QPE) by an iterative Grover search,
    similar to [2].

    Due to the iterative version of the QPE, this algorithm does not require extra ancilla
    qubits, as the originally proposed QAE [3] and the resulting circuits are less complex.

    References:

        [1]: K. Nakaji. Faster Amplitude Estimation, 2020;
            `arXiv:2002.02417 <https://arxiv.org/pdf/2003.02417.pdf>`_
        [2]: D. Grinko et al. Iterative Amplitude Estimation, 2019;
            `arXiv:1912.05559 <http://arxiv.org/abs/1912.05559>`_
        [3]: G. Brassard et al. Quantum Amplitude Amplification and Estimation, 2000;
            `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_

    """

    def __init__(self, N: Optional[Tuple[int, int]] = None,
                 delta: Optional[float] = None,
                 maxiter: Optional[int] = None,
                 a_factory: Optional[CircuitFactory] = None,
                 q_factory: Optional[CircuitFactory] = None,
                 x_factory: Optional[CircuitFactory] = None,
                 i_objective: Optional[List[int]] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        r"""
        Args:
            N: The number of shots for the first and second stage as tuple.
            delta: The probability that the true value is outside of the final confidence interval.
            maxiter: The number of iterations, the maximal power of Q is `2 ** (maxiter - 1)`.
            a_factory: The A operator constructing the amplitudes.
            q_factory: The Q, or Grover, operator.
            x_factory: The X operator, defined as :math:`A \otimes R_y(\sin^{-1}(1/4))`.
            i_objective: The objective qubits, defaults to the last two qubits.
            quantum_instance: The quantum instance or backend to run the circuits.
        """
        super().__init__(a_factory, q_factory, i_objective, quantum_instance)
        self._shots = N
        self._delta = delta
        self._maxiter = maxiter
        self._num_oracle_calls = 0
        self._x_factory = x_factory

    @property
    def x_factory(self) -> Optional[CircuitFactory]:
        r"""The X operator, defined as :math:`A \otimes R_y(\sin^{-1}(1/4))`.

        Returns:
            The X operator, computed based on the current A operator, or None if the A operator
            is not defined.
        """
        if self._x_factory is not None:
            return self._x_factory

        if self._a_factory is not None:
            return XFactory(self._a_factory, scaling_factor=0.25)

        return None

    @x_factory.setter
    def x_factory(self, x_factory: CircuitFactory) -> None:
        """Set the X operator.

        Args:
            x_factory: The new X operator.
        """
        self._x_factory = x_factory

    @property
    def is_rescaled(self):
        """Check if the amplitude has been rescaled."""
        return self._q_factory is None

    @property
    def q_factory(self) -> Optional[CircuitFactory]:
        r"""The Q operator.

        Returns:
            The current Q operator, based on the A operator, of None if the A operator is not set.
        """
        if self._q_factory is not None:
            return self._q_factory

        if self._a_factory is not None:
            s11 = S11Factory(self.x_factory.num_target_qubits)
            return QFactory(self.x_factory, s_psi_0_factory=s11)

        return None

    @q_factory.setter
    def q_factory(self, q_factory):
        """
        Set the Q operator as QFactory.

        Args:
            q_factory (QFactory): the specialized Q operator
        """
        self._q_factory = q_factory

    @property
    def i_objective(self) -> Optional[List[int]]:
        """The objective qubits, if all these qubits are in state |1> the state is marked as good.

        Returns:
            The objective qubits, or None if not set and the A operator is not set.
        """
        if self._i_objective is not None:
            return self._i_objective

        if self._q_factory is not None:
            i_objective = self._q_factory.i_objective
            if not isinstance(i_objective, list):
                return [i_objective]
            return i_objective

        if self._a_factory is not None:
            return [self._a_factory.num_target_qubits - 1, self._a_factory.num_target_qubits]

        if self._x_factory is not None:
            return [self._x_factory.num_target_qubits - 2, self._x_factory.num_target_qubits - 1]

        return None

    @i_objective.setter
    def i_objective(self, i_objective: int) -> None:
        """Set the index of the objective qubit, i.e. the qubit deciding between 'good/bad' states.

        Args:
            i_objective: the index
        """
        self._i_objective = i_objective

    def _cos_estimate(self, k, shots=1024):
        if self._quantum_instance is None:
            raise AquaError('Quantum instance must be set.')

        if self._quantum_instance.is_statevector:
            circuit = self.construct_circuit(k, measurement=False)
            statevector = self._quantum_instance.execute(circuit).get_statevector()

            # sum over all amplitudes where the objective qubits are 1
            prob = 0
            for i, amplitude in enumerate(statevector):
                state = bin(i)[2:].zfill(circuit.num_qubits)[::-1]
                if all(state[i] == '1' for i in self.i_objective):
                    prob = prob + np.abs(amplitude)**2

            cos_estimate = 1 - 2 * prob
        else:
            circuit = self.construct_circuit(k, measurement=True)
            self._quantum_instance.run_config.shots = shots
            counts = self._quantum_instance.execute(circuit).get_counts()
            self._num_oracle_calls += (2 * k + 1) * shots
            cos_estimate = 1 - 2 * counts.get('1' * len(self.i_objective), 0) / shots

        return cos_estimate

    def _chernoff(self, cos, shots):
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
            num_objective = len(self.i_objective)
            c = ClassicalRegister(num_objective)
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
            circuit.measure([q[i] for i in self.i_objective], [c[i] for i in range(num_objective)])

        return circuit

    def _run(self):
        self._num_oracle_calls = 0

        if self._quantum_instance.is_statevector:
            cos = self._cos_estimate(k=0)
            theta = np.arccos(cos) / 2
            theta_ci = [theta, theta]
            theta_cis = [theta_ci]
            num_steps = num_first_stage_steps = 1

        else:
            theta_ci = [0, np.arcsin(0.25)]
            first_stage = True

            theta_cis = []
            num_first_stage_steps = 0
            num_steps = 0

            for j in range(1, self._maxiter + 1):
                num_steps += 1
                if first_stage:
                    num_first_stage_steps += 1
                    c = self._cos_estimate(2**(j - 1), self._shots[0])
                    chernoff_ci = self._chernoff(c, self._shots[0])
                    theta_ci = [np.arccos(x) / (2 ** (j + 1) + 2) for x in chernoff_ci[::-1]]

                    if 2 ** (j + 1) * theta_ci[1] >= 3 * np.pi / 8 and j < self._maxiter:
                        j_0 = j
                        v = 2 ** j * np.sum(theta_ci)
                        first_stage = False
                else:
                    cos = self._cos_estimate(2**(j - 1), self._shots[1])
                    cos_2 = self._cos_estimate(2 ** (j - 1) + 2 ** (j_0 - 1), self._shots[1])
                    sin = (c * np.cos(v) - cos_2) / np.sin(v)
                    rho = np.arctan2(sin, cos)
                    n = int(((2 ** (j + 1) + 2) * theta_ci[1] - rho) / (2 * np.pi))

                    theta_ci = [(2 * np.pi * n + rho + sign * np.pi / 2) / (2 ** (j + 1) + 2)
                                for sign in [-1, 1]]
                theta_cis.append(theta_ci)

        theta = np.mean(theta_ci)
        rescaling = 4 if self.is_rescaled else 1
        value = (rescaling * np.sin(theta)) ** 2
        value_ci = [(rescaling * np.sin(x)) ** 2 for x in theta_ci]

        if self._a_factory is not None:
            value_to_estimation = self._a_factory.value_to_estimation
        else:
            value_to_estimation = self._x_factory.value_to_estimation

        results = {
            'theta': theta,
            'theta_ci': theta_ci,
            'value': value,
            'value_ci': value_ci,
            'estimation': value_to_estimation(value),
            'confidence_interval': [value_to_estimation(x) for x in value_ci],
            'num_oracle_calls': self._num_oracle_calls,
            'theta_cis': theta_cis,
            'num_steps': num_steps,
            'num_first_stage_steps': num_first_stage_steps,
        }

        return results


class XFactory(CircuitFactory):
    """The X operator, rescales the amplitude of the A operator using an ancilla.

    The A operator in Quantum Amplitude Estimation performs the mapping

    .. math::

        todo

    Somce QAE variants, as Faster Amplitude Estimation and QAE Simplified, require the amplitude
    :math:`a` to be smaller than a certain threshold. This rescaling is achieved with with a Pauli-Y
    rotation on an ancilla qubit with an according rotation angle.
    """

    def __init__(self, a_factory: CircuitFactory, scaling_factor: float) -> None:
        """
        Args:
            a_factory: The A operator.
            scaling_factor: The rescaling factor, must be smaller than 1.
        """
        super().__init__(a_factory.num_target_qubits + 1)
        self.a_factory = a_factory
        self.scaling_factor = scaling_factor

    # pylint: disable=unused-argument
    def build(self, qc, q, q_ancillas=None, params=None):
        self.a_factory.build(qc, q, q_ancillas)
        qc.ry(2 * np.arcsin(self.scaling_factor), q[self.num_target_qubits - 1])


class S11Factory(CircuitFactory):
    """Flip if the last two qubits are in |11>"""

    # pylint:disable=unused-argument
    def build(self, qc, q, q_ancillas=None, params=None):
        qc.cz(q[self.num_target_qubits - 2], q[self.num_target_qubits - 1])
