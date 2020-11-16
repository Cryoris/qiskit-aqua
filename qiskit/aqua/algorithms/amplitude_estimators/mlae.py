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

"""The Maximum Likelihood Amplitude Estimation algorithm."""

from typing import Optional, List, Union, Tuple, Callable, Dict
import warnings
import logging
import numpy as np
from scipy.optimize import brute
from scipy.stats import norm, chi2

from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.utils.circuit_factory import CircuitFactory
from qiskit.aqua.utils.validation import validate_min
from .ae_algorithm import AmplitudeEstimationAlgorithm, AmplitudeEstimationAlgorithmResult
from .estimation_problem import EstimationProblem

logger = logging.getLogger(__name__)


class MaximumLikelihoodAmplitudeEstimation(AmplitudeEstimationAlgorithm):
    """The Maximum Likelihood Amplitude Estimation algorithm.

    This class implements the quantum amplitude estimation (QAE) algorithm without phase
    estimation, as introduced in [1]. In comparison to the original QAE algorithm [2],
    this implementation relies solely on different powers of the Grover operator and does not
    require additional evaluation qubits.
    Finally, the estimate is determined via a maximum likelihood estimation, which is why this
    class in named ``MaximumLikelihoodAmplitudeEstimation``.

    References:
        [1]: Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N. (2019).
             Amplitude Estimation without Phase Estimation.
             `arXiv:1904.10246 <https://arxiv.org/abs/1904.10246>`_.
        [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(self, num_oracle_circuits: int,
                 state_preparation: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 grover_operator: Optional[Union[QuantumCircuit, CircuitFactory]] = None,
                 objective_qubits: Optional[List[int]] = None,
                 post_processing: Optional[Callable[[float], float]] = None,
                 a_factory: Optional[CircuitFactory] = None,
                 q_factory: Optional[CircuitFactory] = None,
                 i_objective: Optional[int] = None,
                 likelihood_evals: Optional[int] = None,
                 quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        r"""
        Args:
            num_oracle_circuits: The number of circuits applying different powers of the Grover
                oracle Q. The (`num_oracle_circuits` + 1) executed circuits will be
                `[id, Q^2^0, ..., Q^2^{num_oracle_circuits-1}] A |0>`, where A is the problem
                unitary encoded in the argument `a_factory`.
                Has a minimum value of 1.
            state_preparation: A circuit preparing the input state, referred to as
                :math:`\mathcal{A}`.
            grover_operator: The Grover operator :math:`\mathcal{Q}` used as unitary in the
                phase estimation circuit.
            objective_qubits: A list of qubit indices. A measurement outcome is classified as
                'good' state if all objective qubits are in state :math:`|1\rangle`, otherwise it
                is classified as 'bad'.
            post_processing: A mapping applied to the estimate of :math:`0 \leq a \leq 1`,
                usually used to map the estimate to a target interval.
            a_factory: The CircuitFactory subclass object representing the problem unitary.
            q_factory: The CircuitFactory subclass object representing.
                an amplitude estimation sample (based on a_factory)
            i_objective: The index of the objective qubit, i.e. the qubit marking 'good' solutions
                with the state \|1> and 'bad' solutions with the state \|0>
            likelihood_evals: The number of gridpoints for the maximum search of the likelihood
                function
            quantum_instance: Quantum Instance or Backend
        """
        validate_min('num_oracle_circuits', num_oracle_circuits, 1)

        # support legacy input if passed as positional arguments
        if isinstance(state_preparation, CircuitFactory):
            a_factory = state_preparation
            state_preparation = None

        if isinstance(grover_operator, CircuitFactory):
            q_factory = grover_operator
            grover_operator = None

        if isinstance(objective_qubits, int):
            i_objective = objective_qubits
            objective_qubits = None

        super().__init__(state_preparation=state_preparation,
                         grover_operator=grover_operator,
                         objective_qubits=objective_qubits,
                         post_processing=post_processing,
                         a_factory=a_factory,
                         q_factory=q_factory,
                         i_objective=i_objective,
                         quantum_instance=quantum_instance)

        # get parameters
        self._evaluation_schedule = [0] + [2**j for j in range(num_oracle_circuits)]

        self._likelihood_evals = likelihood_evals
        # default number of evaluations is max(10^5, pi/2 * 10^3 * 2^(m))
        if likelihood_evals is None:
            default = 10000
            self._likelihood_evals = np.maximum(default,
                                                int(np.pi / 2 * 1000 * 2 ** num_oracle_circuits))

        self._circuits = []  # type: List[QuantumCircuit]

        # TODO remove this, once ``self.confidence_interval()`` has reached the end of deprecation`
        self._last_result = None  # type: MaximumLikelihoodAmplitudeEstimationResult

    def construct_circuits(self, estimation_problem: Optional[EstimationProblem] = None,
                           measurement: bool = False) -> List[QuantumCircuit]:
        """Construct the Amplitude Estimation w/o QPE quantum circuits.

        Args:
            estimation_problem: The estimation problem for which to construct the QAE circuit.
            measurement: Boolean flag to indicate if measurement should be included in the circuits.

        Returns:
            A list with the QuantumCircuit objects for the algorithm.
        """
        if isinstance(estimation_problem, bool):
            warnings.warn('The first argument of construct_circuit is now an estimation problem.')
            measurement = estimation_problem
        elif estimation_problem is None:
            warnings.warn('In future, construct_circuit must be passed an estimation problem.')

        if estimation_problem is None:
            estimation_problem = EstimationProblem(self.state_preparation, self.grover_operator,
                                                   self.objective_qubits)

        # keep track of the Q-oracle queries
        circuits = []

        num_qubits = max(estimation_problem.state_preparation.num_qubits,
                         estimation_problem.grover_operator.num_qubits)
        q = QuantumRegister(num_qubits, 'q')
        qc_0 = QuantumCircuit(q, name='qc_a')  # 0 applications of Q, only a single A operator

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            qc_0.add_register(c)

        qc_0.compose(estimation_problem.state_preparation, inplace=True)

        for k in self._evaluation_schedule:
            qc_k = qc_0.copy(name='qc_a_q_%s' % k)

            if k != 0:
                qc_k.compose(estimation_problem.grover_operator.power(k), inplace=True)

            if measurement:
                # real hardware can currently not handle operations after measurements,
                # which might happen if the circuit gets transpiled, hence we're adding
                # a safeguard-barrier
                qc_k.barrier()
                qc_k.measure(estimation_problem.objective_qubits, *c)

            circuits += [qc_k]

        return circuits

    def _get_hits(self, num_state_qubits: int,
                  circuit_results: List[Union[np.ndarray, List[float], Dict[str, int]]],
                  is_good_state: Callable[[str], bool],
                  objective_qubits: List[int],
                  ) -> Tuple[List[float], List[int]]:
        """Get the good and total counts.

        Returns:
            A pair of two lists, ([1-counts per experiment], [shots per experiment]).

        Raises:
            AquaError: If self.run() has not been called yet.
        """
        one_hits = []  # h_k: how often 1 has been measured, for a power Q^(m_k)
        all_hits = []  # shots_k: how often has been measured at a power Q^(m_k)
        if all(isinstance(data, (list, np.ndarray)) for data in circuit_results):
            probabilities = []
            num_qubits = int(np.log2(len(circuit_results[0])))  # the total number of qubits
            for statevector in circuit_results:
                p_k = 0.0
                for i, amplitude in enumerate(statevector):
                    probability = np.abs(amplitude) ** 2
                    # consider only state qubits and revert bit order
                    bitstr = bin(i)[2:].zfill(num_qubits)[-num_state_qubits:][::-1]
                    objectives = [bitstr[index] for index in objective_qubits]
                    if is_good_state(objectives):
                        p_k += probability
                probabilities += [p_k]

            one_hits = probabilities
            all_hits = np.ones_like(one_hits)
        else:
            for counts in circuit_results:
                one_hits += [counts.get('1', 0)]  # return 0 if no key '1' found  #
                all_hits += [sum(counts.values())]

        return one_hits, all_hits

    @staticmethod
    def _compute_fisher_information(result: 'MaximumLikelihoodAmplitudeEstimationResult',
                                    num_sum_terms: Optional[int] = None,
                                    observed: bool = False) -> float:
        """Compute the Fisher information.

        Args:
            result: A maximum likelihood amplitude estimation result.
            num_sum_terms: The number of sum terms to be included in the calculation of the
                Fisher information. By default all values are included.
            observed: If True, compute the observed Fisher information, otherwise the theoretical
                one.

        Returns:
            The computed Fisher information, or np.inf if statevector simulation was used.

        Raises:
            KeyError: Call run() first!
        """
        a = result.a_estimation

        # Corresponding angle to the value a (only use real part of 'a')
        theta_a = np.arcsin(np.sqrt(np.real(a)))

        # Get the number of hits (shots_k) and one-hits (h_k)
        one_hits = result.one_hits
        all_hits = result.all_hits

        # Include all sum terms or just up to a certain term?
        evaluation_schedule = result.evaluation_schedule
        if num_sum_terms is not None:
            evaluation_schedule = evaluation_schedule[:num_sum_terms]
            # not necessary since zip goes as far as shortest list:
            # all_hits = all_hits[:num_sum_terms]
            # one_hits = one_hits[:num_sum_terms]

        # Compute the Fisher information
        fisher_information = None
        if observed:
            # Note, that the observed Fisher information is very unreliable in this algorithm!
            d_loglik = 0
            for shots_k, h_k, m_k in zip(all_hits, one_hits, evaluation_schedule):
                tan = np.tan((2 * m_k + 1) * theta_a)
                d_loglik += (2 * m_k + 1) * (h_k / tan + (shots_k - h_k) * tan)

            d_loglik /= np.sqrt(a * (1 - a))
            fisher_information = d_loglik ** 2 / len(all_hits)

        else:
            fisher_information = sum(shots_k * (2 * m_k + 1)**2
                                     for shots_k, m_k in zip(all_hits, evaluation_schedule))
            fisher_information /= a * (1 - a)

        return fisher_information

    @staticmethod
    def compute_confidence_interval(result: 'MaximumLikelihoodAmplitudeEstimationResult',
                                    alpha: float, kind: str = 'fisher') -> Tuple[float, float]:
        # pylint: disable=wrong-spelling-in-docstring
        """Compute the `alpha` confidence interval using the method `kind`.

        The confidence level is (1 - `alpha`) and supported kinds are 'fisher',
        'likelihood_ratio' and 'observed_fisher' with shorthand
        notations 'fi', 'lr' and 'oi', respectively.

        Args:
            result: A maximum likelihood amplitude estimation result.
            alpha: The confidence level.
            kind: The method to compute the confidence interval. Defaults to 'fisher', which
                computes the theoretical Fisher information.

        Returns:
            The specified confidence interval.

        Raises:
            AquaError: If `run()` hasn't been called yet.
            NotImplementedError: If the method `kind` is not supported.
        """
        # if statevector simulator the estimate is exact
        if all(isinstance(data, (list, np.ndarray)) for data in result.circuit_results):
            return 2 * [result.estimation]

        if kind in ['likelihood_ratio', 'lr']:
            return _likelihood_ratio_confint(result, alpha)

        if kind in ['fisher', 'fi']:
            return _fisher_confint(result, alpha, observed=False)

        if kind in ['observed_fisher', 'observed_information', 'oi']:
            return _fisher_confint(result, alpha, observed=True)

        raise NotImplementedError('CI `{}` is not implemented.'.format(kind))

    def confidence_interval(self, alpha: float, kind: str = 'fisher') -> List[float]:
        # pylint: disable=wrong-spelling-in-docstring
        """Compute the `alpha` confidence interval using the method `kind`.

        The confidence level is (1 - `alpha`) and supported kinds are 'fisher',
        'likelihood_ratio' and 'observed_fisher' with shorthand
        notations 'fi', 'lr' and 'oi', respectively.

        Args:
            alpha: The confidence level.
            kind: The method to compute the confidence interval. Defaults to 'fisher', which
                computes the theoretical Fisher information.

        Returns:
            The specified confidence interval.

        Raises:
            AquaError: If `run()` hasn't been called yet.
            NotImplementedError: If the method `kind` is not supported.
        """
        return list(self.compute_confidence_interval(self._last_result, alpha, kind))

    @staticmethod
    def compute_mle(result: 'MaximumLikelihoodAmplitudeEstimationResult') -> float:
        """Compute the MLE via a grid-search.

        This is a stable approach if sufficient gridpoints are used.

        Args:
            result: An amplitude estimation result object.

        Returns:
            The MLE for the provided result object.
        """
        one_hits = result.one_hits
        all_hits = result.all_hits

        # search range
        eps = 1e-15  # to avoid invalid value in log
        search_range = [0 + eps, np.pi / 2 - eps]

        def loglikelihood(theta):
            # loglik contains the first `it` terms of the full loglikelihood
            loglik = 0
            for i, k in enumerate(result.evaluation_schedule):
                loglik += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_hits[i]
                loglik += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_hits[i] - one_hits[i])
            return -loglik

        est_theta = brute(loglikelihood, [search_range], Ns=result.likelihood_nevals)[0]
        return est_theta

    def estimate(self, estimation_problem: EstimationProblem
                 ) -> 'MaximumLikelihoodAmplitudeEstimationResult':
        if estimation_problem.state_preparation is None:
            raise AquaError('Either the state_preparation variable or the a_factory '
                            '(deprecated) must be set to run the algorithm.')

        result = MaximumLikelihoodAmplitudeEstimationResult()
        result.likelihood_nevals = self._likelihood_evals
        result.evaluation_schedule = self._evaluation_schedule
        result.post_processing = estimation_problem.post_processing

        if self._quantum_instance.is_statevector:

            # run circuit on statevector simulator
            circuits = self.construct_circuits(estimation_problem, measurement=False)
            ret = self._quantum_instance.execute(circuits)

            # get statevectors and construct MLE input
            statevectors = [np.asarray(ret.get_statevector(circuit)) for circuit in circuits]
            result.circuit_results = statevectors

            # to count the number of Q-oracle calls (don't count shots)
            result.shots = 1

        else:
            # run circuit on QASM simulator
            circuits = self.construct_circuits(estimation_problem, measurement=True)
            ret = self._quantum_instance.execute(circuits)

            # get counts and construct MLE input
            result.circuit_results = [ret.get_counts(circuit) for circuit in circuits]

            # to count the number of Q-oracle calls
            result.shots = self._quantum_instance._run_config.shots

        num_state_qubits = circuits[0].num_qubits - circuits[0].num_ancillas
        one_hits, all_hits = self._get_hits(num_state_qubits, result.circuit_results,
                                            estimation_problem.is_good_state,
                                            estimation_problem.objective_qubits)
        result.one_hits = one_hits
        result.all_hits = all_hits

        # run maximum likelihood estimation and construct results
        result.theta = self.compute_mle(result)
        result.a_estimation = np.sin(result.theta)**2
        result.estimation = result.post_processing(result.a_estimation)
        result.fisher_information = self._compute_fisher_information(result)
        result.num_oracle_queries = result.shots * sum(k for k in result.evaluation_schedule)

        confidence_interval = self.compute_confidence_interval(result, alpha=0.05, kind='fisher')
        result.confidence_interval = confidence_interval

        return result

    # def estimate

    # circuits = self.construct_circuits()

    # res = execute(circuits, backend).result()
    # circuit_results = [res.get_counts(circ) for circ in circuits]

    # theta = self.compute_mle(circuit_results, likelihood_nevals=None)

    def _run(self) -> 'MaximumLikelihoodAmplitudeEstimationResult':
        # check if A factory or state_preparation has been set
        if self.state_preparation is None:
            if self._a_factory is None:
                raise AquaError('Either the state_preparation variable or the a_factory '
                                '(deprecated) must be set to run the algorithm.')

        estimation_problem = EstimationProblem(self.state_preparation, self.grover_operator,
                                               self.objective_qubits, self.post_processing)

        result = self.estimate(estimation_problem)
        self._last_result = result

        return result


class MaximumLikelihoodAmplitudeEstimationResult(AmplitudeEstimationAlgorithmResult):
    """ MaximumLikelihoodAmplitudeEstimation Result."""

    @ property
    def circuit_results(self) -> Optional[Union[List[np.ndarray], List[Dict[str, int]]]]:
        """ return circuit results """
        return self.get('circuit_results')

    @ circuit_results.setter
    def circuit_results(self, value: Union[List[np.ndarray], List[Dict[str, int]]]) -> None:
        """ set circuit results """
        self.data['circuit_results'] = value

    @ property
    def theta(self) -> float:
        """ returns theta """
        return self.get('theta')

    @ theta.setter
    def theta(self, value: float) -> None:
        """ set theta """
        self.data['theta'] = value

    @ property
    def likelihood_nevals(self) -> int:
        """ returns theta """
        return self.get('likelihood_nevals')

    @ likelihood_nevals.setter
    def likelihood_nevals(self, value: int) -> None:
        """ set theta """
        self.data['likelihood_nevals'] = value

    @ property
    def one_hits(self) -> List[float]:
        """ returns the percentage of one hits per circuit power """
        return self.get('one_hits')

    @ one_hits.setter
    def one_hits(self, hits: List[float]) -> None:
        """ sets the percentage of one hits per circuit power """
        self.data['one_hits'] = hits

    @ property
    def all_hits(self) -> List[int]:
        """ returns all hits per circuit power """
        return self.get('all_hits')

    @ all_hits.setter
    def all_hits(self, hits: List[int]) -> None:
        """ sets all hits per circuit power """
        self.data['all_hits'] = hits

    @ property
    def evaluation_schedule(self) -> List[int]:
        """ returns the evaluation schedule """
        return self.get('evaluation_schedule')

    @ evaluation_schedule.setter
    def evaluation_schedule(self, evaluation_schedule: List[int]) -> None:
        """ sets the evaluation schedule """
        self.data['evaluation_schedule'] = evaluation_schedule

    @ property
    def fisher_information(self) -> float:
        """ return fisher_information  """
        return self.get('fisher_information')

    @ fisher_information.setter
    def fisher_information(self, value: float) -> None:
        """ set fisher_information """
        self.data['fisher_information'] = value

    @ staticmethod
    def from_dict(a_dict: Dict) -> 'MaximumLikelihoodAmplitudeEstimationResult':
        """ create new object from a dictionary """
        return MaximumLikelihoodAmplitudeEstimationResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == 'statevectors':
            warnings.warn('statevectors deprecated, use circuit_results property.',
                          DeprecationWarning)
            return super().__getitem__('circuit_results')
        elif key == 'counts':
            warnings.warn('counts deprecated, use circuit_results property.', DeprecationWarning)
            return super().__getitem__('circuit_results')

        return super().__getitem__(key)


def _safe_min(array, default=0):
    if len(array) == 0:
        return default
    return np.min(array)


def _safe_max(array, default=(np.pi / 2)):
    if len(array) == 0:
        return default
    return np.max(array)


def _fisher_confint(result: MaximumLikelihoodAmplitudeEstimationResult,
                    alpha: float = 0.05,
                    observed: bool = False) -> Tuple[float, float]:
    """Compute the `alpha` confidence interval based on the Fisher information.

    Args:
        result: A maximum likelihood amplitude estimation results object.
        alpha: The level of the confidence interval (must be <= 0.5), default to 0.05.
        observed: If True, use observed Fisher information.

    Returns:
        float: The alpha confidence interval based on the Fisher information
    Raises:
        AssertionError: Call run() first!
    """
    # Get the (observed) Fisher information
    fisher_information = None
    try:
        fisher_information = result.fisher_information
    except KeyError as ex:
        raise AssertionError("Call run() first!") from ex

    if observed:
        fisher_information = MaximumLikelihoodAmplitudeEstimation._compute_fisher_information(
            result, observed=True)

    normal_quantile = norm.ppf(1 - alpha / 2)
    confint = np.real(result.a_estimation) + \
        normal_quantile / np.sqrt(fisher_information) * np.array([-1, 1])
    mapped_confint = tuple(result.post_processing(bound) for bound in confint)
    return mapped_confint


def _likelihood_ratio_confint(result: MaximumLikelihoodAmplitudeEstimationResult,
                              alpha: float = 0.05,
                              nevals: Optional[int] = None) -> List[float]:
    """Compute the likelihood-ratio confidence interval.

    Args:
        result: A maximum likelihood amplitude estimation results object.
        alpha: The level of the confidence interval (< 0.5), defaults to 0.05.
        nevals: The number of evaluations to find the intersection with the loglikelihood
            function. Defaults to an adaptive value based on the maximal power of Q.

    Returns:
        The alpha-likelihood-ratio confidence interval.
    """
    if nevals is None:
        nevals = result.likelihood_nevals

    def loglikelihood(theta, one_counts, all_counts):
        loglik = 0
        for i, k in enumerate(result.evaluation_schedule):
            loglik += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_counts[i]
            loglik += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_counts[i] - one_counts[i])
        return loglik

    one_counts = result.one_hits
    all_counts = result.all_hits

    eps = 1e-15  # to avoid invalid value in log
    thetas = np.linspace(0 + eps, np.pi / 2 - eps, nevals)
    values = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        values[i] = loglikelihood(theta, one_counts, all_counts)

    loglik_mle = loglikelihood(result.theta, one_counts, all_counts)
    chi2_quantile = chi2.ppf(1 - alpha, df=1)
    thres = loglik_mle - chi2_quantile / 2

    # the (outer) LR confidence interval
    above_thres = thetas[values >= thres]

    # it might happen that the `above_thres` array is empty,
    # to still provide a valid result use safe_min/max which
    # then yield [0, pi/2]
    confint = [_safe_min(above_thres, default=0),
               _safe_max(above_thres, default=np.pi / 2)]
    mapped_confint = tuple(result.post_processing(np.sin(bound) ** 2) for bound in confint)

    return mapped_confint
