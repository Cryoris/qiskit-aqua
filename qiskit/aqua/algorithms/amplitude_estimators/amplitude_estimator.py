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

"""The Amplitude Estimation interface."""

import warnings
from abc import abstractmethod
from typing import Union, Optional, Dict, Callable, Tuple
import numpy as np
from qiskit.providers import BaseBackend, Backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import AlgorithmResult, QuantumAlgorithm

from .estimation_problem import EstimationProblem


class AmplitudeEstimator(QuantumAlgorithm):
    """The Amplitude Estimation interface."""

    def __init__(self,
                 quantum_instance: Optional[Union[Backend, BaseBackend, QuantumInstance]] = None
                 ) -> None:
        super().__init__(quantum_instance)

    @abstractmethod
    def estimate(self, estimation_problem: EstimationProblem) -> 'AmplitudeEstimatorResult':
        """Run the amplitude estimation algorithm.

        Args:
            estimation_problem: An ``EstimationProblem`` describing
        """
        raise NotImplementedError


class AmplitudeEstimatorResult(AlgorithmResult):
    """The results object for amplitude estimation algorithms."""

    @property
    def circuit_result(self) -> Optional[Union[np.ndarray, Dict[str, int]]]:
        """ return circuit result """
        return self.get('circuit_result')

    @circuit_result.setter
    def circuit_result(self, value: Union[np.ndarray, Dict[str, int]]) -> None:
        """ set circuit result """
        self.data['circuit_result'] = value

    @property
    def shots(self) -> int:
        """ return shots """
        return self.get('shots')

    @shots.setter
    def shots(self, value: int) -> None:
        """ set shots """
        self.data['shots'] = value

    @property
    def a_estimation(self) -> float:
        """ return a_estimation """
        return self.get('a_estimation')

    @a_estimation.setter
    def a_estimation(self, value: float) -> None:
        """ set a_estimation """
        self.data['a_estimation'] = value

    @property
    def estimation(self) -> float:
        """ return estimation """
        return self.get('estimation')

    @estimation.setter
    def estimation(self, value: float) -> None:
        """ set estimation """
        self.data['estimation'] = value

    @property
    def num_oracle_queries(self) -> int:
        """ return num_oracle_queries """
        return self.get('num_oracle_queries')

    @num_oracle_queries.setter
    def num_oracle_queries(self, value: int) -> None:
        """ set num_oracle_queries """
        self.data['num_oracle_queries'] = value

    @property
    def post_processing(self) -> Callable[[float], float]:
        """ returns post_processing """
        return self._post_processing
        # return self.get('post_processing')

    @post_processing.setter
    def post_processing(self, post_processing: Callable[[float], float]) -> None:
        """ sets post_processing """
        self._post_processing = post_processing
        # self.data['post_processing'] = post_processing

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """ returns the confidence interval (95% by default) """
        return self.get('confidence_interval')

    @confidence_interval.setter
    def confidence_interval(self, confidence_interval: Tuple[float, float]) -> None:
        """ sets confidence interval """
        self.data['confidence_interval'] = confidence_interval

    @staticmethod
    def from_dict(a_dict: Dict) -> 'AmplitudeEstimationAlgorithmResult':
        """ create new object from a dictionary """
        return AmplitudeEstimatorResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == '95%_confidence_interval':
            warnings.warn('95%_confidence_interval deprecated, use confidence_interval property.',
                          DeprecationWarning)
            return super().__getitem__('confidence_interval')
        elif key == 'value':
            warnings.warn('value deprecated, use a_estimation property.', DeprecationWarning)
            return super().__getitem__('a_estimation')

        return super().__getitem__(key)
