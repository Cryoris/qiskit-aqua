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

""" Test VQE """

import unittest
import warnings
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, unpack, data
from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import RY, RYRZ

from qiskit.aqua import QuantumInstance, aqua_globals, AquaError
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.aqua.components.variational_forms import RY as VarRY, RYRZ as VarRYRZ
from qiskit.aqua.components.optimizers import L_BFGS_B, COBYLA, SPSA, SLSQP
from qiskit.aqua.algorithms import VQE


@ddt
class TestVQE(QiskitAquaTestCase):
    """ Test VQE """

    def setUp(self):
        super().setUp()
        self.seed = 50
        aqua_globals.random_seed = self.seed
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict)

        num_qubits = self.qubit_op.num_qubits
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        self.ryrz_wavefunction = {'wrapped': VarRYRZ(num_qubits),
                                  'circuit': QuantumCircuit(num_qubits).compose(RYRZ(num_qubits)),
                                  'library': RYRZ()}

        self.ry_wavefunction = {'wrapped': VarRY(num_qubits),
                                'circuit': QuantumCircuit(num_qubits).compose(RY(num_qubits)),
                                'library': RY()}
        warnings.filterwarnings('always', category=DeprecationWarning)

    @data('wrapped', 'circuit', 'library')
    def test_vqe(self, mode):
        """ VQE test """
        wavefunction = self.ryrz_wavefunction[mode]
        if mode == 'wrapped':
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        vqe = VQE(self.qubit_op, wavefunction, L_BFGS_B())
        if mode == 'wrapped':
            warnings.filterwarnings('always', category=DeprecationWarning)

        result = vqe.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                                         coupling_map=[[0, 1]],
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503)
        np.testing.assert_array_almost_equal(result.eigenvalue.real, -1.85727503, 5)
        self.assertEqual(len(result.optimal_point), 16)
        self.assertIsNotNone(result.cost_function_evals)
        self.assertIsNotNone(result.optimizer_time)

    def test_vqe_no_varform_params(self):
        """Test specifying a variational form with no parameters raises an error."""
        circuit = QuantumCircuit(self.qubit_op.num_qubits)
        for i in range(circuit.num_qubits):
            circuit.h(i)
            circuit.cx(i, (i + 1) % circuit.num_qubits)
            circuit.rx(0.2, i)

        vqe = VQE(self.qubit_op, circuit)
        with self.assertRaises(RuntimeError):
            vqe.run(BasicAer.get_backend('statevector_simulator'))

    @data(
        (SLSQP, 5, 4),
        (SLSQP, 5, 1),
        (SPSA, 3, 2),  # max_evals_grouped=n is considered as max_evals_grouped=2 if n>2
        (SPSA, 3, 1)
    )
    @unpack
    def test_vqe_optimizers(self, optimizer_cls, places, max_evals_grouped):
        """ VQE Optimizers test """
        result = VQE(self.qubit_op,
                     RYRZ(),
                     optimizer_cls(),
                     max_evals_grouped=max_evals_grouped).run(
                         QuantumInstance(BasicAer.get_backend('statevector_simulator'), shots=1,
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))

        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=places)

    @data('wrapped', 'circuit', 'library')
    def test_vqe_qasm(self, mode):
        """ VQE QASM test """
        backend = BasicAer.get_backend('qasm_simulator')
        optimizer = SPSA(max_trials=300, last_avg=5)
        wavefunction = self.ry_wavefunction[mode]

        if mode == 'wrapped':
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        vqe = VQE(self.qubit_op, wavefunction, optimizer, max_evals_grouped=1)
        if mode == 'wrapped':
            warnings.filterwarnings('always', category=DeprecationWarning)

        quantum_instance = QuantumInstance(backend, shots=10000,
                                           seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        result = vqe.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=2)

    @data('wrapped', 'circuit', 'library')
    def test_vqe_statevector_snapshot_mode(self, mode):
        """ VQE Aer statevector_simulator snapshot mode test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('statevector_simulator')
        wavefunction = self.ry_wavefunction[mode]
        optimizer = L_BFGS_B()

        if mode == 'wrapped':
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        vqe = VQE(self.qubit_op, wavefunction, optimizer, max_evals_grouped=1)
        if mode == 'wrapped':
            warnings.filterwarnings('always', category=DeprecationWarning)

        quantum_instance = QuantumInstance(backend,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqe.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=6)

    @data('wrapped', 'circuit', 'library')
    def test_vqe_qasm_snapshot_mode(self, mode):
        """ VQE Aer qasm_simulator snapshot mode test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('qasm_simulator')
        optimizer = L_BFGS_B()
        wavefunction = self.ry_wavefunction[mode]

        if mode == 'wrapped':
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        vqe = VQE(self.qubit_op, wavefunction, optimizer, max_evals_grouped=1)
        if mode == 'wrapped':
            warnings.filterwarnings('always', category=DeprecationWarning)

        quantum_instance = QuantumInstance(backend, shots=1,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqe.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=6)

    @data('wrapped', 'circuit', 'library')
    def test_vqe_callback(self, mode):
        """ VQE Callback test """
        history = {'eval_count': [], 'parameters': [], 'mean': [], 'std': []}

        def store_intermediate_result(eval_count, parameters, mean, std):
            history['eval_count'].append(eval_count)
            history['parameters'].append(parameters)
            history['mean'].append(mean)
            history['std'].append(std)

        backend = BasicAer.get_backend('qasm_simulator')
        optimizer = COBYLA(maxiter=3)
        wavefunction = self.ry_wavefunction[mode]

        if mode == 'wrapped':
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        vqe = VQE(self.qubit_op, wavefunction, optimizer, callback=store_intermediate_result,
                  auto_conversion=False)
        if mode == 'wrapped':
            warnings.filterwarnings('always', category=DeprecationWarning)

        aqua_globals.random_seed = 50
        quantum_instance = QuantumInstance(backend,
                                           seed_transpiler=50,
                                           shots=1024,
                                           seed_simulator=50)
        vqe.run(quantum_instance)

        self.assertTrue(all(isinstance(count, int) for count in history['eval_count']))
        self.assertTrue(all(isinstance(mean, float) for mean in history['mean']))
        self.assertTrue(all(isinstance(std, float) for std in history['std']))
        for params in history['parameters']:
            self.assertTrue(all(isinstance(param, float) for param in params))

    def test_vqe_reuse(self):
        """ Test vqe reuse """
        vqe = VQE()
        with self.assertRaises(AquaError):
            _ = vqe.run()

        var_form = RY()
        vqe.var_form = var_form
        with self.assertRaises(AquaError):
            _ = vqe.run()

        vqe.operator = self.qubit_op
        with self.assertRaises(AquaError):
            _ = vqe.run()

        qinst = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        vqe.quantum_instance = qinst
        result = vqe.run()
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=5)

        operator = MatrixOperator(np.array([[1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, 2, 0],
                                            [0, 0, 0, 3]]))
        vqe.operator = operator
        result = vqe.run()
        self.assertAlmostEqual(result.eigenvalue.real, -1.0, places=5)

    def test_vqe_mes(self):
        """ Test vqe minimum eigen solver interface """
        vqe = VQE(var_form=RY(), optimizer=COBYLA())
        vqe.set_backend(BasicAer.get_backend('statevector_simulator'))
        result = vqe.compute_minimum_eigenvalue(self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=5)


if __name__ == '__main__':
    unittest.main()
