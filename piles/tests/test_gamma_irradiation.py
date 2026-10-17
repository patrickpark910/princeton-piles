"""Check finite irradiation without a transport run or nuclear data library."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.sparse import csc_array

from python import gamma_saturation as gamma


class MatrixChain:
    def __init__(self, decay):
        self.decay_matrix = csc_array(decay)

    def __len__(self):
        return self.decay_matrix.shape[0]


class IrradiationTests(unittest.TestCase):
    def test_zero_time_retains_initial_passive_inventory(self):
        chain = MatrixChain([[-0.01, 0.], [0.01, 0.]])
        initial = np.array([100., 20.])
        atoms = gamma.irradiate(chain, chain.decay_matrix, initial, 0.)
        assert_allclose(atoms[:, 0], initial, rtol=0., atol=0.)
        assert_allclose(atoms[:, 1], 0., rtol=0., atol=0.)

    def test_decay_without_neutrons_stays_passive(self):
        decay = np.log(2.) / 1800.
        chain = MatrixChain([[-decay, 0.], [decay, 0.]])
        atoms = gamma.irradiate(chain, chain.decay_matrix,
                                np.array([100., 0.]), 3600.)
        assert_allclose(atoms[:, 0], [25., 75.], rtol=1e-12)
        assert_allclose(atoms[:, 1], 0., atol=1e-14)

    def test_activation_with_parent_depletion_matches_analytic_solution(self):
        rate, decay, time_s, n0 = 1e-5, np.log(2.) / 1800., 3600., 1e6
        chain = MatrixChain([[0., 0., 0.],
                             [0., -decay, 0.],
                             [0., decay, 0.]])
        reactions = csc_array([[-rate, 0., 0.],
                               [rate, 0., 0.],
                               [0., 0., 0.]])
        atoms = gamma.irradiate(chain, chain.decay_matrix + reactions,
                                np.array([n0, 0., 0.]), time_s)
        parent = n0 * np.exp(-rate * time_s)
        product = n0 * rate / (decay - rate) * (
            np.exp(-rate * time_s) - np.exp(-decay * time_s))
        assert_allclose(atoms[:, 0], [parent, 0., 0.], rtol=1e-12)
        assert_allclose(atoms[:, 1], [0., product, n0 - parent - product],
                        rtol=1e-12)

    def test_small_product_inventory_is_retained(self):
        rate, n0, time_s = 1e-30, 1e28, 3600.
        chain = MatrixChain(np.zeros((2, 2)))
        matrix = csc_array([[-rate, 0.], [rate, 0.]])
        atoms = gamma.irradiate(chain, matrix, np.array([n0, 0.]), time_s)
        expected = n0 * -np.expm1(-rate * time_s)
        assert_allclose(atoms[1, 1], expected, rtol=1e-12)
        assert_allclose(atoms[1, 0], 0., atol=0.)

    def test_component_sum_matches_full_chain_with_stable_activation(self):
        decay, capture, removal = 2e-4, 1e-4, 3e-4
        chain = MatrixChain([[-decay, 0., 0., 0.],
                             [decay, 0., 0., 0.],
                             [0., 0., -decay, 0.],
                             [0., 0., decay, 0.]])
        reactions = csc_array([[0., 0., 0., 0.],
                               [0., -capture, 0., 0.],
                               [0., capture, -removal, 0.],
                               [0., 0., removal, 0.]])
        matrix = chain.decay_matrix + reactions
        initial = np.array([1e6, 5000., 0., 0.])
        atoms = gamma.irradiate(chain, matrix, initial, 3600.)
        assert_allclose(atoms.sum(axis=1), expm(matrix.toarray() * 3600.) @ initial,
                        rtol=1e-12)
        self.assertGreater(atoms[2, 1], 0.)
        self.assertGreater(atoms[3, 1], 0.)
        assert_allclose(atoms[2:, 0], 0., atol=0.)
        self.assertTrue(np.all(atoms >= 0.))

    def test_invalid_duration_is_rejected(self):
        chain = MatrixChain([[0.]])
        for duration in (-1., np.nan, np.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                gamma.irradiate(chain, chain.decay_matrix, np.array([1.]), duration)


if __name__ == '__main__':
    unittest.main()
