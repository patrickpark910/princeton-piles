"""Check the existing run method and CSV processing with transport mocked."""
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import openmc
import pandas as pd
from numpy.testing import assert_allclose
from scipy.sparse import csc_array

from python import gamma_saturation as gamma
from python.model import Pile


def saved_tally(name, filters, nuclides, scores, values):
    tally = openmc.Tally(name=name)
    tally.filters, tally.nuclides, tally.scores = filters, nuclides, scores
    tally._mean = np.asarray(values, dtype=float).reshape(tally.shape)
    tally._std_dev = np.zeros(tally.shape)
    return tally


class RunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        material = openmc.Material()
        material.add_nuclide('H1', 1.)
        boundary = MagicMock()
        boundary.__neg__.return_value.bounding_box = (np.zeros(3), np.ones(3))
        shells = [SimpleNamespace() for _ in range(5)]
        self.pile = SimpleNamespace(
            path=self.directory, run_openmc=True, n_cycles=10, source_rate=10.,
            model=Mock(), materials=SimpleNamespace(cross_sections='unused.xml'),
            settings=SimpleNamespace(source=[SimpleNamespace(energy=SimpleNamespace(x=[1e6]))]),
            geometry=SimpleNamespace(get_all_materials=lambda: {material.id: material}),
            pile_boundary=boundary, n_trisos=0, m_breeder=material,
            **dict(zip(('m_xo2', 'm_buff', 'm_PyC1', 'm_SiC', 'm_PyC2'), shells)))
        self.pile.openmc = MethodType(Pile.openmc, self.pile)
        self.pile.model.run.side_effect = AssertionError('Transport must not start')
        self.final = self.directory / 'statepoint.10.h5'
        chain = MagicMock()
        chain.__len__.return_value = 1
        chain.nuclides = [SimpleNamespace(name='H1', sources={
            'photon': openmc.stats.Discrete([1000.], [0.5])})]
        chain.reactions, chain.fission_yields = ['(n,gamma)'], [{}]
        chain.decay_matrix = chain.form_matrix.return_value = csc_array([[0.]])
        chain.reduce.return_value = chain
        operator = MagicMock(chain=chain)
        operator.initial_condition.return_value = [np.array([100.])]
        operator.return_value.rates = [None]
        material_filter = openmc.MaterialFilter([material])
        tallies = {
            'MicroXS flux 0': saved_tally('MicroXS flux 0', [material_filter],
                                        ['total'], ['flux'], [2.]),
            'MicroXS RR 0': saved_tally('MicroXS RR 0', [material_filter],
                                      ['H1'], ['(n,gamma)'], [4.]),
            'a02 prompt production': saved_tally('a02 prompt production',
                [openmc.ParticleFilter('neutron'),
                 openmc.ParticleProductionFilter('photon', energies=[0., 1000., 2000.])],
                ['H1'], ['events'], [2., 3.]),
        }
        for target, value in (
            ('Chain.from_xml', chain),
            ('openmc.data.DataLibrary.from_xml', [{'type': 'neutron', 'materials': ['H1']}]),
            ('IndependentOperator', operator),
        ):
            patcher = patch(f'python.gamma_saturation.{target}', return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch('python.gamma_saturation.openmc.StatePoint')
        self.statepoint = patcher.start()
        self.addCleanup(patcher.stop)
        self.statepoint.return_value.__enter__.return_value = SimpleNamespace(
            get_tally=Mock(side_effect=lambda name: tallies[name]))

    def test_existing_statepoint_reformats_csvs_without_transport(self):
        self.final.write_bytes(b'existing statepoint')
        intermediate = self.directory / 'statepoint.5.h5'
        intermediate.touch()
        gamma.run_saturation(self.pile, 'unused.xml', energy_bins=2)
        original = gamma.save_spectrum

        def reformat(path, *args):
            original(path, *args)
            table = pd.read_csv(path)
            table['format_version'] = 2
            table.to_csv(path, index=False)

        with patch.object(gamma, 'save_spectrum', side_effect=reformat):
            gamma.run_saturation(self.pile, 'unused.xml', energy_bins=7)
        self.pile.model.run.assert_not_called()
        self.assertEqual(self.final.read_bytes(), b'existing statepoint')
        self.assertTrue(intermediate.exists())
        for name in ('prompt', 'passive', 'daughter', 'total'):
            self.assertIn('format_version', pd.read_csv(self.directory / f'{name}.csv'))
        prompt = pd.read_csv(self.directory / 'prompt.csv')
        assert_allclose(prompt['E_mid_keV'], [0.5, 1.5])
        assert_allclose(prompt['photons_per_s'], [20., 30.])
        total = pd.read_csv(self.directory / 'total.csv')
        self.assertTrue(total['photons_per_s'].is_monotonic_decreasing)

    def test_new_run_uses_existing_intermediate_cleanup(self):
        intermediate = self.directory / 'statepoint.5.h5'

        def fake_run(cwd):
            self.assertEqual(cwd, self.directory)
            intermediate.touch()
            self.final.touch()
            return self.final

        self.pile.model.run.side_effect = fake_run
        gamma.run_saturation(self.pile, 'unused.xml')
        self.pile.model.run.assert_called_once_with(cwd=self.directory)
        self.assertFalse(intermediate.exists())
        self.assertTrue(self.final.exists())

    def test_no_run_reuses_existing_statepoint(self):
        self.pile.run_openmc = False
        self.final.touch()
        gamma.run_saturation(self.pile, 'unused.xml')
        self.pile.model.run.assert_not_called()
        self.assertTrue((self.directory / 'total.csv').is_file())

    def test_no_run_without_statepoint_stops_processing(self):
        self.pile.run_openmc = False
        gamma.run_saturation(self.pile, 'unused.xml')
        self.pile.model.run.assert_not_called()
        self.statepoint.assert_not_called()
        self.assertEqual(list(self.directory.glob('*.csv')), [])


class RateReaderTests(unittest.TestCase):
    def test_saved_material_order_and_zero_flux(self):
        material_filter = openmc.MaterialFilter([11, 22])
        flux = saved_tally('MicroXS flux 0', [material_filter], ['total'], ['flux'], [2., 0.])
        rates = saved_tally('MicroXS RR 0', [material_filter], ['He4', 'H1'],
                           ['(n,gamma)', '(n,2n)'], [8., 4., 6., 2., 10., 12., 14., 16.])
        statepoint = SimpleNamespace(get_tally=lambda name: flux if name == flux.name else rates)
        fluxes, micros = gamma.read_rate_data(statepoint, [SimpleNamespace(id=22), SimpleNamespace(id=11)])
        assert_allclose(fluxes, [[0.], [2.]])
        assert_allclose(micros[0].data, 0.)
        assert_allclose(micros[1].data[:, :, 0], [[4., 2.], [3., 1.]])
        self.assertEqual(micros[1].nuclides, ['He4', 'H1'])


if __name__ == '__main__':
    unittest.main()
