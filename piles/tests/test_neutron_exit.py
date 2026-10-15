"""Check leakage against six directed beams with negligible scattering.

Run from the repository root:
    PYTHONPATH=piles .venv/bin/python -m unittest discover -s piles/tests -v
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import openmc

from python.neutron_exit import FACES, build_exit_tallies, export_exit_maps


class DirectedBeamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("OPENMC_CROSS_SECTIONS") or openmc.config.get("cross_sections")
        cls.xs = Path(configured) if configured else Path.home() / "openmc/data/endf-b8.1-hdf5/cross_sections.xml"
        if not shutil.which("openmc") or not cls.xs.is_file():
            raise unittest.SkipTest("OpenMC and a neutron cross-section library are required.")

    def run_beams(self, directory, energy_bounds=None):
        openmc.reset_auto_ids()
        boundary = openmc.model.RectangularParallelepiped(
            -10., 10., -10., 10., 0., 50., boundary_type="vacuum")
        # A nonzero density loads real energy bounds. The chance of scattering
        # across this box is negligible at 1e-30 atoms/barn-cm.
        material = openmc.Material()
        material.add_nuclide("H1", 1.)
        material.set_density("atom/b-cm", 1e-30)
        materials = openmc.Materials([material])
        materials.cross_sections = str(self.xs)
        cell = openmc.Cell(region=-boundary, fill=material)
        model = openmc.Model(geometry=openmc.Geometry([cell]), materials=materials)
        point = (2.5, -3.5, 12.5)
        energies = [100., 1e4, 1e5, 1e6, 10e6, 14.1e6]
        sources = []
        for index, energy in enumerate(energies):
            direction = [0., 0., 0.]
            direction[index // 2] = -1. if index % 2 == 0 else 1.
            sources.append(openmc.IndependentSource(
                space=openmc.stats.Point(point),
                angle=openmc.stats.Monodirectional(direction),
                energy=openmc.stats.Discrete([energy], [1.]), strength=(index + 1) / 21.))
        model.settings.source = sources
        model.settings.run_mode = "fixed source"
        model.settings.particles = 2000
        model.settings.batches = 10
        model.settings.output = {"summary": False, "tallies": False}
        reference = openmc.Tally(name="current")
        reference.filters = [openmc.SurfaceFilter(
            [getattr(boundary, face.replace("_", "")) for face in FACES]),
            openmc.ParticleFilter(["neutron"])]
        reference.scores = ["current"]
        if energy_bounds is None:
            exits = build_exit_tallies(boundary, 4., 60)
        else:
            with patch("python.neutron_exit.EXIT_ENERGY_MIN_EV", energy_bounds[0]), \
                 patch("python.neutron_exit.EXIT_ENERGY_MAX_EV", energy_bounds[1]):
                exits = build_exit_tallies(boundary, 4., 60)
        model.tallies = openmc.Tallies([reference, *exits])
        statepoint_path = model.run(cwd=directory, threads=1, output=False)
        return statepoint_path, point, energies

    def test_faces_pixels_spectra_normalization_and_empty_pixels(self):
        with tempfile.TemporaryDirectory(prefix="pile-exit-test-") as directory:
            path, point, energies = self.run_beams(directory)
            output = Path(directory) / "neutron_exit"
            with openmc.StatePoint(path, autolink=False) as statepoint:
                export_exit_maps(statepoint, output, source_rate=1e6)
            total = 0.
            for index, face in enumerate(FACES):
                with np.load(output / f"{face}_spectra.npz", allow_pickle=False) as data:
                    mean = data["current_mean"]
                    spectrum = data["spectrum_mean"]
                    self.assertEqual(spectrum.shape[:2], mean.shape)
                    np.testing.assert_allclose(spectrum.sum(axis=-1), mean, atol=1e-12)
                    axes = [i for i in range(3) if i != index // 2]
                    u = np.searchsorted(data["u_edges_cm"], point[axes[0]], side="right") - 1
                    v = np.searchsorted(data["v_edges_cm"], point[axes[1]], side="right") - 1
                    self.assertEqual(np.count_nonzero(mean), 1)
                    expected = (index + 1) / 21.
                    error = data["current_std_dev"][v, u]
                    self.assertLess(abs(mean[v, u] - expected), 6 * error + 1e-3)
                    self.assertLessEqual(data["modal_energy_lower_eV"][v, u], energies[index])
                    self.assertGreater(data["modal_energy_upper_eV"][v, u], energies[index])
                    self.assertTrue(data["mode_reliable"][v, u])
                    self.assertEqual(np.count_nonzero(np.isfinite(data["modal_energy_eV"])), 1)
                    np.testing.assert_allclose(data["current_density"] * data["pixel_area_cm2"],
                                               mean * 1e6, atol=1e-10)
                    np.testing.assert_allclose(data["spectrum_std_dev"].sum(axis=-1),
                                               data["current_std_dev"], atol=1e-12)
                    total += float(mean.sum())
            self.assertAlmostEqual(total, 1., places=12)
            manifest = json.loads((output / "metadata.json").read_text())
            self.assertEqual(manifest["source_rate_neutrons_per_s"], 1e6)
            for filename in ("exit_current.png", "modal_energy.png"):
                self.assertGreater((output / filename).stat().st_size, 1000)

    def test_out_of_range_energies_remain_in_spectra(self):
        with tempfile.TemporaryDirectory(prefix="pile-exit-guards-") as directory:
            path, _, _ = self.run_beams(directory, energy_bounds=(1e3, 5e6))
            output = Path(directory) / "neutron_exit"
            with openmc.StatePoint(path, autolink=False) as statepoint:
                export_exit_maps(statepoint, output)
            total = 0.
            for face, guard in (("x_min", 0), ("z_min", -1), ("z_max", -1)):
                with np.load(output / f"{face}_spectra.npz", allow_pickle=False) as data:
                    self.assertGreater(data["spectrum_mean"][..., guard].sum(), 0.)
                    self.assertTrue(np.isnan(data["modal_energy_eV"]).all())
            for face in FACES:
                with np.load(output / f"{face}_spectra.npz", allow_pickle=False) as data:
                    total += float(data["spectrum_mean"].sum())
            self.assertAlmostEqual(total, 1., places=12)


if __name__ == "__main__":
    unittest.main()
