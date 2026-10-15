import openmc
import numpy as np

from python.neutron_exit import build_exit_tallies

class Tally:
    
    def build_tallies(self):

        self.tallies = openmc.Tallies()

        # Filters
        cell_filter = openmc.CellFilter(self.cells)
        energy_filter = openmc.EnergyFilter(np.geomspace(1e-3, 14.2e6, 1000))
        surface_filter = openmc.SurfaceFilter(list(self.cells[0].region.get_surfaces().values()))

        # Boundary current
        neutron_filter = openmc.ParticleFilter(["neutron"])
        t_current = self.make_tally("current", ["current"], filters=[surface_filter, neutron_filter])
        t_current_energy = self.make_tally("current spectrum", ["current"], filters=[surface_filter, neutron_filter, energy_filter],)
        t_current.estimator = "analog"
        t_current_energy.estimator = "analog"

        # Flux
        t_flux = self.make_tally("flux", ["flux"], filters=[cell_filter])
        t_flux_energy = self.make_tally("flux spectrum", ["flux"], filters=[cell_filter, energy_filter],)

        nuclides = [n for n in ("U238", "U235", "Th232") if n in self.m_xo2.get_nuclides() ]
        scores = [ "(n,gamma)", "fission", "nu-fission", "scatter", "elastic", "(n,2n)", "(n,3n)",] # "(n,na)", "(n,np)",]
        t_fertile        = self.make_tally("Fertile rxn rates", scores, nuclides=nuclides)
        t_fertile_energy = self.make_tally( "Fertile rxn rates spectrum", scores, filters=[cell_filter, energy_filter], nuclides=nuclides, )

        # Register every tally
        self.tallies.extend([t_current, t_current_energy, t_flux, t_flux_energy, t_fertile, t_fertile_energy])
        self.tallies.extend(build_exit_tallies(
            self.pile_boundary, self.exit_pixel_cm, self.exit_energy_bins))


    @staticmethod
    def make_tally(name, scores, filters:list=None, nuclides:list=None):
        tally = openmc.Tally(name=name)
        tally.scores = scores
        if filters is not None:
            tally.filters = filters
        if nuclides is not None:
            tally.nuclides = nuclides
        return tally
