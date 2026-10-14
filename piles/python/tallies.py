import openmc

class Tally:
    
    def tallies(self):
        """ 
        TALLIES 
        """
        self.tallies = openmc.Tallies() # initialize

        # Filters
        cell_filter         = openmc.CellFilter([cell for cell in self.cells if cell.fill is self.blanket])  # not None])  # dropping tallies on non-blanket cells for expedience --ppark 2026-07-07
        energy_filter       = openmc.EnergyFilter(logspace_per_decade(1e-3, 20e6, 100)) # './helpers/utilities.py'
                
        # Workaround: OpenMC has a bug tallying MT=4 (total inelastic) directly.
        # Instead, tally the 41 discrete-level + continuum channels (MTs 51-91)
        # individually, then collapse them in post-processing.
        # MT_INELASTIC_SCORES = [str(mt) for mt in range(51, 92)]
        MT_SCATTERING_SCORES = ['scatter', 'elastic', '(n,2n)', '(n,3n)', '(n,na)', '(n,np)']
        # instead of this let's try doing scattering - elastic - n,Xn - n,nX = inelastic

        # Current tally
        current_tally        = self.make_tally('current', ['current'], filters=[cellwithvoid_filter, cellfrom_filter])

        # Leakage current spectrum - scope CellFilter to void cells only to keep output manageable
        current_energy_tally = self.make_tally('current spectrum', ['current'], filters=[void_cell_filter, cellfrom_filter, energy_filter])

        # Flux tally 
        flux_tally        = self.make_tally('flux', ['flux'], filters=[cell_filter])
        flux_energy_tally = self.make_tally('flux spectrum', ['flux'], filters=[cell_filter, energy_filter])

        heating_tally        = self.make_tally('heating', ['heating-local'], filters=[cell_filter])
        heating_energy_tally = self.make_tally('heating spectrum', ['heating-local'], filters=[cell_filter, energy_filter])  # not having cell_filter on purpose

        fisq_tally           = self.make_tally('fission-q', ['fission-q-recoverable'], filters=[cell_filter])
        fisq_energy_tally    = self.make_tally('fission-q spectrum', ['fission-q-recoverable'], filters=[cell_filter, energy_filter])  # not having cell_filter on purpose


        # Fertile element reaction rates // MT 4 = inelastic scattering (tallied as MTs 51-91, summed in post-processing)
        # fertile_tally_tot    = self.make_tally('Total fertile rxn rate',     ['(n,gamma)', 'fission', 'nu-fission', *MT_SCATTERING_SCORES], nuclides=['U238', 'U235', 'Th232'])
        fertile_tally        = self.make_tally('Fertile rxn rates',          ['(n,gamma)', 'fission', 'nu-fission', *MT_SCATTERING_SCORES], nuclides=['U238', 'U235', 'Th232'], filters=[cell_filter])
        fertile_energy_tally = self.make_tally('Fertile rxn rates spectrum', ['(n,gamma)', 'fission', 'nu-fission', *MT_SCATTERING_SCORES], nuclides=['U238', 'U235', 'Th232'], filters=[cell_filter, energy_filter])


    @staticmethod
    def make_tally(name, scores, filters:list=None, nuclides:list=None):
        tally = openmc.Tally(name=name)
        tally.scores = scores
        if filters is not None:
            tally.filters = filters
        if nuclides is not None:
            tally.nuclides = nuclides
        return tally