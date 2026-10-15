import openmc

class Settings:
    
    def build_settings(self):

        if self.pile_code == 'a01':
       
            source = openmc.IndependentSource(particle="neutron",
                                                 space=openmc.stats.CartesianIndependent(
                                                     x=openmc.stats.Uniform(-10.0, 10.0),
                                                     y=openmc.stats.Uniform(-10.0, 10.0),
                                                     z=openmc.stats.Discrete([1e-6], [1.0]),),
                                                 angle=openmc.stats.Monodirectional((0.0, 0.0, 1.0)),
                                                energy=openmc.stats.Discrete([14.1e6], [1.0]),)

        # Create settings and assign source
        self.settings = openmc.Settings()
        self.settings.source = source

        # Run type 
        self.settings.run_mode  = 'fixed source'
        self.settings.particles = self.n_particles
        self.settings.batches   = self.n_cycles

        # Dump data to statepoint intermittently to allow restarts later 
        save_sp_every = 5
        self.settings.statepoint = {'batches': sorted(
            set(range(save_sp_every, self.n_cycles + 1, save_sp_every)) | {self.n_cycles})}
        self.settings.output     = {'summary': False, 'tallies': False}
