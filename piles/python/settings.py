import openmc

class Settings:
    
    def settings(self):

        if self.pile_code == 'A01':
       
            source = openmc.IndependentSource()
            source.space = openmc.stats.CylindricalIndependent(
                                        r=openmc.stats.Discrete([self.R0], [1.0]),  # r = R0 major radius [cm]
                                        phi=openmc.stats.Uniform(0.0, 2*np.pi)   ,  # phi = 0 to 2pi
                                        z=openmc.stats.Discrete([0.0], [1.0])   ) # z   = 0
                                                            
            source.particle = 'neutron'
            source.energy   = openmc.stats.Discrete([14.0e6], [1.0])  # 14 MeV
            source.angle    = openmc.stats.Isotropic()

        # Create settings and assign source
        self.settings = openmc.Settings()
        self.settings.source = source

        # Run type 
        self.settings.run_mode  = 'fixed source'
        self.settings.particles = self.n_particles
        self.settings.batches   = self.n_cycles

        # Dump data to statepoint intermittently to allow restarts later 
        save_sp_every = 5
        self.settings.statepoint = {'batches': range(save_sp_every, self.n_cycles + 1, save_sp_every)} 
        self.settings.output     = {'summary': False, 'tallies': False}