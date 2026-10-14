import numpy as np
import openmc

from python.reactor    import *
from python.parameters import *
from python.utilities  import *


class Materials:

    def materials(self):       

        if self.breeder == 'flibe':

            breeder = openmc.Material(name='2(LiF)-BeF2', temperature=self.temp_k)
            breeder.set_density('g/cm3', density_flibe(self.temp_k))
            breeder.add_elements_from_formula('F4Li2Be', 'ao', 
                                              enrichment_target='Li6', 
                                              enrichment_type='ao',  
                                              enrichment=self.li_enrich)

        elif self.breeder == 'teflon':

            breeder = openmc.Material(name='Teflon (PTFE)', temperature=self.temp_k)
            breeder.set_density('g/cm3', density_teflon(self.temp_k))
            breeder.add_elements_from_formula('C2F4', 'ao')

        elif self.breeder == 'grph':

            breeder = openmc.Material(name='Graphite', temperature=self.temp_k)
            breeder.set_density('g/cm3', DENSITY_GRPH)
            breeder.add_elements_from_formula('C', 'ao')

        elif self.breeder == 'cllif':

            breeder = openmc.Material(name='LiF-ClF2', temperature=self.temp_k)
            breeder.set_density('g/cm3', DENSITY_CLLIF)
            breeder.add_elements_from_formula('F3LiCl', 'ao', 
                                              enrichment_target='Li6', 
                                              enrichment_type='ao',  
                                              enrichment=self.li_enrich)


        # TRISO particle

        buff = openmc.Material(name='Buffer')
        buff.set_density('g/cm3', 1.0)
        buff.add_element('C', 1.0)
        buff.add_s_alpha_beta('c_Graphite')

        PyC1 = openmc.Material(name='PyC1')
        PyC1.set_density('g/cm3', 1.9)
        PyC1.add_element('C', 1.0)
        PyC1.add_s_alpha_beta('c_Graphite')

        PyC2 = openmc.Material(name='PyC2')
        PyC2.set_density('g/cm3', 1.87)
        PyC2.add_element('C', 1.0)
        PyC2.add_s_alpha_beta('c_Graphite')

        SiC = openmc.Material(name='SiC')
        SiC.set_density('g/cm3', 3.2)
        SiC.add_element('C', 0.5)
        SiC.add_element('Si', 0.5)

        graphite = openmc.Material()
        graphite.set_density('g/cm3', 1.1995)
        graphite.add_element('C', 1.0)
        graphite.add_s_alpha_beta('c_Graphite')


        # Breeder and fertile material mixed in the blanket breeding regions 
        if self.fertile_isotope == 'U238':

            self.xf4 = openmc.Material(name='fertile', temperature=self.temp_k)
            self.xf4.add_elements_from_formula('UF4','ao') # 'ao' here refers to 1:4 atomic ratio of U:F in UF4
            self.xf4.set_density('g/cm3', DENSITY_UF4) 

            self.xo2 = openmc.Material(name='fertile', temperature=self.temp_k)
            self.xo2.add_elements_from_formula('UO2','ao') # 'ao
            
            
        elif self.fertile_isotope == 'Th232':

            self.xf4 = openmc.Material(name='fertile', temperature=self.temp_k)
            self.xf4.add_elements_from_formula('ThF4','ao') # 'ao' here refers to 1:4 atomic ratio of Th:F in ThF4
            self.xf4.set_density('g/cm3', DENSITY_ThF4) 

            self.xo2 = openmc.Material(name='fertile', temperature=self.temp_k)
            self.xo2.add_elements_from_formula('UO2','ao') 

        vf_xf4, vf_br = calc_xf4_vol_fracs(self.fertile_kgm3, self.fertile_isotope)
        self.breeder_xf4 = openmc.Material.mix_materials([self.xf4, self.breeder], [vf_xf4, vf_br], 'vo') # fractions in 'mix_materials' MUST add up to 1
        self.blanket.temperature = self.temp_k
        
        self.blanket.name = (f"{self.fertile_kgm3:07.2f} kg/m3 | {(vf_xf4*100):.4f} vol% of nominal breeder")

        # Add materials 
        self.materials = openmc.Materials([self.breeder, buff, PyC1, PyC2, SiC, graphite, self.xo2, self.breeder_xf4]) # self.air, 
