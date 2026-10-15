import numpy as np
import openmc

from python.parameters import *
from python.utilities  import *


class Material:

    def build_materials(self):       

        if self.breeder == 'flibe':

            self.m_breeder = openmc.Material(name='2(LiF)-BeF2', temperature=self.temp_k)
            self.m_breeder.set_density('g/cm3', density_flibe(self.temp_k))
            self.m_breeder.add_elements_from_formula('F4Li2Be', 'ao', 
                                              enrichment_target='Li6', 
                                              enrichment_type='ao',  
                                              enrichment=self.li_enrich)

        elif self.breeder == 'teflon':

            self.m_breeder = openmc.Material(name='Teflon (PTFE)', temperature=self.temp_k)
            self.m_breeder.set_density('g/cm3', density_teflon(self.temp_k))
            self.m_breeder.add_elements_from_formula('C2F4', 'ao')

        elif self.breeder == 'grph':

            self.m_breeder = openmc.Material(name='Graphite', temperature=self.temp_k)
            self.m_breeder.set_density('g/cm3', DENSITY['grph'])
            self.m_breeder.add_elements_from_formula('C', 'ao')

        elif self.breeder == 'cllif':

            self.m_breeder = openmc.Material(name='LiF-ClF2', temperature=self.temp_k)
            self.m_breeder.set_density('g/cm3', DENSITY['cllif'])
            self.m_breeder.add_elements_from_formula('F3LiCl', 'ao', 
                                              enrichment_target='Li6', 
                                              enrichment_type='ao',  
                                              enrichment=self.li_enrich)


        # TRISO particle

        self.m_buff = openmc.Material(name='Buffer')
        self.m_buff.set_density('g/cm3', 1.0)
        self.m_buff.add_element('C', 1.0)
        self.m_buff.add_s_alpha_beta('c_Graphite')

        self.m_PyC1 = openmc.Material(name='PyC1')
        self.m_PyC1.set_density('g/cm3', 1.90)
        self.m_PyC1.add_element('C', 1.0)
        self.m_PyC1.add_s_alpha_beta('c_Graphite')

        self.m_PyC2 = openmc.Material(name='PyC2')
        self.m_PyC2.set_density('g/cm3', 1.87)
        self.m_PyC2.add_element('C', 1.0)
        self.m_PyC2.add_s_alpha_beta('c_Graphite')

        self.m_SiC = openmc.Material(name='SiC')
        self.m_SiC.set_density('g/cm3', 3.20)
        self.m_SiC.add_element('C', 0.5)
        self.m_SiC.add_element('Si', 0.5)

        self.m_grph = openmc.Material()
        self.m_grph.set_density('g/cm3', 1.1995)
        self.m_grph.add_element('C', 1.0)
        self.m_grph.add_s_alpha_beta('c_Graphite')


        # Breeder and fertile material mixed in the blanket breeding regions 
        if self.fertile_isotope == 'U238':

            self.m_xf4 = openmc.Material(name='fertile', temperature=self.temp_k)
            self.m_xf4.add_elements_from_formula('UF4','ao') # 'ao' here refers to 1:4 atomic ratio of U:F in UF4
            self.m_xf4.set_density('g/cm3', DENSITY['UF4']) 

            self.m_xo2 = openmc.Material(name='fertile', temperature=self.temp_k)
            self.m_xo2.add_elements_from_formula('UO2','ao') # 'ao' here refers to 1:4 atomic ratio of U:O in UO2
            self.m_xo2.set_density('g/cm3', DENSITY['UO2']) 

        elif self.fertile_isotope == 'Th232':

            self.m_xf4 = openmc.Material(name='fertile', temperature=self.temp_k)
            self.m_xf4.add_elements_from_formula('ThF4','ao') # 'ao' here refers to 1:4 atomic ratio of Th:F in ThF4
            self.m_xf4.set_density('g/cm3', DENSITY['ThF4']) 

            self.m_xo2 = openmc.Material(name='fertile', temperature=self.temp_k)
            self.m_xo2.add_elements_from_formula('ThO2','ao') # 'ao' here refers to 1:4 atomic ratio of Th:O in ThO2
            self.m_xo2.set_density('g/cm3', DENSITY['ThO2']) 

        else:
            raise ValueError(f"Invalid fertile isotope: {self.fertile_isotope}. Must be 'U238' or 'Th232'.")

        # vf_xf4, vf_br = calc_xf4_vol_fracs(self.fertile_kgm3, self.fertile_isotope)
        # self.breeder_xf4 = openmc.Material.mix_materials([self.xf4, self.breeder], [vf_xf4, vf_br], 'vo') # fractions in 'mix_materials' MUST add up to 1
        # self.blanket.temperature = self.temp_k
        # self.blanket.name = (f"{self.fertile_kgm3:07.2f} kg/m3 | {(vf_xf4*100):.4f} vol% of nominal breeder")

        # Add materials 
        self.materials = openmc.Materials([self.m_breeder, self.m_buff, self.m_PyC1, self.m_PyC2, self.m_SiC, self.m_grph, self.m_xo2]) # self.air, , self.breeder_xf4
