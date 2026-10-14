import numpy as np
import openmc

from Python.reactor    import *
from Python.parameters import *
from Python.utilities  import *


class FLiBe(Reactor):

    def initialize(self):

        self.temp_k          = TEMP_K
        self.blanket_name    = 'FLiBe'
        self.blanket_volume  = FLIBE_BL_VOL         # [m³] total breeding region (FLiBe + XF4)
        self.breeder_volume  = FLIBE_BL_VOL         # [m³] FLiBe volume, deducted by adding XF4 in materials()
        self.breeder_density = DENSITY_FLIBE        # [g/cm³]
        if self.breeder_enrich is None:
            self.breeder_enrich  = ENRICH_FLIBE         # [at%]

        print(self.breeder_enrich)
        # Name file based on reactor config - should come out to smth like: tallies_FLiBe_U010kgm3_Li7.5_900K
        s = f"{self.n_particles:.0e}x{self.n_cycles}".replace("+0", "").replace("+", "")
        self.name = f"{self.run_type}_{self.blanket_name}_{self.temp_k}K_Li{self.breeder_enrich:04.1f}_{self.fertile_isotope}_{self.fertile_kgm3:07.2f}kgm3_{s}"           
        self.path = f"./OpenMC/{self.name}"
        
        os.makedirs(self.path, exist_ok=True)

        start_msg = f"\n======== {self.blanket_name} blanket - {self.fertile_isotope} {self.fertile_kgm3:07.2f} kg/m3 - {self.breeder_enrich:4.1f}%-enriched Li - {self.temp_k} K ========"
        print(f"{C.CYAN}{start_msg}{C.END}")


    def materials(self):       

        # ------------------------------------------------------------------
        # Breeder material
        # ------------------------------------------------------------------

        flibe = openmc.Material(name='2(LiF)-BeF2', temperature=self.temp_k)
        flibe.set_density('g/cm3', self.breeder_density)
        flibe.add_elements_from_formula('F4Li2Be', 'ao', 
                                        enrichment_target='Li6', 
                                        enrichment_type='ao',  # I don't know why Alex says this should be wt%. I'm deferring to the MIT papers that say 7.5 at% here. --ppark
                                        enrichment=self.breeder_enrich)

        
        # ------------------------------------------------------------------
        # Breeder and fertile material mixed in the blanket breeding regions 
        # ------------------------------------------------------------------        

        if self.fertile_isotope == 'U238':

            self.fertile = openmc.Material(name='fertile', temperature=self.temp_k)
            self.fertile.add_elements_from_formula('UF4','ao') # 'ao' here refers to 1:4 atomic ratio of U:F in UF4
            self.fertile.set_density('g/cm3', DENSITY_UF4) 
            
        elif self.fertile_isotope == 'Th232':

            self.fertile = openmc.Material(name='fertile', temperature=self.temp_k)
            self.fertile.add_elements_from_formula('ThF4','ao') # 'ao' here refers to 1:4 atomic ratio of Th:F in ThF4
            self.fertile.set_density('g/cm3', DENSITY_ThF4) 

        vf_xf4, vf_flibe = calc_xf4_vol_fracs(self.fertile_kgm3, self.fertile_isotope)

        self.blanket = openmc.Material.mix_materials([self.fertile, flibe], [vf_xf4, vf_flibe], 'vo') # fractions in 'mix_materials' MUST add up to 1
        self.blanket.temperature = self.temp_k
        self.blanket.name = (f"{self.fertile_kgm3:07.2f} kg/m3 | {(vf_xf4*100):.4f} vol% of nominal breeder")


        # ------------------------------------------------------------------
        # Add materials 
        # ------------------------------------------------------------------

        self.materials = openmc.Materials([self.firstwall, self.beryllium, self.structure, self.blanket]) # self.air, 
        # self.materials.export_to_xml(self.path)


        # ------------------------------------------------------------------
        # Debugging printouts
        # ------------------------------------------------------------------

        uh_oh_did_i_make_a_fucky_wucky = True

        if self.run_debug and uh_oh_did_i_make_a_fucky_wucky:
            print(f"")
            print(f"| DEBUG PRINTOUT - MATERIALS")
            print(f"| ")
            print(f"| Breeder = 2(LiF)-BeF2")
            print(f"| Breeder vol: {self.breeder_volume:.6f} [m³] (nominal breeder volume in blanket)")
            print(f"| Blanket vol: {self.blanket_volume:.6f} [m³] (total blanket volume)")
            print(f"| ")
            print(f"| With respect to the NOMINAL BREEDER VOLUME, we have these new volume fractions:")
            print(f"|   vf_xf4   =  {(vf_xf4*100):.6f} vol%")
            print(f"|   vf_flibe =  {(vf_flibe*100):.6f} vol%")
            # print(f"|   check they add up = {(vf_xf4+vf_flibe)*100:.6f} vol%")
            print(f"")


    def geometry(self):

        # ------------------------------------------------------------------
        # Tokamak geometry parameters 
        # ------------------------------------------------------------------

        self.R0, self.a, self.kappa, self.delta = FLIBE_R0, FLIBE_A, FLIBE_KAPPA, FLIBE_DELTA 

        d_fw  = FLIBE_FW_CM 
        d_be  = d_fw  + FLIBE_BE_CM
        d_st0 = d_be  + FLIBE_ST1_CM
        d_br0 = d_st0 + FLIBE_BR1_CM
        d_st1 = d_br0 + FLIBE_ST2_CM
        d_br1 = d_st1 + FLIBE_BR2_CM
        d_st2 = d_br1 + FLIBE_ST3_CM

        self.extent_r = (self.R0 + self.a + d_st2)*1.05 # 105%
        self.extent_z = (self.kappa*self.a + d_st2)*1.05



        # ------------------------------------------------------------------
        # Surfaces 
        # ------------------------------------------------------------------

        # Arrays of a (minor r) and z points
        points_vc  = miller_model(self.R0, self.a, self.kappa, self.delta)                 # coords around vacuum chamber
        points_fw  = miller_model(self.R0, self.a, self.kappa, self.delta, extrude=d_fw)   # outer coords around first wall
        points_be  = miller_model(self.R0, self.a, self.kappa, self.delta, extrude=d_be)   # outer coords around beryllium layer
        points_st0 = miller_model(self.R0, self.a, self.kappa, self.delta, extrude=d_st0)  # outer coords around structural region 0
        points_br0 = miller_model(self.R0, self.a, self.kappa, self.delta, extrude=d_br0)  # outer coords around breeding channel
        points_st1 = miller_model(self.R0, self.a, self.kappa, self.delta, extrude=d_st1)  # outer coords around structural region 1
        points_br1 = miller_model(self.R0, self.a, self.kappa, self.delta, extrude=d_br1)  # outer coords around breeding region 1
        points_st2 = miller_model(self.R0, self.a, self.kappa, self.delta, extrude=d_st2)  # outer coords around structural region 2

        # Create OpenMC surfaces and STORE THEM AS CLASS ATTRIBUTES
        surface_vc  = openmc.model.Polygon(points_vc , basis='rz')  # Plasma-facing surface (inner FW)
        surface_fw  = openmc.model.Polygon(points_fw , basis='rz')  # Outer surface of first wall
        surface_be  = openmc.model.Polygon(points_be , basis='rz')  # Outer surface of beryllium layer
        surface_st0 = openmc.model.Polygon(points_st0, basis='rz')
        surface_br0 = openmc.model.Polygon(points_br0, basis='rz')
        surface_st1 = openmc.model.Polygon(points_st1, basis='rz')
        surface_br1 = openmc.model.Polygon(points_br1, basis='rz')
        surface_st2 = openmc.model.Polygon(points_st2, basis='rz')

        # Toroidal sector with reflective boundaries
        self.sector_angle = np.radians(SECTOR_DEG)
        wedge_plane_a  = openmc.YPlane(y0=0, boundary_type='reflective')
        wedge_plane_b = openmc.Plane(a=-np.sin(self.sector_angle), b=np.cos(self.sector_angle), c=0, d=0, boundary_type='reflective')
        wedge = +wedge_plane_a & -wedge_plane_b

        # Add boundary surfaces
        outer_cylinder = openmc.ZCylinder(r=self.extent_r, boundary_type='vacuum')
        top_plane      = openmc.ZPlane(z0=self.extent_z, boundary_type='vacuum')
        bottom_plane   = openmc.ZPlane(z0=-self.extent_z, boundary_type='vacuum')
        

        # ------------------------------------------------------------------
        # Cells | 10: vc | 11: fw | 41: be | 2X: structure | 3X: breeding
        # ------------------------------------------------------------------
        cell_vc   = openmc.Cell(cell_id=10, region= -surface_vc & wedge)
        cell_vc.importance = {'neutron':1}
        cell_fw   = openmc.Cell(cell_id=11, region= +surface_vc  & -surface_fw  & wedge, fill=self.firstwall) 
        cell_be   = openmc.Cell(cell_id=41, region= +surface_fw  & -surface_be  & wedge, fill=self.beryllium)
        cell_st0  = openmc.Cell(cell_id=21, region= +surface_be  & -surface_st0 & wedge, fill=self.structure)
        cell_br0  = openmc.Cell(cell_id=31, region= +surface_st0 & -surface_br0 & wedge, fill=self.blanket)
        cell_st1  = openmc.Cell(cell_id=22, region= +surface_br0 & -surface_st1 & wedge, fill=self.structure)
        cell_br1  = openmc.Cell(cell_id=32, region= +surface_st1 & -surface_br1 & wedge, fill=self.blanket)
        cell_st2  = openmc.Cell(cell_id=23, region= +surface_br1 & -surface_st2 & wedge, fill=self.structure) 

        # Surrounding air cell with proper boundaries (otherwise causes error with just Polygons)
        cell_void = openmc.Cell(cell_id=99, region= +surface_st2 & -outer_cylinder & +bottom_plane & -top_plane & wedge) # fill=self.air
        cell_void.importance = {'neutron':0}

        self.cells = [cell_vc, cell_fw, cell_be, cell_st0, cell_br0, cell_st1, cell_br1, cell_st2, cell_void]
        self.geometry = openmc.Geometry(openmc.Universe(cells=self.cells))
        # self.geometry.export_to_xml(self.path)

    def settings(self):
        super().settings()  # calls the settings() method from the parent class Reactor
        source = openmc.IndependentSource()
        source.space = openmc.stats.CylindricalIndependent(
                                    r=openmc.stats.Discrete([self.R0], [1.0]),
                                    phi=openmc.stats.Uniform(0.0, self.sector_angle),
                                    z=openmc.stats.Discrete([0.0], [1.0]))
        source.particle = 'neutron'
        source.energy   = openmc.stats.Discrete([14.0e6], [1.0])
        source.angle    = openmc.stats.Isotropic()
        self.settings.source = [source]
