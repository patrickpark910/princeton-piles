import openmc
import os
import numpy as np
import pandas as pd
from PIL import Image
from abc import ABC, abstractmethod

# Import helper functions
from python.parameters import *
from python.utilities  import *
from python.geometries import Geometry
from python.tallies    import Tally
from python.settings   import Settings


class Pile(Geometry, Tally, Settings, ABC):

    def __init__(self, pile_code, breeder, temp_k=TEMP_K,
                 fertile_kgm3=0.0, fertile_isotope='U238', li_enrich=None, 
                 run_type='tallies', n_particles=N_PARTICLES, n_cycles=N_CYCLES, 
                 print_xml=True, run_openmc=True, run_debug=True):

        self.fertile_kgm3 = fertile_kgm3
        self.fertile_isotope = fertile_isotope
        self.fissile_isotope = FISSILE.get(fertile_isotope, None)
        self.li_enrich = li_enrich
        self.temp_k  = temp_k
        self.breeder = breeder
        self.breeder_density = None 

        self.pile_code   = pile_code
        self.run_type    = run_type
        self.print_xml   = print_xml
        self.run_openmc  = run_openmc
        self.run_debug   = run_debug
        self.n_particles = n_particles
        self.n_cycles    = n_cycles

        s = f"{self.n_particles:.0e}x{self.n_cycles}".replace("+0", "").replace("+", "")
        self.name = f"{pile_code}_{run_type}_{breeder}_{temp_k}K_Li{li_enrich:04.1f}_{fertile_isotope}_{fertile_kgm3:07.2f}kgm3_{s}"    
        self.path = None # f"./OpenMC/{self.name}"

        start_msg = f"\n======== {self.blanket_name} blanket - {self.fertile_isotope} {self.fertile_kgm3:07.2f} kg/m3 - {self.breeder_enrich:4.1f}%-enriched Li - {self.temp_k} K ========"
        print(f"{C.CYAN}{start_msg}{C.END}")


    @abstractmethod
    def materials(self):
        pass






    def compile(self):
        
        openmc.reset_auto_ids()
        self.materials()
        self.geometry()
        self.settings()
        self.tallies()

        self.materials.cross_sections = set_xs_path()
        self.model = openmc.model.Model(self.geometry, self.materials, self.settings, self.tallies)

        if self.print_xml:
            self.model.export_to_model_xml(self.path)
            print(f"{C.YELLOW}Comment.{C.END} OpenMC model XML files printed to: {self.path}")
        else:
            print(f"{C.RED}Warning.{C.END} OpenMC model XML files NOT printed (print_xml=False) to: {self.path}")


    def openmc(self):

        if self.run_openmc:
            if has_statepoint(self.path,cycle=self.n_cycles):
                print(f"{C.YELLOW}Warning.{C.END} File {self.path}/statepoint.{str(self.n_cycles).zfill(2)}.h5 already exists, so this OpenMC run will be skipped...")
            else:
                self.model.run(cwd=self.path)

                # Clean up intermediate statepoint files -- KEEP UNDER THIS LAST 'ELSE'!! --ppark 2026-05-01
                for filename in os.listdir(self.path):
                    if filename.startswith('statepoint.') and filename.endswith('.h5'):
                        try:
                            # Extract the integer batch number from the filename
                            batch_num = int(filename.split('.')[1])  
                            # Delete if it's less than the final programmed cycle
                            if batch_num < self.n_cycles:
                                print(f"{C.YELLOW}Comment.{C.END} Removing intermediate statepoint.{batch_num}.h5 (< {self.n_cycles} cycles)...")
                                file_path = os.path.join(self.path, filename)
                                os.remove(file_path)
                        except ValueError:
                            # Safely ignore any improperly formatted statepoint files
                            pass
        else:    
            print(f"{C.RED}Warning.{C.END} OpenMC calculation skipped (run_openmc=False) for: {self.path}")


    def plot(self):

        self.materials()
        self.geometry()

        Image.MAX_IMAGE_PIXELS = None # suppreses DecompressionBombWarning lmao

        settings_plot = openmc.Settings()
        settings_plot.run_mode = 'plot'

        if self.blanket_name in ['ARC', 'ARCBall', 'FLiBe']:
            C = {self.firstwall: (30, 27, 41),    # very dark gray
                      self.structure: (109, 110, 113), # gray
                      self.blanket: (129, 204, 185),   # teal
                      # Void regions will be white by default
                      }

        elif self.blanket_name in ['DCLL']:
            C = {self.firstwall: (30, 27, 41),           # very dark gray
                      self.structure: (109, 110, 113),        # gray
                      self.coolant: (37, 150, 190), # cobalt blue 
                      self.blanket: (129, 204, 185),          # teal
                      self.divider: (109, 110, 113),          # gray
                      self.inner_manifold:(176, 123, 76),     # wood-color
                      # Void regions will be white by default
                      }

        elif self.blanket_name in ['HCPB']:
            C = {self.firstwall: (30, 27, 41),           # very dark gray
                      self.structure: (109, 110, 113),        # gray
                      self.blanket: (129, 204, 185),          # teal
                      # Void regions will be white by default
                      }

        x_width = round(2*self.extent_r)
        z_width = round(2*self.extent_z)
        
        # XY toroidal slice
        xy = openmc.Plot()
        xy.filename = f"{self.blanket_name}_xy" # {self.path}/
        xy.basis = "xy"
        xy.width  = (x_width, x_width)
        xy.pixels = (10*x_width, 10*x_width)
        xy.color_by = "material"
        xy.C = C

        # XZ poloidal slice
        xz = openmc.Plot()
        xz.filename = f"{self.blanket_name}_xz" # {self.path}/
        xz.basis = "xz"
        xz.width  = (x_width, z_width)
        xz.pixels = (10*x_width, 10*z_width)
        xz.color_by = "material"
        xz.C = C

        plots = openmc.Plots([xy, xz])
        model_plot = openmc.model.Model(self.geometry, self.materials, settings_plot)
        model_plot.plots = plots 
        model_plot.plot_geometry(cwd=f"{self.path}")  # writes tokamak_rz.ppm and tokamak_xz.ppm

        for basename in [f"{self.blanket_name}_xy", f"{self.blanket_name}_xz"]:
            ppm_file = os.path.join(self.path, f"{basename}.ppm")
            png_file = os.path.join(self.path, f"{basename}.png")
            if os.path.exists(ppm_file):
                with Image.open(ppm_file) as im:
                    im.save(png_file)
                print(f"{C.YELLOW}Comment.{C.END} Plots '{self.blanket_name}_xy.png', '{self.blanket_name}_xz.png' printed to {self.path}")
            else:
                print(f"{C.YELLOW}Error.{C.END} OpenMC did not print '{self.blanket_name}_xy.ppm', '{self.blanket_name}_xz.ppm' to {self.path}!")