import openmc
import os
import numpy as np
import pandas as pd
from PIL import Image
from abc import ABC, abstractmethod

# Import helper functions
from python.parameters import *
from python.utilities  import *
from python.materials  import Material
from python.geometries import Geometry
from python.tallies    import Tally
from python.settings   import Settings
from python.neutron_exit import exit_grid_id, export_exit_maps


class Pile(Material, Geometry, Tally, Settings, ABC):

    def __init__(self, pile_code, breeder, 
                 temp_k=TEMP_K,
                 fertile_kgm3=0.0, 
                 fertile_isotope='U238', 
                 li_enrich=None, 
                 run_type='tallies', 
                 n_particles=N_PARTICLES, 
                 n_cycles=N_CYCLES, 
                 print_xml=True, 
                 run_openmc=True, 
                 run_debug=True,
                 exit_pixel_cm=EXIT_PIXEL_CM, 
                 exit_energy_bins=EXIT_ENERGY_BINS,
                 source_rate=EXCALIBUR_SOURCE_RATE):

        self.fertile_kgm3    = fertile_kgm3
        self.fertile_isotope = fertile_isotope
        self.fissile_isotope = FISSILE.get(fertile_isotope, None)
        self.li_enrich       = li_enrich
        self.temp_k          = temp_k
        self.breeder         = breeder

        self.pile_code   = pile_code
        self.run_type    = run_type
        self.print_xml   = print_xml
        self.run_openmc  = run_openmc
        self.run_debug   = run_debug
        self.n_particles = n_particles
        self.n_cycles    = n_cycles
        if not np.isfinite(exit_pixel_cm) or exit_pixel_cm <= 0:
            raise ValueError("Exit pixel width must be positive and finite.")
        if isinstance(exit_energy_bins, bool) or not isinstance(exit_energy_bins, (int, np.integer)) or exit_energy_bins < 1:
            raise ValueError("Exit energy bin count must be a positive integer.")
        if source_rate is not None and (not np.isfinite(source_rate) or source_rate <= 0):
            raise ValueError("Source rate must be positive and finite.")
        self.exit_pixel_cm = exit_pixel_cm
        self.exit_energy_bins = exit_energy_bins
        self.source_rate = source_rate

        run_size = f"{self.n_particles:.0e}x{self.n_cycles}".replace("+0", "").replace("+", "")
        self.name = f"{pile_code}_{breeder}_{temp_k}K_{fertile_isotope}_{fertile_kgm3:07.2f}kgm3_{run_size}"    
        # Keep statepoints without exit tallies in their original directories.
        if run_type == 'tallies' and pile_code != 'a02':
            self.name += f"_exit{exit_pixel_cm:g}cm_{exit_energy_bins}g_{exit_grid_id(exit_pixel_cm, exit_energy_bins)}"
        self.path = DIR_OPENMC/self.name

        start_msg = f"\n======== {breeder} blanket - {fertile_isotope} {fertile_kgm3:07.2f} kg/m3 - {self.temp_k} K ========"
        print(f"{C.CYAN}{start_msg}{C.END}")



    def compile(self):
        
        openmc.reset_auto_ids()
        
        self.build_materials()
        self.build_geometry()
        self.build_settings()
        self.build_tallies()

        self.materials.cross_sections = set_xs_path()
        self.model = openmc.model.Model(self.geometry, self.materials, self.settings, self.tallies)

        if self.print_xml:
            self.model.export_to_model_xml(self.path)
            print(f"{C.YELLOW}Comment.{C.END} OpenMC model XML files printed to: {self.path}")
        else:
            print(f"{C.RED}Warning.{C.END} OpenMC model XML files NOT printed (print_xml=False) to: {self.path}")


    def openmc(self):

        statepoint = self.path / f"statepoint.{self.n_cycles}.h5"
        if self.run_openmc:
            if statepoint.is_file():
                print(f"{C.YELLOW}Warning.{C.END} File {self.path}/statepoint.{str(self.n_cycles).zfill(2)}.h5 already exists, so this OpenMC run will be skipped...")
            else:
                statepoint = Path(self.model.run(cwd=self.path))

                # Clean up intermediate statepoint files -- KEEP UNDER THIS LAST 'ELSE'!! --ppark 2026-05-01
                for filename in sorted(os.listdir(self.path)):
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

        return statepoint if statepoint.is_file() else None


    def extract_tallies(self, statepoint):
        with openmc.StatePoint(statepoint, autolink=False) as sp:
            for tally in sp.tallies.values():
                # Save these spectra in compressed arrays instead of large CSV files.
                if tally.name.startswith("neutron exit "):
                    continue
                name = tally.name.replace(" ", "_")
                output = self.path / f"{tally.id:03d}_{name}.csv"
                tally.get_pandas_dataframe().to_csv(output, index=False)
            export_exit_maps(sp, self.path / "neutron_exit", self.source_rate)


    def plot(self):

        if not self.run_openmc:
            print(f"{C.YELLOW}Warning.{C.END} Geometry plotting skipped.")
            return

        self.build_materials()
        self.build_geometry()

        plot_dir = DIR_OPENMC / self.name
        plot_dir.mkdir(parents=True, exist_ok=True)

        colors = {self.m_breeder: (  0, 128, 128),    # Teal
                  self.m_buff:    (170, 170, 170),    # Light gray
                  self.m_PyC1:    ( 80,  80,  80),    # Dark gray
                  self.m_PyC2:    ( 80,  80,  80),    # Dark gray
                  self.m_SiC:     (120, 120, 120),    # Gray
                  self.m_xo2:     (255, 165,   0), }  # Orange 

        plots = openmc.Plots()

        # Full box slices: XY is 20 × 20 cm; XZ is 20 × 50 cm.
        for basis, coordinate in (("xy", self.plot_origin[2]), ("xz", self.plot_origin[1]),):
            plot = openmc.SlicePlot.from_geometry(self.geometry, basis=basis, slice_coord=coordinate,)
            plot.filename = basis
            plot.pixels = tuple(max(1, round(width * 100)) for width in plot.width)
            plots.append(plot)

        # Close view to show the thin TRISO layers.
        if self.n_trisos > 0:
            zoom = openmc.SlicePlot()
            zoom.filename = "triso"
            zoom.basis = "xy"
            zoom.origin = self.plot_origin
            zoom.width = (3 * TRISO_RADII[-1],) * 2
            zoom.pixels = (800, 800)
            plots.append(zoom)

        for plot in plots:
            plot.color_by = "material"
            plot.colors = colors
            plot.background = "white"

        settings_plot = openmc.Settings()
        settings_plot.run_mode = "plot"

        model_plot = openmc.model.Model(geometry=self.geometry, materials=self.materials, settings=settings_plot, plots=plots,)
        model_plot.plot_geometry(cwd=plot_dir)

        for plot in plots:
            png_file = plot_dir / f"{plot.filename}.png"
            ppm_file = plot_dir / f"{plot.filename}.ppm"

            # Convert PPM output if OpenMC was built without PNG support.
            if not png_file.is_file() and ppm_file.is_file():
                with Image.open(ppm_file) as image:
                    image.save(png_file)

            if not png_file.is_file():
                raise FileNotFoundError(f"Plot image missing: {png_file}")

            print(f"{C.GREEN}Comment.{C.END} Plot saved to: {png_file}")
