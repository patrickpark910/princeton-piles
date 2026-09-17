from pathlib import Path

import numpy as np
import openmc


# ============================================================
# User settings
# ============================================================

WALL_THICKNESS_CM = 100.0
DX_CM = 1.0

BATCHES = 50
PARTICLES_PER_BATCH = 20_000

# Placeholder ClLiF composition
CLLIF_LICL_MOL_FRAC = 0.695
CLLIF_DENSITY = {900: 1.550, 293: 1.800}

# Natural elemental boron loading by total material weight
BORATED_PE_BORON_WT_FRAC = 0.05
BORATED_PE_DENSITY = 1.00  # g/cm3; nominal, adjust for the actual material



# ============================================================
# Materials
# ============================================================

def make_flibe(temp_K):

    mat = openmc.Material(name="FLiBe", temperature=temp_K)    
    mat.add_elements_from_formula('F4Li2Be', 'ao', enrichment_target='Li6', enrichment_type='ao', enrichment=7.50)
    mat.set_density('g/cm3', 1.9505)

    return mat


def make_cllif(temp_K, ):

    x = CLLIF_LICL_MOL_FRAC
    density = CLLIF_DENSITY.get(temp_K, CLLIF_DENSITY[293.0])

    # x LiCl + (1-x) LiF / atom counts: Li = 1, Cl = x, F  = 1-x

    mat = openmc.Material(name="ClLiF", temperature=temp_K)
    mat.add_element("Li", 1.0, percent_type="ao")
    mat.add_element("Cl", x, percent_type="ao", enrichment_target="Cl37", enrichment=90.0, enrichment_type="ao",)
    mat.add_element("F", 1.0 - x, percent_type="ao")
    mat.set_density("g/cm3", density)

    return mat


def make_ptfe(temp_K):

    mat = openmc.Material(name="PTFE", temperature=temp_K)

    # CF2 repeating unit
    mat.add_element("C", 1.0, percent_type="ao")
    mat.add_element("F", 2.0, percent_type="ao")
    mat.set_density("g/cm3", 2.20)

    return mat


def make_hdpe(temp_K):

    mat = openmc.Material(name="HDPE", temperature=temp_K)

    # CH2 repeating unit; nominal high-density polyethylene density
    mat.add_element("C", 1.0, percent_type="ao")
    mat.add_element("H", 2.0, percent_type="ao")
    mat.set_density("g/cm3", 0.95)
    mat.add_s_alpha_beta("c_H_in_CH2")

    return mat


def make_borated_polyethylene(temp_K):

    boron_fraction = BORATED_PE_BORON_WT_FRAC
    polyethylene_fraction = 1.0 - boron_fraction
    carbon_mass = openmc.data.atomic_weight("C")
    hydrogen_mass = 2.0 * openmc.data.atomic_weight("H")
    ch2_mass = carbon_mass + hydrogen_mass

    mat = openmc.Material(name="Borated polyethylene", temperature=temp_K)

    # Preserve CH2 stoichiometry in the polyethylene portion.
    mat.add_element("C", polyethylene_fraction * carbon_mass / ch2_mass, percent_type="wo")
    mat.add_element("H", polyethylene_fraction * hydrogen_mass / ch2_mass, percent_type="wo")
    mat.add_element("B", boron_fraction, percent_type="wo")  # Natural B10/B11
    mat.set_density("g/cm3", BORATED_PE_DENSITY)
    mat.add_s_alpha_beta("c_H_in_CH2")

    return mat


def make_graphite(temp_K):

    mat = openmc.Material(name="Graphite", temperature=temp_K)

    mat.add_element("C", 1.0)
    mat.set_density("g/cm3", 1.70)
    mat.add_s_alpha_beta("c_Graphite")

    return mat


def build_materials(temp_K):

    return {"flibe":  make_flibe(temp_K),
            "cllif":  make_cllif(temp_K),
            "teflon": make_ptfe(temp_K),
            "hdpe":   make_hdpe(temp_K),
            "bpe":    make_borated_polyethylene(temp_K),
            "grphC":  make_graphite(temp_K),}


# ============================================================
# Build OpenMC model
# ============================================================

def build_model(material_name, temp_K):

    material_dict = build_materials(temp_K)

    if material_name not in material_dict:
        raise ValueError(f"Unknown material '{material_name}'. Choose from {list(material_dict)}")

    wall_material = material_dict[material_name]

    # ========================================================
    # Geometry
    #
    #          |<--------- 100 cm ---------->|
    #   source |                             |
    #          |          MATERIAL           |
    #          |                             |
    #       x=0                             x=100
    #
    # Reflecting boundaries in y and z make this a laterally infinite slab.
    # ========================================================

    x_left = openmc.XPlane(x0=-1.0, boundary_type="vacuum",)
    x_front = openmc.XPlane(x0=0.0)
    x_back = openmc.XPlane(x0=WALL_THICKNESS_CM,)
    x_right = openmc.XPlane(x0=WALL_THICKNESS_CM + 1.0,boundary_type="vacuum",)

    # 1 cm x 1 cm transverse box
    y_min = openmc.YPlane(y0=-0.5,boundary_type="reflective",)
    y_max = openmc.YPlane(y0=0.5,boundary_type="reflective",)

    z_min = openmc.ZPlane(z0=-0.5,boundary_type="reflective",)
    z_max = openmc.ZPlane(z0=0.5,boundary_type="reflective",)

    transverse_region = (+y_min & -y_max & +z_min & -z_max)

    # Source region
    source_region = (+x_left & -x_front & transverse_region)
    source_cell = openmc.Cell(name="source region",region=source_region,)

    # Wall
    wall_region = (+x_front & -x_back & transverse_region)
    wall_cell = openmc.Cell(name=f"{material_name} wall", fill=wall_material, region=wall_region,)

    # Downstream vacuum
    downstream_region = (+x_back & -x_right & transverse_region )
    downstream_cell = openmc.Cell(name="downstream region", region=downstream_region,)

    geometry = openmc.Geometry([source_cell, wall_cell, downstream_cell])
    materials = openmc.Materials([wall_material])


    # ========================================================
    # Source
    # ========================================================

    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((-0.5, 0.0, 0.0))  # Start just in front of wall
    source.angle = openmc.stats.Monodirectional(reference_uvw=(1.0, 0.0, 0.0))  # All source neutrons directed in +x
    source.energy = openmc.stats.Discrete([14.07e6],[1.0],)
    source.particle = "neutron"

    # ========================================================
    # Settings
    # ========================================================

    settings = openmc.Settings()

    settings.run_mode = "fixed source"
    settings.source = source

    settings.batches = BATCHES
    settings.inactive = 0
    settings.particles = PARTICLES_PER_BATCH

    # ========================================================
    # Tallies
    # ========================================================

    tallies = openmc.Tallies()

    # 1-cm spatial mesh
    # 100 cm / 1 cm = 100 bins
    # bin0 = 0-1 cm, bin1 = 1-2 cm, ... bin99 = 99-100 cm

    nx = int(WALL_THICKNESS_CM / DX_CM)
    mesh = openmc.RegularMesh()
    mesh.dimension = (nx, 1, 1)
    mesh.lower_left = (0.0,-0.5,-0.5,)
    mesh.upper_right = (WALL_THICKNESS_CM,0.5,0.5,)
    mesh_filter = openmc.MeshFilter(mesh)

    # Energy bins / 1e-3 eV -> 20 MeV
    energy_edges  = np.logspace(np.log10(1.0e-3),np.log10(20.0e6),250,)
    energy_filter = openmc.EnergyFilter(energy_edges)

    # Flux(E, x)
    flux_spectrum = openmc.Tally(name="flux_vs_depth_and_energy")
    flux_spectrum.filters = [mesh_filter,energy_filter,]
    flux_spectrum.scores = ["flux",]
    tallies.append(flux_spectrum)

    # Current leaving back of wall
    transmitted = openmc.Tally(name="transmitted_current")
    transmitted.filters = [openmc.SurfaceFilter(x_back)]
    transmitted.scores = ["current"]
    tallies.append(transmitted)

    # ========================================================
    # Model
    # ========================================================

    return openmc.Model(geometry=geometry, materials=materials, settings=settings, tallies=tallies,)


# ============================================================
# Run all materials
# ============================================================

if __name__ == "__main__":

    base_dir = Path("OpenMC")
    base_dir.mkdir(exist_ok=True)

    for (material_name, temp_K) in [("flibe", 900), ("cllif", 900), ("cllif", 294), ("teflon",294), ("hdpe", 294), ("bpe", 294), ("grphC", 294)]:

        print()
        print("=" * 70)
        print(f"Running {material_name}")
        print("=" * 70)

        run_name = f"sigma_{material_name}_{temp_K}K"
        run_directory = base_dir / run_name
        run_directory.mkdir(parents=True, exist_ok=True,)

        statepoint = run_directory / f"statepoint.{BATCHES}.h5"
        if statepoint.is_file():
            print(f"Skipping {run_name}: {statepoint.name} already exists")
            continue

        model = build_model(material_name, temp_K)
        model.export_to_xml(directory=run_directory)
        openmc.config["cross_sections"] = ("/home/patri/openmc/data/endf-b8.1-hdf5/cross_sections.xml")
        openmc.run(cwd=run_directory)

        print(f"Finished {material_name}")