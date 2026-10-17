
# ------------------------------------------------
# Code parameters
# ------------------------------------------------

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # resolve() converts to absolute path, parent goes one above
DIR_FIGURES = ROOT/"figures"
DIR_OPENMC  = ROOT/"outputs"

# Run settings
FERTILE_KGM3 = [0.0, 0.01, 0.10, 1, 10, 100]
BLANKETS     = ['flibe','cllif','grph','lead']
ISOTOPES     = ['U238', 'Th232']
TEMP_K       = 900   # use 294 K for room temp (to align with endf available temps)
N_PARTICLES  = int(1e5)
N_CYCLES     = int(100)
EXCALIBUR_SOURCE_RATE = int(1e8) # [n/s] -- real Excalibur source rate is 5e8-8e8 n/s

# Neutron exit maps. Edge bins also retain energies outside this log grid.
EXIT_PIXEL_CM = 1.0
EXIT_ENERGY_BINS = 300
EXIT_ENERGY_MIN_EV = 1e-5
EXIT_ENERGY_MAX_EV = 20e6
EXIT_MODE_MAX_REL_ERROR = 0.5

A02_IR_TIME = 3600.0 # [s] Irradiation time for a02; source stays on.


# ------------------------------------------------
# General physical-chemical parameters
# ------------------------------------------------

DENSITY = {'UO2': 10.50, 'ThO2': 10.00, 'UF4': 4.50, 'ThF4': 4.00, 
           'flibe': 1.95, 'teflon': 2.1674, 'cllif': 2.42, 'lead': 11.34}  # [g/cm³] at 900 K, 1 atm 


AMU = {'U235': 235.05, 'U238': 238.05, 'Th232': 232.04, 'O': 16.00}  # [g/mol]

U_ENRICH = 0.0072  # [wt-frac] natural uranium enrichment of U-235
AMU['U'] = 1.0 / (U_ENRICH / AMU['U235'] + (1-U_ENRICH) / AMU['U238'])

# Radii of TRISO layers: [fuel kernel, buffer, inner PyC, SiC, outer PyC]
TRISO_RADII  = [0.0215, 0.0315, 0.0350, 0.0385, 0.0425] # [cm] 

# Mass of fertile isotope in a single TRISO particle
FERTILE_MASS_IN_TRISO = {"U238": 3.8254e-4, "Th232": 3.6584e-4} # [g] 

FISSILE = {'U238': 'Pu239', 'Th232': 'U233'}






# ------------------------------------------------
# Breeder-specific parameters
# ------------------------------------------------

# FLiBe (2(LiF)-BeF2)
# NB. can also use utilities.py/flibe_density() to calculate density at different (T, P)
DENSITY_FLIBE    =  1.9505  # [g/cm³] at 900 K, 1 atm -- calculated by utilities.py
ENRICH_FLIBE     =  7.50    # [at%] enrich of Li-6

# Teflon
# T_melt = 326°C / 600 K / 621 F
DENSITY_TEFLON   =  2.1674  # [g/cm³] at 294 K -- calculated by utilities.py/density_teflon(294)

# ClLiF (LiF-ClF2)
# DENSITY_CLLIF   =  2.42  # [g/cm³] 
# ENRICH_CLLIF    =  7.50  # [at%] enrich of Li-6

# Pb
DENSITY_PB      =  11.34  # [g/cm³]
