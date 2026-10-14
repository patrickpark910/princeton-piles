
# ------------------------------------------------
# Universal parameters
# ------------------------------------------------

# Run settings
FERTILE_KGM3 = [0.0, 0.01, 0.10, 1, 10, 100]
BLANKETS     = ['flibe','cllif','grph','pbli','pebble']
ISOTOPES     = ['U238', 'Th232']
TEMP_K       = 900   # use 294 K for room temp (to align with endf available temps)


# ------------------------------------------------
# Breeder-specific parameters
# ------------------------------------------------

# FLiBe (2(LiF)-BeF2)
# NB. can also use utilities.py/flibe_density() to calculate density at different (T, P)
DENSITY_FLIBE    =  1.9505  # [g/cm³] at 900 K, 1 atm -- calculated by utilities.py
ENRICH_FLIBE     =  7.50    # [at%] enrich of Li-6

# Teflon
# T_melt = 326°C / 600 K / 621 F
DENSITY_TEFLON  =  2.1674  # [g/cm³] at 294 K -- calculated by utilities.py/density_teflon(294)

# ClLiF (LiF-ClF2)
# DENSITY_CLLIF   =  2.42  # [g/cm³] 
# ENRICH_CLLIF    =  7.50  # [at%] enrich of Li-6

# Pb
DENSITY_PB    =  11.34  # [g/cm³]

# Li4SiO4-Be
DENSITY_LI4SIO4 =  2.42  # [g/cm³] # should we normalize to ~2.17 for ceramic porosity and 900 K? (pure, room temp g/cm3 = 2.42) --ppark
DENSITY_BE      =  1.85  # [g/cm³] # could normalize to 1.80 for 900 K from 1.85 at STP
ENRICH_HCPB     = 60.00  # [at%] enrich of Li-6
