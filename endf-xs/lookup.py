
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import openmc

# --------------------------------------------------
# Parameters
# --------------------------------------------------

E = 14.07e6
T_IRR = 3600.0       # 1 hour irradiation [s]
R_F = 1.0            # Fission rate during irradiation [fissions/s]
DT = 1.0             # Time step [s]
T_MAX = 200.0        # Time after beam-off [s]

ENDF_PATH = Path("/home/patri/openmc/data/endf-b8.1-hdf5/neutron")

ISOTOPES = [("Th232", "090232"), ("U233",  "092233"),  ("U235",  "092235"), ("U238",  "092238"), ("Pu239", "094239"),]

""""
452  nu*sigma_f     product of average total fission neutron yield and fission cross section
455  nu_d*sigma_f   product of average delayed neutron yield       and fission cross section
456  nu_p*sigma_f   product of average prompt neutron yield        and fission cross section
458  kappa*sigma_f  product of average energy release in MeV       and fission cross section
"""


# Time samples, including the instant of beam-off
times = np.arange(0, T_MAX + DT, DT)
results = {}

# --------------------------------------------------
# Extract nuclear data and calculate emission
# --------------------------------------------------

for label, zaid in ISOTOPES:

    mat = openmc.data.IncidentNeutron.from_hdf5(ENDF_PATH / f"{label}.h5")

    delayed = [p for p in mat[18].products if p.particle == "neutron" and p.emission_mode == "delayed"]
    
    nu = np.array([float(p.yield_(E)) for p in delayed])
    lam = np.array([float(p.decay_rate) for p in delayed])

    # Total delayed neutron yield per fission
    nu_total = np.sum(nu)

    # Fractional saturation after irradiation
    sat = 1.0 - np.exp(-lam * T_IRR)

    # Emission rate at the instant of beam-off
    R0 = R_F * nu * sat

    # Delayed neutron emission rate after beam-off [n/s]
    rates = np.sum(R0[:, None] * np.exp(-lam[:, None] * times), axis=0)

    # All delayed neutrons eventually emitted after beam-off
    total_post = np.sum(R0 / lam)

    # Per fission occurring during irradiation
    post_per_fission = total_post / (R_F * T_IRR)

    results[label] = {"rates": rates, "nu_d": nu_total}

    print(f"\n{label}")
    print(f"  Total nu_d:          {nu_total:.6f} n/fis")
    print(f"  Beam-off rate:       {rates[0]:.6f} n/s")
    print(f"  Post-beam neutrons:  {total_post:.6f}")
    print(f"  Post-beam n/fission: {post_per_fission:.8f}")

# --------------------------------------------------
# Plot: delayed neutron emission rate
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for label, result in results.items():

    legend_label = rf"{label} ($\nu_d$ = {result['nu_d']:.4f} n/fis)"

    axes[0].plot(times, result["rates"], linewidth=2, label=legend_label)
    axes[1].plot(times, result["rates"], linewidth=2, label=legend_label)

axes[0].set_title("Linear Time")
axes[1].set_title(f"Logarithmic Time (linear below {DT:g} s)")
axes[1].set_xscale("symlog", linthresh=DT) # Include t=0 with a linear section up to DT, then use logarithmic scaling.

for ax in axes:
    # Apply 3% padding in each axis scale.
    ax.margins(x=0.03)
    ax.set_xlabel("Time after beam-off [s]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

axes[0].set_ylabel("Delayed neutron emission rate [n/s]")

fig.suptitle("Delayed Neutron Emission Rate after Beam Shut-off\n14.1-MeV beam irradiates to equilibrium rate of 1 fis/s")
plt.tight_layout()
plt.savefig("/endf-xs/delayed_neutron_emission.png", dpi=300)
plt.savefig("/endf-xs/delayed_neutron_emission.pdf",)
plt.show()
