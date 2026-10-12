"""Plot total delayed-neutron yield per fission from ENDF-derived OpenMC data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc.data


ENDF_PATH = Path("/home/patri/openmc/data/endf-b8.1-hdf5/neutron")

ISOTOPES = [
    ("Th232", "090232"),
    ("U233", "092233"),
    ("U235", "092235"),
    ("U238", "092238"),
    ("Pu239", "094239"),
]

E_MIN = 1.0e-5       # Minimum incident neutron energy [eV]
E_MAX = 20.0e6       # Maximum incident neutron energy [eV]
E_LOG_MIN = 1.0e-2   # Minimum energy shown on the logarithmic plot [eV]
E_REFERENCE = 14.07e6
N_POINTS = 2000


def main():
    results = {}

    for label, zaid in ISOTOPES:
        mat = openmc.data.IncidentNeutron.from_hdf5(ENDF_PATH / f"{label}.h5")
        delayed = [
            p for p in mat[18].products
            if p.particle == "neutron" and p.emission_mode == "delayed"
        ]
        if not delayed:
            raise ValueError(f"No delayed-neutron yields found for {label}.")

        native_grids = [
            p.yield_.x for p in delayed
            if isinstance(p.yield_, openmc.data.Tabulated1D)
        ]
        if any(E_MIN < grid[0] or E_MAX > grid[-1] for grid in native_grids):
            raise ValueError(f"Requested energy range exceeds the yield data for {label}.")

        # Include native data knots and samples for both energy scales.
        energy = np.unique(np.concatenate([
            np.geomspace(E_MIN, E_MAX, N_POINTS),
            np.linspace(E_MIN, E_MAX, N_POINTS),
            np.array([E_REFERENCE, E_LOG_MIN]),
            *native_grids,
        ]))
        energy = energy[(energy >= E_MIN) & (energy <= E_MAX)]

        # Sum all delayed groups. This is yield per fission, integrated over decay time.
        nu_d = np.sum([p.yield_(energy) for p in delayed], axis=0)
        if not np.all(np.isfinite(nu_d)) or np.any(nu_d < 0):
            raise ValueError(f"Invalid delayed-neutron yields for {label}.")
        results[label] = {"energy": energy, "nu_d": nu_d}

        reference_yield = sum(float(p.yield_(E_REFERENCE)) for p in delayed)
        print(f"{label} ({zaid}): nu_d({E_REFERENCE / 1e6:g} MeV) = {reference_yield:.6f} n/fission")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for label, result in results.items():
        axes[0].plot(result["energy"] / 1e6, result["nu_d"], linewidth=2, label=label)
        log_range = result["energy"] >= E_LOG_MIN
        axes[1].plot(result["energy"][log_range], result["nu_d"][log_range], linewidth=2, label=label)

    axes[0].set_title("Linear Energy")
    axes[0].set_xlabel("Incident neutron energy [MeV]")
    axes[1].set_title("Logarithmic Energy")
    axes[1].set_xlabel("Incident neutron energy [eV]")
    axes[1].set_xscale("log")

    for ax in axes:
        ax.margins(x=0.03)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)

    axes[0].set_ylabel(r"Delayed neutron yield, $\nu_d$ [n/fission]")
    fig.suptitle("Delayed Neutron Yield per Fission\nENDF/B-VIII.1")
    fig.tight_layout()
    output_dir = Path(__file__).resolve().parent
    fig.savefig(output_dir / "nu_d_over_energy.png", dpi=300)
    fig.savefig(output_dir / "nu_d_over_energy.pdf")
    plt.show()

    return results, fig


if __name__ == "__main__":
    main()
