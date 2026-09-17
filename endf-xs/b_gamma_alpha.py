"""Export abundance-weighted natural boron (n,gamma) plus (n,alpha) using OpenMC."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmc
import openmc.data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "openmc/data/endf-b8.1-hdf5/neutron")
    parser.add_argument("--temperature", default="294K")
    parser.add_argument("--library", default="ENDF/B-VIII.1")
    args = parser.parse_args()
    abundances = {name: fraction for name, fraction in openmc.data.NATURAL_ABUNDANCE.items() if name.startswith("B") and name[1:].isdigit()}
    np.testing.assert_allclose(sum(abundances.values()), 1.0, rtol=0, atol=1e-12)
    reactions = {}
    sources = {}
    temperatures = []
    for name in abundances:
        path = args.data_dir / f"{name}.h5"
        data = openmc.data.IncidentNeutron.from_hdf5(str(path))
        if data.name != name:
            raise ValueError(f"Expected {name}, got {data.name}")
        for mt in (102, 107):
            reactions[(name, mt)] = data[mt].xs[args.temperature]
        temperatures.append(float(data.kTs[data.temperatures.index(args.temperature)] / openmc.data.K_BOLTZMANN))
        sources[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    np.testing.assert_allclose(temperatures, temperatures[0], rtol=0, atol=1e-6)
    # Keep all isotope reaction knots over their common supported energy range.
    low = max(xs.x[0] for (name, mt), xs in reactions.items() if mt == 102)
    high = min(xs.x[-1] for xs in reactions.values())
    energy = np.unique(np.concatenate([xs.x[(xs.x >= low) & (xs.x <= high)] for xs in reactions.values()]))
    capture = sum(abundances[name] * np.where(energy >= xs.x[0], xs(energy), 0.0) for (name, mt), xs in reactions.items())
    assert np.all(np.diff(energy) > 0)
    assert np.all(np.isfinite(capture)) and np.all(capture >= 0)
    # Check the mixture independently with NumPy interpolation at all knots.
    for xs in reactions.values():
        assert np.all(xs.interpolation == 2), "Independent check requires linear-linear data"
    independent = sum(abundances[name] * np.interp(energy, xs.x, xs.y, left=0.0) for (name, mt), xs in reactions.items())
    np.testing.assert_allclose(capture, independent, rtol=1e-12, atol=0)
    stem = Path(__file__).resolve().parent / f"B_gamma-alpha_{args.temperature}"
    table = np.column_stack((energy, capture))
    np.savetxt(stem.with_suffix(".csv"), table, delimiter=",", header="energy_eV,cross_section_barns", comments="", fmt="%.17e")
    np.testing.assert_array_equal(np.loadtxt(stem.with_suffix(".csv"), delimiter=",", skiprows=1), table)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 6.4), layout="constrained")
    ax.loglog(energy, np.ma.masked_less_equal(capture, 0), color="#176aa6", linewidth=1.05, label=r"Natural B(n,$\gamma$) + B(n,$\alpha$)  [MT 102 + 107]")
    ax.set(title=f"Natural boron: gamma + alpha cross sections\n{args.library}  |  {temperatures[0]:.1f} K", xlabel="Incident neutron energy (eV)", ylabel="Microscopic cross section (barns per natural B atom)", xlim=(energy[0], energy[-1]))
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    ax.legend(framealpha=0.95)
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    metadata = {
        "element": "natural boron", "reaction": "(n,gamma) + (n,alpha)", "MT": [102, 107],
        "library": args.library, "temperature_key": args.temperature, "temperature_K": temperatures[0],
        "openmc_version": openmc.__version__, "isotope_atom_fractions": abundances,
        "abundance_source": "openmc.data.NATURAL_ABUNDANCE", "sources": sources,
        "rows": len(energy), "energy_range_eV": [float(energy[0]), float(energy[-1])],
        "columns": {"energy_eV": "Incident neutron energy in eV", "cross_section_barns": "Summed microscopic gamma + alpha cross section in barns per natural B atom"},
        "method": "Sum of isotope MT 102 and MT 107 cross sections weighted by natural atom fractions, evaluated with OpenMC on the union of all isotope reaction knots within the common supported incident-energy range; alpha cross sections are zero below their thresholds. No downsampling.",
        "note": "Pointwise cross sections; no self-shielding or spectrum averaging. Unresolved-resonance probability tables are not sampled.",
        "checks": "Abundances sum to one; all isotopes have matching temperatures; weighted mixture matches independent linear interpolation; CSV round-trip preserves every value."
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    thermal = sum(abundances[name] * (xs(0.0253) if 0.0253 >= xs.x[0] else 0.0) for (name, mt), xs in reactions.items())
    print(f"Wrote {stem.name}.csv/.png/.svg/.json: {len(energy)} points, {energy[0]:g} to {energy[-1]:g} eV")
    print(f"Gamma + alpha at 0.0253 eV: {thermal:.8g} barns per natural B atom")


if __name__ == "__main__":
    main()
