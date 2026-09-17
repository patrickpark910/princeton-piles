"""Export abundance-weighted natural cadmium radiative capture using OpenMC."""
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
    abundances = {name: fraction for name, fraction in openmc.data.NATURAL_ABUNDANCE.items() if name.startswith("Cd")}
    np.testing.assert_allclose(sum(abundances.values()), 1.0, rtol=0, atol=1e-12)
    reactions = {}
    sources = {}
    temperatures = []
    for name in abundances:
        path = args.data_dir / f"{name}.h5"
        data = openmc.data.IncidentNeutron.from_hdf5(str(path))
        if data.name != name:
            raise ValueError(f"Expected {name}, got {data.name}")
        reactions[name] = data[102].xs[args.temperature]
        temperatures.append(float(data.kTs[data.temperatures.index(args.temperature)] / openmc.data.K_BOLTZMANN))
        sources[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    np.testing.assert_allclose(temperatures, temperatures[0], rtol=0, atol=1e-6)
    # Keep all isotope reaction knots over their common supported energy range.
    low = max(xs.x[0] for xs in reactions.values())
    high = min(xs.x[-1] for xs in reactions.values())
    energy = np.unique(np.concatenate([xs.x[(xs.x >= low) & (xs.x <= high)] for xs in reactions.values()]))
    capture = sum(abundances[name] * xs(energy) for name, xs in reactions.items())
    assert np.all(np.diff(energy) > 0)
    assert np.all(np.isfinite(capture)) and np.all(capture >= 0)
    # Check the mixture independently with NumPy interpolation at all knots.
    for xs in reactions.values():
        assert np.all(xs.interpolation == 2), "Independent check requires linear-linear data"
    independent = sum(abundances[name] * np.interp(energy, xs.x, xs.y) for name, xs in reactions.items())
    np.testing.assert_allclose(capture, independent, rtol=1e-12, atol=0)
    stem = Path(__file__).resolve().parent / f"Cd_capture_{args.temperature}"
    table = np.column_stack((energy, capture))
    np.savetxt(stem.with_suffix(".csv"), table, delimiter=",", header="energy_eV,cross_section_barns", comments="", fmt="%.17e")
    np.testing.assert_array_equal(np.loadtxt(stem.with_suffix(".csv"), delimiter=",", skiprows=1), table)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 6.4), layout="constrained")
    ax.loglog(energy, np.ma.masked_less_equal(capture, 0), color="#176aa6", linewidth=1.05, label=r"Natural Cd(n,$\gamma$)  [MT 102]")
    ax.set(title=f"Natural cadmium neutron capture\n{args.library}  |  {temperatures[0]:.1f} K", xlabel="Incident neutron energy (eV)", ylabel="Microscopic cross section (barns per natural Cd atom)", xlim=(energy[0], energy[-1]))
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    ax.legend(framealpha=0.95)
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    metadata = {
        "element": "natural cadmium", "reaction": "radiative capture (n,gamma)", "MT": 102,
        "library": args.library, "temperature_key": args.temperature, "temperature_K": temperatures[0],
        "openmc_version": openmc.__version__, "isotope_atom_fractions": abundances,
        "abundance_source": "openmc.data.NATURAL_ABUNDANCE", "sources": sources,
        "rows": len(energy), "energy_range_eV": [float(energy[0]), float(energy[-1])],
        "columns": {"energy_eV": "Incident neutron energy in eV", "cross_section_barns": "Microscopic capture cross section in barns per natural Cd atom"},
        "method": "Sum of isotope MT 102 cross sections weighted by natural atom fractions, evaluated with OpenMC on the union of all isotope capture knots within the common supported energy range. No downsampling.",
        "note": "Pointwise cross sections; no self-shielding or spectrum averaging. Unresolved-resonance probability tables are not sampled.",
        "checks": "Abundances sum to one; all isotopes have matching temperatures; weighted mixture matches independent linear interpolation; CSV round-trip preserves every value."
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    thermal = sum(abundances[name] * xs(0.0253) for name, xs in reactions.items())
    print(f"Wrote {stem.name}.csv/.png/.svg/.json: {len(energy)} points, {energy[0]:g} to {energy[-1]:g} eV")
    print(f"Capture at 0.0253 eV: {thermal:.8g} barns per natural Cd atom")


if __name__ == "__main__":
    main()
