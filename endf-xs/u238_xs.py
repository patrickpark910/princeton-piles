"""Plot and export U-238 microscopic cross sections using OpenMC data."""
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
    parser.add_argument("--data", type=Path, default=Path.home() / "openmc/data/endf-b8.1-hdf5/neutron/U238.h5")
    parser.add_argument("--temperature", default="294K", help="Exact temperature key in the HDF5 library")
    parser.add_argument("--library", default="ENDF/B-VIII.1", help="Source library label for plot and metadata")
    args = parser.parse_args()
    data = openmc.data.IncidentNeutron.from_hdf5(str(args.data))
    if data.name != "U238":
        raise ValueError(f"Expected U238, got {data.name}")
    if args.temperature not in data.temperatures:
        raise ValueError(f"Available temperatures: {data.temperatures}")
    energy = data.energy[args.temperature]
    values = []
    for mt in (18, 102):
        xs = data[mt].xs[args.temperature]
        result = np.zeros_like(energy)
        in_range = energy >= xs.x[0]
        if energy[-1] > xs.x[-1]:
            raise ValueError("Energy grid extends beyond the reaction data")
        result[in_range] = xs(energy[in_range])
        assert np.all(np.isfinite(result)) and np.all(result >= 0)
        # Independently check every native reaction knot against stored ordinates.
        indices = np.searchsorted(energy, xs.x)
        np.testing.assert_array_equal(energy[indices], xs.x)
        np.testing.assert_allclose(result[indices], xs.y, rtol=1e-12, atol=0)
        values.append(result)
    assert np.all(np.diff(energy) > 0)
    output = Path(__file__).resolve().parent
    stem = output / f"u238_fission_capture_{args.temperature}"
    csv_files = {}
    for reaction, xs in zip(("fission", "gamma"), values):
        table = np.column_stack((energy, xs))
        csv_path = output / f"U238_{reaction}_{args.temperature}.csv"
        np.savetxt(csv_path, table, delimiter=",", header="energy_eV,cross_section_barns", comments="", fmt="%.17e")
        np.testing.assert_array_equal(np.loadtxt(csv_path, delimiter=",", skiprows=1), table)
        csv_files[reaction] = csv_path.name

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 6.4), layout="constrained")
    for xs, label, color in zip(values, [r"$^{238}$U(n,fission)  [MT 18]", r"$^{238}$U(n,$\gamma$)  [MT 102]"], ["#c75a20", "#176aa6"]):
        ax.loglog(energy, np.ma.masked_less_equal(xs, 0), label=label, color=color, linewidth=1.05)
    actual_temperature = float(data.kTs[data.temperatures.index(args.temperature)] / openmc.data.K_BOLTZMANN)
    ax.set(title=f"Uranium-238 neutron cross sections\n{args.library}  |  {actual_temperature:.1f} K", xlabel="Incident neutron energy (eV)", ylabel="Microscopic cross section (barns)", xlim=(energy[0], energy[-1]))
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    ax.legend(loc="lower left", framealpha=0.95)
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    metadata = {
        "nuclide": data.name, "library": args.library,
        "source_file": str(args.data.resolve()),
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "openmc_version": openmc.__version__, "temperature_key": args.temperature,
        "temperature_K": actual_temperature,
        "rows": len(energy), "energy_range_eV": [float(energy[0]), float(energy[-1])],
        "csv_files": csv_files, "reaction_MT": {"fission": 18, "gamma": 102}, "columns": {"energy_eV": "Incident neutron energy, eV", "cross_section_barns": "Microscopic reaction cross section, barns"},
        "method": "OpenMC IncidentNeutron.from_hdf5; native full temperature energy grid, without rebinning or downsampling. Zero below a reaction threshold. CSV round-trip and native reaction ordinates verified.",
        "note": "Pointwise evaluated microscopic cross sections, not spectrum-averaged or self-shielded group constants. Unresolved-resonance probability tables are not sampled. Zero values are retained in CSV and masked on the logarithmic plot.",
        "openmc_documentation": "https://docs.openmc.org/en/stable/pythonapi/generated/openmc.data.IncidentNeutron.html"
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    for e in (0.0253, 1e6, 14e6):
        print(f"E={e:g} eV: fission={data[18].xs[args.temperature](e):.8g} b, capture={data[102].xs[args.temperature](e):.8g} b")


if __name__ == "__main__":
    main()
