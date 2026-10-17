# Neutron exit maps

Run from the repository root:

```bash
.venv/bin/python piles/piles.py -p a01 -b flibe -i U238 -f 0.1
```

Each tally run writes a `neutron_exit` directory inside its run directory in
`piles/outputs`. Run names include the pixel width, energy-bin count, and grid
identifier. This
keeps earlier runs without exit tallies in their original directories.

- `exit_current.png`: outward neutron current density on all six faces.
- `modal_energy.png`: modal neutron energy on all six faces.
- `<face>_spectra.npz`: the full energy spectrum and statistical errors for
  every pixel. Face names are `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`.
- `<face>_pixels.csv`: pixel bounds, areas, currents, errors, and modal energies.
- `metadata.json`: units, axis order, statistical criteria, and face totals.

The default maximum pixel width is 1 cm. The default spectrum has 300
logarithmic bins from 1e-5 eV to 20 MeV, plus two guard bins. The guard bins
retain neutrons below and above that range. Their outer edges are 0 and
1e30 eV. The upper edge is above the energies supported by nuclear libraries.

Use `--exit_pixel_cm 2` or `--exit_energy_bins 100` to change the resolution.
Non-divisible face lengths use equal pixels no wider than the requested width.

Maps use neutrons/cm2/s, scaled with `EXCALIBUR_SOURCE_RATE` from
`python/parameters.py`. The stored spectra remain per source neutron.
The source in `settings.py` has unit strength.

Use `--no_run` with the same model and resolution arguments to make the maps
again from an existing statepoint.

## Read a pixel spectrum

```python
import numpy as np

data = np.load("PATH_TO_RUN/neutron_exit/x_min_spectra.npz")
edges = data["energy_edges_eV"]
mean = data["spectrum_mean"][v_pixel, u_pixel, :]
std_dev = data["spectrum_std_dev"][v_pixel, u_pixel, :]
```

`mean[k]` is the number of escaping neutrons in energy bin `k`, per source
neutron, integrated over that pixel. Its uncertainty is `std_dev[k]`, one
standard error of the estimated mean. Divide by `pixel_area_cm2[v_pixel,
u_pixel]` for current density. Multiply by the physical source rate for rates
per second. No energy-bin width is included in these arrays.

The array order is `[v_pixel, u_pixel, energy_bin]`. Pixel indices start at 0.
`u_axis`, `v_axis`, `u_edges_cm`, and `v_edges_cm` give the global coordinates.
Faces are not mirrored. Both x faces use y horizontally and z vertically.
Both y faces use x horizontally and z vertically. Both z faces use x
horizontally and y vertically.

The modal energy is the geometric center of the bin with the largest
`spectrum_mean / delta(ln E)`. The guard bins do not have a log-energy mode.
A pixel has no finite mode if a guard bin dominates or no neutrons were tallied.
The PNG also masks pixels when the selected bin's relative error exceeds 50%.
The unmasked estimate, its bin limits, and `mode_reliable` remain in the data.
This flag tests the selected bin's precision. It does not establish a unique
peak when competing peaks have similar heights.

Currents have a separate energy-integrated tally. Its uncertainty includes
correlations between energy bins. Do not estimate this error by adding the
individual bin errors in quadrature. Export checks each pixel's spectral sum
against its current, and each face's pixel sum against the original boundary
current tally.

These are steady fixed-source results for the current composition. They do
not assume activation saturation or simulate one hour of irradiation.

## Check the calculation

```bash
PYTHONPATH=piles .venv/bin/python -m unittest discover -s piles/tests -v
```

The checks use six directed neutron sources in material with negligible
scattering. They test face selection, outward signs, pixel coordinates,
spectra, modal energy bins, normalization, empty pixels, and guard bins.
They require OpenMC and a neutron data library. Set `OPENMC_CROSS_SECTIONS`
when the library is not at the default local path.

# Gamma source spectra (a02)

```bash
.venv/bin/python piles/piles.py -p a02 -b flibe -i U238 -f 0.01
```

The source definition and transport settings come from `python/settings.py`.
The current source energy is 14.07 MeV. Prompt scaling and depletion use
`pile.source_rate`, whose default is `EXCALIBUR_SOURCE_RATE` from
`python/parameters.py` (currently 1e8 n/s). OpenMC uses normal thread selection
and displays its terminal output.
`A02_IR_TIME` in `python/parameters.py` sets the irradiation time in seconds.
Its default is 3600 s (1 hr). The spectra give the emission rate at that time,
with the neutron source still on.
a02 prepares prompt, material-flux, and reaction-rate tallies, then calls
`Pile.openmc()`. That method reuses an existing final statepoint or runs
transport and deletes intermediate statepoints. Saved tallies supply the R2S
rates through OpenMC's `MicroXS` and `IndependentOperator`. `CRAM48` evolves
the inventories through the irradiation time. No leakage calculation is
performed.

`piles/outputs/<run>/` contains three spectra: `prompt.csv`,
`daughter.csv`, and `passive.csv`. Each gives `nuclide`, `E_low_keV`,
`E_high_keV`, `E_mid_keV`, `photons_per_s`, and `std_dev_photons_per_s`.
Energy bounds and midpoints are in keV. The midpoint is `(low + high) / 2`.
`total.csv` combines all rows from these three files. Its `component` column
identifies `prompt`, `passive`, or `daughter`. Rows are sorted by
`photons_per_s`, from largest to smallest.
All three use the same energy bins and can be summed. Zero bins are omitted.
Prompt labels identify neutron targets. Decay labels identify decay parents.
Passive includes initial radioactivity and its daughter chains. Daughter
includes neutron-produced radionuclides and their decay chains. Both decay
components include neutron removal during irradiation, without double counting.
The passive and daughter inventories are evolved together. Their sum equals
the full irradiated inventory. Decay uncertainties are blank; inventory
uncertainties are not calculated.

The model holds the neutron field fixed. The inventories include parent
depletion, product decay, and further activation of stable products.
The prompt spectrum uses the initial composition and neutron field.
Rates are averaged over each material. Fission yields
use the available library energy nearest the configured monoenergetic source
energy. The supplied chain and
cross-section library determine the included reaction and decay paths.

Use `--chain_file PATH` to select a decay chain and `--gamma_energy_bins N`
to change the default 1000 logarithmic bins from 1 keV to 20 MeV. Guard bins
retain lower and higher energies. Transport files stay in the existing run
directory, with the final statepoint named `statepoint.<batches>.h5`.
Each call regenerates the CSV files from the returned statepoint. Thus, CSV
format changes do not require another transport run. `--no_run` uses an
existing final statepoint; if none exists, a02 returns without processing.
The saved prompt tally supplies its nuclide labels and photon energy bins.
No metadata file or additional output subdirectory is created.

All pile codes use the same `20 x 20 x 50` TRISO search lattice. This grid
preserves the individual TRISO positions, radii, and material layers.

Check the irradiation calculation:

```bash
PYTHONPATH=piles .venv/bin/python -m unittest discover -s piles/tests -v
```

These checks compare the inventories with analytic decay and activation
solutions. They also check the component sum, statepoint reuse, CSV
regeneration, and cleanup. Transport is mocked for the run checks. They
require OpenMC's Python package, but no simulation or nuclear data library.
