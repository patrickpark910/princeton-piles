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

Maps use neutrons/cm2 per source neutron. Supply `--source_rate RATE` in
neutrons/s to get current-density maps in neutrons/cm2/s. This scales the
reported density; it does not change the transport calculation or the stored
per-source spectra. For example, `--source_rate 1e8` sets a physical rate of
100 million source neutrons/s. The source in `settings.py` has unit strength.

Use `--no_run` with the same model and resolution arguments to make the maps
again from an existing statepoint. You can change `--source_rate` on this call.

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
