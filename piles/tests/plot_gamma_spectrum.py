"""Save spectra for the 24 isotope-component pairs with the highest total rates.

Run from the repository root:
    .venv/bin/python piles/tests/plot_gamma_spectrum.py
    .venv/bin/python piles/tests/plot_gamma_spectrum.py path/to/total.csv -o plot.png

Requires matplotlib, numpy, and pandas. No OpenMC installation is required.
Rates are in photons/s per energy bin, as in the CSV file. Select pairs by
the sum of photons_per_s across all energy bins, not by the largest peak.
Both axes use logarithmic scales. If the first bin starts at zero, the
energy axis starts at that bin's midpoint.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV = (Path(__file__).resolve().parents[1] / 'outputs'
               / 'a02_flibe_900K_U238_0001.00kgm3_1e6x10' / 'total.csv')
TOP_COUNT = 24


def read_spectra(path):
    """Read the bin rates and rank each nuclide-component pair by its sum."""
    columns = ['nuclide', 'component', 'E_low_keV', 'E_high_keV', 'photons_per_s']
    data = pd.read_csv(path, usecols=columns)
    if data.empty or data[['nuclide', 'component']].isna().any().any():
        raise ValueError('The CSV must contain spectra with nuclide and component labels.')
    values = data[['E_low_keV', 'E_high_keV', 'photons_per_s']].to_numpy()
    if (not np.isfinite(values).all() or (values < 0.).any()
            or (data.E_high_keV <= data.E_low_keV).any()):
        raise ValueError('Energy bounds and rates must be finite and nonnegative. '
                         'Each upper energy bound must exceed its lower bound.')
    totals = data.groupby(['nuclide', 'component']).photons_per_s.sum()
    top = totals[totals > 0.].sort_values(ascending=False, kind='stable').head(TOP_COUNT)
    if top.empty:
        raise ValueError('The CSV contains no positive photon rates.')
    return data, top


def plot_spectra(data, top, output):
    """Plot the selected spectra on the CSV energy grid and save a PNG file."""
    # Recover omitted zero bins. Do not connect peaks across those bins.
    edges = np.unique(data[['E_low_keV', 'E_high_keV']].to_numpy())
    low_indices = np.searchsorted(edges, data.E_low_keV)
    high_indices = np.searchsorted(edges, data.E_high_keV)
    if np.any(high_indices != low_indices + 1):
        raise ValueError('The spectra must use a common energy grid without overlapping bins.')
    data = data.assign(bin_index=low_indices)
    midpoints = (edges[:-1] + edges[1:]) / 2000.
    # Zero cannot appear on a logarithmic axis. Retain its bin's midpoint.
    energy_min = edges[0] / 1000. if edges[0] > 0. else midpoints[0]
    groups = data.groupby(['nuclide', 'component'])
    colors = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('Dark2').colors[:4])
    styles = {'prompt': '-', 'daughter': '--', 'passive': ':'}

    fig, ax = plt.subplots(figsize=(16, 10), layout='constrained')
    try:
        for rank, ((nuclide, component), total) in enumerate(top.items(), 1):
            group = groups.get_group((nuclide, component))
            rates = np.zeros(len(edges) - 1)
            np.add.at(rates, group.bin_index.to_numpy(), group.photons_per_s.to_numpy())
            label = f'{rank:2d}. {nuclide} ({component}): {total:.3e} /s'
            ax.stairs(np.where(rates > 0., rates, np.nan), edges / 1000.,
                      baseline=None, color=colors[rank - 1],
                      linestyle=styles.get(component, '-'), linewidth=1.2, label=label)
            # Mark positive bins so that isolated narrow peaks remain visible.
            positive = rates > 0.
            ax.plot(midpoints[positive], rates[positive], '.', color=colors[rank - 1],
                    markersize=2.5)
        ax.set(xscale='log', yscale='log', xlim=(energy_min, edges[-1] / 1000.),
               xlabel='Gamma energy [MeV]', ylabel='Photon rate [photons/s per bin]',
               title=f'Gamma spectra: {len(top)} highest isotope-component combinations\n'
                     'Ranked by total photon rate across all energy bins')
        ax.grid(True, which='major', alpha=0.25)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
                  fontsize=9, title='Isotope (component): total photon rate', frameon=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format='png', dpi=180)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', type=Path, default=DEFAULT_CSV,
                        help='Input total.csv. The default is the a02 FLiBe run.')
    parser.add_argument('-o', '--output', type=Path,
                        help='Output PNG. The default is gamma_spectrum_top24.png beside the CSV.')
    args = parser.parse_args()
    output = args.output or args.csv.with_name('gamma_spectrum_top24.png')
    if output.suffix.lower() != '.png':
        parser.error('The output file must have a .png extension.')
    try:
        data, top = read_spectra(args.csv)
        plot_spectra(data, top, output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(top.rename('photons_per_s').to_string())
    print(f'Saved {output.resolve()}')


if __name__ == '__main__':
    main()
