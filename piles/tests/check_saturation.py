"""Run with: .venv/bin/python piles/tests/check_saturation.py

Compare finite irradiation with the former a02 fixed-parent equilibrium.
The initial model has natural uranium in UO2 and no added decay daughters.
All products remain in the system, including radon. Stable products can
react again. The neutron field stays fixed, as in transport-independent R2S.

The reported time is the first sampled time within 1% of the reference inventory
that stays within 1% through the remaining samples (at least three samples).
It is a comparison with the former model, not proof of infinite-time equilibrium.
CSV fields with 'a02' refer to this former equilibrium reference.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import openmc
from openmc.deplete import Chain, IndependentOperator
from openmc.deplete.cram import CRAM48
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python.gamma_saturation import add_rate_tallies, read_rate_data
from python.model import Pile
from python.parameters import TRISO_RADII


# Change the model and time range here. The source rate is the global constant.
BREEDER = 'flibe'
FERTILE_KGM3 = 0.01
PARTICLES, BATCHES = 20000, 5
MAX_YEARS, TIME_POINTS = 1e7, 160
TOLERANCE = 0.01
YEAR = 365.25 * 86400.
OUTPUT = Path(__file__).resolve().parents[1] / 'outputs' / 'saturation_check'


def equilibrium(chain, matrix, initial):
    """Retain the former a02 equilibrium model for this comparison only."""
    fixed = np.flatnonzero(initial > 0.)
    active = np.array([i for i, n in enumerate(chain.nuclides)
                       if n.half_life is not None and initial[i] == 0.], dtype=int)
    atoms = np.zeros((len(chain), 2))
    atoms[fixed, 0] = initial[fixed]
    if not len(active):
        return atoms
    loss = -matrix.diagonal()[active]
    passive = chain.decay_matrix[active][:, fixed] @ initial[fixed]
    total = matrix[active][:, fixed] @ initial[fixed]
    source = np.column_stack([passive, total - passive])
    balance = -matrix[active][:, active] @ diags(1. / loss)
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        flow = spsolve(balance.tocsc(), source)
    residual = np.linalg.norm(balance @ flow - source, 1) / max(np.linalg.norm(source, 1), 1e-300)
    if (not np.all(np.isfinite(flow)) or residual > 1e-7
            or np.any(flow < -1e-9 * max(source.max(), 1e-300))):
        raise ValueError('The radioactive products have no finite nonnegative equilibrium.')
    atoms[active] = np.maximum(flow, 0.) / loss[:, None]
    return atoms


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pile = Pile('a02', BREEDER, fertile_isotope='U238', fertile_kgm3=FERTILE_KGM3,
                n_particles=PARTICLES, n_cycles=BATCHES, print_xml=False)
    pile.path = OUTPUT
    pile.compile()
    source_energy = float(pile.settings.source[0].energy.x[0])

    # Exact total volumes of the existing TRISO layers and breeder.
    shells = [pile.m_xo2, pile.m_buff, pile.m_PyC1, pile.m_SiC, pile.m_PyC2]
    volumes = pile.n_trisos * 4 * np.pi / 3 * np.diff(np.r_[0., TRISO_RADII]**3)
    for material, volume in zip(shells, volumes):
        material.volume = float(volume)
    lower, upper = (-pile.pile_boundary).bounding_box
    pile.m_breeder.volume = float(np.prod(upper - lower) - volumes.sum())
    materials = sorted(pile.geometry.get_all_materials().values(), key=lambda m: m.id)
    for material in materials:
        material.depletable = True
    chain_file = Path(pile.materials.cross_sections).parent.parent / 'chain_endfb81_fast.xml'
    targets = {n for material in materials for n in material.get_nuclides()}
    chain = Chain.from_xml(chain_file).reduce(targets)
    print(f'Neutron source: {pile.source_rate:g} n/s, {source_energy / 1e6:g} MeV.', flush=True)
    add_rate_tallies(pile, materials, chain)
    statepoint = pile.openmc()
    if statepoint is None:
        return
    with openmc.StatePoint(statepoint, autolink=False) as sp:
        fluxes, micros = read_rate_data(sp, materials)
    operator = IndependentOperator(
        materials, fluxes, micros, chain_file=chain, normalization_mode='source-rate',
        fission_yield_opts={'energy': source_energy})
    inventories = operator.initial_condition()
    rates = operator(inventories, pile.source_rate).rates
    chain = operator.chain
    matrices = [chain.form_matrix(rates[i], chain.fission_yields[i])
                for i in range(len(materials))]
    reference = sum(equilibrium(chain, matrix, atoms).sum(axis=1)
                    for matrix, atoms in zip(matrices, inventories))

    # Logarithmic samples plus exact times for an hour, a day, and a year.
    times = np.unique(np.r_[0., np.geomspace(1., MAX_YEARS * YEAR, TIME_POINTS),
                            [t for t in (3600., 86400., YEAR) if t <= MAX_YEARS * YEAR]])
    history = [np.sum(inventories, axis=0)]
    print(f'Evolve complete inventories through {MAX_YEARS:g} years.', flush=True)
    for step, dt in enumerate(np.diff(times), 1):
        inventories = [np.maximum(CRAM48(matrix, atoms, float(dt)), 0.)
                       for matrix, atoms in zip(matrices, inventories)]
        history.append(np.sum(inventories, axis=0))
        if step % 40 == 0:
            print(f'  Time: {times[step] / YEAR:.4g} years.', flush=True)
    history = np.asarray(history)
    if not np.isfinite(history).all():
        raise ValueError('The inventory calculation produced nonfinite values.')

    summaries, buildup = [], []
    for i, nuclide in enumerate(chain.nuclides):
        if reference[i] == 0. and not np.any(history[:, i] > 0.):
            continue
        decay = np.log(2) / nuclide.half_life if nuclide.half_life is not None else 0.
        spectrum = nuclide.sources.get('photon')
        photons_per_atom = spectrum.integral() if spectrum is not None else 0.
        fraction = history[:, i] / reference[i] if reference[i] > 0. else np.full(len(times), np.nan)
        near = np.abs(fraction - 1.) <= TOLERANCE
        retained = np.logical_and.accumulate(near[::-1])[::-1]
        hits = np.flatnonzero(retained[:-2])
        reached = float(times[hits[0]] / YEAR) if len(hits) else np.nan
        status = 'within_tolerance' if len(hits) else 'not_reached_in_time_range'
        if nuclide.half_life is None:
            reached, status = np.nan, 'stable: no radioactive saturation target'
        elif reference[i] == 0.:
            reached, status = np.nan, 'no_a02_equilibrium_target'
        summaries.append(dict(
            nuclide=nuclide.name, half_life_years=nuclide.half_life / YEAR if decay else np.nan,
            a02_atoms=reference[i], a02_photons_per_s=reference[i] * photons_per_atom,
            sampled_saturation_years=reached, status=status,
            final_fraction_of_a02=fraction[-1]))
        buildup.extend(dict(time_s=t, time_years=t / YEAR, nuclide=nuclide.name,
                            atoms=n, activity_Bq=n * decay, photons_per_s=n * photons_per_atom,
                            fraction_of_a02=f)
                       for t, n, f in zip(times, history[:, i], fraction))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / 'saturation_times.csv', index=False)
    pd.DataFrame(buildup).to_csv(OUTPUT / 'buildup.csv', index=False)
    selected = ['Th234', 'Pa234_m1', 'U234', 'Th230', 'Ra226', 'Pb214', 'Bi214',
                'Pb210', 'Bi210', 'Bi208']
    print(summary.set_index('nuclide').reindex(selected)[
        ['sampled_saturation_years', 'status', 'a02_photons_per_s']].to_string())
    print(f'Outputs: {OUTPUT}')
    print('Times compare with the former a02 equilibrium; they are upper sample bounds.')
    print('Stable products and parent depletion are included in the time history.')
    print(f'Fission yields use the library energy nearest {source_energy / 1e6:g} MeV, as in a02.')


if __name__ == '__main__':
    main()
