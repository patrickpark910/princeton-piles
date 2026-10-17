"""a02: gamma source spectra at the end of irradiation, in photons/s per bin."""
import numpy as np
import openmc
from openmc.deplete import Chain, IndependentOperator, MicroXS
from openmc.deplete.cram import CRAM48
import pandas as pd
from scipy.sparse import bmat, diags

from python.parameters import A02_IR_TIME, TRISO_RADII


def add_rate_tallies(pile, materials, chain):
    """Prepare the existing one-group rate tallies without running transport."""
    library = openmc.data.DataLibrary.from_xml(pile.materials.cross_sections)
    available = {name for entry in library if entry['type'] == 'neutron'
                 for name in entry['materials']}
    nuclides = [n.name for n in chain.nuclides if n.name in available]
    material_filter = openmc.MaterialFilter(materials)
    flux = openmc.Tally(name='MicroXS flux 0')
    flux.filters, flux.scores = [material_filter], ['flux']
    rates = openmc.Tally(name='MicroXS RR 0')
    rates.filters, rates.nuclides = [material_filter], nuclides
    rates.scores, rates.multiply_density = chain.reactions, False
    pile.model.tallies.extend([flux, rates])


def read_rate_data(statepoint, materials):
    """Read rates in material order, using the saved nuclides and reactions."""
    selection = dict(filters=[openmc.MaterialFilter],
                     filter_bins=[tuple(m.id for m in materials)])
    flux = statepoint.get_tally(name='MicroXS flux 0').get_values(**selection)
    flux = flux.reshape(len(materials), 1)
    tally = statepoint.get_tally(name='MicroXS RR 0')
    rates = tally.get_values(**selection)[..., None]
    divisor = flux[:, None, None, :]
    xs = np.divide(rates, divisor, out=np.zeros_like(rates), where=divisor != 0.)
    return list(flux), [MicroXS(data, tally.nuclides, tally.scores) for data in xs]


def irradiate(chain, matrix, initial, time_s):
    """Evolve passive and neutron-produced atoms with the source on.

    Passive atoms follow decay paths and can be removed by neutron reactions.
    Atoms produced by neutron reactions enter the daughter inventory, which
    follows the full reaction and decay chain. Their sum solves dN/dt = A N.
    Use a coupled solve to avoid subtraction of nearly equal inventories.
    """
    if not np.isfinite(time_s) or time_s < 0.:
        raise ValueError('Irradiation time must be finite and nonnegative.')
    if time_s == 0.:
        return np.column_stack([initial, np.zeros_like(initial)])
    reactions = matrix - chain.decay_matrix
    removal = diags(reactions.diagonal())
    passive = chain.decay_matrix + removal
    production = reactions - removal
    coupled = bmat([[passive, None], [production, matrix]], format='csc')
    start = np.r_[initial, np.zeros_like(initial)]
    atoms = CRAM48(coupled, start, float(time_s)).reshape(2, len(chain)).T
    if not np.isfinite(atoms).all():
        raise ValueError('The irradiation calculation produced nonfinite inventories.')
    return np.maximum(atoms, 0.)


def bin_spectrum(distribution, edges):
    """Integrate native OpenMC decay distributions on the prompt energy grid."""
    if isinstance(distribution, openmc.stats.Discrete):
        return np.histogram(distribution.x, edges, weights=distribution.p)[0]
    if isinstance(distribution, openmc.stats.Mixture):
        return sum(weight * bin_spectrum(part, edges)
                   for weight, part in zip(distribution.probability, distribution.distribution))
    if isinstance(distribution, openmc.stats.Tabular):
        nodes = np.unique(np.r_[distribution.x,
            edges[(edges > distribution.x[0]) & (edges < distribution.x[-1])]])
        values = np.array([distribution.evaluate(float(x)) for x in nodes])
        # Restore exact knots; evaluate() returns zero at support boundaries.
        values[np.searchsorted(nodes, distribution.x[:len(distribution.p)])] = distribution.p
        values[-1] = distribution.p[-1]
        expanded = openmc.stats.Tabular(nodes, values, distribution.interpolation)
        return np.maximum(0., np.diff(np.interp(
            edges, nodes, expanded.cdf(), left=0., right=expanded.integral())))
    raise TypeError(f'Unsupported photon spectrum: {type(distribution).__name__}')


def save_spectrum(path, names, values, errors, edges):
    """Write energy bounds and arithmetic midpoints in keV."""
    edges_keV = edges / 1e3
    rows = [(name, edges_keV[k], edges_keV[k + 1],
             (edges_keV[k] + edges_keV[k + 1]) / 2., values[k, i], errors[k, i])
            for i, name in enumerate(names) for k in np.flatnonzero(values[:, i] > 0.)]
    pd.DataFrame(rows, columns=['nuclide', 'E_low_keV', 'E_high_keV', 'E_mid_keV',
                               'photons_per_s', 'std_dev_photons_per_s']).to_csv(path, index=False)


def save_total_spectrum(directory):
    spectra = [pd.read_csv(directory / f'{component}.csv').assign(component=component)
               for component in ('prompt', 'passive', 'daughter')]
    total = pd.concat(spectra, ignore_index=True)
    total.sort_values('photons_per_s', ascending=False, kind='stable').to_csv(
        directory / 'total.csv', index=False)


def run_saturation(pile, chain_file, energy_bins=1000):
    """Keep the driver entry point; export spectra at A02_IR_TIME, source on."""
    directory = pile.path
    if not np.isfinite(A02_IR_TIME) or A02_IR_TIME <= 0.:
        raise ValueError('A02_IR_TIME must be positive and finite, in seconds.')
    directory.mkdir(parents=True, exist_ok=True)
    edges = np.r_[0., np.geomspace(1e3, 20e6, energy_bins + 1), 1e30]
    source_energy = float(pile.settings.source[0].energy.x[0])

    # Average over each material. Use exact volumes for the existing spheres.
    shells = [pile.m_xo2, pile.m_buff, pile.m_PyC1, pile.m_SiC, pile.m_PyC2]
    volumes = pile.n_trisos * 4 * np.pi / 3 * np.diff(np.r_[0., TRISO_RADII]**3)
    for material, volume in zip(shells, volumes):
        material.volume = float(volume)
    lower, upper = (-pile.pile_boundary).bounding_box
    pile.m_breeder.volume = float(np.prod(upper - lower) - volumes.sum())
    materials = sorted(pile.geometry.get_all_materials().values(), key=lambda m: m.id)
    for material in materials:
        material.depletable = True
    targets = sorted({name for m in materials for name in m.get_nuclides()})
    chain = Chain.from_xml(chain_file).reduce(targets)
    prompt = openmc.Tally(name='a02 prompt production')
    prompt.filters = [openmc.ParticleFilter('neutron'),
                      openmc.ParticleProductionFilter('photon', energies=edges)]
    prompt.nuclides, prompt.scores, prompt.estimator = targets, ['events'], 'analog'
    pile.model.tallies = [prompt]
    add_rate_tallies(pile, materials, chain)
    statepoint = pile.openmc()
    if statepoint is None:
        return directory
    with openmc.StatePoint(statepoint, autolink=False) as sp:
        fluxes, micros = read_rate_data(sp, materials)
        tally = sp.get_tally(name=prompt.name)
        edges = tally.find_filter(openmc.ParticleProductionFilter).energies
        targets = tally.nuclides
        means = tally.mean.reshape(len(edges) - 1, len(targets)) * pile.source_rate
        errors = tally.std_dev.reshape(means.shape) * pile.source_rate
    save_spectrum(directory / 'prompt.csv', targets, means, errors, edges)

    operator = IndependentOperator(
        materials, fluxes, micros, chain_file=chain, normalization_mode='source-rate',
        fission_yield_opts={'energy': source_energy})
    initial = operator.initial_condition()
    rates = operator(initial, pile.source_rate).rates
    chain = operator.chain
    atoms = np.zeros((len(chain), 2))
    for i, inventory in enumerate(initial):
        matrix = chain.form_matrix(rates[i], chain.fission_yields[i])
        atoms += irradiate(chain, matrix, inventory, A02_IR_TIME)
    names, spectra = [], []
    for i, parent in enumerate(chain.nuclides):
        distribution = parent.sources.get('photon')
        if distribution is not None and np.any(atoms[i] > 0.):
            names.append(parent.name)
            spectra.append(bin_spectrum(distribution, edges)[:, None] * atoms[i])
    values = np.stack(spectra, axis=1) if spectra else np.zeros((len(edges) - 1, 0, 2))
    for column, component in enumerate(('passive', 'daughter')):
        save_spectrum(directory / f'{component}.csv', names, values[:, :, column],
                      np.full(values.shape[:2], np.nan), edges)
    save_total_spectrum(directory)
    print(f'a02 gamma spectra at {A02_IR_TIME:g} s, source on: {directory}')
    return directory
