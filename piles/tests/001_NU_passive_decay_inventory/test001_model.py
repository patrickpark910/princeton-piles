"""
TEST 001 - Natural UO2/ThO2 Passive Decay Inventory

See test001_main.py.
No geometry, settings, or tallies necessary for this decay-only depletion calculation.
    
This file is 100% human-written by P. J. Park
File create: 2026-10-17 (Sat)
Last update: ---
"""

import openmc
import openmc.deplete
import numpy as np

def make_model(chain_path="/home/patri/openmc/data/chain_endfb81_fast.xml"):

    # --------------------------------
    # Materials (from PNNL-1587, 2021)
    # --------------------------------

    m_uo2 = openmc.Material(name='UO2', material_id=1)
    m_uo2.set_density('g/cm3', 10.96)
    m_uo2.depletable = True
    m_uo2.volume = 1.0 # [cm³]

    m_uo2.add_nuclide('U234', 0.000090)  
    m_uo2.add_nuclide('U235', 0.010124)  
    m_uo2.add_nuclide('U236', 0.000046)
    m_uo2.add_nuclide('U238', 0.323072)  
    m_uo2.add_nuclide('O16',  0.665047)
    m_uo2.add_nuclide('O17',  0.000253)
    m_uo2.add_nuclide('O18',  0.001367)


    m_tho2 = openmc.Material(name='ThO2', material_id=2)
    m_tho2.set_density('g/cm3', 9.70)
    m_tho2.depletable = True
    m_tho2.volume = 1.0 # [cm³]

    m_tho2.add_nuclide('Th232', 0.333333)
    m_tho2.add_nuclide('O16',  0.665047)
    m_tho2.add_nuclide('O17',  0.000253)
    m_tho2.add_nuclide('O18',  0.001367)


    materials = openmc.Materials([m_uo2, m_tho2])


    # --------------------------------
    # Settings - decay-only depletion
    # --------------------------------

    # openmc.deplete.IndependentOperator(materials, fluxes, micros, chain_file=None, keff=None, 
    #                                    normalization_mode='fission-q', fission_q=None, 
    #                                    prev_results=None, reduce_chain_level=None, fission_yield_opts=None)
    #
    # materials (iterable of openmc.Material)
    # fluxes (list of numpy.ndarray) - Flux in each group in [n-cm/src] for each domain
    # micros (list of MicroXS) - Cross-sections in [b] for each domain. If the MicroXS object is empty, a decay-only calculation will be run.
    # chain_file (PathLike or Chain, optional) - Path to the depletion chain XML file or instance of openmc.deplete.Chain. Defaults to openmc.config['chain_file'].
    # normalization_mode ({"fission-q", "source-rate"}) 
    #
    # openmc.deplete.MicroXS(data: ndarray, nuclides: list[str], reactions: list[str])
    #
    # data (numpy.ndarray of floats) - 3D array containing microscopic cross section values for each nuclide, reaction, and energy group. 
    #                                  Cross section values are assumed to be in [b], and indexed by [nuclide, reaction, energy group].
    # nuclides (list of str) - List of nuclide symbols for that have data for at least one reaction.
    # reactions (list of str) - List of reactions. All reactions must match those in openmc.deplete.chain.REACTIONS
    #

    fluxes = [0.0, 0.0]
    micros = [openmc.deplete.MicroXS(np.zeros((0, 0, 0)), [], []) for _ in materials]

    chain    = openmc.deplete.Chain.from_xml(chain_path).reduce(['U234','U235','U236','U238','Th232','O16','O17','O18'])
    operator = openmc.deplete.IndependentOperator(materials, fluxes, micros, chain_file=chain, normalization_mode='source-rate')

    times     = [0] + np.geomspace(1/24, 10*365.25, num=300).tolist()  # [days]
    timesteps = np.diff(times)  # OpenMC expects timestep durations, not cumulative times

    integrator = openmc.deplete.PredictorIntegrator(operator=operator, timesteps=timesteps, timestep_units='d', source_rates=0.0, solver='cram48')

    return materials, integrator