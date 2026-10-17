"""
TEST 001 - Natural UO2/ThO2 Passive Decay Inventory

Measures isotopic evolution of natural UO2, ThO2 over time. Used as inputs for Pile materials,
since for U-238, the presence of Pa234m1 and its passive decay gamma of 1 MeV is important to passively
detecting U-238.

NB1. For passive radioactive decay, the thermal and fast depletion chains will give identical results.
The depletion equation is:

    dN/dt = A*N = (A_decay + A_flux)*N

for vector of nuclide inventories (N), depletion matrix (A [1/sec]), and rate of change dN/dt [atoms/sec].
Because neutron flux is 0, any depletion chain will give the same A = A_decay.

NB2. Because I am new to OpenMC depletion, here's how 'openmc.deplete' works. It splits the depletion 
calculation into three steps:
    
    (1) Operator    (2) Integrator  (3) Depletion solver

(1) Operator determines how reaction rates are obtained. The two choices are
    (a) CoupledOperator     - runs OpenMC transport
    (b) IndependentOperator - you supply neutron flux and effective microscopic cross-sections
For passive decay, phi = 0 so reaction rate R = N*sigma*phi = 0, so we use (b) with all zeros.

(2) Integrator evolve the composition through time. OpenMC v.0.16 has eight integrators and we use the
simplest one, PredictorIntegrator, that uses discrete beginning-of-step reaction rates. This is because
for passive decay, the depletion matrix is constant:

    N(t+dt) = exp(A*dt) * N(t) = N(t+dt) = exp(A_decay*dt) * N(t).

(3) Solver solves the Bateman equation. There's either:
    (a) CRAM16 - 16th-order Chebyshev Rational Approximation Method, faster but less accurate
    (b) CRAM48 - 48th-order Chebyshev Rational Approximation Method
    (c) Custom function
Both CRAMs approximate the matrix exponential exp(A*dt)
    
This file is 100% human-written by P. J. Park
File create: 2026-10-17 (Sat)
Last update: ---
"""

import os, sys, openmc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from test001_model import make_model

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  
from plotstyle import C


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CHAIN_PATH  = "/home/patri/openmc/data/chain_endfb81_fast.xml"
RESULT_PATH = SCRIPT_DIR / "decay_results.h5"

PARENTS = {'UO2': ('U234', 'U235', 'U236', 'U238'), 'ThO2': ('Th232',)}  # Need comma after 'Th232', to register as tuple
MAT_IDS = {'UO2': '1', 'ThO2': '2'}

MIN_PLOT_ABUNDANCE = 1e-16


def main():

    # OpenMC model setup from test001_model.py
    materials, integrator = make_model(chain_path=CHAIN_PATH)

    # Run OpenMC if result h5 does not exist
    if not RESULT_PATH.is_file():
        print(f"{C.Y}Comment. Running OpenMC...{C.X}")
        integrator.integrate(path=RESULT_PATH)

    # openmc.deplete.Results acts as a list that stores the results from each depletion step 
    # and provides extra methods for interrogating these results.
    
    print(f"{C.Y}Comment. Unpacking depletion results...{C.X}")
    results = openmc.deplete.Results(RESULT_PATH)

    datasets = {}

    for m, p in PARENTS.items():

        print(f"{C.Y}Comment. Extracting decay of {p}...{C.X}")
        df = extract_results(results, m, p)

        df_path = SCRIPT_DIR / f"isotopics_{m}.csv"
        print(f"{C.Y}Comment. Saving data to: {df_path}{C.X}")
        df.to_csv(df_path, index=False)

        datasets[m] = df

    # Plot isotopics
    for m, df in datasets.items():

        plot_isotopics(df, m)

    # plt.show()




def extract_results(results, material, parents):
    """ 
    Extract nuclide inventories normalized to the initial total number of U or Th atoms
    
    Returns DataFrame:
        rows: cumulative time
        columns: nuclides
        units: atoms per original U/Th atom
    """
    
    # results[i] is a openmc.deplete.StepResult object
    mat_id = MAT_IDS[material]
    initial = results[0]
    times   = results.get_times(time_units='d')
    atoms   = np.stack([step[mat_id, :] for step in results]) 

    # Initial uranium or thorium inventory
    # (basically summing up the uranium isotopes)
    N0 = sum( initial[mat_id, nuclide] for nuclide in parents )

    # Initialize dataframe
    data = {'time_days': times}

    for nuc, idx in sorted(initial.index_nuc.items()):

        # Ignore oxygen isotopes for now
        if nuc.startswith('O'):
            continue

        abundance = atoms[:, idx] / N0

        if np.any(abundance > 0):
            data[nuc] = abundance

    return pd.DataFrame(data)


def plot_isotopics(df, material_name):

    fig, ax = plt.subplots(figsize=(9, 6))
    times = df["time_days"].to_numpy()
    # Sort nuclides by maximum abundance
    nuclides = sorted(df.columns.drop("time_days"), key=lambda nuc: df[nuc].max(), reverse=True)

    for nuc in nuclides:

        abundance = df[nuc].to_numpy()

        # Exclude insignificant isotopes from plots
        if np.max(abundance) < MIN_PLOT_ABUNDANCE:
            continue

        # Logarithmic axes require positive values
        mask = (times > 0) & (abundance > 0)

        if np.any(mask):
            # ax.plot(times[mask], abundance[mask], label=nuc)

            x, y = times[mask], abundance[mask]
            line, = ax.plot(x, y)
            ax.annotate(nuc, xy=(x[0], y[0]), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", color=line.get_color())

    element = "U" if material_name == "UO2" else "Th"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(f"Atoms per original {element} atom")
    ax.set_title(f"{material_name} natural decay, 1 hr -- 10 yrs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()

    path = SCRIPT_DIR / f"isotopics_{material_name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")

    print(f"Saved: {path}")

    return fig
    

if __name__ == '__main__':
    main()
