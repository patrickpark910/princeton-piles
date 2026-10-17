import os, sys
import numpy as np
from scipy.optimize import fsolve

from python.parameters import *


def density_flibe(temp_k, pressure_kpa=101.325):
    """Calculate FLiBe density from Humrickhouse (2017), INL-44148.

    Args:
        temp_k: temperature [K], between 788 and 1093 K
        pressure_kpa: pressure [kPa], default 101.325 kPa (1 atm)

    Returns:
        density [g/cm³], as a float or NumPy array

    Raises:
        ValueError: if temperature is outside 788--1093 K
        RuntimeError: if the EOS solver fails to converge

    Notes:
        Uses the liquid-phase soft-sphere Helmholtz EOS.
        Temperature bounds reflect the approximate range of
        experimental density data, not formal EOS validity limits.
    """

    # Ensure temp range is within experimentally-validated bounds 
    temps = np.asarray(temp_k, dtype=float)
    if np.any((temps < 788) | (temps > 1093)):
        raise ValueError("FLiBe temp must be in: [788, 1093] K.")

    # Equation-of-state constants
    R_s = 84.078       # J/(kg K)
    T_m = 732.3        # K
    rho_m = 1991.6     # kg/m³
    pressure_pa = float(pressure_kpa) * 1000.0

    n = np.array([0.9307, 36.62, 45.17, 32.29, -104.1])
    t = np.array([0.9599, 0.9663, 1.014, 0.4468, 0.9629])
    d = np.array([0, 0, 3.151, 1.924, 1.998])

    def solve(T):
        def residual(rho):
            term = np.sum(n * d * (T_m / T)**t * (rho / rho_m)**d)
            return rho * R_s * T * (1.0 + term) - pressure_pa

        rho, _, ier, msg = fsolve(residual, x0=1900.0, full_output=True)
        
        if ier != 1: 
            raise RuntimeError(f"Density solver failed at {T} K: {msg}")

        return rho[0] / 1000.0  # kg/m³ -> g/cm³

    densities = np.array([solve(T) for T in temps.flat]).reshape(temps.shape)

    return densities.item() if densities.ndim == 0 else densities


def density_teflon(temp_k, rho_25=2.162):
    """Calculate Teflon (PTFE) density from Kirby (1956).

    Args:
        temp_k: temperature [K], between 83.15 and 573.15 K
        rho_25: reference density [g/cm³] at 25°C / 298.15 K, default 2.162 g/cm³ (Kirby annealed sample A).

    Returns:
        Density [g/cm³], as a float or NumPy array.

    Raises:
        ValueError: If temperature is outside 83.15--573.15 K.

    Notes:
        Uses mean linear thermal expansion coefficients from Kirby Tb.2, at 1 atm.
        Assumes isotropic expansion and constant mass.
        Interpolates relative length, not expansion coefficients.

    References:
        R. K. Kirby (1956), J Research of NBS, 57 (2), 91--94, doi:10.6028/jres.057.010
    """
    # Ensure temperature is within the experimental range.
    temps = np.asarray(temp_k, dtype=float)

    if np.any((temps < 83.15) | (temps > 573.15)):
        raise ValueError("Teflon temp must be in: [83.15, 573.15] K.")

    if not np.isfinite(rho_25) or rho_25 <= 0:
        raise ValueError("Reference density must be positive and finite.")

    # Kirby (1956), Table 2: mean linear expansion relative to 25°C
    temp_c = np.array([-190, -150, -100, -50, 0, 25, 50, 100, 150, 200, 250, 300])
    alpha = np.array([86, 96, 112, 135, 200, 0, 124, 124, 135, 151, 174, 218]) * 1e-6  # 1/K; value at 25°C is an unused placeholder.

    # Relative linear dimensions and volumetric expansion
    length_ratio = 1.0 + alpha * (temp_c - 25.0)
    expansion = np.interp(temps - 273.15, temp_c, length_ratio)

    densities = rho_25 / expansion**3

    return densities.item() if densities.ndim == 0 else densities


def calc_fertile_mass_in_triso(fertile_isotope):
    """Return fertile isotope masses [g] per TRISO particle."""

    kernel_volume = (4.0 / 3.0) * np.pi * TRISO_RADII[0]**3  # [cm³]

    if fertile_isotope == "U238":
        return {"U238": (kernel_volume * DENSITY['UO2'] * AMU['U'] / (AMU['U'] + 2.0 * AMU['O']) * (1-U_ENRICH))}
    elif fertile_isotope == "Th232":
        return {"Th232": (kernel_volume * DENSITY['ThO2'] * AMU['Th232'] / (AMU['Th232'] + 2.0 * AMU['O'])),}
    else:
        raise ValueError("Invalid fertile isotope. Choose 'U238' or 'Th232'.")


def lattice_coords(lower_left, shape, pitch):
    """
    Calculates the center coordinates of all cells in a rectangular lattice
    and returns them as a list of tuples.
    
    Args:
        lower_left (iterable of float): (x, y, z) coordinates of the lower-left corner.
        shape (iterable of int): number of cells (Nx, Ny, Nz).
        pitch (iterable of float): width, height, and depth of each cell (Dx, Dy, Dz).
        
    Returns: 
        coords (list of tuple): flat list containing (x, y, z) center coordinates.
    """
    nx, ny, nz = shape
    dx, dy, dz = pitch
    lx, ly, lz = lower_left

    coords = []
    
    for i in range(nx):
        x = lx + (i + 0.5) * dx
        for j in range(ny):
            y = ly + (j + 0.5) * dy
            for k in range(nz):
                z = lz + (k + 0.5) * dz
                coords.append((x, y, z))
                
    return coords


def set_xs_path():
    """
    Temporary solution for finding xs files between WSL and Ubuntu on Computing Cluster without editing PATH --ppark 2025-06-28
    """
    xs_path_wsl = '/home/patri/openmc/data/endf-b8.1-hdf5/cross_sections.xml'
    xs_path = None # os.environ.get("OPENMC_CROSS_SECTIONS")
    
    if xs_path is None:
        # print(f"{C.YELLOW}Warning.{C.END} 'OPENMC_CROSS_SECTIONS' is NOT set in PATH.")
        print(os.path.isfile(xs_path_wsl))
        if os.path.isfile(xs_path_wsl):
            print(f"{C.YELLOW}Warning.{C.END} 'OPENMC_CROSS_SECTIONS' manually set to {xs_path_wsl}.")
            return xs_path_wsl
        
        else:
            sys.exit(f"{C.RED}Fatal. Cannot find cross section XML!{C.END}")
    
    else:
        print(f"{C.GREEN}'OPENMC_CROSS_SECTIONS' found!{C.END}")
        return xs_path


def has_statepoint(directory_path,cycle=None):
    """
    Check if any file starting with 'statepoint' exists in the given directory.
    
    Args:
        directory_path (str): Path to the directory to search
    
    Returns:
        bool: True if a file starting with 'statepoint' is found, False otherwise
    """
    found = False
    for filename in os.listdir(directory_path):
        if cycle:
            cycle_str = str(cycle).zfill(2)  # Pad cycle number with zeros to get 5 -> '05'
            if filename.startswith(f"statepoint.{cycle_str}"):
                found = True
        else:
            if filename.startswith("statepoint"):
                found = True
    return found


if __name__ == "__main__":
    # TEST TEST TEST
    print(calc_fertile_mass_in_triso("U238"))
    print(calc_fertile_mass_in_triso("Th232"))
    print(AMU['U'])