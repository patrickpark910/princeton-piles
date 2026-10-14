import numpy as np
from scipy.optimize import fsolve


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


if __name__ == "__main__":
    # TEST TEST TEST
    print("FLiBe density at 900 K:", density_flibe(900))
    print("Teflon density at 294 K:", density_teflon(294))