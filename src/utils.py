"""
Utility functions: density-depth relationships and helpers.

Density-depth functions model how the density contrast between
sediment and basement changes with depth due to compaction.

References:
    Athy, L.F. (1930). Density, porosity and compaction of sedimentary rocks.
    Cordell, L. (1973). Gravity analysis using an exponential density-depth function.
    Litinsky, V.A. (1989). Concept of effective density: key to gravity depth determinations.
    Chakravarthi, V. et al. (2002). Parabolic density function in sedimentary basin modelling.
"""

import numpy as np


def exponential_density(z, drho_0=-500.0, lam=0.0003):
    """
    Exponential compaction density-depth function.

    Δρ(z) = Δρ₀ · exp(-λz)

    As depth increases, density contrast decreases (sediment becomes
    denser due to compaction, approaching basement density).

    Parameters
    ----------
    z : float or array
        Depth in meters (positive downward)
    drho_0 : float
        Surface density contrast in kg/m^3 (negative: sediment lighter than basement)
        Typical: -400 to -600 kg/m^3
    lam : float
        Compaction parameter in 1/meters
        Typical: 0.0001 to 0.001

    Returns
    -------
    drho : float or array
        Density contrast at depth z in kg/m^3
    """
    return drho_0 * np.exp(-lam * z)


def constant_density(z, drho_0=-500.0):
    """
    Constant density contrast (no compaction).

    Parameters
    ----------
    z : float or array
        Depth in meters (not used, included for consistent interface)
    drho_0 : float
        Density contrast in kg/m^3

    Returns
    -------
    drho : float or array
        Constant density contrast
    """
    return drho_0 * np.ones_like(np.atleast_1d(np.asarray(z, dtype=float)))


def hyperbolic_density(z, drho_0=-500.0, beta=0.0003):
    """
    Hyperbolic density-depth function (Litinsky, 1989).

    Δρ(z) = Δρ₀ / (1 + βz)

    Parameters
    ----------
    z : float or array
        Depth in meters
    drho_0 : float
        Surface density contrast in kg/m^3
    beta : float
        Compaction parameter in 1/meters
    """
    return drho_0 / (1.0 + beta * np.atleast_1d(np.asarray(z, dtype=float)))


def parabolic_density(z, drho_0=-500.0, alpha=1e-7):
    """
    Parabolic density-depth function (Chakravarthi et al., 2002).

    Δρ(z) = Δρ₀ · (1 - αz²)

    Parameters
    ----------
    z : float or array
        Depth in meters
    drho_0 : float
        Surface density contrast in kg/m^3
    alpha : float
        Compaction parameter in 1/meters^2
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return drho_0 * (1.0 - alpha * z**2)


def make_density_func(func_type='exponential', **kwargs):
    """
    Factory function to create a density-depth function.

    Parameters
    ----------
    func_type : str
        One of 'exponential', 'constant', 'hyperbolic', 'parabolic'
    **kwargs : dict
        Parameters for the chosen function

    Returns
    -------
    func : callable
        density_func(z) -> density contrast in kg/m^3
    """
    funcs = {
        'exponential': exponential_density,
        'constant': constant_density,
        'hyperbolic': hyperbolic_density,
        'parabolic': parabolic_density,
    }
    if func_type not in funcs:
        raise ValueError(f"Unknown density function: {func_type}. "
                         f"Choose from {list(funcs.keys())}")
    base_func = funcs[func_type]
    return lambda z: base_func(z, **kwargs)
