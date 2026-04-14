"""
Synthetic Basement Model Generator

Creates known true basement depth profiles for testing the inversion.
The true model allows us to validate that the MCMC correctly recovers
the depths and that the uncertainty quantification is well-calibrated.
"""

import numpy as np
from .forward_model import compute_gravity_for_basin_fast, compute_gravity_for_basin
from .utils import make_density_func


def create_synthetic_basin_2d(n_blocks=30, profile_length=100e3,
                               block_y_width=50e3, base_depth=2000.0,
                               seed=42):
    """
    Create a 2D synthetic sedimentary basin profile with realistic
    geological features: a graben (deep trough), a horst (shallow ridge),
    and a regional tilt.

    Parameters
    ----------
    n_blocks : int
        Number of blocks along the profile
    profile_length : float
        Total length of profile in meters
    block_y_width : float
        Half-width of blocks in y-direction (meters) for 2.5D approximation
    base_depth : float
        Background/reference depth in meters
    seed : int
        Random seed for reproducibility

    Returns
    -------
    model : dict with keys:
        'true_depths' : array (N,) - true basement depth at each block (meters)
        'block_x_edges' : array (N+1,) - x-edges of blocks (meters)
        'block_x_centers' : array (N,) - x-centers of blocks (meters)
        'block_y_width' : float - half-width in y (meters)
        'profile_length' : float - total profile length (meters)
        'n_blocks' : int
    """
    np.random.seed(seed)

    dx = profile_length / n_blocks
    block_x_edges = np.linspace(0, profile_length, n_blocks + 1)
    block_x_centers = (block_x_edges[:-1] + block_x_edges[1:]) / 2.0

    x = block_x_centers

    # Build basement depth profile with geological features
    # 1. Base depth
    depths = np.ones(n_blocks) * base_depth

    # 2. Graben (deep trough) - Gaussian depression
    graben_center = 0.35 * profile_length
    graben_width = 0.12 * profile_length
    graben_amplitude = 1500.0  # extra depth in meters
    depths += graben_amplitude * np.exp(
        -(x - graben_center)**2 / (2 * graben_width**2)
    )

    # 3. Horst (shallow ridge) - negative Gaussian
    horst_center = 0.65 * profile_length
    horst_width = 0.08 * profile_length
    horst_amplitude = 800.0
    depths -= horst_amplitude * np.exp(
        -(x - horst_center)**2 / (2 * horst_width**2)
    )

    # 4. Regional tilt
    tilt_slope = 500.0 / profile_length  # 500m over full profile
    depths += tilt_slope * x

    # Ensure minimum depth
    depths = np.maximum(depths, 300.0)

    return {
        'true_depths': depths,
        'block_x_edges': block_x_edges,
        'block_x_centers': block_x_centers,
        'block_y_width': block_y_width,
        'profile_length': profile_length,
        'n_blocks': n_blocks,
    }


def generate_synthetic_gravity(model, density_func, obs_per_block=1,
                                noise_std=0.2, n_sublayers=10, seed=42):
    """
    Compute synthetic gravity data from the true basement model and
    add realistic measurement noise.

    Parameters
    ----------
    model : dict
        From create_synthetic_basin_2d()
    density_func : callable
        density_func(z) -> density contrast in kg/m^3
    obs_per_block : int
        Number of observation stations per block (1 = one per block center)
    noise_std : float
        Standard deviation of Gaussian noise in mGal
    n_sublayers : int
        Number of sublayers for depth-dependent density approximation
    seed : int
        Random seed for noise

    Returns
    -------
    data : dict with keys:
        'obs_x' : array (M,) - station x-coordinates (meters)
        'gravity_true' : array (M,) - noise-free gravity (mGal)
        'gravity_obs' : array (M,) - observed gravity with noise (mGal)
        'noise_std' : float - noise level (mGal)
    """
    np.random.seed(seed)

    # Observation stations at block centers
    if obs_per_block == 1:
        obs_x = model['block_x_centers'].copy()
    else:
        obs_x = np.linspace(
            model['block_x_edges'][0],
            model['block_x_edges'][-1],
            model['n_blocks'] * obs_per_block
        )

    # Compute true gravity (no noise)
    gravity_true = compute_gravity_for_basin_fast(
        obs_x=obs_x,
        block_x_edges=model['block_x_edges'],
        block_y_width=model['block_y_width'],
        depths=model['true_depths'],
        density_func=density_func,
        n_sublayers=n_sublayers,
    )

    # Add measurement noise
    noise = np.random.normal(0, noise_std, size=len(obs_x))
    gravity_obs = gravity_true + noise

    return {
        'obs_x': obs_x,
        'gravity_true': gravity_true,
        'gravity_obs': gravity_obs,
        'noise_std': noise_std,
    }


def create_synthetic_basin_3d(nx_blocks=10, ny_blocks=10,
                               x_length=100e3, y_length=100e3,
                               base_depth=2000.0, seed=42):
    """
    Create a realistic 3D synthetic sedimentary basin (2D grid of blocks).

    Models an intracratonic rift basin (~100x100 km) with:
    - Main depocenter (half-graben) with deep sediment accumulation
    - Secondary sub-basin separated by an intra-basinal high
    - Fault-bounded basement horst/ridge
    - Regional NE-dipping tilt from tectonic subsidence
    - Basin margin shallowing (realistic tapering at edges)

    Parameters
    ----------
    nx_blocks, ny_blocks : int
        Number of blocks in x and y directions
    x_length, y_length : float
        Total domain extent in meters
    base_depth : float
        Background/reference basement depth in meters
    seed : int
        Random seed for reproducibility

    Returns
    -------
    model : dict with keys:
        'true_depths' : array (Nx, Ny) - true basement depth at each block
        'block_x_edges' : array (Nx+1,)
        'block_y_edges' : array (Ny+1,)
        'block_x_centers' : array (Nx,)
        'block_y_centers' : array (Ny,)
        'x_length', 'y_length' : float
        'nx_blocks', 'ny_blocks' : int
    """
    np.random.seed(seed)

    block_x_edges = np.linspace(0, x_length, nx_blocks + 1)
    block_y_edges = np.linspace(0, y_length, ny_blocks + 1)
    block_x_centers = (block_x_edges[:-1] + block_x_edges[1:]) / 2.0
    block_y_centers = (block_y_edges[:-1] + block_y_edges[1:]) / 2.0

    # 2D coordinate grids (indexing='ij': first index=x, second=y)
    X, Y = np.meshgrid(block_x_centers, block_y_centers, indexing='ij')

    depths = np.ones((nx_blocks, ny_blocks)) * base_depth

    # 1. Main depocenter (half-graben) — deepest part of basin
    #    Elongated N-S, located at ~35% x, ~50% y
    cx1, cy1 = 0.35 * x_length, 0.50 * y_length
    sx1, sy1 = 0.12 * x_length, 0.18 * y_length
    depths += 2000.0 * np.exp(-((X - cx1)**2 / (2 * sx1**2) +
                                 (Y - cy1)**2 / (2 * sy1**2)))

    # 2. Secondary depocenter — smaller sub-basin
    cx2, cy2 = 0.72 * x_length, 0.30 * y_length
    sx2, sy2 = 0.10 * x_length, 0.12 * y_length
    depths += 1000.0 * np.exp(-((X - cx2)**2 / (2 * sx2**2) +
                                 (Y - cy2)**2 / (2 * sy2**2)))

    # 3. Basement horst/ridge — shallow area between depocenters
    cx3, cy3 = 0.55 * x_length, 0.50 * y_length
    sx3, sy3 = 0.08 * x_length, 0.08 * y_length
    depths -= 800.0 * np.exp(-((X - cx3)**2 / (2 * sx3**2) +
                                (Y - cy3)**2 / (2 * sy3**2)))

    # 4. Regional NE tilt (gentle monocline)
    depths += (300.0 / x_length) * X + (200.0 / y_length) * Y

    # 5. Basin margin shallowing — smooth taper at edges
    #    Use a product of edge tapers so the basin is deeper in center
    margin_width_x = 0.15 * x_length
    margin_width_y = 0.15 * y_length
    # Taper from 0 at edge to 1 at margin_width distance from edge
    taper_x = np.clip(np.minimum(X, x_length - X) / margin_width_x, 0, 1)
    taper_y = np.clip(np.minimum(Y, y_length - Y) / margin_width_y, 0, 1)
    taper = taper_x * taper_y
    # Apply taper: reduce extra depth (above base) at margins
    extra_depth = depths - base_depth * 0.5  # keep minimum ~1000m at margins
    depths = base_depth * 0.5 + extra_depth * taper

    # Ensure minimum depth
    depths = np.maximum(depths, 300.0)

    return {
        'true_depths': depths,
        'block_x_edges': block_x_edges,
        'block_y_edges': block_y_edges,
        'block_x_centers': block_x_centers,
        'block_y_centers': block_y_centers,
        'x_length': x_length,
        'y_length': y_length,
        'nx_blocks': nx_blocks,
        'ny_blocks': ny_blocks,
    }


def generate_synthetic_gravity_3d(model, density_func, noise_std=0.3,
                                    n_sublayers=10, seed=42):
    """
    Compute synthetic gravity data from a 3D basin model and add noise.

    Parameters
    ----------
    model : dict
        From create_synthetic_basin_3d()
    density_func : callable
        density_func(z) -> density contrast in kg/m^3
    noise_std : float
        Standard deviation of Gaussian noise in mGal
    n_sublayers : int
        Number of sublayers for depth-dependent density
    seed : int
        Random seed for noise

    Returns
    -------
    data : dict with keys:
        'obs_x' : array (M,) - flattened x-coordinates
        'obs_y' : array (M,) - flattened y-coordinates
        'obs_x_grid' : array (Nx, Ny) - 2D grid x-coordinates
        'obs_y_grid' : array (Nx, Ny) - 2D grid y-coordinates
        'gravity_true' : array (M,) - noise-free gravity
        'gravity_obs' : array (M,) - observed gravity with noise
        'noise_std' : float
    """
    np.random.seed(seed)

    # Observation points at block centers (2D grid)
    obs_x_grid, obs_y_grid = np.meshgrid(
        model['block_x_centers'], model['block_y_centers'], indexing='ij'
    )
    obs_x = obs_x_grid.ravel()
    obs_y = obs_y_grid.ravel()

    # Compute true gravity using full 3D forward model
    gravity_true = compute_gravity_for_basin(
        obs_x=obs_x,
        obs_y=obs_y,
        block_x_edges=model['block_x_edges'],
        block_y_edges=model['block_y_edges'],
        depths=model['true_depths'],
        density_func=density_func,
        n_sublayers=n_sublayers,
    )

    # Add measurement noise
    noise = np.random.normal(0, noise_std, size=len(obs_x))
    gravity_obs = gravity_true + noise

    return {
        'obs_x': obs_x,
        'obs_y': obs_y,
        'obs_x_grid': obs_x_grid,
        'obs_y_grid': obs_y_grid,
        'gravity_true': gravity_true,
        'gravity_obs': gravity_obs,
        'noise_std': noise_std,
    }
