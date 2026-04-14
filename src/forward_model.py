"""
Forward Model: Gravity of Rectangular Prisms (Nagy et al., 2000)

Computes the vertical gravitational acceleration (g_z) produced by
rectangular prisms with known dimensions and density contrast.

This is the standard analytical formula used throughout geophysics
for computing gravity from block models.

Reference:
    Nagy, D., Papp, G., and Benedek, J. (2000). The gravitational potential
    and its derivatives for the prism. J. Geod., 74, 552-560.
"""

import numpy as np

# Gravitational constant in SI units (m^3 kg^-1 s^-2)
G = 6.67430e-11

# Conversion: 1 mGal = 1e-5 m/s^2
SI_TO_MGAL = 1e5


def _gz_prism_single(x1, x2, y1, y2, z1, z2, density):
    """
    Compute vertical gravity (g_z) at the origin due to a single
    rectangular prism with constant density.

    The observation point is at the origin (0, 0, 0).
    Prism boundaries are defined in a coordinate system centered
    on the observation point.

    Parameters
    ----------
    x1, x2 : float
        Prism boundaries in x (meters), x1 < x2
    y1, y2 : float
        Prism boundaries in y (meters), y1 < y2
    z1, z2 : float
        Prism boundaries in z (meters), z1 < z2
        (z positive downward in geophysics convention)
    density : float
        Density contrast in kg/m^3

    Returns
    -------
    gz : float
        Vertical gravity in m/s^2 (multiply by SI_TO_MGAL for mGal)

    Notes
    -----
    Formula from Nagy et al. (2000), Eq. for g_z:
        g_z = G * rho * sum_{i,j,k} mu_{ijk} * [
            x * ln(y + r) + y * ln(x + r) - z * arctan(xy / (zr))
        ]
    where r = sqrt(x^2 + y^2 + z^2) and mu_{ijk} = (-1)^(i+j+k)
    """
    result = 0.0
    for i, x in enumerate([x1, x2]):
        for j, y in enumerate([y1, y2]):
            for k, z in enumerate([z1, z2]):
                mu = (-1) ** (i + j + k)
                r = np.sqrt(x**2 + y**2 + z**2)

                # Handle singularities
                if r < 1e-10:
                    continue

                term = 0.0

                # x * ln(y + r)
                yr = y + r
                if abs(yr) > 1e-10:
                    term += x * np.log(yr)

                # y * ln(x + r)
                xr = x + r
                if abs(xr) > 1e-10:
                    term += y * np.log(xr)

                # -z * arctan(xy / (zr))
                if abs(z * r) > 1e-10:
                    term -= z * np.arctan2(x * y, z * r)

                result += mu * term

    return G * density * result


def gz_prism(obs_x, obs_y, obs_z, prism_x1, prism_x2,
             prism_y1, prism_y2, prism_z1, prism_z2, density):
    """
    Compute vertical gravity at observation point(s) due to a single
    rectangular prism with constant density.

    Parameters
    ----------
    obs_x, obs_y, obs_z : float or array
        Observation point coordinates (meters). z positive downward.
        For surface measurements, obs_z = 0.
    prism_x1, prism_x2 : float
        Prism x boundaries (meters)
    prism_y1, prism_y2 : float
        Prism y boundaries (meters)
    prism_z1, prism_z2 : float
        Prism z boundaries (meters), z positive downward
    density : float
        Density contrast (kg/m^3)

    Returns
    -------
    gz : float or array
        Vertical gravity in mGal
    """
    obs_x = np.atleast_1d(np.asarray(obs_x, dtype=float))
    obs_y = np.atleast_1d(np.asarray(obs_y, dtype=float))
    obs_z = np.atleast_1d(np.asarray(obs_z, dtype=float))

    gz = np.zeros_like(obs_x)

    for p in range(len(obs_x)):
        # Shift prism coordinates relative to observation point
        x1 = prism_x1 - obs_x[p]
        x2 = prism_x2 - obs_x[p]
        y1 = prism_y1 - obs_y[p]
        y2 = prism_y2 - obs_y[p]
        z1 = prism_z1 - obs_z[p]
        z2 = prism_z2 - obs_z[p]

        gz[p] = _gz_prism_single(x1, x2, y1, y2, z1, z2, density)

    return gz * SI_TO_MGAL


def gz_prisms_vectorized(obs_x, obs_y, obs_z, prisms, densities):
    """
    Compute total vertical gravity at observation points due to
    multiple rectangular prisms. Vectorized for speed.

    Parameters
    ----------
    obs_x, obs_y, obs_z : array of shape (M,)
        Observation point coordinates (meters)
    prisms : array of shape (N, 6)
        Each row: [x1, x2, y1, y2, z1, z2] in meters
    densities : array of shape (N,)
        Density contrast for each prism (kg/m^3)

    Returns
    -------
    gz_total : array of shape (M,)
        Total vertical gravity at each observation point in mGal
    """
    obs_x = np.atleast_1d(np.asarray(obs_x, dtype=float))
    obs_y = np.atleast_1d(np.asarray(obs_y, dtype=float))
    obs_z = np.atleast_1d(np.asarray(obs_z, dtype=float))
    prisms = np.asarray(prisms, dtype=float)
    densities = np.asarray(densities, dtype=float)

    M = len(obs_x)
    N = len(prisms)
    gz_total = np.zeros(M)

    for p in range(M):
        gz_sum = 0.0
        for n in range(N):
            x1 = prisms[n, 0] - obs_x[p]
            x2 = prisms[n, 1] - obs_x[p]
            y1 = prisms[n, 2] - obs_y[p]
            y2 = prisms[n, 3] - obs_y[p]
            z1 = prisms[n, 4] - obs_z[p]
            z2 = prisms[n, 5] - obs_z[p]

            gz_sum += _gz_prism_single(x1, x2, y1, y2, z1, z2,
                                       densities[n])
        gz_total[p] = gz_sum

    return gz_total * SI_TO_MGAL


def compute_single_block_gravity(obs_x, obs_y, x1, x2, y1, y2,
                                  depth, density_func, n_sublayers=10):
    """
    Compute gravity contribution from a SINGLE block at all observation points.

    Used by the incremental MCMC: instead of recomputing all blocks,
    only recompute the one block whose depth changed.

    Parameters
    ----------
    obs_x, obs_y : array of shape (M,)
        Observation point coordinates (meters)
    x1, x2 : float
        Block x-boundaries (meters)
    y1, y2 : float
        Block y-boundaries (meters)
    depth : float
        Basement depth for this block (meters, positive downward)
    density_func : callable
        density_func(z) -> density contrast (kg/m^3)
    n_sublayers : int
        Number of sublayers for depth-dependent density

    Returns
    -------
    gz : array of shape (M,)
        Gravity contribution from this block at each observation point (mGal)
    """
    obs_x = np.atleast_1d(np.asarray(obs_x, dtype=float))
    obs_y = np.atleast_1d(np.asarray(obs_y, dtype=float))
    M = len(obs_x)

    if depth <= 0:
        return np.zeros(M)

    prism_list = []
    density_list = []

    dz = depth / n_sublayers
    for isub in range(n_sublayers):
        z_top = isub * dz
        z_bot = (isub + 1) * dz
        z_mid = (z_top + z_bot) / 2.0
        rho = density_func(z_mid)
        prism_list.append([x1, x2, y1, y2, z_top, z_bot])
        density_list.append(rho)

    prisms = np.array(prism_list)
    densities = np.array(density_list)
    obs_z = np.zeros(M)

    return gz_prisms_vectorized(obs_x, obs_y, obs_z, prisms, densities)


def compute_gravity_for_basin(obs_x, obs_y, block_x_edges, block_y_edges,
                               depths, density_func, n_sublayers=10):
    """
    Compute gravity at surface stations for a sedimentary basin model.

    The basin is divided into vertical columns (blocks). Each block extends
    from the surface (z=0) to its basement depth. The density contrast
    within each block varies with depth according to density_func.

    To handle depth-dependent density, each block is subdivided into
    thin horizontal sublayers, each with approximately constant density.

    Parameters
    ----------
    obs_x : array of shape (M,)
        x-coordinates of observation stations (meters)
    obs_y : array of shape (M,)
        y-coordinates of observation stations (meters)
    block_x_edges : array of shape (Nx+1,)
        x-edges of blocks (meters)
    block_y_edges : array of shape (Ny+1,)
        y-edges of blocks (meters)
    depths : array of shape (Nx, Ny) or (Nx,) for 2D profile
        Basement depth for each block (meters, positive downward)
    density_func : callable
        Function that takes depth z (meters) and returns density
        contrast (kg/m^3). Example: lambda z: -500 * np.exp(-0.0003 * z)
    n_sublayers : int
        Number of sublayers to approximate depth-dependent density

    Returns
    -------
    gz : array of shape (M,)
        Total gravity at each station in mGal
    """
    obs_x = np.atleast_1d(np.asarray(obs_x, dtype=float))
    obs_y = np.atleast_1d(np.asarray(obs_y, dtype=float))
    obs_z = np.zeros_like(obs_x)  # Surface observation

    depths = np.atleast_2d(depths)
    if depths.ndim == 1:
        depths = depths.reshape(-1, 1)

    Nx = len(block_x_edges) - 1
    Ny = len(block_y_edges) - 1

    # Build list of all prisms and their densities
    prism_list = []
    density_list = []

    for ix in range(Nx):
        x1 = block_x_edges[ix]
        x2 = block_x_edges[ix + 1]
        for iy in range(Ny):
            y1 = block_y_edges[iy]
            y2 = block_y_edges[iy + 1]

            depth = depths[ix, iy]
            if depth <= 0:
                continue

            # Subdivide into sublayers for depth-dependent density
            dz = depth / n_sublayers
            for isub in range(n_sublayers):
                z_top = isub * dz
                z_bot = (isub + 1) * dz
                z_mid = (z_top + z_bot) / 2.0

                # Density at midpoint of sublayer
                rho = density_func(z_mid)

                prism_list.append([x1, x2, y1, y2, z_top, z_bot])
                density_list.append(rho)

    if len(prism_list) == 0:
        return np.zeros(len(obs_x))

    prisms = np.array(prism_list)
    densities = np.array(density_list)

    return gz_prisms_vectorized(obs_x, obs_y, obs_z, prisms, densities)


def compute_gravity_for_basin_fast(obs_x, block_x_edges, block_y_width,
                                    depths, density_func, n_sublayers=10):
    """
    Fast version for 2D profiles. Observation points and blocks are
    along the x-axis. Each block has the same y-width (2.5D approximation).

    Parameters
    ----------
    obs_x : array of shape (M,)
        x-coordinates of observation stations (meters)
    block_x_edges : array of shape (N+1,)
        x-edges of blocks (meters)
    block_y_width : float
        Half-width of blocks in y-direction (meters).
        Blocks extend from -block_y_width to +block_y_width.
    depths : array of shape (N,)
        Basement depth for each block (meters, positive downward)
    density_func : callable
        density_func(z) returns density contrast in kg/m^3
    n_sublayers : int
        Number of sublayers per block

    Returns
    -------
    gz : array of shape (M,)
        Total gravity at each station in mGal
    """
    obs_x = np.atleast_1d(np.asarray(obs_x, dtype=float))
    obs_y = np.zeros_like(obs_x)
    obs_z = np.zeros_like(obs_x)

    N = len(depths)
    y1 = -block_y_width
    y2 = block_y_width

    prism_list = []
    density_list = []

    for i in range(N):
        x1 = block_x_edges[i]
        x2 = block_x_edges[i + 1]
        depth = depths[i]

        if depth <= 0:
            continue

        dz = depth / n_sublayers
        for isub in range(n_sublayers):
            z_top = isub * dz
            z_bot = (isub + 1) * dz
            z_mid = (z_top + z_bot) / 2.0

            rho = density_func(z_mid)
            prism_list.append([x1, x2, y1, y2, z_top, z_bot])
            density_list.append(rho)

    if len(prism_list) == 0:
        return np.zeros(len(obs_x))

    prisms = np.array(prism_list)
    densities = np.array(density_list)

    return gz_prisms_vectorized(obs_x, obs_y, obs_z, prisms, densities)
