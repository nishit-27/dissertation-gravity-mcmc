"""
Bayesian MCMC Inversion for Basement Depth

Implements the Metropolis-Hastings algorithm to sample the posterior
distribution of basement depths given observed gravity data.

The posterior is: P(m|d) ∝ P(d|m) × P(m)
- P(d|m) = likelihood = exp(-S(m)/σ²)  where S = ½Σ(g_obs - g_calc)²
- P(m) = prior = uniform within bounds × smoothness penalty

The MCMC chain produces an ensemble of plausible depth models.
The spread of this ensemble IS the uncertainty.

References:
    Metropolis, N. et al. (1953). Equation of state calculations.
    Hastings, W.K. (1970). Monte Carlo sampling methods using Markov chains.
    Field et al. (2026). GravMCMC. Geosci. Model Dev.
    Rossi, L. (2017). Bayesian gravity inversion. PhD thesis, Politecnico di Milano.
"""

import time
import numpy as np
from .forward_model import (compute_gravity_for_basin_fast,
                             compute_gravity_for_basin,
                             compute_single_block_gravity)


def compute_misfit(gravity_obs, gravity_calc):
    """Sum of squared residuals divided by 2."""
    return 0.5 * np.sum((gravity_obs - gravity_calc) ** 2)


def run_mcmc(obs_x, gravity_obs, block_x_edges, block_y_width,
             density_func, noise_std,
             n_iterations=50000, step_size=100.0,
             depth_min=200.0, depth_max=6000.0,
             smoothness_weight=0.0, n_sublayers=10,
             initial_depths=None, seed=42, verbose=True):
    """
    Run Metropolis-Hastings MCMC inversion for basement depth.

    At each iteration:
    1. Randomly select one block
    2. Propose a new depth = current + Normal(0, step_size)
    3. Compute forward gravity for the proposed model
    4. Accept/reject based on Metropolis criterion

    Parameters
    ----------
    obs_x : array (M,)
        Observation station x-coordinates (meters)
    gravity_obs : array (M,)
        Observed gravity data (mGal)
    block_x_edges : array (N+1,)
        Block x-boundaries (meters)
    block_y_width : float
        Half-width in y-direction (meters)
    density_func : callable
        density_func(z) -> density contrast (kg/m^3)
    noise_std : float
        Estimated data noise level (mGal)
    n_iterations : int
        Total number of MCMC iterations
    step_size : float
        Standard deviation of proposal perturbation (meters)
    depth_min, depth_max : float
        Prior bounds on depth (meters)
    smoothness_weight : float
        Weight for smoothness penalty (0 = no smoothness constraint)
    n_sublayers : int
        Sublayers for depth-dependent density
    initial_depths : array (N,) or None
        Initial depth model. If None, uses uniform mid-range.
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    result : dict with keys:
        'chain' : array (n_accepted, N) - all accepted depth models
        'misfit_chain' : array (n_accepted,) - misfit for each accepted model
        'acceptance_rate' : float
        'n_iterations' : int
        'all_misfits' : array (n_iterations,) - misfit at each iteration
    """
    np.random.seed(seed)

    N = len(block_x_edges) - 1  # number of blocks
    sigma2 = noise_std ** 2

    # Initialize depth model
    if initial_depths is not None:
        current_depths = np.array(initial_depths, dtype=float).copy()
    else:
        current_depths = np.ones(N) * (depth_min + depth_max) / 2.0

    # Compute initial gravity and misfit
    current_gravity = compute_gravity_for_basin_fast(
        obs_x, block_x_edges, block_y_width,
        current_depths, density_func, n_sublayers
    )
    current_misfit = compute_misfit(gravity_obs, current_gravity)

    # Add smoothness penalty
    if smoothness_weight > 0:
        smooth_penalty = smoothness_weight * np.sum(
            np.diff(current_depths) ** 2
        )
        current_energy = current_misfit / sigma2 + smooth_penalty
    else:
        current_energy = current_misfit / sigma2

    # Storage
    chain = [current_depths.copy()]
    misfit_chain = [current_misfit]
    all_misfits = np.zeros(n_iterations)
    n_accepted = 0

    if verbose:
        print(f"Starting MCMC: {n_iterations} iterations, {N} blocks")
        print(f"Initial misfit: {current_misfit:.4f}")
        print(f"Step size: {step_size:.1f} m, Noise std: {noise_std:.3f} mGal")
        print("-" * 50)

    for it in range(n_iterations):
        # 1. Randomly select a block to perturb
        idx = np.random.randint(0, N)

        # 2. Propose new depth
        proposed_depths = current_depths.copy()
        perturbation = np.random.normal(0, step_size)
        proposed_depths[idx] += perturbation

        # 3. Check prior bounds
        if proposed_depths[idx] < depth_min or proposed_depths[idx] > depth_max:
            all_misfits[it] = current_misfit
            continue  # Reject — outside prior bounds

        # 4. Compute forward gravity for proposed model
        proposed_gravity = compute_gravity_for_basin_fast(
            obs_x, block_x_edges, block_y_width,
            proposed_depths, density_func, n_sublayers
        )

        # 5. Compute proposed misfit and energy
        proposed_misfit = compute_misfit(gravity_obs, proposed_gravity)

        if smoothness_weight > 0:
            smooth_penalty = smoothness_weight * np.sum(
                np.diff(proposed_depths) ** 2
            )
            proposed_energy = proposed_misfit / sigma2 + smooth_penalty
        else:
            proposed_energy = proposed_misfit / sigma2

        # 6. Metropolis acceptance criterion
        delta_energy = proposed_energy - current_energy
        if delta_energy < 0:
            # Better model — always accept
            accept = True
        else:
            # Worse model — accept with probability exp(-delta)
            accept = np.random.uniform() < np.exp(-delta_energy)

        # 7. Update
        if accept:
            current_depths = proposed_depths
            current_gravity = proposed_gravity
            current_misfit = proposed_misfit
            current_energy = proposed_energy
            n_accepted += 1
            chain.append(current_depths.copy())
            misfit_chain.append(current_misfit)

        all_misfits[it] = current_misfit

        # Progress reporting
        if verbose and (it + 1) % (n_iterations // 10) == 0:
            acc_rate = n_accepted / (it + 1) * 100
            print(f"  Iteration {it+1:6d}/{n_iterations} | "
                  f"Misfit: {current_misfit:10.4f} | "
                  f"Accept rate: {acc_rate:5.1f}%")

    acceptance_rate = n_accepted / n_iterations
    if verbose:
        print("-" * 50)
        print(f"Done. Acceptance rate: {acceptance_rate*100:.1f}%")
        print(f"Final misfit: {current_misfit:.4f}")
        print(f"Total accepted models: {len(chain)}")

    return {
        'chain': np.array(chain),
        'misfit_chain': np.array(misfit_chain),
        'acceptance_rate': acceptance_rate,
        'n_iterations': n_iterations,
        'all_misfits': all_misfits,
    }


def run_mcmc_joint(obs_x, gravity_obs, block_x_edges, block_y_width,
                   drho_0, noise_std,
                   n_iterations=50000,
                   step_depth=150.0, step_lambda=0.00005,
                   depth_min=200.0, depth_max=6000.0,
                   lambda_min=0.00001, lambda_max=0.005,
                   lambda_init=0.0005,
                   prob_perturb_lambda=0.2,
                   smoothness_weight=0.0, n_sublayers=10,
                   initial_depths=None, borehole_constraints=None,
                   seed=42, verbose=True):
    """
    Joint Bayesian MCMC inversion for basement depth AND lambda
    (compaction parameter in exponential density function).

    At each iteration, randomly choose to perturb either:
      - One block's depth (probability = 1 - prob_perturb_lambda)
      - Lambda for all blocks (probability = prob_perturb_lambda)

    The density function is: drho(z) = drho_0 * exp(-lambda * z)
    where drho_0 is fixed and lambda is estimated by MCMC.

    Parameters
    ----------
    obs_x : array (M,)
        Observation station x-coordinates (meters)
    gravity_obs : array (M,)
        Observed gravity data (mGal)
    block_x_edges : array (N+1,)
        Block x-boundaries (meters)
    block_y_width : float
        Half-width in y-direction (meters)
    drho_0 : float
        Surface density contrast (kg/m^3), fixed (e.g., -500)
    noise_std : float
        Estimated data noise level (mGal)
    n_iterations : int
        Total MCMC iterations
    step_depth : float
        Std of depth proposal perturbation (meters)
    step_lambda : float
        Std of lambda proposal perturbation (1/m)
    depth_min, depth_max : float
        Prior bounds on depth (meters)
    lambda_min, lambda_max : float
        Prior bounds on lambda (1/m)
    lambda_init : float
        Initial guess for lambda (1/m)
    prob_perturb_lambda : float
        Probability of choosing to perturb lambda at each iteration (0 to 1)
    smoothness_weight : float
        Weight for smoothness penalty
    n_sublayers : int
        Sublayers for depth-dependent density
    initial_depths : array (N,) or None
        Initial depth model
    borehole_constraints : dict or None
        Dictionary mapping block index (0-based) to known depth (meters).
        Example: {1: 2449, 6: 1591} means block 1 is locked at 2449m
        and block 6 is locked at 1591m. These blocks are never perturbed.
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    result : dict with keys:
        'chain' : array (K, N) - accepted depth models
        'lambda_chain' : array (K,) - accepted lambda values
        'misfit_chain' : array (K,) - misfit for each accepted model
        'acceptance_rate' : float
        'depth_acceptance_rate' : float
        'lambda_acceptance_rate' : float
        'n_iterations' : int
        'all_misfits' : array (n_iterations,)
        'all_lambdas' : array (n_iterations,)
    """
    np.random.seed(seed)

    N = len(block_x_edges) - 1
    sigma2 = noise_std ** 2

    # Handle borehole constraints
    if borehole_constraints is None:
        borehole_constraints = {}
    borehole_set = set(borehole_constraints.keys())

    # Initialize depths
    if initial_depths is not None:
        current_depths = np.array(initial_depths, dtype=float).copy()
    else:
        current_depths = np.ones(N) * (depth_min + depth_max) / 2.0

    # Lock borehole depths to known values
    for idx, depth in borehole_constraints.items():
        current_depths[idx] = depth

    # Free blocks (not constrained by boreholes)
    free_blocks = [i for i in range(N) if i not in borehole_set]

    # Initialize lambda
    current_lambda = lambda_init

    # Build density function with current lambda
    def make_density(lam):
        return lambda z: drho_0 * np.exp(-lam * z)

    current_density_func = make_density(current_lambda)

    # Compute initial gravity and misfit
    current_gravity = compute_gravity_for_basin_fast(
        obs_x, block_x_edges, block_y_width,
        current_depths, current_density_func, n_sublayers
    )
    current_misfit = compute_misfit(gravity_obs, current_gravity)

    if smoothness_weight > 0:
        current_energy = current_misfit / sigma2 + smoothness_weight * np.sum(
            np.diff(current_depths) ** 2)
    else:
        current_energy = current_misfit / sigma2

    # Storage
    chain = [current_depths.copy()]
    lambda_chain = [current_lambda]
    misfit_chain = [current_misfit]
    all_misfits = np.zeros(n_iterations)
    all_lambdas = np.zeros(n_iterations)
    all_lambdas[0] = current_lambda

    n_accepted = 0
    n_depth_proposed = 0
    n_depth_accepted = 0
    n_lambda_proposed = 0
    n_lambda_accepted = 0

    if verbose:
        print(f"Starting Joint MCMC: {n_iterations} iterations, {N} blocks")
        print(f"Estimating: {len(free_blocks)} free depths + 1 lambda")
        if borehole_constraints:
            print(f"Borehole constraints: {len(borehole_constraints)} blocks locked")
            for idx, depth in sorted(borehole_constraints.items()):
                print(f"  Block {idx+1} (index {idx}): locked at {depth:.0f} m")
        print(f"Initial misfit: {current_misfit:.4f}")
        print(f"Initial lambda: {current_lambda:.6f}")
        print(f"Lambda perturbation probability: {prob_perturb_lambda*100:.0f}%")
        print(f"Step sizes: depth={step_depth:.0f}m, lambda={step_lambda:.6f}")
        print("-" * 60)

    progress_every = max(500, n_iterations // 50)
    t_start = time.time()

    for it in range(n_iterations):
        # Decide: perturb depth or lambda?
        perturb_lambda = np.random.uniform() < prob_perturb_lambda

        if perturb_lambda:
            # --- PERTURB LAMBDA ---
            n_lambda_proposed += 1

            proposed_lambda = current_lambda + np.random.normal(0, step_lambda)

            # Check prior bounds
            if proposed_lambda < lambda_min or proposed_lambda > lambda_max:
                all_misfits[it] = current_misfit
                all_lambdas[it] = current_lambda
                continue

            # Recompute gravity with new lambda (all blocks affected)
            proposed_density_func = make_density(proposed_lambda)
            proposed_gravity = compute_gravity_for_basin_fast(
                obs_x, block_x_edges, block_y_width,
                current_depths, proposed_density_func, n_sublayers
            )
            proposed_misfit = compute_misfit(gravity_obs, proposed_gravity)

            if smoothness_weight > 0:
                proposed_energy = proposed_misfit / sigma2 + smoothness_weight * np.sum(
                    np.diff(current_depths) ** 2)
            else:
                proposed_energy = proposed_misfit / sigma2

            # Metropolis criterion
            delta_energy = proposed_energy - current_energy
            if delta_energy < 0:
                accept = True
            else:
                accept = np.random.uniform() < np.exp(-delta_energy)

            if accept:
                current_lambda = proposed_lambda
                current_density_func = proposed_density_func
                current_gravity = proposed_gravity
                current_misfit = proposed_misfit
                current_energy = proposed_energy
                n_lambda_accepted += 1
                n_accepted += 1
                chain.append(current_depths.copy())
                lambda_chain.append(current_lambda)
                misfit_chain.append(current_misfit)

        else:
            # --- PERTURB ONE DEPTH ---
            n_depth_proposed += 1

            # Only pick from free blocks (not boreholes)
            idx = free_blocks[np.random.randint(0, len(free_blocks))]
            proposed_depths = current_depths.copy()
            proposed_depths[idx] += np.random.normal(0, step_depth)

            # Check prior bounds
            if proposed_depths[idx] < depth_min or proposed_depths[idx] > depth_max:
                all_misfits[it] = current_misfit
                all_lambdas[it] = current_lambda
                continue

            # Compute gravity (lambda unchanged)
            proposed_gravity = compute_gravity_for_basin_fast(
                obs_x, block_x_edges, block_y_width,
                proposed_depths, current_density_func, n_sublayers
            )
            proposed_misfit = compute_misfit(gravity_obs, proposed_gravity)

            if smoothness_weight > 0:
                proposed_energy = proposed_misfit / sigma2 + smoothness_weight * np.sum(
                    np.diff(proposed_depths) ** 2)
            else:
                proposed_energy = proposed_misfit / sigma2

            # Metropolis criterion
            delta_energy = proposed_energy - current_energy
            if delta_energy < 0:
                accept = True
            else:
                accept = np.random.uniform() < np.exp(-delta_energy)

            if accept:
                current_depths = proposed_depths
                current_gravity = proposed_gravity
                current_misfit = proposed_misfit
                current_energy = proposed_energy
                n_depth_accepted += 1
                n_accepted += 1
                chain.append(current_depths.copy())
                lambda_chain.append(current_lambda)
                misfit_chain.append(current_misfit)

        all_misfits[it] = current_misfit
        all_lambdas[it] = current_lambda

        # Progress reporting
        if verbose and (it + 1) % progress_every == 0:
            elapsed = time.time() - t_start
            frac = (it + 1) / n_iterations
            eta = elapsed * (1.0 - frac) / max(frac, 1e-9)
            rate = (it + 1) / max(elapsed, 1e-9)
            acc_rate = n_accepted / (it + 1) * 100
            print(f"  [{frac*100:5.1f}%] iter {it+1:>7d}/{n_iterations} | "
                  f"misfit {current_misfit:8.2f} | "
                  f"λ {current_lambda:.2e} | "
                  f"accept {acc_rate:5.1f}% | "
                  f"{rate:6.0f} it/s | "
                  f"elapsed {elapsed/60:5.1f} min | "
                  f"eta {eta/60:5.1f} min",
                  flush=True)

    # Final statistics
    acceptance_rate = n_accepted / n_iterations
    depth_acc = n_depth_accepted / max(n_depth_proposed, 1)
    lambda_acc = n_lambda_accepted / max(n_lambda_proposed, 1)

    if verbose:
        print("-" * 60)
        total_min = (time.time() - t_start) / 60
        print(f"Done in {total_min:.1f} min. Overall acceptance: {acceptance_rate*100:.1f}%")
        print(f"  Depth acceptance:  {depth_acc*100:.1f}% ({n_depth_accepted}/{n_depth_proposed})")
        print(f"  Lambda acceptance: {lambda_acc*100:.1f}% ({n_lambda_accepted}/{n_lambda_proposed})")
        print(f"Final misfit: {current_misfit:.4f}")
        print(f"Final lambda: {current_lambda:.6f}")
        print(f"Total accepted models: {len(chain)}")

    return {
        'chain': np.array(chain),
        'lambda_chain': np.array(lambda_chain),
        'misfit_chain': np.array(misfit_chain),
        'acceptance_rate': acceptance_rate,
        'depth_acceptance_rate': depth_acc,
        'lambda_acceptance_rate': lambda_acc,
        'n_iterations': n_iterations,
        'all_misfits': all_misfits,
        'all_lambdas': all_lambdas,
    }


def process_joint_chain(result, burn_in_frac=0.5, thin=1):
    """
    Post-process the joint MCMC chain (depths + lambda).

    Returns
    -------
    posterior : dict with keys for depths AND lambda statistics
    """
    chain = result['chain']
    lambda_chain = result['lambda_chain']
    n_total = len(chain)
    burn_in = int(n_total * burn_in_frac)

    depth_samples = chain[burn_in::thin]
    lambda_samples = lambda_chain[burn_in::thin]

    return {
        # Depth statistics
        'samples': depth_samples,
        'mean': np.mean(depth_samples, axis=0),
        'std': np.std(depth_samples, axis=0),
        'ci_5': np.percentile(depth_samples, 5, axis=0),
        'ci_95': np.percentile(depth_samples, 95, axis=0),
        'ci_2_5': np.percentile(depth_samples, 2.5, axis=0),
        'ci_97_5': np.percentile(depth_samples, 97.5, axis=0),
        'n_samples': len(depth_samples),
        # Lambda statistics
        'lambda_samples': lambda_samples,
        'lambda_mean': np.mean(lambda_samples),
        'lambda_std': np.std(lambda_samples),
        'lambda_ci_5': np.percentile(lambda_samples, 5),
        'lambda_ci_95': np.percentile(lambda_samples, 95),
        'lambda_ci_2_5': np.percentile(lambda_samples, 2.5),
        'lambda_ci_97_5': np.percentile(lambda_samples, 97.5),
    }


def process_chain(result, burn_in_frac=0.5, thin=1):
    """
    Post-process the MCMC chain: remove burn-in and thin.

    Parameters
    ----------
    result : dict
        Output from run_mcmc()
    burn_in_frac : float
        Fraction of chain to discard as burn-in (0 to 1)
    thin : int
        Keep every thin-th sample

    Returns
    -------
    posterior : dict with keys:
        'samples' : array (M, N) - posterior depth samples
        'mean' : array (N,) - posterior mean depth per block
        'std' : array (N,) - posterior standard deviation per block
        'ci_5' : array (N,) - 5th percentile (lower bound of 90% CI)
        'ci_95' : array (N,) - 95th percentile (upper bound of 90% CI)
        'ci_2_5' : array (N,) - 2.5th percentile (lower bound of 95% CI)
        'ci_97_5' : array (N,) - 97.5th percentile (upper bound of 95% CI)
        'n_samples' : int
    """
    chain = result['chain']
    n_total = len(chain)
    burn_in = int(n_total * burn_in_frac)

    # Remove burn-in and thin
    samples = chain[burn_in::thin]

    return {
        'samples': samples,
        'mean': np.mean(samples, axis=0),
        'std': np.std(samples, axis=0),
        'ci_5': np.percentile(samples, 5, axis=0),
        'ci_95': np.percentile(samples, 95, axis=0),
        'ci_2_5': np.percentile(samples, 2.5, axis=0),
        'ci_97_5': np.percentile(samples, 97.5, axis=0),
        'n_samples': len(samples),
    }


def compute_coverage(true_depths, posterior, ci_level=90):
    """
    Compute what fraction of true depths fall within the credible interval.

    For well-calibrated UQ, a 90% CI should contain ~90% of true values.

    Parameters
    ----------
    true_depths : array (N,)
        True basement depths
    posterior : dict
        Output from process_chain()
    ci_level : int
        Credible interval level (90 or 95)

    Returns
    -------
    coverage : float
        Fraction of true depths within the CI (should be ~ci_level/100)
    """
    if ci_level == 90:
        lower = posterior['ci_5']
        upper = posterior['ci_95']
    elif ci_level == 95:
        lower = posterior['ci_2_5']
        upper = posterior['ci_97_5']
    else:
        raise ValueError("ci_level must be 90 or 95")

    within = np.sum((true_depths >= lower) & (true_depths <= upper))
    return within / np.asarray(true_depths).size


def run_mcmc_3d(obs_x, obs_y, gravity_obs,
                block_x_edges, block_y_edges,
                density_func, noise_std,
                n_iterations=50000, step_size=100.0,
                depth_min=200.0, depth_max=6000.0,
                smoothness_weight=0.0, n_sublayers=10,
                initial_depths=None, seed=42, verbose=True):
    """
    Run Metropolis-Hastings MCMC for a 3D grid of blocks (fixed lambda).

    Uses incremental forward model: only recomputes gravity for the
    one block whose depth changes, giving ~100x speedup over brute force.

    Parameters
    ----------
    obs_x, obs_y : array (M,)
        Observation station coordinates (meters)
    gravity_obs : array (M,)
        Observed gravity data (mGal)
    block_x_edges : array (Nx+1,)
        Block x-boundaries (meters)
    block_y_edges : array (Ny+1,)
        Block y-boundaries (meters)
    density_func : callable
        density_func(z) -> density contrast (kg/m^3)
    noise_std : float
        Estimated data noise level (mGal)
    n_iterations : int
        Total number of MCMC iterations
    step_size : float
        Standard deviation of proposal perturbation (meters)
    depth_min, depth_max : float
        Prior bounds on depth (meters)
    smoothness_weight : float
        Weight for 2D smoothness penalty (0 = no smoothness)
    n_sublayers : int
        Sublayers for depth-dependent density
    initial_depths : array (Nx, Ny) or None
        Initial depth model. If None, uses uniform mid-range.
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    result : dict with keys:
        'chain' : array (n_accepted, Nx, Ny) - accepted depth models
        'misfit_chain' : array (n_accepted,) - misfit for each accepted
        'acceptance_rate' : float
        'n_iterations' : int
        'all_misfits' : array (n_iterations,) - misfit at each iteration
    """
    np.random.seed(seed)

    Nx = len(block_x_edges) - 1
    Ny = len(block_y_edges) - 1
    M = len(obs_x)
    sigma2 = noise_std ** 2

    # Initialize depth model
    if initial_depths is not None:
        current_depths = np.array(initial_depths, dtype=float).copy()
        assert current_depths.shape == (Nx, Ny), \
            f"initial_depths shape {current_depths.shape} != expected ({Nx}, {Ny})"
    else:
        current_depths = np.ones((Nx, Ny)) * (depth_min + depth_max) / 2.0

    # --- Precompute gravity per block for incremental updates ---
    gravity_blocks = np.zeros((Nx, Ny, M))
    for ix in range(Nx):
        x1 = block_x_edges[ix]
        x2 = block_x_edges[ix + 1]
        for iy in range(Ny):
            y1 = block_y_edges[iy]
            y2 = block_y_edges[iy + 1]
            gravity_blocks[ix, iy, :] = compute_single_block_gravity(
                obs_x, obs_y, x1, x2, y1, y2,
                current_depths[ix, iy], density_func, n_sublayers
            )

    current_gravity = np.sum(gravity_blocks, axis=(0, 1))
    current_misfit = compute_misfit(gravity_obs, current_gravity)

    # 2D smoothness helper
    def compute_smoothness_2d(depths):
        sx = np.sum(np.diff(depths, axis=0) ** 2)
        sy = np.sum(np.diff(depths, axis=1) ** 2)
        return sx + sy

    # Initial energy
    if smoothness_weight > 0:
        current_energy = current_misfit / sigma2 + \
            smoothness_weight * compute_smoothness_2d(current_depths)
    else:
        current_energy = current_misfit / sigma2

    # Storage
    chain = [current_depths.copy()]
    misfit_chain = [current_misfit]
    all_misfits = np.zeros(n_iterations)
    n_accepted = 0

    if verbose:
        print(f"Starting 3D MCMC: {n_iterations} iterations, "
              f"{Nx}x{Ny}={Nx*Ny} blocks")
        print(f"Initial misfit: {current_misfit:.4f}")
        print(f"Step size: {step_size:.1f} m, Noise std: {noise_std:.3f} mGal")
        print(f"Using incremental forward model (100x speedup)")
        print("-" * 60)

    for it in range(n_iterations):
        # 1. Randomly select a block (ix, iy)
        ix = np.random.randint(0, Nx)
        iy = np.random.randint(0, Ny)

        # 2. Propose new depth for that block
        old_depth = current_depths[ix, iy]
        perturbation = np.random.normal(0, step_size)
        new_depth = old_depth + perturbation

        # 3. Check prior bounds
        if new_depth < depth_min or new_depth > depth_max:
            all_misfits[it] = current_misfit
            continue

        # 4. Incremental forward model: recompute only the changed block
        x1 = block_x_edges[ix]
        x2 = block_x_edges[ix + 1]
        y1 = block_y_edges[iy]
        y2 = block_y_edges[iy + 1]

        new_block_gravity = compute_single_block_gravity(
            obs_x, obs_y, x1, x2, y1, y2,
            new_depth, density_func, n_sublayers
        )
        proposed_gravity = (current_gravity
                            - gravity_blocks[ix, iy]
                            + new_block_gravity)

        # 5. Compute proposed misfit and energy
        proposed_misfit = compute_misfit(gravity_obs, proposed_gravity)

        if smoothness_weight > 0:
            # Temporarily set new depth to compute smoothness
            proposed_depths = current_depths.copy()
            proposed_depths[ix, iy] = new_depth
            proposed_energy = proposed_misfit / sigma2 + \
                smoothness_weight * compute_smoothness_2d(proposed_depths)
        else:
            proposed_energy = proposed_misfit / sigma2

        # 6. Metropolis acceptance criterion
        delta_energy = proposed_energy - current_energy
        if delta_energy < 0:
            accept = True
        else:
            accept = np.random.uniform() < np.exp(-delta_energy)

        # 7. Update
        if accept:
            current_depths[ix, iy] = new_depth
            gravity_blocks[ix, iy] = new_block_gravity
            current_gravity = proposed_gravity
            current_misfit = proposed_misfit
            current_energy = proposed_energy
            n_accepted += 1
            chain.append(current_depths.copy())
            misfit_chain.append(current_misfit)

        all_misfits[it] = current_misfit

        # Progress reporting
        if verbose and (it + 1) % (n_iterations // 10) == 0:
            acc_rate = n_accepted / (it + 1) * 100
            print(f"  Iteration {it+1:6d}/{n_iterations} | "
                  f"Misfit: {current_misfit:10.4f} | "
                  f"Accept rate: {acc_rate:5.1f}%")

    acceptance_rate = n_accepted / n_iterations
    if verbose:
        print("-" * 60)
        print(f"Done. Acceptance rate: {acceptance_rate*100:.1f}%")
        print(f"Final misfit: {current_misfit:.4f}")
        print(f"Total accepted models: {len(chain)}")

    return {
        'chain': np.array(chain),
        'misfit_chain': np.array(misfit_chain),
        'acceptance_rate': acceptance_rate,
        'n_iterations': n_iterations,
        'all_misfits': all_misfits,
    }


def process_chain_3d(result, burn_in_frac=0.5, thin=1):
    """
    Post-process 3D MCMC chain: remove burn-in and thin.

    Parameters
    ----------
    result : dict
        Output from run_mcmc_3d()
    burn_in_frac : float
        Fraction of chain to discard as burn-in (0 to 1)
    thin : int
        Keep every thin-th sample

    Returns
    -------
    posterior : dict with keys:
        'samples' : array (M, Nx, Ny) - posterior depth samples
        'mean' : array (Nx, Ny) - posterior mean depth per block
        'std' : array (Nx, Ny) - posterior standard deviation
        'ci_5' : array (Nx, Ny) - 5th percentile
        'ci_95' : array (Nx, Ny) - 95th percentile
        'ci_2_5' : array (Nx, Ny) - 2.5th percentile
        'ci_97_5' : array (Nx, Ny) - 97.5th percentile
        'n_samples' : int
    """
    chain = result['chain']
    n_total = len(chain)
    burn_in = int(n_total * burn_in_frac)

    samples = chain[burn_in::thin]

    return {
        'samples': samples,
        'mean': np.mean(samples, axis=0),
        'std': np.std(samples, axis=0),
        'ci_5': np.percentile(samples, 5, axis=0),
        'ci_95': np.percentile(samples, 95, axis=0),
        'ci_2_5': np.percentile(samples, 2.5, axis=0),
        'ci_97_5': np.percentile(samples, 97.5, axis=0),
        'n_samples': len(samples),
    }


def run_mcmc_3d_joint(obs_x, obs_y, gravity_obs,
                      block_x_edges, block_y_edges,
                      drho_0, noise_std,
                      n_iterations=50000,
                      step_depth=100.0, step_lambda=0.00003,
                      depth_min=0.0, depth_max=6000.0,
                      lambda_min=0.00001, lambda_max=0.003,
                      lambda_init=0.0003,
                      prob_perturb_lambda=0.2,
                      borehole_constraints=None,
                      smoothness_weight=0.0, n_sublayers=10,
                      initial_depths=None, seed=42, verbose=True):
    """
    Joint Bayesian MCMC inversion for 3D grid of blocks: basement depth
    AND lambda (compaction parameter in exponential density function).

    Combines the incremental forward model from run_mcmc_3d (only recompute
    one block when depth changes) with the joint depth+lambda estimation
    and borehole constraints from run_mcmc_joint.

    At each iteration, randomly choose to perturb either:
      - One block's depth (probability = 1 - prob_perturb_lambda)
        -> incremental update: only recompute that block's gravity
      - Lambda for all blocks (probability = prob_perturb_lambda)
        -> must recompute ALL blocks (density function changed globally)

    The density function is: drho(z) = drho_0 * exp(-lambda * z)
    where drho_0 is fixed and lambda is estimated by MCMC.

    Parameters
    ----------
    obs_x, obs_y : array (M,)
        Observation station coordinates (meters)
    gravity_obs : array (M,)
        Observed gravity data (mGal)
    block_x_edges : array (Nx+1,)
        Block x-boundaries (meters)
    block_y_edges : array (Ny+1,)
        Block y-boundaries (meters)
    drho_0 : float
        Surface density contrast (kg/m^3), fixed (e.g., -500)
    noise_std : float
        Estimated data noise level (mGal)
    n_iterations : int
        Total MCMC iterations
    step_depth : float
        Std of depth proposal perturbation (meters)
    step_lambda : float
        Std of lambda proposal perturbation (1/m)
    depth_min, depth_max : float
        Prior bounds on depth (meters)
    lambda_min, lambda_max : float
        Prior bounds on lambda (1/m)
    lambda_init : float
        Initial guess for lambda (1/m)
    prob_perturb_lambda : float
        Probability of choosing to perturb lambda at each iteration (0 to 1)
    borehole_constraints : dict or None
        Dictionary mapping (ix, iy) tuples to known depth (meters).
        Example: {(3, 5): 450.0, (7, 2): 120.0} means block (3,5) is
        locked at 450m and block (7,2) is locked at 120m.
    smoothness_weight : float
        Weight for 2D smoothness penalty (0 = no smoothness)
    n_sublayers : int
        Sublayers for depth-dependent density
    initial_depths : array (Nx, Ny) or None
        Initial depth model. If None, uses uniform mid-range.
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    result : dict with keys:
        'chain' : array (n_accepted, Nx, Ny) - accepted depth models
        'lambda_chain' : array (n_accepted,) - accepted lambda values
        'misfit_chain' : array (n_accepted,) - misfit for each accepted model
        'acceptance_rate' : float
        'depth_acceptance_rate' : float
        'lambda_acceptance_rate' : float
        'n_iterations' : int
        'all_misfits' : array (n_iterations,) - misfit at each iteration
        'all_lambdas' : array (n_iterations,) - lambda at each iteration
    """
    np.random.seed(seed)

    Nx = len(block_x_edges) - 1
    Ny = len(block_y_edges) - 1
    M = len(obs_x)
    sigma2 = noise_std ** 2

    # Handle borehole constraints
    if borehole_constraints is None:
        borehole_constraints = {}
    borehole_set = set(borehole_constraints.keys())

    # Build list of free blocks (not constrained by boreholes)
    free_blocks = [(ix, iy) for ix in range(Nx) for iy in range(Ny)
                   if (ix, iy) not in borehole_set]

    # Initialize depth model
    if initial_depths is not None:
        current_depths = np.array(initial_depths, dtype=float).copy()
        assert current_depths.shape == (Nx, Ny), \
            f"initial_depths shape {current_depths.shape} != expected ({Nx}, {Ny})"
    else:
        current_depths = np.ones((Nx, Ny)) * (depth_min + depth_max) / 2.0

    # Lock borehole depths to known values
    for (ix, iy), depth in borehole_constraints.items():
        current_depths[ix, iy] = depth

    # Initialize lambda
    current_lambda = lambda_init

    # Build density function with current lambda
    def make_density(lam):
        return lambda z: drho_0 * np.exp(-lam * z)

    current_density_func = make_density(current_lambda)

    # --- Precompute gravity per block for incremental updates ---
    gravity_blocks = np.zeros((Nx, Ny, M))
    for ix in range(Nx):
        x1 = block_x_edges[ix]
        x2 = block_x_edges[ix + 1]
        for iy in range(Ny):
            y1 = block_y_edges[iy]
            y2 = block_y_edges[iy + 1]
            gravity_blocks[ix, iy, :] = compute_single_block_gravity(
                obs_x, obs_y, x1, x2, y1, y2,
                current_depths[ix, iy], current_density_func, n_sublayers
            )

    current_gravity = np.sum(gravity_blocks, axis=(0, 1))
    current_misfit = compute_misfit(gravity_obs, current_gravity)

    # 2D smoothness helper
    def compute_smoothness_2d(depths):
        sx = np.sum(np.diff(depths, axis=0) ** 2)
        sy = np.sum(np.diff(depths, axis=1) ** 2)
        return sx + sy

    # Initial energy
    if smoothness_weight > 0:
        current_energy = current_misfit / sigma2 + \
            smoothness_weight * compute_smoothness_2d(current_depths)
    else:
        current_energy = current_misfit / sigma2

    # Storage
    chain = [current_depths.copy()]
    lambda_chain = [current_lambda]
    misfit_chain = [current_misfit]
    all_misfits = np.zeros(n_iterations)
    all_lambdas = np.zeros(n_iterations)
    all_lambdas[0] = current_lambda

    n_accepted = 0
    n_depth_proposed = 0
    n_depth_accepted = 0
    n_lambda_proposed = 0
    n_lambda_accepted = 0

    if verbose:
        print(f"Starting 3D Joint MCMC: {n_iterations} iterations, "
              f"{Nx}x{Ny}={Nx*Ny} blocks")
        print(f"Estimating: {len(free_blocks)} free depths + 1 lambda")
        if borehole_constraints:
            print(f"Borehole constraints: {len(borehole_constraints)} blocks locked")
            for (bx, by), bd in sorted(borehole_constraints.items()):
                print(f"  Block ({bx}, {by}): locked at {bd:.0f} m")
        print(f"Initial misfit: {current_misfit:.4f}")
        print(f"Initial lambda: {current_lambda:.6f}")
        print(f"Lambda perturbation probability: {prob_perturb_lambda*100:.0f}%")
        print(f"Step sizes: depth={step_depth:.1f}m, lambda={step_lambda:.6f}")
        print(f"Using incremental forward model (100x speedup for depth steps)")
        print("-" * 60)

    # Progress-reporting cadence: ~50 lines over the run, min 500 iters between prints
    progress_every = max(500, n_iterations // 50)
    t_start = time.time()

    for it in range(n_iterations):
        # Decide: perturb depth or lambda?
        perturb_lambda = np.random.uniform() < prob_perturb_lambda

        if perturb_lambda:
            # --- PERTURB LAMBDA ---
            n_lambda_proposed += 1

            proposed_lambda = current_lambda + np.random.normal(0, step_lambda)

            # Check prior bounds
            if proposed_lambda < lambda_min or proposed_lambda > lambda_max:
                all_misfits[it] = current_misfit
                all_lambdas[it] = current_lambda
                continue

            # Lambda changed -> must recompute ALL blocks
            proposed_density_func = make_density(proposed_lambda)

            # Save old gravity_blocks in case we reject
            old_gravity_blocks = gravity_blocks.copy()

            # Recompute gravity for every block with new density
            for bix in range(Nx):
                x1 = block_x_edges[bix]
                x2 = block_x_edges[bix + 1]
                for biy in range(Ny):
                    y1 = block_y_edges[biy]
                    y2 = block_y_edges[biy + 1]
                    gravity_blocks[bix, biy, :] = compute_single_block_gravity(
                        obs_x, obs_y, x1, x2, y1, y2,
                        current_depths[bix, biy],
                        proposed_density_func, n_sublayers
                    )

            proposed_gravity = np.sum(gravity_blocks, axis=(0, 1))
            proposed_misfit = compute_misfit(gravity_obs, proposed_gravity)

            if smoothness_weight > 0:
                proposed_energy = proposed_misfit / sigma2 + \
                    smoothness_weight * compute_smoothness_2d(current_depths)
            else:
                proposed_energy = proposed_misfit / sigma2

            # Metropolis criterion
            delta_energy = proposed_energy - current_energy
            if delta_energy < 0:
                accept = True
            else:
                accept = np.random.uniform() < np.exp(-delta_energy)

            if accept:
                current_lambda = proposed_lambda
                current_density_func = proposed_density_func
                current_gravity = proposed_gravity
                current_misfit = proposed_misfit
                current_energy = proposed_energy
                n_lambda_accepted += 1
                n_accepted += 1
                chain.append(current_depths.copy())
                lambda_chain.append(current_lambda)
                misfit_chain.append(current_misfit)
                # gravity_blocks already updated in place
            else:
                # Restore gravity_blocks to old values
                gravity_blocks[:] = old_gravity_blocks

        else:
            # --- PERTURB ONE DEPTH ---
            n_depth_proposed += 1

            # Pick random block from free blocks (not borehole-constrained)
            fb_idx = np.random.randint(0, len(free_blocks))
            ix, iy = free_blocks[fb_idx]

            # Propose new depth
            old_depth = current_depths[ix, iy]
            new_depth = old_depth + np.random.normal(0, step_depth)

            # Check prior bounds
            if new_depth < depth_min or new_depth > depth_max:
                all_misfits[it] = current_misfit
                all_lambdas[it] = current_lambda
                continue

            # Incremental forward model: recompute only the changed block
            x1 = block_x_edges[ix]
            x2 = block_x_edges[ix + 1]
            y1 = block_y_edges[iy]
            y2 = block_y_edges[iy + 1]

            new_block_gravity = compute_single_block_gravity(
                obs_x, obs_y, x1, x2, y1, y2,
                new_depth, current_density_func, n_sublayers
            )
            proposed_gravity = (current_gravity
                                - gravity_blocks[ix, iy]
                                + new_block_gravity)

            proposed_misfit = compute_misfit(gravity_obs, proposed_gravity)

            if smoothness_weight > 0:
                proposed_depths = current_depths.copy()
                proposed_depths[ix, iy] = new_depth
                proposed_energy = proposed_misfit / sigma2 + \
                    smoothness_weight * compute_smoothness_2d(proposed_depths)
            else:
                proposed_energy = proposed_misfit / sigma2

            # Metropolis criterion
            delta_energy = proposed_energy - current_energy
            if delta_energy < 0:
                accept = True
            else:
                accept = np.random.uniform() < np.exp(-delta_energy)

            if accept:
                current_depths[ix, iy] = new_depth
                gravity_blocks[ix, iy] = new_block_gravity
                current_gravity = proposed_gravity
                current_misfit = proposed_misfit
                current_energy = proposed_energy
                n_depth_accepted += 1
                n_accepted += 1
                chain.append(current_depths.copy())
                lambda_chain.append(current_lambda)
                misfit_chain.append(current_misfit)

        all_misfits[it] = current_misfit
        all_lambdas[it] = current_lambda

        # Progress reporting
        if verbose and (it + 1) % progress_every == 0:
            elapsed = time.time() - t_start
            frac = (it + 1) / n_iterations
            eta = elapsed * (1.0 - frac) / max(frac, 1e-9)
            rate = (it + 1) / max(elapsed, 1e-9)
            acc_rate = n_accepted / (it + 1) * 100
            print(f"  [{frac*100:5.1f}%] iter {it+1:>7d}/{n_iterations} | "
                  f"misfit {current_misfit:8.2f} | "
                  f"λ {current_lambda:.2e} | "
                  f"accept {acc_rate:5.1f}% | "
                  f"{rate:6.0f} it/s | "
                  f"elapsed {elapsed/60:5.1f} min | "
                  f"eta {eta/60:5.1f} min",
                  flush=True)

    # Final statistics
    acceptance_rate = n_accepted / n_iterations
    depth_acc = n_depth_accepted / max(n_depth_proposed, 1)
    lambda_acc = n_lambda_accepted / max(n_lambda_proposed, 1)

    if verbose:
        print("-" * 60)
        total_min = (time.time() - t_start) / 60
        print(f"Done in {total_min:.1f} min. Overall acceptance: {acceptance_rate*100:.1f}%")
        print(f"  Depth acceptance:  {depth_acc*100:.1f}% "
              f"({n_depth_accepted}/{n_depth_proposed})")
        print(f"  Lambda acceptance: {lambda_acc*100:.1f}% "
              f"({n_lambda_accepted}/{n_lambda_proposed})")
        print(f"Final misfit: {current_misfit:.4f}")
        print(f"Final lambda: {current_lambda:.6f}")
        print(f"Total accepted models: {len(chain)}")

    return {
        'chain': np.array(chain),
        'lambda_chain': np.array(lambda_chain),
        'misfit_chain': np.array(misfit_chain),
        'acceptance_rate': acceptance_rate,
        'depth_acceptance_rate': depth_acc,
        'lambda_acceptance_rate': lambda_acc,
        'n_iterations': n_iterations,
        'all_misfits': all_misfits,
        'all_lambdas': all_lambdas,
    }


def process_chain_3d_joint(result, burn_in_frac=0.5, thin=1):
    """
    Post-process 3D joint MCMC chain: remove burn-in, thin,
    and compute statistics for both depths and lambda.

    Parameters
    ----------
    result : dict
        Output from run_mcmc_3d_joint()
    burn_in_frac : float
        Fraction of chain to discard as burn-in (0 to 1)
    thin : int
        Keep every thin-th sample

    Returns
    -------
    posterior : dict with keys:
        Depth statistics (all Nx x Ny arrays):
        'samples' : array (M, Nx, Ny) - posterior depth samples
        'mean' : array (Nx, Ny) - posterior mean depth per block
        'std' : array (Nx, Ny) - posterior standard deviation
        'ci_5' : array (Nx, Ny) - 5th percentile
        'ci_95' : array (Nx, Ny) - 95th percentile
        'ci_2_5' : array (Nx, Ny) - 2.5th percentile
        'ci_97_5' : array (Nx, Ny) - 97.5th percentile
        'n_samples' : int

        Lambda statistics (scalars):
        'lambda_samples' : array - post-burn-in lambda samples
        'lambda_mean' : float
        'lambda_std' : float
        'lambda_ci_5' : float
        'lambda_ci_95' : float
    """
    chain = result['chain']
    lambda_chain = result['lambda_chain']
    n_total = len(chain)
    burn_in = int(n_total * burn_in_frac)

    depth_samples = chain[burn_in::thin]
    lambda_samples = lambda_chain[burn_in::thin]

    return {
        # Depth statistics
        'samples': depth_samples,
        'mean': np.mean(depth_samples, axis=0),
        'std': np.std(depth_samples, axis=0),
        'ci_5': np.percentile(depth_samples, 5, axis=0),
        'ci_95': np.percentile(depth_samples, 95, axis=0),
        'ci_2_5': np.percentile(depth_samples, 2.5, axis=0),
        'ci_97_5': np.percentile(depth_samples, 97.5, axis=0),
        'n_samples': len(depth_samples),
        # Lambda statistics
        'lambda_samples': lambda_samples,
        'lambda_mean': np.mean(lambda_samples),
        'lambda_std': np.std(lambda_samples),
        'lambda_ci_5': np.percentile(lambda_samples, 5),
        'lambda_ci_95': np.percentile(lambda_samples, 95),
    }
