"""
Edwards AFB Real Data Inversion
================================
Bayesian MCMC inversion of gravity data from Edwards Air Force Base.
Jointly estimates basement depth AND density compaction parameter (lambda)
with borehole constraints and full uncertainty quantification.

Data source: USGS Open-File Report 2019-1128 (Langenheim et al., 2019)
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import (
    prepare_edwards_data, subsample_gravity, assign_wells_to_blocks
)
from src.mcmc_inversion import run_mcmc_3d_joint, process_chain_3d_joint
from src.utils import exponential_density

# ============================================================
# Configuration
# ============================================================
# Quick test settings (tune first, then scale up)
QUICK_TEST = True  # Set False for production run

if QUICK_TEST:
    NX, NY = 12, 10            # 120 blocks
    N_ITERATIONS = 30000
    SUBSAMPLE_SPACING = 3000   # meters — ~200 stations
    N_CONSTRAINT_WELLS = 5
    TAG = "quick"
else:
    NX, NY = 15, 12            # 180 blocks
    N_ITERATIONS = 100000
    SUBSAMPLE_SPACING = 2500   # meters — ~300 stations
    N_CONSTRAINT_WELLS = 10
    TAG = "production"

# Study area bounds (decimal degrees)
STUDY_BOUNDS = {
    'lon_min': -117.95,
    'lon_max': -117.50,
    'lat_min': 34.82,
    'lat_max': 35.10,
}

# MCMC parameters
STEP_DEPTH = 50.0          # meters (shallower basin than synthetic)
STEP_LAMBDA = 0.00005      # 1/m
NOISE_STD = 0.5            # mGal (measurement + terrain + model errors)
DEPTH_MIN = 0.0            # bedrock at surface in some areas
DEPTH_MAX = 1500.0         # USGS max ~1219m + margin
LAMBDA_INIT = 0.0003       # starting guess
LAMBDA_MIN = 0.00005
LAMBDA_MAX = 0.003
PROB_PERTURB_LAMBDA = 0.2
SMOOTHNESS_WEIGHT = 1e-6
DRHO_0 = -500.0            # surface density contrast (fixed)
N_SUBLAYERS = 10

print("=" * 60)
print(f"Edwards AFB Bayesian MCMC Inversion ({TAG})")
print("=" * 60)

# ============================================================
# 1. Load and preprocess data
# ============================================================
print("\n[1/7] Loading and preprocessing data...")

data = prepare_edwards_data('real_data/edwards_afb/', study_bounds=STUDY_BOUNDS)

grav = data['gravity']
bwells = data['basement_wells']
bawells = data['basin_wells']
dgrid = data['depth_grid']

print(f"  Gravity stations: {len(grav['x'])}")
print(f"  Basement wells: {len(bwells['x'])}")
print(f"  Study area: {grav['x'].max()/1000:.0f} x {grav['y'].max()/1000:.0f} km")
print(f"  Regional correction applied: {data['regional_correction']:.2f} mGal")

# Apply regional correction to gravity
gravity_corrected = grav['isostatic'] - data['regional_correction']
print(f"  Corrected gravity range: {gravity_corrected.min():.1f} to {gravity_corrected.max():.1f} mGal")

# ============================================================
# 2. Set up block grid
# ============================================================
print(f"\n[2/7] Setting up {NX}x{NY} block grid...")

x_max = grav['x'].max()
y_max = grav['y'].max()

block_x_edges = np.linspace(0, x_max, NX + 1)
block_y_edges = np.linspace(0, y_max, NY + 1)

block_size_x = block_x_edges[1] - block_x_edges[0]
block_size_y = block_y_edges[1] - block_y_edges[0]
print(f"  Block size: {block_size_x/1000:.1f} x {block_size_y/1000:.1f} km")
print(f"  Total blocks: {NX * NY}")

# ============================================================
# 3. Subsample gravity stations
# ============================================================
print(f"\n[3/7] Subsampling gravity stations (spacing={SUBSAMPLE_SPACING}m)...")

obs_x, obs_y, obs_gravity, n_per_cell = subsample_gravity(
    grav['x'], grav['y'], gravity_corrected, SUBSAMPLE_SPACING
)
print(f"  Subsampled: {len(grav['x'])} -> {len(obs_x)} stations")
print(f"  Gravity range: {obs_gravity.min():.1f} to {obs_gravity.max():.1f} mGal")

# ============================================================
# 4. Select borehole constraints
# ============================================================
print(f"\n[4/7] Selecting {N_CONSTRAINT_WELLS} borehole constraints...")

# Only use wells with depth > 10m (skip surface/shallow ones)
deep_mask = bwells['depth_m'] > 10.0
deep_x = bwells['x'][deep_mask]
deep_y = bwells['y'][deep_mask]
deep_depths = bwells['depth_m'][deep_mask]
deep_ids = np.array(bwells['well_id'])[deep_mask]
print(f"  Wells with depth > 10m: {deep_mask.sum()}")

# Select spatially dispersed wells using a grid-based strategy
# Divide study area into subregions and pick one well per region
n_sub_x = int(np.ceil(np.sqrt(N_CONSTRAINT_WELLS * x_max / y_max)))
n_sub_y = int(np.ceil(N_CONSTRAINT_WELLS / n_sub_x))
sub_x_edges = np.linspace(0, x_max, n_sub_x + 1)
sub_y_edges = np.linspace(0, y_max, n_sub_y + 1)

constraint_indices = []
for sx in range(n_sub_x):
    for sy in range(n_sub_y):
        if len(constraint_indices) >= N_CONSTRAINT_WELLS:
            break
        # Find wells in this subregion
        in_sub = (
            (deep_x >= sub_x_edges[sx]) & (deep_x < sub_x_edges[sx + 1]) &
            (deep_y >= sub_y_edges[sy]) & (deep_y < sub_y_edges[sy + 1])
        )
        if in_sub.sum() > 0:
            # Pick the well closest to center of subregion
            cx = (sub_x_edges[sx] + sub_x_edges[sx + 1]) / 2
            cy = (sub_y_edges[sy] + sub_y_edges[sy + 1]) / 2
            dists = (deep_x[in_sub] - cx)**2 + (deep_y[in_sub] - cy)**2
            idx_in_sub = np.where(in_sub)[0][np.argmin(dists)]
            constraint_indices.append(idx_in_sub)

constraint_indices = constraint_indices[:N_CONSTRAINT_WELLS]

# Assign selected wells to blocks
constraint_x = deep_x[constraint_indices]
constraint_y = deep_y[constraint_indices]
constraint_depths = deep_depths[constraint_indices]
constraint_ids = deep_ids[constraint_indices]

borehole_constraints, constraint_log = assign_wells_to_blocks(
    constraint_x, constraint_y, constraint_depths,
    block_x_edges, block_y_edges
)

print(f"  Selected {len(borehole_constraints)} constraint wells:")
for (ix, iy), depth in sorted(borehole_constraints.items()):
    print(f"    Block ({ix},{iy}): {depth:.1f} m")

# Identify validation wells (all basement wells NOT used as constraints)
constraint_block_set = set(borehole_constraints.keys())
all_well_assignments, _ = assign_wells_to_blocks(
    bwells['x'], bwells['y'], bwells['depth_m'],
    block_x_edges, block_y_edges
)
validation_wells = {k: v for k, v in all_well_assignments.items()
                    if k not in constraint_block_set}
print(f"  Validation wells (held out): {len(validation_wells)} blocks with wells")

# ============================================================
# 5. Initial depth model
# ============================================================
print(f"\n[5/7] Computing initial depth model from USGS grid...")

# Average USGS depths within each block for a better starting point
initial_depths = np.ones((NX, NY)) * (DEPTH_MIN + DEPTH_MAX) / 2.0

if len(dgrid['x']) > 0:
    for ix in range(NX):
        for iy in range(NY):
            mask = (
                (dgrid['x'] >= block_x_edges[ix]) &
                (dgrid['x'] < block_x_edges[ix + 1]) &
                (dgrid['y'] >= block_y_edges[iy]) &
                (dgrid['y'] < block_y_edges[iy + 1])
            )
            if mask.sum() > 0:
                initial_depths[ix, iy] = np.clip(
                    np.mean(dgrid['depth_m'][mask]),
                    DEPTH_MIN + 1, DEPTH_MAX - 1
                )

print(f"  Initial depth range: {initial_depths.min():.0f} - {initial_depths.max():.0f} m")

# ============================================================
# 6. Run MCMC
# ============================================================
print(f"\n[6/7] Running 3D Joint MCMC ({N_ITERATIONS} iterations)...")
print(f"  Parameters: step_depth={STEP_DEPTH}, step_lambda={STEP_LAMBDA}")
print(f"  noise_std={NOISE_STD}, smoothness={SMOOTHNESS_WEIGHT}")
print(f"  lambda_init={LAMBDA_INIT}, drho_0={DRHO_0}")
print(f"  Observation stations: {len(obs_x)}")

t0 = time.time()

result = run_mcmc_3d_joint(
    obs_x=obs_x,
    obs_y=obs_y,
    gravity_obs=obs_gravity,
    block_x_edges=block_x_edges,
    block_y_edges=block_y_edges,
    drho_0=DRHO_0,
    noise_std=NOISE_STD,
    n_iterations=N_ITERATIONS,
    step_depth=STEP_DEPTH,
    step_lambda=STEP_LAMBDA,
    depth_min=DEPTH_MIN,
    depth_max=DEPTH_MAX,
    lambda_min=LAMBDA_MIN,
    lambda_max=LAMBDA_MAX,
    lambda_init=LAMBDA_INIT,
    prob_perturb_lambda=PROB_PERTURB_LAMBDA,
    borehole_constraints=borehole_constraints,
    smoothness_weight=SMOOTHNESS_WEIGHT,
    n_sublayers=N_SUBLAYERS,
    initial_depths=initial_depths,
    seed=42,
    verbose=True
)

elapsed = time.time() - t0
print(f"\nRuntime: {elapsed/60:.1f} minutes")

# ============================================================
# 7. Process and validate
# ============================================================
print(f"\n[7/7] Processing results and validating...")

posterior = process_chain_3d_joint(result, burn_in_frac=0.5, thin=1)

mean_depths = posterior['mean']
std_depths = posterior['std']
lambda_mean = posterior['lambda_mean']
lambda_std = posterior['lambda_std']

# --- Validation against held-out wells ---
print(f"\n{'='*60}")
print(f"RESULTS — Edwards AFB ({TAG})")
print(f"{'='*60}")

print(f"\nMCMC Diagnostics:")
print(f"  Overall acceptance:  {result['acceptance_rate']*100:.1f}%")
print(f"  Depth acceptance:    {result['depth_acceptance_rate']*100:.1f}%")
print(f"  Lambda acceptance:   {result['lambda_acceptance_rate']*100:.1f}%")
print(f"  Posterior samples:   {posterior['n_samples']}")

print(f"\nRecovered Density Parameter:")
print(f"  Lambda: {lambda_mean:.6f} +/- {lambda_std:.6f}")
print(f"  90% CI: [{posterior['lambda_ci_5']:.6f}, {posterior['lambda_ci_95']:.6f}]")
print(f"  (USGS equivalent: ~0.00065)")

print(f"\nDepth Statistics:")
print(f"  Mean depth range: {mean_depths.min():.0f} - {mean_depths.max():.0f} m")
print(f"  Mean uncertainty: {std_depths.mean():.0f} m")

# Validation against held-out wells
if len(validation_wells) > 0:
    errors = []
    covered_90 = 0
    covered_95 = 0
    n_val = 0

    print(f"\nValidation against {len(validation_wells)} held-out well locations:")
    for (ix, iy), well_depth in sorted(validation_wells.items()):
        if 0 <= ix < NX and 0 <= iy < NY:
            est = mean_depths[ix, iy]
            std = std_depths[ix, iy]
            ci5 = posterior['ci_5'][ix, iy]
            ci95 = posterior['ci_95'][ix, iy]
            ci2_5 = posterior['ci_2_5'][ix, iy]
            ci97_5 = posterior['ci_97_5'][ix, iy]

            err = est - well_depth
            errors.append(err)
            if ci5 <= well_depth <= ci95:
                covered_90 += 1
            if ci2_5 <= well_depth <= ci97_5:
                covered_95 += 1
            n_val += 1

    errors = np.array(errors)
    rms = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)

    print(f"\n  Well Validation Summary:")
    print(f"  RMS error:        {rms:.1f} m")
    print(f"  MAE:              {mae:.1f} m")
    print(f"  Bias:             {bias:.1f} m")
    print(f"  90% CI coverage:  {covered_90}/{n_val} ({covered_90/n_val*100:.0f}%)")
    print(f"  95% CI coverage:  {covered_95}/{n_val} ({covered_95/n_val*100:.0f}%)")
    print(f"  (Target: 90% and 95% respectively)")

# ============================================================
# Save results
# ============================================================
out_dir = f'results/exp_edwards_{TAG}/'
os.makedirs(out_dir, exist_ok=True)

np.savez(os.path.join(out_dir, 'results_data.npz'),
         mean_depths=mean_depths,
         std_depths=std_depths,
         ci_5=posterior['ci_5'],
         ci_95=posterior['ci_95'],
         ci_2_5=posterior['ci_2_5'],
         ci_97_5=posterior['ci_97_5'],
         lambda_mean=lambda_mean,
         lambda_std=lambda_std,
         block_x_edges=block_x_edges,
         block_y_edges=block_y_edges,
         obs_x=obs_x,
         obs_y=obs_y,
         obs_gravity=obs_gravity)

# ============================================================
# Quick plots
# ============================================================
# Depth map
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im0 = axes[0].pcolormesh(block_x_edges/1000, block_y_edges/1000, mean_depths.T,
                          cmap='viridis_r', shading='flat')
plt.colorbar(im0, ax=axes[0], label='Depth (m)')
axes[0].set_title('MCMC Posterior Mean Depth')
axes[0].set_xlabel('X (km)')
axes[0].set_ylabel('Y (km)')
axes[0].set_aspect('equal')

# Plot constraint wells
for (ix, iy), d in borehole_constraints.items():
    cx = (block_x_edges[ix] + block_x_edges[ix+1]) / 2000
    cy = (block_y_edges[iy] + block_y_edges[iy+1]) / 2000
    axes[0].plot(cx, cy, 'r^', markersize=8, markeredgecolor='black')

im1 = axes[1].pcolormesh(block_x_edges/1000, block_y_edges/1000, std_depths.T,
                          cmap='hot_r', shading='flat')
plt.colorbar(im1, ax=axes[1], label='Std Dev (m)')
axes[1].set_title('Posterior Uncertainty')
axes[1].set_xlabel('X (km)')
axes[1].set_ylabel('Y (km)')
axes[1].set_aspect('equal')

fig.suptitle(f'Edwards AFB MCMC Inversion ({TAG})', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'depth_and_uncertainty.png'), dpi=150)
print(f"\nSaved: {out_dir}depth_and_uncertainty.png")
plt.close(fig)

# Lambda trace
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(result['all_lambdas'], linewidth=0.3, alpha=0.7)
axes[0].axhline(lambda_mean, color='red', linewidth=1, label=f'Mean={lambda_mean:.6f}')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Lambda (1/m)')
axes[0].set_title('Lambda Trace')
axes[0].legend()

lambda_samples = posterior['lambda_samples']
axes[1].hist(lambda_samples, bins=50, density=True, alpha=0.7, color='steelblue')
axes[1].axvline(lambda_mean, color='red', linewidth=2, label=f'Mean={lambda_mean:.6f}')
axes[1].axvline(posterior['lambda_ci_5'], color='red', linestyle='--', linewidth=1)
axes[1].axvline(posterior['lambda_ci_95'], color='red', linestyle='--', linewidth=1,
                label=f'90% CI')
axes[1].set_xlabel('Lambda (1/m)')
axes[1].set_ylabel('Density')
axes[1].set_title('Lambda Posterior')
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'lambda_results.png'), dpi=150)
print(f"Saved: {out_dir}lambda_results.png")
plt.close(fig)

# Misfit trace
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(result['all_misfits'], linewidth=0.3, alpha=0.7)
ax.set_xlabel('Iteration')
ax.set_ylabel('Misfit')
ax.set_title('Misfit Trace')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'misfit_trace.png'), dpi=150)
print(f"Saved: {out_dir}misfit_trace.png")
plt.close(fig)

# Validation scatter plot
if len(validation_wells) > 0:
    fig, ax = plt.subplots(figsize=(8, 8))

    well_d = []
    mcmc_d = []
    mcmc_std = []
    for (ix, iy), wd in validation_wells.items():
        if 0 <= ix < NX and 0 <= iy < NY:
            well_d.append(wd)
            mcmc_d.append(mean_depths[ix, iy])
            mcmc_std.append(std_depths[ix, iy])

    well_d = np.array(well_d)
    mcmc_d = np.array(mcmc_d)
    mcmc_std = np.array(mcmc_std)

    ax.errorbar(well_d, mcmc_d, yerr=mcmc_std, fmt='o', markersize=5,
                capsize=3, alpha=0.7, color='steelblue')

    lim = max(well_d.max(), mcmc_d.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='1:1 line')
    ax.set_xlabel('Well Depth (m)')
    ax.set_ylabel('MCMC Estimated Depth (m)')
    ax.set_title(f'Validation: MCMC vs Wells (RMS={rms:.0f}m, n={n_val})')
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'validation_scatter.png'), dpi=150)
    print(f"Saved: {out_dir}validation_scatter.png")
    plt.close(fig)

print(f"\nAll results saved to: {out_dir}")
print("=" * 60)
