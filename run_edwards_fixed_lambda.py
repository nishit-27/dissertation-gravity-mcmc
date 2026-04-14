"""
Edwards AFB — Fixed Lambda Inversion (Fast)
=============================================
Uses existing run_mcmc_3d() with fixed density. No lambda estimation.
Much faster — pure incremental forward model, no all-block recomputation.
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
from src.mcmc_inversion import run_mcmc_3d, process_chain_3d
from src.utils import make_density_func

# ============================================================
# Configuration
# ============================================================
NX, NY = 12, 10
N_ITERATIONS = 30000
SUBSAMPLE_SPACING = 3000  # meters

STUDY_BOUNDS = {
    'lon_min': -117.95,
    'lon_max': -117.50,
    'lat_min': 34.82,
    'lat_max': 35.10,
}

# Fixed density — typical literature values
DRHO_0 = -500.0
LAMBDA = 0.0003
density_func = make_density_func('exponential', drho_0=DRHO_0, lam=LAMBDA)

STEP_SIZE = 50.0
NOISE_STD = 0.5
DEPTH_MIN = 0.0
DEPTH_MAX = 1500.0
SMOOTHNESS_WEIGHT = 1e-6
N_SUBLAYERS = 10

print("=" * 60)
print("Edwards AFB — Fixed Lambda Inversion (Fast)")
print("=" * 60)
print(f"Density: drho_0={DRHO_0}, lambda={LAMBDA} (fixed)")

# ============================================================
# 1. Load data
# ============================================================
print("\n[1/5] Loading data...")
data = prepare_edwards_data('real_data/edwards_afb/', study_bounds=STUDY_BOUNDS)

grav = data['gravity']
bwells = data['basement_wells']
dgrid = data['depth_grid']

gravity_corrected = grav['isostatic'] - data['regional_correction']
print(f"  Stations: {len(grav['x'])}, Regional correction: {data['regional_correction']:.2f} mGal")

# ============================================================
# 2. Grid + subsample
# ============================================================
print(f"\n[2/5] Setting up {NX}x{NY} grid, subsampling...")

x_max = grav['x'].max()
y_max = grav['y'].max()
block_x_edges = np.linspace(0, x_max, NX + 1)
block_y_edges = np.linspace(0, y_max, NY + 1)

obs_x, obs_y, obs_gravity, n_per_cell = subsample_gravity(
    grav['x'], grav['y'], gravity_corrected, SUBSAMPLE_SPACING
)
print(f"  Subsampled: {len(grav['x'])} -> {len(obs_x)} stations")
print(f"  Block size: {(block_x_edges[1]-block_x_edges[0])/1000:.1f} x {(block_y_edges[1]-block_y_edges[0])/1000:.1f} km")

# ============================================================
# 3. Initial depths from USGS
# ============================================================
print(f"\n[3/5] Initial depths from USGS model...")

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
# 4. Run MCMC (fixed lambda — FAST)
# ============================================================
print(f"\n[4/5] Running MCMC ({N_ITERATIONS} iterations, fixed lambda)...")
t0 = time.time()

result = run_mcmc_3d(
    obs_x=obs_x,
    obs_y=obs_y,
    gravity_obs=obs_gravity,
    block_x_edges=block_x_edges,
    block_y_edges=block_y_edges,
    density_func=density_func,
    noise_std=NOISE_STD,
    n_iterations=N_ITERATIONS,
    step_size=STEP_SIZE,
    depth_min=DEPTH_MIN,
    depth_max=DEPTH_MAX,
    smoothness_weight=SMOOTHNESS_WEIGHT,
    n_sublayers=N_SUBLAYERS,
    initial_depths=initial_depths,
    seed=42,
    verbose=True
)

elapsed = time.time() - t0
print(f"\nRuntime: {elapsed/60:.1f} minutes")

# ============================================================
# 5. Results + Validation
# ============================================================
print(f"\n[5/5] Processing results...")
posterior = process_chain_3d(result, burn_in_frac=0.5, thin=1)

mean_depths = posterior['mean']
std_depths = posterior['std']

# Validate against ALL 114 basement wells (none used as constraints)
all_wells, _ = assign_wells_to_blocks(
    bwells['x'], bwells['y'], bwells['depth_m'],
    block_x_edges, block_y_edges
)

errors = []
covered_90 = 0
covered_95 = 0
n_val = 0

for (ix, iy), well_depth in all_wells.items():
    if 0 <= ix < NX and 0 <= iy < NY:
        est = mean_depths[ix, iy]
        ci5 = posterior['ci_5'][ix, iy]
        ci95 = posterior['ci_95'][ix, iy]
        ci2_5 = posterior['ci_2_5'][ix, iy]
        ci97_5 = posterior['ci_97_5'][ix, iy]
        errors.append(est - well_depth)
        if ci5 <= well_depth <= ci95:
            covered_90 += 1
        if ci2_5 <= well_depth <= ci97_5:
            covered_95 += 1
        n_val += 1

errors = np.array(errors)
rms = np.sqrt(np.mean(errors**2))
mae = np.mean(np.abs(errors))
bias = np.mean(errors)

print(f"\n{'='*60}")
print(f"RESULTS — Edwards AFB (Fixed Lambda)")
print(f"{'='*60}")
print(f"Acceptance rate:    {result['acceptance_rate']*100:.1f}%")
print(f"Posterior samples:  {posterior['n_samples']}")
print(f"Depth range:        {mean_depths.min():.0f} - {mean_depths.max():.0f} m")
print(f"Mean uncertainty:   {std_depths.mean():.0f} m")
print(f"\nValidation ({n_val} wells, ALL independent):")
print(f"  RMS error:        {rms:.1f} m")
print(f"  MAE:              {mae:.1f} m")
print(f"  Bias:             {bias:.1f} m")
print(f"  90% CI coverage:  {covered_90}/{n_val} ({covered_90/n_val*100:.0f}%)")
print(f"  95% CI coverage:  {covered_95}/{n_val} ({covered_95/n_val*100:.0f}%)")
print(f"  Runtime:          {elapsed/60:.1f} min")

# ============================================================
# Save + Plots
# ============================================================
out_dir = 'results/exp_edwards_fixed_lambda/'
os.makedirs(out_dir, exist_ok=True)

np.savez(os.path.join(out_dir, 'results_data.npz'),
         mean_depths=mean_depths, std_depths=std_depths,
         ci_5=posterior['ci_5'], ci_95=posterior['ci_95'],
         ci_2_5=posterior['ci_2_5'], ci_97_5=posterior['ci_97_5'],
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         obs_x=obs_x, obs_y=obs_y, obs_gravity=obs_gravity)

# Depth + uncertainty maps
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im0 = axes[0].pcolormesh(block_x_edges/1000, block_y_edges/1000, mean_depths.T,
                          cmap='viridis_r', shading='flat')
plt.colorbar(im0, ax=axes[0], label='Depth (m)')
axes[0].set_title('Posterior Mean Depth')
axes[0].set_xlabel('X (km)')
axes[0].set_ylabel('Y (km)')
axes[0].set_aspect('equal')

im1 = axes[1].pcolormesh(block_x_edges/1000, block_y_edges/1000, std_depths.T,
                          cmap='hot_r', shading='flat')
plt.colorbar(im1, ax=axes[1], label='Std Dev (m)')
axes[1].set_title('Posterior Uncertainty')
axes[1].set_xlabel('X (km)')
axes[1].set_ylabel('Y (km)')
axes[1].set_aspect('equal')

fig.suptitle(f'Edwards AFB — Fixed Lambda (drho_0={DRHO_0}, λ={LAMBDA})', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'depth_and_uncertainty.png'), dpi=150)
plt.close(fig)

# Validation scatter
fig, ax = plt.subplots(figsize=(8, 8))
well_d = []
mcmc_d = []
mcmc_s = []
for (ix, iy), wd in all_wells.items():
    if 0 <= ix < NX and 0 <= iy < NY:
        well_d.append(wd)
        mcmc_d.append(mean_depths[ix, iy])
        mcmc_s.append(std_depths[ix, iy])

well_d = np.array(well_d)
mcmc_d = np.array(mcmc_d)
mcmc_s = np.array(mcmc_s)

ax.errorbar(well_d, mcmc_d, yerr=mcmc_s, fmt='o', markersize=5,
            capsize=3, alpha=0.7, color='steelblue')
lim = max(well_d.max(), mcmc_d.max()) * 1.1 + 10
ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='1:1 line')
ax.set_xlabel('Well Depth (m)')
ax.set_ylabel('MCMC Estimated Depth (m)')
ax.set_title(f'Validation: MCMC vs Wells (RMS={rms:.0f}m, 90%cov={covered_90/n_val*100:.0f}%)')
ax.legend()
ax.set_aspect('equal')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'validation_scatter.png'), dpi=150)
plt.close(fig)

# Misfit trace
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(result['all_misfits'], linewidth=0.3, alpha=0.7)
ax.set_xlabel('Iteration')
ax.set_ylabel('Misfit')
ax.set_title('Misfit Trace')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'misfit_trace.png'), dpi=150)
plt.close(fig)

print(f"\nAll saved to: {out_dir}")
print("=" * 60)
