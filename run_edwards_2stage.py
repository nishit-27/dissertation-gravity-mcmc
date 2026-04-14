"""
Edwards AFB — 2-Stage Inversion
================================
Stage 1: Estimate lambda from 2-3 boreholes using slab approximation (fast)
Stage 2: Depth-only MCMC with the estimated lambda (using run_mcmc_3d)

Robust approach that decouples density estimation from depth inversion.
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
STUDY_BOUNDS = {
    'lon_min': -117.95,
    'lon_max': -117.50,
    'lat_min': 34.82,
    'lat_max': 35.10,
}
NX, NY = 12, 10
N_ITERATIONS = 15000  # smaller for testing
SUBSAMPLE_SPACING = 3000  # gives ~150 stations
DRHO_0 = -500.0        # Fixed surface density contrast (literature)
STEP_SIZE = 50.0
NOISE_STD = 1.0        # Slightly larger for real data noise
DEPTH_MIN = 0.0
DEPTH_MAX = 1500.0
SMOOTHNESS_WEIGHT = 1e-6
N_SUBLAYERS = 10

print("=" * 60)
print("Edwards AFB — 2-Stage Bayesian Inversion")
print("=" * 60)

# ============================================================
# 1. Load data
# ============================================================
print("\n[1] Loading data...")
data = prepare_edwards_data('real_data/edwards_afb/', study_bounds=STUDY_BOUNDS)
grav = data['gravity']
bwells = data['basement_wells']
dgrid = data['depth_grid']

gravity_corrected = grav['isostatic'] - data['regional_correction']
print(f"  Stations: {len(grav['x'])}, Regional correction: {data['regional_correction']:.2f} mGal")

# ============================================================
# 2. Pick 3 boreholes for Stage 1 (lambda calibration)
# ============================================================
print("\n[2] Selecting 3 boreholes for lambda calibration...")

# Need wells with varying depths: shallow, medium, deep
# Filter: wells with depth between 50m-700m (exclude surface outcrops and outliers)
mask_usable = (bwells['depth_m'] > 50) & (bwells['depth_m'] < 700)
usable_depths = bwells['depth_m'][mask_usable]
usable_x = bwells['x'][mask_usable]
usable_y = bwells['y'][mask_usable]
usable_ids = np.array(bwells['well_id'])[mask_usable]

# Pick 3 wells: near 25th percentile, median, 75th percentile of depth
sorted_idx = np.argsort(usable_depths)
n_usable = len(sorted_idx)
pick_indices = [sorted_idx[n_usable//4], sorted_idx[n_usable//2], sorted_idx[3*n_usable//4]]

cal_x = usable_x[pick_indices]
cal_y = usable_y[pick_indices]
cal_depths = usable_depths[pick_indices]
cal_ids = usable_ids[pick_indices]

print(f"  Selected calibration wells:")
for i, (wid, d) in enumerate(zip(cal_ids, cal_depths)):
    print(f"    {i+1}. {wid}: depth = {d:.1f} m")

# ============================================================
# 3. Get observed gravity at each calibration borehole
# ============================================================
print("\n[3] Getting observed gravity at calibration boreholes...")

def gravity_at_point(px, py, obs_x, obs_y, gravity, radius_m=1500):
    """Average gravity at observations within radius of point."""
    dists = np.sqrt((obs_x - px)**2 + (obs_y - py)**2)
    mask = dists < radius_m
    if mask.sum() == 0:
        # Expand radius if nothing found
        mask = dists < radius_m * 3
    if mask.sum() == 0:
        return gravity[np.argmin(dists)]
    weights = 1.0 / (dists[mask] + 100)  # inverse distance weighting
    return np.sum(weights * gravity[mask]) / np.sum(weights)

cal_gravity = np.array([
    gravity_at_point(cal_x[i], cal_y[i], grav['x'], grav['y'], gravity_corrected)
    for i in range(3)
])

print(f"  Observed gravity at wells:")
for i in range(3):
    print(f"    Well {i+1}: depth={cal_depths[i]:.0f}m, gravity={cal_gravity[i]:.3f} mGal")

# ============================================================
# 4. STAGE 1: Estimate lambda via slab approximation
# ============================================================
print("\n[4] STAGE 1: Estimating lambda from boreholes...")

# Bouguer slab formula with exponential density compaction:
# Δg(z) = 2πG × Δρ₀ × (1 - exp(-λz)) / λ
# where Δρ₀ is in kg/m³, z in m, Δg in mGal
# Conversion: 1 mGal = 1e-5 m/s², so multiply by 1e5 for mGal
# 2πG = 2π × 6.67e-11 = 4.19e-10 m³/(kg·s²)
# For density contrast -500 kg/m³ at depth 500m with lambda 0.0003:
# Δρ_eff = -500 × (1-exp(-0.15))/0.15 = -500 × 0.1393/0.15 ≈ -464 kg/m³
# Δg = 2π × 6.67e-11 × 500 × (-464) × 1e5 = -9.7 mGal

G = 6.67430e-11
TWO_PI_G_MGAL = 2 * np.pi * G * 1e5  # converts to mGal when density in kg/m³ and depth in m

def slab_gravity(depth, drho_0, lam):
    """Bouguer slab gravity with exponential density compaction."""
    if lam * depth < 1e-6:
        return TWO_PI_G_MGAL * drho_0 * depth  # avoid numerical issue
    return TWO_PI_G_MGAL * drho_0 * (1 - np.exp(-lam * depth)) / lam

# Grid search over lambda
lambda_grid = np.linspace(0.00005, 0.002, 200)
misfits = []

for lam in lambda_grid:
    predicted = np.array([slab_gravity(d, DRHO_0, lam) for d in cal_depths])
    misfit = np.sum((cal_gravity - predicted) ** 2)
    misfits.append(misfit)

misfits = np.array(misfits)
best_lambda = lambda_grid[np.argmin(misfits)]

# Compute uncertainty from misfit curvature (approx 1-sigma from chi^2 + 1)
min_misfit = misfits.min()
within_1sigma = lambda_grid[misfits < min_misfit + np.std(cal_gravity)**2]
lambda_low = within_1sigma.min() if len(within_1sigma) > 0 else best_lambda
lambda_high = within_1sigma.max() if len(within_1sigma) > 0 else best_lambda

print(f"\n  Best lambda: {best_lambda:.6f} /m")
print(f"  Approximate 1-sigma range: [{lambda_low:.6f}, {lambda_high:.6f}]")
print(f"  (USGS equivalent: ~0.00065)")

# Plot misfit curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(lambda_grid, misfits, 'b-')
axes[0].axvline(best_lambda, color='red', linewidth=2, label=f'Best λ={best_lambda:.6f}')
axes[0].set_xlabel('Lambda (1/m)')
axes[0].set_ylabel('Misfit')
axes[0].set_title('Stage 1: Lambda Misfit Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot gravity vs depth: observed vs model
depth_plot = np.linspace(0, 1000, 100)
model_plot = np.array([slab_gravity(z, DRHO_0, best_lambda) for z in depth_plot])
axes[1].plot(depth_plot, model_plot, 'b-', label=f'Slab model, λ={best_lambda:.6f}')
axes[1].scatter(cal_depths, cal_gravity, c='red', s=80, zorder=5,
                label='Calibration boreholes')
for i in range(3):
    axes[1].annotate(f'  W{i+1}', (cal_depths[i], cal_gravity[i]), fontsize=9)
axes[1].set_xlabel('Well Depth (m)')
axes[1].set_ylabel('Gravity Anomaly (mGal)')
axes[1].set_title('Stage 1: Calibration Fit')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

out_dir = 'results/exp_edwards_2stage/'
os.makedirs(out_dir, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'stage1_lambda_calibration.png'), dpi=150)
plt.close(fig)
print(f"  Saved: {out_dir}stage1_lambda_calibration.png")

# ============================================================
# 5. STAGE 2: Depth-only MCMC with estimated lambda
# ============================================================
print(f"\n[5] STAGE 2: Running depth-only MCMC with lambda={best_lambda:.6f}")

# Grid
x_max = grav['x'].max()
y_max = grav['y'].max()
block_x_edges = np.linspace(0, x_max, NX + 1)
block_y_edges = np.linspace(0, y_max, NY + 1)

# Subsample gravity
obs_x, obs_y, obs_gravity, n_per_cell = subsample_gravity(
    grav['x'], grav['y'], gravity_corrected, SUBSAMPLE_SPACING
)
print(f"  Observations: {len(obs_x)} stations")
print(f"  Grid: {NX}x{NY} blocks ({(block_x_edges[1]-block_x_edges[0])/1000:.1f} km)")

# ============================================================
# DATA DIAGNOSTIC FIGURES (before MCMC)
# ============================================================
print("\n  Generating data diagnostic figures...")

# Load original gravity stations (before study-area clipping) for context
from src.data_loader import load_gravity_data, convert_to_utm
grav_all = load_gravity_data('real_data/edwards_afb/gravity_data.csv')
e_all, n_all = convert_to_utm(grav_all['lon'], grav_all['lat'])
x_all = e_all - data['origin_easting']
y_all = n_all - data['origin_northing']

# Figure: Data processing steps
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: All stations (original)
ax = axes[0, 0]
ax.scatter(x_all/1000, y_all/1000, c=grav_all['isostatic'], cmap='RdYlBu_r',
           s=5, alpha=0.6, vmin=-45, vmax=-4)
ax.set_title(f'Step 1: All {len(x_all)} Original Gravity Stations\n(color = isostatic residual, mGal)')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_aspect('equal')
# Study area
sx_min = 0; sx_max = grav['x'].max()/1000
sy_min = 0; sy_max = grav['y'].max()/1000
ax.add_patch(plt.Rectangle((sx_min, sy_min), sx_max-sx_min, sy_max-sy_min,
                            linewidth=2, edgecolor='black', facecolor='none',
                            linestyle='--', label='Study area'))
ax.legend(loc='upper left')

# Panel 2: Stations in study area after clipping
ax = axes[0, 1]
ax.scatter(grav['x']/1000, grav['y']/1000, c=gravity_corrected, cmap='RdYlBu_r',
           s=10, alpha=0.7, vmin=-30, vmax=15)
ax.set_title(f'Step 2: {len(grav["x"])} Stations in Study Area\n(regional correction applied: {data["regional_correction"]:.2f} mGal)')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_aspect('equal')
# Grid
for xe in block_x_edges:
    ax.axvline(xe/1000, color='gray', linewidth=0.3, alpha=0.5)
for ye in block_y_edges:
    ax.axhline(ye/1000, color='gray', linewidth=0.3, alpha=0.5)

# Panel 3: Subsampled stations (what we actually use)
ax = axes[1, 0]
sc = ax.scatter(obs_x/1000, obs_y/1000, c=obs_gravity, cmap='RdYlBu_r',
                s=50, alpha=0.9, vmin=-30, vmax=15,
                edgecolors='black', linewidths=0.5)
plt.colorbar(sc, ax=ax, label='Gravity (mGal)')
ax.set_title(f'Step 3: {len(obs_x)} Subsampled Stations (spacing={SUBSAMPLE_SPACING}m)\nTHESE are used in MCMC')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_aspect('equal')
for xe in block_x_edges:
    ax.axvline(xe/1000, color='gray', linewidth=0.3, alpha=0.5)
for ye in block_y_edges:
    ax.axhline(ye/1000, color='gray', linewidth=0.3, alpha=0.5)

# Panel 4: Wells + grid
ax = axes[1, 1]
# All basement wells
ax.scatter(bwells['x']/1000, bwells['y']/1000,
           c=bwells['depth_m'], cmap='viridis_r', s=40, marker='o',
           edgecolors='black', linewidths=0.5, label=f'Basement wells ({len(bwells["x"])})',
           vmin=0, vmax=700, zorder=3)
# 3 calibration wells
ax.scatter(cal_x/1000, cal_y/1000, s=250, marker='*', c='red',
           edgecolors='black', linewidths=1.5, zorder=5,
           label='Stage 1 calibration wells (3)')
# Label calibration wells
for i in range(3):
    ax.annotate(f' W{i+1}\n {cal_depths[i]:.0f}m',
                (cal_x[i]/1000, cal_y[i]/1000),
                fontsize=10, fontweight='bold', zorder=6)
# Colorbar for depths
sc = ax.collections[0]
plt.colorbar(sc, ax=ax, label='Well depth to bedrock (m)')
# Grid
for xe in block_x_edges:
    ax.axvline(xe/1000, color='gray', linewidth=0.3, alpha=0.5)
for ye in block_y_edges:
    ax.axhline(ye/1000, color='gray', linewidth=0.3, alpha=0.5)
ax.set_title(f'Step 4: Boreholes + Block Grid ({NX}x{NY}={NX*NY} blocks)\nRed stars = Stage 1 calibration wells')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=9)

fig.suptitle('Edwards AFB — Data Processing Pipeline', fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'data_pipeline.png'), dpi=150)
plt.close(fig)
print(f"  Saved: {out_dir}data_pipeline.png")

# Station-per-block heatmap
fig, ax = plt.subplots(figsize=(12, 8))
station_counts = np.zeros((NX, NY))
for sx, sy in zip(obs_x, obs_y):
    ix = int(np.searchsorted(block_x_edges, sx) - 1)
    iy = int(np.searchsorted(block_y_edges, sy) - 1)
    if 0 <= ix < NX and 0 <= iy < NY:
        station_counts[ix, iy] += 1

im = ax.pcolormesh(block_x_edges/1000, block_y_edges/1000, station_counts.T,
                    cmap='YlOrRd', shading='flat')
plt.colorbar(im, ax=ax, label='Stations per block (after subsampling)')

# Annotate each cell
for ix in range(NX):
    for iy in range(NY):
        cx = (block_x_edges[ix] + block_x_edges[ix+1]) / 2000
        cy = (block_y_edges[iy] + block_y_edges[iy+1]) / 2000
        cnt = int(station_counts[ix, iy])
        color = 'white' if cnt > 3 else 'black'
        ax.text(cx, cy, str(cnt), ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

# Overlay boreholes
ax.scatter(bwells['x']/1000, bwells['y']/1000, c='blue', s=20, marker='^',
           edgecolors='white', linewidths=0.5, zorder=3, label='Basement wells')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_title(f'Station Count per Block (total subsampled: {len(obs_x)})')
ax.set_aspect('equal')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'stations_per_block.png'), dpi=150)
plt.close(fig)
print(f"  Saved: {out_dir}stations_per_block.png")

# Initial depths from USGS
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

# Density function with estimated lambda
density_func = make_density_func('exponential', drho_0=DRHO_0, lam=best_lambda)

t0 = time.time()
result = run_mcmc_3d(
    obs_x=obs_x, obs_y=obs_y, gravity_obs=obs_gravity,
    block_x_edges=block_x_edges, block_y_edges=block_y_edges,
    density_func=density_func, noise_std=NOISE_STD,
    n_iterations=N_ITERATIONS, step_size=STEP_SIZE,
    depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
    smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
    initial_depths=initial_depths, seed=42, verbose=True
)
elapsed = time.time() - t0
print(f"\n  Runtime: {elapsed/60:.1f} min")

posterior = process_chain_3d(result, burn_in_frac=0.5)
mean_depths = posterior['mean']
std_depths = posterior['std']

# ============================================================
# 6. Validation against held-out wells
# ============================================================
print("\n[6] Validation against wells...")

# Use ALL 114 wells (even the 3 calibration ones — they're not "used as constraints" in Stage 2)
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
        ci5, ci95 = posterior['ci_5'][ix, iy], posterior['ci_95'][ix, iy]
        ci2_5, ci97_5 = posterior['ci_2_5'][ix, iy], posterior['ci_97_5'][ix, iy]
        errors.append(est - well_depth)
        if ci5 <= well_depth <= ci95: covered_90 += 1
        if ci2_5 <= well_depth <= ci97_5: covered_95 += 1
        n_val += 1

errors = np.array(errors)
rms = np.sqrt(np.mean(errors**2))
mae = np.mean(np.abs(errors))
bias = np.mean(errors)

print(f"\n{'='*60}")
print(f"RESULTS — Edwards AFB 2-Stage Inversion")
print(f"{'='*60}")
print(f"\nStage 1 (Lambda Calibration):")
print(f"  Calibration wells: 3 (depths {cal_depths.min():.0f}-{cal_depths.max():.0f} m)")
print(f"  Estimated lambda: {best_lambda:.6f} /m")
print(f"  Method: Bouguer slab approximation + grid search")

print(f"\nStage 2 (Depth MCMC):")
print(f"  Acceptance rate:    {result['acceptance_rate']*100:.1f}%")
print(f"  Posterior samples:  {posterior['n_samples']}")
print(f"  Depth range:        {mean_depths.min():.0f} - {mean_depths.max():.0f} m")
print(f"  Mean uncertainty:   {std_depths.mean():.0f} m")

print(f"\nValidation ({n_val} wells):")
print(f"  RMS error:        {rms:.1f} m")
print(f"  MAE:              {mae:.1f} m")
print(f"  Bias:             {bias:.1f} m")
print(f"  90% CI coverage:  {covered_90}/{n_val} ({covered_90/n_val*100:.0f}%)")
print(f"  95% CI coverage:  {covered_95}/{n_val} ({covered_95/n_val*100:.0f}%)")

# Save
np.savez(os.path.join(out_dir, 'results_data.npz'),
         best_lambda=best_lambda, cal_depths=cal_depths, cal_gravity=cal_gravity,
         mean_depths=mean_depths, std_depths=std_depths,
         ci_5=posterior['ci_5'], ci_95=posterior['ci_95'],
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         rms=rms, mae=mae, bias=bias, coverage_90=covered_90/n_val, coverage_95=covered_95/n_val)

# ============================================================
# Plots
# ============================================================
# Depth + uncertainty
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
im0 = axes[0].pcolormesh(block_x_edges/1000, block_y_edges/1000, mean_depths.T,
                          cmap='viridis_r', shading='flat')
plt.colorbar(im0, ax=axes[0], label='Depth (m)')
axes[0].set_title(f'MCMC Posterior Mean Depth (λ={best_lambda:.5f})')
axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)'); axes[0].set_aspect('equal')

# Plot calibration wells
for i in range(3):
    axes[0].plot(cal_x[i]/1000, cal_y[i]/1000, 'r^', markersize=12, markeredgecolor='black')

im1 = axes[1].pcolormesh(block_x_edges/1000, block_y_edges/1000, std_depths.T,
                          cmap='hot_r', shading='flat')
plt.colorbar(im1, ax=axes[1], label='Std Dev (m)')
axes[1].set_title('Posterior Uncertainty')
axes[1].set_xlabel('X (km)'); axes[1].set_ylabel('Y (km)'); axes[1].set_aspect('equal')

fig.suptitle(f'Edwards AFB — 2-Stage Inversion (RMS={rms:.0f}m, 90%cov={covered_90/n_val*100:.0f}%)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'depth_and_uncertainty.png'), dpi=150)
plt.close(fig)

# Validation scatter
fig, ax = plt.subplots(figsize=(8, 8))
well_d = []; mcmc_d = []; mcmc_s = []
for (ix, iy), wd in all_wells.items():
    if 0 <= ix < NX and 0 <= iy < NY:
        well_d.append(wd); mcmc_d.append(mean_depths[ix, iy]); mcmc_s.append(std_depths[ix, iy])
well_d = np.array(well_d); mcmc_d = np.array(mcmc_d); mcmc_s = np.array(mcmc_s)
ax.errorbar(well_d, mcmc_d, yerr=mcmc_s, fmt='o', markersize=5, capsize=3, alpha=0.7, color='steelblue')
lim = max(well_d.max(), mcmc_d.max()) * 1.1 + 10
ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='1:1 line')
ax.set_xlabel('Well Depth (m)'); ax.set_ylabel('MCMC Estimated Depth (m)')
ax.set_title(f'Validation (RMS={rms:.0f}m, n={n_val})')
ax.legend(); ax.set_aspect('equal'); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'validation_scatter.png'), dpi=150)
plt.close(fig)

print(f"\nAll saved to: {out_dir}")
print("=" * 60)
