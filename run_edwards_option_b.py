"""
Edwards AFB — Option B: Spatially-Varying Regional Correction
==============================================================
Improved preprocessing + 2-stage inversion.

Pipeline:
  Step A: Fit polynomial surface to isostatic gravity at shallow wells (depth<10m)
          → this is the 'basement gravity' (smooth regional trend)
  Step B: Subtract from all gravity → proper basin-fill signal
  Stage 1: Calibrate lambda using deep wells (now physically valid)
  Stage 2: Depth-only MCMC with estimated lambda
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import (
    prepare_edwards_data, subsample_gravity, assign_wells_to_blocks,
    load_gravity_data, convert_to_utm
)
from src.mcmc_inversion import run_mcmc_3d, process_chain_3d
from src.utils import make_density_func

# ============================================================
# Configuration
# ============================================================
STUDY_BOUNDS = {
    'lon_min': -117.95, 'lon_max': -117.50,
    'lat_min': 34.82,   'lat_max': 35.10,
}
NX, NY = 12, 10
N_ITERATIONS = 15000
SUBSAMPLE_SPACING = 3000
DRHO_0 = -500.0
STEP_SIZE = 50.0
NOISE_STD = 1.0
DEPTH_MIN = 0.0
DEPTH_MAX = 1500.0
SMOOTHNESS_WEIGHT = 1e-6
N_SUBLAYERS = 10
POLY_DEGREE = 2  # Quadratic polynomial for regional trend

print("=" * 60)
print("Edwards AFB — Option B: Spatially-Varying Regional Correction")
print("=" * 60)

out_dir = 'results/exp_edwards_option_b/'
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# 1. Load data (no global correction yet — we'll do spatial)
# ============================================================
print("\n[1] Loading data...")
data = prepare_edwards_data('real_data/edwards_afb/', study_bounds=STUDY_BOUNDS)
grav = data['gravity']
bwells = data['basement_wells']
dgrid = data['depth_grid']

# Undo the global correction so we can apply spatial one
gravity_raw_iso = grav['isostatic']  # uncorrected isostatic residual
print(f"  Stations: {len(grav['x'])}, basement wells: {len(bwells['x'])}")

# ============================================================
# 2. Step A: Fit polynomial to gravity at shallow wells
# ============================================================
print(f"\n[2] Step A: Fitting {POLY_DEGREE}-degree polynomial to shallow wells...")

# Shallow wells = bedrock effectively at surface (depth < 10m)
shallow_mask = bwells['depth_m'] < 10.0
shallow_x = bwells['x'][shallow_mask]
shallow_y = bwells['y'][shallow_mask]
print(f"  Shallow wells (depth<10m): {shallow_mask.sum()}")

# Get gravity at each shallow well (interpolate from nearest stations)
def gravity_at_point(px, py, obs_x, obs_y, gravity, radius_m=1500):
    dists = np.sqrt((obs_x - px)**2 + (obs_y - py)**2)
    mask = dists < radius_m
    if mask.sum() == 0:
        mask = dists < radius_m * 3
    if mask.sum() == 0:
        return gravity[np.argmin(dists)]
    weights = 1.0 / (dists[mask] + 100)
    return np.sum(weights * gravity[mask]) / np.sum(weights)

shallow_gravity = np.array([
    gravity_at_point(shallow_x[i], shallow_y[i], grav['x'], grav['y'], gravity_raw_iso)
    for i in range(shallow_mask.sum())
])
print(f"  Gravity at shallow wells: {shallow_gravity.min():.1f} to {shallow_gravity.max():.1f} mGal")

# Build design matrix for polynomial: 1, x, y, x^2, xy, y^2 (degree 2)
def build_poly_design(x, y, degree=2):
    """Return design matrix for polynomial of given degree."""
    terms = [np.ones_like(x)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            terms.append((x ** (d - i)) * (y ** i))
    return np.column_stack(terms)

# Fit: gravity ≈ polynomial(x, y)
A_shallow = build_poly_design(shallow_x, shallow_y, POLY_DEGREE)
coeffs, residuals, rank, sv = np.linalg.lstsq(A_shallow, shallow_gravity, rcond=None)

# Fit quality
fitted = A_shallow @ coeffs
fit_rms = np.sqrt(np.mean((shallow_gravity - fitted) ** 2))
print(f"  Polynomial fit RMS (at shallow wells): {fit_rms:.2f} mGal")
print(f"  Coefficients: {coeffs}")

# ============================================================
# 3. Step B: Subtract polynomial from all gravity
# ============================================================
print("\n[3] Step B: Applying spatially-varying correction...")
A_all = build_poly_design(grav['x'], grav['y'], POLY_DEGREE)
regional_field = A_all @ coeffs
gravity_corrected = gravity_raw_iso - regional_field

print(f"  Before correction: gravity range [{gravity_raw_iso.min():.1f}, {gravity_raw_iso.max():.1f}] mGal")
print(f"  After correction:  gravity range [{gravity_corrected.min():.1f}, {gravity_corrected.max():.1f}] mGal")

# Verify: corrected gravity at shallow wells should be ~0
shallow_corrected = np.array([
    gravity_at_point(shallow_x[i], shallow_y[i], grav['x'], grav['y'], gravity_corrected)
    for i in range(shallow_mask.sum())
])
print(f"  Corrected gravity at shallow wells: mean={shallow_corrected.mean():.3f}, std={shallow_corrected.std():.3f} mGal (should be ~0)")

# ============================================================
# 4. Visualize regional field + corrected gravity
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Raw gravity
ax = axes[0]
sc = ax.scatter(grav['x']/1000, grav['y']/1000, c=gravity_raw_iso, cmap='RdYlBu_r',
                s=6, alpha=0.7, vmin=-45, vmax=-4)
plt.colorbar(sc, ax=ax, label='Isostatic Residual (mGal)')
ax.scatter(shallow_x/1000, shallow_y/1000, c='black', s=20, marker='o',
           edgecolors='white', linewidths=0.5, label=f'Shallow wells ({shallow_mask.sum()})')
ax.set_title('Step A: Raw Isostatic Residual')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
ax.legend(fontsize=9)

# Regional field (polynomial)
ax = axes[1]
# Evaluate on grid for visualization
xv = np.linspace(0, grav['x'].max(), 100)
yv = np.linspace(0, grav['y'].max(), 100)
Xv, Yv = np.meshgrid(xv, yv)
A_grid = build_poly_design(Xv.flatten(), Yv.flatten(), POLY_DEGREE)
regional_grid = (A_grid @ coeffs).reshape(Xv.shape)
im = ax.contourf(Xv/1000, Yv/1000, regional_grid, levels=20, cmap='RdYlBu_r')
plt.colorbar(im, ax=ax, label='Regional Field (mGal)')
ax.scatter(shallow_x/1000, shallow_y/1000, c='black', s=20, marker='o',
           edgecolors='white', linewidths=0.5)
ax.set_title(f'Fitted Polynomial (degree {POLY_DEGREE})')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')

# Corrected gravity
ax = axes[2]
sc = ax.scatter(grav['x']/1000, grav['y']/1000, c=gravity_corrected, cmap='RdYlBu_r',
                s=6, alpha=0.7, vmin=-25, vmax=5)
plt.colorbar(sc, ax=ax, label='Corrected Gravity (mGal)')
ax.scatter(shallow_x/1000, shallow_y/1000, c='black', s=20, marker='o',
           edgecolors='white', linewidths=0.5)
ax.set_title('Step B: Basin-Fill Gravity (corrected)')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')

fig.suptitle('Option B: Spatially-Varying Regional Correction', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'regional_correction.png'), dpi=150)
plt.close(fig)
print(f"  Saved: {out_dir}regional_correction.png")

# ============================================================
# 5. Stage 1: Pick deep wells, estimate lambda
# ============================================================
print("\n[4] Stage 1: Picking deep wells for lambda calibration...")

# Now pick DEEPEST wells (where basin signal dominates)
deep_mask = bwells['depth_m'] > 200.0
deep_depths = bwells['depth_m'][deep_mask]
deep_x = bwells['x'][deep_mask]
deep_y = bwells['y'][deep_mask]
deep_ids = np.array(bwells['well_id'])[deep_mask]
print(f"  Wells with depth>200m: {deep_mask.sum()}")

# Pick 5 deepest wells
sorted_idx = np.argsort(deep_depths)[::-1]  # descending
n_cal = min(5, len(sorted_idx))
cal_indices = sorted_idx[:n_cal]

cal_x = deep_x[cal_indices]
cal_y = deep_y[cal_indices]
cal_depths = deep_depths[cal_indices]
cal_ids = deep_ids[cal_indices]

cal_gravity = np.array([
    gravity_at_point(cal_x[i], cal_y[i], grav['x'], grav['y'], gravity_corrected)
    for i in range(n_cal)
])

print(f"  Calibration wells (deepest):")
for i in range(n_cal):
    print(f"    {i+1}. {cal_ids[i]}: depth={cal_depths[i]:.0f}m, corrected_gravity={cal_gravity[i]:.3f} mGal")

# ============================================================
# Stage 1 fit
# ============================================================
G = 6.67430e-11
TWO_PI_G_MGAL = 2 * np.pi * G * 1e5

def slab_gravity(depth, drho_0, lam):
    if lam * depth < 1e-6:
        return TWO_PI_G_MGAL * drho_0 * depth
    return TWO_PI_G_MGAL * drho_0 * (1 - np.exp(-lam * depth)) / lam

lambda_grid = np.linspace(0.00005, 0.002, 400)
misfits = np.array([
    np.sum((cal_gravity - np.array([slab_gravity(d, DRHO_0, lam) for d in cal_depths]))**2)
    for lam in lambda_grid
])
best_lambda = lambda_grid[np.argmin(misfits)]
min_misfit = misfits.min()

# 1-sigma range from misfit curvature
sigma_thresh = min_misfit + len(cal_depths) * np.std(cal_gravity)**2 * 0.5
within_1s = lambda_grid[misfits < sigma_thresh]
lambda_low = within_1s.min() if len(within_1s) > 0 else best_lambda
lambda_high = within_1s.max() if len(within_1s) > 0 else best_lambda

print(f"\n  Stage 1 result:")
print(f"    Best lambda: {best_lambda:.6f} /m")
print(f"    1-sigma range: [{lambda_low:.6f}, {lambda_high:.6f}]")
print(f"    (USGS equivalent: ~0.00065)")

# Stage 1 diagnostic plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(lambda_grid, misfits, 'b-')
axes[0].axvline(best_lambda, color='red', linewidth=2, label=f'Best λ={best_lambda:.6f}')
axes[0].axvline(0.00065, color='green', linewidth=1.5, linestyle='--', label='USGS equiv (0.00065)')
axes[0].set_xlabel('Lambda (1/m)')
axes[0].set_ylabel('Misfit')
axes[0].set_title('Stage 1: Lambda Misfit Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

depth_plot = np.linspace(0, 1000, 200)
model_plot = np.array([slab_gravity(z, DRHO_0, best_lambda) for z in depth_plot])
axes[1].plot(depth_plot, model_plot, 'b-', linewidth=2, label=f'Best fit (λ={best_lambda:.6f})')
# USGS reference
model_usgs = np.array([slab_gravity(z, -530, 0.00065) for z in depth_plot])
axes[1].plot(depth_plot, model_usgs, 'g--', linewidth=1, label='USGS values (-530, 0.00065)')
axes[1].scatter(cal_depths, cal_gravity, c='red', s=100, zorder=5, label='Calibration wells')
for i in range(n_cal):
    axes[1].annotate(f' W{i+1}', (cal_depths[i], cal_gravity[i]), fontsize=10)
axes[1].set_xlabel('Well Depth (m)')
axes[1].set_ylabel('Corrected Gravity (mGal)')
axes[1].set_title('Stage 1: Slab Model Fit')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'stage1_calibration.png'), dpi=150)
plt.close(fig)
print(f"  Saved: {out_dir}stage1_calibration.png")

# ============================================================
# 6. Setup grid and subsample
# ============================================================
x_max = grav['x'].max()
y_max = grav['y'].max()
block_x_edges = np.linspace(0, x_max, NX + 1)
block_y_edges = np.linspace(0, y_max, NY + 1)

obs_x, obs_y, obs_gravity, _ = subsample_gravity(
    grav['x'], grav['y'], gravity_corrected, SUBSAMPLE_SPACING
)
print(f"\n  Subsampled: {len(grav['x'])} -> {len(obs_x)} stations")

# Data pipeline figure
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# All original stations
grav_all = load_gravity_data('real_data/edwards_afb/gravity_data.csv')
e_all, n_all = convert_to_utm(grav_all['lon'], grav_all['lat'])
x_all = e_all - data['origin_easting']
y_all = n_all - data['origin_northing']
ax = axes[0, 0]
ax.scatter(x_all/1000, y_all/1000, c=grav_all['isostatic'], cmap='RdYlBu_r',
           s=5, alpha=0.6, vmin=-45, vmax=-4)
ax.add_patch(plt.Rectangle((0, 0), x_max/1000, y_max/1000,
                            linewidth=2, edgecolor='black', facecolor='none',
                            linestyle='--', label='Study area'))
ax.set_title(f'Step 1: All {len(x_all)} Original Stations')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
ax.legend()

# After regional correction (in study area)
ax = axes[0, 1]
ax.scatter(grav['x']/1000, grav['y']/1000, c=gravity_corrected, cmap='RdYlBu_r',
           s=10, alpha=0.7, vmin=-25, vmax=5)
ax.set_title(f'Step 2: {len(grav["x"])} Stations + Spatial Correction')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
for xe in block_x_edges:
    ax.axvline(xe/1000, color='gray', linewidth=0.3, alpha=0.5)
for ye in block_y_edges:
    ax.axhline(ye/1000, color='gray', linewidth=0.3, alpha=0.5)

# Subsampled stations
ax = axes[1, 0]
sc = ax.scatter(obs_x/1000, obs_y/1000, c=obs_gravity, cmap='RdYlBu_r',
                s=50, alpha=0.9, vmin=-25, vmax=5,
                edgecolors='black', linewidths=0.5)
plt.colorbar(sc, ax=ax, label='Gravity (mGal)')
ax.set_title(f'Step 3: {len(obs_x)} Subsampled — USED IN MCMC')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
for xe in block_x_edges:
    ax.axvline(xe/1000, color='gray', linewidth=0.3, alpha=0.5)
for ye in block_y_edges:
    ax.axhline(ye/1000, color='gray', linewidth=0.3, alpha=0.5)

# Wells + calibration highlighted
ax = axes[1, 1]
sc = ax.scatter(bwells['x']/1000, bwells['y']/1000,
                c=bwells['depth_m'], cmap='viridis_r', s=40, marker='o',
                edgecolors='black', linewidths=0.5, vmin=0, vmax=700,
                label=f'Basement wells ({len(bwells["x"])})', zorder=3)
ax.scatter(cal_x/1000, cal_y/1000, s=250, marker='*', c='red',
           edgecolors='black', linewidths=1.5, zorder=5,
           label=f'Stage 1 wells ({n_cal})')
for i in range(n_cal):
    ax.annotate(f' W{i+1}:{cal_depths[i]:.0f}m',
                (cal_x[i]/1000, cal_y[i]/1000),
                fontsize=9, fontweight='bold', zorder=6)
plt.colorbar(sc, ax=ax, label='Depth (m)')
for xe in block_x_edges:
    ax.axvline(xe/1000, color='gray', linewidth=0.3, alpha=0.5)
for ye in block_y_edges:
    ax.axhline(ye/1000, color='gray', linewidth=0.3, alpha=0.5)
ax.set_title(f'Step 4: Wells + {NX}x{NY} Block Grid')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=9)

fig.suptitle('Edwards AFB — Data Pipeline (Option B)', fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'data_pipeline.png'), dpi=150)
plt.close(fig)
print(f"  Saved: {out_dir}data_pipeline.png")

# ============================================================
# 7. Stage 2: Depth MCMC
# ============================================================
print(f"\n[5] Stage 2: Depth-only MCMC with lambda={best_lambda:.6f}")

# Initial depths from USGS
initial_depths = np.ones((NX, NY)) * (DEPTH_MIN + DEPTH_MAX) / 2.0
if len(dgrid['x']) > 0:
    for ix in range(NX):
        for iy in range(NY):
            mask = ((dgrid['x'] >= block_x_edges[ix]) &
                    (dgrid['x'] < block_x_edges[ix + 1]) &
                    (dgrid['y'] >= block_y_edges[iy]) &
                    (dgrid['y'] < block_y_edges[iy + 1]))
            if mask.sum() > 0:
                initial_depths[ix, iy] = np.clip(
                    np.mean(dgrid['depth_m'][mask]), DEPTH_MIN + 1, DEPTH_MAX - 1)

density_func = make_density_func('exponential', drho_0=DRHO_0, lam=best_lambda)

t0 = time.time()
result = run_mcmc_3d(
    obs_x=obs_x, obs_y=obs_y, gravity_obs=obs_gravity,
    block_x_edges=block_x_edges, block_y_edges=block_y_edges,
    density_func=density_func, noise_std=NOISE_STD,
    n_iterations=N_ITERATIONS, step_size=STEP_SIZE,
    depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
    smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
    initial_depths=initial_depths, seed=42, verbose=True)
elapsed = time.time() - t0
print(f"\n  Runtime: {elapsed/60:.1f} min")

posterior = process_chain_3d(result, burn_in_frac=0.5)
mean_depths = posterior['mean']
std_depths = posterior['std']

# ============================================================
# 8. Validation
# ============================================================
all_wells, _ = assign_wells_to_blocks(
    bwells['x'], bwells['y'], bwells['depth_m'],
    block_x_edges, block_y_edges)

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
print(f"RESULTS — Option B (Spatial Correction)")
print(f"{'='*60}")
print(f"Polynomial fit RMS:   {fit_rms:.2f} mGal")
print(f"Estimated lambda:     {best_lambda:.6f} /m (USGS: ~0.00065)")
print(f"\nStage 2:")
print(f"  Acceptance:         {result['acceptance_rate']*100:.1f}%")
print(f"  Samples:            {posterior['n_samples']}")
print(f"  Depth range:        {mean_depths.min():.0f} - {mean_depths.max():.0f} m")
print(f"  Mean uncertainty:   {std_depths.mean():.0f} m")
print(f"\nValidation ({n_val} wells):")
print(f"  RMS:                {rms:.1f} m")
print(f"  MAE:                {mae:.1f} m")
print(f"  Bias:               {bias:.1f} m")
print(f"  90% CI coverage:    {covered_90}/{n_val} ({covered_90/n_val*100:.0f}%)")
print(f"  95% CI coverage:    {covered_95}/{n_val} ({covered_95/n_val*100:.0f}%)")

np.savez(os.path.join(out_dir, 'results_data.npz'),
         best_lambda=best_lambda, poly_coeffs=coeffs,
         mean_depths=mean_depths, std_depths=std_depths,
         ci_5=posterior['ci_5'], ci_95=posterior['ci_95'],
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         rms=rms, mae=mae, bias=bias,
         coverage_90=covered_90/n_val, coverage_95=covered_95/n_val)

# Depth + uncertainty
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
im0 = axes[0].pcolormesh(block_x_edges/1000, block_y_edges/1000, mean_depths.T,
                          cmap='viridis_r', shading='flat')
plt.colorbar(im0, ax=axes[0], label='Depth (m)')
axes[0].set_title(f'MCMC Mean Depth (λ={best_lambda:.5f})')
axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)'); axes[0].set_aspect('equal')
for i in range(n_cal):
    axes[0].plot(cal_x[i]/1000, cal_y[i]/1000, 'r*', markersize=15, markeredgecolor='black')

im1 = axes[1].pcolormesh(block_x_edges/1000, block_y_edges/1000, std_depths.T,
                          cmap='hot_r', shading='flat')
plt.colorbar(im1, ax=axes[1], label='Std Dev (m)')
axes[1].set_title('Uncertainty')
axes[1].set_xlabel('X (km)'); axes[1].set_ylabel('Y (km)'); axes[1].set_aspect('equal')

fig.suptitle(f'Option B: RMS={rms:.0f}m, 90%cov={covered_90/n_val*100:.0f}%', fontsize=14)
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
ax.plot([0, lim], [0, lim], 'k--', label='1:1 line')
ax.set_xlabel('Well Depth (m)'); ax.set_ylabel('MCMC Depth (m)')
ax.set_title(f'Validation — RMS={rms:.0f}m, n={n_val}, 90%cov={covered_90/n_val*100:.0f}%')
ax.legend(); ax.set_aspect('equal'); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'validation_scatter.png'), dpi=150)
plt.close(fig)

print(f"\nAll saved to: {out_dir}")
print("=" * 60)
