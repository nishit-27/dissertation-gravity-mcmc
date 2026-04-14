"""
Cauvery Basin (India) — 3D MCMC Gravity Inversion for Basement Depth
=====================================================================
Real-data application using ICGEM XGM2019e Bouguer anomaly over the onshore
Pudukkottai–Thanjavur sub-basin (Ariyalur-Pondicherry system), Southern
Cauvery Basin.

Bbox:   9.7°–10.7°N, 78.6°–79.6°E (~111 × 109 km, 95% onshore)
Data:   ICGEM XGM2019e_2159 Bouguer anomaly, 0.05° grid = 441 stations
Model:  10×10 MCMC block grid, exponential compaction, FIXED density
Param:  Δρ₀ = −550 kg/m³, λ = 5.0×10⁻⁴ /m (fit to Ganguli & Pal 2023 Table 2)
Bench:  Ganguli & Pal (2023), Front. Earth Sci., 10.3389/feart.2023.1190106

Adapted from run_3d_50k_smooth_test.py (Exp 7 — synthetic) and
run_edwards_fixed_lambda.py (Edwards AFB real-data pipeline).
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mcmc_inversion import run_mcmc_3d, process_chain_3d
from src.utils import make_density_func

# ============================================================
# Configuration
# ============================================================
DATA_CSV = 'real_data/cauvery_south/gravity_stations.csv'

# MCMC block grid (matches Exp 7)
NX, NY = 10, 10
N_ITERATIONS = 50_000

# Density — literature-calibrated (Ganguli & Pal 2023 Table 2 +
# Rao et al. 2019 Cauvery petrophysics; fit to exponential)
DRHO_0 = -550.0       # kg/m³ (surface contrast: sediment 2.17 vs basement 2.72 g/cc)
LAMBDA = 5.0e-4       # 1/m (≈0.5 km⁻¹)

# MCMC params (match Exp 7)
STEP_SIZE = 150.0
SMOOTHNESS_WEIGHT = 1e-6
N_SUBLAYERS = 10

# Depth bounds (Ganguli & Pal 2023 report 3000–5400 m in adjacent depocenters)
DEPTH_MIN = 0.0
DEPTH_MAX = 6000.0

# Data noise (mGal) — XGM2019e commission error + unmodelled short-wavelength
NOISE_STD = 1.0

OUT_DIR = 'results/exp_cauvery_real'

# ============================================================
# 1. Load ICGEM XGM2019e Bouguer anomaly
# ============================================================
print("=" * 65)
print("CAUVERY BASIN — 3D MCMC INVERSION (Real Data, Fixed Density)")
print("=" * 65)
print(f"Data: {DATA_CSV}")
print(f"Density: Δρ₀ = {DRHO_0} kg/m³, λ = {LAMBDA} /m (fixed)")

print("\n[1/5] Loading ICGEM gravity data...")
arr = np.loadtxt(DATA_CSV, delimiter=',', skiprows=1)
lon, lat, bg = arr[:, 0], arr[:, 1], arr[:, 2]
print(f"  Stations: {len(bg)}")
print(f"  Lon: {lon.min():.3f} to {lon.max():.3f}  Lat: {lat.min():.3f} to {lat.max():.3f}")
print(f"  Raw Bouguer: {bg.min():.2f} to {bg.max():.2f} mGal (mean={bg.mean():.2f})")

# ============================================================
# 2. Project lat/lon to local meters (equirectangular at mid-lat)
# ============================================================
R = 6371000.0
lat0 = (lat.min() + lat.max()) / 2.0
cos_lat0 = np.cos(np.deg2rad(lat0))
obs_x = (lon - lon.min()) * np.deg2rad(1.0) * R * cos_lat0
obs_y = (lat - lat.min()) * np.deg2rad(1.0) * R
print(f"  Local X: 0 to {obs_x.max()/1000:.2f} km  Y: 0 to {obs_y.max()/1000:.2f} km")

# ============================================================
# 3. Regional correction (remove bilinear trend)
#    Long-wavelength Bouguer reflects Moho/crust, not basin. Remove via
#    least-squares bilinear plane fit. The residual is the basin signal.
# ============================================================
A = np.column_stack([np.ones_like(obs_x), obs_x, obs_y])
coef, *_ = np.linalg.lstsq(A, bg, rcond=None)
bg_regional = A @ coef
bg_residual = bg - bg_regional
print(f"\n[2/5] Regional correction (bilinear detrend)...")
print(f"  Regional plane: {coef[0]:.2f} + {coef[1]*1e3:.4f}·x_km + {coef[2]*1e3:.4f}·y_km")
print(f"  Residual Bouguer: {bg_residual.min():.2f} to {bg_residual.max():.2f} mGal")
print(f"  Residual std: {bg_residual.std():.2f} mGal")

# Calibrate so shallow-basement areas (highest residual) → depth ~ 0
# i.e. gravity effect of basin = residual − max(residual); all ≤ 0
gravity_obs = bg_residual - bg_residual.max()
print(f"  Calibrated (basin signal): {gravity_obs.min():.2f} to {gravity_obs.max():.2f} mGal")

# ============================================================
# 4. Set up 10×10 MCMC block grid
# ============================================================
print(f"\n[3/5] Setting up {NX}x{NY} block grid...")
block_x_edges = np.linspace(0, obs_x.max(), NX + 1)
block_y_edges = np.linspace(0, obs_y.max(), NY + 1)
bx_km = (block_x_edges[1] - block_x_edges[0]) / 1000
by_km = (block_y_edges[1] - block_y_edges[0]) / 1000
print(f"  Block size: {bx_km:.2f} × {by_km:.2f} km  |  Total blocks: {NX*NY}")

# Initial depths from a rough relationship: residual mGal → depth
# Using Bouguer slab ≈ 2πGΔρ·z ≈ 0.042·Δρ·z (mGal). For Δρ=−350, z(km)≈|g|/14.7.
# But we want a uniform sensible starting point inside bounds.
initial_depths = np.full((NX, NY), 2000.0)

density_func = make_density_func('exponential', drho_0=DRHO_0, lam=LAMBDA)

# ============================================================
# 5. Run MCMC
# ============================================================
print(f"\n[4/5] Running MCMC ({N_ITERATIONS:,} iterations, fixed density)...")
t0 = time.time()

result = run_mcmc_3d(
    obs_x=obs_x,
    obs_y=obs_y,
    gravity_obs=gravity_obs,
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
    verbose=True,
)
elapsed = time.time() - t0
print(f"\n  Runtime: {elapsed/60:.1f} min")
print(f"  Acceptance: {result['acceptance_rate']*100:.1f}%")

# ============================================================
# 6. Process posterior + save + plot
# ============================================================
print(f"\n[5/5] Processing posterior...")
posterior = process_chain_3d(result, burn_in_frac=0.5, thin=1)
mean_d = posterior['mean']
std_d = posterior['std']

print(f"  Posterior samples: {posterior['n_samples']}")
print(f"  Depth range: {mean_d.min():.0f} – {mean_d.max():.0f} m")
print(f"  Mean uncertainty (std): {std_d.mean():.0f} m")

# Ganguli & Pal 2023 benchmark comparison
print(f"\n  Benchmark (Ganguli & Pal 2023, adjacent 78-79E/9-10N):")
print(f"    Their depocenter range: 3000 – 5400 m")
print(f"    Our depocenter (deepest block): {mean_d.max():.0f} ± {std_d[np.unravel_index(mean_d.argmax(), mean_d.shape)]:.0f} m")

os.makedirs(OUT_DIR, exist_ok=True)

# Save everything for later analysis
np.savez(os.path.join(OUT_DIR, 'results_data.npz'),
         mean_depths=mean_d, std_depths=std_d,
         ci_5=posterior['ci_5'], ci_95=posterior['ci_95'],
         ci_2_5=posterior['ci_2_5'], ci_97_5=posterior['ci_97_5'],
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
         lon=lon, lat=lat, bouguer_raw=bg, bouguer_residual=bg_residual,
         regional_coef=coef,
         drho_0=DRHO_0, lam=LAMBDA,
         acceptance_rate=result['acceptance_rate'],
         runtime_min=elapsed/60)

# Depth + uncertainty maps
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
im0 = axes[0].pcolormesh(block_x_edges/1000, block_y_edges/1000, mean_d.T,
                         cmap='viridis_r', shading='flat')
plt.colorbar(im0, ax=axes[0], label='Depth (m)')
axes[0].set_title(f'Posterior Mean Basement Depth\nrange {mean_d.min():.0f}–{mean_d.max():.0f} m')
axes[0].set_xlabel('X (km)')
axes[0].set_ylabel('Y (km)')
axes[0].set_aspect('equal')

im1 = axes[1].pcolormesh(block_x_edges/1000, block_y_edges/1000, std_d.T,
                         cmap='hot_r', shading='flat')
plt.colorbar(im1, ax=axes[1], label='Std Dev (m)')
axes[1].set_title(f'Posterior Uncertainty (std)\nmean {std_d.mean():.0f} m')
axes[1].set_xlabel('X (km)')
axes[1].set_ylabel('Y (km)')
axes[1].set_aspect('equal')

fig.suptitle(f'Cauvery Basin — 3D MCMC Inversion (XGM2019e, Δρ₀={DRHO_0}, λ={LAMBDA})', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'depth_and_uncertainty.png'), dpi=150)
plt.close(fig)

# Bouguer anomaly + residual maps
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
sc = axes[0].scatter(lon, lat, c=bg, cmap='RdBu_r', s=40)
plt.colorbar(sc, ax=axes[0], label='mGal')
axes[0].set_title('Raw Bouguer (XGM2019e)')
axes[0].set_xlabel('Lon °E'); axes[0].set_ylabel('Lat °N')

sc = axes[1].scatter(lon, lat, c=bg_regional, cmap='RdBu_r', s=40)
plt.colorbar(sc, ax=axes[1], label='mGal')
axes[1].set_title('Regional (bilinear plane)')
axes[1].set_xlabel('Lon °E')

sc = axes[2].scatter(lon, lat, c=bg_residual, cmap='RdBu_r', s=40)
plt.colorbar(sc, ax=axes[2], label='mGal')
axes[2].set_title('Residual (basin signal)')
axes[2].set_xlabel('Lon °E')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'bouguer_decomposition.png'), dpi=150)
plt.close(fig)

# Misfit trace
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(result['all_misfits'], linewidth=0.3, alpha=0.7)
ax.axvline(x=N_ITERATIONS // 2, color='red', linestyle='--', label='Burn-in cutoff')
ax.set_xlabel('Iteration'); ax.set_ylabel('Misfit')
ax.set_title(f'MCMC Convergence  (acceptance {result["acceptance_rate"]*100:.1f}%)')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'misfit_trace.png'), dpi=150)
plt.close(fig)

# Per-block depth table
print(f"\n  Per-Block Depth (m) ± std:")
print(f"  {'':>8s}", end='')
for j in range(NY):
    print(f"   Y{j+1:>2d}   ", end='')
print()
for i in range(NX):
    print(f"  X{i+1:>2d}", end='  ')
    for j in range(NY):
        print(f"{mean_d[i,j]:5.0f}±{std_d[i,j]:3.0f} ", end='')
    print()

print(f"\n{'='*65}")
print(f"DONE — results saved to: {OUT_DIR}/")
print(f"{'='*65}")
