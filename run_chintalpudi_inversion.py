"""
Chintalpudi Sub-Basin (Krishna–Godavari, India) — 3D MCMC Gravity Inversion
===========================================================================
Real-data benchmark. 61x41 grid of Bouguer gravity (1 km spacing) digitized
from published maps. Validated against Chakravarthi & Sundararajan (2007),
Geophysics 72(2), I23-I32, and ONGC borehole depth of 2.935 km.

Quick test: 10,000 iterations.
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
DATA_DIR = 'real_data/chintalpudi'
OUT_DIR = 'results/exp_chintalpudi_10k'

NX, NY = 10, 10
N_ITERATIONS = 10_000

# Exponential density (Chakravarthi & Sundararajan 2007 used similar)
DRHO_0 = -500.0
LAMBDA = 5.0e-4

STEP_SIZE = 150.0
SMOOTHNESS_WEIGHT = 1e-6
N_SUBLAYERS = 10

DEPTH_MIN = 0.0
DEPTH_MAX = 5000.0     # ONGC borehole = 2935 m; allow margin
NOISE_STD = 0.5        # mGal, typical digitization noise

print("=" * 70)
print("CHINTALPUDI SUB-BASIN — 3D MCMC INVERSION (Real Data, 10k)")
print("=" * 70)

# ============================================================
# 1. Load data
# ============================================================
print("\n[1/5] Loading Chintalpudi gridded data...")
xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))   # (40, 60)
yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))  # (40, 60)
bd = np.loadtxt(os.path.join(DATA_DIR, 'basement_depth.txt'))    # (41, 60)
print(f"  x_meshgrid: {xg.shape}, range {xg.min():.0f}–{xg.max():.0f} m")
print(f"  y_meshgrid: {yg.shape}, range {yg.min():.0f}–{yg.max():.0f} m")
print(f"  gravity:    {gv.shape}, range {gv.min():.2f}–{gv.max():.2f} mGal")
print(f"  basement:   {bd.shape}, range {bd.min():.0f}–{bd.max():.0f} m")

obs_x = xg.flatten()
obs_y = yg.flatten()
gravity_obs = gv.flatten()
print(f"  Stations: {len(obs_x)}")

# ============================================================
# 2. Block grid
# ============================================================
print(f"\n[2/5] Setting up {NX}x{NY} MCMC block grid...")
block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)
bx_km = (block_x_edges[1] - block_x_edges[0]) / 1000
by_km = (block_y_edges[1] - block_y_edges[0]) / 1000
print(f"  Block size: {bx_km:.2f} × {by_km:.2f} km")

initial_depths = np.full((NX, NY), 1500.0)
density_func = make_density_func('exponential', drho_0=DRHO_0, lam=LAMBDA)

# ============================================================
# 3. Run MCMC
# ============================================================
print(f"\n[3/5] Running MCMC ({N_ITERATIONS:,} iterations)...")
t0 = time.time()
result = run_mcmc_3d(
    obs_x=obs_x, obs_y=obs_y, gravity_obs=gravity_obs,
    block_x_edges=block_x_edges, block_y_edges=block_y_edges,
    density_func=density_func, noise_std=NOISE_STD,
    n_iterations=N_ITERATIONS, step_size=STEP_SIZE,
    depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
    smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
    initial_depths=initial_depths, seed=42, verbose=True,
)
elapsed = time.time() - t0
print(f"\n  Runtime: {elapsed/60:.1f} min  |  Acceptance: {result['acceptance_rate']*100:.1f}%")

# ============================================================
# 4. Posterior + validation
# ============================================================
print(f"\n[4/5] Processing posterior...")
posterior = process_chain_3d(result, burn_in_frac=0.5, thin=1)
mean_d = posterior['mean']
std_d = posterior['std']

print(f"  Depth range: {mean_d.min():.0f} – {mean_d.max():.0f} m")
print(f"  Mean uncertainty: {std_d.mean():.0f} m")

# Downsample ground-truth (40x60 or 41x60) → (NX, NY) via block averaging
bd_sub = bd[:40, :60] if bd.shape[0] == 41 else bd
truth_blocks = np.zeros((NX, NY))
by_cells = bd_sub.shape[0] // NY   # y dimension = rows
bx_cells = bd_sub.shape[1] // NX   # x dimension = cols
for i in range(NX):
    for j in range(NY):
        tile = bd_sub[j*by_cells:(j+1)*by_cells, i*bx_cells:(i+1)*bx_cells]
        truth_blocks[i, j] = tile.mean()

rms = np.sqrt(np.mean((mean_d - truth_blocks) ** 2))
bias = np.mean(mean_d - truth_blocks)
ci_lo, ci_hi = posterior['ci_5'], posterior['ci_95']
coverage = np.mean((truth_blocks >= ci_lo) & (truth_blocks <= ci_hi))

print(f"\n  === VALIDATION vs. published basement model ===")
print(f"  Truth range: {truth_blocks.min():.0f} – {truth_blocks.max():.0f} m")
print(f"  RMS error:   {rms:.0f} m")
print(f"  Bias:        {bias:+.0f} m")
print(f"  90% CI coverage: {coverage*100:.0f}%")
print(f"  ONGC borehole basement: 2935 m  |  Our deepest block: {mean_d.max():.0f} m")
print(f"  Chakravarthi 2007 estimate: 3100 m (5.6% error vs borehole)")

# ============================================================
# 5. Save + plot
# ============================================================
os.makedirs(OUT_DIR, exist_ok=True)
np.savez(os.path.join(OUT_DIR, 'results_data.npz'),
         mean_depths=mean_d, std_depths=std_d,
         ci_5=posterior['ci_5'], ci_95=posterior['ci_95'],
         truth_blocks=truth_blocks,
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
         drho_0=DRHO_0, lam=LAMBDA,
         rms=rms, bias=bias, coverage=coverage,
         acceptance_rate=result['acceptance_rate'],
         runtime_min=elapsed/60,
         all_misfits=np.asarray(result['all_misfits']))

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

im = axes[0, 0].pcolormesh(block_x_edges/1000, block_y_edges/1000, mean_d.T,
                           cmap='viridis_r', shading='flat')
plt.colorbar(im, ax=axes[0, 0], label='Depth (m)')
axes[0, 0].set_title(f'Posterior Mean  ({mean_d.min():.0f}–{mean_d.max():.0f} m)')
axes[0, 0].set_xlabel('X (km)'); axes[0, 0].set_ylabel('Y (km)')
axes[0, 0].set_aspect('equal')

im = axes[0, 1].pcolormesh(block_x_edges/1000, block_y_edges/1000, truth_blocks.T,
                           cmap='viridis_r', shading='flat',
                           vmin=mean_d.min(), vmax=mean_d.max())
plt.colorbar(im, ax=axes[0, 1], label='Depth (m)')
axes[0, 1].set_title(f'Published Ground Truth  ({truth_blocks.min():.0f}–{truth_blocks.max():.0f} m)')
axes[0, 1].set_xlabel('X (km)'); axes[0, 1].set_ylabel('Y (km)')
axes[0, 1].set_aspect('equal')

im = axes[1, 0].pcolormesh(block_x_edges/1000, block_y_edges/1000, std_d.T,
                           cmap='hot_r', shading='flat')
plt.colorbar(im, ax=axes[1, 0], label='Std (m)')
axes[1, 0].set_title(f'Posterior Uncertainty (mean {std_d.mean():.0f} m)')
axes[1, 0].set_xlabel('X (km)'); axes[1, 0].set_ylabel('Y (km)')
axes[1, 0].set_aspect('equal')

diff = mean_d - truth_blocks
vmax = np.abs(diff).max()
im = axes[1, 1].pcolormesh(block_x_edges/1000, block_y_edges/1000, diff.T,
                           cmap='RdBu_r', shading='flat', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=axes[1, 1], label='Error (m)')
axes[1, 1].set_title(f'Recovered − Truth  (RMS {rms:.0f} m, bias {bias:+.0f} m)')
axes[1, 1].set_xlabel('X (km)'); axes[1, 1].set_ylabel('Y (km)')
axes[1, 1].set_aspect('equal')

fig.suptitle(f'Chintalpudi Sub-Basin — MCMC Inversion (10k) vs Published Model',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'chintalpudi_summary.png'), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(result['all_misfits'], linewidth=0.3, alpha=0.7)
ax.axvline(x=N_ITERATIONS // 2, color='red', linestyle='--', label='Burn-in')
ax.set_xlabel('Iteration'); ax.set_ylabel('Misfit')
ax.set_title(f'MCMC Convergence  (acceptance {result["acceptance_rate"]*100:.1f}%)')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'misfit_trace.png'), dpi=150)
plt.close(fig)

print(f"\n{'='*70}")
print(f"DONE — results saved to: {OUT_DIR}/")
print(f"{'='*70}")
