#!/usr/bin/env python3
"""
3D Synthetic Test: 10x10 grid, 10000 MCMC iterations, fixed lambda.
Extends the 1D profile inversion to a full 2D grid of blocks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.synthetic import create_synthetic_basin_3d, generate_synthetic_gravity_3d
from src.utils import make_density_func
from src.mcmc_inversion import run_mcmc_3d, process_chain_3d, compute_coverage
from src.forward_model import compute_gravity_for_basin
from src.visualization import (plot_depth_comparison, plot_depth_3d_surface,
                                plot_uncertainty_map, plot_uncertainty_3d_surface,
                                plot_depth_cross_sections, plot_gravity_fit_3d,
                                save_results)

# Create results directory
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'exp06_3d_depth_only')
os.makedirs(results_dir, exist_ok=True)

print("=" * 65)
print("3D BAYESIAN MCMC GRAVITY INVERSION - SYNTHETIC TEST")
print("10x10 grid, fixed lambda, incremental forward model")
print("=" * 65)

# =====================================================
# 1. Create synthetic 3D basin
# =====================================================
print("\n[1/6] Creating synthetic 3D basin...")
model = create_synthetic_basin_3d(nx_blocks=10, ny_blocks=10,
                                   x_length=100e3, y_length=100e3)
true_d = model['true_depths']
print(f"  Grid: {model['nx_blocks']}x{model['ny_blocks']} = "
      f"{model['nx_blocks']*model['ny_blocks']} blocks")
print(f"  Domain: {model['x_length']/1e3:.0f} x {model['y_length']/1e3:.0f} km")
print(f"  Block size: {model['x_length']/model['nx_blocks']/1e3:.0f} x "
      f"{model['y_length']/model['ny_blocks']/1e3:.0f} km")
print(f"  Depth range: {true_d.min():.0f} to {true_d.max():.0f} m")
print(f"  Mean depth: {true_d.mean():.0f} m")

# Print true depth grid
print("\n  True Depth Grid (m):")
x_km = model['block_x_centers'] / 1e3
y_km = model['block_y_centers'] / 1e3
print(f"  {'':>8s}", end='')
for j in range(model['ny_blocks']):
    print(f"  Y={y_km[j]:4.0f}", end='')
print()
for i in range(model['nx_blocks']):
    print(f"  X={x_km[i]:4.0f}", end='')
    for j in range(model['ny_blocks']):
        print(f"  {true_d[i,j]:6.0f}", end='')
    print()

# =====================================================
# 2. Generate synthetic gravity
# =====================================================
print("\n[2/6] Generating synthetic gravity data...")
density_func = make_density_func('exponential', drho_0=-500.0, lam=0.0003)
data = generate_synthetic_gravity_3d(model, density_func,
                                       noise_std=0.3, n_sublayers=10)
g_obs = data['gravity_obs']
print(f"  Observation stations: {len(data['obs_x'])} "
      f"({model['nx_blocks']}x{model['ny_blocks']} grid)")
print(f"  Gravity range: {g_obs.min():.2f} to {g_obs.max():.2f} mGal")
print(f"  Noise: {data['noise_std']} mGal")

# =====================================================
# 3. Run 3D MCMC
# =====================================================
n_iter = 10000
print(f"\n[3/6] Running 3D MCMC inversion ({n_iter:,} iterations)...")
t_start = time.time()

result = run_mcmc_3d(
    obs_x=data['obs_x'],
    obs_y=data['obs_y'],
    gravity_obs=data['gravity_obs'],
    block_x_edges=model['block_x_edges'],
    block_y_edges=model['block_y_edges'],
    density_func=density_func,
    noise_std=data['noise_std'],
    n_iterations=n_iter,
    step_size=150.0,
    depth_min=300.0,
    depth_max=6000.0,
    smoothness_weight=0.0,
    n_sublayers=10,
    verbose=True,
)

elapsed = time.time() - t_start
print(f"\n  Runtime: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print(f"  Speed: {n_iter/elapsed:.0f} iterations/second")

# =====================================================
# 4. Process chain
# =====================================================
print("\n[4/6] Processing posterior...")
posterior = process_chain_3d(result, burn_in_frac=0.5, thin=1)
print(f"  Post burn-in samples: {posterior['n_samples']}")

rms = np.sqrt(np.mean((posterior['mean'] - true_d)**2))
cov90 = compute_coverage(true_d, posterior, ci_level=90)
cov95 = compute_coverage(true_d, posterior, ci_level=95)

print(f"  RMS error: {rms:.1f} m")
print(f"  90% CI coverage: {cov90*100:.0f}%")
print(f"  95% CI coverage: {cov95*100:.0f}%")
print(f"  Acceptance rate: {result['acceptance_rate']*100:.1f}%")

# Print per-block comparison
est_d = posterior['mean']
print(f"\n  Per-Block Results (True | Estimated | Error | Std):")
print(f"  {'':>8s}", end='')
for j in range(model['ny_blocks']):
    print(f"    Y={y_km[j]:4.0f}   ", end='')
print()
for i in range(model['nx_blocks']):
    print(f"  X={x_km[i]:4.0f}", end='')
    for j in range(model['ny_blocks']):
        err = abs(est_d[i,j] - true_d[i,j])
        print(f"  {est_d[i,j]:5.0f}±{posterior['std'][i,j]:3.0f}", end='')
    print()

# =====================================================
# 5. Generate all plots
# =====================================================
print("\n[5/6] Generating plots...")

# Save all data for later reuse
save_results(model, data, result, posterior,
              density_params={'drho_0': -500.0, 'lam': 0.0003},
              save_path=os.path.join(results_dir, 'results_data.npz'))

# Plot 1: 2D depth comparison (jet colormap, smooth)
plot_depth_comparison(model, posterior,
                       save_path=os.path.join(results_dir, 'depth_comparison.png'))

# Plot 2: 3D surface plots
plot_depth_3d_surface(model, posterior,
                       save_path=os.path.join(results_dir, 'depth_3d_surface.png'))

# Plot 3: Uncertainty maps
plot_uncertainty_map(model, posterior,
                      save_path=os.path.join(results_dir, 'uncertainty_map.png'))

# Plot 4: 3D uncertainty surface
plot_uncertainty_3d_surface(model, posterior,
                             save_path=os.path.join(results_dir, 'uncertainty_3d_surface.png'))

# Plot 5: Cross-sections through the grid
plot_depth_cross_sections(model, posterior,
                           save_path=os.path.join(results_dir, 'cross_sections.png'))

# Plot 6: Gravity data fit
plot_gravity_fit_3d(model, data, posterior, density_func,
                     save_path=os.path.join(results_dir, 'gravity_fit.png'))

# Plot 5: MCMC diagnostics (reuse existing trace plot logic)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax1 = axes[0]
ax1.plot(result['all_misfits'], 'b-', linewidth=0.3, alpha=0.7)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Misfit S(m)', fontsize=12)
ax1.set_title('MCMC Convergence: Misfit vs Iteration', fontsize=13,
              fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
burn_in_iter = n_iter // 2
ax1.axvline(x=burn_in_iter, color='red', linestyle='--', linewidth=1.5,
            label=f'Burn-in cutoff ({burn_in_iter})')
ax1.legend(fontsize=10)

ax2 = axes[1]
window = 200
misfits = result['all_misfits']
accepted = np.zeros(len(misfits))
for i in range(1, len(misfits)):
    if misfits[i] != misfits[i-1]:
        accepted[i] = 1
running_rate = np.convolve(accepted, np.ones(window)/window, mode='valid') * 100
ax2.plot(range(len(running_rate)), running_rate, 'g-', linewidth=0.8)
ax2.axhline(y=20, color='red', linestyle=':', alpha=0.5)
ax2.axhline(y=50, color='red', linestyle=':', alpha=0.5)
ax2.fill_between(range(len(running_rate)), 20, 50, alpha=0.08, color='green',
                 label='Target range (20-50%)')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)
ax2.set_title(f'Running Acceptance Rate (window={window})', fontsize=13,
              fontweight='bold')
ax2.set_ylim(0, 80)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'mcmc_diagnostics.png'),
            dpi=150, bbox_inches='tight')
print(f"  Saved: results/exp06_3d_depth_only/mcmc_diagnostics.png")
plt.close()

# =====================================================
# 6. Summary
# =====================================================
print("\n" + "=" * 65)
print("3D MCMC INVERSION COMPLETE")
print("=" * 65)
print(f"  Grid: {model['nx_blocks']}x{model['ny_blocks']} blocks")
print(f"  Iterations: {n_iter:,}")
print(f"  Runtime: {elapsed:.1f}s")
print(f"  Acceptance rate: {result['acceptance_rate']*100:.1f}%")
print(f"  RMS depth error: {rms:.1f} m")
print(f"  90% CI coverage: {cov90*100:.0f}%")
print(f"  95% CI coverage: {cov95*100:.0f}%")
print(f"  Results: {results_dir}")
print("=" * 65)
