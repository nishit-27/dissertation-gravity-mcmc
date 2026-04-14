#!/usr/bin/env python3
"""
Full synthetic test: 10 blocks, 10000 MCMC iterations.
Generates all plots for dissertation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.synthetic import create_synthetic_basin_2d, generate_synthetic_gravity
from src.utils import make_density_func
from src.mcmc_inversion import run_mcmc, process_chain, compute_coverage
from src.forward_model import compute_gravity_for_basin_fast

# Create results directory
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

print("=" * 60)
print("BAYESIAN MCMC GRAVITY INVERSION - SYNTHETIC TEST")
print("=" * 60)

# =====================================================
# 1. Create synthetic basin
# =====================================================
print("\n[1/5] Creating synthetic basin...")
model = create_synthetic_basin_2d(n_blocks=10, profile_length=80e3)
print(f"  Blocks: {model['n_blocks']}")
print(f"  Profile: {model['profile_length']/1e3:.0f} km")
print(f"  Depth range: {model['true_depths'].min():.0f} to {model['true_depths'].max():.0f} m")

# =====================================================
# 2. Generate synthetic gravity
# =====================================================
print("\n[2/5] Generating synthetic gravity data...")
density_func = make_density_func('exponential', drho_0=-500.0, lam=0.0003)
data = generate_synthetic_gravity(model, density_func, noise_std=0.3, n_sublayers=10)
print(f"  Stations: {len(data['obs_x'])}")
print(f"  Gravity range: {data['gravity_obs'].min():.2f} to {data['gravity_obs'].max():.2f} mGal")
print(f"  Noise: {data['noise_std']} mGal")

# =====================================================
# 3. Run MCMC
# =====================================================
print("\n[3/5] Running MCMC inversion (10,000 iterations)...")
result = run_mcmc(
    obs_x=data['obs_x'],
    gravity_obs=data['gravity_obs'],
    block_x_edges=model['block_x_edges'],
    block_y_width=model['block_y_width'],
    density_func=density_func,
    noise_std=data['noise_std'],
    n_iterations=10000,
    step_size=150.0,
    depth_min=300.0,
    depth_max=6000.0,
    smoothness_weight=0.0,
    n_sublayers=10,
    verbose=True,
)

# =====================================================
# 4. Process chain
# =====================================================
print("\n[4/5] Processing posterior...")
posterior = process_chain(result, burn_in_frac=0.5, thin=1)
print(f"  Post burn-in samples: {posterior['n_samples']}")

rms = np.sqrt(np.mean((posterior['mean'] - model['true_depths'])**2))
cov90 = compute_coverage(model['true_depths'], posterior, ci_level=90)
cov95 = compute_coverage(model['true_depths'], posterior, ci_level=95)

print(f"  RMS error: {rms:.1f} m")
print(f"  90% CI coverage: {cov90*100:.0f}%")
print(f"  95% CI coverage: {cov95*100:.0f}%")

# Print per-block results
print("\n  Block |  x (km) | True (m) | Mean (m) | Std (m) | Error (m)")
print("  " + "-" * 65)
x_km = model['block_x_centers'] / 1e3
for i in range(model['n_blocks']):
    err = abs(posterior['mean'][i] - model['true_depths'][i])
    print(f"  {i+1:5d} | {x_km[i]:7.1f} | {model['true_depths'][i]:8.0f} | "
          f"{posterior['mean'][i]:8.0f} | {posterior['std'][i]:7.0f} | {err:8.0f}")

# =====================================================
# 5. Generate all plots
# =====================================================
print("\n[5/5] Generating plots...")

# --- PLOT 1: Basement depth with uncertainty ---
fig, axes = plt.subplots(3, 1, figsize=(12, 14),
                         gridspec_kw={'height_ratios': [2.5, 1, 1]})

ax1 = axes[0]
ax1.fill_between(x_km, posterior['ci_2_5'], posterior['ci_97_5'],
                 color='#3498db', alpha=0.12, label='95% Credible Interval')
ax1.fill_between(x_km, posterior['ci_5'], posterior['ci_95'],
                 color='#3498db', alpha=0.25, label='90% Credible Interval')
ax1.plot(x_km, model['true_depths'], 'k-', linewidth=2.5,
         marker='s', markersize=6, label='True Basement')
ax1.plot(x_km, posterior['mean'], 'r--', linewidth=2,
         marker='o', markersize=5, label='MCMC Mean Estimate')
ax1.set_xlabel('Distance along profile (km)', fontsize=12)
ax1.set_ylabel('Depth (m)', fontsize=12)
ax1.set_title('Basement Depth: True vs MCMC Estimate with Uncertainty',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='lower left')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3)
ax1.text(0.98, 0.02,
         f'RMS: {rms:.1f} m | 90% Coverage: {cov90*100:.0f}% | Samples: {posterior["n_samples"]}',
         transform=ax1.transAxes, fontsize=10, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Uncertainty bar chart
ax2 = axes[1]
colors = plt.cm.Reds(posterior['std'] / max(posterior['std']))
ax2.bar(x_km, posterior['std'], width=(x_km[1]-x_km[0])*0.7,
        color=colors, edgecolor='darkred', alpha=0.8)
ax2.set_xlabel('Distance along profile (km)', fontsize=12)
ax2.set_ylabel('Uncertainty (m)', fontsize=12)
ax2.set_title('Posterior Standard Deviation (Uncertainty) per Block', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Gravity fit
ax3 = axes[2]
# Compute gravity from MCMC mean model
gravity_mcmc = compute_gravity_for_basin_fast(
    data['obs_x'], model['block_x_edges'], model['block_y_width'],
    posterior['mean'], density_func, n_sublayers=10
)
ax3.plot(data['obs_x']/1e3, data['gravity_obs'], 'ko', markersize=6,
         label='Observed (with noise)', zorder=3)
ax3.plot(data['obs_x']/1e3, data['gravity_true'], 'b-', linewidth=1.5,
         label='True (noise-free)')
ax3.plot(data['obs_x']/1e3, gravity_mcmc, 'r--', linewidth=2,
         label='MCMC Mean Model')
ax3.set_xlabel('Distance along profile (km)', fontsize=12)
ax3.set_ylabel('Gravity Anomaly (mGal)', fontsize=12)
ax3.set_title('Gravity Data Fit', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'basement_depth_uncertainty.png'),
            dpi=150, bbox_inches='tight')
print("  Saved: results/basement_depth_uncertainty.png")
plt.close()

# --- PLOT 2: MCMC Diagnostics (trace + acceptance) ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax1 = axes[0]
ax1.plot(result['all_misfits'], 'b-', linewidth=0.3, alpha=0.7)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Misfit S(m)', fontsize=12)
ax1.set_title('MCMC Convergence: Misfit vs Iteration', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
burn_in_iter = result['n_iterations'] // 2
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
ax2.set_title(f'Running Acceptance Rate (window={window})', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 80)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'mcmc_diagnostics.png'),
            dpi=150, bbox_inches='tight')
print("  Saved: results/mcmc_diagnostics.png")
plt.close()

# --- PLOT 3: Posterior ensemble (all models) ---
fig, ax = plt.subplots(figsize=(12, 6))
chain = result['chain']
burn_in = int(len(chain) * 0.5)
samples = chain[burn_in:]
n_show = min(500, len(samples))
indices = np.linspace(0, len(samples)-1, n_show, dtype=int)
for i in indices:
    ax.plot(x_km, samples[i], color='#3498db', alpha=0.04, linewidth=0.8)
ax.plot(x_km, model['true_depths'], 'k-', linewidth=2.5,
        marker='s', markersize=6, label='True Basement')
ax.plot(x_km, np.mean(samples, axis=0), 'r--', linewidth=2,
        marker='o', markersize=5, label='MCMC Mean')
ax.set_xlabel('Distance along profile (km)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title(f'Posterior Ensemble: {n_show} Accepted Models from MCMC',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'posterior_ensemble.png'),
            dpi=150, bbox_inches='tight')
print("  Saved: results/posterior_ensemble.png")
plt.close()

# --- PLOT 4: Depth histograms at selected blocks ---
selected = [0, 2, 4, 7, 9]  # blocks to show
fig, axes = plt.subplots(1, len(selected), figsize=(4*len(selected), 5))

for ax, idx in zip(axes, selected):
    s = posterior['samples'][:, idx]
    true_d = model['true_depths'][idx]
    ax.hist(s, bins=40, density=True, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(x=true_d, color='black', linewidth=2, linestyle='-',
               label=f'True: {true_d:.0f}m')
    ax.axvline(x=posterior['mean'][idx], color='red', linewidth=2, linestyle='--',
               label=f'Mean: {posterior["mean"][idx]:.0f}m')
    ax.axvspan(posterior['ci_5'][idx], posterior['ci_95'][idx],
               alpha=0.15, color='orange', label='90% CI')
    ax.set_xlabel('Depth (m)', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.set_title(f'Block {idx+1}\nx={x_km[idx]:.0f} km', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)

plt.suptitle('Posterior Depth Distributions at Selected Blocks',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'depth_histograms.png'),
            dpi=150, bbox_inches='tight')
print("  Saved: results/depth_histograms.png")
plt.close()

# --- PLOT 5: Density function visualization ---
fig, ax = plt.subplots(figsize=(6, 6))
z = np.linspace(0, 5000, 100)
rho = density_func(z)
ax.plot(rho, z, 'b-', linewidth=2)
ax.set_xlabel('Density Contrast (kg/m³)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title('Exponential Density-Depth Function\n$\\Delta\\rho(z) = \\Delta\\rho_0 \\cdot e^{-\\lambda z}$',
             fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.text(0.95, 0.05,
        f'$\\Delta\\rho_0$ = -500 kg/m³\n$\\lambda$ = 0.0003 /m',
        transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'density_function.png'),
            dpi=150, bbox_inches='tight')
print("  Saved: results/density_function.png")
plt.close()

print("\n" + "=" * 60)
print("ALL DONE!")
print(f"Results saved in: {results_dir}")
print("=" * 60)
