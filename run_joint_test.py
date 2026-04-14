#!/usr/bin/env python3
"""
Joint inversion test: estimate BOTH depths AND lambda simultaneously.
10 blocks, 10000 iterations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.synthetic import create_synthetic_basin_2d, generate_synthetic_gravity
from src.utils import make_density_func
from src.mcmc_inversion import run_mcmc_joint, process_joint_chain, compute_coverage
from src.forward_model import compute_gravity_for_basin_fast

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

# TRUE parameters
TRUE_DRHO_0 = -500.0
TRUE_LAMBDA = 0.0003

print("=" * 60)
print("JOINT BAYESIAN MCMC: DEPTHS + LAMBDA")
print("=" * 60)
print(f"True lambda: {TRUE_LAMBDA}")
print(f"True drho_0: {TRUE_DRHO_0} kg/m^3 (fixed)")

# 1. Create synthetic basin
print("\n[1/5] Creating synthetic basin...")
model = create_synthetic_basin_2d(n_blocks=10, profile_length=80e3)

# 2. Generate gravity with TRUE lambda
print("[2/5] Generating synthetic gravity...")
true_density_func = make_density_func('exponential', drho_0=TRUE_DRHO_0, lam=TRUE_LAMBDA)
data = generate_synthetic_gravity(model, true_density_func, noise_std=0.3, n_sublayers=10)
print(f"  Gravity range: {data['gravity_obs'].min():.2f} to {data['gravity_obs'].max():.2f} mGal")

# 3. Run joint MCMC — start lambda far from true to test recovery
print("\n[3/5] Running Joint MCMC (10,000 iterations)...")
print(f"  Starting lambda at 0.0005 (true = {TRUE_LAMBDA})")
result = run_mcmc_joint(
    obs_x=data['obs_x'],
    gravity_obs=data['gravity_obs'],
    block_x_edges=model['block_x_edges'],
    block_y_width=model['block_y_width'],
    drho_0=TRUE_DRHO_0,
    noise_std=data['noise_std'],
    n_iterations=10000,
    step_depth=150.0,
    step_lambda=0.00003,
    depth_min=300.0,
    depth_max=6000.0,
    lambda_min=0.00001,
    lambda_max=0.003,
    lambda_init=0.0005,  # start far from true (0.0003)
    prob_perturb_lambda=0.2,
    n_sublayers=10,
    verbose=True,
)

# 4. Process chain
print("\n[4/5] Processing posterior...")
posterior = process_joint_chain(result, burn_in_frac=0.5)

rms = np.sqrt(np.mean((posterior['mean'] - model['true_depths'])**2))
cov90 = compute_coverage(model['true_depths'], posterior, ci_level=90)

print(f"\n  === DEPTH RESULTS ===")
print(f"  RMS error: {rms:.1f} m")
print(f"  90% CI coverage: {cov90*100:.0f}%")
print(f"\n  Block |  x (km) | True (m) | Mean (m) | Std (m) | Error (m)")
print(f"  " + "-" * 65)
x_km = model['block_x_centers'] / 1e3
for i in range(model['n_blocks']):
    err = abs(posterior['mean'][i] - model['true_depths'][i])
    print(f"  {i+1:5d} | {x_km[i]:7.1f} | {model['true_depths'][i]:8.0f} | "
          f"{posterior['mean'][i]:8.0f} | {posterior['std'][i]:7.0f} | {err:8.0f}")

print(f"\n  === LAMBDA RESULTS ===")
print(f"  True lambda:     {TRUE_LAMBDA:.6f}")
print(f"  Estimated mean:  {posterior['lambda_mean']:.6f}")
print(f"  Estimated std:   {posterior['lambda_std']:.6f}")
print(f"  90% CI:          [{posterior['lambda_ci_5']:.6f}, {posterior['lambda_ci_95']:.6f}]")
print(f"  Error:           {abs(posterior['lambda_mean'] - TRUE_LAMBDA):.6f}")
print(f"  True within 90% CI: {posterior['lambda_ci_5'] <= TRUE_LAMBDA <= posterior['lambda_ci_95']}")

# 5. Plots
print("\n[5/5] Generating plots...")

# --- PLOT 1: Basement depth with uncertainty (same as before) ---
fig, axes = plt.subplots(3, 1, figsize=(12, 14),
                         gridspec_kw={'height_ratios': [2.5, 1, 1]})

ax1 = axes[0]
ax1.fill_between(x_km, posterior['ci_2_5'], posterior['ci_97_5'],
                 color='#3498db', alpha=0.12, label='95% CI')
ax1.fill_between(x_km, posterior['ci_5'], posterior['ci_95'],
                 color='#3498db', alpha=0.25, label='90% CI')
ax1.plot(x_km, model['true_depths'], 'k-', linewidth=2.5,
         marker='s', markersize=6, label='True Basement')
ax1.plot(x_km, posterior['mean'], 'r--', linewidth=2,
         marker='o', markersize=5, label='MCMC Mean')
ax1.set_xlabel('Distance (km)', fontsize=12)
ax1.set_ylabel('Depth (m)', fontsize=12)
ax1.set_title('Joint Inversion: Basement Depth (depths + lambda estimated simultaneously)',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='lower left')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3)
ax1.text(0.98, 0.02,
         f'RMS: {rms:.1f}m | 90% Cov: {cov90*100:.0f}% | Samples: {posterior["n_samples"]}',
         transform=ax1.transAxes, fontsize=10, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Uncertainty
ax2 = axes[1]
colors = plt.cm.Reds(posterior['std'] / max(posterior['std']))
ax2.bar(x_km, posterior['std'], width=(x_km[1]-x_km[0])*0.7,
        color=colors, edgecolor='darkred', alpha=0.8)
ax2.set_xlabel('Distance (km)', fontsize=12)
ax2.set_ylabel('Uncertainty (m)', fontsize=12)
ax2.set_title('Depth Uncertainty per Block', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Gravity fit
ax3 = axes[2]
mean_density = lambda z: TRUE_DRHO_0 * np.exp(-posterior['lambda_mean'] * z)
gravity_mcmc = compute_gravity_for_basin_fast(
    data['obs_x'], model['block_x_edges'], model['block_y_width'],
    posterior['mean'], mean_density, n_sublayers=10
)
ax3.plot(data['obs_x']/1e3, data['gravity_obs'], 'ko', markersize=6, label='Observed')
ax3.plot(data['obs_x']/1e3, data['gravity_true'], 'b-', linewidth=1.5, label='True')
ax3.plot(data['obs_x']/1e3, gravity_mcmc, 'r--', linewidth=2, label='MCMC Mean')
ax3.set_xlabel('Distance (km)', fontsize=12)
ax3.set_ylabel('Gravity (mGal)', fontsize=12)
ax3.set_title('Gravity Fit', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'joint_depth_uncertainty.png'), dpi=150, bbox_inches='tight')
print("  Saved: results/joint_depth_uncertainty.png")
plt.close()

# --- PLOT 2: Lambda trace + histogram ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Lambda trace
ax1 = axes[0, 0]
ax1.plot(result['all_lambdas'], 'b-', linewidth=0.3, alpha=0.7)
ax1.axhline(y=TRUE_LAMBDA, color='red', linewidth=2, linestyle='-', label=f'True: {TRUE_LAMBDA}')
ax1.axvline(x=result['n_iterations']//2, color='gray', linestyle='--', label='Burn-in cutoff')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Lambda (1/m)', fontsize=12)
ax1.set_title('Lambda Trace', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Lambda histogram (post burn-in)
ax2 = axes[0, 1]
ax2.hist(posterior['lambda_samples'], bins=50, density=True,
         color='#2ecc71', alpha=0.7, edgecolor='white')
ax2.axvline(x=TRUE_LAMBDA, color='black', linewidth=2, linestyle='-',
            label=f'True: {TRUE_LAMBDA:.4f}')
ax2.axvline(x=posterior['lambda_mean'], color='red', linewidth=2, linestyle='--',
            label=f'Mean: {posterior["lambda_mean"]:.4f}')
ax2.axvspan(posterior['lambda_ci_5'], posterior['lambda_ci_95'],
            alpha=0.15, color='orange', label='90% CI')
ax2.set_xlabel('Lambda (1/m)', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.set_title('Lambda Posterior Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Misfit trace
ax3 = axes[1, 0]
ax3.plot(result['all_misfits'], 'b-', linewidth=0.3, alpha=0.7)
ax3.set_xlabel('Iteration', fontsize=12)
ax3.set_ylabel('Misfit S(m)', fontsize=12)
ax3.set_title('Misfit Convergence', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.axvline(x=result['n_iterations']//2, color='gray', linestyle='--')
ax3.grid(True, alpha=0.3)

# Density function comparison
ax4 = axes[1, 1]
z = np.linspace(0, 5000, 100)
rho_true = TRUE_DRHO_0 * np.exp(-TRUE_LAMBDA * z)
rho_mean = TRUE_DRHO_0 * np.exp(-posterior['lambda_mean'] * z)
rho_low = TRUE_DRHO_0 * np.exp(-posterior['lambda_ci_95'] * z)  # higher lambda = faster decay
rho_high = TRUE_DRHO_0 * np.exp(-posterior['lambda_ci_5'] * z)  # lower lambda = slower decay
ax4.fill_betweenx(z, rho_low, rho_high, alpha=0.2, color='#3498db', label='90% CI')
ax4.plot(rho_true, z, 'k-', linewidth=2.5, label=f'True (lambda={TRUE_LAMBDA})')
ax4.plot(rho_mean, z, 'r--', linewidth=2, label=f'Estimated (lambda={posterior["lambda_mean"]:.5f})')
ax4.set_xlabel('Density Contrast (kg/m^3)', fontsize=12)
ax4.set_ylabel('Depth (m)', fontsize=12)
ax4.set_title('Density Function: True vs Estimated', fontsize=13, fontweight='bold')
ax4.invert_yaxis()
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.suptitle('Joint Inversion: Lambda Recovery & Diagnostics',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'joint_lambda_results.png'), dpi=150, bbox_inches='tight')
print("  Saved: results/joint_lambda_results.png")
plt.close()

# --- PLOT 3: Ensemble ---
fig, ax = plt.subplots(figsize=(12, 6))
chain = result['chain']
burn_in = int(len(chain) * 0.5)
samples = chain[burn_in:]
n_show = min(500, len(samples))
indices = np.linspace(0, len(samples)-1, n_show, dtype=int)
for i in indices:
    ax.plot(x_km, samples[i], color='#3498db', alpha=0.04, linewidth=0.8)
ax.plot(x_km, model['true_depths'], 'k-', linewidth=2.5, marker='s', markersize=6, label='True')
ax.plot(x_km, np.mean(samples, axis=0), 'r--', linewidth=2, marker='o', markersize=5, label='MCMC Mean')
ax.set_xlabel('Distance (km)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_title('Joint Inversion: Posterior Ensemble', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'joint_ensemble.png'), dpi=150, bbox_inches='tight')
print("  Saved: results/joint_ensemble.png")
plt.close()

print("\n" + "=" * 60)
print("JOINT INVERSION COMPLETE!")
print(f"True lambda:      {TRUE_LAMBDA:.6f}")
print(f"Estimated lambda:  {posterior['lambda_mean']:.6f} +/- {posterior['lambda_std']:.6f}")
print(f"Depth RMS:         {rms:.1f} m")
print("=" * 60)
