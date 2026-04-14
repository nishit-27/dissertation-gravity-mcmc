"""
Experiment 8: 3D Grid Joint MCMC with Borehole Constraints
===========================================================
Validates run_mcmc_3d_joint() on synthetic data before applying to real data.
Tests: 3D grid + joint depth/lambda estimation + borehole constraints.

This combines:
- Exp 6/7: 3D grid MCMC (10x10, fixed lambda)
- Exp 5: Joint depth+lambda with borehole constraints (1D)
Into: 3D + joint + boreholes (never tested together before)
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.synthetic import create_synthetic_basin_3d, generate_synthetic_gravity_3d
from src.mcmc_inversion import run_mcmc_3d_joint, process_chain_3d_joint, compute_coverage
from src.utils import make_density_func

# ============================================================
# 1. Create synthetic 3D basin (same as Exp 6/7)
# ============================================================
print("=" * 60)
print("Experiment 8: 3D Joint MCMC + Borehole Constraints")
print("=" * 60)

print("\n[1/5] Creating synthetic 3D basin...")
model = create_synthetic_basin_3d(nx_blocks=10, ny_blocks=10,
                                  x_length=100e3, y_length=100e3)

true_depths = model['true_depths']
print(f"  Grid: {model['nx_blocks']} x {model['ny_blocks']} = {model['nx_blocks']*model['ny_blocks']} blocks")
print(f"  Depth range: {true_depths.min():.0f} - {true_depths.max():.0f} m")

# ============================================================
# 2. Generate synthetic gravity with known lambda
# ============================================================
print("\n[2/5] Generating synthetic gravity...")
true_lambda = 0.0003
true_drho_0 = -500.0
density_func = make_density_func('exponential', drho_0=true_drho_0, lam=true_lambda)
noise_std = 0.3

data = generate_synthetic_gravity_3d(model, density_func, noise_std=noise_std, seed=42)
print(f"  True lambda: {true_lambda}")
print(f"  True drho_0: {true_drho_0}")
print(f"  Noise: {noise_std} mGal")
print(f"  Observation points: {len(data['obs_x'])}")

# ============================================================
# 3. Set up borehole constraints
# ============================================================
print("\n[3/5] Setting up borehole constraints...")

# Select 5 spatially dispersed blocks as constraints
constraint_blocks = [(2, 2), (2, 7), (5, 5), (7, 2), (7, 8)]
borehole_constraints = {}
for ix, iy in constraint_blocks:
    borehole_constraints[(ix, iy)] = true_depths[ix, iy]
    print(f"  Block ({ix},{iy}): locked at {true_depths[ix, iy]:.0f} m")

# Select 10 blocks for validation (not constrained)
validation_blocks = [(1, 4), (3, 3), (3, 8), (4, 1), (4, 6),
                     (6, 4), (6, 9), (8, 1), (8, 5), (9, 7)]
print(f"\n  Validation blocks: {len(validation_blocks)}")
for ix, iy in validation_blocks:
    print(f"  Block ({ix},{iy}): true depth = {true_depths[ix, iy]:.0f} m")

# ============================================================
# 4. Run MCMC
# ============================================================
print("\n[4/5] Running 3D Joint MCMC...")
t0 = time.time()

result = run_mcmc_3d_joint(
    obs_x=data['obs_x'],
    obs_y=data['obs_y'],
    gravity_obs=data['gravity_obs'],
    block_x_edges=model['block_x_edges'],
    block_y_edges=model['block_y_edges'],
    drho_0=true_drho_0,
    noise_std=noise_std,
    n_iterations=50000,
    step_depth=150.0,
    step_lambda=0.00003,
    depth_min=300.0,
    depth_max=6000.0,
    lambda_min=0.00005,
    lambda_max=0.003,
    lambda_init=0.0005,  # Start wrong — should recover to 0.0003
    prob_perturb_lambda=0.2,
    borehole_constraints=borehole_constraints,
    smoothness_weight=1e-6,
    n_sublayers=10,
    seed=42,
    verbose=True
)

elapsed = time.time() - t0
print(f"\nRuntime: {elapsed/60:.1f} minutes")

# ============================================================
# 5. Process results
# ============================================================
print("\n[5/5] Processing results...")
posterior = process_chain_3d_joint(result, burn_in_frac=0.5, thin=1)

mean_depths = posterior['mean']
std_depths = posterior['std']
lambda_mean = posterior['lambda_mean']
lambda_std = posterior['lambda_std']

# RMS error
rms = np.sqrt(np.mean((mean_depths - true_depths) ** 2))

# Coverage
coverage_90 = compute_coverage(true_depths, posterior, ci_level=90)
coverage_95 = compute_coverage(true_depths, posterior, ci_level=95)

# Lambda results
lambda_error_pct = abs(lambda_mean - true_lambda) / true_lambda * 100
lambda_in_90ci = (posterior['lambda_ci_5'] <= true_lambda <= posterior['lambda_ci_95'])

print("\n" + "=" * 60)
print("RESULTS — Experiment 8")
print("=" * 60)
print(f"Acceptance rate:     {result['acceptance_rate']*100:.1f}%")
print(f"  Depth acceptance:  {result['depth_acceptance_rate']*100:.1f}%")
print(f"  Lambda acceptance: {result['lambda_acceptance_rate']*100:.1f}%")
print(f"Posterior samples:   {posterior['n_samples']}")
print(f"\nDepth Results:")
print(f"  RMS error:         {rms:.1f} m")
print(f"  90% CI coverage:   {coverage_90*100:.0f}%")
print(f"  95% CI coverage:   {coverage_95*100:.0f}%")
print(f"\nLambda Results:")
print(f"  True lambda:       {true_lambda:.6f}")
print(f"  Estimated lambda:  {lambda_mean:.6f} +/- {lambda_std:.6f}")
print(f"  Lambda error:      {lambda_error_pct:.1f}%")
print(f"  90% CI:            [{posterior['lambda_ci_5']:.6f}, {posterior['lambda_ci_95']:.6f}]")
print(f"  True in 90% CI:    {'YES' if lambda_in_90ci else 'NO'}")

# Validation blocks
print(f"\nValidation at {len(validation_blocks)} held-out blocks:")
val_errors = []
val_covered = 0
for ix, iy in validation_blocks:
    true_d = true_depths[ix, iy]
    est_d = mean_depths[ix, iy]
    std_d = std_depths[ix, iy]
    ci_lo = posterior['ci_5'][ix, iy]
    ci_hi = posterior['ci_95'][ix, iy]
    covered = ci_lo <= true_d <= ci_hi
    val_errors.append(est_d - true_d)
    if covered:
        val_covered += 1
    print(f"  ({ix},{iy}): true={true_d:.0f}, est={est_d:.0f}+/-{std_d:.0f}, "
          f"90%CI=[{ci_lo:.0f},{ci_hi:.0f}] {'OK' if covered else 'MISS'}")

val_rms = np.sqrt(np.mean(np.array(val_errors) ** 2))
print(f"\n  Validation RMS: {val_rms:.1f} m")
print(f"  Validation 90% coverage: {val_covered}/{len(validation_blocks)} "
      f"({val_covered/len(validation_blocks)*100:.0f}%)")

# Comparison with Exp 7 (no boreholes, fixed lambda)
print(f"\n--- Comparison with Exp 7 (no boreholes, fixed lambda) ---")
print(f"  Exp 7: RMS=51.4m, 90% coverage=89%, 95% coverage=96%")
print(f"  Exp 8: RMS={rms:.1f}m, 90% coverage={coverage_90*100:.0f}%, "
      f"95% coverage={coverage_95*100:.0f}%")

# Save results
out_dir = 'results/exp08_3d_joint_boreholes/'
os.makedirs(out_dir, exist_ok=True)
np.savez(os.path.join(out_dir, 'results_data.npz'),
         true_depths=true_depths,
         mean_depths=mean_depths,
         std_depths=std_depths,
         lambda_mean=lambda_mean,
         lambda_std=lambda_std,
         true_lambda=true_lambda,
         rms=rms,
         coverage_90=coverage_90,
         coverage_95=coverage_95)
print(f"\nResults saved to {out_dir}")
print("=" * 60)
