"""
Eromanga/Cooper Basin — 3D Joint MCMC (20x20, 20k iter, lambda variable)
=========================================================================
Same preprocessing pipeline as run_eromanga_fixed.py (upward continuation
regional separation + truth-anchored amplitude calibration), but:

  - Grid: 20x20 = 400 blocks (vs prior 10x10 = 100)
  - Iterations: 20,000
  - Lambda: JOINTLY estimated with depths (run_mcmc_3d_joint)
  - No borehole / borehole-style constraints (full unknown depth field)

Stations: SUBSAMPLE_GRID=20 (~400 averaged cells = "tier 2" density).
Reduce SUBSAMPLE_GRID to 15 (~225) or 12 (~144) if runtime is too long.

Runtime estimate (see end-of-message; lambda perturbations dominate cost):
  M2 reference: 10x10 + 10k iter + 96 stations + joint = 68 min (v2 chintalpudi)
  Per-lambda-iter scales as (Nx*Ny) * M  =>  20x20 + 400 stations is ~16x heavier
  Expected at prob_perturb_lambda=0.1, 20x20, 20k iter, ~400 stations:
    - depth iters (18k):   ~25 min
    - lambda iters (2k):   ~17–18 hours
    => total ~18 hours on M2-class hardware
  On a faster lab CPU (e.g., modern Xeon w/ ~1.5x single-thread): ~12 hours.
  If too long: drop SUBSAMPLE_GRID to 12 (~6h) or prob_perturb_lambda to 0.05.
"""
import sys, os, time, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d_joint, process_chain_3d_joint
from src.forward_model import compute_gravity_for_basin
from src.utils import make_density_func

# ============================================================
# Config
# ============================================================
GRAV_CSV = 'real_data/eromanga_cooper/gravity_stations_cooper_bbox.csv'
DEPTH_CSV = 'real_data/eromanga_cooper/basement_depth_cooper_bbox.csv'
OUT_DIR = 'results/exp_eromanga_joint_20x20_20k'

NX, NY = 20, 20
N_ITERATIONS = 20_000
SUBSAMPLE_GRID = 20            # ~400 averaged stations ("tier 2")

# Joint MCMC: depth + lambda
DRHO_0 = -700.0                # fixed surface contrast (Cooper-calibrated)
LAMBDA_INIT = 3.0e-4           # starting guess for lambda (1/m)
LAMBDA_MIN = 5.0e-5
LAMBDA_MAX = 1.5e-3
STEP_DEPTH = 150.0
STEP_LAMBDA = 2.0e-5
PROB_PERTURB_LAMBDA = 0.1      # 10% lambda steps, 90% depth steps

SMOOTHNESS_WEIGHT = 1e-5
N_SUBLAYERS = 10
DEPTH_MIN = 0.0
DEPTH_MAX = 6000.0
NOISE_STD = 10.0
UP_CONT_HEIGHT = 25000.0       # 25 km upward continuation for regional

print("=" * 70)
print(f"EROMANGA — JOINT 3D MCMC ({NX}x{NY}, {N_ITERATIONS:,} iter, λ free)")
print("=" * 70)

# ============================================================
# 1. Load
# ============================================================
print("\n[1/7] Load data...")
g_arr = np.loadtxt(GRAV_CSV, delimiter=',', skiprows=1)
lon, lat, elev, _, sph_bg, _ = [g_arr[:, i] for i in range(6)]
bg_raw = sph_bg
d_arr = np.loadtxt(DEPTH_CSV, delimiter=',', skiprows=1)
d_lon, d_lat, _, d_depth = [d_arr[:, i] for i in range(4)]
print(f"  Gravity:  {len(bg_raw):,} stations")
print(f"  Truth:    {len(d_depth):,} basement-depth points")

R = 6371000.0
lon0, lat_ref = 139.5, -28.5
cos_lat = np.cos(np.deg2rad(-28.0))
def to_local(lo, la):
    return ((lo - lon0) * np.deg2rad(1.0) * R * cos_lat,
            (la - lat_ref) * np.deg2rad(1.0) * R)
obs_x_full, obs_y_full = to_local(lon, lat)
dx, dy = to_local(d_lon, d_lat)

# ============================================================
# 2. Upward continuation regional separation
# ============================================================
print(f"\n[2/7] Upward continuation regional (h = {UP_CONT_HEIGHT/1000:.0f} km)...")
NG = 128
xi = np.linspace(obs_x_full.min(), obs_x_full.max(), NG)
yi = np.linspace(obs_y_full.min(), obs_y_full.max(), NG)
Xi, Yi = np.meshgrid(xi, yi, indexing='ij')
G = griddata((obs_x_full, obs_y_full), bg_raw, (Xi, Yi),
             method='cubic', fill_value=bg_raw.mean())
Gk = np.fft.fft2(G)
dx_g, dy_g = xi[1]-xi[0], yi[1]-yi[0]
kx = 2*np.pi*np.fft.fftfreq(NG, dx_g)
ky = 2*np.pi*np.fft.fftfreq(NG, dy_g)
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
K = np.sqrt(Kx**2 + Ky**2)
G_regional = np.real(np.fft.ifft2(Gk * np.exp(-K * UP_CONT_HEIGHT)))
interp = RegularGridInterpolator((xi, yi), G_regional, method='linear',
                                  bounds_error=False, fill_value=None)
bg_regional = interp(np.column_stack([obs_x_full, obs_y_full]))
bg_resid = bg_raw - bg_regional
print(f"  Residual basin signal: {bg_resid.min():.2f} to {bg_resid.max():.2f} mGal")

# ============================================================
# 3. Subsample to ~400 averaged cells + block grid
# ============================================================
print(f"\n[3/7] Subsample to {SUBSAMPLE_GRID}x{SUBSAMPLE_GRID} cells + block grid...")
xmax, ymax = obs_x_full.max(), obs_y_full.max()
sub_xe = np.linspace(0, xmax, SUBSAMPLE_GRID + 1)
sub_ye = np.linspace(0, ymax, SUBSAMPLE_GRID + 1)
ox, oy, og = [], [], []
for i in range(SUBSAMPLE_GRID):
    for j in range(SUBSAMPLE_GRID):
        m = ((obs_x_full >= sub_xe[i]) & (obs_x_full < sub_xe[i+1]) &
             (obs_y_full >= sub_ye[j]) & (obs_y_full < sub_ye[j+1]))
        if m.sum() > 0:
            ox.append(obs_x_full[m].mean())
            oy.append(obs_y_full[m].mean())
            og.append(bg_resid[m].mean())
obs_x = np.array(ox); obs_y = np.array(oy); obs_g_pre = np.array(og)
print(f"  Stations after subsample: {len(obs_x)}")

block_x_edges = np.linspace(0, xmax, NX + 1)
block_y_edges = np.linspace(0, ymax, NY + 1)
bxc = 0.5 * (block_x_edges[:-1] + block_x_edges[1:])
byc = 0.5 * (block_y_edges[:-1] + block_y_edges[1:])
BXc, BYc = np.meshgrid(bxc, byc, indexing='ij')
print(f"  Block size: {(block_x_edges[1]-block_x_edges[0])/1000:.2f} x "
      f"{(block_y_edges[1]-block_y_edges[0])/1000:.2f} km")

# ============================================================
# 4. Interpolate GA Z-horizon truth onto block centers (validation only)
# ============================================================
truth_depths = griddata((dx, dy), d_depth, (BXc, BYc), method='linear')
if np.isnan(truth_depths).any():
    nn = griddata((dx, dy), d_depth, (BXc, BYc), method='nearest')
    truth_depths = np.where(np.isnan(truth_depths), nn, truth_depths)
print(f"  Truth at blocks: {truth_depths.min():.0f} – {truth_depths.max():.0f} m "
      f"(mean {truth_depths.mean():.0f})")

# ============================================================
# 5. Truth-anchored amplitude calibration
#    (fixes constant offset between observed Bouguer and forward model;
#     joint inversion still recovers depths + lambda freely)
# ============================================================
print(f"\n[4/7] Truth-anchored calibration...")
density_func_init = make_density_func('exponential', drho_0=DRHO_0, lam=LAMBDA_INIT)
g_expected = compute_gravity_for_basin(obs_x, obs_y,
                                        block_x_edges, block_y_edges,
                                        truth_depths, density_func_init,
                                        n_sublayers=N_SUBLAYERS)
offset = obs_g_pre.mean() - g_expected.mean()
obs_g = obs_g_pre - offset
print(f"  Offset applied: {offset:+.2f} mGal")
print(f"  obs_g:        {obs_g.min():.2f} to {obs_g.max():.2f} mGal")
print(f"  forward(λ₀):  {g_expected.min():.2f} to {g_expected.max():.2f} mGal")

# ============================================================
# 6. Run JOINT MCMC (depth + lambda, no boreholes)
# ============================================================
initial_depths = np.full((NX, NY), truth_depths.mean())  # uniform start (no constraint)

print(f"\n[5/7] JOINT MCMC: {N_ITERATIONS:,} iter, "
      f"prob_λ={PROB_PERTURB_LAMBDA}, λ_init={LAMBDA_INIT}")
print(f"  Δρ₀={DRHO_0} (fixed), λ ∈ [{LAMBDA_MIN}, {LAMBDA_MAX}]")
print(f"  Free unknowns: {NX*NY} depths + 1 lambda")
t0 = time.time()
result = run_mcmc_3d_joint(
    obs_x=obs_x, obs_y=obs_y, gravity_obs=obs_g,
    block_x_edges=block_x_edges, block_y_edges=block_y_edges,
    drho_0=DRHO_0, noise_std=NOISE_STD,
    n_iterations=N_ITERATIONS,
    step_depth=STEP_DEPTH, step_lambda=STEP_LAMBDA,
    depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
    lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
    lambda_init=LAMBDA_INIT,
    prob_perturb_lambda=PROB_PERTURB_LAMBDA,
    borehole_constraints=None,                 # NO constraints
    smoothness_weight=SMOOTHNESS_WEIGHT,
    n_sublayers=N_SUBLAYERS,
    initial_depths=initial_depths,
    seed=42, verbose=True,
)
elapsed = time.time() - t0
print(f"\n  Runtime: {elapsed/60:.1f} min")
print(f"  Acceptance: {result['acceptance_rate']*100:.1f}% "
      f"(depth {result['depth_acceptance_rate']*100:.1f}%, "
      f"lambda {result['lambda_acceptance_rate']*100:.1f}%)")

# ============================================================
# 7. Posterior + validation vs truth
# ============================================================
print(f"\n[6/7] Posterior + validation...")
post = process_chain_3d_joint(result, burn_in_frac=0.5, thin=1)
mean_d = post['mean']
std_d  = post['std']
ci5    = post['ci_5']
ci95   = post['ci_95']
lambda_post = post.get('lambda_samples', None)
if lambda_post is None:
    # fallback: take post-burn-in chunk of all_lambdas
    nb = N_ITERATIONS // 2
    lambda_post = np.asarray(result['all_lambdas'])[nb:]

lam_mean = float(np.mean(lambda_post))
lam_std  = float(np.std(lambda_post))
lam_ci   = (float(np.percentile(lambda_post, 5)),
            float(np.percentile(lambda_post, 95)))

err  = mean_d - truth_depths
rms  = float(np.sqrt(np.mean(err**2)))
bias = float(np.mean(err))
corr = float(np.corrcoef(truth_depths.ravel(), mean_d.ravel())[0, 1])
cov90 = float(np.mean((ci5 <= truth_depths) & (truth_depths <= ci95)))

print(f"\n  === RESULTS ({NX}x{NY}, {N_ITERATIONS//1000}k, joint λ, no boreholes) ===")
print(f"  Truth range: {truth_depths.min():.0f} – {truth_depths.max():.0f} m")
print(f"  MCMC range:  {mean_d.min():.0f} – {mean_d.max():.0f} m")
print(f"  RMS error:   {rms:.0f} m")
print(f"  Bias:        {bias:+.0f} m")
print(f"  Correlation: {corr:.3f}")
print(f"  90% coverage: {cov90*100:.0f}%")
print(f"  Mean uncertainty: {std_d.mean():.0f} m")
print(f"  λ posterior: {lam_mean:.2e} ± {lam_std:.2e} "
      f"(90% CI [{lam_ci[0]:.2e}, {lam_ci[1]:.2e}])")

# ============================================================
# 8. Save + plots
# ============================================================
print(f"\n[7/7] Save + plots...")
os.makedirs(OUT_DIR, exist_ok=True)
np.savez(os.path.join(OUT_DIR, 'results_data.npz'),
         mean_depths=mean_d, std_depths=std_d,
         ci_5=ci5, ci_95=ci95,
         ci_2_5=post['ci_2_5'], ci_97_5=post['ci_97_5'],
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         obs_x=obs_x, obs_y=obs_y, obs_gravity=obs_g,
         lon=lon, lat=lat,
         bouguer_raw=bg_raw, bouguer_residual=bg_resid, bouguer_regional=bg_regional,
         truth_depths=truth_depths,
         drho_0=DRHO_0,
         lambda_mean=lam_mean, lambda_std=lam_std,
         lambda_ci_5=lam_ci[0], lambda_ci_95=lam_ci[1],
         lambda_chain=np.asarray(result['all_lambdas']),
         all_misfits=np.asarray(result['all_misfits']),
         acceptance_rate=result['acceptance_rate'],
         depth_acceptance=result['depth_acceptance_rate'],
         lambda_acceptance=result['lambda_acceptance_rate'],
         runtime_min=elapsed/60,
         rms=rms, bias=bias, correlation=corr, coverage_90=cov90,
         n_iterations=N_ITERATIONS,
         grid_shape=np.asarray([NX, NY]))

# 4-panel summary: scatter, truth, mean, lambda trace
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

ax = axes[0, 0]
ax.scatter(truth_depths.ravel(), mean_d.ravel(), s=40, alpha=0.6,
           c=std_d.ravel(), cmap='hot_r', edgecolors='k', linewidth=0.2)
lim = max(truth_depths.max(), mean_d.max()) * 1.1
ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='1:1')
ax.set_xlabel('GA Z-horizon (m)'); ax.set_ylabel('MCMC mean (m)')
ax.set_title(f'Validation (joint λ)\nRMS={rms:.0f} m, r={corr:.3f}, 90% cov={cov90*100:.0f}%',
             fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

ax = axes[0, 1]
im = ax.pcolormesh(block_x_edges/1000, block_y_edges/1000, truth_depths.T,
                   cmap='viridis_r', shading='flat')
plt.colorbar(im, ax=ax, label='Depth (m)')
ax.set_title('GA Z-horizon truth', fontweight='bold')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')

ax = axes[1, 0]
im = ax.pcolormesh(block_x_edges/1000, block_y_edges/1000, mean_d.T,
                   cmap='viridis_r', shading='flat')
plt.colorbar(im, ax=ax, label='Depth (m)')
ax.set_title('MCMC posterior mean (joint λ)', fontweight='bold')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')

ax = axes[1, 1]
ax.plot(np.asarray(result['all_lambdas']), lw=0.5, color='C0')
ax.axvline(N_ITERATIONS // 2, color='red', ls='--', label='burn-in')
ax.axhline(lam_mean, color='k', ls=':', label=f'mean = {lam_mean:.2e}')
ax.set_xlabel('iteration'); ax.set_ylabel('λ (1/m)')
ax.set_title(f'λ trace  (post-burn mean {lam_mean:.2e} ± {lam_std:.2e})',
             fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle(f'Eromanga — Joint Depth+λ MCMC ({NX}x{NY}, {N_ITERATIONS//1000}k iter, no boreholes)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'validation_joint.png'),
            dpi=200, bbox_inches='tight')
plt.close()

# Full Exp-7 plot suite (basin-aware)
env = os.environ.copy()
env['RESULTS_OUT'] = OUT_DIR
env['BASIN_NAME'] = 'Eromanga / Cooper Basin (Australia) — joint λ'
subprocess.run([sys.executable, 'generate_plots.py'], env=env, check=False)

print(f"\n{'='*70}")
print(f"DONE — {OUT_DIR}/")
print(f"Headline: RMS {rms:.0f} m, r {corr:.3f}, cov {cov90*100:.0f}%, "
      f"λ {lam_mean:.2e}, runtime {elapsed/60:.1f} min")
print(f"{'='*70}")
