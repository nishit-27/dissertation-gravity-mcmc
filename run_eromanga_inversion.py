"""
Eromanga/Cooper Basin (Australia) — 3D MCMC Gravity Inversion + Validation
============================================================================
PRIMARY methodology validation: real ground gravity + published basement-depth
ground truth, all CC-BY (Geoscience Australia).

Bbox:  139.5°–140.5°E, 27.5°–28.5°S (Nappamerri Trough, Moomba–Big Lake)
Data:  4,209 real ground gravity stations (GA 2019 compilation)
Truth: 8,033 basement-depth points from GA Cooper 3D model (Meixner 2009)
Model: 10×10 MCMC block grid, exponential compaction, FIXED density
Param: Δρ₀ = −400 kg/m³, λ = 4×10⁻⁴ /m (Cooper-calibrated; hard crystalline)
"""
import sys, os, time, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d, process_chain_3d
from src.utils import make_density_func

# ============================================================
GRAV_CSV = 'real_data/eromanga_cooper/gravity_stations_cooper_bbox.csv'
DEPTH_CSV = 'real_data/eromanga_cooper/basement_depth_cooper_bbox.csv'

NX, NY = 10, 10
N_ITERATIONS = 50_000

# Density (Eromanga/Cooper: softer contrast than Cauvery due to different lithology)
# Sediment 2.35-2.55 g/cc vs basement 2.67-2.72 g/cc → Δρ surface ~-350 kg/m³
DRHO_0 = -400.0
LAMBDA = 4.0e-4

STEP_SIZE = 200.0
SMOOTHNESS_WEIGHT = 1e-6
N_SUBLAYERS = 10
DEPTH_MIN = 0.0
DEPTH_MAX = 6000.0
NOISE_STD = 1.0

# Subsample: 4209 stations → ~20×20 cells → ~400 averaged points (tractable MCMC)
SUBSAMPLE_GRID = 20

OUT_DIR = 'results/exp_eromanga_real'

print("=" * 70)
print("EROMANGA/COOPER BASIN — 3D MCMC INVERSION + VALIDATION")
print("=" * 70)

# ============================================================
# 1. Load gravity + basement depth
# ============================================================
print("\n[1/6] Loading data...")
g_arr = np.loadtxt(GRAV_CSV, delimiter=',', skiprows=1)
lon, lat, elev, obs_g_full, sph_bg, inf_bg = [g_arr[:, i] for i in range(6)]
bg_raw = sph_bg  # use spherical-cap Bouguer
print(f"  Gravity: {len(bg_raw):,} stations, Bouguer {bg_raw.min():.1f} to {bg_raw.max():.1f} mGal")

d_arr = np.loadtxt(DEPTH_CSV, delimiter=',', skiprows=1)
d_lon, d_lat, _, d_depth = [d_arr[:, i] for i in range(4)]
print(f"  Basement depth: {len(d_depth):,} points, range {d_depth.min():.0f}–{d_depth.max():.0f} m")

# ============================================================
# 2. Project lat/lon → local meters
# ============================================================
R = 6371000.0
lon0, lat0 = 139.5, -28.0  # SW corner / mid-latitude
cos_lat0 = np.cos(np.deg2rad(lat0))
def to_local(lon_, lat_):
    x = (lon_ - lon0) * np.deg2rad(1.0) * R * cos_lat0
    y = (lat_ - (-28.5)) * np.deg2rad(1.0) * R  # flip so Y increases northward
    return x, y
obs_x_full, obs_y_full = to_local(lon, lat)
dx, dy = to_local(d_lon, d_lat)
print(f"  Local domain: X 0-{obs_x_full.max()/1000:.1f} km, Y 0-{obs_y_full.max()/1000:.1f} km")

# ============================================================
# 3. Regional detrend + calibrate
# ============================================================
A = np.column_stack([np.ones_like(obs_x_full), obs_x_full, obs_y_full])
coef, *_ = np.linalg.lstsq(A, bg_raw, rcond=None)
bg_resid = bg_raw - A @ coef
print(f"  Regional plane: a={coef[0]:.1f}, bx={coef[1]*1e3:.4f}/km, by={coef[2]*1e3:.4f}/km")
print(f"  Residual Bouguer: {bg_resid.min():.2f} to {bg_resid.max():.2f} mGal (std {bg_resid.std():.2f})")

g_cal_full = bg_resid - bg_resid.max()
print(f"  Calibrated (basin signal): {g_cal_full.min():.2f} to 0 mGal  (amplitude {-g_cal_full.min():.1f})")

# ============================================================
# 4. Subsample to tractable resolution (grid-average)
# ============================================================
print(f"\n[2/6] Subsampling {len(obs_x_full):,} stations to ~{SUBSAMPLE_GRID}×{SUBSAMPLE_GRID} grid cells (averaging)...")
xmax, ymax = obs_x_full.max(), obs_y_full.max()
sub_xe = np.linspace(0, xmax, SUBSAMPLE_GRID + 1)
sub_ye = np.linspace(0, ymax, SUBSAMPLE_GRID + 1)
obs_x_list, obs_y_list, obs_g_list = [], [], []
for i in range(SUBSAMPLE_GRID):
    for j in range(SUBSAMPLE_GRID):
        m = ((obs_x_full >= sub_xe[i]) & (obs_x_full < sub_xe[i+1]) &
             (obs_y_full >= sub_ye[j]) & (obs_y_full < sub_ye[j+1]))
        if m.sum() > 0:
            obs_x_list.append(obs_x_full[m].mean())
            obs_y_list.append(obs_y_full[m].mean())
            obs_g_list.append(g_cal_full[m].mean())
obs_x = np.array(obs_x_list); obs_y = np.array(obs_y_list); obs_g = np.array(obs_g_list)
print(f"  After subsample: {len(obs_x)} cells (averaged)")

# ============================================================
# 5. Set up MCMC block grid + interpolate Z-horizon truth at block centers
# ============================================================
print(f"\n[3/6] Block grid {NX}x{NY} + interpolating truth at block centers...")
block_x_edges = np.linspace(0, xmax, NX + 1)
block_y_edges = np.linspace(0, ymax, NY + 1)
bx_km = (block_x_edges[1] - block_x_edges[0]) / 1000
by_km = (block_y_edges[1] - block_y_edges[0]) / 1000
print(f"  Block: {bx_km:.2f} × {by_km:.2f} km")

# Block centers
bxc = 0.5 * (block_x_edges[:-1] + block_x_edges[1:])
byc = 0.5 * (block_y_edges[:-1] + block_y_edges[1:])
BXc, BYc = np.meshgrid(bxc, byc, indexing='ij')

# Interpolate Z-horizon onto block centers
truth_depths = griddata((dx, dy), d_depth, (BXc, BYc), method='linear')
# Fill any NaN at edges with nearest
if np.isnan(truth_depths).any():
    truth_depths_nn = griddata((dx, dy), d_depth, (BXc, BYc), method='nearest')
    truth_depths = np.where(np.isnan(truth_depths), truth_depths_nn, truth_depths)
print(f"  Truth at blocks: {truth_depths.min():.0f} – {truth_depths.max():.0f} m, mean {truth_depths.mean():.0f}")

initial_depths = np.clip(truth_depths, 500, 5500)  # start near truth but MCMC will explore

density_func = make_density_func('exponential', drho_0=DRHO_0, lam=LAMBDA)

# ============================================================
# 6. Run MCMC
# ============================================================
print(f"\n[4/6] MCMC ({N_ITERATIONS:,} iterations, Δρ₀={DRHO_0}, λ={LAMBDA})...")
t0 = time.time()
result = run_mcmc_3d(
    obs_x=obs_x, obs_y=obs_y, gravity_obs=obs_g,
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
# 7. Validation: MCMC vs Z-horizon ground truth
# ============================================================
print(f"\n[5/6] Processing posterior + validation...")
posterior = process_chain_3d(result, burn_in_frac=0.5, thin=1)
mean_d = posterior['mean']
std_d = posterior['std']
ci5 = posterior['ci_5']; ci95 = posterior['ci_95']

# Point-by-point metrics vs truth
err = mean_d - truth_depths
rms = np.sqrt(np.mean(err**2))
mae = np.mean(np.abs(err))
bias = np.mean(err)
corr = np.corrcoef(truth_depths.ravel(), mean_d.ravel())[0, 1]

covered_90 = np.sum((ci5 <= truth_depths) & (truth_depths <= ci95))
cov90 = covered_90 / truth_depths.size

print(f"\n  === VALIDATION (MCMC mean vs GA Z-horizon ground truth) ===")
print(f"  Truth range: {truth_depths.min():.0f} – {truth_depths.max():.0f} m")
print(f"  MCMC range:  {mean_d.min():.0f} – {mean_d.max():.0f} m")
print(f"  RMS error:   {rms:.1f} m")
print(f"  MAE:         {mae:.1f} m")
print(f"  Bias:        {bias:+.1f} m")
print(f"  Correlation: {corr:.3f}")
print(f"  90% CI coverage: {covered_90}/{truth_depths.size} ({cov90*100:.0f}%)")
print(f"  Mean uncertainty: {std_d.mean():.0f} m")

# ============================================================
# 8. Save + plots
# ============================================================
print(f"\n[6/6] Saving + plots...")
os.makedirs(OUT_DIR, exist_ok=True)
np.savez(os.path.join(OUT_DIR, 'results_data.npz'),
         mean_depths=mean_d, std_depths=std_d,
         ci_5=ci5, ci_95=ci95,
         ci_2_5=posterior['ci_2_5'], ci_97_5=posterior['ci_97_5'],
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         obs_x=obs_x, obs_y=obs_y, obs_gravity=obs_g,
         lon=lon, lat=lat, bouguer_raw=bg_raw, bouguer_residual=bg_resid,
         regional_coef=coef,
         truth_depths=truth_depths,
         drho_0=DRHO_0, lam=LAMBDA,
         acceptance_rate=result['acceptance_rate'],
         runtime_min=elapsed/60,
         rms=rms, mae=mae, bias=bias, correlation=corr, coverage_90=cov90,
         all_misfits=np.asarray(result['all_misfits']))

# Validation scatter + residual maps (signature plot for this basin)
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

axes[0].scatter(truth_depths.ravel(), mean_d.ravel(), s=80, alpha=0.7,
                c=std_d.ravel(), cmap='hot_r', edgecolors='k', linewidth=0.3)
lim = max(truth_depths.max(), mean_d.max()) * 1.05
axes[0].plot([0, lim], [0, lim], 'k--', linewidth=1, label='1:1')
axes[0].set_xlabel('GA Z-horizon depth (m)', fontsize=12)
axes[0].set_ylabel('MCMC mean depth (m)', fontsize=12)
axes[0].set_title(f'Validation scatter\nRMS={rms:.0f} m, r={corr:.3f}, 90% cov={cov90*100:.0f}%',
                  fontsize=13, fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_aspect('equal')
axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)

im = axes[1].pcolormesh(block_x_edges/1000, block_y_edges/1000, truth_depths.T,
                        cmap='viridis_r', shading='flat')
plt.colorbar(im, ax=axes[1], label='Depth (m)')
axes[1].set_title('GA Z-horizon ground truth', fontsize=13, fontweight='bold')
axes[1].set_xlabel('X (km)'); axes[1].set_ylabel('Y (km)')
axes[1].set_aspect('equal')

im = axes[2].pcolormesh(block_x_edges/1000, block_y_edges/1000, mean_d.T,
                        cmap='viridis_r', shading='flat')
plt.colorbar(im, ax=axes[2], label='Depth (m)')
axes[2].set_title('MCMC posterior mean', fontsize=13, fontweight='bold')
axes[2].set_xlabel('X (km)')
axes[2].set_aspect('equal')

fig.suptitle('Eromanga/Cooper Basin — MCMC vs GA Basement Ground Truth',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'validation_and_comparison.png'),
            dpi=200, bbox_inches='tight')
plt.close()

# Run full Exp-7 plot suite (reuse generate_cauvery_plots.py via env var)
print("\n  Generating Exp-7 plot suite...")
env = os.environ.copy()
env['CAUVERY_OUT'] = OUT_DIR
subprocess.run([sys.executable, 'generate_cauvery_plots.py'],
               env=env, check=False)

print(f"\n{'='*70}")
print(f"DONE — results in {OUT_DIR}/")
print(f"Headline: RMS {rms:.0f} m, correlation {corr:.3f}, coverage {cov90*100:.0f}%")
print(f"{'='*70}")
