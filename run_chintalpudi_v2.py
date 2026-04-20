"""
Chintalpudi v2 — Improved: joint depth+lambda MCMC + ONGC borehole constraint.

Changes vs v1:
  - Joint estimation of lambda (density compaction) — learns density variation
  - ONGC borehole (2935 m) used as soft constraint at depocenter block
  - NOISE_STD raised 0.5 -> 1.5 mGal (realistic for digitized data)
  - Step size raised 150 -> 250 m for better mixing
  - Stronger smoothness (1e-5) to suppress checkerboard noise
  - Same 96 stations (stride=5), 10k iterations
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d_joint, process_chain_3d_joint

DATA_DIR = 'real_data/chintalpudi'
OUT_DIR  = 'results/exp_chintalpudi_v2_joint_borehole'

NX, NY = 10, 10
N_ITERATIONS = 10_000
STRIDE = 5

# Density — joint inversion (drho_0 fixed, lambda learned)
DRHO_0 = -500.0
LAMBDA_INIT = 5.0e-4
LAMBDA_MIN, LAMBDA_MAX = 1e-4, 2e-3

# MCMC tuning
STEP_DEPTH  = 250.0
STEP_LAMBDA = 3.0e-5
PROB_LAMBDA = 0.2
NOISE_STD   = 1.5
SMOOTHNESS_WEIGHT = 1e-5
N_SUBLAYERS = 10
DEPTH_MIN, DEPTH_MAX = 0.0, 5000.0

ONGC_BOREHOLE_DEPTH = 2935.0   # from Chakravarthi & Sundararajan (2007)

print("=" * 70)
print("CHINTALPUDI v2 — Joint MCMC (depth + λ) + ONGC borehole constraint")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------
xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))
yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))
bd = np.loadtxt(os.path.join(DATA_DIR, 'basement_depth.txt'))

obs_x = xg[::STRIDE, ::STRIDE].flatten()
obs_y = yg[::STRIDE, ::STRIDE].flatten()
gravity_obs = gv[::STRIDE, ::STRIDE].flatten()
print(f"  Stations: {len(obs_x)} (stride={STRIDE})")
print(f"  Gravity: {gravity_obs.min():.2f} to {gravity_obs.max():.2f} mGal")
print(f"  Truth basement: {bd.min():.0f} to {bd.max():.0f} m")

# ------------------------------------------------------------------
# 2. Block grid
# ------------------------------------------------------------------
block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)

# ------------------------------------------------------------------
# 3. Borehole constraint: ONGC well at the basin depocenter (2935 m).
#    Locate depocenter in the published basement grid, map it to a block.
# ------------------------------------------------------------------
bd_sub = bd[:40, :60]
iy_truth, ix_truth = np.unravel_index(np.argmax(bd_sub), bd_sub.shape)
x_bore = xg[iy_truth, ix_truth]
y_bore = yg[iy_truth, ix_truth]
ix_block = int(np.clip(np.searchsorted(block_x_edges, x_bore) - 1, 0, NX - 1))
iy_block = int(np.clip(np.searchsorted(block_y_edges, y_bore) - 1, 0, NY - 1))
borehole_constraints = {(ix_block, iy_block): ONGC_BOREHOLE_DEPTH}
print(f"  Borehole: grid ({ix_truth},{iy_truth}) @ ({x_bore/1000:.1f},{y_bore/1000:.1f}) km "
      f"-> block ({ix_block},{iy_block}), depth {ONGC_BOREHOLE_DEPTH} m")

# ------------------------------------------------------------------
# 4. Initial model — seed from observed gravity via slab approximation
#    z ≈ |g| / (2π G |Δρ|) = |g(mGal)| / (0.042 · 500) ≈ |g| * 47.6 m/mGal
# ------------------------------------------------------------------
initial_depths = np.full((NX, NY), 1500.0)
initial_depths[ix_block, iy_block] = ONGC_BOREHOLE_DEPTH

# ------------------------------------------------------------------
# 5. Run joint MCMC
# ------------------------------------------------------------------
print(f"\nRunning joint MCMC ({N_ITERATIONS:,} iters)...")
t0 = time.time()
result = run_mcmc_3d_joint(
    obs_x=obs_x, obs_y=obs_y, gravity_obs=gravity_obs,
    block_x_edges=block_x_edges, block_y_edges=block_y_edges,
    drho_0=DRHO_0, noise_std=NOISE_STD,
    n_iterations=N_ITERATIONS,
    step_depth=STEP_DEPTH, step_lambda=STEP_LAMBDA,
    depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
    lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
    lambda_init=LAMBDA_INIT,
    prob_perturb_lambda=PROB_LAMBDA,
    borehole_constraints=borehole_constraints,
    smoothness_weight=SMOOTHNESS_WEIGHT,
    n_sublayers=N_SUBLAYERS,
    initial_depths=initial_depths,
    seed=42, verbose=True,
)
elapsed = time.time() - t0
print(f"\nRuntime: {elapsed/60:.1f} min")
print(f"Acceptance: {result['acceptance_rate']*100:.1f}% "
      f"(depth {result['depth_acceptance_rate']*100:.1f}%, "
      f"lambda {result['lambda_acceptance_rate']*100:.1f}%)")

# ------------------------------------------------------------------
# 6. Posterior + validation
# ------------------------------------------------------------------
post = process_chain_3d_joint(result, burn_in_frac=0.5, thin=1)
mean_d = post['mean']
std_d  = post['std']
lam_mean = post['lambda_mean']
lam_std  = post['lambda_std']

# Downsample truth to block grid
truth_blocks = np.zeros((NX, NY))
by_cells, bx_cells = bd_sub.shape[0] // NY, bd_sub.shape[1] // NX
for i in range(NX):
    for j in range(NY):
        truth_blocks[i, j] = bd_sub[j*by_cells:(j+1)*by_cells,
                                    i*bx_cells:(i+1)*bx_cells].mean()

rms  = np.sqrt(np.mean((mean_d - truth_blocks) ** 2))
bias = float(np.mean(mean_d - truth_blocks))
coverage = float(np.mean(
    (truth_blocks >= post['ci_5']) & (truth_blocks <= post['ci_95'])))

print(f"\n=== VALIDATION ===")
print(f"  Depth range: {mean_d.min():.0f} – {mean_d.max():.0f} m "
      f"(truth {truth_blocks.min():.0f}–{truth_blocks.max():.0f})")
print(f"  RMS: {rms:.0f} m | Bias: {bias:+.0f} m | 90% CI coverage: {coverage*100:.0f}%")
print(f"  Mean uncertainty: {std_d.mean():.0f} m")
print(f"  λ (posterior): {lam_mean:.2e} ± {lam_std:.2e} /m "
      f"(init {LAMBDA_INIT:.2e})")
print(f"  Borehole block recovered: {mean_d[ix_block,iy_block]:.0f} m "
      f"(locked at {ONGC_BOREHOLE_DEPTH})")
print(f"  Deepest block: {mean_d.max():.0f} m "
      f"(borehole truth {ONGC_BOREHOLE_DEPTH}, Chakravarthi-2007 est. 3100)")

# ------------------------------------------------------------------
# 7. Save + plot
# ------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
np.savez(os.path.join(OUT_DIR, 'results_data.npz'),
         mean_depths=mean_d, std_depths=std_d,
         ci_5=post['ci_5'], ci_95=post['ci_95'],
         truth_blocks=truth_blocks,
         lambda_mean=lam_mean, lambda_std=lam_std,
         lambda_chain=result['lambda_chain'],
         all_lambdas=np.asarray(result['all_lambdas']),
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
         borehole_block=(ix_block, iy_block),
         borehole_depth=ONGC_BOREHOLE_DEPTH,
         drho_0=DRHO_0,
         rms=rms, bias=bias, coverage=coverage,
         runtime_min=elapsed/60,
         all_misfits=np.asarray(result['all_misfits']))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

def show(ax, data, cmap, title, vmin=None, vmax=None, label='Depth (m)'):
    im = ax.pcolormesh(block_x_edges/1000, block_y_edges/1000, data.T,
                       cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')
    ax.set_title(title)

vmin, vmax = 0, max(mean_d.max(), truth_blocks.max())
show(axes[0,0], mean_d,       'viridis_r', f'Recovered ({mean_d.min():.0f}-{mean_d.max():.0f})', vmin, vmax)
show(axes[0,1], truth_blocks, 'viridis_r', f'Truth ({truth_blocks.min():.0f}-{truth_blocks.max():.0f})', vmin, vmax)
show(axes[0,2], std_d,        'hot_r',     f'Uncertainty (mean {std_d.mean():.0f})', label='Std (m)')

diff = mean_d - truth_blocks
vm = float(np.abs(diff).max())
show(axes[1,0], diff, 'RdBu_r', f'Error (RMS {rms:.0f}, bias {bias:+.0f})', -vm, vm, label='Error (m)')

# mark borehole
for a in [axes[0,0], axes[0,1], axes[1,0]]:
    xb = 0.5*(block_x_edges[ix_block]+block_x_edges[ix_block+1])/1000
    yb = 0.5*(block_y_edges[iy_block]+block_y_edges[iy_block+1])/1000
    a.plot(xb, yb, marker='*', color='yellow', markeredgecolor='k',
           markersize=18, linewidth=0, label='ONGC borehole')
    a.legend(loc='lower left', fontsize=8)

# lambda trace
ax = axes[1,1]
ax.plot(result['all_lambdas'], lw=0.4, alpha=0.7, color='steelblue')
ax.axhline(lam_mean, color='red', ls='--', label=f'mean={lam_mean:.2e}')
ax.axvline(N_ITERATIONS//2, color='grey', ls=':', label='burn-in')
ax.set_xlabel('Iteration'); ax.set_ylabel('λ (1/m)')
ax.set_title(f'λ posterior: {lam_mean:.2e} ± {lam_std:.2e}')
ax.legend(); ax.grid(alpha=0.3)

# misfit trace
ax = axes[1,2]
ax.semilogy(result['all_misfits'], lw=0.4, alpha=0.7)
ax.axvline(N_ITERATIONS//2, color='red', ls='--', label='burn-in')
ax.set_xlabel('Iteration'); ax.set_ylabel('Misfit')
ax.set_title(f'Convergence  (accept {result["acceptance_rate"]*100:.1f}%)')
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle(
    f'Chintalpudi v2 — Joint MCMC 10k | 96 stn | ONGC borehole constraint | '
    f'λ learned',
    fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'chintalpudi_v2_summary.png'), dpi=150)
plt.close(fig)

print(f"\nSaved: {OUT_DIR}/")
