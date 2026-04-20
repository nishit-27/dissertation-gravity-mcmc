"""
Chintalpudi v4 — Multi-well constraint + tuning (fixed λ, 96 stations, 10k).

Changes vs v3:
  - 7 borehole constraints distributed across basin (deep, mid, edge)
  - NOISE_STD raised 1.5 -> 3.0 mGal (realistic digitization+model error)
  - STEP_DEPTH raised 250 -> 350 m to lower acceptance toward ~30%
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d_joint, process_chain_3d_joint

DATA_DIR = 'real_data/chintalpudi'
OUT_DIR  = 'results/exp_chintalpudi_v4_multiwell'

NX, NY = 10, 10
N_ITERATIONS = 10_000
STRIDE = 5
DRHO_0, LAMBDA_FIXED = -500.0, 5.0e-4
STEP_DEPTH = 350.0
NOISE_STD  = 3.0
SMOOTHNESS_WEIGHT = 1e-5
N_SUBLAYERS = 10
DEPTH_MIN, DEPTH_MAX = 0.0, 5000.0

print("=" * 70)
print("CHINTALPUDI v4 — Fixed-λ + MULTI-WELL constraint")
print("=" * 70)

xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))
yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))
bd = np.loadtxt(os.path.join(DATA_DIR, 'basement_depth.txt'))

obs_x = xg[::STRIDE, ::STRIDE].flatten()
obs_y = yg[::STRIDE, ::STRIDE].flatten()
gravity_obs = gv[::STRIDE, ::STRIDE].flatten()
print(f"  Stations: {len(obs_x)}")

block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)

# -------------------------------------------------------------
# Build multi-well constraints by block-averaging the truth
# at 7 strategic block locations (deep, mid, edge).
# -------------------------------------------------------------
bd_sub = bd[:40, :60]
truth_blocks = np.zeros((NX, NY))
by_cells, bx_cells = bd_sub.shape[0]//NY, bd_sub.shape[1]//NX
for i in range(NX):
    for j in range(NY):
        truth_blocks[i, j] = bd_sub[j*by_cells:(j+1)*by_cells,
                                    i*bx_cells:(i+1)*bx_cells].mean()

# 7 "wells":  main depocenter + 2 secondary deep + 2 mid-slope + 2 basin edge
well_blocks = [
    tuple(np.unravel_index(np.argmax(truth_blocks), truth_blocks.shape)),  # depocenter
    (3, 3),   # secondary deep (west of depocenter)
    (6, 7),   # secondary deep (east)
    (2, 5),   # mid-slope
    (7, 4),   # mid-slope east
    (1, 1),   # SW edge (shallow)
    (8, 8),   # NE edge (shallow)
]
borehole_constraints = {}
for (i, j) in well_blocks:
    borehole_constraints[(int(i), int(j))] = float(truth_blocks[i, j])
print(f"  Wells: {len(borehole_constraints)}")
for (i,j), z in borehole_constraints.items():
    print(f"    block ({i:>2d},{j:>2d})  depth {z:5.0f} m")

initial_depths = np.full((NX, NY), 1500.0)
for (i, j), z in borehole_constraints.items():
    initial_depths[i, j] = z

print(f"\nRunning MCMC (fixed λ={LAMBDA_FIXED}, noise={NOISE_STD} mGal, "
      f"{N_ITERATIONS:,} iters)...")
t0 = time.time()
result = run_mcmc_3d_joint(
    obs_x=obs_x, obs_y=obs_y, gravity_obs=gravity_obs,
    block_x_edges=block_x_edges, block_y_edges=block_y_edges,
    drho_0=DRHO_0, noise_std=NOISE_STD,
    n_iterations=N_ITERATIONS,
    step_depth=STEP_DEPTH, step_lambda=0.0,
    depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
    lambda_min=LAMBDA_FIXED*0.99, lambda_max=LAMBDA_FIXED*1.01,
    lambda_init=LAMBDA_FIXED,
    prob_perturb_lambda=0.0,
    borehole_constraints=borehole_constraints,
    smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
    initial_depths=initial_depths, seed=42, verbose=True,
)
elapsed = time.time() - t0
print(f"\nRuntime: {elapsed/60:.1f} min | Accept: {result['acceptance_rate']*100:.1f}%")

post = process_chain_3d_joint(result, burn_in_frac=0.5, thin=1)
mean_d, std_d = post['mean'], post['std']
ci_lo, ci_hi = post['ci_5'], post['ci_95']

rms  = float(np.sqrt(np.mean((mean_d - truth_blocks)**2)))
bias = float(np.mean(mean_d - truth_blocks))
cov  = float(np.mean((truth_blocks >= ci_lo) & (truth_blocks <= ci_hi)))
free_mask = np.ones_like(mean_d, dtype=bool)
for (i, j) in borehole_constraints:
    free_mask[i, j] = False
rms_free = float(np.sqrt(np.mean((mean_d - truth_blocks)[free_mask]**2)))
cov_free = float(np.mean(((truth_blocks >= ci_lo) &
                          (truth_blocks <= ci_hi))[free_mask]))

print(f"\n=== VALIDATION (v4 vs v3) ===")
print(f"  Recovered: {mean_d.min():.0f}–{mean_d.max():.0f} m  (truth {truth_blocks.min():.0f}–{truth_blocks.max():.0f})")
print(f"  Overall RMS: {rms:.0f} m  |  free-blocks only: {rms_free:.0f} m  (v3 was 594 m)")
print(f"  Bias: {bias:+.0f} m")
print(f"  90% CI coverage — all: {cov*100:.0f}%  |  free blocks: {cov_free*100:.0f}%  (v3: 25%)")
print(f"  Mean uncertainty: {std_d.mean():.0f} m")

os.makedirs(OUT_DIR, exist_ok=True)
np.savez(os.path.join(OUT_DIR, 'results_data.npz'),
         mean_depths=mean_d, std_depths=std_d,
         ci_5=ci_lo, ci_95=ci_hi, truth_blocks=truth_blocks,
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
         well_blocks=np.array(list(borehole_constraints.keys())),
         well_depths=np.array(list(borehole_constraints.values())),
         drho_0=DRHO_0, lam=LAMBDA_FIXED,
         rms=rms, rms_free=rms_free, bias=bias,
         coverage=cov, coverage_free=cov_free,
         runtime_min=elapsed/60,
         all_misfits=np.asarray(result['all_misfits']))

# quick summary plot
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
def show(ax, data, cmap, title, vmin=None, vmax=None, label='Depth (m)'):
    im = ax.pcolormesh(block_x_edges/1000, block_y_edges/1000, data.T,
                       cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
    ax.set_title(title)

vmin, vmax = 0, max(mean_d.max(), truth_blocks.max())
show(axes[0,0], mean_d, 'viridis_r', f'Recovered ({mean_d.min():.0f}-{mean_d.max():.0f})', vmin, vmax)
show(axes[0,1], truth_blocks, 'viridis_r', f'Truth', vmin, vmax)
show(axes[1,0], std_d, 'hot_r', f'Uncertainty (mean {std_d.mean():.0f})', label='Std (m)')
diff = mean_d - truth_blocks
vm = float(np.abs(diff).max())
show(axes[1,1], diff, 'RdBu_r', f'Error (RMS {rms:.0f}, free-RMS {rms_free:.0f})', -vm, vm, label='Error (m)')

# mark all wells
for a in [axes[0,0], axes[0,1], axes[1,1]]:
    for (i, j) in borehole_constraints:
        xb = 0.5*(block_x_edges[i]+block_x_edges[i+1])/1000
        yb = 0.5*(block_y_edges[j]+block_y_edges[j+1])/1000
        a.plot(xb, yb, marker='*', color='yellow',
               markeredgecolor='k', markersize=16, linewidth=0)

fig.suptitle(f'Chintalpudi v4 — 7-well constraint | 96 stn | fixed λ | 10k',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'chintalpudi_v4_summary.png'), dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/")
