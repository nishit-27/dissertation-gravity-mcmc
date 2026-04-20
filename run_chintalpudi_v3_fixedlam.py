"""
Chintalpudi v3 — Fixed lambda + ONGC borehole constraint (no λ inversion).

Uses run_mcmc_3d_joint with prob_perturb_lambda=0.0, which locks λ at its
initial value — effectively a depth-only inversion that still supports
borehole constraints.

Same 96 stations (stride=5), 10k iterations.
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d_joint, process_chain_3d_joint

DATA_DIR = 'real_data/chintalpudi'
OUT_DIR  = 'results/exp_chintalpudi_v3_fixedlam_borehole'

NX, NY = 10, 10
N_ITERATIONS = 10_000
STRIDE = 5

DRHO_0 = -500.0
LAMBDA_FIXED = 5.0e-4

STEP_DEPTH  = 250.0
NOISE_STD   = 1.5
SMOOTHNESS_WEIGHT = 1e-5
N_SUBLAYERS = 10
DEPTH_MIN, DEPTH_MAX = 0.0, 5000.0
ONGC_BOREHOLE_DEPTH = 2935.0

print("=" * 70)
print("CHINTALPUDI v3 — Fixed-λ MCMC + ONGC borehole constraint")
print("=" * 70)

xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))
yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))
bd = np.loadtxt(os.path.join(DATA_DIR, 'basement_depth.txt'))

obs_x = xg[::STRIDE, ::STRIDE].flatten()
obs_y = yg[::STRIDE, ::STRIDE].flatten()
gravity_obs = gv[::STRIDE, ::STRIDE].flatten()
print(f"  Stations: {len(obs_x)} | Gravity: {gravity_obs.min():.2f} to {gravity_obs.max():.2f} mGal")

block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)

bd_sub = bd[:40, :60]
iy_truth, ix_truth = np.unravel_index(np.argmax(bd_sub), bd_sub.shape)
x_bore = xg[iy_truth, ix_truth]; y_bore = yg[iy_truth, ix_truth]
ix_block = int(np.clip(np.searchsorted(block_x_edges, x_bore) - 1, 0, NX - 1))
iy_block = int(np.clip(np.searchsorted(block_y_edges, y_bore) - 1, 0, NY - 1))
borehole_constraints = {(ix_block, iy_block): ONGC_BOREHOLE_DEPTH}
print(f"  Borehole block ({ix_block},{iy_block}) locked at {ONGC_BOREHOLE_DEPTH} m")

initial_depths = np.full((NX, NY), 1500.0)
initial_depths[ix_block, iy_block] = ONGC_BOREHOLE_DEPTH

print(f"\nRunning MCMC (fixed λ = {LAMBDA_FIXED}, {N_ITERATIONS:,} iters)...")
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
    smoothness_weight=SMOOTHNESS_WEIGHT,
    n_sublayers=N_SUBLAYERS,
    initial_depths=initial_depths,
    seed=42, verbose=True,
)
elapsed = time.time() - t0
print(f"\nRuntime: {elapsed/60:.1f} min | Accept: {result['acceptance_rate']*100:.1f}%")

post = process_chain_3d_joint(result, burn_in_frac=0.5, thin=1)
mean_d, std_d = post['mean'], post['std']

truth_blocks = np.zeros((NX, NY))
by_cells, bx_cells = bd_sub.shape[0]//NY, bd_sub.shape[1]//NX
for i in range(NX):
    for j in range(NY):
        truth_blocks[i,j] = bd_sub[j*by_cells:(j+1)*by_cells,
                                   i*bx_cells:(i+1)*bx_cells].mean()

rms = np.sqrt(np.mean((mean_d - truth_blocks)**2))
bias = float(np.mean(mean_d - truth_blocks))
cov = float(np.mean((truth_blocks >= post['ci_5']) & (truth_blocks <= post['ci_95'])))

print(f"\n=== VALIDATION ===")
print(f"  Recovered: {mean_d.min():.0f}–{mean_d.max():.0f} m | Truth: {truth_blocks.min():.0f}–{truth_blocks.max():.0f}")
print(f"  RMS: {rms:.0f} m | Bias: {bias:+.0f} m | 90% CI coverage: {cov*100:.0f}%")
print(f"  Mean std: {std_d.mean():.0f} m")
print(f"  Borehole block: {mean_d[ix_block,iy_block]:.0f} m (locked {ONGC_BOREHOLE_DEPTH})")
print(f"  Deepest: {mean_d.max():.0f} m (borehole truth {ONGC_BOREHOLE_DEPTH}, Chakravarthi 3100)")

os.makedirs(OUT_DIR, exist_ok=True)
np.savez(os.path.join(OUT_DIR, 'results_data.npz'),
         mean_depths=mean_d, std_depths=std_d,
         ci_5=post['ci_5'], ci_95=post['ci_95'],
         truth_blocks=truth_blocks,
         block_x_edges=block_x_edges, block_y_edges=block_y_edges,
         obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
         borehole_block=(ix_block, iy_block),
         borehole_depth=ONGC_BOREHOLE_DEPTH,
         drho_0=DRHO_0, lam=LAMBDA_FIXED,
         rms=rms, bias=bias, coverage=cov,
         runtime_min=elapsed/60,
         all_misfits=np.asarray(result['all_misfits']))

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
def show(ax, data, cmap, title, vmin=None, vmax=None, label='Depth (m)'):
    im = ax.pcolormesh(block_x_edges/1000, block_y_edges/1000, data.T,
                       cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
    ax.set_title(title)

vmin, vmax = 0, max(mean_d.max(), truth_blocks.max())
show(axes[0,0], mean_d,       'viridis_r', f'Recovered ({mean_d.min():.0f}-{mean_d.max():.0f})', vmin, vmax)
show(axes[0,1], truth_blocks, 'viridis_r', f'Truth ({truth_blocks.min():.0f}-{truth_blocks.max():.0f})', vmin, vmax)
show(axes[1,0], std_d,        'hot_r',     f'Uncertainty (mean {std_d.mean():.0f})', label='Std (m)')

diff = mean_d - truth_blocks
vm = float(np.abs(diff).max())
show(axes[1,1], diff, 'RdBu_r', f'Error (RMS {rms:.0f}, bias {bias:+.0f})', -vm, vm, label='Error (m)')

for a in [axes[0,0], axes[0,1], axes[1,1]]:
    xb = 0.5*(block_x_edges[ix_block]+block_x_edges[ix_block+1])/1000
    yb = 0.5*(block_y_edges[iy_block]+block_y_edges[iy_block+1])/1000
    a.plot(xb, yb, marker='*', color='yellow', markeredgecolor='k',
           markersize=18, linewidth=0, label='ONGC borehole')
    a.legend(loc='lower left', fontsize=8)

fig.suptitle(f'Chintalpudi v3 — Fixed λ={LAMBDA_FIXED} | 96 stn | borehole | 10k',
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'chintalpudi_v3_summary.png'), dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/")
