"""
Chintalpudi v5 — 20x20 unconstrained (NO borehole) — 100k iterations.

Identical to run_chintalpudi_v5_50k.py except:
  - N_ITERATIONS = 100_000 (publication-quality mixing)
  - OUT_DIR      = 'results/exp_chintalpudi_v5_100k'
  - POSTERIOR_THIN = 50 (same total thinned-sample count)

Use this when you want tighter posteriors. Expected runtime ~2× the 50k run.
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d_joint, process_chain_3d_joint

# ======================================================================
# CONFIG
# ======================================================================
DATA_DIR = 'real_data/chintalpudi'
OUT_DIR  = 'results/exp_chintalpudi_v5_100k'

NX, NY            = 20, 20
N_ITERATIONS      = 100_000
STRIDE            = 1
DRHO_0            = -500.0
LAMBDA_FIXED      = 5.0e-4
STEP_DEPTH        = 300.0
NOISE_STD         = 3.0
SMOOTHNESS_WEIGHT = 3e-5
N_SUBLAYERS       = 10
DEPTH_MIN, DEPTH_MAX = 0.0, 5000.0
BURN_IN_FRAC      = 0.5
POSTERIOR_THIN    = 50


def main():
    print("=" * 72)
    print(f"CHINTALPUDI v5 — {NX}×{NY} unconstrained | {N_ITERATIONS:,} iters")
    print("=" * 72)

    xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))
    yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
    gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))
    bd = np.loadtxt(os.path.join(DATA_DIR, 'basement_depth.txt'))

    obs_x       = xg[::STRIDE, ::STRIDE].flatten()
    obs_y       = yg[::STRIDE, ::STRIDE].flatten()
    gravity_obs = gv[::STRIDE, ::STRIDE].flatten()
    print(f"  Stations:    {len(obs_x)} (stride={STRIDE})")
    print(f"  Unknowns:    {NX*NY}")
    print(f"  Data/param:  {len(obs_x)/(NX*NY):.2f}")

    block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
    block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)
    dx_km = (block_x_edges[1] - block_x_edges[0]) / 1000.0
    dy_km = (block_y_edges[1] - block_y_edges[0]) / 1000.0
    print(f"  Block size:  {dx_km:.2f} × {dy_km:.2f} km")

    truth_x = np.loadtxt(os.path.join(DATA_DIR, 'x_coords.txt'))
    truth_y = np.loadtxt(os.path.join(DATA_DIR, 'y_coords.txt'))
    bd_cells = 0.25 * (bd[:-1, :-1] + bd[1:, :-1] + bd[:-1, 1:] + bd[1:, 1:])
    truth_xc = 0.5 * (truth_x[:-1] + truth_x[1:])
    truth_yc = 0.5 * (truth_y[:-1] + truth_y[1:])
    ix_of = np.clip(np.digitize(truth_xc, block_x_edges) - 1, 0, NX - 1)
    iy_of = np.clip(np.digitize(truth_yc, block_y_edges) - 1, 0, NY - 1)

    truth_blocks = np.zeros((NX, NY))
    counts = np.zeros((NX, NY), dtype=int)
    for jj, iy in enumerate(iy_of):
        for ii, ix in enumerate(ix_of):
            truth_blocks[ix, iy] += bd_cells[jj, ii]
            counts[ix, iy] += 1
    truth_blocks = np.where(counts > 0,
                            truth_blocks / np.maximum(counts, 1), np.nan)
    if np.isnan(truth_blocks).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(np.isnan(truth_blocks),
                                     return_distances=False, return_indices=True)
        truth_blocks = truth_blocks[tuple(idx)]
    print(f"  Truth rebin: {truth_blocks.min():.0f}–{truth_blocks.max():.0f} m "
          f"(mean {truth_blocks.mean():.0f})")

    initial_depths = np.full((NX, NY), 1500.0)
    print(f"\nMCMC: fixed λ={LAMBDA_FIXED}, σ={NOISE_STD} mGal, "
          f"smooth={SMOOTHNESS_WEIGHT:g}")
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
        borehole_constraints=None,
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        initial_depths=initial_depths, seed=42, verbose=True,
    )
    elapsed = time.time() - t0
    acc = result['acceptance_rate']
    print(f"\nRuntime: {elapsed/60:.1f} min | Accept: {acc*100:.1f}%")

    post = process_chain_3d_joint(result, burn_in_frac=BURN_IN_FRAC, thin=1)
    mean_d, std_d = post['mean'], post['std']
    ci_lo, ci_hi = post['ci_5'], post['ci_95']
    ci_lo2, ci_hi2 = post['ci_2_5'], post['ci_97_5']
    thinned = post['samples'][::POSTERIOR_THIN]
    print(f"  Samples saved: {thinned.shape[0]} (thin={POSTERIOR_THIN})")

    rms  = float(np.sqrt(np.mean((mean_d - truth_blocks)**2)))
    bias = float(np.mean(mean_d - truth_blocks))
    cov90 = float(np.mean((truth_blocks >= ci_lo) & (truth_blocks <= ci_hi)))
    cov95 = float(np.mean((truth_blocks >= ci_lo2) & (truth_blocks <= ci_hi2)))

    borehole_xy = (30_000.0, 25_000.0)
    ib = int(np.clip(np.digitize([borehole_xy[0]], block_x_edges)[0] - 1, 0, NX-1))
    jb = int(np.clip(np.digitize([borehole_xy[1]], block_y_edges)[0] - 1, 0, NY-1))
    bore_depth_recov = float(mean_d[ib, jb])
    bore_depth_std   = float(std_d[ib, jb])

    print(f"\n=== VALIDATION (truth rebinned — NOT a constraint) ===")
    print(f"  Recovered:   {mean_d.min():.0f}–{mean_d.max():.0f} m "
          f"(truth {truth_blocks.min():.0f}–{truth_blocks.max():.0f})")
    print(f"  RMS:         {rms:.0f} m")
    print(f"  Bias:        {bias:+.0f} m")
    print(f"  90% CI cov:  {cov90*100:.0f}%")
    print(f"  95% CI cov:  {cov95*100:.0f}%")
    print(f"  Mean σ:      {std_d.mean():.0f} m")
    print(f"  ONGC block: recovered {bore_depth_recov:.0f} ± {bore_depth_std:.0f} m "
          f"vs reported 2935 m")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'results_data.npz')
    np.savez_compressed(
        out_path,
        mean_depths=mean_d, std_depths=std_d,
        ci_5=ci_lo, ci_95=ci_hi, ci_2_5=ci_lo2, ci_97_5=ci_hi2,
        posterior_samples_thinned=thinned.astype(np.float32),
        posterior_thin=POSTERIOR_THIN, burn_in_frac=BURN_IN_FRAC,
        truth_blocks=truth_blocks,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
        drho_0=DRHO_0, lam=LAMBDA_FIXED, noise_std=NOISE_STD,
        step_depth=STEP_DEPTH, smoothness_weight=SMOOTHNESS_WEIGHT,
        n_sublayers=N_SUBLAYERS,
        all_misfits=np.asarray(result['all_misfits']),
        acceptance_rate=acc, n_iterations=N_ITERATIONS,
        runtime_min=elapsed/60,
        rms=rms, bias=bias, coverage_90=cov90, coverage_95=cov95,
        borehole_xy=np.asarray(borehole_xy), borehole_depth=2935.0,
        borehole_block=np.asarray([ib, jb]),
        experiment='chintalpudi_v5_100k',
        grid_shape=np.asarray([NX, NY]),
    )
    print(f"\nSaved: {out_path}")
    print(f"Now run: python generate_chintalpudi_v5_plots.py {OUT_DIR}")


if __name__ == '__main__':
    main()
