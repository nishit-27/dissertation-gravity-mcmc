"""
Chintalpudi v5 — 20x20 unconstrained (NO borehole) — 50k iterations.

Literature-grounded parameters (Chakravarthi & Sundararajan 2007, Geophysics 72(2)):
  - Δρ₀ = -500 kg/m³  (exact paper value for Chintalpudi)
  - λ   = 5.0e-4 /m   (best exp-fit to their parabolic density function)
  - σ   = 3.0 mGal    (digitized-map noise tolerance)

Design:
  - 20×20 = 400 blocks  (block size ≈ 3.0 × 2.0 km, matches gravity resolution limit)
  - All 2400 stations   (stride=1 — 6:1 data/unknown oversampling)
  - No borehole constraints (fully data-driven)
  - Fixed λ (same functional form as Chak 2007)

Outputs everything needed for `generate_chintalpudi_v5_plots.py` to produce
the full Exp-7 plot suite without re-running MCMC.
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
OUT_DIR  = 'results/exp_chintalpudi_v5_50k'

NX, NY            = 20, 20
N_ITERATIONS      = 50_000
STRIDE            = 1                  # use ALL 2400 stations
DRHO_0            = -500.0             # Chakravarthi & Sundararajan 2007
LAMBDA_FIXED      = 5.0e-4             # exp fit to their parabolic PDF
STEP_DEPTH        = 300.0
NOISE_STD         = 3.0
SMOOTHNESS_WEIGHT = 3e-5
N_SUBLAYERS       = 10
DEPTH_MIN, DEPTH_MAX = 0.0, 5000.0
BURN_IN_FRAC      = 0.5
POSTERIOR_THIN    = 25                 # keep every 25th post-burn-in sample


def main():
    print("=" * 72)
    print(f"CHINTALPUDI v5 — {NX}×{NY} unconstrained | {N_ITERATIONS:,} iters")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Rebin ground truth onto NX×NY block grid (validation only)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Run MCMC
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Posterior stats + thinned samples
    # ------------------------------------------------------------------
    post = process_chain_3d_joint(result, burn_in_frac=BURN_IN_FRAC, thin=1)
    mean_d, std_d = post['mean'], post['std']
    ci_lo, ci_hi = post['ci_5'], post['ci_95']
    ci_lo2, ci_hi2 = post['ci_2_5'], post['ci_97_5']

    # thinned posterior samples for future histograms / re-analysis
    thinned = post['samples'][::POSTERIOR_THIN]
    print(f"  Samples saved: {thinned.shape[0]} (thin={POSTERIOR_THIN})")

    rms  = float(np.sqrt(np.mean((mean_d - truth_blocks)**2)))
    bias = float(np.mean(mean_d - truth_blocks))
    cov90 = float(np.mean((truth_blocks >= ci_lo) & (truth_blocks <= ci_hi)))
    cov95 = float(np.mean((truth_blocks >= ci_lo2) & (truth_blocks <= ci_hi2)))

    # borehole pixel (ONGC 2.935 km) — nearest block center
    borehole_xy = (30_000.0, 25_000.0)   # approx depocenter, see Chak 2007 Fig 5a
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

    # ------------------------------------------------------------------
    # Save everything needed to regenerate plots
    # ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'results_data.npz')
    np.savez_compressed(
        out_path,
        # posterior stats
        mean_depths=mean_d, std_depths=std_d,
        ci_5=ci_lo, ci_95=ci_hi, ci_2_5=ci_lo2, ci_97_5=ci_hi2,
        posterior_samples_thinned=thinned.astype(np.float32),
        posterior_thin=POSTERIOR_THIN, burn_in_frac=BURN_IN_FRAC,
        # truth and geometry
        truth_blocks=truth_blocks,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        # inputs
        obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
        # physics params
        drho_0=DRHO_0, lam=LAMBDA_FIXED, noise_std=NOISE_STD,
        step_depth=STEP_DEPTH, smoothness_weight=SMOOTHNESS_WEIGHT,
        n_sublayers=N_SUBLAYERS,
        # diagnostics
        all_misfits=np.asarray(result['all_misfits']),
        acceptance_rate=acc, n_iterations=N_ITERATIONS,
        runtime_min=elapsed/60,
        # validation metrics
        rms=rms, bias=bias, coverage_90=cov90, coverage_95=cov95,
        # borehole reference (for plots — not used in inversion)
        borehole_xy=np.asarray(borehole_xy), borehole_depth=2935.0,
        borehole_block=np.asarray([ib, jb]),
        # version tag
        experiment='chintalpudi_v5_50k',
        grid_shape=np.asarray([NX, NY]),
    )
    print(f"\nSaved: {out_path}")
    print(f"Now run: python generate_chintalpudi_v5_plots.py {OUT_DIR}")


if __name__ == '__main__':
    main()
