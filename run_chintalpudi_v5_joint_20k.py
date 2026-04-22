"""
Chintalpudi v5 — JOINT depth + λ MCMC (20×20, 20k iter, NO boreholes)
======================================================================
Same setup as run_chintalpudi_v5_50k.py, but:

  - Iterations:   20,000  (vs 50k)
  - λ:            JOINTLY estimated (prob_perturb_lambda > 0)
  - Boreholes:    NONE (already absent in v5_50k — kept that way)
  - Stations:     STRIDE=2 → ~600 averaged stations ("tier 2")

Why a smaller tier than v5_50k (STRIDE=1, 2400 stations)?
  Joint-λ steps recompute *all* 400 blocks → cost ∝ Nx·Ny·M.
  All-stations + joint-λ at 20k iters would take days. Tier 2 keeps it
  to roughly an overnight run on a fast lab CPU.

Literature-grounded fixed Δρ₀ (Chakravarthi & Sundararajan 2007):
  Δρ₀ = -500 kg/m³, prior λ ∈ [1e-4, 1e-3] (broad around their 5e-4 best fit).

Runtime estimate (calibrated against chintalpudi v2: 10×10, 10k iter,
96 stations, prob_λ=0.2, joint → 68 min on M2):

  Per-block forward cost ≈ 19.6 ms × (M / 96)
  Depth iter cost   = 1 × per-block-cost
  λ iter cost       = Nx·Ny × per-block-cost  (full grid recompute)

  This run: 20×20, M≈600, 20k iter, prob_λ=0.10
    depth iters (18,000): ~37 min
    λ iters     (2,000):  ~27 hours
    => ~28 hours on M2 single-thread; ~18–20 h on a faster lab Xeon.

  If too slow, in this file change either:
    - PROB_PERTURB_LAMBDA = 0.05   → roughly halves time (~14 h)
    - STRIDE = 3                   → ~270 stations, ~12 h at prob=0.1
    - Both                         → ~6 h
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
OUT_DIR  = 'results/exp_chintalpudi_v5_joint_20k'

NX, NY            = 20, 20
N_ITERATIONS      = 20_000
STRIDE            = 2                  # tier 2: ~600 stations (vs 2400 with stride=1)

# Density: Δρ₀ fixed (paper value), λ JOINTLY inverted with broad prior
DRHO_0            = -500.0             # Chakravarthi & Sundararajan 2007
LAMBDA_INIT       = 5.0e-4             # Chak 2007 best-fit starting point
LAMBDA_MIN        = 1.0e-4
LAMBDA_MAX        = 1.0e-3
STEP_DEPTH        = 300.0
STEP_LAMBDA       = 3.0e-5
PROB_PERTURB_LAMBDA = 0.10             # 10% λ proposals, 90% depth proposals

NOISE_STD         = 3.0                # digitized-map noise tolerance
SMOOTHNESS_WEIGHT = 3e-5
N_SUBLAYERS       = 10
DEPTH_MIN, DEPTH_MAX = 0.0, 5000.0
BURN_IN_FRAC      = 0.5
POSTERIOR_THIN    = 25                 # keep every 25th post-burn-in sample

ONGC_DEPTH = 2935.0                    # real ground truth (Agarwal 1995, single borehole)


def main():
    print("=" * 72)
    print(f"CHINTALPUDI v5 JOINT — {NX}×{NY} | {N_ITERATIONS:,} iter | "
          f"λ free | NO boreholes")
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
    print(f"  Stations:    {len(obs_x)} (stride={STRIDE} → tier 2)")
    print(f"  Unknowns:    {NX*NY} depths + 1 λ")
    print(f"  Data/param:  {len(obs_x)/(NX*NY+1):.2f}")

    block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
    block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)
    dx_km = (block_x_edges[1] - block_x_edges[0]) / 1000.0
    dy_km = (block_y_edges[1] - block_y_edges[0]) / 1000.0
    print(f"  Block size:  {dx_km:.2f} × {dy_km:.2f} km")

    # ------------------------------------------------------------------
    # Rebin Chakravarthi 2007 reference basement (digitized Fig 5f) onto
    # the 20×20 MCMC grid. NOT measured truth — only ONGC borehole is.
    # ------------------------------------------------------------------
    ref_x = np.loadtxt(os.path.join(DATA_DIR, 'x_coords.txt'))
    ref_y = np.loadtxt(os.path.join(DATA_DIR, 'y_coords.txt'))
    bd_cells = 0.25 * (bd[:-1, :-1] + bd[1:, :-1] + bd[:-1, 1:] + bd[1:, 1:])
    ref_xc = 0.5 * (ref_x[:-1] + ref_x[1:])
    ref_yc = 0.5 * (ref_y[:-1] + ref_y[1:])
    ix_of = np.clip(np.digitize(ref_xc, block_x_edges) - 1, 0, NX - 1)
    iy_of = np.clip(np.digitize(ref_yc, block_y_edges) - 1, 0, NY - 1)

    ref_blocks = np.zeros((NX, NY))
    counts = np.zeros((NX, NY), dtype=int)
    for jj, iy in enumerate(iy_of):
        for ii, ix in enumerate(ix_of):
            ref_blocks[ix, iy] += bd_cells[jj, ii]
            counts[ix, iy] += 1
    ref_blocks = np.where(counts > 0, ref_blocks / np.maximum(counts, 1), np.nan)
    if np.isnan(ref_blocks).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(np.isnan(ref_blocks),
                                     return_distances=False, return_indices=True)
        ref_blocks = ref_blocks[tuple(idx)]
    print(f"  Chak2007 ref: {ref_blocks.min():.0f}–{ref_blocks.max():.0f} m "
          f"(mean {ref_blocks.mean():.0f})")

    # ------------------------------------------------------------------
    # Run JOINT MCMC: depth + λ, no constraints
    # ------------------------------------------------------------------
    initial_depths = np.full((NX, NY), 1500.0)
    print(f"\nMCMC: Δρ₀={DRHO_0} (fixed), λ ∈ [{LAMBDA_MIN}, {LAMBDA_MAX}], "
          f"λ₀={LAMBDA_INIT}")
    print(f"      σ={NOISE_STD} mGal, smooth={SMOOTHNESS_WEIGHT:g}, "
          f"prob_λ={PROB_PERTURB_LAMBDA}")
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
        prob_perturb_lambda=PROB_PERTURB_LAMBDA,
        borehole_constraints=None,             # NO boreholes / depth locks
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        initial_depths=initial_depths, seed=42, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nRuntime:  {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print(f"Accept:   overall {result['acceptance_rate']*100:.1f}%, "
          f"depth {result['depth_acceptance_rate']*100:.1f}%, "
          f"λ {result['lambda_acceptance_rate']*100:.1f}%")

    # ------------------------------------------------------------------
    # Posterior stats (depths + λ)
    # ------------------------------------------------------------------
    post = process_chain_3d_joint(result, burn_in_frac=BURN_IN_FRAC, thin=1)
    mean_d, std_d = post['mean'], post['std']
    ci_lo, ci_hi   = post['ci_5'],  post['ci_95']
    ci_lo2, ci_hi2 = post['ci_2_5'], post['ci_97_5']
    thinned = post['samples'][::POSTERIOR_THIN]
    print(f"  Depth samples saved: {thinned.shape[0]} (thin={POSTERIOR_THIN})")

    lam_mean = float(post['lambda_mean'])
    lam_std  = float(post['lambda_std'])
    lam_ci   = (float(post['lambda_ci_5']), float(post['lambda_ci_95']))

    # Agreement vs Chak 2007 inversion (benchmark, NOT measured truth)
    rms_ref  = float(np.sqrt(np.mean((mean_d - ref_blocks)**2)))
    bias_ref = float(np.mean(mean_d - ref_blocks))
    cov90 = float(np.mean((ref_blocks >= ci_lo) & (ref_blocks <= ci_hi)))
    cov95 = float(np.mean((ref_blocks >= ci_lo2) & (ref_blocks <= ci_hi2)))

    # ONGC borehole (real ground truth) at depocenter
    ib, jb = [int(k) for k in np.unravel_index(np.argmax(ref_blocks), ref_blocks.shape)]
    xc_depo = 0.5*(block_x_edges[ib] + block_x_edges[ib+1])
    yc_depo = 0.5*(block_y_edges[jb] + block_y_edges[jb+1])
    bore_recov = float(mean_d[ib, jb])
    bore_std   = float(std_d[ib, jb])
    err_pct    = 100.0 * (bore_recov - ONGC_DEPTH) / ONGC_DEPTH
    in_ci      = ci_lo[ib, jb] <= ONGC_DEPTH <= ci_hi[ib, jb]

    print(f"\n=== AGREEMENT vs Chakravarthi 2007 (benchmark) ===")
    print(f"  Recovered:   {mean_d.min():.0f}–{mean_d.max():.0f} m "
          f"(Chak {ref_blocks.min():.0f}–{ref_blocks.max():.0f})")
    print(f"  RMS:         {rms_ref:.0f} m  |  Bias: {bias_ref:+.0f} m")
    print(f"  Coverage:    90% {cov90*100:.0f}%  |  95% {cov95*100:.0f}%")
    print(f"  Mean σ:      {std_d.mean():.0f} m")
    print(f"\n=== λ POSTERIOR (jointly inverted) ===")
    print(f"  λ mean:  {lam_mean:.3e} ± {lam_std:.3e} /m")
    print(f"  λ 90%CI: [{lam_ci[0]:.3e}, {lam_ci[1]:.3e}]")
    print(f"  Chak 2007 best fit: 5.0e-04 /m")
    print(f"\n=== ONGC BOREHOLE (real ground truth, single point) ===")
    print(f"  Depocenter block ({ib},{jb}) at ({xc_depo/1000:.1f}, {yc_depo/1000:.1f}) km")
    print(f"  Recovered: {bore_recov:.0f} ± {bore_std:.0f} m  vs ONGC {ONGC_DEPTH:.0f} m")
    print(f"  Error: {err_pct:+.1f}%  |  ONGC within 90% CI: {'YES' if in_ci else 'NO'} "
          f"(CI {ci_lo[ib,jb]:.0f}–{ci_hi[ib,jb]:.0f} m)")

    # ------------------------------------------------------------------
    # Save everything (compatible w/ generate_chintalpudi_v5_plots.py)
    # ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'results_data.npz')
    np.savez_compressed(
        out_path,
        # depth posterior
        mean_depths=mean_d, std_depths=std_d,
        ci_5=ci_lo, ci_95=ci_hi, ci_2_5=ci_lo2, ci_97_5=ci_hi2,
        posterior_samples_thinned=thinned.astype(np.float32),
        posterior_thin=POSTERIOR_THIN, burn_in_frac=BURN_IN_FRAC,
        # λ posterior
        lambda_mean=lam_mean, lambda_std=lam_std,
        lambda_ci_5=lam_ci[0], lambda_ci_95=lam_ci[1],
        lambda_chain=np.asarray(result['all_lambdas']),
        lambda_samples_post=np.asarray(post['lambda_samples']),
        # benchmarks / inputs
        ref_blocks=ref_blocks,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
        # physics + sampler params
        drho_0=DRHO_0, lam=lam_mean,         # 'lam' field for plot-suite compat
        lambda_init=LAMBDA_INIT,
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
        prob_perturb_lambda=PROB_PERTURB_LAMBDA,
        noise_std=NOISE_STD,
        step_depth=STEP_DEPTH, step_lambda=STEP_LAMBDA,
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        # diagnostics
        all_misfits=np.asarray(result['all_misfits']),
        acceptance_rate=result['acceptance_rate'],
        depth_acceptance=result['depth_acceptance_rate'],
        lambda_acceptance=result['lambda_acceptance_rate'],
        n_iterations=N_ITERATIONS, runtime_min=elapsed/60,
        # agreement metrics
        rms_ref=rms_ref, bias_ref=bias_ref,
        coverage_90=cov90, coverage_95=cov95,
        # borehole (ground truth)
        borehole_xy=np.asarray([xc_depo, yc_depo]),
        borehole_depth=ONGC_DEPTH,
        borehole_block=np.asarray([ib, jb]),
        chak2007_reported_depocenter=2830.0,
        # version tag
        experiment='chintalpudi_v5_joint_20k',
        grid_shape=np.asarray([NX, NY]),
    )
    print(f"\nSaved: {out_path}")

    # Generate the full plot suite (re-uses v5 plot generator)
    try:
        from generate_chintalpudi_v5_plots import generate_all
        generate_all(OUT_DIR)
    except Exception as e:
        print(f"  (plot generation skipped: {e})")


if __name__ == '__main__':
    main()
