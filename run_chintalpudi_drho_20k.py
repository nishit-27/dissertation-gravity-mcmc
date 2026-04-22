"""
Chintalpudi — JOINT depth + constant-Δρ MCMC (PRODUCTION, 20k iter)
====================================================================
Production-quality version of run_chintalpudi_drho_quicktest.py.

Density model:  Δρ(z) = constant  (NO λ, NO compaction)
Estimating:     100 block depths + 1 global Δρ
Constraints:    NONE (no boreholes, no priors beyond uniform bounds)

Quick-test result (5k iter, 96 stations, 8 min):
  - Δρ converged to -308.5 ± 5.2 kg/m³ (tight unimodal posterior)
  - Depocenter recovered 2415 ± 391 m  vs ONGC truth 2935 m
  - ONGC INSIDE 90% CI [1951, 3078]  (first time across all runs)
  - Gravity RMS residual 1.69 mGal (vs 3 mGal noise — excellent fit)

This production run scales to:
  - 20,000 iterations (4× more for depocenter mixing)
  - STRIDE=4 → ~150 stations (1.5:1 data/param, better-conditioned)
  - Saves thinned posterior samples (for depocenter histogram in plot 08)
  - Auto-calls v5-style plot suite at end

Expected runtime: ~50–60 min on M2-class CPU.

Usage:
    python run_chintalpudi_drho_20k.py
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d_joint_drho, process_chain_3d_joint_drho

# ============================================================
DATA_DIR = 'real_data/chintalpudi'
OUT_DIR  = 'results/exp_chintalpudi_drho_20k'

NX, NY            = 10, 10
N_ITERATIONS      = 20_000
STRIDE            = 4                    # ~150 stations (1.5:1 data/param)

# Constant Δρ joint inversion (NO lambda, NO depth-decay)
DRHO_INIT         = -200.0               # mid-prior start
DRHO_MIN          = -500.0
DRHO_MAX          = -50.0
STEP_DRHO         = 6.0                  # slightly tighter than quicktest (better mixing in 20k)
PROB_PERTURB_DRHO = 0.10

STEP_DEPTH        = 250.0
NOISE_STD         = 3.0
SMOOTHNESS_WEIGHT = 1e-6                 # very weak (don't flatten depocenter)
N_SUBLAYERS       = 5                    # constant Δρ → fewer sublayers OK
DEPTH_MIN, DEPTH_MAX = 0.0, 5500.0
BURN_IN_FRAC      = 0.5
POSTERIOR_THIN    = 25                   # save every 25th post-burn-in sample

ONGC_DEPTH = 2935.0


def main():
    print("=" * 72)
    print(f"CHINTALPUDI PRODUCTION — depth + constant Δρ joint MCMC")
    print(f"  {NX}×{NY} grid | {N_ITERATIONS:,} iter | STRIDE={STRIDE}")
    print("=" * 72)

    # Load
    xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))
    yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
    gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))
    bd = np.loadtxt(os.path.join(DATA_DIR, 'basement_depth.txt'))

    obs_x = xg[::STRIDE, ::STRIDE].flatten()
    obs_y = yg[::STRIDE, ::STRIDE].flatten()
    gravity_obs = gv[::STRIDE, ::STRIDE].flatten()
    print(f"  Stations: {len(obs_x)}  | Unknowns: {NX*NY} depths + 1 Δρ")
    print(f"  Data/param: {len(obs_x)/(NX*NY+1):.2f}")
    print(f"  Gravity range: {gravity_obs.min():.1f} to {gravity_obs.max():.1f} mGal")

    block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
    block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)
    dx_km = (block_x_edges[1] - block_x_edges[0]) / 1000.0
    dy_km = (block_y_edges[1] - block_y_edges[0]) / 1000.0
    print(f"  Block size: {dx_km:.2f} × {dy_km:.2f} km")

    # Rebin Chak 2007 reference (digitized Fig 5f) for benchmarking
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
    ib, jb = [int(k) for k in np.unravel_index(np.argmax(ref_blocks), ref_blocks.shape)]
    xc_depo = 0.5*(block_x_edges[ib] + block_x_edges[ib+1])
    yc_depo = 0.5*(block_y_edges[jb] + block_y_edges[jb+1])
    print(f"  Chak2007 ref: {ref_blocks.min():.0f}–{ref_blocks.max():.0f} m "
          f"(depocenter block ({ib},{jb}))")

    # MCMC
    initial_depths = np.full((NX, NY), 1500.0)
    print(f"\nMCMC: Δρ joint, prior [{DRHO_MIN}, {DRHO_MAX}], init {DRHO_INIT}")
    print(f"      σ={NOISE_STD} mGal, smooth={SMOOTHNESS_WEIGHT:g}, "
          f"prob_Δρ={PROB_PERTURB_DRHO}")
    t0 = time.time()
    result = run_mcmc_3d_joint_drho(
        obs_x=obs_x, obs_y=obs_y, gravity_obs=gravity_obs,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        noise_std=NOISE_STD,
        n_iterations=N_ITERATIONS,
        step_depth=STEP_DEPTH, step_drho=STEP_DRHO,
        depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
        drho_min=DRHO_MIN, drho_max=DRHO_MAX, drho_init=DRHO_INIT,
        prob_perturb_drho=PROB_PERTURB_DRHO,
        smoothness_weight=SMOOTHNESS_WEIGHT,
        n_sublayers=N_SUBLAYERS,
        initial_depths=initial_depths,
        seed=42, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nRuntime: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print(f"Accept:   overall {result['acceptance_rate']*100:.1f}%, "
          f"depth {result['depth_acceptance_rate']*100:.1f}%, "
          f"Δρ {result['drho_acceptance_rate']*100:.1f}%")

    # Posterior
    post = process_chain_3d_joint_drho(result, burn_in_frac=BURN_IN_FRAC)
    mean_d, std_d = post['mean'], post['std']
    ci_lo, ci_hi   = post['ci_5'],  post['ci_95']
    ci_lo2, ci_hi2 = post['ci_2_5'], post['ci_97_5']
    thinned = post['samples'][::POSTERIOR_THIN]
    drho_mean = post['drho_mean']
    drho_std  = post['drho_std']
    drho_ci   = (post['drho_ci_5'], post['drho_ci_95'])

    # Metrics
    rms_ref  = float(np.sqrt(np.mean((mean_d - ref_blocks)**2)))
    bias_ref = float(np.mean(mean_d - ref_blocks))
    cov90 = float(np.mean((ref_blocks >= ci_lo) & (ref_blocks <= ci_hi)))
    cov95 = float(np.mean((ref_blocks >= ci_lo2) & (ref_blocks <= ci_hi2)))

    bore_recov = float(mean_d[ib, jb])
    bore_std   = float(std_d[ib, jb])
    err_pct    = 100.0 * (bore_recov - ONGC_DEPTH) / ONGC_DEPTH
    in_ci      = ci_lo[ib, jb] <= ONGC_DEPTH <= ci_hi[ib, jb]

    print(f"\n=== AGREEMENT vs Chakravarthi 2007 (benchmark) ===")
    print(f"  Recovered: {mean_d.min():.0f}–{mean_d.max():.0f} m  "
          f"(Chak {ref_blocks.min():.0f}–{ref_blocks.max():.0f})")
    print(f"  RMS:       {rms_ref:.0f} m  | Bias: {bias_ref:+.0f} m")
    print(f"  Coverage:  90% {cov90*100:.0f}% | 95% {cov95*100:.0f}%")
    print(f"  Mean σ:    {std_d.mean():.0f} m")
    print(f"\n=== Δρ POSTERIOR (jointly inverted) ===")
    print(f"  Δρ mean:   {drho_mean:.1f} ± {drho_std:.1f} kg/m³")
    print(f"  Δρ 90%CI:  [{drho_ci[0]:.1f}, {drho_ci[1]:.1f}]")
    print(f"\n=== ONGC borehole (real ground truth, single point) ===")
    print(f"  Depocenter ({ib},{jb}) at ({xc_depo/1000:.1f}, {yc_depo/1000:.1f}) km")
    print(f"  Recovered: {bore_recov:.0f} ± {bore_std:.0f} m  vs ONGC {ONGC_DEPTH:.0f} m")
    print(f"  Error: {err_pct:+.1f}%  | ONGC in 90% CI: "
          f"{'YES' if in_ci else 'NO'}  ({ci_lo[ib,jb]:.0f}–{ci_hi[ib,jb]:.0f} m)")

    # Save (compatible with generate_chintalpudi_drho_plots.py)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'results_data.npz')
    np.savez_compressed(
        out_path,
        # depth posterior
        mean_depths=mean_d, std_depths=std_d,
        ci_5=ci_lo, ci_95=ci_hi, ci_2_5=ci_lo2, ci_97_5=ci_hi2,
        posterior_samples_thinned=thinned.astype(np.float32),
        posterior_thin=POSTERIOR_THIN, burn_in_frac=BURN_IN_FRAC,
        # Δρ posterior
        drho_mean=drho_mean, drho_std=drho_std,
        drho_ci_5=drho_ci[0], drho_ci_95=drho_ci[1],
        drho_chain=np.asarray(result['all_drhos']),
        drho_samples_post=np.asarray(post['drho_samples']),
        # benchmarks / inputs
        ref_blocks=ref_blocks,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
        # physics + sampler params
        drho_init=DRHO_INIT, drho_min=DRHO_MIN, drho_max=DRHO_MAX,
        prob_perturb_drho=PROB_PERTURB_DRHO,
        noise_std=NOISE_STD,
        step_depth=STEP_DEPTH, step_drho=STEP_DRHO,
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        # diagnostics
        all_misfits=np.asarray(result['all_misfits']),
        acceptance_rate=result['acceptance_rate'],
        depth_acceptance=result['depth_acceptance_rate'],
        drho_acceptance=result['drho_acceptance_rate'],
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
        experiment='chintalpudi_drho_20k',
        grid_shape=np.asarray([NX, NY]),
    )
    print(f"\nSaved: {out_path}")

    # Auto-generate v5-style plot suite
    try:
        from generate_chintalpudi_drho_plots import generate_all
        generate_all(OUT_DIR)
    except Exception as e:
        print(f"  (plot generation skipped: {e})")
        print(f"  Re-run later with: python generate_chintalpudi_drho_plots.py {OUT_DIR}")


if __name__ == '__main__':
    main()
