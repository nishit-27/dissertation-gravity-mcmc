"""
Chintalpudi — FINAL production MCMC inversion.

Method
------
Bayesian Metropolis-Hastings inversion of Chintalpudi sub-basin (Krishna-Godavari,
India) Bouguer gravity for basement depth. 3D rectangular-prism forward model
(Nagy 2000) with depth-dependent density contrast following the Rao (1990) /
Chakravarthi & Sundararajan (2007) parabolic density-depth function:

    Δρ(z) = Δρ₀ · (α / (α + z))²

with published constants for Krishna-Godavari sediments (Δρ₀ = -550 kg/m³,
α = 2000 m). Density law is FIXED; MCMC estimates only the 400 block depths.

NO borehole constraint is used in the inversion. The ONGC well (Agarwal 1995,
2935 m at the depocenter) is used ONLY as a blind post-hoc check in
diagnostics. The digitized Chakravarthi & Sundararajan (2007) Figure 5f
basement map is used as a reference-inversion benchmark (not ground truth).

Configuration
-------------
  grid        : 20 x 20 blocks (400 unknowns)
  stations    : stride=2 → ~650 (data/param ≈ 1.6)
  iterations  : 100,000
  density law : Rao/Chakravarthi parabolic (FIXED)
  burn-in     : 50%, thin = 50 for posterior samples

Outputs (in results/exp_chintalpudi_FINAL/)
-------
  results_data.npz          — mean, std, CI, thinned posterior samples,
                              station data, reference map, config
  01_depth_comparison.png   — 2D depth: ours | Chak 2007 | difference
  02_depth_3d_surface.png   — 3D perspective of depth
  03_uncertainty_map.png    — 2D posterior std
  04_uncertainty_3d_surface.png — 3D depth colored by std
  05_cross_sections.png     — E-W and N-S cross-sections, 90% CI
  06_gravity_fit.png        — observed | predicted | residual
  07_accuracy.png           — recovered vs Chak 2007 scatter + histogram
  08_mcmc_diagnostics.png   — misfit trace + convergence + depocenter posterior

Expected runtime
----------------
Typical desktop (3-4 GHz): ~3-4 hours. Scales linearly with station count
and iteration count.

Usage
-----
    python run_chintalpudi_FINAL.py

Requires real_data/chintalpudi/{x_meshgrid.txt, y_meshgrid.txt,
observed_gravity.txt, basement_depth.txt, x_coords.txt, y_coords.txt}.

References
----------
  Nagy, D. (2000). The gravitational potential and its derivatives for the
      prism. J. Geodesy 74:552-560.
  Rao, D.B. (1986). Modelling of sedimentary basins from gravity anomalies
      with variable density contrast. Geophys. J. R. Astr. Soc. 84:207-212.
  Chakravarthi, V. & Sundararajan, N. (2007). 3D gravity inversion of basement
      relief — a depth-dependent density approach. Geophysics 72(2):I23-I32.
  Agarwal, B.N.P. (1995). ONGC well report, Chintalpudi sub-basin.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d, process_chain_3d

# ======================================================================
# CONFIGURATION
# ======================================================================
DATA_DIR = 'real_data/chintalpudi'
OUT_DIR  = 'results/exp_chintalpudi_FINAL'

# Grid
NX, NY            = 20, 20

# MCMC
N_ITERATIONS      = 100_000
BURN_IN_FRAC      = 0.5
POSTERIOR_THIN    = 50
STEP_DEPTH        = 300.0           # meters — proposal std for depth
SEED              = 42

# Data subsampling (stride=2 gives ~650 stations on 41x61 grid)
STRIDE            = 2

# Rao/Chakravarthi parabolic density (FIXED, not inverted)
DRHO_0            = -550.0          # kg/m³ at surface
ALPHA             = 2000.0          # m — compaction length scale

# Likelihood / prior
NOISE_STD         = 1.5             # mGal — gravity data uncertainty
SMOOTHNESS_WEIGHT = 1e-5            # 2D Laplacian prior
DEPTH_MIN         = 0.0
DEPTH_MAX         = 5000.0
N_SUBLAYERS       = 10              # sublayer count for density integration

# Benchmark
ONGC_DEPTH        = 2935.0          # m — blind check only
CHAK2007_DEPOCENTER = 2830.0        # m — published reference


# ======================================================================
def parabolic_density(z):
    """Rao (1990) / Chakravarthi & Sundararajan (2007) parabolic density law.

        Δρ(z) = Δρ₀ · (α / (α + z))²

    At z=0: Δρ = Δρ₀ (max contrast). At z→∞: Δρ → 0 (full compaction).
    """
    z = np.asarray(z, dtype=float)
    return DRHO_0 * (ALPHA / (ALPHA + z)) ** 2


def _rebin_reference(bd, ref_x, ref_y, block_x_edges, block_y_edges, nx, ny):
    """Rebin the Chakravarthi 2007 digitized basement map to MCMC block grid."""
    from scipy.ndimage import distance_transform_edt
    bd_cells = 0.25 * (bd[:-1, :-1] + bd[1:, :-1] + bd[:-1, 1:] + bd[1:, 1:])
    ref_xc = 0.5 * (ref_x[:-1] + ref_x[1:])
    ref_yc = 0.5 * (ref_y[:-1] + ref_y[1:])
    ix_of = np.clip(np.digitize(ref_xc, block_x_edges) - 1, 0, nx - 1)
    iy_of = np.clip(np.digitize(ref_yc, block_y_edges) - 1, 0, ny - 1)
    agg = np.zeros((nx, ny))
    counts = np.zeros((nx, ny), dtype=int)
    for jj, iy in enumerate(iy_of):
        for ii, ix in enumerate(ix_of):
            agg[ix, iy] += bd_cells[jj, ii]
            counts[ix, iy] += 1
    ref = np.where(counts > 0, agg / np.maximum(counts, 1), np.nan)
    if np.isnan(ref).any():
        idx = distance_transform_edt(np.isnan(ref),
                                     return_distances=False, return_indices=True)
        ref = ref[tuple(idx)]
    return ref


def main():
    print("=" * 76)
    print("CHINTALPUDI — FINAL MCMC INVERSION")
    print(f"  Grid:       {NX}x{NY} = {NX*NY} unknowns")
    print(f"  Iterations: {N_ITERATIONS:,}")
    print(f"  Density:    Rao parabolic  Δρ₀={DRHO_0:.0f} kg/m³, α={ALPHA:.0f} m (FIXED)")
    print(f"  Output:     {OUT_DIR}")
    print("=" * 76)

    # -- Load real data ------------------------------------------------
    xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))
    yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
    gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))
    bd = np.loadtxt(os.path.join(DATA_DIR, 'basement_depth.txt'))
    ref_x = np.loadtxt(os.path.join(DATA_DIR, 'x_coords.txt'))
    ref_y = np.loadtxt(os.path.join(DATA_DIR, 'y_coords.txt'))

    obs_x       = xg[::STRIDE, ::STRIDE].flatten()
    obs_y       = yg[::STRIDE, ::STRIDE].flatten()
    gravity_obs = gv[::STRIDE, ::STRIDE].flatten()

    block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
    block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)
    dx_km = (block_x_edges[1] - block_x_edges[0]) / 1000.0
    dy_km = (block_y_edges[1] - block_y_edges[0]) / 1000.0

    print(f"\n  Stations:    {len(obs_x)} (stride={STRIDE})")
    print(f"  Data/param:  {len(obs_x) / (NX*NY):.2f}")
    print(f"  Block size:  {dx_km:.2f} x {dy_km:.2f} km")
    print(f"  Gravity:     {gravity_obs.min():.2f} to {gravity_obs.max():.2f} mGal")

    # -- Reference (benchmark, NOT ground truth) -----------------------
    ref_blocks = _rebin_reference(bd, ref_x, ref_y,
                                   block_x_edges, block_y_edges, NX, NY)
    print(f"  Chak2007 (rebinned {NX}x{NY}): "
          f"{ref_blocks.min():.0f}-{ref_blocks.max():.0f} m (mean {ref_blocks.mean():.0f})")

    # -- Density sanity table -----------------------------------------
    print(f"\n  Parabolic Δρ(z) table:")
    for z in (0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000):
        print(f"     z = {z:5d} m  →  Δρ = {parabolic_density(z):7.1f} kg/m³")

    # -- Run MCMC ------------------------------------------------------
    print(f"\nStarting MCMC — this will take several hours on a typical desktop.")
    initial_depths = np.full((NX, NY), 1500.0)
    t0 = time.time()
    result = run_mcmc_3d(
        obs_x=obs_x, obs_y=obs_y, gravity_obs=gravity_obs,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        density_func=parabolic_density, noise_std=NOISE_STD,
        n_iterations=N_ITERATIONS, step_size=STEP_DEPTH,
        depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        initial_depths=initial_depths, seed=SEED, verbose=True,
    )
    elapsed = time.time() - t0
    acc = result['acceptance_rate']
    print(f"\nMCMC done: {elapsed/60:.1f} min | accept {acc*100:.1f}%")

    # -- Post-process --------------------------------------------------
    post = process_chain_3d(result, burn_in_frac=BURN_IN_FRAC, thin=1)
    mean_d = post['mean']
    std_d  = post['std']
    ci_5   = post['ci_5']; ci_95   = post['ci_95']
    ci_2_5 = post['ci_2_5']; ci_97_5 = post['ci_97_5']
    samples_thinned = post['samples'][::POSTERIOR_THIN].astype(np.float32)

    # Metrics
    rms_ref  = float(np.sqrt(np.mean((mean_d - ref_blocks) ** 2)))
    bias_ref = float(np.mean(mean_d - ref_blocks))
    cov90    = float(np.mean((ref_blocks >= ci_5) & (ref_blocks <= ci_95)))
    cov95    = float(np.mean((ref_blocks >= ci_2_5) & (ref_blocks <= ci_97_5)))

    ib, jb = [int(k) for k in np.unravel_index(np.argmax(ref_blocks), ref_blocks.shape)]
    xc_d = 0.5 * (block_x_edges[ib] + block_x_edges[ib+1])
    yc_d = 0.5 * (block_y_edges[jb] + block_y_edges[jb+1])
    depo_mean = float(mean_d[ib, jb])
    depo_std  = float(std_d[ib, jb])
    depo_err_pct = 100.0 * (depo_mean - ONGC_DEPTH) / ONGC_DEPTH
    depo_in_ci = bool(ci_5[ib, jb] <= ONGC_DEPTH <= ci_95[ib, jb])

    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"  Depth range:  {mean_d.min():.0f} - {mean_d.max():.0f} m")
    print(f"  vs Chak 2007 (digitized benchmark, NOT ground truth):")
    print(f"    RMS difference: {rms_ref:.0f} m")
    print(f"    Bias:           {bias_ref:+.0f} m")
    print(f"    90% CI coverage: {cov90*100:.0f}%")
    print(f"    95% CI coverage: {cov95*100:.0f}%")
    print(f"  Mean posterior std: {std_d.mean():.0f} m")
    print(f"\n  Blind ONGC borehole check (NOT used in inversion):")
    print(f"    Depocenter block ({ib},{jb}) at ({xc_d/1000:.1f}, {yc_d/1000:.1f}) km")
    print(f"    Recovered:       {depo_mean:.0f} +/- {depo_std:.0f} m")
    print(f"    90% CI:          {ci_5[ib,jb]:.0f} - {ci_95[ib,jb]:.0f} m")
    print(f"    ONGC drill:      {ONGC_DEPTH:.0f} m  (blind)")
    print(f"    Error:           {depo_err_pct:+.1f}%")
    print(f"    ONGC in 90% CI:  {'YES' if depo_in_ci else 'NO'}")
    print(f"    Chak 2007:       {CHAK2007_DEPOCENTER:.0f} m (their inversion)")

    # -- Save ----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_npz = os.path.join(OUT_DIR, 'results_data.npz')
    np.savez_compressed(
        out_npz,
        # posterior stats
        mean_depths=mean_d, std_depths=std_d,
        ci_5=ci_5, ci_95=ci_95, ci_2_5=ci_2_5, ci_97_5=ci_97_5,
        posterior_samples_thinned=samples_thinned,
        posterior_thin=POSTERIOR_THIN, burn_in_frac=BURN_IN_FRAC,
        # benchmark
        ref_blocks=ref_blocks,
        # grid & data
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
        # config (for reproducibility)
        drho_0=DRHO_0, alpha=ALPHA, density_law='rao_parabolic',
        noise_std=NOISE_STD, step_depth=STEP_DEPTH,
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        depth_min=DEPTH_MIN, depth_max=DEPTH_MAX, seed=SEED, stride=STRIDE,
        grid_shape=np.asarray([NX, NY]),
        # diagnostics
        all_misfits=np.asarray(result['all_misfits']),
        acceptance_rate=acc, n_iterations=N_ITERATIONS,
        runtime_min=elapsed/60,
        # metrics
        rms_ref=rms_ref, bias_ref=bias_ref,
        coverage_90=cov90, coverage_95=cov95,
        # blind ground-truth check
        borehole_xy=np.asarray([xc_d, yc_d]),
        borehole_depth=ONGC_DEPTH,
        borehole_block=np.asarray([ib, jb]),
        chak2007_reported_depocenter=CHAK2007_DEPOCENTER,
        experiment='chintalpudi_FINAL',
    )
    print(f"\nSaved npz: {out_npz}")

    # -- Generate full 8-plot suite ------------------------------------
    print("\nGenerating 8-plot suite...")
    from generate_v6_plots import generate_all
    generate_all(OUT_DIR)

    print("\n" + "=" * 76)
    print(f"ALL DONE in {elapsed/60:.1f} min (MCMC) + plotting")
    print(f"Results:  {OUT_DIR}/")
    print("=" * 76)


if __name__ == '__main__':
    main()
