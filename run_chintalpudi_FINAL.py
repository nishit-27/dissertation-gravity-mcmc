"""
Chintalpudi — FINAL production MCMC inversion + full 8-plot suite.
SINGLE-FILE version: does MCMC and writes all 8 plots. No other scripts needed.

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
2935 m at the depocenter) is used ONLY as a blind post-hoc check. The
digitized Chakravarthi & Sundararajan (2007) Figure 5f basement map is used
as a reference-inversion benchmark (not ground truth).

Configuration
-------------
  grid        : 20 x 20 blocks (400 unknowns)
  stations    : stride=2 → ~650 (data/param ≈ 1.6)
  iterations  : 100,000
  density law : Rao/Chakravarthi parabolic (FIXED)
  burn-in     : 50%, thin = 50 for posterior samples

Outputs (in results/exp_chintalpudi_FINAL/)
-------
  results_data.npz          — full state (posterior, config, metrics, samples)
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
Typical desktop (3-4 GHz): ~3-4 hours.

Usage
-----
    python run_chintalpudi_FINAL.py

Requires real_data/chintalpudi/{x_meshgrid, y_meshgrid, observed_gravity,
basement_depth, x_coords, y_coords}.txt and src/{mcmc_inversion,
forward_model}.py.

References
----------
  Nagy, D. (2000). J. Geodesy 74:552-560.
  Rao, D.B. (1986). Geophys. J. R. Astr. Soc. 84:207-212.
  Chakravarthi, V. & Sundararajan, N. (2007). Geophysics 72(2):I23-I32.
  Agarwal, B.N.P. (1995). ONGC well report, Chintalpudi sub-basin.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d, process_chain_3d
from src.forward_model import compute_gravity_for_basin

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
STEP_DEPTH        = 300.0
SEED              = 42

# Data subsampling (stride=2 gives ~650 stations on 41x61 grid)
STRIDE            = 2

# Rao/Chakravarthi parabolic density (FIXED, not inverted)
DRHO_0            = -550.0
ALPHA             = 2000.0

# Likelihood / prior
NOISE_STD         = 1.5
SMOOTHNESS_WEIGHT = 1e-5
DEPTH_MIN         = 0.0
DEPTH_MAX         = 5000.0
N_SUBLAYERS       = 10

# Benchmarks (blind check only — not used in inversion)
ONGC_DEPTH          = 2935.0
CHAK2007_DEPOCENTER = 2830.0

REF_LABEL  = 'Chakravarthi 2007 (reference inversion)'
ONGC_LABEL = 'ONGC borehole (2935 m, real)'


# ======================================================================
# DENSITY LAW
# ======================================================================
def parabolic_density(z):
    """Rao (1990) / Chakravarthi & Sundararajan (2007) parabolic density law.

        Δρ(z) = Δρ₀ · (α / (α + z))²

    At z=0: Δρ = Δρ₀ (max contrast). At z→∞: Δρ → 0 (full compaction).
    """
    z = np.asarray(z, dtype=float)
    return DRHO_0 * (ALPHA / (ALPHA + z)) ** 2


# ======================================================================
# UTILITIES
# ======================================================================
def _rebin_reference(bd, ref_x, ref_y, block_x_edges, block_y_edges, nx, ny):
    """Rebin the Chakravarthi 2007 digitized basement map to block grid."""
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


# ======================================================================
# PLOT FUNCTIONS (Exp-7 standard 8-plot suite)
# ======================================================================
def plot_01_depth_comparison(ctx):
    mean_d, ref = ctx['mean_d'], ctx['ref_blocks']
    ext = ctx['extent']; xc_d, yc_d, bz = ctx['depo_xyz']
    rms, bias = ctx['rms_ref'], ctx['bias_ref']
    vmax = max(mean_d.max(), ref.max())
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, arr, t in zip(ax[:2], [mean_d, ref],
                         [f'Our MCMC (posterior mean)\n{mean_d.min():.0f}–{mean_d.max():.0f} m',
                          f'{REF_LABEL}\n{ref.min():.0f}–{ref.max():.0f} m']):
        im = a.imshow(arr.T, origin='lower', extent=ext, cmap='viridis_r',
                      vmin=0, vmax=vmax, aspect='auto')
        a.scatter([xc_d], [yc_d], marker='*', s=280, c='gold',
                  edgecolor='k', zorder=5, label=ONGC_LABEL)
        a.set_title(t); a.set_xlabel('X (km)')
        plt.colorbar(im, ax=a, label='Depth (m)')
    ax[0].set_ylabel('Y (km)'); ax[0].legend(loc='lower right', fontsize=8)
    diff = mean_d - ref
    im = ax[2].imshow(diff.T, origin='lower', extent=ext, cmap='RdBu_r',
                     vmin=-1500, vmax=1500, aspect='auto')
    ax[2].scatter([xc_d], [yc_d], marker='*', s=280, c='gold',
                  edgecolor='k', zorder=5)
    ax[2].set_title(f'Difference (ours − Chak)\nRMS {rms:.0f} m, bias {bias:+.0f} m')
    ax[2].set_xlabel('X (km)')
    plt.colorbar(im, ax=ax[2], label='Difference (m)')
    fig.suptitle(f"{ctx['tag']} — 2D depth comparison "
                 f"(★ = ONGC borehole, only real ground truth)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_depth_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_02_depth_3d_surface(ctx):
    mean_d, ref = ctx['mean_d'], ctx['ref_blocks']
    xc, yc = ctx['xc'], ctx['yc']
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    xc_d, yc_d, bz = ctx['depo_xyz']
    fig = plt.figure(figsize=(15, 6))
    for k, (arr, t) in enumerate(zip(
            [mean_d, ref],
            [f'Our MCMC (posterior mean)\n{mean_d.min():.0f}–{mean_d.max():.0f} m',
             f'{REF_LABEL}\n{ref.min():.0f}–{ref.max():.0f} m'])):
        ax = fig.add_subplot(1, 2, k+1, projection='3d')
        s = ax.plot_surface(X, Y, -arr, cmap='viridis_r', alpha=0.9,
                            edgecolor='none')
        ax.scatter([xc_d], [yc_d], [-bz], marker='*', color='gold',
                   edgecolor='k', s=180, depthshade=False)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m)')
        ax.set_title(t); ax.invert_zaxis()
        fig.colorbar(s, ax=ax, shrink=0.6, label='Depth (m)', pad=0.1)
    fig.suptitle(f"{ctx['tag']} — 3D basement surface")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_depth_3d_surface.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def plot_03_uncertainty_map(ctx):
    std_d = ctx['std_d']; ext = ctx['extent']
    xc_d, yc_d, _ = ctx['depo_xyz']
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(std_d.T, origin='lower', extent=ext, cmap='hot_r', aspect='auto')
    ax.scatter([xc_d], [yc_d], marker='*', s=280, c='gold',
               edgecolor='k', zorder=5, label=ONGC_LABEL)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'Posterior uncertainty (std)\n'
                 f'min {std_d.min():.0f}, mean {std_d.mean():.0f}, '
                 f'max {std_d.max():.0f} m')
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax, label='Posterior std (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_uncertainty_map.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_04_uncertainty_3d_surface(ctx):
    mean_d, std_d = ctx['mean_d'], ctx['std_d']
    xc, yc = ctx['xc'], ctx['yc']
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(vmin=std_d.min(), vmax=std_d.max())
    colors = plt.cm.hot_r(norm(std_d))
    ax.plot_surface(X, Y, -mean_d, facecolors=colors, alpha=0.9,
                    edgecolor='none', rstride=1, cstride=1)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='hot_r')
    mappable.set_array(std_d)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label='Posterior std (m)', pad=0.1)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m)')
    ax.set_title(f"{ctx['tag']} — 3D basement colored by uncertainty")
    ax.invert_zaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_uncertainty_3d_surface.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def plot_05_cross_sections(ctx):
    mean_d, ref = ctx['mean_d'], ctx['ref_blocks']
    ci_lo, ci_hi = ctx['ci_5'], ctx['ci_95']
    xc, yc = ctx['xc'], ctx['yc']
    ib, jb = ctx['ib'], ctx['jb']
    xc_d, yc_d, bz = ctx['depo_xyz']
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    ax.plot(xc, mean_d[:, jb], 'b-', lw=2, label='Our MCMC (mean)')
    ax.fill_between(xc, ci_lo[:, jb], ci_hi[:, jb], alpha=0.3, color='b',
                    label='Our 90% CI')
    ax.plot(xc, ref[:, jb], 'k--', lw=2, label=REF_LABEL)
    ax.plot(xc_d, bz, marker='*', color='gold', markeredgecolor='k',
            markersize=18, linewidth=0, label=ONGC_LABEL)
    ax.invert_yaxis(); ax.set_xlabel('X (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'E–W cross-section at Y = {yc_d:.1f} km'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(yc, mean_d[ib, :], 'b-', lw=2, label='Our MCMC (mean)')
    ax.fill_between(yc, ci_lo[ib, :], ci_hi[ib, :], alpha=0.3, color='b',
                    label='Our 90% CI')
    ax.plot(yc, ref[ib, :], 'k--', lw=2, label=REF_LABEL)
    ax.plot(yc_d, bz, marker='*', color='gold', markeredgecolor='k',
            markersize=18, linewidth=0, label=ONGC_LABEL)
    ax.invert_yaxis(); ax.set_xlabel('Y (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'N–S cross-section at X = {xc_d:.1f} km'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"{ctx['tag']} — Cross-sections with 90% CI")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '05_cross_sections.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_06_gravity_fit(ctx):
    obs_x, obs_y, g_obs = ctx['obs_x'], ctx['obs_y'], ctx['g_obs']
    mean_d = ctx['mean_d']
    bx, by = ctx['bx'], ctx['by']
    g_pred = compute_gravity_for_basin(obs_x, obs_y, bx, by, mean_d,
                                       parabolic_density, n_sublayers=N_SUBLAYERS)
    residual = g_obs - g_pred
    rms_g = float(np.sqrt(np.mean(residual**2)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, c, title, vm in [
        (axes[0], g_obs, f'Observed ({g_obs.min():.1f} to {g_obs.max():.1f} mGal)', None),
        (axes[1], g_pred, 'Predicted (from MCMC mean)', None),
        (axes[2], residual, f'Residual  RMS {rms_g:.2f} mGal',
         float(np.abs(residual).max()))]:
        if vm is None:
            sc = ax.scatter(obs_x/1000, obs_y/1000, c=c, cmap='RdBu_r',
                            s=18, vmin=g_obs.min(), vmax=g_obs.max())
        else:
            sc = ax.scatter(obs_x/1000, obs_y/1000, c=c, cmap='RdBu_r',
                            s=18, vmin=-vm, vmax=vm)
        plt.colorbar(sc, ax=ax, label='mGal')
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_aspect('equal'); ax.set_title(title)
    fig.suptitle(f"{ctx['tag']} — Gravity fit  |  "
                 f"Rao parabolic Δρ₀={DRHO_0:.0f}, α={ALPHA:.0f} m")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '06_gravity_fit.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_07_accuracy(ctx):
    mean_d, std_d, ref = ctx['mean_d'], ctx['std_d'], ctx['ref_blocks']
    rms, bias = ctx['rms_ref'], ctx['bias_ref']
    ib, jb = ctx['ib'], ctx['jb']
    _, _, bz = ctx['depo_xyz']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    ax.errorbar(ref.flatten(), mean_d.flatten(), yerr=std_d.flatten(),
                fmt='o', alpha=0.55, ms=3, capsize=2, color='steelblue',
                label='MCMC blocks (± σ)')
    lim = [0, max(ref.max(), mean_d.max())*1.05]
    ax.plot(lim, lim, 'k--', label='y = x (perfect agreement)')
    ax.plot(ref[ib, jb], mean_d[ib, jb], marker='o', color='blue',
            markersize=12, markeredgecolor='k',
            label=f'Depocenter ({mean_d[ib, jb]:.0f} m)')
    ax.axhline(bz, color='gold', ls='--', lw=1.5,
               label=f'ONGC borehole ({int(bz)} m, real)')
    ax.set_xlabel('Chak 2007 reference depth (m)')
    ax.set_ylabel('Our MCMC depth (m)')
    ax.set_title(f'Agreement vs Chak 2007 (RMS {rms:.0f} m)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax = axes[1]
    diff = (mean_d - ref).flatten()
    ax.hist(diff, bins=30, color='salmon', edgecolor='k', alpha=0.85)
    ax.axvline(bias, color='b', lw=2, label=f'bias {bias:+.0f} m')
    ax.set_xlabel('Our − Chak (m)'); ax.set_ylabel('Count')
    ax.set_title(f'Per-block difference distribution (std {diff.std():.0f} m)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(f"{ctx['tag']} — Agreement with published inversion")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '07_accuracy.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_08_diagnostics(ctx):
    misfits = ctx['all_misfits']; acc = ctx['acceptance_rate']
    samples_thin = ctx['samples_thinned']
    ib, jb = ctx['ib'], ctx['jb']
    _, _, bz = ctx['depo_xyz']
    post_mean = float(ctx['mean_d'][ib, jb])
    n = len(misfits)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.semilogy(misfits, lw=0.4, alpha=0.7)
    ax.axvline(n//2, color='r', ls='--', label='burn-in (50%)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Misfit (log)')
    ax.set_title(f'Misfit trace (accept {acc*100:.1f}%)')
    ax.legend(); ax.grid(alpha=0.3, which='both')
    ax = axes[1]
    window = max(200, n // 200)
    running = np.convolve(np.asarray(misfits), np.ones(window)/window, mode='valid')
    ax.plot(np.arange(len(running)) + window, running, lw=0.8)
    ax.set_xlabel('Iteration'); ax.set_ylabel(f'Rolling-mean misfit (w={window})')
    ax.set_title('Misfit convergence'); ax.grid(alpha=0.3)
    ax = axes[2]
    bore_samples = samples_thin[:, ib, jb]
    ax.hist(bore_samples, bins=40, color='slateblue', edgecolor='k', alpha=0.85)
    ax.axvline(bz, color='gold', lw=3, label=ONGC_LABEL)
    ax.axvline(post_mean, color='blue', lw=2,
               label=f'Our posterior mean ({post_mean:.0f} m)')
    ax.axvline(CHAK2007_DEPOCENTER, color='darkgreen', lw=1.5, ls='--',
               label=f'Chak 2007 ({CHAK2007_DEPOCENTER:.0f} m)')
    ax.set_xlabel('Depth (m)'); ax.set_ylabel('Count')
    ax.set_title(f'Posterior at depocenter block ({ib},{jb})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{ctx['tag']} — MCMC diagnostics "
                 f"(right panel: blind ground-truth check)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '08_mcmc_diagnostics.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================
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

    # -- Reference -----------------------------------------------------
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
    mean_d = post['mean']; std_d = post['std']
    ci_5 = post['ci_5']; ci_95 = post['ci_95']
    ci_2_5 = post['ci_2_5']; ci_97_5 = post['ci_97_5']
    samples_thinned = post['samples'][::POSTERIOR_THIN].astype(np.float32)

    rms_ref  = float(np.sqrt(np.mean((mean_d - ref_blocks) ** 2)))
    bias_ref = float(np.mean(mean_d - ref_blocks))
    cov90    = float(np.mean((ref_blocks >= ci_5) & (ref_blocks <= ci_95)))
    cov95    = float(np.mean((ref_blocks >= ci_2_5) & (ref_blocks <= ci_97_5)))

    ib, jb = [int(k) for k in np.unravel_index(np.argmax(ref_blocks), ref_blocks.shape)]
    xc_d = 0.5 * (block_x_edges[ib] + block_x_edges[ib+1]) / 1000.0
    yc_d = 0.5 * (block_y_edges[jb] + block_y_edges[jb+1]) / 1000.0
    depo_mean = float(mean_d[ib, jb]); depo_std = float(std_d[ib, jb])
    depo_err_pct = 100.0 * (depo_mean - ONGC_DEPTH) / ONGC_DEPTH
    depo_in_ci = bool(ci_5[ib, jb] <= ONGC_DEPTH <= ci_95[ib, jb])

    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"  Depth range:  {mean_d.min():.0f} - {mean_d.max():.0f} m")
    print(f"  vs Chak 2007 (digitized benchmark):")
    print(f"    RMS difference:  {rms_ref:.0f} m")
    print(f"    Bias:            {bias_ref:+.0f} m")
    print(f"    90% CI coverage: {cov90*100:.0f}%")
    print(f"    95% CI coverage: {cov95*100:.0f}%")
    print(f"  Mean posterior std: {std_d.mean():.0f} m")
    print(f"\n  Blind ONGC borehole check (NOT used in inversion):")
    print(f"    Depocenter block ({ib},{jb}) at ({xc_d:.1f}, {yc_d:.1f}) km")
    print(f"    Recovered:       {depo_mean:.0f} +/- {depo_std:.0f} m")
    print(f"    90% CI:          {ci_5[ib,jb]:.0f} - {ci_95[ib,jb]:.0f} m")
    print(f"    ONGC drill:      {ONGC_DEPTH:.0f} m")
    print(f"    Error:           {depo_err_pct:+.1f}%")
    print(f"    ONGC in 90% CI:  {'YES' if depo_in_ci else 'NO'}")
    print(f"    Chak 2007:       {CHAK2007_DEPOCENTER:.0f} m (their inversion)")

    # -- Save ----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_npz = os.path.join(OUT_DIR, 'results_data.npz')
    np.savez_compressed(
        out_npz,
        mean_depths=mean_d, std_depths=std_d,
        ci_5=ci_5, ci_95=ci_95, ci_2_5=ci_2_5, ci_97_5=ci_97_5,
        posterior_samples_thinned=samples_thinned,
        posterior_thin=POSTERIOR_THIN, burn_in_frac=BURN_IN_FRAC,
        ref_blocks=ref_blocks,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
        drho_0=DRHO_0, alpha=ALPHA, density_law='rao_parabolic',
        noise_std=NOISE_STD, step_depth=STEP_DEPTH,
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        depth_min=DEPTH_MIN, depth_max=DEPTH_MAX, seed=SEED, stride=STRIDE,
        grid_shape=np.asarray([NX, NY]),
        all_misfits=np.asarray(result['all_misfits']),
        acceptance_rate=acc, n_iterations=N_ITERATIONS,
        runtime_min=elapsed/60,
        rms_ref=rms_ref, bias_ref=bias_ref,
        coverage_90=cov90, coverage_95=cov95,
        borehole_xy=np.asarray([xc_d*1000, yc_d*1000]),
        borehole_depth=ONGC_DEPTH,
        borehole_block=np.asarray([ib, jb]),
        chak2007_reported_depocenter=CHAK2007_DEPOCENTER,
        experiment='chintalpudi_FINAL',
    )
    print(f"\nSaved npz: {out_npz}")

    # -- Generate all 8 plots -----------------------------------------
    print("\nGenerating 8-plot suite...")
    ctx = {
        'tag': 'Chintalpudi FINAL',
        'mean_d': mean_d, 'std_d': std_d,
        'ci_5': ci_5, 'ci_95': ci_95,
        'ref_blocks': ref_blocks,
        'bx': block_x_edges, 'by': block_y_edges,
        'xc': 0.5 * (block_x_edges[:-1] + block_x_edges[1:]) / 1000.0,
        'yc': 0.5 * (block_y_edges[:-1] + block_y_edges[1:]) / 1000.0,
        'extent': [block_x_edges[0]/1000, block_x_edges[-1]/1000,
                   block_y_edges[0]/1000, block_y_edges[-1]/1000],
        'ib': ib, 'jb': jb,
        'depo_xyz': (xc_d, yc_d, ONGC_DEPTH),
        'rms_ref': rms_ref, 'bias_ref': bias_ref,
        'obs_x': obs_x, 'obs_y': obs_y, 'g_obs': gravity_obs,
        'all_misfits': result['all_misfits'],
        'acceptance_rate': acc,
        'samples_thinned': samples_thinned,
    }
    for fn in (plot_01_depth_comparison, plot_02_depth_3d_surface,
               plot_03_uncertainty_map, plot_04_uncertainty_3d_surface,
               plot_05_cross_sections, plot_06_gravity_fit,
               plot_07_accuracy, plot_08_diagnostics):
        try:
            fn(ctx)
            print(f"  ✓ {fn.__name__}")
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")

    print("\n" + "=" * 76)
    print(f"ALL DONE in {elapsed/60:.1f} min (MCMC) + plotting")
    print(f"Results:  {OUT_DIR}/")
    print("=" * 76)


if __name__ == '__main__':
    main()
