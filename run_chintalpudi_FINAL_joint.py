"""
Chintalpudi — FINAL joint MCMC: basement depths + α (compaction length).
SINGLE-FILE: does MCMC and writes all 9 plots (8 standard + α diagnostics).

Difference from run_chintalpudi_FINAL.py
----------------------------------------
FINAL holds the density law fixed:   α = 2000 m (literature).
FINAL_joint estimates α from data:   α ~ Uniform(1000, 3500) m.

Δρ₀ stays fixed at -550 kg/m³ (tightly constrained from rock physics).
All other settings (grid, stations, iterations, smoothness, noise) identical.

Density law (Rao 1990 / Chakravarthi & Sundararajan 2007):
    Δρ(z) = Δρ₀ · (α / (α + z))²

This run propagates the uncertainty in α into the depth posterior, producing
honest joint UQ. Expect slightly wider depth CIs than the fixed-α FINAL.

Configuration
-------------
  grid        : 20 x 20 (400 depth unknowns + 1 α)
  stations    : stride=2  (~650 stations)
  iterations  : 100,000
  α prior     : Uniform(1000, 3500) m
  proposal    : depth σ=300 m (80%), α σ=200 m (20%)

Outputs (in results/exp_chintalpudi_FINAL_joint/)
-------
  results_data.npz          — posterior (depths + α), config, samples
  01–08 .png                — standard 8-plot suite (same as FINAL)
  09_alpha_posterior.png    — α trace + posterior histogram + prior band

Expected runtime: ~3.5–4.5 hours on a typical desktop (slightly longer
than FINAL because α-proposals require full forward recompute).

References
----------
  Rao, D.B. (1986, 1990). Geophys. J. R. Astr. Soc.
  Chakravarthi, V. & Sundararajan, N. (2007). Geophysics 72(2):I23–I32.
  Pallero, J.L.G. et al. (2017). Geophys. J. Int. — joint density/depth.
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
from src.mcmc_inversion import run_mcmc_3d_rao_joint, process_chain_3d_rao_joint
from src.forward_model import compute_gravity_for_basin

# ======================================================================
# CONFIGURATION
# ======================================================================
DATA_DIR = 'real_data/chintalpudi'
OUT_DIR  = 'results/exp_chintalpudi_FINAL_joint'

# Grid
NX, NY            = 20, 20

# MCMC
N_ITERATIONS      = 100_000
BURN_IN_FRAC      = 0.5
POSTERIOR_THIN    = 50
STEP_DEPTH        = 300.0
STEP_ALPHA        = 200.0
PROB_PERTURB_ALPHA = 0.2
SEED              = 42

# Data subsampling
STRIDE            = 2

# Rao parabolic density — α is FREE, Δρ₀ FIXED
DRHO_0            = -550.0              # kg/m³ (fixed from rock physics)
ALPHA_PRIOR_MIN   = 1000.0              # m  (non-informative bounds)
ALPHA_PRIOR_MAX   = 3500.0              # m
ALPHA_INIT        = 2000.0              # m  (Chakravarthi 2007 literature value)

# Likelihood / prior on depths
NOISE_STD         = 1.5
SMOOTHNESS_WEIGHT = 1e-5
DEPTH_MIN         = 0.0
DEPTH_MAX         = 5000.0
N_SUBLAYERS       = 10

# Benchmarks (blind check only)
ONGC_DEPTH          = 2935.0
CHAK2007_DEPOCENTER = 2830.0

REF_LABEL  = 'Chakravarthi 2007 (reference inversion)'
ONGC_LABEL = 'ONGC borehole (2935 m, real)'


# ======================================================================
# UTILITIES
# ======================================================================
def parabolic_density(z, alpha):
    """Δρ(z; α) = Δρ₀·(α/(α+z))²  — used for post-run plotting only."""
    z = np.asarray(z, dtype=float)
    return DRHO_0 * (alpha / (alpha + z)) ** 2


def _rebin_reference(bd, ref_x, ref_y, bxe, bye, nx, ny):
    from scipy.ndimage import distance_transform_edt
    bd_cells = 0.25 * (bd[:-1, :-1] + bd[1:, :-1] + bd[:-1, 1:] + bd[1:, 1:])
    rxc = 0.5 * (ref_x[:-1] + ref_x[1:])
    ryc = 0.5 * (ref_y[:-1] + ref_y[1:])
    ix_of = np.clip(np.digitize(rxc, bxe) - 1, 0, nx - 1)
    iy_of = np.clip(np.digitize(ryc, bye) - 1, 0, ny - 1)
    agg = np.zeros((nx, ny))
    cnt = np.zeros((nx, ny), dtype=int)
    for jj, iy in enumerate(iy_of):
        for ii, ix in enumerate(ix_of):
            agg[ix, iy] += bd_cells[jj, ii]
            cnt[ix, iy] += 1
    ref = np.where(cnt > 0, agg / np.maximum(cnt, 1), np.nan)
    if np.isnan(ref).any():
        idx = distance_transform_edt(np.isnan(ref),
                                     return_distances=False, return_indices=True)
        ref = ref[tuple(idx)]
    return ref


# ======================================================================
# PLOTS 01–08  (v5 style — identical to FINAL)
# ======================================================================
def plot_01(ctx):
    mean_d, ref = ctx['mean_d'], ctx['ref_blocks']
    bx, by = ctx['bx']/1000, ctx['by']/1000
    xc_d, yc_d, bz = ctx['depo_xyz']
    rms, bias = ctx['rms_ref'], ctx['bias_ref']
    vmax = max(mean_d.max(), ref.max())
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, arr, t in zip(ax[:2], [mean_d, ref],
                         [f'Our MCMC (posterior mean)\n{mean_d.min():.0f}–{mean_d.max():.0f} m',
                          f'{REF_LABEL}\n{ref.min():.0f}–{ref.max():.0f} m']):
        pm = a.pcolormesh(bx, by, arr.T, cmap='viridis_r',
                          vmin=0, vmax=vmax, shading='flat')
        a.scatter([xc_d], [yc_d], marker='*', s=200, c='gold',
                  edgecolors='k', zorder=5, label=ONGC_LABEL)
        a.set_title(t); a.set_xlabel('X (km)'); a.set_aspect('equal')
        plt.colorbar(pm, ax=a, label='Depth (m)')
    ax[0].set_ylabel('Y (km)'); ax[0].legend(loc='lower right', fontsize=8)
    diff = mean_d - ref
    pm = ax[2].pcolormesh(bx, by, diff.T, cmap='RdBu_r',
                          vmin=-1500, vmax=1500, shading='flat')
    ax[2].scatter([xc_d], [yc_d], marker='*', s=200, c='gold',
                  edgecolors='k', zorder=5)
    ax[2].set_title(f'Difference (ours − Chak)\nRMS {rms:.0f} m, bias {bias:+.0f} m')
    ax[2].set_xlabel('X (km)'); ax[2].set_aspect('equal')
    plt.colorbar(pm, ax=ax[2], label='Difference (m)')
    fig.suptitle(f"{ctx['tag']} — 2D depth comparison "
                 f"(★ = ONGC borehole, only real ground truth)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '01_depth_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_02(ctx):
    mean_d, ref = ctx['mean_d'], ctx['ref_blocks']
    xc, yc = ctx['xc'], ctx['yc']
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    xc_d, yc_d, bz = ctx['depo_xyz']
    fig = plt.figure(figsize=(16, 7))
    for k, (arr, t) in enumerate(zip(
            [mean_d, ref],
            [f'Our MCMC (posterior mean)\n{mean_d.min():.0f}–{mean_d.max():.0f} m',
             f'{REF_LABEL}\n{ref.min():.0f}–{ref.max():.0f} m'])):
        ax = fig.add_subplot(1, 2, k+1, projection='3d')
        s = ax.plot_surface(X, Y, -arr, cmap='viridis_r',
                            edgecolor='k', linewidth=0.15, alpha=0.95)
        ax.scatter([xc_d], [yc_d], [-bz], marker='*', color='gold',
                   edgecolors='k', s=200, depthshade=False)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m)')
        ax.set_title(t); ax.view_init(elev=28, azim=-120)
        fig.colorbar(s, ax=ax, shrink=0.6, label='Depth (m)', pad=0.1)
    fig.suptitle(f"{ctx['tag']} — 3D basement surface")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '02_depth_3d_surface.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def plot_03(ctx):
    std_d = ctx['std_d']
    bx, by = ctx['bx']/1000, ctx['by']/1000
    xc_d, yc_d, _ = ctx['depo_xyz']
    fig, ax = plt.subplots(figsize=(9, 7))
    pm = ax.pcolormesh(bx, by, std_d.T, cmap='hot_r', shading='flat')
    ax.set_aspect('equal')
    ax.scatter([xc_d], [yc_d], marker='*', s=200, c='gold',
               edgecolors='k', zorder=5, label=ONGC_LABEL)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'Posterior uncertainty (std)\n'
                 f'min {std_d.min():.0f}, mean {std_d.mean():.0f}, '
                 f'max {std_d.max():.0f} m')
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(pm, ax=ax, label='Posterior std (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_uncertainty_map.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_04(ctx):
    mean_d, std_d = ctx['mean_d'], ctx['std_d']
    xc, yc = ctx['xc'], ctx['yc']
    xc_d, yc_d, bz = ctx['depo_xyz']
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    fig = plt.figure(figsize=(10.5, 7))
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(vmin=std_d.min(), vmax=std_d.max())
    colors = plt.cm.hot_r(norm(std_d))
    ax.plot_surface(X, Y, -mean_d, facecolors=colors,
                    edgecolor='k', linewidth=0.15, alpha=0.95,
                    rstride=1, cstride=1)
    ax.scatter([xc_d], [yc_d], [-bz], marker='*', color='gold',
               edgecolors='k', s=200, depthshade=False)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='hot_r')
    mappable.set_array(std_d)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label='Posterior std (m)', pad=0.1)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m)')
    ax.set_title(f"{ctx['tag']} — 3D basement colored by uncertainty\n"
                 f"σ: min {std_d.min():.0f}, mean {std_d.mean():.0f}, "
                 f"max {std_d.max():.0f} m")
    ax.view_init(elev=28, azim=-120)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_uncertainty_3d_surface.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def plot_05(ctx):
    mean_d, ref = ctx['mean_d'], ctx['ref_blocks']
    ci_lo, ci_hi = ctx['ci_5'], ctx['ci_95']
    xc, yc = ctx['xc'], ctx['yc']
    ib, jb = ctx['ib'], ctx['jb']
    xc_d, yc_d, bz = ctx['depo_xyz']
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    ax.plot(xc, mean_d[:, jb], 'b-', lw=2, label='Our MCMC (mean)')
    ax.fill_between(xc, ci_lo[:, jb], ci_hi[:, jb],
                    color='lightsteelblue', alpha=0.6, label='Our 90% CI')
    ax.plot(xc, ref[:, jb], 'k--', lw=2, label=REF_LABEL)
    ax.plot(xc_d, bz, marker='*', color='gold', markeredgecolor='k',
            markersize=18, linewidth=0, label=ONGC_LABEL)
    ax.invert_yaxis(); ax.set_xlabel('X (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'E–W cross-section at Y = {yc_d:.1f} km'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(yc, mean_d[ib, :], 'b-', lw=2, label='Our MCMC (mean)')
    ax.fill_between(yc, ci_lo[ib, :], ci_hi[ib, :],
                    color='lightsteelblue', alpha=0.6, label='Our 90% CI')
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


def plot_06(ctx):
    obs_x, obs_y, g_obs = ctx['obs_x'], ctx['obs_y'], ctx['g_obs']
    mean_d = ctx['mean_d']; alpha_mean = ctx['alpha_mean']
    bx, by = ctx['bx'], ctx['by']
    dens_func = lambda z: parabolic_density(z, alpha_mean)
    g_pred = compute_gravity_for_basin(obs_x, obs_y, bx, by, mean_d,
                                       dens_func, n_sublayers=N_SUBLAYERS)
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
                 f"Rao parabolic Δρ₀={DRHO_0:.0f}, α={alpha_mean:.0f} m (posterior mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '06_gravity_fit.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_07(ctx):
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
    ax.hist(diff, bins=30, color='indianred', edgecolor='k', alpha=0.85)
    ax.axvline(bias, color='b', lw=2, label=f'bias {bias:+.0f} m')
    ax.set_xlabel('Our − Chak (m)'); ax.set_ylabel('Count')
    ax.set_title(f'Per-block difference distribution (std {diff.std():.0f} m)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(f"{ctx['tag']} — Agreement with published inversion")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '07_accuracy.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_08(ctx):
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


def plot_09_alpha(ctx):
    """α trace + posterior histogram + prior band."""
    all_alphas = ctx['all_alphas']
    alpha_samples = ctx['alpha_samples']
    a_mean = ctx['alpha_mean']; a_std = ctx['alpha_std']
    a_ci5 = ctx['alpha_ci_5']; a_ci95 = ctx['alpha_ci_95']
    a_acc = ctx['alpha_acceptance_rate']
    n = len(all_alphas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    ax.plot(all_alphas, lw=0.5, alpha=0.8, color='slateblue')
    ax.axhline(a_mean, color='blue', lw=2, label=f'posterior mean {a_mean:.0f} m')
    ax.axhspan(a_ci5, a_ci95, color='lightsteelblue', alpha=0.4,
               label=f'90% CI [{a_ci5:.0f}, {a_ci95:.0f}]')
    ax.axvline(n//2, color='r', ls='--', label='burn-in (50%)')
    ax.axhline(ALPHA_PRIOR_MIN, color='grey', ls=':', label='prior bounds')
    ax.axhline(ALPHA_PRIOR_MAX, color='grey', ls=':')
    ax.set_xlabel('Iteration'); ax.set_ylabel('α (m)')
    ax.set_title(f'α trace  (α-acceptance {a_acc*100:.1f}%)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.hist(alpha_samples, bins=40, color='slateblue', edgecolor='k', alpha=0.85)
    ax.axvline(a_mean, color='blue', lw=2, label=f'mean {a_mean:.0f}')
    ax.axvline(a_ci5, color='b', ls='--', lw=1.5)
    ax.axvline(a_ci95, color='b', ls='--', lw=1.5,
               label=f'90% CI')
    ax.axvline(2000, color='green', ls='--', lw=1.5,
               label='Chakravarthi 2007 (2000)')
    ax.set_xlabel('α (m)'); ax.set_ylabel('Count')
    ax.set_title(f'α posterior (mean {a_mean:.0f} ± {a_std:.0f})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{ctx['tag']} — α (compaction length) diagnostics")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '09_alpha_posterior.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 76)
    print("CHINTALPUDI — FINAL JOINT MCMC (depths + α)")
    print(f"  Grid:       {NX}x{NY} = {NX*NY} depth unknowns + 1 α")
    print(f"  Iterations: {N_ITERATIONS:,}")
    print(f"  Density:    Rao parabolic  Δρ₀={DRHO_0:.0f} (fixed), "
          f"α ~ U({ALPHA_PRIOR_MIN:.0f}, {ALPHA_PRIOR_MAX:.0f}) m")
    print(f"  α init:     {ALPHA_INIT:.0f} m")
    print(f"  Output:     {OUT_DIR}")
    print("=" * 76)

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
    print(f"  Data/param:  {len(obs_x) / (NX*NY + 1):.2f}")
    print(f"  Block size:  {dx_km:.2f} x {dy_km:.2f} km")
    print(f"  Gravity:     {gravity_obs.min():.2f} to {gravity_obs.max():.2f} mGal")

    ref_blocks = _rebin_reference(bd, ref_x, ref_y,
                                   block_x_edges, block_y_edges, NX, NY)
    print(f"  Chak2007 ({NX}x{NY}): {ref_blocks.min():.0f}-{ref_blocks.max():.0f} m "
          f"(mean {ref_blocks.mean():.0f})")

    initial_depths = np.full((NX, NY), 1500.0)
    print(f"\nStarting joint MCMC (may take 3.5–4.5 hours).")
    t0 = time.time()
    result = run_mcmc_3d_rao_joint(
        obs_x=obs_x, obs_y=obs_y, gravity_obs=gravity_obs,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        drho_0=DRHO_0, noise_std=NOISE_STD,
        n_iterations=N_ITERATIONS,
        step_depth=STEP_DEPTH, step_alpha=STEP_ALPHA,
        depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
        alpha_min=ALPHA_PRIOR_MIN, alpha_max=ALPHA_PRIOR_MAX,
        alpha_init=ALPHA_INIT,
        prob_perturb_alpha=PROB_PERTURB_ALPHA,
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        initial_depths=initial_depths, seed=SEED, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nMCMC done: {elapsed/60:.1f} min")

    post = process_chain_3d_rao_joint(result, burn_in_frac=BURN_IN_FRAC, thin=1)
    mean_d = post['mean']; std_d = post['std']
    ci_5, ci_95 = post['ci_5'], post['ci_95']
    ci_2_5, ci_97_5 = post['ci_2_5'], post['ci_97_5']
    samples_thinned = post['samples'][::POSTERIOR_THIN].astype(np.float32)
    alpha_samples = post['alpha_samples']
    a_mean = post['alpha_mean']; a_std = post['alpha_std']
    a_ci5, a_ci95 = post['alpha_ci_5'], post['alpha_ci_95']

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
    print(f"  α posterior:   {a_mean:.0f} ± {a_std:.0f} m  "
          f"(90% CI {a_ci5:.0f}–{a_ci95:.0f})  [prior U({ALPHA_PRIOR_MIN:.0f}, {ALPHA_PRIOR_MAX:.0f})]")
    print(f"  α-acceptance:  {result['alpha_acceptance_rate']*100:.1f}%")
    print(f"\n  Depth range:   {mean_d.min():.0f} - {mean_d.max():.0f} m")
    print(f"  vs Chak 2007 benchmark:")
    print(f"    RMS:    {rms_ref:.0f} m   Bias: {bias_ref:+.0f} m   "
          f"90% cov {cov90*100:.0f}%, 95% cov {cov95*100:.0f}%")
    print(f"  Mean posterior σ (depth): {std_d.mean():.0f} m")
    print(f"\n  Blind ONGC borehole check:")
    print(f"    Depocenter ({ib},{jb}) at ({xc_d:.1f}, {yc_d:.1f}) km")
    print(f"    Recovered: {depo_mean:.0f} ± {depo_std:.0f} m  "
          f"(CI {ci_5[ib,jb]:.0f}–{ci_95[ib,jb]:.0f})")
    print(f"    ONGC:      {ONGC_DEPTH:.0f} m   err {depo_err_pct:+.1f}%   "
          f"in 90% CI: {'YES' if depo_in_ci else 'NO'}")
    print(f"    Chak 2007: {CHAK2007_DEPOCENTER:.0f} m")

    # -- Save ----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_npz = os.path.join(OUT_DIR, 'results_data.npz')
    np.savez_compressed(
        out_npz,
        mean_depths=mean_d, std_depths=std_d,
        ci_5=ci_5, ci_95=ci_95, ci_2_5=ci_2_5, ci_97_5=ci_97_5,
        posterior_samples_thinned=samples_thinned,
        alpha_samples=alpha_samples, all_alphas=np.asarray(result['all_alphas']),
        alpha_mean=a_mean, alpha_std=a_std, alpha_ci_5=a_ci5, alpha_ci_95=a_ci95,
        posterior_thin=POSTERIOR_THIN, burn_in_frac=BURN_IN_FRAC,
        ref_blocks=ref_blocks,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
        drho_0=DRHO_0, alpha_prior_min=ALPHA_PRIOR_MIN, alpha_prior_max=ALPHA_PRIOR_MAX,
        alpha_init=ALPHA_INIT, density_law='rao_parabolic_joint_alpha',
        noise_std=NOISE_STD, step_depth=STEP_DEPTH, step_alpha=STEP_ALPHA,
        prob_perturb_alpha=PROB_PERTURB_ALPHA,
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        depth_min=DEPTH_MIN, depth_max=DEPTH_MAX, seed=SEED, stride=STRIDE,
        grid_shape=np.asarray([NX, NY]),
        all_misfits=np.asarray(result['all_misfits']),
        acceptance_rate=result['acceptance_rate'],
        depth_acceptance_rate=result['depth_acceptance_rate'],
        alpha_acceptance_rate=result['alpha_acceptance_rate'],
        n_iterations=N_ITERATIONS, runtime_min=elapsed/60,
        rms_ref=rms_ref, bias_ref=bias_ref,
        coverage_90=cov90, coverage_95=cov95,
        borehole_xy=np.asarray([xc_d*1000, yc_d*1000]),
        borehole_depth=ONGC_DEPTH,
        borehole_block=np.asarray([ib, jb]),
        chak2007_reported_depocenter=CHAK2007_DEPOCENTER,
        experiment='chintalpudi_FINAL_joint',
    )
    print(f"\nSaved npz: {out_npz}")

    # -- Plots ---------------------------------------------------------
    print("\nGenerating 9-plot suite...")
    ctx = {
        'tag': 'Chintalpudi FINAL_joint',
        'mean_d': mean_d, 'std_d': std_d,
        'ci_5': ci_5, 'ci_95': ci_95,
        'ref_blocks': ref_blocks,
        'bx': block_x_edges, 'by': block_y_edges,
        'xc': 0.5 * (block_x_edges[:-1] + block_x_edges[1:]) / 1000.0,
        'yc': 0.5 * (block_y_edges[:-1] + block_y_edges[1:]) / 1000.0,
        'ib': ib, 'jb': jb,
        'depo_xyz': (xc_d, yc_d, ONGC_DEPTH),
        'rms_ref': rms_ref, 'bias_ref': bias_ref,
        'obs_x': obs_x, 'obs_y': obs_y, 'g_obs': gravity_obs,
        'all_misfits': result['all_misfits'],
        'acceptance_rate': result['acceptance_rate'],
        'samples_thinned': samples_thinned,
        'all_alphas': result['all_alphas'],
        'alpha_samples': alpha_samples,
        'alpha_mean': a_mean, 'alpha_std': a_std,
        'alpha_ci_5': a_ci5, 'alpha_ci_95': a_ci95,
        'alpha_acceptance_rate': result['alpha_acceptance_rate'],
    }
    for fn in (plot_01, plot_02, plot_03, plot_04, plot_05,
               plot_06, plot_07, plot_08, plot_09_alpha):
        try:
            fn(ctx)
            print(f"  ✓ {fn.__name__}")
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")

    print("\n" + "=" * 76)
    print(f"ALL DONE in {elapsed/60:.1f} min (MCMC) + plotting")
    print(f"Results: {OUT_DIR}/")
    print("=" * 76)


if __name__ == '__main__':
    main()
