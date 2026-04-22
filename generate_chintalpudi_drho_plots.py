"""
Plot suite for Chintalpudi joint depth + constant-Δρ MCMC runs.

Reads results_data.npz from a results directory and writes 9 PNGs:
  01_depth_comparison.png     — 2D: ours vs Chak 2007 vs difference
  02_depth_3d_surface.png     — 3D perspective, side-by-side
  03_uncertainty_map.png      — 2D posterior std
  04_uncertainty_3d_surface.png — 3D depth colored by std
  05_cross_sections.png       — E–W & N–S with 90% CI bands
  06_gravity_fit.png          — observed, predicted (constant Δρ), residual
  07_accuracy.png             — scatter + per-block error histogram
  08_mcmc_diagnostics.png     — misfit trace + rolling mean
  09_drho_posterior.png       — Δρ trace + posterior histogram

Usage:
  python generate_chintalpudi_drho_plots.py results/exp_chintalpudi_drho_quicktest
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.forward_model import compute_gravity_for_basin

REF_LABEL  = 'Chakravarthi 2007 (digitized)'
ONGC_LABEL = 'ONGC borehole (2935 m, real)'


def load(res_dir):
    npz = np.load(os.path.join(res_dir, 'results_data.npz'), allow_pickle=True)
    d = {k: npz[k] for k in npz.files}
    for k in list(d.keys()):
        if d[k].ndim == 0:
            d[k] = d[k].item()
    return d


def _depocenter(d):
    ref = d['ref_blocks']
    bx, by = d['block_x_edges'], d['block_y_edges']
    if 'borehole_block' in d:
        ib, jb = int(d['borehole_block'][0]), int(d['borehole_block'][1])
    else:
        ib, jb = [int(k) for k in np.unravel_index(np.argmax(ref), ref.shape)]
    xc = 0.5*(bx[ib] + bx[ib+1]) / 1000
    yc = 0.5*(by[jb] + by[jb+1]) / 1000
    bz = float(d.get('borehole_depth', 2935.0))
    return ib, jb, xc, yc, bz


def _const_density_func(drho):
    return lambda z: float(drho) * np.ones_like(np.asarray(z, dtype=float))


def plot_01_depth_comparison(d, res_dir):
    bx, by = d['block_x_edges'], d['block_y_edges']
    ours = d['mean_depths']
    ref  = d['ref_blocks']
    diff = ours - ref
    ib, jb, xc_d, yc_d, bz = _depocenter(d)
    drho_mean = float(d['drho_mean'])

    vmax_d = max(ours.max(), ref.max())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, Z, title, vmin, vmax, cmap in [
        (axes[0], ref,  REF_LABEL,                       0,    vmax_d, 'viridis_r'),
        (axes[1], ours, f'Our MCMC mean (constant Δρ={drho_mean:.0f})', 0, vmax_d, 'viridis_r'),
        (axes[2], diff, 'Difference (ours − ref)',       -float(np.abs(diff).max()),
                                                          float(np.abs(diff).max()), 'RdBu_r'),
    ]:
        im = ax.pcolormesh(bx/1000, by/1000, Z.T, shading='flat',
                            cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Depth (m)')
        ax.scatter([xc_d], [yc_d], marker='*', s=320, c='gold',
                   edgecolor='k', linewidth=1.2, label='ONGC')
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_aspect('equal'); ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('Chintalpudi (constant Δρ joint MCMC) — Depth comparison',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '01_depth_comparison.png'), dpi=150)
    plt.close(fig)


def plot_02_depth_3d_surface(d, res_dir):
    """Match v5 style: fixed view angle, edge lines, ONGC star, depth plotted as -z."""
    mean_d, ref = d['mean_depths'], d['ref_blocks']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:]) / 1000
    yc = 0.5*(by[:-1] + by[1:]) / 1000
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig = plt.figure(figsize=(16, 7))
    for k, (data, title) in enumerate([
        (mean_d, 'Our MCMC (posterior mean)'),
        (ref,    REF_LABEL)]):
        ax = fig.add_subplot(1, 2, k+1, projection='3d')
        surf = ax.plot_surface(Xc, Yc, -data, cmap='viridis_r',
                               edgecolor='k', linewidth=0.15, alpha=0.95)
        ax.scatter([xc_d], [yc_d], [-bz], color='gold',
                   edgecolors='k', s=200, marker='*', label=ONGC_LABEL)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_zlabel('Depth (m, down)')
        ax.set_title(f'{title}\n{data.min():.0f}–{data.max():.0f} m')
        fig.colorbar(surf, ax=ax, shrink=0.55, label='Depth (m)')
        ax.view_init(elev=28, azim=-120)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Chintalpudi — 3D basement surface (★ = ONGC borehole)',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '02_depth_3d_surface.png'), dpi=150)
    plt.close(fig)


def plot_03_uncertainty_map(d, res_dir):
    bx, by = d['block_x_edges'], d['block_y_edges']
    std = d['std_depths']
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.pcolormesh(bx/1000, by/1000, std.T, shading='flat',
                        cmap='hot_r')
    plt.colorbar(im, ax=ax, label='Posterior std (m)')
    ax.scatter([xc_d], [yc_d], marker='*', s=320, c='gold',
               edgecolor='k', label='ONGC')
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
    ax.set_title(f'Per-block depth uncertainty  '
                 f'(mean σ {std.mean():.0f} m, max σ {std.max():.0f} m)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '03_uncertainty_map.png'), dpi=150)
    plt.close(fig)


def plot_04_uncertainty_3d(d, res_dir):
    """Match v5 style: fixed view angle, edge lines, ONGC star, depth as -z."""
    mean_d, std_d = d['mean_depths'], d['std_depths']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:]) / 1000
    yc = 0.5*(by[:-1] + by[1:]) / 1000
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig = plt.figure(figsize=(10.5, 7))
    ax = fig.add_subplot(111, projection='3d')
    norm = (std_d - std_d.min()) / (std_d.max() - std_d.min() + 1e-9)
    colors = plt.cm.hot_r(norm)
    ax.plot_surface(Xc, Yc, -mean_d, facecolors=colors,
                    edgecolor='k', linewidth=0.15, shade=False, alpha=0.95)
    ax.scatter([xc_d], [yc_d], [-bz], color='gold',
               edgecolors='k', s=200, marker='*', label=ONGC_LABEL)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_zlabel('Depth (m, down)')
    ax.set_title(f'Basement with posterior uncertainty\n'
                 f'std {std_d.min():.0f}–{std_d.max():.0f} m '
                 f'(mean {std_d.mean():.0f})')
    ax.view_init(elev=28, azim=-120)
    ax.legend(loc='upper left', fontsize=8)
    m = plt.cm.ScalarMappable(cmap='hot_r',
            norm=plt.Normalize(vmin=std_d.min(), vmax=std_d.max()))
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.65, label='Std (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '04_uncertainty_3d_surface.png'), dpi=150)
    plt.close(fig)


def plot_05_cross_sections(d, res_dir):
    bx, by = d['block_x_edges'], d['block_y_edges']
    ours, std = d['mean_depths'], d['std_depths']
    ci_lo, ci_hi = d['ci_5'], d['ci_95']
    ref = d['ref_blocks']
    ib, jb, xc_d, yc_d, bz = _depocenter(d)
    Nx, Ny = ours.shape
    bxc = 0.5*(bx[:-1] + bx[1:]) / 1000
    byc = 0.5*(by[:-1] + by[1:]) / 1000

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # E-W slice through depocenter (vary X, fix Y=jb)
    ax = axes[0]
    ax.fill_between(bxc, ci_lo[:, jb], ci_hi[:, jb], color='lightsteelblue',
                    alpha=0.6, label='90% CI')
    ax.plot(bxc, ours[:, jb], 'b-', lw=2, label='MCMC mean')
    ax.plot(bxc, ref[:, jb], 'k--', lw=1.5, label=REF_LABEL)
    ax.axvline(xc_d, color='gold', ls=':', lw=2,
               label=f'ONGC X={xc_d:.1f} km')
    ax.scatter([xc_d], [bz], marker='*', s=200, c='gold',
               edgecolor='k', zorder=10)
    ax.invert_yaxis()
    ax.set_xlabel('X (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'E–W cross-section at Y={yc_d:.1f} km (depocenter row)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # N-S slice through depocenter
    ax = axes[1]
    ax.fill_between(byc, ci_lo[ib, :], ci_hi[ib, :], color='lightsteelblue',
                    alpha=0.6, label='90% CI')
    ax.plot(byc, ours[ib, :], 'b-', lw=2, label='MCMC mean')
    ax.plot(byc, ref[ib, :], 'k--', lw=1.5, label=REF_LABEL)
    ax.axvline(yc_d, color='gold', ls=':', lw=2,
               label=f'ONGC Y={yc_d:.1f} km')
    ax.scatter([yc_d], [bz], marker='*', s=200, c='gold',
               edgecolor='k', zorder=10)
    ax.invert_yaxis()
    ax.set_xlabel('Y (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'N–S cross-section at X={xc_d:.1f} km (depocenter col)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Chintalpudi — depth profiles with 90% credible band',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '05_cross_sections.png'), dpi=150)
    plt.close(fig)


def plot_06_gravity_fit(d, res_dir):
    bx, by = d['block_x_edges'], d['block_y_edges']
    obs_x, obs_y, g_obs = d['obs_x'], d['obs_y'], d['obs_gravity']
    drho_mean = float(d['drho_mean'])
    mean_d = d['mean_depths']

    density_func = _const_density_func(drho_mean)
    g_pred = compute_gravity_for_basin(obs_x, obs_y, bx, by, mean_d,
                                        density_func, n_sublayers=5)
    residual = g_obs - g_pred
    rms_g = float(np.sqrt(np.mean(residual**2)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, c, title, vm in [
        (axes[0], g_obs,    f'Observed ({g_obs.min():.1f} to {g_obs.max():.1f} mGal)', None),
        (axes[1], g_pred,   f'Predicted (MCMC mean, Δρ={drho_mean:.0f})', None),
        (axes[2], residual, f'Residual = obs − pred (RMS {rms_g:.2f} mGal)',
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
    fig.suptitle('Chintalpudi — gravity data fit (constant Δρ model)',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '06_gravity_fit.png'), dpi=150)
    plt.close(fig)


def plot_07_accuracy(d, res_dir):
    mean_d, std_d, ref = d['mean_depths'], d['std_depths'], d['ref_blocks']
    rms = float(np.sqrt(np.mean((mean_d - ref)**2)))
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    ax = axes[0]
    ax.errorbar(ref.flatten(), mean_d.flatten(), yerr=std_d.flatten(),
                fmt='o', alpha=0.55, ms=3, capsize=2, color='steelblue',
                label='MCMC blocks (with σ)')
    lim = [0, max(ref.max(), mean_d.max())*1.05]
    ax.plot(lim, lim, 'k--', label='y = x')
    ax.plot(ref[ib, jb], mean_d[ib, jb], marker='o', color='blue',
            markersize=12, markeredgecolor='k',
            label=f'Depocenter ({mean_d[ib,jb]:.0f}±{std_d[ib,jb]:.0f})')
    ax.axhline(bz, color='gold', lw=1.8, ls=':',
               label=f'ONGC ({bz:.0f} m)')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel(f'{REF_LABEL} depth (m)')
    ax.set_ylabel('Our MCMC depth (m)')
    ax.set_title(f'Per-block agreement (RMS {rms:.0f} m)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    err = (mean_d - ref).flatten()
    ax.hist(err, bins=30, color='indianred', edgecolor='k', alpha=0.85)
    ax.axvline(0, color='k', ls='--')
    ax.axvline(err.mean(), color='blue', lw=2,
               label=f'bias {err.mean():+.0f} m')
    ax.set_xlabel('Our MCMC − Chak 2007 (m)'); ax.set_ylabel('Count')
    ax.set_title(f'Per-block residual (std {err.std():.0f} m)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Chintalpudi — Agreement with Chakravarthi 2007 inversion',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '07_accuracy.png'), dpi=150)
    plt.close(fig)


def plot_08_diagnostics(d, res_dir):
    """v5-style: misfit trace + rolling mean + (if available) depocenter posterior."""
    misfits = np.asarray(d['all_misfits'])
    acc = float(d.get('acceptance_rate', np.nan))
    n = len(misfits)
    samples_thin = d.get('posterior_samples_thinned', None)
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    n_panels = 3 if samples_thin is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5*n_panels, 4.6))
    if n_panels == 2:
        axes = list(axes)

    ax = axes[0]
    ax.semilogy(misfits, lw=0.3, alpha=0.7)
    ax.axvline(n//2, color='r', ls='--', label='burn-in (50%)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Misfit (log)')
    ax.set_title(f'Misfit trace (accept {acc*100:.1f}%)')
    ax.legend(); ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    window = max(200, n // 200)
    running = np.convolve(misfits, np.ones(window)/window, mode='valid')
    ax.plot(np.arange(len(running)) + window, running, lw=0.6)
    ax.set_xlabel('Iteration'); ax.set_ylabel(f'Rolling-mean misfit (w={window})')
    ax.set_title('Misfit convergence')
    ax.grid(alpha=0.3)

    if n_panels == 3:
        ax = axes[2]
        samples_bore = samples_thin[:, ib, jb]
        ax.hist(samples_bore, bins=40, color='slateblue',
                edgecolor='k', alpha=0.85)
        ax.axvline(bz, color='gold', lw=2.5,
                   label=f'ONGC borehole ({bz:.0f} m, real)')
        ax.axvline(float(samples_bore.mean()), color='blue', lw=2,
                   label=f'Our posterior mean ({samples_bore.mean():.0f} m)')
        chak_depo = d.get('chak2007_reported_depocenter', None)
        if chak_depo is not None:
            ax.axvline(float(chak_depo), color='darkgreen', lw=1.5, ls='--',
                       label=f'Chak 2007 inversion ({float(chak_depo):.0f} m)')
        ax.set_xlabel('Depth (m)'); ax.set_ylabel('Count')
        ax.set_title(f'Posterior at depocenter block ({ib},{jb})')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Chintalpudi — MCMC diagnostics '
                 '(right panel: ground-truth check)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '08_mcmc_diagnostics.png'), dpi=150)
    plt.close(fig)


def plot_09_drho_posterior(d, res_dir):
    drho_chain = np.asarray(d['drho_chain'])
    n = len(drho_chain)
    burn = n // 2
    post = drho_chain[burn:]
    drho_mean = float(d['drho_mean'])
    drho_std  = float(d['drho_std'])
    drho_ci   = (float(d['drho_ci_5']), float(d['drho_ci_95']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    ax = axes[0]
    ax.plot(drho_chain, lw=0.5, color='C0', alpha=0.85)
    ax.axvline(burn, color='r', ls='--', label='burn-in (50%)')
    ax.axhline(drho_mean, color='k', ls=':',
               label=f'post-burn mean {drho_mean:.1f}')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Δρ (kg/m³)')
    ax.set_title(f'Δρ trace  (post-burn {drho_mean:.1f} ± {drho_std:.1f} kg/m³)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(post, bins=40, color='slateblue', edgecolor='k', alpha=0.85)
    ax.axvline(drho_mean, color='blue', lw=2,
               label=f'mean {drho_mean:.1f}')
    ax.axvline(drho_ci[0], color='k', ls=':',
               label=f'5% {drho_ci[0]:.1f}')
    ax.axvline(drho_ci[1], color='k', ls=':',
               label=f'95% {drho_ci[1]:.1f}')
    ax.set_xlabel('Δρ (kg/m³)'); ax.set_ylabel('Count')
    ax.set_title('Posterior distribution of Δρ (post-burn-in)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Chintalpudi — Δρ joint-posterior diagnostics',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '09_drho_posterior.png'), dpi=150)
    plt.close(fig)


def generate_all(res_dir):
    if not os.path.isdir(res_dir):
        raise FileNotFoundError(f'results dir does not exist: {res_dir}')
    d = load(res_dir)
    print(f"\nPlotting {res_dir}/results_data.npz")
    print(f"  experiment: {d.get('experiment','?')}")
    print(f"  Δρ mean:    {float(d['drho_mean']):.1f} ± {float(d['drho_std']):.1f} kg/m³")
    print(f"  iterations: {int(d['n_iterations']):,}")

    plot_01_depth_comparison(d, res_dir)
    plot_02_depth_3d_surface(d, res_dir)
    plot_03_uncertainty_map(d, res_dir)
    plot_04_uncertainty_3d(d, res_dir)
    plot_05_cross_sections(d, res_dir)
    plot_06_gravity_fit(d, res_dir)
    plot_07_accuracy(d, res_dir)
    plot_08_diagnostics(d, res_dir)
    plot_09_drho_posterior(d, res_dir)

    print(f"\nGenerated:")
    for f in sorted(os.listdir(res_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")


def main():
    res_dir = (sys.argv[1] if len(sys.argv) > 1
               else 'results/exp_chintalpudi_drho_quicktest')
    generate_all(res_dir)


if __name__ == '__main__':
    main()
