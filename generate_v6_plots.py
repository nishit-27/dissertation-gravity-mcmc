"""
Generalized 8-plot suite for v6 runs. Supports both exponential (lam) and
Rao/Chakravarthi parabolic (alpha) density laws via 'density_law' field in npz.

USAGE
    python generate_v6_plots.py <results_dir>
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.forward_model import compute_gravity_for_basin


REF_LABEL = 'Chakravarthi 2007 (reference inversion)'
ONGC_LABEL = 'ONGC borehole (2935 m, real)'


def load(res_dir):
    npz = np.load(os.path.join(res_dir, 'results_data.npz'), allow_pickle=True)
    d = {k: npz[k] for k in npz.files}
    for k in list(d.keys()):
        if d[k].ndim == 0:
            d[k] = d[k].item()
    if 'ref_blocks' not in d and 'truth_blocks' in d:
        d['ref_blocks'] = d['truth_blocks']
    return d


def _density_func(d):
    """Return a density_func(z) callable based on npz contents."""
    law = str(d.get('density_law', 'exponential'))
    drho_0 = float(d['drho_0'])
    if 'alpha' in d and ('parab' in law.lower() or 'rao' in law.lower()):
        alpha = float(d['alpha'])
        return lambda z: drho_0 * (alpha / (alpha + np.asarray(z, dtype=float)))**2, \
               f'Rao parabolic Δρ₀={drho_0:.0f}, α={alpha:.0f} m'
    if 'z_compaction' in d:
        zc = float(d['z_compaction'])
        def f(z):
            z = np.asarray(z, dtype=float)
            r = np.clip(1 - z/zc, 0, None)
            return drho_0 * r * r
        return f, f'parab(1-z/z_c)² Δρ₀={drho_0:.0f}, z_c={zc:.0f}'
    lam = float(d.get('lam', 5e-4))
    return (lambda z: drho_0 * np.exp(-lam * np.asarray(z, dtype=float))), \
           f'exponential Δρ₀={drho_0:.0f}, λ={lam:.1e}'


def _depocenter(d):
    ref = d['ref_blocks']
    bx, by = d['block_x_edges'], d['block_y_edges']
    if 'borehole_block' in d:
        ib, jb = int(d['borehole_block'][0]), int(d['borehole_block'][1])
    else:
        ib, jb = [int(k) for k in np.unravel_index(np.argmax(ref), ref.shape)]
    xc = 0.5*(bx[ib] + bx[ib+1])/1000
    yc = 0.5*(by[jb] + by[jb+1])/1000
    bz = float(d.get('borehole_depth', 2935.0))
    return ib, jb, xc, yc, bz


def _tag(d):
    return str(d.get('experiment', ''))


def plot_01(d, res_dir):
    mean_d, ref = d['mean_depths'], d['ref_blocks']
    bx, by = d['block_x_edges'], d['block_y_edges']
    ext = [bx[0]/1000, bx[-1]/1000, by[0]/1000, by[-1]/1000]
    ib, jb, xc_d, yc_d, bz = _depocenter(d)
    rms = float(np.sqrt(np.mean((mean_d - ref)**2)))
    bias = float(np.mean(mean_d - ref))
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
    fig.suptitle(f'{_tag(d)} — 2D depth comparison (★ = ONGC borehole, only real ground truth)')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '01_depth_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_02(d, res_dir):
    mean_d, ref = d['mean_depths'], d['ref_blocks']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:])/1000
    yc = 0.5*(by[:-1] + by[1:])/1000
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig = plt.figure(figsize=(15, 6))
    for k, (arr, t) in enumerate(zip([mean_d, ref],
                                     [f'Our MCMC (posterior mean)\n'
                                      f'{mean_d.min():.0f}–{mean_d.max():.0f} m',
                                      f'{REF_LABEL}\n'
                                      f'{ref.min():.0f}–{ref.max():.0f} m'])):
        ax = fig.add_subplot(1, 2, k+1, projection='3d')
        s = ax.plot_surface(X, Y, -arr, cmap='viridis_r', alpha=0.9,
                            edgecolor='none')
        ax.scatter([xc_d], [yc_d], [-bz], marker='*', color='gold',
                   edgecolor='k', s=180, depthshade=False, label=ONGC_LABEL)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m)')
        ax.set_title(t)
        ax.invert_zaxis()
        fig.colorbar(s, ax=ax, shrink=0.6, label='Depth (m)', pad=0.1)
    fig.suptitle(f'{_tag(d)} — 3D basement surface')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '02_depth_3d_surface.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def plot_03(d, res_dir):
    std_d = d['std_depths']
    bx, by = d['block_x_edges'], d['block_y_edges']
    ext = [bx[0]/1000, bx[-1]/1000, by[0]/1000, by[-1]/1000]
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(std_d.T, origin='lower', extent=ext, cmap='hot_r',
                   aspect='auto')
    ax.scatter([xc_d], [yc_d], marker='*', s=280, c='gold',
               edgecolor='k', zorder=5, label=ONGC_LABEL)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'Posterior uncertainty (std)\n'
                 f'min {std_d.min():.0f}, mean {std_d.mean():.0f}, '
                 f'max {std_d.max():.0f} m')
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax, label='Posterior std (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '03_uncertainty_map.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_04(d, res_dir):
    mean_d, std_d = d['mean_depths'], d['std_depths']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:])/1000
    yc = 0.5*(by[:-1] + by[1:])/1000
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
    ax.set_title(f'{_tag(d)} — 3D basement colored by uncertainty')
    ax.invert_zaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '04_uncertainty_3d_surface.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


def plot_05(d, res_dir):
    mean_d, ref = d['mean_depths'], d['ref_blocks']
    ci_lo, ci_hi = d['ci_5'], d['ci_95']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:])/1000
    yc = 0.5*(by[:-1] + by[1:])/1000
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    ax.plot(xc, mean_d[:, jb], 'b-', lw=2, label='Our MCMC (mean)')
    ax.fill_between(xc, ci_lo[:, jb], ci_hi[:, jb], alpha=0.3, color='b',
                    label='Our 90% CI')
    ax.plot(xc, ref[:, jb], 'k--', lw=2, label=REF_LABEL)
    ax.plot(xc_d, bz, marker='*', color='gold', markeredgecolor='k',
            markersize=18, linewidth=0, label=ONGC_LABEL)
    ax.invert_yaxis(); ax.set_xlabel('X (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'E–W cross-section at Y = {yc_d:.1f} km')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(yc, mean_d[ib, :], 'b-', lw=2, label='Our MCMC (mean)')
    ax.fill_between(yc, ci_lo[ib, :], ci_hi[ib, :], alpha=0.3, color='b',
                    label='Our 90% CI')
    ax.plot(yc, ref[ib, :], 'k--', lw=2, label=REF_LABEL)
    ax.plot(yc_d, bz, marker='*', color='gold', markeredgecolor='k',
            markersize=18, linewidth=0, label=ONGC_LABEL)
    ax.invert_yaxis(); ax.set_xlabel('Y (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'N–S cross-section at X = {xc_d:.1f} km')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'{_tag(d)} — Cross-sections with 90% CI')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '05_cross_sections.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_06(d, res_dir):
    bx, by = d['block_x_edges'], d['block_y_edges']
    obs_x, obs_y = d['obs_x'], d['obs_y']
    g_obs = d['obs_gravity']
    mean_d = d['mean_depths']
    dens_func, dens_tag = _density_func(d)

    g_pred = compute_gravity_for_basin(obs_x, obs_y, bx, by, mean_d,
                                       dens_func, n_sublayers=10)
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
    fig.suptitle(f'{_tag(d)} — Gravity fit  |  {dens_tag}')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '06_gravity_fit.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_07(d, res_dir):
    mean_d, std_d, ref = d['mean_depths'], d['std_depths'], d['ref_blocks']
    rms = float(np.sqrt(np.mean((mean_d - ref)**2)))
    bias = float(np.mean(mean_d - ref))
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

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

    fig.suptitle(f'{_tag(d)} — Agreement with published inversion')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '07_accuracy.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_08(d, res_dir):
    misfits = d['all_misfits']
    acc = float(d.get('acceptance_rate', np.nan))
    n = len(misfits)
    samples_thin = d.get('posterior_samples_thinned', None)
    ib, jb, xc_d, yc_d, bz = _depocenter(d)
    post_mean = float(d['mean_depths'][ib, jb])

    # Always 3 panels; right is posterior histogram (from samples if available,
    # else CI bounds only)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.semilogy(misfits, lw=0.4, alpha=0.7)
    ax.axvline(n//2, color='r', ls='--', label='burn-in (50%)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Misfit (log)')
    ax.set_title(f'Misfit trace (accept {acc*100:.1f}%)')
    ax.legend(); ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    mf = np.asarray(misfits)
    window = max(200, n // 200)
    running = np.convolve(mf, np.ones(window)/window, mode='valid')
    ax.plot(np.arange(len(running)) + window, running, lw=0.8)
    ax.set_xlabel('Iteration'); ax.set_ylabel(f'Rolling-mean misfit (w={window})')
    ax.set_title('Misfit convergence'); ax.grid(alpha=0.3)

    ax = axes[2]
    if samples_thin is not None:
        bore_samples = samples_thin[:, ib, jb]
        ax.hist(bore_samples, bins=40, color='slateblue',
                edgecolor='k', alpha=0.85)
    else:
        # Fall back: show CI range as a filled band
        ci_lo = float(d['ci_5'][ib, jb]); ci_hi = float(d['ci_95'][ib, jb])
        ax.axvspan(ci_lo, ci_hi, color='slateblue', alpha=0.3,
                   label=f'90% CI ({ci_lo:.0f}–{ci_hi:.0f})')
    ax.axvline(bz, color='gold', lw=3, label=ONGC_LABEL)
    ax.axvline(post_mean, color='blue', lw=2,
               label=f'Our posterior mean ({post_mean:.0f} m)')
    chak_depo = d.get('chak2007_reported_depocenter', 2830.0)
    try:
        ax.axvline(float(chak_depo), color='darkgreen', lw=1.5, ls='--',
                   label=f'Chak 2007 ({float(chak_depo):.0f} m)')
    except Exception:
        pass
    ax.set_xlabel('Depth (m)'); ax.set_ylabel('Count')
    ax.set_title(f'Posterior at depocenter block ({ib},{jb})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'{_tag(d)} — MCMC diagnostics (right panel: blind ground-truth check)')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '08_mcmc_diagnostics.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def generate_all(res_dir):
    d = load(res_dir)
    print(f"\nPlotting {res_dir}/results_data.npz")
    print(f"  experiment: {_tag(d)}")
    print(f"  grid:       {tuple(d['grid_shape'])}")
    print(f"  iters:      {int(d['n_iterations']):,}")
    for fn in (plot_01, plot_02, plot_03, plot_04, plot_05,
               plot_06, plot_07, plot_08):
        try:
            fn(d, res_dir)
            print(f"  ✓ {fn.__name__}")
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"  Done. Plots in {res_dir}/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_v6_plots.py <results_dir>")
        sys.exit(1)
    generate_all(sys.argv[1])
