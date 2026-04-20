"""
Generate the full Exp-7 plot suite for Chintalpudi v5 runs from saved npz.

USAGE
    python generate_chintalpudi_v5_plots.py [results_dir]

    results_dir defaults to 'results/exp_chintalpudi_v5_50k'.

REFERENCE TERMINOLOGY
    "Chak 2007 reference" = Chakravarthi & Sundararajan (2007) Figure 5(f),
      their Marquardt-inversion basement surface, digitized. This is a
      published-inversion BENCHMARK, NOT measured ground truth.

    "ONGC borehole (2935 m)" = Agarwal (1995). This is the ONLY measured
      ground-truth depth for Chintalpudi — a single point at the depocenter.

PRODUCES (8 plots)
    01_depth_comparison.png       — recovered vs Chak2007 ref vs difference (2D)
    02_depth_3d_surface.png       — 3D perspective of recovered & reference
    03_uncertainty_map.png        — 2D posterior std-dev
    04_uncertainty_3d_surface.png — 3D depth colored by std
    05_cross_sections.png         — E–W and N–S with 90% CI
    06_gravity_fit.png            — observed, predicted, residual
    07_accuracy.png               — recovered vs Chak2007 scatter + histogram
    08_mcmc_diagnostics.png       — misfit trace, rolling mean, ONGC posterior
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.forward_model import compute_gravity_for_basin
from src.utils import make_density_func


REF_LABEL = 'Chakravarthi 2007 (Marquardt inversion)'
ONGC_LABEL = 'ONGC borehole (2935 m, real)'


def load(res_dir):
    npz = np.load(os.path.join(res_dir, 'results_data.npz'), allow_pickle=True)
    d = {k: npz[k] for k in npz.files}
    for k in list(d.keys()):
        if d[k].ndim == 0:
            d[k] = d[k].item()
    # back-compat: older runs saved as 'truth_blocks'
    if 'ref_blocks' not in d and 'truth_blocks' in d:
        d['ref_blocks'] = d['truth_blocks']
    return d


def _depocenter(d):
    """Return (ib, jb, xc_km, yc_km, ONGC_depth) — the depocenter block."""
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


def plot_01_depth_comparison(d, res_dir):
    mean_d, ref = d['mean_depths'], d['ref_blocks']
    bx, by = d['block_x_edges'], d['block_y_edges']
    diff = mean_d - ref
    rms = float(np.sqrt(np.mean(diff**2)))
    vmin, vmax = 0.0, max(mean_d.max(), ref.max())
    vm = float(np.abs(diff).max())
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    panels = [
        (axes[0], mean_d, 'viridis_r',
         f'Our MCMC (posterior mean)\n{mean_d.min():.0f}–{mean_d.max():.0f} m',
         (vmin, vmax), 'Depth (m)'),
        (axes[1], ref, 'viridis_r',
         f'{REF_LABEL}\n{ref.min():.0f}–{ref.max():.0f} m',
         (vmin, vmax), 'Depth (m)'),
        (axes[2], diff, 'RdBu_r',
         f'Difference (ours − Chak 2007)\nRMS {rms:.0f} m, bias {diff.mean():+.0f} m',
         (-vm, vm), 'Difference (m)')]
    for ax, data, cmap, title, vv, lbl in panels:
        im = ax.pcolormesh(bx/1000, by/1000, data.T, cmap=cmap,
                           vmin=vv[0], vmax=vv[1], shading='flat')
        plt.colorbar(im, ax=ax, label=lbl)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_aspect('equal'); ax.set_title(title)
        ax.plot(xc_d, yc_d, marker='*', color='gold',
                markeredgecolor='k', markersize=18,
                label=ONGC_LABEL)
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle(f"Chintalpudi v5 — 2D depth comparison "
                 f"(★ = ONGC borehole, only real ground truth)")
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '01_depth_comparison.png'), dpi=150)
    plt.close(fig)


def plot_02_depth_3d_surface(d, res_dir):
    mean_d, ref = d['mean_depths'], d['ref_blocks']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:])/1000
    yc = 0.5*(by[:-1] + by[1:])/1000
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
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m, down)')
        ax.set_title(f'{title}\n{data.min():.0f}–{data.max():.0f} m')
        fig.colorbar(surf, ax=ax, shrink=0.55, label='Depth (m)')
        ax.view_init(elev=28, azim=-120); ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Chintalpudi v5 — 3D basement surface (★ = ONGC borehole)')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '02_depth_3d_surface.png'), dpi=150)
    plt.close(fig)


def plot_03_uncertainty_map(d, res_dir):
    std_d = d['std_depths']
    bx, by = d['block_x_edges'], d['block_y_edges']
    ib, jb, xc_d, yc_d, bz = _depocenter(d)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.pcolormesh(bx/1000, by/1000, std_d.T, cmap='hot_r', shading='flat')
    plt.colorbar(im, ax=ax, label='Posterior std (m)')
    ax.plot(xc_d, yc_d, marker='*', color='gold',
            markeredgecolor='k', markersize=18, label=ONGC_LABEL)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
    ax.set_title(f'Posterior uncertainty (std)\n'
                 f'min {std_d.min():.0f}, mean {std_d.mean():.0f}, '
                 f'max {std_d.max():.0f} m')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '03_uncertainty_map.png'), dpi=150)
    plt.close(fig)


def plot_04_uncertainty_3d(d, res_dir):
    mean_d, std_d = d['mean_depths'], d['std_depths']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:])/1000
    yc = 0.5*(by[:-1] + by[1:])/1000
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
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m)')
    ax.set_title(f'Basement with posterior uncertainty\n'
                 f'std {std_d.min():.0f}–{std_d.max():.0f} m '
                 f'(mean {std_d.mean():.0f})')
    ax.view_init(elev=28, azim=-120); ax.legend(loc='upper left', fontsize=8)
    m = plt.cm.ScalarMappable(cmap='hot_r',
            norm=plt.Normalize(vmin=std_d.min(), vmax=std_d.max()))
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.65, label='Std (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '04_uncertainty_3d_surface.png'), dpi=150)
    plt.close(fig)


def plot_05_cross_sections(d, res_dir):
    mean_d, ref = d['mean_depths'], d['ref_blocks']
    ci_lo, ci_hi = d['ci_5'], d['ci_95']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:])/1000
    yc = 0.5*(by[:-1] + by[1:])/1000
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    ax.plot(xc, mean_d[:, jb], 'b-', lw=2, label='Our MCMC (mean)')
    ax.fill_between(xc, ci_lo[:, jb], ci_hi[:, jb],
                    alpha=0.3, color='b', label='Our 90% CI')
    ax.plot(xc, ref[:, jb], 'k--', lw=2, label=REF_LABEL)
    ax.plot(xc_d, bz, marker='*', color='gold', markeredgecolor='k',
            markersize=18, linewidth=0, label=ONGC_LABEL)
    ax.invert_yaxis(); ax.set_xlabel('X (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'E–W cross-section at Y = {yc_d:.1f} km (depocenter row)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(yc, mean_d[ib, :], 'b-', lw=2, label='Our MCMC (mean)')
    ax.fill_between(yc, ci_lo[ib, :], ci_hi[ib, :],
                    alpha=0.3, color='b', label='Our 90% CI')
    ax.plot(yc, ref[ib, :], 'k--', lw=2, label=REF_LABEL)
    ax.plot(yc_d, bz, marker='*', color='gold', markeredgecolor='k',
            markersize=18, linewidth=0, label=ONGC_LABEL)
    ax.invert_yaxis(); ax.set_xlabel('Y (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'N–S cross-section at X = {xc_d:.1f} km (depocenter col)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Chintalpudi v5 — Cross-sections with 90% CI')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '05_cross_sections.png'), dpi=150)
    plt.close(fig)


def plot_06_gravity_fit(d, res_dir):
    bx, by = d['block_x_edges'], d['block_y_edges']
    obs_x, obs_y, g_obs = d['obs_x'], d['obs_y'], d['obs_gravity']
    drho_0 = float(d['drho_0']); lam = float(d['lam'])
    mean_d = d['mean_depths']

    density_func = make_density_func('exponential', drho_0=drho_0, lam=lam)
    g_pred = compute_gravity_for_basin(obs_x, obs_y, bx, by, mean_d,
                                       density_func, n_sublayers=10)
    residual = g_obs - g_pred
    rms_g = float(np.sqrt(np.mean(residual**2)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, c, title, vm in [
        (axes[0], g_obs, f'Observed ({g_obs.min():.1f} to {g_obs.max():.1f} mGal)', None),
        (axes[1], g_pred, 'Predicted (from MCMC mean)', None),
        (axes[2], residual, f'Residual = obs − pred (RMS {rms_g:.2f} mGal)',
         float(np.abs(residual).max()))]:
        if vm is None:
            sc = ax.scatter(obs_x/1000, obs_y/1000, c=c, cmap='RdBu_r',
                            s=10, vmin=g_obs.min(), vmax=g_obs.max())
        else:
            sc = ax.scatter(obs_x/1000, obs_y/1000, c=c, cmap='RdBu_r',
                            s=10, vmin=-vm, vmax=vm)
        plt.colorbar(sc, ax=ax, label='mGal')
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_aspect('equal'); ax.set_title(title)
    fig.suptitle('Chintalpudi v5 — Gravity data fit')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '06_gravity_fit.png'), dpi=150)
    plt.close(fig)


def plot_07_accuracy(d, res_dir):
    mean_d, std_d, ref = d['mean_depths'], d['std_depths'], d['ref_blocks']
    rms = float(np.sqrt(np.mean((mean_d - ref)**2)))
    ib, jb, xc_d, yc_d, bz = _depocenter(d)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # Scatter: our MCMC mean vs Chak 2007 reference, color-coded per-block uncertainty
    ax = axes[0]
    ax.errorbar(ref.flatten(), mean_d.flatten(), yerr=std_d.flatten(),
                fmt='o', alpha=0.55, ms=3, capsize=2, color='steelblue',
                label='MCMC blocks (with σ)')
    lim = [0, max(ref.max(), mean_d.max())*1.05]
    ax.plot(lim, lim, 'k--', label='y = x (perfect agreement)')
    # mark the depocenter block with ONGC truth on Y-axis
    ax.plot(ref[ib, jb], mean_d[ib, jb], marker='o', color='blue',
            markersize=12, markeredgecolor='k',
            label=f'Depocenter (our {mean_d[ib,jb]:.0f}±{std_d[ib,jb]:.0f})')
    ax.axhline(bz, color='gold', lw=1.8, ls=':',
               label=f'ONGC borehole ({bz:.0f} m, real)')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel(f'{REF_LABEL} depth (m)')
    ax.set_ylabel('Our MCMC depth (m)')
    ax.set_title(f'Our MCMC vs Chak 2007 reference (RMS {rms:.0f} m)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Histogram of (ours − reference)
    ax = axes[1]
    err = (mean_d - ref).flatten()
    ax.hist(err, bins=30, color='indianred', edgecolor='k', alpha=0.85)
    ax.axvline(0, color='k', ls='--')
    ax.axvline(err.mean(), color='blue', ls='-',
               label=f'bias {err.mean():+.0f} m')
    ax.set_xlabel('Our MCMC − Chak 2007 (m)'); ax.set_ylabel('Count')
    ax.set_title(f'Per-block difference distribution (std {err.std():.0f} m)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Chintalpudi v5 — Agreement with published inversion')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '07_accuracy.png'), dpi=150)
    plt.close(fig)


def plot_08_diagnostics(d, res_dir):
    misfits = d['all_misfits']
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
    mf = np.asarray(misfits)
    window = max(200, n // 200)
    running = np.convolve(mf, np.ones(window)/window, mode='valid')
    ax.plot(np.arange(len(running)) + window, running, lw=0.6)
    ax.set_xlabel('Iteration'); ax.set_ylabel(f'Rolling-mean misfit (w={window})')
    ax.set_title('Misfit convergence')
    ax.grid(alpha=0.3)

    if n_panels == 3:
        ax = axes[2]
        samples_bore = samples_thin[:, ib, jb]
        ax.hist(samples_bore, bins=40, color='slateblue',
                edgecolor='k', alpha=0.85)
        ax.axvline(bz, color='gold', lw=2.5, label=ONGC_LABEL)
        ax.axvline(float(samples_bore.mean()), color='blue', lw=2,
                   label=f'Our posterior mean ({samples_bore.mean():.0f} m)')
        chak_depo = d.get('chak2007_reported_depocenter', None)
        if chak_depo is not None:
            ax.axvline(float(chak_depo), color='darkgreen', lw=1.5, ls='--',
                       label=f'Chak 2007 inversion ({float(chak_depo):.0f} m)')
        ax.set_xlabel('Depth (m)'); ax.set_ylabel('Count')
        ax.set_title(f'Posterior at depocenter block ({ib},{jb})')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'Chintalpudi v5 — MCMC diagnostics '
                 f'(right panel: ground-truth check)')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '08_mcmc_diagnostics.png'), dpi=150)
    plt.close(fig)


def main():
    res_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/exp_chintalpudi_v5_50k'
    if not os.path.isdir(res_dir):
        print(f'ERROR: results dir does not exist: {res_dir}')
        sys.exit(1)
    d = load(res_dir)
    print(f"Loaded {res_dir}/results_data.npz")
    print(f"  experiment: {d.get('experiment','?')}")
    print(f"  grid:       {tuple(d['grid_shape'])}")
    print(f"  iters:      {int(d['n_iterations']):,}")
    print(f"  NOTE: 'reference' = Chak 2007 Fig 5(f) inversion (NOT measured truth)")
    print(f"        'ground truth' = ONGC borehole, 2935 m, single point only")

    plot_01_depth_comparison(d, res_dir)
    plot_02_depth_3d_surface(d, res_dir)
    plot_03_uncertainty_map(d, res_dir)
    plot_04_uncertainty_3d(d, res_dir)
    plot_05_cross_sections(d, res_dir)
    plot_06_gravity_fit(d, res_dir)
    plot_07_accuracy(d, res_dir)
    plot_08_diagnostics(d, res_dir)

    print(f"\nGenerated in {res_dir}:")
    for f in sorted(os.listdir(res_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
