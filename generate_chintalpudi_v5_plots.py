"""
Generate the full Exp-7 plot suite for Chintalpudi v5 runs from saved npz.

USAGE
    python generate_chintalpudi_v5_plots.py [results_dir]

    results_dir defaults to 'results/exp_chintalpudi_v5_50k'.
    Expects <results_dir>/results_data.npz produced by run_chintalpudi_v5_50k.py
    or run_chintalpudi_v5_100k.py.

PRODUCES (8 plots)
    01_depth_comparison.png       — recovered vs truth vs error (2D)
    02_depth_3d_surface.png       — 3D perspective of recovered & truth
    03_uncertainty_map.png        — 2D posterior std-dev
    04_uncertainty_3d_surface.png — 3D depth colored by std
    05_cross_sections.png         — E–W and N–S with 90% CI
    06_gravity_fit.png            — observed, predicted, residual
    07_accuracy.png               — recovered-vs-truth scatter + error histogram
    08_mcmc_diagnostics.png       — misfit trace, running acceptance, posterior
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


def load(res_dir):
    npz = np.load(os.path.join(res_dir, 'results_data.npz'), allow_pickle=True)
    d = {k: npz[k] for k in npz.files}
    # unwrap 0-d arrays
    for k in list(d.keys()):
        if d[k].ndim == 0:
            d[k] = d[k].item()
    return d


def plot_01_depth_comparison(d, res_dir):
    mean_d, truth = d['mean_depths'], d['truth_blocks']
    bx, by = d['block_x_edges'], d['block_y_edges']
    err = mean_d - truth
    rms = float(np.sqrt(np.mean(err**2)))
    vmin, vmax = 0.0, max(mean_d.max(), truth.max())
    vm = float(np.abs(err).max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, data, cmap, title, vminmax, lbl in [
        (axes[0], mean_d, 'viridis_r',
         f'Recovered (MCMC posterior mean)\n{mean_d.min():.0f}–{mean_d.max():.0f} m',
         (vmin, vmax), 'Depth (m)'),
        (axes[1], truth, 'viridis_r',
         f'Published truth (Chakravarthi 2007)\n{truth.min():.0f}–{truth.max():.0f} m',
         (vmin, vmax), 'Depth (m)'),
        (axes[2], err, 'RdBu_r',
         f'Error = recovered − truth\nRMS = {rms:.0f} m, bias {err.mean():+.0f} m',
         (-vm, vm), 'Error (m)')]:
        im = ax.pcolormesh(bx/1000, by/1000, data.T, cmap=cmap,
                           vmin=vminmax[0], vmax=vminmax[1], shading='flat')
        plt.colorbar(im, ax=ax, label=lbl)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_aspect('equal'); ax.set_title(title)
    fig.suptitle(f"Chintalpudi v5 ({d.get('experiment','')}) — 2D depth comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '01_depth_comparison.png'), dpi=150)
    plt.close(fig)


def plot_02_depth_3d_surface(d, res_dir):
    mean_d, truth = d['mean_depths'], d['truth_blocks']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:])/1000
    yc = 0.5*(by[:-1] + by[1:])/1000
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
    bxy = d.get('borehole_block', None)
    bz  = d.get('borehole_depth', 2935.0)

    fig = plt.figure(figsize=(16, 7))
    for k, (data, title) in enumerate([
        (mean_d, 'Recovered (MCMC posterior mean)'),
        (truth,  'Published truth (Chakravarthi 2007)')]):
        ax = fig.add_subplot(1, 2, k+1, projection='3d')
        surf = ax.plot_surface(Xc, Yc, -data, cmap='viridis_r',
                               edgecolor='k', linewidth=0.15, alpha=0.95)
        if bxy is not None:
            ib, jb = int(bxy[0]), int(bxy[1])
            ax.scatter([xc[ib]], [yc[jb]], [-bz], color='yellow',
                       edgecolors='k', s=160, marker='*', label='ONGC borehole')
            ax.legend()
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m, down)')
        ax.set_title(f'{title}\n{data.min():.0f}–{data.max():.0f} m')
        fig.colorbar(surf, ax=ax, shrink=0.55, label='Depth (m)')
        ax.view_init(elev=28, azim=-120)
    fig.suptitle('Chintalpudi v5 — 3D basement surface')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '02_depth_3d_surface.png'), dpi=150)
    plt.close(fig)


def plot_03_uncertainty_map(d, res_dir):
    std_d = d['std_depths']
    bx, by = d['block_x_edges'], d['block_y_edges']
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.pcolormesh(bx/1000, by/1000, std_d.T, cmap='hot_r', shading='flat')
    plt.colorbar(im, ax=ax, label='Posterior std (m)')
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
    ax.set_title(f'Posterior uncertainty (std)\n'
                 f'min {std_d.min():.0f}, mean {std_d.mean():.0f}, '
                 f'max {std_d.max():.0f} m')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '03_uncertainty_map.png'), dpi=150)
    plt.close(fig)


def plot_04_uncertainty_3d(d, res_dir):
    mean_d, std_d = d['mean_depths'], d['std_depths']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:])/1000
    yc = 0.5*(by[:-1] + by[1:])/1000
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')

    fig = plt.figure(figsize=(10.5, 7))
    ax = fig.add_subplot(111, projection='3d')
    norm = (std_d - std_d.min()) / (std_d.max() - std_d.min() + 1e-9)
    colors = plt.cm.hot_r(norm)
    ax.plot_surface(Xc, Yc, -mean_d, facecolors=colors,
                    edgecolor='k', linewidth=0.15, shade=False, alpha=0.95)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m)')
    ax.set_title(f'Basement with posterior uncertainty\n'
                 f'std {std_d.min():.0f}–{std_d.max():.0f} m '
                 f'(mean {std_d.mean():.0f})')
    ax.view_init(elev=28, azim=-120)
    m = plt.cm.ScalarMappable(cmap='hot_r',
            norm=plt.Normalize(vmin=std_d.min(), vmax=std_d.max()))
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.65, label='Std (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '04_uncertainty_3d_surface.png'), dpi=150)
    plt.close(fig)


def plot_05_cross_sections(d, res_dir):
    mean_d, truth = d['mean_depths'], d['truth_blocks']
    ci_lo, ci_hi = d['ci_5'], d['ci_95']
    bx, by = d['block_x_edges'], d['block_y_edges']
    xc = 0.5*(bx[:-1] + bx[1:])/1000
    yc = 0.5*(by[:-1] + by[1:])/1000
    # pick cross-sections through row/col of maximum truth depth
    idx = np.unravel_index(np.argmax(truth), truth.shape)
    ib, jb = idx

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    ax.plot(xc, mean_d[:, jb], 'b-', lw=2, label='MCMC mean')
    ax.fill_between(xc, ci_lo[:, jb], ci_hi[:, jb],
                    alpha=0.3, color='b', label='90% CI')
    ax.plot(xc, truth[:, jb], 'k--', lw=2, label='Truth (Chak 2007)')
    ax.invert_yaxis(); ax.set_xlabel('X (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'E–W cross-section at Y = {yc[jb]:.1f} km (depocenter row)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(yc, mean_d[ib, :], 'b-', lw=2, label='MCMC mean')
    ax.fill_between(yc, ci_lo[ib, :], ci_hi[ib, :],
                    alpha=0.3, color='b', label='90% CI')
    ax.plot(yc, truth[ib, :], 'k--', lw=2, label='Truth (Chak 2007)')
    ax.invert_yaxis(); ax.set_xlabel('Y (km)'); ax.set_ylabel('Depth (m)')
    ax.set_title(f'N–S cross-section at X = {xc[ib]:.1f} km (depocenter col)')
    ax.legend(); ax.grid(alpha=0.3)

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
        (axes[2], residual, f'Residual (RMS {rms_g:.2f} mGal)',
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
    fig.suptitle('Chintalpudi v5 — Gravity fit')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '06_gravity_fit.png'), dpi=150)
    plt.close(fig)


def plot_07_accuracy(d, res_dir):
    mean_d, std_d, truth = d['mean_depths'], d['std_depths'], d['truth_blocks']
    rms = float(np.sqrt(np.mean((mean_d - truth)**2)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    ax.errorbar(truth.flatten(), mean_d.flatten(), yerr=std_d.flatten(),
                fmt='o', alpha=0.55, ms=3, capsize=2, color='steelblue')
    lim = [0, max(truth.max(), mean_d.max())*1.05]
    ax.plot(lim, lim, 'k--', label='y = x (perfect)')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('Truth depth (m)'); ax.set_ylabel('Recovered depth (m)')
    ax.set_title(f'Recovered vs truth (RMS {rms:.0f} m)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    err = (mean_d - truth).flatten()
    ax.hist(err, bins=30, color='indianred', edgecolor='k', alpha=0.85)
    ax.axvline(0, color='k', ls='--')
    ax.axvline(err.mean(), color='blue', ls='-',
               label=f'bias {err.mean():+.0f} m')
    ax.set_xlabel('Error (m)'); ax.set_ylabel('Count')
    ax.set_title(f'Error distribution (std {err.std():.0f} m)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('Chintalpudi v5 — Depth recovery accuracy')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '07_accuracy.png'), dpi=150)
    plt.close(fig)


def plot_08_diagnostics(d, res_dir):
    misfits = d['all_misfits']
    acc = float(d.get('acceptance_rate', np.nan))
    n = len(misfits)
    samples_thin = d.get('posterior_samples_thinned', None)
    bxy = d.get('borehole_block', None)

    n_panels = 3 if samples_thin is not None and bxy is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5*n_panels, 4.4))
    if n_panels == 2:
        axes = list(axes)

    ax = axes[0]
    ax.semilogy(misfits, lw=0.3, alpha=0.7)
    ax.axvline(n//2, color='r', ls='--', label='burn-in')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Misfit')
    ax.set_title(f'Misfit trace (accept {acc*100:.1f}%)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    mf = np.asarray(misfits)
    window = max(200, n // 200)
    running = np.convolve(mf, np.ones(window)/window, mode='valid')
    ax.plot(np.arange(len(running)) + window, running, lw=0.6)
    ax.set_xlabel('Iteration'); ax.set_ylabel(f'Misfit (rolling mean, w={window})')
    ax.set_title('Rolling-window misfit'); ax.grid(alpha=0.3)

    if n_panels == 3:
        ax = axes[2]
        ib, jb = int(bxy[0]), int(bxy[1])
        samples_bore = samples_thin[:, ib, jb]
        bz = float(d.get('borehole_depth', 2935.0))
        ax.hist(samples_bore, bins=40, color='slateblue',
                edgecolor='k', alpha=0.85)
        ax.axvline(bz, color='gold', lw=2,
                   label=f'ONGC borehole ({bz:.0f} m)')
        ax.axvline(float(samples_bore.mean()), color='blue', lw=2,
                   label=f'post mean ({samples_bore.mean():.0f})')
        ax.set_xlabel('Depth (m)'); ax.set_ylabel('Count')
        ax.set_title(f'Posterior at depocenter block ({ib},{jb})')
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('Chintalpudi v5 — MCMC diagnostics')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, '08_mcmc_diagnostics.png'), dpi=150)
    plt.close(fig)


def main():
    res_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/exp_chintalpudi_v5_50k'
    if not os.path.isdir(res_dir):
        print(f'ERROR: results dir does not exist: {res_dir}')
        sys.exit(1)
    d = load(res_dir)
    print(f"Loaded {res_dir}/results_data.npz  "
          f"(experiment={d.get('experiment','?')}, "
          f"grid {tuple(d['grid_shape'])}, "
          f"{int(d['n_iterations']):,} iters)")

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
