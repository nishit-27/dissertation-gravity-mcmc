"""
Visualization functions for Bayesian MCMC gravity inversion results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
import os


def plot_basement_with_uncertainty(model, posterior, data, save_path=None):
    """
    Main result plot: True basement vs MCMC estimate with uncertainty bands.
    """
    x_km = model['block_x_centers'] / 1e3
    true_depths = model['true_depths']

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

    # --- Panel 1: Basement depth with uncertainty ---
    ax1 = axes[0]
    ax1.fill_between(x_km, posterior['ci_5'], posterior['ci_95'],
                     color='#3498db', alpha=0.25, label='90% Credible Interval')
    ax1.fill_between(x_km, posterior['ci_2_5'], posterior['ci_97_5'],
                     color='#3498db', alpha=0.1, label='95% Credible Interval')
    ax1.plot(x_km, true_depths, 'k-', linewidth=2.5, label='True Basement')
    ax1.plot(x_km, posterior['mean'], 'r--', linewidth=2, label='MCMC Mean Estimate')

    ax1.set_xlabel('Distance along profile (km)', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title('Basement Depth: True vs MCMC Estimate with Uncertainty', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    # Add RMS and coverage text
    from .mcmc_inversion import compute_coverage
    rms = np.sqrt(np.mean((posterior['mean'] - true_depths)**2))
    cov90 = compute_coverage(true_depths, posterior, ci_level=90)
    ax1.text(0.02, 0.02,
             f'RMS Error: {rms:.1f} m\n90% Coverage: {cov90*100:.0f}%\nSamples: {posterior["n_samples"]}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # --- Panel 2: Uncertainty (std) along profile ---
    ax2 = axes[1]
    ax2.bar(x_km, posterior['std'], width=(x_km[1]-x_km[0])*0.8,
            color='#e74c3c', alpha=0.7, edgecolor='darkred')
    ax2.set_xlabel('Distance along profile (km)', fontsize=12)
    ax2.set_ylabel('Std Dev (m)', fontsize=12)
    ax2.set_title('Depth Uncertainty Along Profile', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Gravity fit ---
    ax3 = axes[2]
    ax3.plot(data['obs_x'] / 1e3, data['gravity_obs'], 'ko', markersize=5, label='Observed (with noise)')
    ax3.plot(data['obs_x'] / 1e3, data['gravity_true'], 'b-', linewidth=1.5, label='True (no noise)')

    # Compute gravity from MCMC mean
    from .forward_model import compute_gravity_for_basin_fast
    from .utils import make_density_func
    # We'll pass density_func through data dict or recompute
    ax3.set_xlabel('Distance along profile (km)', fontsize=12)
    ax3.set_ylabel('Gravity (mGal)', fontsize=12)
    ax3.set_title('Gravity Data', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_trace_and_acceptance(result, save_path=None):
    """
    MCMC diagnostics: misfit trace plot and acceptance rate.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- Misfit trace ---
    ax1 = axes[0]
    ax1.plot(result['all_misfits'], 'b-', linewidth=0.3, alpha=0.7)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Misfit S(m)', fontsize=12)
    ax1.set_title('MCMC Convergence: Misfit vs Iteration', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Mark burn-in
    burn_in = result['n_iterations'] // 2
    ax1.axvline(x=burn_in, color='red', linestyle='--', linewidth=1.5, label=f'Burn-in cutoff ({burn_in})')
    ax1.legend(fontsize=10)

    # --- Running acceptance rate ---
    ax2 = axes[1]
    window = max(100, result['n_iterations'] // 50)
    chain_misfits = result['all_misfits']
    # Compute running acceptance: count changes in misfit
    accepted = np.zeros(len(chain_misfits))
    for i in range(1, len(chain_misfits)):
        if chain_misfits[i] != chain_misfits[i-1]:
            accepted[i] = 1
    # Running average
    running_rate = np.convolve(accepted, np.ones(window)/window, mode='valid') * 100
    ax2.plot(range(len(running_rate)), running_rate, 'g-', linewidth=0.8)
    ax2.axhline(y=20, color='red', linestyle=':', alpha=0.5, label='Target range (20-50%)')
    ax2.axhline(y=50, color='red', linestyle=':', alpha=0.5)
    ax2.fill_between(range(len(running_rate)), 20, 50, alpha=0.1, color='green')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax2.set_title(f'Running Acceptance Rate (window={window})', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 80)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_depth_histograms(model, posterior, block_indices=None, save_path=None):
    """
    Posterior depth histograms at selected blocks.
    """
    if block_indices is None:
        n = model['n_blocks']
        block_indices = [0, n//4, n//2, 3*n//4, n-1]

    n_plots = len(block_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    x_km = model['block_x_centers'] / 1e3

    for ax, idx in zip(axes, block_indices):
        samples = posterior['samples'][:, idx]
        true_d = model['true_depths'][idx]

        ax.hist(samples, bins=40, density=True, color='#3498db', alpha=0.7, edgecolor='white')
        ax.axvline(x=true_d, color='black', linewidth=2, linestyle='-', label=f'True: {true_d:.0f}m')
        ax.axvline(x=posterior['mean'][idx], color='red', linewidth=2, linestyle='--',
                   label=f'Mean: {posterior["mean"][idx]:.0f}m')

        # CI shading
        ax.axvspan(posterior['ci_5'][idx], posterior['ci_95'][idx],
                   alpha=0.15, color='orange', label='90% CI')

        ax.set_xlabel('Depth (m)', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.set_title(f'Block {idx+1}\nx = {x_km[idx]:.1f} km', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

    plt.suptitle('Posterior Depth Distributions at Selected Blocks', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_all_chain_models(model, result, burn_in_frac=0.5, save_path=None):
    """
    Plot all accepted models (post burn-in) as semi-transparent lines
    to show the posterior ensemble.
    """
    x_km = model['block_x_centers'] / 1e3
    chain = result['chain']
    burn_in = int(len(chain) * burn_in_frac)
    samples = chain[burn_in:]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot subset of samples (max 500 for visibility)
    n_show = min(500, len(samples))
    indices = np.linspace(0, len(samples)-1, n_show, dtype=int)
    for i in indices:
        ax.plot(x_km, samples[i], color='#3498db', alpha=0.03, linewidth=0.5)

    ax.plot(x_km, model['true_depths'], 'k-', linewidth=2.5, label='True Basement')
    ax.plot(x_km, np.mean(samples, axis=0), 'r--', linewidth=2, label='MCMC Mean')

    ax.set_xlabel('Distance along profile (km)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Posterior Ensemble: {n_show} Models from MCMC Chain', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# =====================================================================
# 3D Grid Visualization Functions
# =====================================================================

# Colormap: jet_r = red (shallow) → blue (deep), matching geophysics convention
DEPTH_CMAP = 'jet_r'
UNCERT_CMAP = 'hot_r'
GRAV_CMAP = 'jet'


def _interpolate_grid(x_km, y_km, values, factor=5):
    """Interpolate block-center values to a finer grid for smooth plotting."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((x_km, y_km), values, method='linear',
                                      bounds_error=False, fill_value=None)
    x_fine = np.linspace(x_km[0], x_km[-1], len(x_km) * factor)
    y_fine = np.linspace(y_km[0], y_km[-1], len(y_km) * factor)
    Xf, Yf = np.meshgrid(x_fine, y_fine, indexing='ij')
    pts = np.column_stack([Xf.ravel(), Yf.ravel()])
    Zf = interp(pts).reshape(Xf.shape)
    return x_fine, y_fine, Xf, Yf, Zf


def _annotate_cells(ax, x_centers_km, y_centers_km, values, fmt='.0f',
                    fontsize=9, color='black'):
    """Helper to annotate cell values on a pcolormesh plot."""
    for i in range(len(x_centers_km)):
        for j in range(len(y_centers_km)):
            ax.text(x_centers_km[i], y_centers_km[j],
                    f'{values[i, j]:{fmt}}',
                    ha='center', va='center', fontsize=fontsize,
                    color=color, fontweight='bold')


def plot_depth_comparison(model, posterior, save_path=None):
    """
    Main result: True vs Estimated depth maps (smooth, jet colormap).
    """
    from .mcmc_inversion import compute_coverage

    x_km = model['block_x_centers'] / 1e3
    y_km = model['block_y_centers'] / 1e3
    true_d = model['true_depths']
    est_d = posterior['mean']

    rms = np.sqrt(np.mean((est_d - true_d)**2))
    cov90 = compute_coverage(true_d, posterior, ci_level=90)

    vmin = min(true_d.min(), est_d.min())
    vmax = max(true_d.max(), est_d.max())

    # Interpolate for smooth appearance
    _, _, Xf1, Yf1, Zf_true = _interpolate_grid(x_km, y_km, true_d)
    _, _, Xf2, Yf2, Zf_est = _interpolate_grid(x_km, y_km, est_d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.pcolormesh(Xf1, Yf1, Zf_true, cmap=DEPTH_CMAP,
                          vmin=vmin, vmax=vmax, shading='auto')
    ax1.set_xlabel('X (km)', fontsize=14)
    ax1.set_ylabel('Y (km)', fontsize=14)
    ax1.set_title('True Basement Depth', fontsize=16, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=12)

    im2 = ax2.pcolormesh(Xf2, Yf2, Zf_est, cmap=DEPTH_CMAP,
                          vmin=vmin, vmax=vmax, shading='auto')
    ax2.set_xlabel('X (km)', fontsize=14)
    ax2.set_ylabel('Y (km)', fontsize=14)
    ax2.set_title('MCMC Estimated Depth', fontsize=16, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.tick_params(labelsize=12)

    fig.subplots_adjust(bottom=0.18)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Depth (m)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(f'Basement Depth Comparison  |  RMS Error: {rms:.0f} m  |  '
                 f'90% Coverage: {cov90*100:.0f}%',
                 fontsize=17, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_depth_3d_surface(model, posterior, save_path=None):
    """
    3D perspective surface plots: True vs Estimated basement depth.
    Depth shown as negative z (surface at 0, deeper = more negative).
    """
    from mpl_toolkits.mplot3d import Axes3D
    from .mcmc_inversion import compute_coverage

    x_km = model['block_x_centers'] / 1e3
    y_km = model['block_y_centers'] / 1e3
    true_d = model['true_depths']
    est_d = posterior['mean']

    rms = np.sqrt(np.mean((est_d - true_d)**2))
    cov90 = compute_coverage(true_d, posterior, ci_level=90)

    # Interpolate for smooth surfaces
    _, _, Xf, Yf, Zf_true = _interpolate_grid(x_km, y_km, true_d, factor=6)
    _, _, _, _, Zf_est = _interpolate_grid(x_km, y_km, est_d, factor=6)

    # Show depth as negative (surface = 0, deeper = negative)
    Zf_true_neg = -Zf_true
    Zf_est_neg = -Zf_est

    vmin = -max(true_d.max(), est_d.max())
    vmax = -min(true_d.min(), est_d.min())

    fig = plt.figure(figsize=(18, 8))

    # True depth surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(Xf, Yf, Zf_true_neg, cmap=DEPTH_CMAP,
                              vmin=vmin, vmax=vmax,
                              edgecolor='none', alpha=0.95,
                              rstride=1, cstride=1)
    ax1.set_xlabel('X (km)', fontsize=12, labelpad=10)
    ax1.set_ylabel('Y (km)', fontsize=12, labelpad=10)
    ax1.set_zlabel('Depth (m)', fontsize=12, labelpad=10)
    ax1.set_title('True Basement Depth', fontsize=15, fontweight='bold',
                   pad=15)
    ax1.view_init(elev=30, azim=225)
    ax1.tick_params(labelsize=9)

    # Estimated depth surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(Xf, Yf, Zf_est_neg, cmap=DEPTH_CMAP,
                              vmin=vmin, vmax=vmax,
                              edgecolor='none', alpha=0.95,
                              rstride=1, cstride=1)
    ax2.set_xlabel('X (km)', fontsize=12, labelpad=10)
    ax2.set_ylabel('Y (km)', fontsize=12, labelpad=10)
    ax2.set_zlabel('Depth (m)', fontsize=12, labelpad=10)
    ax2.set_title('MCMC Estimated Depth', fontsize=15, fontweight='bold',
                   pad=15)
    ax2.view_init(elev=30, azim=225)
    ax2.tick_params(labelsize=9)

    # Match z-limits
    zmin = min(ax1.get_zlim()[0], ax2.get_zlim()[0])
    zmax = max(ax1.get_zlim()[1], ax2.get_zlim()[1])
    ax1.set_zlim(zmin, zmax)
    ax2.set_zlim(zmin, zmax)

    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.025])
    cbar = fig.colorbar(surf1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Depth (m)', fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.suptitle(f'3D Basement Surface  |  RMS Error: {rms:.0f} m  |  '
                 f'90% Coverage: {cov90*100:.0f}%',
                 fontsize=17, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_uncertainty_map(model, posterior, save_path=None):
    """
    Error and uncertainty maps (smooth, interpolated).
    """
    x_km = model['block_x_centers'] / 1e3
    y_km = model['block_y_centers'] / 1e3

    abs_error = np.abs(posterior['mean'] - model['true_depths'])
    uncertainty = posterior['std']

    _, _, Xf1, Yf1, Zf_err = _interpolate_grid(x_km, y_km, abs_error)
    _, _, Xf2, Yf2, Zf_unc = _interpolate_grid(x_km, y_km, uncertainty)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.pcolormesh(Xf1, Yf1, Zf_err, cmap='OrRd', shading='auto')
    ax1.set_xlabel('X (km)', fontsize=14)
    ax1.set_ylabel('Y (km)', fontsize=14)
    ax1.set_title('Absolute Depth Error |Est - True|', fontsize=15,
                   fontweight='bold')
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=12)
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cb1.set_label('Error (m)', fontsize=13)

    im2 = ax2.pcolormesh(Xf2, Yf2, Zf_unc, cmap=UNCERT_CMAP, shading='auto')
    ax2.set_xlabel('X (km)', fontsize=14)
    ax2.set_ylabel('Y (km)', fontsize=14)
    ax2.set_title('Posterior Uncertainty (Std Dev)', fontsize=15,
                   fontweight='bold')
    ax2.set_aspect('equal')
    ax2.tick_params(labelsize=12)
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cb2.set_label('Std Dev (m)', fontsize=13)

    fig.suptitle('Depth Error vs Uncertainty',
                 fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_uncertainty_3d_surface(model, posterior, save_path=None):
    """
    3D surface plot of posterior uncertainty (std dev).
    """
    from mpl_toolkits.mplot3d import Axes3D

    x_km = model['block_x_centers'] / 1e3
    y_km = model['block_y_centers'] / 1e3
    uncertainty = posterior['std']

    _, _, Xf, Yf, Zf = _interpolate_grid(x_km, y_km, uncertainty, factor=6)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Xf, Yf, Zf, cmap=UNCERT_CMAP,
                            edgecolor='none', alpha=0.95,
                            rstride=1, cstride=1)
    ax.set_xlabel('X (km)', fontsize=13, labelpad=10)
    ax.set_ylabel('Y (km)', fontsize=13, labelpad=10)
    ax.set_zlabel('Std Dev (m)', fontsize=13, labelpad=10)
    ax.set_title('Posterior Depth Uncertainty', fontsize=16, fontweight='bold',
                  pad=15)
    ax.view_init(elev=30, azim=225)
    ax.tick_params(labelsize=10)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Std Dev (m)', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_depth_cross_sections(model, posterior, save_path=None):
    """
    Cross-section plots through the 3D grid showing depth + uncertainty bands.
    Top: X-direction slice at middle Y.
    Bottom: Y-direction slice at middle X.
    """
    x_km = model['block_x_centers'] / 1e3
    y_km = model['block_y_centers'] / 1e3
    true_d = model['true_depths']
    Nx, Ny = true_d.shape

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # --- X-direction cross-section (at middle Y) ---
    iy_mid = Ny // 2
    ax = ax1
    ax.fill_between(x_km, posterior['ci_2_5'][:, iy_mid],
                     posterior['ci_97_5'][:, iy_mid],
                     color='#3498db', alpha=0.12, label='95% CI')
    ax.fill_between(x_km, posterior['ci_5'][:, iy_mid],
                     posterior['ci_95'][:, iy_mid],
                     color='#3498db', alpha=0.25, label='90% CI')
    ax.plot(x_km, true_d[:, iy_mid], 'k-', linewidth=2.5,
            marker='s', markersize=6, label='True Basement')
    ax.plot(x_km, posterior['mean'][:, iy_mid], 'r--', linewidth=2,
            marker='o', markersize=5, label='MCMC Mean')
    ax.set_xlabel('X Distance (km)', fontsize=14)
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_title(f'X-Direction Cross-Section at Y = {y_km[iy_mid]:.0f} km',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='lower left')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # --- Y-direction cross-section (at middle X) ---
    ix_mid = Nx // 2
    ax = ax2
    ax.fill_between(y_km, posterior['ci_2_5'][ix_mid, :],
                     posterior['ci_97_5'][ix_mid, :],
                     color='#3498db', alpha=0.12, label='95% CI')
    ax.fill_between(y_km, posterior['ci_5'][ix_mid, :],
                     posterior['ci_95'][ix_mid, :],
                     color='#3498db', alpha=0.25, label='90% CI')
    ax.plot(y_km, true_d[ix_mid, :], 'k-', linewidth=2.5,
            marker='s', markersize=6, label='True Basement')
    ax.plot(y_km, posterior['mean'][ix_mid, :], 'r--', linewidth=2,
            marker='o', markersize=5, label='MCMC Mean')
    ax.set_xlabel('Y Distance (km)', fontsize=14)
    ax.set_ylabel('Depth (m)', fontsize=14)
    ax.set_title(f'Y-Direction Cross-Section at X = {x_km[ix_mid]:.0f} km',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='lower left')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_gravity_fit_3d(model, data, posterior, density_func,
                         n_sublayers=10, save_path=None):
    """
    Gravity data fit: Observed vs Computed vs Residual maps (smooth).
    """
    from .forward_model import compute_gravity_for_basin

    x_km = model['block_x_centers'] / 1e3
    y_km = model['block_y_centers'] / 1e3
    Nx = model['nx_blocks']
    Ny = model['ny_blocks']

    gravity_calc = compute_gravity_for_basin(
        data['obs_x'], data['obs_y'],
        model['block_x_edges'], model['block_y_edges'],
        posterior['mean'], density_func, n_sublayers
    )

    g_obs_grid = data['gravity_obs'].reshape(Nx, Ny)
    g_calc_grid = gravity_calc.reshape(Nx, Ny)
    residual_grid = g_obs_grid - g_calc_grid

    gmin = min(g_obs_grid.min(), g_calc_grid.min())
    gmax = max(g_obs_grid.max(), g_calc_grid.max())

    # Interpolate
    _, _, Xf1, Yf1, Zf_obs = _interpolate_grid(x_km, y_km, g_obs_grid)
    _, _, Xf2, Yf2, Zf_calc = _interpolate_grid(x_km, y_km, g_calc_grid)
    _, _, Xf3, Yf3, Zf_res = _interpolate_grid(x_km, y_km, residual_grid)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    im1 = ax1.pcolormesh(Xf1, Yf1, Zf_obs, cmap=GRAV_CMAP,
                          vmin=gmin, vmax=gmax, shading='auto')
    ax1.set_xlabel('X (km)', fontsize=13)
    ax1.set_ylabel('Y (km)', fontsize=13)
    ax1.set_title('Observed Gravity', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=11)
    fig.colorbar(im1, ax=ax1, shrink=0.8, label='mGal')

    im2 = ax2.pcolormesh(Xf2, Yf2, Zf_calc, cmap=GRAV_CMAP,
                          vmin=gmin, vmax=gmax, shading='auto')
    ax2.set_xlabel('X (km)', fontsize=13)
    ax2.set_ylabel('Y (km)', fontsize=13)
    ax2.set_title('Computed Gravity (MCMC Mean)', fontsize=14,
                   fontweight='bold')
    ax2.set_aspect('equal')
    ax2.tick_params(labelsize=11)
    fig.colorbar(im2, ax=ax2, shrink=0.8, label='mGal')

    res_max = max(abs(residual_grid.min()), abs(residual_grid.max()))
    if res_max < 1e-10:
        res_max = 1.0
    im3 = ax3.pcolormesh(Xf3, Yf3, Zf_res, cmap='RdBu_r',
                          vmin=-res_max, vmax=res_max, shading='auto')
    ax3.set_xlabel('X (km)', fontsize=13)
    ax3.set_ylabel('Y (km)', fontsize=13)
    ax3.set_title('Residual (Obs - Calc)', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.tick_params(labelsize=11)
    fig.colorbar(im3, ax=ax3, shrink=0.8, label='mGal')

    rms_grav = np.sqrt(np.mean(residual_grid**2))
    fig.suptitle(f'Gravity Data Fit  |  RMS Residual: {rms_grav:.3f} mGal',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def save_results(model, data, result, posterior, density_params,
                  save_path='results_data.npz'):
    """
    Save all MCMC results to a .npz file for later reuse and plotting.

    Parameters
    ----------
    model : dict from create_synthetic_basin_3d()
    data : dict from generate_synthetic_gravity_3d()
    result : dict from run_mcmc_3d()
    posterior : dict from process_chain_3d()
    density_params : dict with keys 'drho_0', 'lam'
    save_path : str
    """
    np.savez_compressed(
        save_path,
        # Model
        true_depths=model['true_depths'],
        block_x_edges=model['block_x_edges'],
        block_y_edges=model['block_y_edges'],
        block_x_centers=model['block_x_centers'],
        block_y_centers=model['block_y_centers'],
        nx_blocks=model['nx_blocks'],
        ny_blocks=model['ny_blocks'],
        x_length=model['x_length'],
        y_length=model['y_length'],
        # Data
        obs_x=data['obs_x'],
        obs_y=data['obs_y'],
        gravity_true=data['gravity_true'],
        gravity_obs=data['gravity_obs'],
        noise_std=data['noise_std'],
        # MCMC result
        chain=result['chain'],
        misfit_chain=result['misfit_chain'],
        acceptance_rate=result['acceptance_rate'],
        n_iterations=result['n_iterations'],
        all_misfits=result['all_misfits'],
        # Posterior
        posterior_mean=posterior['mean'],
        posterior_std=posterior['std'],
        posterior_ci_5=posterior['ci_5'],
        posterior_ci_95=posterior['ci_95'],
        posterior_ci_2_5=posterior['ci_2_5'],
        posterior_ci_97_5=posterior['ci_97_5'],
        n_samples=posterior['n_samples'],
        # Density params
        drho_0=density_params['drho_0'],
        lam=density_params['lam'],
    )
    print(f"Saved all results to: {save_path}")


def load_results(path):
    """
    Load saved results and reconstruct model/data/result/posterior dicts.

    Returns
    -------
    model, data, result, posterior, density_params : dicts
    """
    d = np.load(path, allow_pickle=False)

    model = {
        'true_depths': d['true_depths'],
        'block_x_edges': d['block_x_edges'],
        'block_y_edges': d['block_y_edges'],
        'block_x_centers': d['block_x_centers'],
        'block_y_centers': d['block_y_centers'],
        'nx_blocks': int(d['nx_blocks']),
        'ny_blocks': int(d['ny_blocks']),
        'x_length': float(d['x_length']),
        'y_length': float(d['y_length']),
    }

    data = {
        'obs_x': d['obs_x'],
        'obs_y': d['obs_y'],
        'gravity_true': d['gravity_true'],
        'gravity_obs': d['gravity_obs'],
        'noise_std': float(d['noise_std']),
    }

    result = {
        'chain': d['chain'],
        'misfit_chain': d['misfit_chain'],
        'acceptance_rate': float(d['acceptance_rate']),
        'n_iterations': int(d['n_iterations']),
        'all_misfits': d['all_misfits'],
    }

    posterior = {
        'mean': d['posterior_mean'],
        'std': d['posterior_std'],
        'ci_5': d['posterior_ci_5'],
        'ci_95': d['posterior_ci_95'],
        'ci_2_5': d['posterior_ci_2_5'],
        'ci_97_5': d['posterior_ci_97_5'],
        'n_samples': int(d['n_samples']),
    }

    density_params = {
        'drho_0': float(d['drho_0']),
        'lam': float(d['lam']),
    }

    return model, data, result, posterior, density_params
