"""
Generate Exp-7-style Plots for All Real Data Experiments
=========================================================
For each real data experiment, creates the same set of figures as Exp 7:
  - depth_comparison.png        (MCMC vs USGS depth model)
  - depth_3d_surface.png        (3D perspective plots)
  - uncertainty_map.png         (difference + std dev maps)
  - uncertainty_3d_surface.png  (3D uncertainty surface)
  - cross_sections.png          (X and Y cross-sections with uncertainty)
  - gravity_fit.png             (observed, computed, residual)
  - mcmc_diagnostics.png        (misfit trace + acceptance trace)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from src.data_loader import prepare_edwards_data


# ============================================================
# Load real data for validation
# ============================================================
STUDY_BOUNDS = {
    'lon_min': -117.95, 'lon_max': -117.50,
    'lat_min': 34.82,   'lat_max': 35.10,
}

print("Loading real data for USGS model reference...")
data = prepare_edwards_data('real_data/edwards_afb/', study_bounds=STUDY_BOUNDS)
dgrid = data['depth_grid']
bwells = data['basement_wells']

def get_usgs_on_grid(block_x_edges, block_y_edges):
    """Average USGS depths within each block."""
    NX = len(block_x_edges) - 1
    NY = len(block_y_edges) - 1
    usgs_grid = np.zeros((NX, NY))
    for ix in range(NX):
        for iy in range(NY):
            mask = ((dgrid['x'] >= block_x_edges[ix]) &
                    (dgrid['x'] < block_x_edges[ix + 1]) &
                    (dgrid['y'] >= block_y_edges[iy]) &
                    (dgrid['y'] < block_y_edges[iy + 1]))
            usgs_grid[ix, iy] = np.mean(dgrid['depth_m'][mask]) if mask.sum() > 0 else np.nan
    return usgs_grid


# ============================================================
# Plotting functions
# ============================================================
def plot_depth_comparison(mcmc_depth, usgs_depth, block_x_edges, block_y_edges,
                          title, save_path):
    """Side-by-side: MCMC mean depth vs USGS model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    vmax = max(np.nanmax(mcmc_depth), np.nanmax(usgs_depth))
    vmin = 0

    # MCMC mean
    im0 = axes[0].pcolormesh(block_x_edges/1000, block_y_edges/1000, mcmc_depth.T,
                              cmap='viridis_r', shading='flat', vmin=vmin, vmax=vmax)
    plt.colorbar(im0, ax=axes[0], label='Depth (m)')
    axes[0].set_title('MCMC Posterior Mean')
    axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)'); axes[0].set_aspect('equal')

    # USGS model
    im1 = axes[1].pcolormesh(block_x_edges/1000, block_y_edges/1000, usgs_depth.T,
                              cmap='viridis_r', shading='flat', vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=axes[1], label='Depth (m)')
    axes[1].set_title('USGS Deterministic Model')
    axes[1].set_xlabel('X (km)'); axes[1].set_ylabel('Y (km)'); axes[1].set_aspect('equal')

    # Difference
    diff = mcmc_depth - usgs_depth
    vmax_d = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
    im2 = axes[2].pcolormesh(block_x_edges/1000, block_y_edges/1000, diff.T,
                              cmap='RdBu_r', shading='flat', vmin=-vmax_d, vmax=vmax_d)
    plt.colorbar(im2, ax=axes[2], label='MCMC - USGS (m)')
    axes[2].set_title(f'Difference (RMS={np.sqrt(np.nanmean(diff**2)):.0f}m)')
    axes[2].set_xlabel('X (km)'); axes[2].set_ylabel('Y (km)'); axes[2].set_aspect('equal')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_depth_3d_surface(mcmc_depth, usgs_depth, block_x_edges, block_y_edges,
                           title, save_path):
    """3D surface plots of MCMC and USGS depths."""
    fig = plt.figure(figsize=(16, 7))

    # Block centers for 3D surface
    xc = 0.5 * (block_x_edges[:-1] + block_x_edges[1:]) / 1000
    yc = 0.5 * (block_y_edges[:-1] + block_y_edges[1:]) / 1000
    XC, YC = np.meshgrid(xc, yc, indexing='ij')

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(XC, YC, -mcmc_depth, cmap='viridis_r', alpha=0.9,
                              edgecolor='black', linewidth=0.2)
    ax1.set_title('MCMC Depth (negative down)')
    ax1.set_xlabel('X (km)'); ax1.set_ylabel('Y (km)'); ax1.set_zlabel('-Depth (m)')
    fig.colorbar(surf1, ax=ax1, shrink=0.6, label='Depth (m)')

    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(XC, YC, -usgs_depth, cmap='viridis_r', alpha=0.9,
                              edgecolor='black', linewidth=0.2)
    ax2.set_title('USGS Depth (negative down)')
    ax2.set_xlabel('X (km)'); ax2.set_ylabel('Y (km)'); ax2.set_zlabel('-Depth (m)')
    fig.colorbar(surf2, ax=ax2, shrink=0.6, label='Depth (m)')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_uncertainty_map(mcmc_depth, std_depth, usgs_depth, block_x_edges, block_y_edges,
                         title, save_path):
    """|MCMC - USGS| and std dev side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    abs_err = np.abs(mcmc_depth - usgs_depth)
    vmax = max(np.nanmax(abs_err), np.nanmax(std_depth))

    im0 = axes[0].pcolormesh(block_x_edges/1000, block_y_edges/1000, abs_err.T,
                              cmap='hot_r', shading='flat', vmin=0, vmax=vmax)
    plt.colorbar(im0, ax=axes[0], label='|MCMC - USGS| (m)')
    axes[0].set_title(f'Absolute Difference (mean={np.nanmean(abs_err):.0f}m)')
    axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)'); axes[0].set_aspect('equal')

    im1 = axes[1].pcolormesh(block_x_edges/1000, block_y_edges/1000, std_depth.T,
                              cmap='hot_r', shading='flat', vmin=0, vmax=vmax)
    plt.colorbar(im1, ax=axes[1], label='Std Dev (m)')
    axes[1].set_title(f'Posterior Uncertainty (mean={np.nanmean(std_depth):.0f}m)')
    axes[1].set_xlabel('X (km)'); axes[1].set_ylabel('Y (km)'); axes[1].set_aspect('equal')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_uncertainty_3d_surface(std_depth, block_x_edges, block_y_edges,
                                 title, save_path):
    """3D surface of posterior uncertainty."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    xc = 0.5 * (block_x_edges[:-1] + block_x_edges[1:]) / 1000
    yc = 0.5 * (block_y_edges[:-1] + block_y_edges[1:]) / 1000
    XC, YC = np.meshgrid(xc, yc, indexing='ij')

    surf = ax.plot_surface(XC, YC, std_depth, cmap='hot_r', alpha=0.9,
                            edgecolor='black', linewidth=0.2)
    ax.set_title('Posterior Uncertainty Surface')
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Std Dev (m)')
    fig.colorbar(surf, ax=ax, shrink=0.6, label='Std Dev (m)')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_depth_cross_sections(mcmc_depth, std_depth, usgs_depth, ci_5, ci_95,
                               block_x_edges, block_y_edges, title, save_path):
    """Cross-sections through middle of grid with CI bands."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    xc = 0.5 * (block_x_edges[:-1] + block_x_edges[1:]) / 1000
    yc = 0.5 * (block_y_edges[:-1] + block_y_edges[1:]) / 1000
    NX, NY = mcmc_depth.shape

    # X cross-section (at middle Y)
    iy_mid = NY // 2
    axes[0].plot(xc, mcmc_depth[:, iy_mid], 'b-', linewidth=2, label='MCMC mean')
    axes[0].fill_between(xc, ci_5[:, iy_mid], ci_95[:, iy_mid],
                          alpha=0.3, color='blue', label='90% CI')
    axes[0].plot(xc, usgs_depth[:, iy_mid], 'r--', linewidth=2, label='USGS model')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Depth (m)')
    axes[0].set_title(f'X Cross-section (Y = {yc[iy_mid]:.0f} km)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Y cross-section (at middle X)
    ix_mid = NX // 2
    axes[1].plot(yc, mcmc_depth[ix_mid, :], 'b-', linewidth=2, label='MCMC mean')
    axes[1].fill_between(yc, ci_5[ix_mid, :], ci_95[ix_mid, :],
                          alpha=0.3, color='blue', label='90% CI')
    axes[1].plot(yc, usgs_depth[ix_mid, :], 'r--', linewidth=2, label='USGS model')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Y (km)'); axes[1].set_ylabel('Depth (m)')
    axes[1].set_title(f'Y Cross-section (X = {xc[ix_mid]:.0f} km)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gravity_fit(mcmc_depth, obs_x, obs_y, obs_gravity,
                     block_x_edges, block_y_edges, density_func,
                     title, save_path, n_sublayers=10):
    """Observed, computed, and residual gravity."""
    from src.forward_model import compute_gravity_for_basin

    computed = compute_gravity_for_basin(
        obs_x, obs_y, block_x_edges, block_y_edges,
        mcmc_depth, density_func, n_sublayers)
    residual = obs_gravity - computed

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    vmax = max(abs(obs_gravity.min()), abs(obs_gravity.max()))
    for ax, data_arr, ttl, lbl in [
        (axes[0], obs_gravity, 'Observed', 'Gravity (mGal)'),
        (axes[1], computed, 'Computed (from MCMC mean)', 'Gravity (mGal)'),
    ]:
        sc = ax.scatter(obs_x/1000, obs_y/1000, c=data_arr, cmap='RdYlBu_r',
                        s=50, vmin=-vmax, vmax=vmax, edgecolors='black', linewidths=0.3)
        plt.colorbar(sc, ax=ax, label=lbl)
        ax.set_title(ttl)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')

    vmax_r = max(abs(residual.min()), abs(residual.max()))
    sc = axes[2].scatter(obs_x/1000, obs_y/1000, c=residual, cmap='RdYlBu_r',
                          s=50, vmin=-vmax_r, vmax=vmax_r,
                          edgecolors='black', linewidths=0.3)
    plt.colorbar(sc, ax=axes[2], label='Residual (mGal)')
    rms_r = np.sqrt(np.mean(residual**2))
    axes[2].set_title(f'Residual (RMS={rms_r:.2f} mGal)')
    axes[2].set_xlabel('X (km)'); axes[2].set_ylabel('Y (km)'); axes[2].set_aspect('equal')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_mcmc_diagnostics(all_misfits, acceptance_rate, burn_in_frac, title, save_path):
    """Misfit trace and running acceptance rate."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    n_it = len(all_misfits)
    burn_in_idx = int(n_it * burn_in_frac)

    axes[0].semilogy(all_misfits, linewidth=0.5, alpha=0.7)
    axes[0].axvline(burn_in_idx, color='red', linestyle='--', label=f'Burn-in ({burn_in_frac*100:.0f}%)')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Misfit (log scale)')
    axes[0].set_title('Misfit Trace')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Running acceptance (approximate)
    axes[1].axhline(acceptance_rate * 100, color='blue', linewidth=2,
                     label=f'Mean: {acceptance_rate*100:.1f}%')
    axes[1].axhline(25, color='green', linestyle=':', label='Ideal range (25-50%)')
    axes[1].axhline(50, color='green', linestyle=':')
    axes[1].set_xlim(0, n_it)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Acceptance Rate (%)')
    axes[1].set_title('Acceptance Rate')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# Generate plots for each experiment
# ============================================================
experiments = [
    {
        'name': 'Edwards — Fixed Lambda (Exp A)',
        'dir': 'results/exp_edwards_fixed_lambda/',
        'lambda': 0.0003,
        'drho_0': -500.0,
    },
    {
        'name': 'Edwards — 2-Stage DC Correction (Exp B)',
        'dir': 'results/exp_edwards_2stage/',
        'lambda': 0.002,
        'drho_0': -500.0,
    },
    {
        'name': 'Edwards — 2-Stage Spatial Correction (Exp C / Option B)',
        'dir': 'results/exp_edwards_option_b/',
        'lambda': 0.00005,
        'drho_0': -500.0,
    },
]

for exp in experiments:
    npz_path = os.path.join(exp['dir'], 'results_data.npz')
    if not os.path.exists(npz_path):
        print(f"Skipping {exp['name']}: no results_data.npz found")
        continue

    print(f"\n{'='*60}")
    print(f"Generating plots for: {exp['name']}")
    print(f"{'='*60}")

    npz = np.load(npz_path, allow_pickle=True)
    mcmc_depth = npz['mean_depths']
    std_depth = npz['std_depths']
    ci_5 = npz['ci_5']
    ci_95 = npz['ci_95']
    block_x_edges = npz['block_x_edges']
    block_y_edges = npz['block_y_edges']

    print(f"  Grid: {mcmc_depth.shape}")
    print(f"  Depth range: {mcmc_depth.min():.0f} - {mcmc_depth.max():.0f} m")
    print(f"  Mean std:    {std_depth.mean():.0f} m")

    # Compute USGS on same grid
    usgs_grid = get_usgs_on_grid(block_x_edges, block_y_edges)

    # 1. Depth comparison
    plot_depth_comparison(mcmc_depth, usgs_grid, block_x_edges, block_y_edges,
                          exp['name'],
                          os.path.join(exp['dir'], 'depth_comparison.png'))
    print("  Saved: depth_comparison.png")

    # 2. 3D surface
    plot_depth_3d_surface(mcmc_depth, usgs_grid, block_x_edges, block_y_edges,
                           exp['name'],
                           os.path.join(exp['dir'], 'depth_3d_surface.png'))
    print("  Saved: depth_3d_surface.png")

    # 3. Uncertainty map
    plot_uncertainty_map(mcmc_depth, std_depth, usgs_grid, block_x_edges, block_y_edges,
                         exp['name'],
                         os.path.join(exp['dir'], 'uncertainty_map.png'))
    print("  Saved: uncertainty_map.png")

    # 4. 3D uncertainty
    plot_uncertainty_3d_surface(std_depth, block_x_edges, block_y_edges,
                                 exp['name'],
                                 os.path.join(exp['dir'], 'uncertainty_3d_surface.png'))
    print("  Saved: uncertainty_3d_surface.png")

    # 5. Cross-sections
    plot_depth_cross_sections(mcmc_depth, std_depth, usgs_grid, ci_5, ci_95,
                               block_x_edges, block_y_edges,
                               exp['name'],
                               os.path.join(exp['dir'], 'cross_sections.png'))
    print("  Saved: cross_sections.png")

    # 6. Gravity fit (needs obs_x, obs_y, obs_gravity from saved data)
    if 'obs_x' in npz and 'obs_y' in npz and 'obs_gravity' in npz:
        from src.utils import make_density_func
        df = make_density_func('exponential', drho_0=exp['drho_0'], lam=exp['lambda'])
        plot_gravity_fit(mcmc_depth, npz['obs_x'], npz['obs_y'], npz['obs_gravity'],
                         block_x_edges, block_y_edges, df,
                         exp['name'],
                         os.path.join(exp['dir'], 'gravity_fit.png'))
        print("  Saved: gravity_fit.png")

    # 7. MCMC diagnostics — we don't have saved all_misfits, skip unless available
    # (could add this to saves next time)

print(f"\n{'='*60}")
print("All plots generated.")
print(f"{'='*60}")
