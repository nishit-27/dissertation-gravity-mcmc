"""
Generate Exp-7-style Plot Suite for Cauvery Basin MCMC Results
================================================================
Loads results/exp_cauvery_real/results_data.npz and creates:
  1. depth_map.png              (2D posterior mean depth)
  2. depth_3d_surface.png       (3D perspective surface)
  3. uncertainty_map.png        (2D posterior std-dev map)
  4. uncertainty_3d_surface.png (3D std-dev surface)
  5. cross_sections.png         (X and Y cross-sections with 90% CI)
  6. gravity_fit.png            (observed, computed, residual)
  7. mcmc_diagnostics.png       (if all_misfits available in npz)

Adapted from Exp 7 visualizations for real-data (no ground truth).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator

from src.forward_model import compute_gravity_for_basin
from src.utils import make_density_func

NPZ_PATH = 'results/exp_cauvery_real/results_data.npz'
OUT_DIR = 'results/exp_cauvery_real'
DEPTH_CMAP = 'viridis_r'   # deeper = darker (intuitive)
UNCERT_CMAP = 'hot_r'

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading {NPZ_PATH} ...")
d = np.load(NPZ_PATH)
mean_d = d['mean_depths']
std_d = d['std_depths']
ci_5 = d['ci_5']
ci_95 = d['ci_95']
bx = d['block_x_edges']; by = d['block_y_edges']
obs_x = d['obs_x']; obs_y = d['obs_y']; obs_g = d['obs_gravity']
drho0 = float(d['drho_0']); lam = float(d['lam'])

nx, ny = mean_d.shape
x_km = 0.5 * (bx[:-1] + bx[1:]) / 1000
y_km = 0.5 * (by[:-1] + by[1:]) / 1000
X_km, Y_km = np.meshgrid(x_km, y_km, indexing='ij')
print(f"Grid: {nx}x{ny}, Depth range: {mean_d.min():.0f} - {mean_d.max():.0f} m")


def interp_smooth(Z, factor=5):
    """Interpolate a 2D grid to a finer grid for smooth visualization."""
    nx_, ny_ = Z.shape
    x_fine = np.linspace(0, nx_-1, nx_*factor)
    y_fine = np.linspace(0, ny_-1, ny_*factor)
    Xf, Yf = np.meshgrid(x_fine, y_fine, indexing='ij')
    interp = RegularGridInterpolator((np.arange(nx_), np.arange(ny_)), Z,
                                     method='linear')
    Zf = interp(np.column_stack([Xf.ravel(), Yf.ravel()])).reshape(Xf.shape)
    # Map back to km
    xf_km = np.interp(x_fine, np.arange(nx_), x_km)
    yf_km = np.interp(y_fine, np.arange(ny_), y_km)
    Xfk, Yfk = np.meshgrid(xf_km, yf_km, indexing='ij')
    return Xfk, Yfk, Zf


# ============================================================
# 1. Depth map (2D)
# ============================================================
print("[1/7] depth_map.png")
Xf, Yf, Zf = interp_smooth(mean_d)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(Xf, Yf, Zf, cmap=DEPTH_CMAP, shading='auto')
cb = plt.colorbar(im, ax=ax, label='Basement Depth (m)')
cb.ax.tick_params(labelsize=12)
ax.set_xlabel('X (km)', fontsize=13)
ax.set_ylabel('Y (km)', fontsize=13)
ax.set_title(f'Cauvery — MCMC Posterior Mean Basement Depth\n'
             f'Range: {mean_d.min():.0f} – {mean_d.max():.0f} m  |  '
             f'Ganguli & Pal 2023: 3000 – 5400 m',
             fontsize=13, fontweight='bold')
ax.set_aspect('equal')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'depth_map.png'), dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# 2. 3D depth surface
# ============================================================
print("[2/7] depth_3d_surface.png")
Xf6, Yf6, Zf6 = interp_smooth(mean_d, factor=6)
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Xf6, Yf6, -Zf6, cmap=DEPTH_CMAP,
                       edgecolor='none', alpha=0.95,
                       rstride=1, cstride=1)
ax.set_xlabel('X (km)', fontsize=11, labelpad=10)
ax.set_ylabel('Y (km)', fontsize=11, labelpad=10)
ax.set_zlabel('Depth (m, negative down)', fontsize=11, labelpad=10)
ax.set_title(f'Cauvery — 3D Basement Surface\nmax depth {mean_d.max():.0f} m',
             fontsize=13, fontweight='bold', pad=15)
ax.view_init(elev=30, azim=225)
fig.colorbar(surf, shrink=0.6, label='Depth (m, negative down)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'depth_3d_surface.png'),
            dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# 3. Uncertainty map (2D)
# ============================================================
print("[3/7] uncertainty_map.png")
Xf, Yf, Uf = interp_smooth(std_d)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(Xf, Yf, Uf, cmap=UNCERT_CMAP, shading='auto')
cb = plt.colorbar(im, ax=ax, label='Posterior Std (m)')
cb.ax.tick_params(labelsize=12)
ax.set_xlabel('X (km)', fontsize=13)
ax.set_ylabel('Y (km)', fontsize=13)
ax.set_title(f'Cauvery — Posterior Uncertainty (1σ)\n'
             f'Mean: {std_d.mean():.0f} m  |  Max: {std_d.max():.0f} m',
             fontsize=13, fontweight='bold')
ax.set_aspect('equal')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'uncertainty_map.png'),
            dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# 4. 3D uncertainty surface
# ============================================================
print("[4/7] uncertainty_3d_surface.png")
Xf6u, Yf6u, Uf6 = interp_smooth(std_d, factor=6)
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Xf6u, Yf6u, Uf6, cmap=UNCERT_CMAP,
                       edgecolor='none', alpha=0.95,
                       rstride=1, cstride=1)
ax.set_xlabel('X (km)', fontsize=11, labelpad=10)
ax.set_ylabel('Y (km)', fontsize=11, labelpad=10)
ax.set_zlabel('Std Dev (m)', fontsize=11, labelpad=10)
ax.set_title(f'Cauvery — 3D Posterior Uncertainty',
             fontsize=13, fontweight='bold', pad=15)
ax.view_init(elev=30, azim=225)
fig.colorbar(surf, shrink=0.6, label='Std (m)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'uncertainty_3d_surface.png'),
            dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# 5. Cross-sections with CI bands
# ============================================================
print("[5/7] cross_sections.png")
iy_mid = ny // 2
ix_mid = nx // 2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

# X-section (across the basin at mid-Y)
ax1.plot(x_km, mean_d[:, iy_mid], 'b-o', linewidth=2, markersize=6,
         label='MCMC mean')
ax1.fill_between(x_km, ci_5[:, iy_mid], ci_95[:, iy_mid],
                 alpha=0.3, color='blue', label='90% CI')
ax1.set_xlabel('X (km)', fontsize=12)
ax1.set_ylabel('Depth (m)', fontsize=12)
ax1.set_title(f'E–W Cross-Section at Y = {y_km[iy_mid]:.1f} km',
              fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Y-section (across the basin at mid-X)
ax2.plot(y_km, mean_d[ix_mid, :], 'r-o', linewidth=2, markersize=6,
         label='MCMC mean')
ax2.fill_between(y_km, ci_5[ix_mid, :], ci_95[ix_mid, :],
                 alpha=0.3, color='red', label='90% CI')
ax2.set_xlabel('Y (km)', fontsize=12)
ax2.set_ylabel('Depth (m)', fontsize=12)
ax2.set_title(f'N–S Cross-Section at X = {x_km[ix_mid]:.1f} km',
              fontsize=13, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cross_sections.png'),
            dpi=200, bbox_inches='tight')
plt.close()


# ============================================================
# 6. Gravity fit (observed, computed, residual)
# ============================================================
print("[6/7] gravity_fit.png  (forward-modeling posterior mean...)")
density_func = make_density_func('exponential', drho_0=drho0, lam=lam)
g_calc = compute_gravity_for_basin(obs_x, obs_y,
                                    bx, by, mean_d,
                                    density_func, n_sublayers=10)
residual = obs_g - g_calc
rms = np.sqrt(np.mean(residual**2))

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
vmin = min(obs_g.min(), g_calc.min())
vmax = max(obs_g.max(), g_calc.max())

sc0 = axes[0].scatter(obs_x/1000, obs_y/1000, c=obs_g, cmap='RdBu_r',
                      vmin=vmin, vmax=vmax, s=60, edgecolors='k', linewidth=0.2)
axes[0].set_title(f'Observed (basin signal)\nrange {obs_g.min():.1f} to {obs_g.max():.1f} mGal',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)')
axes[0].set_aspect('equal')
plt.colorbar(sc0, ax=axes[0], label='mGal')

sc1 = axes[1].scatter(obs_x/1000, obs_y/1000, c=g_calc, cmap='RdBu_r',
                      vmin=vmin, vmax=vmax, s=60, edgecolors='k', linewidth=0.2)
axes[1].set_title(f'Computed (MCMC mean)\nrange {g_calc.min():.1f} to {g_calc.max():.1f} mGal',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('X (km)')
axes[1].set_aspect('equal')
plt.colorbar(sc1, ax=axes[1], label='mGal')

rmax = np.abs(residual).max()
sc2 = axes[2].scatter(obs_x/1000, obs_y/1000, c=residual, cmap='RdBu_r',
                      vmin=-rmax, vmax=rmax, s=60, edgecolors='k', linewidth=0.2)
axes[2].set_title(f'Residual (obs − calc)\nRMS = {rms:.2f} mGal',
                  fontsize=12, fontweight='bold')
axes[2].set_xlabel('X (km)')
axes[2].set_aspect('equal')
plt.colorbar(sc2, ax=axes[2], label='mGal')

fig.suptitle('Cauvery — Gravity Fit', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'gravity_fit.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print(f"  Gravity fit RMS: {rms:.2f} mGal")


# ============================================================
# 7. MCMC diagnostics (if all_misfits saved)
# ============================================================
print("[7/7] mcmc_diagnostics.png")
if 'all_misfits' in d.files:
    misfits = d['all_misfits']
    n_iter = len(misfits)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(misfits, 'b-', linewidth=0.3, alpha=0.7)
    axes[0].axvline(n_iter//2, color='red', ls='--', label=f'Burn-in ({n_iter//2:,})')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Misfit')
    axes[0].set_title('MCMC Convergence', fontsize=13, fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    window = 500
    accepted = np.zeros(n_iter)
    for i in range(1, n_iter):
        if misfits[i] != misfits[i-1]:
            accepted[i] = 1
    running = np.convolve(accepted, np.ones(window)/window, mode='valid') * 100
    axes[1].plot(running, 'g-', linewidth=0.8)
    axes[1].fill_between(np.arange(len(running)), 20, 50, alpha=0.1,
                         color='green', label='Target 20–50%')
    axes[1].axhline(20, color='red', ls=':', alpha=0.5)
    axes[1].axhline(50, color='red', ls=':', alpha=0.5)
    axes[1].set_ylim(0, 80)
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Acceptance Rate (%)')
    axes[1].set_title(f'Running Acceptance (window {window})', fontsize=13, fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mcmc_diagnostics.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved full diagnostics (trace + acceptance).")
else:
    # Keep existing misfit_trace.png as the diagnostic (saved during Run 1)
    print("  (all_misfits not in npz — keeping existing misfit_trace.png from Run 1.)")

print(f"\nDone. All plots in: {OUT_DIR}/")
