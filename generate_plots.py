"""
Generic Exp-7 Plot Suite (basin-aware)
=======================================
Loads <OUT_DIR>/results_data.npz and creates the standard 7-plot suite,
labeled correctly for whichever basin produced the run. If the npz
contains `truth_depths` (i.e. real ground truth, like Eromanga), an
additional validation comparison plot is generated.

Configured via environment variables:
  RESULTS_OUT     Output directory containing results_data.npz
  BASIN_NAME      Title prefix (e.g. 'Cauvery', 'Eromanga / Cooper Basin')
  BENCHMARK_TXT   Optional text appended to depth-map title
                  (e.g. 'Ganguli & Pal 2023: 3000 – 5400 m'
                   or   'GA Z-horizon: <truth.min>–<truth.max> m')

Defaults preserve old Cauvery behavior so existing call sites keep working.
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

OUT_DIR = os.environ.get('RESULTS_OUT', os.environ.get('CAUVERY_OUT',
                         'results/exp_cauvery_real_run2'))
BASIN = os.environ.get('BASIN_NAME', 'Cauvery')
BENCHMARK = os.environ.get('BENCHMARK_TXT', '')
NPZ_PATH = os.path.join(OUT_DIR, 'results_data.npz')
DEPTH_CMAP = 'viridis_r'
UNCERT_CMAP = 'hot_r'

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Loading {NPZ_PATH}  |  basin={BASIN}")
d = np.load(NPZ_PATH)
mean_d = d['mean_depths']; std_d = d['std_depths']
ci_5 = d['ci_5']; ci_95 = d['ci_95']
bx = d['block_x_edges']; by = d['block_y_edges']
obs_x = d['obs_x']; obs_y = d['obs_y']; obs_g = d['obs_gravity']
drho0 = float(d['drho_0']); lam = float(d['lam'])
truth_depths = d['truth_depths'] if 'truth_depths' in d.files else None
has_truth = truth_depths is not None

# Auto benchmark text
if not BENCHMARK and has_truth:
    BENCHMARK = f'GA Z-horizon truth: {truth_depths.min():.0f}–{truth_depths.max():.0f} m'

nx, ny = mean_d.shape
x_km = 0.5 * (bx[:-1] + bx[1:]) / 1000
y_km = 0.5 * (by[:-1] + by[1:]) / 1000
print(f"Grid: {nx}x{ny}, depth {mean_d.min():.0f}–{mean_d.max():.0f} m, truth={'YES' if has_truth else 'NO'}")


def interp_smooth(Z, factor=5):
    nx_, ny_ = Z.shape
    xf = np.linspace(0, nx_-1, nx_*factor)
    yf = np.linspace(0, ny_-1, ny_*factor)
    Xf, Yf = np.meshgrid(xf, yf, indexing='ij')
    interp = RegularGridInterpolator((np.arange(nx_), np.arange(ny_)), Z, method='linear')
    Zf = interp(np.column_stack([Xf.ravel(), Yf.ravel()])).reshape(Xf.shape)
    xfk = np.interp(xf, np.arange(nx_), x_km)
    yfk = np.interp(yf, np.arange(ny_), y_km)
    Xfk, Yfk = np.meshgrid(xfk, yfk, indexing='ij')
    return Xfk, Yfk, Zf


# 1. Depth map
print("[1/8] depth_map.png")
Xf, Yf, Zf = interp_smooth(mean_d)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(Xf, Yf, Zf, cmap=DEPTH_CMAP, shading='auto')
plt.colorbar(im, ax=ax, label='Basement Depth (m)')
title = f'{BASIN} — MCMC Posterior Mean Basement Depth\nMCMC range: {mean_d.min():.0f} – {mean_d.max():.0f} m'
if BENCHMARK:
    title += f'   |   {BENCHMARK}'
ax.set_title(title, fontsize=12, fontweight='bold')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/depth_map.png', dpi=200, bbox_inches='tight'); plt.close()

# 2. 3D depth surface
print("[2/8] depth_3d_surface.png")
Xf6, Yf6, Zf6 = interp_smooth(mean_d, factor=6)
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Xf6, Yf6, -Zf6, cmap=DEPTH_CMAP, edgecolor='none', alpha=0.95)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m, neg = down)')
ax.set_title(f'{BASIN} — 3D Basement Surface\nmax depth {mean_d.max():.0f} m',
             fontsize=13, fontweight='bold')
ax.view_init(elev=30, azim=225)
fig.colorbar(surf, shrink=0.6)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/depth_3d_surface.png', dpi=200, bbox_inches='tight'); plt.close()

# 3. Uncertainty map
print("[3/8] uncertainty_map.png")
Xf, Yf, Uf = interp_smooth(std_d)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(Xf, Yf, Uf, cmap=UNCERT_CMAP, shading='auto')
plt.colorbar(im, ax=ax, label='Posterior Std (m)')
ax.set_title(f'{BASIN} — Posterior Uncertainty (1σ)\nmean {std_d.mean():.0f} m, max {std_d.max():.0f} m',
             fontsize=13, fontweight='bold')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/uncertainty_map.png', dpi=200, bbox_inches='tight'); plt.close()

# 4. 3D uncertainty
print("[4/8] uncertainty_3d_surface.png")
Xf6u, Yf6u, Uf6 = interp_smooth(std_d, factor=6)
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Xf6u, Yf6u, Uf6, cmap=UNCERT_CMAP, edgecolor='none', alpha=0.95)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Std (m)')
ax.set_title(f'{BASIN} — 3D Posterior Uncertainty', fontsize=13, fontweight='bold')
ax.view_init(elev=30, azim=225)
fig.colorbar(surf, shrink=0.6)
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/uncertainty_3d_surface.png', dpi=200, bbox_inches='tight'); plt.close()

# 5. Cross-sections (if truth, overlay)
print("[5/8] cross_sections.png")
iy_mid = ny // 2; ix_mid = nx // 2
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
ax1.plot(x_km, mean_d[:, iy_mid], 'b-o', linewidth=2, markersize=6, label='MCMC mean')
ax1.fill_between(x_km, ci_5[:, iy_mid], ci_95[:, iy_mid], alpha=0.3, color='blue', label='90% CI')
if has_truth:
    ax1.plot(x_km, truth_depths[:, iy_mid], 'r-s', linewidth=2, markersize=6, label='Truth (GA Z-horizon)')
ax1.set_xlabel('X (km)'); ax1.set_ylabel('Depth (m)'); ax1.invert_yaxis()
ax1.set_title(f'{BASIN} — E–W Cross-Section at Y = {y_km[iy_mid]:.1f} km', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3); ax1.legend()

ax2.plot(y_km, mean_d[ix_mid, :], 'b-o', linewidth=2, markersize=6, label='MCMC mean')
ax2.fill_between(y_km, ci_5[ix_mid, :], ci_95[ix_mid, :], alpha=0.3, color='blue', label='90% CI')
if has_truth:
    ax2.plot(y_km, truth_depths[ix_mid, :], 'r-s', linewidth=2, markersize=6, label='Truth (GA Z-horizon)')
ax2.set_xlabel('Y (km)'); ax2.set_ylabel('Depth (m)'); ax2.invert_yaxis()
ax2.set_title(f'{BASIN} — N–S Cross-Section at X = {x_km[ix_mid]:.1f} km', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3); ax2.legend()
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/cross_sections.png', dpi=200, bbox_inches='tight'); plt.close()

# 6. Gravity fit
print("[6/8] gravity_fit.png")
density_func = make_density_func('exponential', drho_0=drho0, lam=lam)
g_calc = compute_gravity_for_basin(obs_x, obs_y, bx, by, mean_d, density_func, n_sublayers=10)
residual = obs_g - g_calc
rms = np.sqrt(np.mean(residual**2))
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
vmin = min(obs_g.min(), g_calc.min()); vmax = max(obs_g.max(), g_calc.max())
sc0 = axes[0].scatter(obs_x/1000, obs_y/1000, c=obs_g, cmap='RdBu_r', vmin=vmin, vmax=vmax, s=60, edgecolors='k', linewidth=0.2)
axes[0].set_title(f'Observed (basin signal)\n{obs_g.min():.1f} to {obs_g.max():.1f} mGal', fontweight='bold')
plt.colorbar(sc0, ax=axes[0], label='mGal'); axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)'); axes[0].set_aspect('equal')
sc1 = axes[1].scatter(obs_x/1000, obs_y/1000, c=g_calc, cmap='RdBu_r', vmin=vmin, vmax=vmax, s=60, edgecolors='k', linewidth=0.2)
axes[1].set_title(f'Computed (MCMC mean)\n{g_calc.min():.1f} to {g_calc.max():.1f} mGal', fontweight='bold')
plt.colorbar(sc1, ax=axes[1], label='mGal'); axes[1].set_xlabel('X (km)'); axes[1].set_aspect('equal')
rmax = np.abs(residual).max()
sc2 = axes[2].scatter(obs_x/1000, obs_y/1000, c=residual, cmap='RdBu_r', vmin=-rmax, vmax=rmax, s=60, edgecolors='k', linewidth=0.2)
axes[2].set_title(f'Residual (obs − calc)\nRMS = {rms:.2f} mGal', fontweight='bold')
plt.colorbar(sc2, ax=axes[2], label='mGal'); axes[2].set_xlabel('X (km)'); axes[2].set_aspect('equal')
fig.suptitle(f'{BASIN} — Gravity Fit', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUT_DIR}/gravity_fit.png', dpi=200, bbox_inches='tight'); plt.close()
print(f"  Gravity fit RMS: {rms:.2f} mGal")

# 7. MCMC diagnostics
print("[7/8] mcmc_diagnostics.png")
if 'all_misfits' in d.files:
    misfits = d['all_misfits']; n_iter = len(misfits)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(misfits, 'b-', linewidth=0.3, alpha=0.7)
    axes[0].axvline(n_iter//2, color='red', ls='--', label=f'Burn-in ({n_iter//2:,})')
    axes[0].set_yscale('log'); axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Misfit')
    axes[0].set_title(f'{BASIN} — MCMC Convergence', fontsize=13, fontweight='bold')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    window = max(50, n_iter // 200)
    accepted = (np.diff(misfits, prepend=misfits[0]) != 0).astype(float)
    running = np.convolve(accepted, np.ones(window)/window, mode='valid') * 100
    axes[1].plot(running, 'g-', linewidth=0.8)
    axes[1].fill_between(np.arange(len(running)), 20, 50, alpha=0.1, color='green', label='Target 20–50%')
    axes[1].axhline(20, color='red', ls=':'); axes[1].axhline(50, color='red', ls=':')
    axes[1].set_ylim(0, 100); axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Acceptance Rate (%)')
    axes[1].set_title(f'Running Acceptance (window {window})', fontsize=13, fontweight='bold')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT_DIR}/mcmc_diagnostics.png', dpi=200, bbox_inches='tight'); plt.close()

# 8. Validation vs truth (only if truth available)
if has_truth:
    print("[8/8] validation_vs_truth.png  (real comparison)")
    err = mean_d - truth_depths
    rms_d = np.sqrt(np.mean(err**2))
    bias = np.mean(err)
    corr = np.corrcoef(truth_depths.ravel(), mean_d.ravel())[0, 1]
    cov90 = int(np.sum((ci_5 <= truth_depths) & (truth_depths <= ci_95))) / truth_depths.size

    fig, axes = plt.subplots(1, 4, figsize=(26, 6.5))
    # truth map
    Xf_t, Yf_t, Zf_t = interp_smooth(truth_depths)
    vmn = min(truth_depths.min(), mean_d.min()); vmx = max(truth_depths.max(), mean_d.max())
    im = axes[0].pcolormesh(Xf_t, Yf_t, Zf_t, cmap=DEPTH_CMAP, vmin=vmn, vmax=vmx, shading='auto')
    plt.colorbar(im, ax=axes[0], label='Depth (m)')
    axes[0].set_title(f'Real (GA Z-horizon)\n{truth_depths.min():.0f}–{truth_depths.max():.0f} m',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)'); axes[0].set_aspect('equal')
    # mcmc map
    Xf_m, Yf_m, Zf_m = interp_smooth(mean_d)
    im = axes[1].pcolormesh(Xf_m, Yf_m, Zf_m, cmap=DEPTH_CMAP, vmin=vmn, vmax=vmx, shading='auto')
    plt.colorbar(im, ax=axes[1], label='Depth (m)')
    axes[1].set_title(f'MCMC mean\n{mean_d.min():.0f}–{mean_d.max():.0f} m',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X (km)'); axes[1].set_aspect('equal')
    # error map
    err_smooth = interp_smooth(err)[2]
    emax = max(abs(err.min()), abs(err.max()))
    im = axes[2].pcolormesh(Xf_m, Yf_m, err_smooth, cmap='RdBu_r', vmin=-emax, vmax=emax, shading='auto')
    plt.colorbar(im, ax=axes[2], label='MCMC − Truth (m)')
    axes[2].set_title(f'Error map\nbias={bias:+.0f} m, RMS={rms_d:.0f} m',
                      fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X (km)'); axes[2].set_aspect('equal')
    # scatter
    axes[3].scatter(truth_depths.ravel(), mean_d.ravel(), s=80, alpha=0.7,
                    c=std_d.ravel(), cmap='hot_r', edgecolors='k', linewidth=0.3)
    lim = max(truth_depths.max(), mean_d.max()) * 1.1
    axes[3].plot([0, lim], [0, lim], 'k--', linewidth=1, label='1:1')
    axes[3].set_xlabel('Truth depth (m)'); axes[3].set_ylabel('MCMC depth (m)')
    axes[3].set_title(f'Scatter\nr={corr:.3f}, 90% cov={cov90*100:.0f}%', fontsize=12, fontweight='bold')
    axes[3].legend(); axes[3].grid(alpha=0.3); axes[3].set_aspect('equal')
    axes[3].set_xlim(0, lim); axes[3].set_ylim(0, lim)
    fig.suptitle(f'{BASIN} — MCMC vs Real Ground Truth (GA Cooper 3D)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/validation_vs_truth.png', dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Validation: RMS={rms_d:.0f} m, bias={bias:+.0f} m, r={corr:.3f}, cov={cov90*100:.0f}%")
else:
    print("[8/8] no truth — skipping validation_vs_truth.png")

# 9. 3D side-by-side (only if truth available)
if has_truth:
    print("[9/9] depth_3d_comparison.png  (real vs MCMC, 3D side-by-side)")
    Xf6t, Yf6t, Zf6t = interp_smooth(truth_depths, factor=6)
    Xf6m, Yf6m, Zf6m = interp_smooth(mean_d, factor=6)
    vmn = -max(truth_depths.max(), mean_d.max())
    vmx = -min(truth_depths.min(), mean_d.min())

    fig = plt.figure(figsize=(20, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    s1 = ax1.plot_surface(Xf6t, Yf6t, -Zf6t, cmap=DEPTH_CMAP,
                          vmin=vmn, vmax=vmx, edgecolor='none', alpha=0.95)
    ax1.set_xlabel('X (km)', labelpad=10); ax1.set_ylabel('Y (km)', labelpad=10)
    ax1.set_zlabel('Depth (m, neg = down)', labelpad=10)
    ax1.set_title(f'Real (GA Z-horizon)\n{truth_depths.min():.0f}–{truth_depths.max():.0f} m',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.view_init(elev=30, azim=225)

    ax2 = fig.add_subplot(122, projection='3d')
    s2 = ax2.plot_surface(Xf6m, Yf6m, -Zf6m, cmap=DEPTH_CMAP,
                          vmin=vmn, vmax=vmx, edgecolor='none', alpha=0.95)
    ax2.set_xlabel('X (km)', labelpad=10); ax2.set_ylabel('Y (km)', labelpad=10)
    ax2.set_zlabel('Depth (m, neg = down)', labelpad=10)
    ax2.set_title(f'MCMC posterior mean\n{mean_d.min():.0f}–{mean_d.max():.0f} m',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.view_init(elev=30, azim=225)
    # Lock both axes to same z-range for fair comparison
    zmin = min(-truth_depths.max(), -mean_d.max())
    zmax = max(-truth_depths.min(), -mean_d.min())
    ax1.set_zlim(zmin, zmax); ax2.set_zlim(zmin, zmax)

    fig.colorbar(s1, ax=[ax1, ax2], shrink=0.5, label='Depth (m, neg = down)')
    fig.suptitle(f'{BASIN} — 3D Basement Surface: Real vs MCMC',
                 fontsize=16, fontweight='bold')
    plt.savefig(f'{OUT_DIR}/depth_3d_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

print(f"\nDone. Plots in: {OUT_DIR}/")
