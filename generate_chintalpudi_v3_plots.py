"""Generate the full Exp-7 plot suite for Chintalpudi v3 results."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.forward_model import compute_gravity_for_basin
from src.utils import make_density_func

RES_DIR = 'results/exp_chintalpudi_v3_fixedlam_borehole'
d = np.load(os.path.join(RES_DIR, 'results_data.npz'))
mean_d = d['mean_depths']; std_d = d['std_depths']
truth  = d['truth_blocks']
ci_lo, ci_hi = d['ci_5'], d['ci_95']
bx, by = d['block_x_edges'], d['block_y_edges']
obs_x, obs_y, g_obs = d['obs_x'], d['obs_y'], d['obs_gravity']
drho_0 = float(d['drho_0']); lam = float(d['lam'])
ib, jb = int(d['borehole_block'][0]), int(d['borehole_block'][1])
bore_z = float(d['borehole_depth'])
NX, NY = mean_d.shape

xc = 0.5*(bx[:-1] + bx[1:]) / 1000   # block centers km
yc = 0.5*(by[:-1] + by[1:]) / 1000
Xc, Yc = np.meshgrid(xc, yc, indexing='ij')

# -----------------------------------------------------------------
# 1. 3D surface — recovered vs truth side-by-side (inverted z)
# -----------------------------------------------------------------
fig = plt.figure(figsize=(16, 7))
for k, (data, title) in enumerate([(mean_d, 'Recovered (MCMC mean)'),
                                   (truth,  'Published truth')]):
    ax = fig.add_subplot(1, 2, k+1, projection='3d')
    surf = ax.plot_surface(Xc, Yc, -data, cmap='viridis_r',
                           edgecolor='k', linewidth=0.15, alpha=0.95)
    ax.scatter([xc[ib]], [yc[jb]], [-bore_z], color='yellow',
               edgecolors='k', s=160, marker='*', label='ONGC borehole')
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_zlabel('Depth (m, down)')
    ax.set_title(f'{title}\n{data.min():.0f}–{data.max():.0f} m')
    fig.colorbar(surf, ax=ax, shrink=0.55, label='Depth (m)')
    ax.view_init(elev=28, azim=-120)
    ax.legend()
fig.suptitle('Chintalpudi — 3D Basement Surface Comparison', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(RES_DIR, 'plot_3d_comparison.png'), dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 2. 3D uncertainty surface (std on top of depth surface)
# -----------------------------------------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
norm_std = (std_d - std_d.min()) / (std_d.max() - std_d.min() + 1e-9)
colors = plt.cm.hot_r(norm_std)
ax.plot_surface(Xc, Yc, -mean_d, facecolors=colors, edgecolor='k',
                linewidth=0.15, shade=False, alpha=0.95)
ax.scatter([xc[ib]], [yc[jb]], [-bore_z], color='yellow',
           edgecolors='k', s=180, marker='*', label='borehole')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Depth (m)')
ax.set_title(f'Basement with Posterior Uncertainty (hot_r scale)\n'
             f'std {std_d.min():.0f}–{std_d.max():.0f} m, mean {std_d.mean():.0f} m')
ax.view_init(elev=28, azim=-120); ax.legend()
m = plt.cm.ScalarMappable(cmap='hot_r',
        norm=plt.Normalize(vmin=std_d.min(), vmax=std_d.max()))
m.set_array([])
fig.colorbar(m, ax=ax, shrink=0.65, label='Std (m)')
fig.tight_layout()
fig.savefig(os.path.join(RES_DIR, 'plot_3d_uncertainty.png'), dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 3. Cross-sections (central X row and Y col)
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
j0 = jb
ax = axes[0]
ax.plot(xc, mean_d[:, j0], 'b-', lw=2, label='MCMC mean')
ax.fill_between(xc, ci_lo[:, j0], ci_hi[:, j0], alpha=0.3, color='b', label='90% CI')
ax.plot(xc, truth[:, j0], 'k--', lw=2, label='Truth')
ax.axvline(xc[ib], color='gold', lw=2, label=f'Borehole ({bore_z:.0f} m)')
ax.invert_yaxis(); ax.set_xlabel('X (km)'); ax.set_ylabel('Depth (m)')
ax.set_title(f'E–W cross-section at Y = {yc[j0]:.1f} km (borehole row)')
ax.legend(); ax.grid(alpha=0.3)

i0 = ib
ax = axes[1]
ax.plot(yc, mean_d[i0, :], 'b-', lw=2, label='MCMC mean')
ax.fill_between(yc, ci_lo[i0, :], ci_hi[i0, :], alpha=0.3, color='b', label='90% CI')
ax.plot(yc, truth[i0, :], 'k--', lw=2, label='Truth')
ax.axvline(yc[jb], color='gold', lw=2, label=f'Borehole ({bore_z:.0f} m)')
ax.invert_yaxis(); ax.set_xlabel('Y (km)'); ax.set_ylabel('Depth (m)')
ax.set_title(f'N–S cross-section at X = {xc[i0]:.1f} km (borehole col)')
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle('Chintalpudi v3 — Cross-Sections with Uncertainty')
fig.tight_layout()
fig.savefig(os.path.join(RES_DIR, 'plot_cross_sections.png'), dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 4. Gravity fit — observed vs predicted
# -----------------------------------------------------------------
density_func = make_density_func('exponential', drho_0=drho_0, lam=lam)
g_pred = compute_gravity_for_basin(obs_x, obs_y, bx, by, mean_d,
                                   density_func, n_sublayers=10)
residual = g_obs - g_pred
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc = axes[0].scatter(obs_x/1000, obs_y/1000, c=g_obs, cmap='RdBu_r', s=40,
                     edgecolors='k', linewidths=0.3)
plt.colorbar(sc, ax=axes[0], label='mGal')
axes[0].set_title(f'Observed ({g_obs.min():.1f} to {g_obs.max():.1f} mGal)')
axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)'); axes[0].set_aspect('equal')

sc = axes[1].scatter(obs_x/1000, obs_y/1000, c=g_pred, cmap='RdBu_r', s=40,
                     edgecolors='k', linewidths=0.3,
                     vmin=g_obs.min(), vmax=g_obs.max())
plt.colorbar(sc, ax=axes[1], label='mGal')
axes[1].set_title(f'Predicted (from MCMC mean)')
axes[1].set_xlabel('X (km)'); axes[1].set_aspect('equal')

vm = float(np.abs(residual).max())
sc = axes[2].scatter(obs_x/1000, obs_y/1000, c=residual, cmap='RdBu_r', s=40,
                     edgecolors='k', linewidths=0.3, vmin=-vm, vmax=vm)
plt.colorbar(sc, ax=axes[2], label='mGal')
rms_g = np.sqrt(np.mean(residual**2))
axes[2].set_title(f'Residual (RMS {rms_g:.2f} mGal)')
axes[2].set_xlabel('X (km)'); axes[2].set_aspect('equal')
fig.suptitle('Chintalpudi v3 — Gravity Fit')
fig.tight_layout()
fig.savefig(os.path.join(RES_DIR, 'plot_gravity_fit.png'), dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 5. Truth vs recovered scatter + histograms
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.errorbar(truth.flatten(), mean_d.flatten(), yerr=std_d.flatten(),
            fmt='o', alpha=0.6, ms=4, capsize=2, color='steelblue')
lim = [0, max(truth.max(), mean_d.max()) * 1.05]
ax.plot(lim, lim, 'k--', label='y = x')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Truth depth (m)'); ax.set_ylabel('Recovered depth (m)')
rms_d = float(np.sqrt(np.mean((mean_d - truth)**2)))
ax.set_title(f'Recovered vs Truth (RMS {rms_d:.0f} m)')
ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')

ax = axes[1]
err = (mean_d - truth).flatten()
ax.hist(err, bins=25, color='indianred', edgecolor='k', alpha=0.85)
ax.axvline(0, color='k', ls='--')
ax.axvline(err.mean(), color='blue', ls='-', label=f'bias {err.mean():+.0f}')
ax.set_xlabel('Error (m)'); ax.set_ylabel('Count')
ax.set_title(f'Error distribution (std {err.std():.0f} m)')
ax.legend(); ax.grid(alpha=0.3)
fig.suptitle('Chintalpudi v3 — Depth Recovery Accuracy')
fig.tight_layout()
fig.savefig(os.path.join(RES_DIR, 'plot_accuracy.png'), dpi=150)
plt.close(fig)

# -----------------------------------------------------------------
# 6. Posterior histograms for 4 representative blocks
# -----------------------------------------------------------------
misfits = d['all_misfits']
# (can't easily get per-block chain without reloading full result; skip chain hist)
# Instead do a diagnostic: misfit trace
fig, ax = plt.subplots(figsize=(11, 4))
ax.semilogy(misfits, lw=0.3, alpha=0.7)
ax.axvline(len(misfits)//2, color='r', ls='--', label='burn-in')
ax.set_xlabel('Iteration'); ax.set_ylabel('Misfit')
ax.set_title(f'MCMC Misfit Trace — Chintalpudi v3')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RES_DIR, 'plot_diagnostics.png'), dpi=150)
plt.close(fig)

print("Generated in", RES_DIR)
for f in sorted(os.listdir(RES_DIR)):
    print(" -", f)
