"""Plot: which stations were used (subsampled) vs. all stations on the grid."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = 'real_data/chintalpudi'
OUT_DIR = 'results/exp_chintalpudi_10k_100stn'
STRIDE = 5

xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))
yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))

all_x = xg.flatten() / 1000
all_y = yg.flatten() / 1000
sel_x = xg[::STRIDE, ::STRIDE].flatten() / 1000
sel_y = yg[::STRIDE, ::STRIDE].flatten() / 1000

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
im = ax.pcolormesh(xg/1000, yg/1000, gv, cmap='RdBu_r', shading='auto', alpha=0.6)
plt.colorbar(im, ax=ax, label='Bouguer anomaly (mGal)')
ax.scatter(all_x, all_y, c='lightgrey', s=3, label=f'All {len(all_x)} stations', zorder=2)
ax.scatter(sel_x, sel_y, c='red', s=40, marker='s',
           edgecolors='black', linewidths=0.6,
           label=f'Used: {len(sel_x)} (stride={STRIDE})', zorder=3)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title(f'Chintalpudi — Station selection for MCMC\n'
             f'{len(sel_x)} of {len(all_x)} stations used ({100*len(sel_x)/len(all_x):.0f}%)')
ax.legend(loc='upper right')
ax.set_aspect('equal')

ax = axes[1]
ax.scatter(all_x, all_y, c='lightgrey', s=4, label=f'Not used ({len(all_x)-len(sel_x)})')
ax.scatter(sel_x, sel_y, c='red', s=50, marker='s',
           edgecolors='black', linewidths=0.8,
           label=f'Used in inversion ({len(sel_x)})')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title('Station map (no background)')
ax.legend(loc='upper right')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

fig.suptitle('Chintalpudi Sub-Basin: Subsampled Station Coverage', fontsize=13)
fig.tight_layout()
out = os.path.join(OUT_DIR, 'station_selection_map.png')
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")
