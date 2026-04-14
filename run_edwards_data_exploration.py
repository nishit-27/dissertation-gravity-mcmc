"""
Edwards AFB Data Exploration
=============================
Visualize gravity stations, wells, and proposed inversion grid
to guide grid design decisions before running MCMC.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import (
    load_gravity_data, load_basement_wells, load_basin_wells,
    load_usgs_depth_grid, convert_to_utm, prepare_edwards_data,
    subsample_gravity
)

# ============================================================
# 1. Load raw data (decimal degrees for map plotting)
# ============================================================
print("=" * 60)
print("Edwards AFB Data Exploration")
print("=" * 60)

data_dir = 'real_data/edwards_afb/'

grav = load_gravity_data(os.path.join(data_dir, 'gravity_data.csv'))
bwells = load_basement_wells(os.path.join(data_dir, 'basement_wells.csv'))
bawells = load_basin_wells(os.path.join(data_dir, 'basin_wells.csv'))
usgs = load_usgs_depth_grid(os.path.join(data_dir, 'depth_to_bedrock.csv'))

print(f"\nGravity stations: {len(grav['lat'])}")
print(f"  Lat: {grav['lat'].min():.4f} to {grav['lat'].max():.4f}")
print(f"  Lon: {grav['lon'].min():.4f} to {grav['lon'].max():.4f}")
print(f"  Isostatic: {grav['isostatic'].min():.1f} to {grav['isostatic'].max():.1f} mGal")

print(f"\nBasement wells: {len(bwells['lat'])}")
print(f"  Depth range: {bwells['depth_m'].min():.0f} to {bwells['depth_m'].max():.0f} m")

print(f"\nBasin wells: {len(bawells['lat'])}")
print(f"  Depth range: {bawells['total_depth_m'].min():.0f} to {bawells['total_depth_m'].max():.0f} m")

print(f"\nUSGS depth grid: {len(usgs['lat'])} points")
print(f"  Depth range: {usgs['depth_m'].min():.0f} to {usgs['depth_m'].max():.0f} m")

# ============================================================
# 2. Define study area
# ============================================================
# Focus on the area with good well coverage and gravity data
# Wells are concentrated around lat 34.85-35.08, lon -117.9 to -117.5
# Extend slightly beyond for padding
study_bounds = {
    'lon_min': -117.95,
    'lon_max': -117.50,
    'lat_min': 34.82,
    'lat_max': 35.10,
}

# Clip gravity stations to study area
mask_grav = (
    (grav['lon'] >= study_bounds['lon_min']) &
    (grav['lon'] <= study_bounds['lon_max']) &
    (grav['lat'] >= study_bounds['lat_min']) &
    (grav['lat'] <= study_bounds['lat_max'])
)
n_in_study = mask_grav.sum()
print(f"\nGravity stations in study area: {n_in_study} / {len(grav['lat'])}")

# Clip wells to study area
mask_bw = (
    (bwells['lon'] >= study_bounds['lon_min']) &
    (bwells['lon'] <= study_bounds['lon_max']) &
    (bwells['lat'] >= study_bounds['lat_min']) &
    (bwells['lat'] <= study_bounds['lat_max'])
)
n_bw_in = mask_bw.sum()
print(f"Basement wells in study area: {n_bw_in} / {len(bwells['lat'])}")

# Study area size in km
lat_range_km = (study_bounds['lat_max'] - study_bounds['lat_min']) * 111.0
lon_range_km = (study_bounds['lon_max'] - study_bounds['lon_min']) * 111.0 * np.cos(np.radians(34.96))
print(f"Study area: {lon_range_km:.1f} km (E-W) x {lat_range_km:.1f} km (N-S)")

# ============================================================
# 3. Proposed block grid
# ============================================================
nx, ny = 15, 12  # 15 blocks E-W, 12 blocks N-S
block_lon_edges = np.linspace(study_bounds['lon_min'], study_bounds['lon_max'], nx + 1)
block_lat_edges = np.linspace(study_bounds['lat_min'], study_bounds['lat_max'], ny + 1)

block_size_lon_km = (block_lon_edges[1] - block_lon_edges[0]) * 111.0 * np.cos(np.radians(34.96))
block_size_lat_km = (block_lat_edges[1] - block_lat_edges[0]) * 111.0
print(f"\nProposed grid: {nx} x {ny} = {nx*ny} blocks")
print(f"Block size: {block_size_lon_km:.1f} km (E-W) x {block_size_lat_km:.1f} km (N-S)")

# Count stations per block
station_counts = np.zeros((nx, ny))
grav_lon_in = grav['lon'][mask_grav]
grav_lat_in = grav['lat'][mask_grav]
for i in range(len(grav_lon_in)):
    ix = np.searchsorted(block_lon_edges, grav_lon_in[i]) - 1
    iy = np.searchsorted(block_lat_edges, grav_lat_in[i]) - 1
    if 0 <= ix < nx and 0 <= iy < ny:
        station_counts[ix, iy] += 1

print(f"\nStations per block: min={station_counts.min():.0f}, "
      f"max={station_counts.max():.0f}, mean={station_counts.mean():.1f}")
print(f"Blocks with 0 stations: {(station_counts == 0).sum()}")

# ============================================================
# 4. Create output directory
# ============================================================
out_dir = 'results/edwards_exploration/'
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# FIGURE 1: All gravity stations colored by isostatic residual
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

sc = ax.scatter(grav['lon'], grav['lat'],
                c=grav['isostatic'], cmap='RdYlBu_r', s=8, alpha=0.7,
                vmin=-45, vmax=-4, zorder=2)
plt.colorbar(sc, ax=ax, label='Isostatic Residual Gravity (mGal)', shrink=0.8)

# Study area rectangle
rect = plt.Rectangle(
    (study_bounds['lon_min'], study_bounds['lat_min']),
    study_bounds['lon_max'] - study_bounds['lon_min'],
    study_bounds['lat_max'] - study_bounds['lat_min'],
    linewidth=2, edgecolor='black', facecolor='none', linestyle='--', zorder=3,
    label='Study area'
)
ax.add_patch(rect)

# Wells
ax.scatter(bwells['lon'], bwells['lat'], c='magenta', s=30, marker='^',
           edgecolors='black', linewidths=0.5, zorder=4, label=f'Basement wells ({len(bwells["lat"])})')
ax.scatter(bawells['lon'], bawells['lat'], c='yellow', s=20, marker='s',
           edgecolors='black', linewidths=0.5, zorder=4, label=f'Basin wells ({len(bawells["lat"])})')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Edwards AFB — All Gravity Stations + Wells')
ax.legend(loc='upper left', fontsize=9)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig1_all_stations.png'), dpi=150)
print(f"\nSaved: {out_dir}fig1_all_stations.png")
plt.close(fig)

# ============================================================
# FIGURE 2: Study area with block grid overlay
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Only stations in study area
sc = ax.scatter(grav_lon_in, grav_lat_in,
                c=grav['isostatic'][mask_grav], cmap='RdYlBu_r', s=12, alpha=0.8,
                vmin=-45, vmax=-4, zorder=2)
plt.colorbar(sc, ax=ax, label='Isostatic Residual Gravity (mGal)', shrink=0.8)

# Block grid
for x in block_lon_edges:
    ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
for y in block_lat_edges:
    ax.axhline(y, color='gray', linewidth=0.5, alpha=0.5, zorder=1)

# Wells in study area
bw_mask = mask_bw
ax.scatter(bwells['lon'][bw_mask], bwells['lat'][bw_mask],
           c='magenta', s=50, marker='^', edgecolors='black', linewidths=0.5,
           zorder=4, label=f'Basement wells in study ({bw_mask.sum()})')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'Study Area with {nx}x{ny} Block Grid ({nx*ny} blocks)')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(study_bounds['lon_min'] - 0.02, study_bounds['lon_max'] + 0.02)
ax.set_ylim(study_bounds['lat_min'] - 0.01, study_bounds['lat_max'] + 0.01)
ax.set_aspect('equal')
ax.grid(False)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig2_study_area_grid.png'), dpi=150)
print(f"Saved: {out_dir}fig2_study_area_grid.png")
plt.close(fig)

# ============================================================
# FIGURE 3: Station density per block (heatmap)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# station_counts is (nx, ny) — plot as image with lon on x, lat on y
im = ax.pcolormesh(block_lon_edges, block_lat_edges, station_counts.T,
                   cmap='YlOrRd', shading='flat', zorder=1)
plt.colorbar(im, ax=ax, label='Number of gravity stations per block')

# Annotate each cell with count
for ix in range(nx):
    for iy in range(ny):
        cx = (block_lon_edges[ix] + block_lon_edges[ix + 1]) / 2
        cy = (block_lat_edges[iy] + block_lat_edges[iy + 1]) / 2
        count = int(station_counts[ix, iy])
        color = 'white' if count > 15 else 'black'
        ax.text(cx, cy, str(count), ha='center', va='center',
                fontsize=7, fontweight='bold', color=color, zorder=3)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'Gravity Station Density per Block ({nx}x{ny} grid)')
ax.set_aspect('equal')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig3_station_density.png'), dpi=150)
print(f"Saved: {out_dir}fig3_station_density.png")
plt.close(fig)

# ============================================================
# FIGURE 4: Basement well depths
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

sc = ax.scatter(bwells['lon'], bwells['lat'],
                c=bwells['depth_m'], cmap='viridis_r', s=60, marker='^',
                edgecolors='black', linewidths=0.5, zorder=4,
                vmin=0, vmax=650)
plt.colorbar(sc, ax=ax, label='Depth to Bedrock (m)', shrink=0.8)

# Study area
rect = plt.Rectangle(
    (study_bounds['lon_min'], study_bounds['lat_min']),
    study_bounds['lon_max'] - study_bounds['lon_min'],
    study_bounds['lat_max'] - study_bounds['lat_min'],
    linewidth=2, edgecolor='black', facecolor='none', linestyle='--', zorder=3
)
ax.add_patch(rect)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Basement Well Depths (meters below surface)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig4_well_depths.png'), dpi=150)
print(f"Saved: {out_dir}fig4_well_depths.png")
plt.close(fig)

# ============================================================
# FIGURE 5: USGS depth model
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Clip USGS grid to study area
mask_usgs = (
    (usgs['lon'] >= study_bounds['lon_min']) &
    (usgs['lon'] <= study_bounds['lon_max']) &
    (usgs['lat'] >= study_bounds['lat_min']) &
    (usgs['lat'] <= study_bounds['lat_max'])
)

sc = ax.scatter(usgs['lon'][mask_usgs], usgs['lat'][mask_usgs],
                c=usgs['depth_m'][mask_usgs], cmap='viridis_r',
                s=1, alpha=0.8, vmin=0, vmax=800, zorder=1)
plt.colorbar(sc, ax=ax, label='USGS Depth to Bedrock (m)', shrink=0.8)

# Overlay wells
ax.scatter(bwells['lon'][mask_bw], bwells['lat'][mask_bw],
           c='magenta', s=40, marker='^', edgecolors='black', linewidths=0.5,
           zorder=4, label='Basement wells')

# Grid
for x in block_lon_edges:
    ax.axvline(x, color='white', linewidth=0.3, alpha=0.5, zorder=2)
for y in block_lat_edges:
    ax.axhline(y, color='white', linewidth=0.3, alpha=0.5, zorder=2)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('USGS Depth to Bedrock Model (Langenheim et al., 2019)')
ax.legend(loc='upper left')
ax.set_xlim(study_bounds['lon_min'] - 0.02, study_bounds['lon_max'] + 0.02)
ax.set_ylim(study_bounds['lat_min'] - 0.01, study_bounds['lat_max'] + 0.01)
ax.set_aspect('equal')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig5_usgs_depth_model.png'), dpi=150)
print(f"Saved: {out_dir}fig5_usgs_depth_model.png")
plt.close(fig)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Study area: {lon_range_km:.1f} x {lat_range_km:.1f} km")
print(f"Grid: {nx} x {ny} = {nx*ny} blocks")
print(f"Block size: {block_size_lon_km:.1f} x {block_size_lat_km:.1f} km")
print(f"Gravity stations in study area: {n_in_study}")
print(f"Basement wells in study area: {n_bw_in}")
print(f"Blocks with 0 stations: {(station_counts == 0).sum()}")
print(f"Min/max stations per block: {station_counts.min():.0f} / {station_counts.max():.0f}")
print(f"\nFigures saved to: {out_dir}")
print("=" * 60)
