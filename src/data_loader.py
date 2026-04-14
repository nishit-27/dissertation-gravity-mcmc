"""
Data Loader for Edwards AFB Gravity Data (USGS OFR 2019-1128)

Loads and preprocesses four CSV data files from the Edwards Air Force
Base study area (Antelope Valley, California):
  1. Gravity station data (observed gravity, Bouguer anomalies)
  2. Basement wells (wells that reached bedrock)
  3. Basin wells (wells within sedimentary basin, may not reach bedrock)
  4. USGS depth-to-bedrock grid (model from Saltus & Jachens, 1995)

Converts all coordinates from geographic (lat/lon) to UTM Zone 11N,
shifts to local origin, and prepares everything for MCMC inversion.

References:
    Ponce, D.A. and Langenheim, V.E. (2019). Gravity and magnetic data
    of the Edwards Air Force Base region. USGS OFR 2019-1128.
"""

import csv
import numpy as np


# ---------------------------------------------------------------------------
# 1. Load gravity station data
# ---------------------------------------------------------------------------

def load_gravity_data(filepath):
    """
    Load gravity station data from CSV (no header, 14 columns).

    Columns:
        name, latd, latm, lond, lonm, elev_ft, obs_grav, faa, sba,
        itc, ttc, code, cba, iso

    Coordinate conversion:
        lat = latd + latm / 60.0
        lon = lond - lonm / 60.0   (lond is negative, lonm positive)

    Parameters
    ----------
    filepath : str
        Path to gravity_data.csv

    Returns
    -------
    data : dict with keys:
        'station_id' : list of str
        'lat'        : ndarray, decimal degrees
        'lon'        : ndarray, decimal degrees
        'elevation_m': ndarray, meters
        'cba'        : ndarray, complete Bouguer anomaly (mGal)
        'isostatic'  : ndarray, isostatic residual anomaly (mGal)
    """
    station_ids = []
    lats = []
    lons = []
    elevations = []
    cbas = []
    isos = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 14:
                continue
            try:
                name = row[0].strip()
                latd = float(row[1])
                latm = float(row[2])
                lond = float(row[3])
                lonm = float(row[4])
                elev_ft = float(row[5])
                cba = float(row[12])
                iso = float(row[13])
            except (ValueError, IndexError):
                # Skip malformed or header/separator lines
                continue

            lat = latd + latm / 60.0
            lon = lond - lonm / 60.0

            station_ids.append(name)
            lats.append(lat)
            lons.append(lon)
            elevations.append(elev_ft * 0.3048)
            cbas.append(cba)
            isos.append(iso)

    return {
        'station_id': station_ids,
        'lat': np.array(lats),
        'lon': np.array(lons),
        'elevation_m': np.array(elevations),
        'cba': np.array(cbas),
        'isostatic': np.array(isos),
    }


# ---------------------------------------------------------------------------
# 2. Load basement wells
# ---------------------------------------------------------------------------

def load_basement_wells(filepath):
    """
    Load wells that reached bedrock (basement wells).

    CSV: no header, 5 columns:
        Well_ID, latitude, longitude, well_depth_to_bedrock_ft,
        model_depth_bedrock_ft

    Parameters
    ----------
    filepath : str
        Path to basement_wells.csv

    Returns
    -------
    data : dict with keys:
        'well_id'       : list of str
        'lat'           : ndarray, decimal degrees
        'lon'           : ndarray, decimal degrees
        'depth_m'       : ndarray, measured well depth to bedrock (meters)
        'model_depth_m' : ndarray, model-predicted depth (meters)
    """
    well_ids = []
    lats = []
    lons = []
    depths = []
    model_depths = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                well_id = row[0].strip()
                lat = float(row[1])
                lon = float(row[2])
                depth_ft = float(row[3])
                model_ft = float(row[4])
            except (ValueError, IndexError):
                continue

            well_ids.append(well_id)
            lats.append(lat)
            lons.append(lon)
            depths.append(depth_ft * 0.3048)
            model_depths.append(model_ft * 0.3048)

    return {
        'well_id': well_ids,
        'lat': np.array(lats),
        'lon': np.array(lons),
        'depth_m': np.array(depths),
        'model_depth_m': np.array(model_depths),
    }


# ---------------------------------------------------------------------------
# 3. Load basin wells (note: lon and lat columns are SWAPPED)
# ---------------------------------------------------------------------------

def load_basin_wells(filepath):
    """
    Load wells within the sedimentary basin (may not reach bedrock).

    CSV: no header, 5 columns:
        Well_ID, **longitude**, **latitude**, total_depth_ft,
        model_depth_bedrock_ft

    NOTE: longitude and latitude are SWAPPED compared to basement_wells.

    Parameters
    ----------
    filepath : str
        Path to basin_wells.csv

    Returns
    -------
    data : dict with keys:
        'well_id'        : list of str
        'lat'            : ndarray, decimal degrees
        'lon'            : ndarray, decimal degrees
        'total_depth_m'  : ndarray, total well depth (meters)
        'model_depth_m'  : ndarray, model-predicted bedrock depth (meters)
    """
    well_ids = []
    lats = []
    lons = []
    total_depths = []
    model_depths = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                well_id = row[0].strip()
                lon = float(row[1])   # longitude comes first!
                lat = float(row[2])   # latitude comes second!
                total_depth_ft = float(row[3])
                model_ft = float(row[4])
            except (ValueError, IndexError):
                continue

            well_ids.append(well_id)
            lats.append(lat)
            lons.append(lon)
            total_depths.append(total_depth_ft * 0.3048)
            model_depths.append(model_ft * 0.3048)

    return {
        'well_id': well_ids,
        'lat': np.array(lats),
        'lon': np.array(lons),
        'total_depth_m': np.array(total_depths),
        'model_depth_m': np.array(model_depths),
    }


# ---------------------------------------------------------------------------
# 4. Load USGS depth-to-bedrock grid
# ---------------------------------------------------------------------------

def load_usgs_depth_grid(filepath):
    """
    Load USGS depth-to-bedrock grid (Saltus & Jachens model).

    CSV: no header, 3 columns:
        longitude, latitude, depth_to_bedrock_ft

    Depths are NEGATIVE (below surface). We take abs() and convert to meters.

    Parameters
    ----------
    filepath : str
        Path to depth_to_bedrock.csv

    Returns
    -------
    data : dict with keys:
        'lon'     : ndarray, decimal degrees
        'lat'     : ndarray, decimal degrees
        'depth_m' : ndarray, depth to bedrock (meters, positive)
    """
    lons = []
    lats = []
    depths = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                lon = float(row[0])
                lat = float(row[1])
                depth_ft = float(row[2])
            except (ValueError, IndexError):
                continue

            lons.append(lon)
            lats.append(lat)
            depths.append(abs(depth_ft) * 0.3048)

    return {
        'lon': np.array(lons),
        'lat': np.array(lats),
        'depth_m': np.array(depths),
    }


# ---------------------------------------------------------------------------
# 5. Geographic to UTM coordinate conversion
# ---------------------------------------------------------------------------

def convert_to_utm(lons, lats, zone=11):
    """
    Convert geographic coordinates (lon, lat) to UTM easting/northing.

    Uses pyproj to transform from WGS84 (EPSG:4326) to UTM Zone N
    (default: Zone 11N, EPSG:32611 for Edwards AFB / southern California).

    Parameters
    ----------
    lons : array-like
        Longitudes in decimal degrees
    lats : array-like
        Latitudes in decimal degrees
    zone : int
        UTM zone number (default 11 for Edwards AFB)

    Returns
    -------
    easting : ndarray
        UTM easting in meters
    northing : ndarray
        UTM northing in meters
    """
    from pyproj import Transformer

    epsg_utm = 32600 + zone  # e.g., 32611 for Zone 11N
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True
    )

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    easting, northing = transformer.transform(lons, lats)

    return np.asarray(easting), np.asarray(northing)


# ---------------------------------------------------------------------------
# 6. Subsample scattered gravity to regular grid spacing
# ---------------------------------------------------------------------------

def subsample_gravity(obs_x, obs_y, gravity, spacing_m):
    """
    Bin scattered gravity stations into grid cells and take median values.

    For each cell that contains at least one station, compute the median
    gravity and the centroid of station positions. This reduces the data
    to an approximately regular spacing for the inversion.

    Parameters
    ----------
    obs_x : ndarray (M,)
        Station x-coordinates (meters, e.g., UTM easting)
    obs_y : ndarray (M,)
        Station y-coordinates (meters, e.g., UTM northing)
    gravity : ndarray (M,)
        Gravity values at each station (mGal)
    spacing_m : float
        Grid cell size in meters

    Returns
    -------
    new_x : ndarray (K,)
        Centroid x-coordinates of occupied cells
    new_y : ndarray (K,)
        Centroid y-coordinates of occupied cells
    gravity_sub : ndarray (K,)
        Median gravity in each occupied cell
    n_per_cell : ndarray (K,)
        Number of stations in each cell
    """
    obs_x = np.asarray(obs_x, dtype=float)
    obs_y = np.asarray(obs_y, dtype=float)
    gravity = np.asarray(gravity, dtype=float)

    # Compute bin indices for each station
    ix = np.floor(obs_x / spacing_m).astype(int)
    iy = np.floor(obs_y / spacing_m).astype(int)

    # Group by (ix, iy)
    cells = {}
    for k in range(len(obs_x)):
        key = (ix[k], iy[k])
        if key not in cells:
            cells[key] = []
        cells[key].append(k)

    new_x = []
    new_y = []
    gravity_sub = []
    n_per_cell = []

    for key, indices in cells.items():
        idx = np.array(indices)
        new_x.append(np.mean(obs_x[idx]))
        new_y.append(np.mean(obs_y[idx]))
        gravity_sub.append(np.median(gravity[idx]))
        n_per_cell.append(len(idx))

    return (np.array(new_x), np.array(new_y),
            np.array(gravity_sub), np.array(n_per_cell))


# ---------------------------------------------------------------------------
# 7. Assign wells to inversion blocks
# ---------------------------------------------------------------------------

def assign_wells_to_blocks(well_x, well_y, well_depths, block_x_edges,
                           block_y_edges, well_ids=None):
    """
    Assign wells to 2D (ix, iy) grid blocks defined by block edges.

    Uses np.digitize to find which block each well falls into. If
    multiple wells land in the same block, their depths are averaged.

    Parameters
    ----------
    well_x : ndarray (W,)
        Well x-coordinates (meters, UTM easting)
    well_y : ndarray (W,)
        Well y-coordinates (meters, UTM northing)
    well_depths : ndarray (W,)
        Well depths to bedrock (meters)
    block_x_edges : ndarray (Nx+1,)
        Block boundaries in x (meters)
    block_y_edges : ndarray (Ny+1,)
        Block boundaries in y (meters)
    well_ids : list of str or None
        Well identifiers for traceability

    Returns
    -------
    constraints : dict
        {(ix, iy): depth_m} where depth_m is the average depth of wells
        in that block. Suitable for the borehole_constraints parameter
        in the MCMC inversion.
    assignment_log : list of tuples
        [(well_id, ix, iy, depth_m), ...] for traceability
    """
    well_x = np.asarray(well_x, dtype=float)
    well_y = np.asarray(well_y, dtype=float)
    well_depths = np.asarray(well_depths, dtype=float)

    if well_ids is None:
        well_ids = [f"W{i}" for i in range(len(well_x))]

    # np.digitize returns 1-based indices; subtract 1 for 0-based
    ix_all = np.digitize(well_x, block_x_edges) - 1
    iy_all = np.digitize(well_y, block_y_edges) - 1

    n_x = len(block_x_edges) - 1
    n_y = len(block_y_edges) - 1

    # Collect wells per block
    block_wells = {}     # (ix, iy) -> list of depths
    assignment_log = []

    for k in range(len(well_x)):
        ix = int(ix_all[k])
        iy = int(iy_all[k])

        # Skip wells outside the grid
        if ix < 0 or ix >= n_x or iy < 0 or iy >= n_y:
            continue

        key = (ix, iy)
        if key not in block_wells:
            block_wells[key] = []
        block_wells[key].append(well_depths[k])

        assignment_log.append((well_ids[k], ix, iy, well_depths[k]))

    # Average depths for blocks with multiple wells
    constraints = {}
    for key, depths_list in block_wells.items():
        constraints[key] = float(np.mean(depths_list))

    return constraints, assignment_log


# ---------------------------------------------------------------------------
# 8. Master preparation function
# ---------------------------------------------------------------------------

def prepare_edwards_data(data_dir, study_bounds=None):
    """
    Load, convert, and prepare all Edwards AFB data for MCMC inversion.

    Steps:
        1. Load all 4 CSV files
        2. Convert all coordinates to UTM Zone 11N (meters)
        3. Optionally clip to geographic study bounds
        4. Shift coordinates so the SW corner = (0, 0)
        5. Apply regional gravity correction: subtract mean isostatic
           gravity at shallow basement wells (depth < 10 m)
        6. Return comprehensive dict with everything

    Parameters
    ----------
    data_dir : str
        Path to directory containing the 4 CSV files:
        gravity_data.csv, basement_wells.csv, basin_wells.csv,
        depth_to_bedrock.csv
    study_bounds : dict or None
        If provided, clip all data to these geographic bounds:
        {'lon_min': float, 'lon_max': float,
         'lat_min': float, 'lat_max': float}

    Returns
    -------
    result : dict with keys:
        'gravity' : dict — gravity station data with UTM coords
        'basement_wells' : dict — basement wells with UTM coords
        'basin_wells' : dict — basin wells with UTM coords
        'depth_grid' : dict — USGS depth grid with UTM coords
        'origin_easting' : float — UTM easting of SW corner (for un-shifting)
        'origin_northing' : float — UTM northing of SW corner
        'regional_correction' : float — mGal value subtracted from isostatic
        'study_bounds' : dict or None — the bounds used for clipping
    """
    import os

    # --- 1. Load all files ---
    grav = load_gravity_data(os.path.join(data_dir, 'gravity_data.csv'))
    bwells = load_basement_wells(os.path.join(data_dir, 'basement_wells.csv'))
    bawells = load_basin_wells(os.path.join(data_dir, 'basin_wells.csv'))
    dgrid = load_usgs_depth_grid(os.path.join(data_dir, 'depth_to_bedrock.csv'))

    print(f"Loaded gravity stations: {len(grav['lat'])}")
    print(f"Loaded basement wells:   {len(bwells['lat'])}")
    print(f"Loaded basin wells:      {len(bawells['lat'])}")
    print(f"Loaded depth grid pts:   {len(dgrid['lat'])}")

    # --- 2. Clip to study bounds (geographic) BEFORE UTM conversion ---
    if study_bounds is not None:
        lo_min = study_bounds['lon_min']
        lo_max = study_bounds['lon_max']
        la_min = study_bounds['lat_min']
        la_max = study_bounds['lat_max']

        # Gravity stations
        mask = ((grav['lon'] >= lo_min) & (grav['lon'] <= lo_max) &
                (grav['lat'] >= la_min) & (grav['lat'] <= la_max))
        grav = {
            'station_id': [grav['station_id'][i] for i in range(len(mask)) if mask[i]],
            'lat': grav['lat'][mask],
            'lon': grav['lon'][mask],
            'elevation_m': grav['elevation_m'][mask],
            'cba': grav['cba'][mask],
            'isostatic': grav['isostatic'][mask],
        }

        # Basement wells
        mask = ((bwells['lon'] >= lo_min) & (bwells['lon'] <= lo_max) &
                (bwells['lat'] >= la_min) & (bwells['lat'] <= la_max))
        bwells = {
            'well_id': [bwells['well_id'][i] for i in range(len(mask)) if mask[i]],
            'lat': bwells['lat'][mask],
            'lon': bwells['lon'][mask],
            'depth_m': bwells['depth_m'][mask],
            'model_depth_m': bwells['model_depth_m'][mask],
        }

        # Basin wells
        mask = ((bawells['lon'] >= lo_min) & (bawells['lon'] <= lo_max) &
                (bawells['lat'] >= la_min) & (bawells['lat'] <= la_max))
        bawells = {
            'well_id': [bawells['well_id'][i] for i in range(len(mask)) if mask[i]],
            'lat': bawells['lat'][mask],
            'lon': bawells['lon'][mask],
            'total_depth_m': bawells['total_depth_m'][mask],
            'model_depth_m': bawells['model_depth_m'][mask],
        }

        # Depth grid
        mask = ((dgrid['lon'] >= lo_min) & (dgrid['lon'] <= lo_max) &
                (dgrid['lat'] >= la_min) & (dgrid['lat'] <= la_max))
        dgrid = {
            'lon': dgrid['lon'][mask],
            'lat': dgrid['lat'][mask],
            'depth_m': dgrid['depth_m'][mask],
        }

        print(f"\nAfter clipping to study bounds:")
        print(f"  Gravity stations: {len(grav['lat'])}")
        print(f"  Basement wells:   {len(bwells['lat'])}")
        print(f"  Basin wells:      {len(bawells['lat'])}")
        print(f"  Depth grid pts:   {len(dgrid['lat'])}")

    # --- 3. Convert all coordinates to UTM Zone 11N ---
    grav['x'], grav['y'] = convert_to_utm(grav['lon'], grav['lat'])
    bwells['x'], bwells['y'] = convert_to_utm(bwells['lon'], bwells['lat'])
    bawells['x'], bawells['y'] = convert_to_utm(bawells['lon'], bawells['lat'])
    dgrid['x'], dgrid['y'] = convert_to_utm(dgrid['lon'], dgrid['lat'])

    # --- 4. Shift so SW corner = (0, 0) ---
    all_x = np.concatenate([grav['x'], bwells['x'], bawells['x'], dgrid['x']])
    all_y = np.concatenate([grav['y'], bwells['y'], bawells['y'], dgrid['y']])

    origin_easting = float(np.min(all_x))
    origin_northing = float(np.min(all_y))

    grav['x'] -= origin_easting
    grav['y'] -= origin_northing
    bwells['x'] -= origin_easting
    bwells['y'] -= origin_northing
    bawells['x'] -= origin_easting
    bawells['y'] -= origin_northing
    dgrid['x'] -= origin_easting
    dgrid['y'] -= origin_northing

    extent_x = float(np.max(all_x) - origin_easting)
    extent_y = float(np.max(all_y) - origin_northing)
    print(f"\nStudy area extent: {extent_x/1000:.1f} x {extent_y/1000:.1f} km")

    # --- 5. Regional gravity correction ---
    # Subtract mean isostatic gravity at shallow basement wells (depth < 10 m).
    # These are essentially "basement outcrops" where the gravity signal should
    # be near zero for an ideal basin-only anomaly.
    shallow_mask = bwells['depth_m'] < 10.0
    n_shallow = int(np.sum(shallow_mask))

    if n_shallow > 0:
        regional_correction = float(np.mean(grav['isostatic']))
        # Match shallow wells to nearby gravity stations and compute correction
        # Use nearest gravity station to each shallow well
        from scipy.spatial import cKDTree
        grav_tree = cKDTree(np.column_stack([grav['x'], grav['y']]))

        shallow_xy = np.column_stack([
            bwells['x'][shallow_mask],
            bwells['y'][shallow_mask],
        ])
        _, nearest_idx = grav_tree.query(shallow_xy)

        regional_correction = float(np.mean(grav['isostatic'][nearest_idx]))

        print(f"\nRegional correction: {regional_correction:.2f} mGal")
        print(f"  (mean isostatic at {n_shallow} shallow basement wells, "
              f"depth < 10 m)")
    else:
        regional_correction = 0.0
        print("\nNo shallow basement wells found — skipping regional correction")

    grav['isostatic_corrected'] = grav['isostatic'] - regional_correction
    grav['cba_corrected'] = grav['cba'] - regional_correction

    # --- 6. Return everything ---
    return {
        'gravity': grav,
        'basement_wells': bwells,
        'basin_wells': bawells,
        'depth_grid': dgrid,
        'origin_easting': origin_easting,
        'origin_northing': origin_northing,
        'regional_correction': regional_correction,
        'study_bounds': study_bounds,
    }
