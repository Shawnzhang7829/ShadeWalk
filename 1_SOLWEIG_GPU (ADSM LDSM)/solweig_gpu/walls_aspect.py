#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
import os
import numpy as np
from osgeo import gdal
from scipy.ndimage import rotate
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import gc
gdal.UseExceptions()

# Wall height threshold
walllimit = 3.0

def findwalls(dem_array, walllimit):
    """
    Identify walls in a Digital Surface Model (DSM) based on height threshold.
    
    Walls are detected by comparing each cell to its immediate neighbors.
    A wall exists where the elevation difference exceeds the threshold.
    
    Args:
        dem_array (np.ndarray): 2D array of elevation values (DSM)
        walllimit (float): Minimum height difference (m) to be considered a wall
    
    Returns:
        np.ndarray: 2D array of wall heights. Zero where no wall exists.
    """
    col, row = dem_array.shape
    walls = np.zeros((col, row))
    domain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            dom = dem_array[j - 1:j + 2, i - 1:i + 2]
            walls[j, i] = np.max(dom[domain == 1])

    walls = walls - dem_array
    walls[walls < walllimit] = 0

    walls[:, 0] = 0
    walls[:, -1] = 0
    walls[0, :] = 0
    walls[-1, :] = 0

    return walls

def cart2pol(x, y, units='deg'):
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        x (np.ndarray or float): X coordinate(s)
        y (np.ndarray or float): Y coordinate(s)
        units (str): Output angle units ('deg' or 'rad'). Default: 'deg'
    
    Returns:
        tuple: (theta, radius) where theta is angle and radius is distance
    """
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if units in ['deg', 'degs']:
        theta = theta * 180 / np.pi
    return theta, radius

def get_ders(dsm, scale):
    """
    Calculate slope derivatives (aspect and gradient) from DSM.
    
    Args:
        dsm (np.ndarray): Digital Surface Model array
        scale (float): Pixel size in meters
    
    Returns:
        tuple: (aspect, gradient) where:
            - aspect: slope orientation in radians
            - gradient: slope magnitude
    """
    dx = 1 / scale
    fy, fx = np.gradient(dsm, dx, dx)
    asp, grad = cart2pol(fy, fx, 'rad')
    grad = np.arctan(grad)
    asp = -asp
    asp[asp < 0] += 2 * np.pi
    return grad, asp

def filter1Goodwin_as_aspect_v3(walls, scale, a):
    """
    Calculate wall aspect (orientation) using directional filtering.
    
    This function determines the orientation of walls by rotating a directional
    filter and finding the direction with maximum wall presence.
    
    Args:
        walls (np.ndarray): Binary array indicating wall locations
        scale (float): Pixel size in meters
        a (np.ndarray): Aspect array from DSM derivatives
    
    Returns:
        np.ndarray: Wall aspect in degrees [0-360], where 0=North, 90=East, 180=South, 270=West
    """
    row, col = a.shape
    filtersize = int(np.floor((scale + 1e-10) * 9))
    if filtersize <= 2:
        filtersize = 3
    elif filtersize != 9 and filtersize % 2 == 0:
        filtersize += 1

    n = filtersize - 1
    filthalveceil = int(np.ceil(filtersize / 2.))
    filthalvefloor = int(np.floor(filtersize / 2.))

    filtmatrix = np.zeros((filtersize, filtersize))
    buildfilt = np.zeros((filtersize, filtersize))
    filtmatrix[:, filthalveceil - 1] = 1
    buildfilt[filthalveceil - 1, :filthalvefloor] = 1
    buildfilt[filthalveceil - 1, filthalveceil:] = 2

    y = np.zeros((row, col))
    z = np.zeros((row, col))
    x = np.zeros((row, col))
    walls = (walls > 0).astype(np.uint8)

    for h in range(0, 180):
        filtmatrix1 = np.round(rotate(filtmatrix, h, order=1, reshape=False, mode='nearest'))
        filtmatrixbuild = np.round(rotate(buildfilt, h, order=0, reshape=False, mode='nearest'))
        index = 270 - h

        if h in [150, 30]:
            filtmatrixbuild[:, n] = 0
        if index == 225:
            filtmatrix1[0, 0] = filtmatrix1[n, n] = 1
        if index == 135:
            filtmatrix1[0, n] = filtmatrix1[n, 0] = 1

        for i in range(filthalveceil - 1, row - filthalveceil - 1):
            for j in range(filthalveceil - 1, col - filthalveceil - 1):
                if walls[i, j] == 1:
                    wallscut = walls[i - filthalvefloor:i + filthalvefloor + 1,
                                     j - filthalvefloor:j + filthalvefloor + 1] * filtmatrix1
                    dsmcut = a[i - filthalvefloor:i + filthalvefloor + 1,
                               j - filthalvefloor:j + filthalvefloor + 1]
                    if z[i, j] < wallscut.sum():
                        z[i, j] = wallscut.sum()
                        x[i, j] = 1 if np.sum(dsmcut[filtmatrixbuild == 1]) > np.sum(dsmcut[filtmatrixbuild == 2]) else 2
                        y[i, j] = index

    y[x == 1] -= 180
    y[y < 0] += 360

    grad, asp = get_ders(a, scale)
    y += ((walls == 1) & (y == 0)) * (asp / (math.pi / 180.))

    return y

def _coverage_fraction(arr):
    """
    Because all tiles have no blank pixels, coverage is:
    count(arr > 0) / total_pixels
    """
    if arr is None:
        return 0.0
    total = arr.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(arr > 0)) / float(total)


def _is_sparse_tile(building_dsm_path, tree_path, min_building_fraction=0.01, min_tree_fraction=0.01):
    """
    A tile is considered sparse if BOTH:
        building_fraction < min_building_fraction
        tree_fraction < min_tree_fraction
    """
    dsm_ds = gdal.Open(building_dsm_path)
    if dsm_ds is None:
        raise FileNotFoundError(f"Could not open Building_DSM tile: {building_dsm_path}")
    dsm_arr = dsm_ds.GetRasterBand(1).ReadAsArray()
    dsm_ds = None

    tree_ds = gdal.Open(tree_path)
    if tree_ds is None:
        raise FileNotFoundError(f"Could not open Trees tile: {tree_path}")
    tree_arr = tree_ds.GetRasterBand(1).ReadAsArray()
    tree_ds = None

    building_fraction = _coverage_fraction(dsm_arr)
    tree_fraction = _coverage_fraction(tree_arr)

    sparse = (building_fraction < min_building_fraction) and (tree_fraction < min_tree_fraction)
    return sparse, building_fraction, tree_fraction

def process_file_parallel(args):
    """
    Process one Building_DSM tile to calculate walls and aspect.
    """
    (
        filename,
        building_dsm_folder_path,
        tree_folder_path,
        wall_output_path,
        aspect_output_path,
        skip_existing,
        skip_sparse_tiles,
        min_building_fraction,
        min_tree_fraction
    ) = args

    building_dsm_path = os.path.join(building_dsm_folder_path, filename)

    suffix = filename[13:-4]
    wall_name = f"walls_{suffix}.tif"
    aspect_name = f"aspect_{suffix}.tif"
    wall_path = os.path.join(wall_output_path, wall_name)
    aspect_path = os.path.join(aspect_output_path, aspect_name)

    if skip_existing and os.path.exists(wall_path) and os.path.exists(aspect_path):
        print(f"Skipping existing wall/aspect for {filename}")
        return filename

    dataset = None

    try:
        dataset = gdal.Open(building_dsm_path)
        if dataset is None:
            print(f"Could not open {filename}")
            return filename

        band = dataset.GetRasterBand(1)
        a = band.ReadAsArray().astype(np.float32)

        if a is None:
            print(f"Skipping {filename}, invalid Building_DSM.")
            dataset = None
            return filename

        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        dataset = None

        scale = 1 / geotransform[1]

        walls = findwalls(a, walllimit)
        aspects = filter1Goodwin_as_aspect_v3(walls, scale, a)

        driver = gdal.GetDriverByName('GTiff')
        rows, cols = walls.shape

        wall_ds = driver.Create(wall_path, cols, rows, 1, gdal.GDT_Float32)
        wall_ds.SetGeoTransform(geotransform)
        wall_ds.SetProjection(projection)
        wall_ds.GetRasterBand(1).WriteArray(walls)
        wall_ds.FlushCache()
        wall_ds = None

        aspect_ds = driver.Create(aspect_path, cols, rows, 1, gdal.GDT_Float32)
        aspect_ds.SetGeoTransform(geotransform)
        aspect_ds.SetProjection(projection)
        aspect_ds.GetRasterBand(1).WriteArray(aspects)
        aspect_ds.FlushCache()
        aspect_ds = None

    except Exception as e:
        print(f"Error processing {filename}: {e}")

    finally:
        dataset = None

    return filename

def run_parallel_processing(
    building_dsm_folder_path,
    tree_folder_path,
    wall_output_path,
    aspect_output_path,
    skip_existing=False,
    skip_sparse_tiles=True,
    min_building_fraction=0.01,
    min_tree_fraction=0.01
):
    """
    Process all Building_DSM tiles in parallel to calculate walls and aspects.

    Parameters
    ----------
    building_dsm_folder_path : str
        Folder containing Building_DSM_*.tif
    tree_folder_path : str
        Folder containing Trees_*.tif
    wall_output_path : str
        Output folder for walls_*.tif
    aspect_output_path : str
        Output folder for aspect_*.tif
    skip_existing : bool
        Reuse existing wall/aspect outputs if both files already exist
    skip_sparse_tiles : bool
        Skip tiles where BOTH building and tree coverage are below threshold
    min_building_fraction : float
        e.g. 0.01 means 1%
    min_tree_fraction : float
        e.g. 0.01 means 1%
    """
    os.makedirs(wall_output_path, exist_ok=True)
    os.makedirs(aspect_output_path, exist_ok=True)

    dsm_files = [
        f for f in os.listdir(building_dsm_folder_path)
        if f.endswith('.tif') and not f.startswith('.')
    ]

    args_list = []
    skipped_sparse = []

    for f in dsm_files:
        building_dsm_path = os.path.join(building_dsm_folder_path, f)

        # Building_DSM_0_0.tif -> Trees_0_0.tif
        tree_filename = f.replace("Building_DSM_", "Trees_", 1)
        tree_path = os.path.join(tree_folder_path, tree_filename)

        # Building_DSM_0_0.tif -> walls_0_0.tif / aspect_0_0.tif
        suffix = f[13:-4]
        wall_name = f"walls_{suffix}.tif"
        aspect_name = f"aspect_{suffix}.tif"
        wall_path = os.path.join(wall_output_path, wall_name)
        aspect_path = os.path.join(aspect_output_path, aspect_name)

        if skip_existing and os.path.exists(wall_path) and os.path.exists(aspect_path):
            print(f"Skipping existing wall/aspect for {f}")
            continue

        if skip_sparse_tiles:
            if not os.path.exists(tree_path):
                print(f"Missing tree tile for {f}: {tree_path}")
                continue

            sparse, building_fraction, tree_fraction = _is_sparse_tile(
                building_dsm_path,
                tree_path,
                min_building_fraction=min_building_fraction,
                min_tree_fraction=min_tree_fraction
            )

            print(
                f"[CHECK] {f}: "
                f"building={building_fraction:.2%}, tree={tree_fraction:.2%}"
            )

            if sparse:
                print(
                    f"Skipping sparse tile before wall/aspect: {f} "
                    f"(building={building_fraction:.2%}, tree={tree_fraction:.2%})"
                )
                skipped_sparse.append(f)
                continue

        args_list.append(
            (
                f,
                building_dsm_folder_path,
                tree_folder_path,
                wall_output_path,
                aspect_output_path,
                skip_existing,
                False,  
                min_building_fraction,
                min_tree_fraction
            )
        )

    print(f"Total tiles found: {len(dsm_files)}")
    print(f"Tiles submitted for wall/aspect: {len(args_list)}")
    print(f"Skipped sparse tiles: {len(skipped_sparse)}")

    cpu_count = os.cpu_count() or 1
    if os.name == 'nt':
        max_workers = min(12, max(1, cpu_count // 2))
    else:
        max_workers = min(32, cpu_count)

    print(f"Using {max_workers} parallel workers")

    if not args_list:
        print("No tiles to process after filtering.")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file_parallel, args): args[0] for args in args_list}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Computing Wall Height and Aspect",
            mininterval=1
        ):
            _ = future.result()

