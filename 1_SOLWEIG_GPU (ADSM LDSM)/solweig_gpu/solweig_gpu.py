
from typing import Optional, Union

def thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    ldsm_filename: Optional[str] = None,
    arcade_filename: Optional[str] = None,
    arcade_base_filename: Optional[str] = None,
    building_remain_filename: Optional[str] = None,
    landcover_filename: Optional[str] = None,
    wallheight_filename: Union[str, bool] = None,
    wallaspect_filename: Union[str, bool] = None,
    tile_size=3600,
    overlap=20,
    use_own_met=True,
    start_time=None,
    end_time=None,
    data_source_type=None,
    data_folder=None,
    own_met_file=None,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    shadow_category=False,
    only_shadow=False,
    reuse_tiles=False,
    reuse_wall_aspect=False,
    skip_sparse_tiles=True,
    min_building_fraction=0.01,
    min_tree_fraction=0.01,
    min_ldsm_fraction=0.0,
    canopy_height_ratio=0.23,
    ldsm_bottom_height_ratio=0.90,
    shelter_transmittance=0.0,
    arcade_transmittance=0.0,
):
    """
    Main function to compute urban thermal comfort using the SOLWEIG-GPU model.

    Supports optional LDSM input for urban shelter/shading structures. When
    `ldsm_filename` is provided, the function expects preprocessing to create
    `processed_inputs/LDSM/` tiles and forwards `ldsm_path` plus LDSM-related
    parameters into `compute_utci()`.

    Notes
    -----
    - If your `preprocessor.ppr()` has already been extended to accept
      `ldsm_filename`, this function will pass it automatically.
    - If `ldsm_filename` is provided but `ppr()` does not support it yet,
      this function raises a clear error telling you to update the preprocessor.
    """

    import inspect
    import os
    import numpy as np
    import torch
    from osgeo import gdal

    from .preprocessor import ppr
    from .utci_process import compute_utci, map_files_by_key
    from .walls_aspect import run_parallel_processing

    def _coverage_fraction(arr):
        if arr is None:
            return 0.0
        total = arr.size
        if total == 0:
            return 0.0
        return float(np.count_nonzero(arr > 0)) / float(total)

    def _read_fraction(path, label):
        if path is None:
            return 0.0
        ds = gdal.Open(path)
        if ds is None:
            raise FileNotFoundError(f"Could not open {label} tile: {path}")
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        return _coverage_fraction(arr)

    def _is_sparse_tile(
        building_dsm_path,
        tree_path,
        ldsm_path=None,
        min_building_fraction=0.01,
        min_tree_fraction=0.01,
        min_ldsm_fraction=0.0,
    ):
        building_fraction = _read_fraction(building_dsm_path, "Building_DSM")
        tree_fraction = _read_fraction(tree_path, "Trees")
        ldsm_fraction = _read_fraction(ldsm_path, "LDSM") if ldsm_path is not None else 0.0

        sparse = (
            (building_fraction < min_building_fraction)
            and (tree_fraction < min_tree_fraction)
            and (ldsm_fraction < min_ldsm_fraction)
        )
        return sparse, building_fraction, tree_fraction, ldsm_fraction

    preprocess_dir = os.path.join(base_path, "processed_inputs")
    os.makedirs(preprocess_dir, exist_ok=True)

    ppr_kwargs = dict(
        base_path=base_path,
        building_dsm_filename=building_dsm_filename,
        dem_filename=dem_filename,
        trees_filename=trees_filename,
        landcover_filename=landcover_filename,
        wallheight_filename=wallheight_filename,
        wallaspect_filename=wallaspect_filename,
        tile_size=tile_size,
        overlap=overlap,
        selected_date_str=selected_date_str,
        use_own_met=use_own_met,
        start_time=start_time,
        end_time=end_time,
        data_source_type=data_source_type,
        data_folder=data_folder,
        own_met_file=own_met_file,
        preprocess_dir=preprocess_dir,
        reuse_tiles=reuse_tiles,
    )

    ppr_sig = inspect.signature(ppr)

    if ldsm_filename is not None:
        if "ldsm_filename" in ppr_sig.parameters:
            ppr_kwargs["ldsm_filename"] = ldsm_filename
        else:
            raise TypeError(
                "ldsm_filename was provided, but preprocessor.ppr() does not accept "
                "'ldsm_filename' yet. Please update preprocessor.py as well."
            )

    if (arcade_filename is None) != (arcade_base_filename is None):
        raise ValueError("arcade_filename and arcade_base_filename must be provided together.")
    if arcade_filename is not None:
        if "arcade_filename" in ppr_sig.parameters:
            ppr_kwargs["arcade_filename"] = arcade_filename
            ppr_kwargs["arcade_base_filename"] = arcade_base_filename
        else:
            raise TypeError(
                "arcade_filename was provided, but preprocessor.ppr() does not accept "
                "'arcade_filename' yet. Please update preprocessor.py as well."
            )

    if building_remain_filename is not None:
        if "building_remain_filename" in ppr_sig.parameters:
            ppr_kwargs["building_remain_filename"] = building_remain_filename
        else:
            raise TypeError(
                "building_remain_filename was provided, but preprocessor.ppr() does not accept "
                "'building_remain_filename' yet. Please update preprocessor.py as well."
            )

    if "wallheight_filename" not in ppr_sig.parameters or "wallaspect_filename" not in ppr_sig.parameters:
        raise TypeError(
            "thermal_comfort() now passes 'wallheight_filename' and 'wallaspect_filename', "
            "but preprocessor.ppr() has not been updated yet. Please update preprocessor.py as well."
        )

    ppr(**ppr_kwargs)

    skipped_tiles_file = os.path.join(preprocess_dir, "skipped_sparse_tiles.txt")
    with open(skipped_tiles_file, "w", encoding="utf-8") as f:
        if ldsm_filename is None:
            f.write("tile_key,building_fraction,tree_fraction\n")
        else:
            f.write("tile_key,building_fraction,tree_fraction,ldsm_fraction\n")

    base_output_path = os.path.join(base_path, "output_folder")
    inputMet = os.path.join(preprocess_dir, "metfiles")
    building_dsm_dir = os.path.join(preprocess_dir, "Building_DSM")
    tree_dir = os.path.join(preprocess_dir, "Trees")
    ldsm_dir = os.path.join(preprocess_dir, "LDSM") if ldsm_filename is not None else None
    adsm_dir = os.path.join(preprocess_dir, "ADSM") if arcade_filename is not None else None
    adsmb_dir = os.path.join(preprocess_dir, "ADSMB") if arcade_filename is not None else None
    bremain_dir = os.path.join(preprocess_dir, "BREMAIN") if building_remain_filename is not None else None
    dem_dir = os.path.join(preprocess_dir, "DEM")
    landcover_dir = os.path.join(preprocess_dir, "Landcover") if landcover_filename is not None else None
    walls_dir = os.path.join(preprocess_dir, "walls")
    aspect_dir = os.path.join(preprocess_dir, "aspect")

    use_input_wallheight = wallheight_filename not in [None, False, ""]
    use_input_wallaspect = wallaspect_filename not in [None, False, ""]

    if use_input_wallheight != use_input_wallaspect:
        raise ValueError(
            "wallheight_filename and wallaspect_filename must be provided together, "
            "or both set to False."
        )

    if use_input_wallheight and use_input_wallaspect:
        print("Using user-provided wall height / wall aspect rasters; skipping wall/aspect computation.")
    else:
        run_parallel_processing(
            building_dsm_dir,
            tree_dir,
            walls_dir,
            aspect_dir,
            skip_existing=reuse_wall_aspect,
            skip_sparse_tiles=skip_sparse_tiles,
            min_building_fraction=min_building_fraction,
            min_tree_fraction=min_tree_fraction
        )

    print("Running Solweig ...")

    building_dsm_map = map_files_by_key(building_dsm_dir, ".tif")
    tree_map = map_files_by_key(tree_dir, ".tif")
    ldsm_map = map_files_by_key(ldsm_dir, ".tif") if ldsm_dir else {}
    adsm_map = map_files_by_key(adsm_dir, ".tif") if adsm_dir else {}
    adsmb_map = map_files_by_key(adsmb_dir, ".tif") if adsmb_dir else {}
    bremain_map = map_files_by_key(bremain_dir, ".tif") if bremain_dir else {}
    dem_map = map_files_by_key(dem_dir, ".tif")
    landcover_map = map_files_by_key(landcover_dir, ".tif") if landcover_dir else {}
    walls_map = map_files_by_key(walls_dir, ".tif")
    aspect_map = map_files_by_key(aspect_dir, ".tif")
    met_map = map_files_by_key(inputMet, ".txt")

    common_keys = set(building_dsm_map) & set(tree_map) & set(dem_map) & set(met_map)
    if landcover_dir:
        common_keys &= set(landcover_map)
    if ldsm_dir:
        common_keys &= set(ldsm_map)
    if adsm_dir:
        common_keys &= set(adsm_map)
        common_keys &= set(adsmb_map)
    if bremain_dir:
        common_keys &= set(bremain_map)

    def _numeric_key(k: str):
        try:
            x_str, y_str = k.split("_")
            return (int(x_str), int(y_str))
        except Exception:
            return (10**18, 10**18)

    sorted_keys = sorted(common_keys, key=_numeric_key)

    for key in sorted_keys:
        print(f"Processing tile: {key}")

        dsm_path = building_dsm_map[key]
        tree_path = tree_map[key]
        ldsm_path = ldsm_map.get(key) if ldsm_dir else None
        adsm_path = adsm_map.get(key) if adsm_dir else None
        adsmb_path = adsmb_map.get(key) if adsm_dir else None
        bremain_path = bremain_map.get(key) if bremain_dir else None
        dem_path = dem_map[key]
        met_path = met_map[key]
        walls_path = walls_map.get(key)
        aspect_path = aspect_map.get(key)
        landcover_path = landcover_map.get(key) if landcover_dir else None

        if skip_sparse_tiles:
            sparse, building_fraction, tree_fraction, ldsm_fraction = _is_sparse_tile(
                dsm_path,
                tree_path,
                ldsm_path=ldsm_path,
                min_building_fraction=min_building_fraction,
                min_tree_fraction=min_tree_fraction,
                min_ldsm_fraction=min_ldsm_fraction,
            )
            if sparse:
                if ldsm_filename is None:
                    print(
                        f"Skipping sparse tile in main simulation: {key} "
                        f"(building={building_fraction:.2%}, tree={tree_fraction:.2%})"
                    )
                    with open(skipped_tiles_file, "a", encoding="utf-8") as f:
                        f.write(f"{key},{building_fraction:.6f},{tree_fraction:.6f}\n")
                else:
                    print(
                        f"Skipping sparse tile in main simulation: {key} "
                        f"(building={building_fraction:.2%}, tree={tree_fraction:.2%}, "
                        f"ldsm={ldsm_fraction:.2%})"
                    )
                    with open(skipped_tiles_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"{key},{building_fraction:.6f},{tree_fraction:.6f},"
                            f"{ldsm_fraction:.6f}\n"
                        )
                continue

        if walls_path is None or aspect_path is None:
            print(f"Skipping tile {key}: missing wall or aspect file.")
            continue

        output_folder = os.path.join(base_output_path, key)
        os.makedirs(output_folder, exist_ok=True)

        met_file_data = np.loadtxt(met_path, skiprows=1)

        compute_kwargs = dict(
            building_dsm_path=dsm_path,
            tree_path=tree_path,
            dem_path=dem_path,
            walls_path=walls_path,
            aspect_path=aspect_path,
            landcover_path=landcover_path,
            met_file_data=met_file_data,
            output_path=output_folder,
            number=key,
            selected_date_str=selected_date_str,
            save_tmrt=save_tmrt,
            save_svf=save_svf,
            save_kup=save_kup,
            save_kdown=save_kdown,
            save_lup=save_lup,
            save_ldown=save_ldown,
            save_shadow=save_shadow,
            shadow_category=shadow_category,
            only_shadow=only_shadow,
            canopy_height_ratio=canopy_height_ratio,
            ldsm_bottom_height_ratio=ldsm_bottom_height_ratio,
            shelter_transmittance=shelter_transmittance,
        )

        compute_sig = inspect.signature(compute_utci)
        if "ldsm_path" in compute_sig.parameters:
            compute_kwargs["ldsm_path"] = ldsm_path
        elif ldsm_path is not None:
            raise TypeError(
                "ldsm_filename was provided, but utci_process.compute_utci() does not accept "
                "'ldsm_path' yet. Please update utci_process.py as well."
            )

        if "adsm_path" in compute_sig.parameters:
            compute_kwargs["adsm_path"] = adsm_path
            compute_kwargs["adsm_base_path"] = adsmb_path
            compute_kwargs["arcade_transmittance"] = arcade_transmittance
        elif adsm_path is not None:
            raise TypeError(
                "arcade_filename was provided, but utci_process.compute_utci() does not accept "
                "'adsm_path' yet. Please update utci_process.py as well."
            )

        if "bremain_path" in compute_sig.parameters:
            compute_kwargs["bremain_path"] = bremain_path
        elif bremain_path is not None:
            raise TypeError(
                "building_remain_filename was provided, but utci_process.compute_utci() does not accept "
                "'bremain_path' yet. Please update utci_process.py as well."
            )

        compute_utci(**compute_kwargs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
