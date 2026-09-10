# -*- coding: utf-8 -*-
"""Citywide SG only_shadow + 13-class run, recomputing shadows with the new arcade/remain layers. Output to TIF_shadow_newarcade\output_folder (does not overwrite the old TIF folder).
 Run: python-qgis-ltr.bat run_newarcade.py (torch+gdal+local arcade solweig)."""
import sys
sys.path.insert(0, r'D:\Claude\SVI_FFW\Module\SOLWEIG-GPU')   # force the local arcade-customised version instead of the pip release
from solweig_gpu import thermal_comfort
import time

BASE = r"D:\Claude\SVI_FFW\TIF_shadow_newarcade"
MET  = r"D:\Claude\SVI_FFW\TIF\Forcing_data\S50_Clementi Road.txt"

def main():
    t0 = time.perf_counter()
    thermal_comfort(
        base_path=BASE,
        selected_date_str='2026-03-01',
        building_dsm_filename='SUB_SG_Polygon_DSMremain_1m.tif',   # new remain (DEM+height)
        dem_filename='SUB_SG_Polygon_DEM_1m.tif',                  # hard link to the old DEM
        trees_filename='SUB_SG_Polygon_CDSMclean_1m.tif',          # CDSM with nodata cleaned
        landcover_filename=None,
        ldsm_filename='SUB_SG_Polygon_LDSMpednet_1m.tif',          # linkway @3m
        arcade_filename='SUB_SG_Polygon_ADSM_1m.tif',              # new arcade top bld_h
        arcade_base_filename='SUB_SG_Polygon_ADSMB_1m.tif',        # new arcade base arc_h
        building_remain_filename='SUB_SG_Polygon_BREMAIN_1m.tif',  # new remain mask (class 12)
        wallheight_filename='SUB_SG_Polygon_WALLS0_1m.tif',        # hard link to the zero walls
        wallaspect_filename='SUB_SG_Polygon_ASPECT0_1m.tif',
        tile_size=4000,
        overlap=100,
        use_own_met=True,
        start_time='2026-03-01 00:00:00',
        end_time='2026-03-01 23:00:00',
        own_met_file=MET,
        save_tmrt=False,
        save_shadow=True,
        shadow_category=True,
        only_shadow=True,
        skip_sparse_tiles=True,
        reuse_tiles=True,
    )
    print(f'[newarcade city run done] {(time.perf_counter()-t0)/60:.1f} min')

if __name__ == "__main__":
    main()
