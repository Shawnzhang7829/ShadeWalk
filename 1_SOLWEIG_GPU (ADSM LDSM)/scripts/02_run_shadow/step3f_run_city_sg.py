"""Step3f: citywide SG only_shadow + 12-class category (identical to the notebook "single run" cell). QGIS env."""
from solweig_gpu import thermal_comfort
import time

BASE=r"D:\Claude\SVI_FFW\TIF"
MET =r"D:\Claude\SVI_FFW\TIF\Forcing_data\S50_Clementi Road.txt"

def main():
    start=time.perf_counter()
    thermal_comfort(
        base_path=BASE,
        selected_date_str='2026-03-01',
        building_dsm_filename='SUB_SG_Polygon_DSMremain_1m.tif',  # DEM+building_remain.height, pixel-consistent with the class-12 mask (replaces the old DSMcarved)
        dem_filename='SUB_SG_Polygon_DEM_1m.tif',
        trees_filename='SUB_SG_Polygon_CDSMclean_1m.tif',  # in the original CDSM, tree-free pixels are nodata (-3.4e38); cleaned to 0
        landcover_filename=None,
        ldsm_filename='SUB_SG_Polygon_LDSMpednet_1m.tif',  # new pednet vectors @3 m (the old government raster had 31.3M px at an anomalous 255 m; discarded)
        arcade_filename='SUB_SG_Polygon_ADSM_1m.tif',
        arcade_base_filename='SUB_SG_Polygon_ADSMB_1m.tif',
        building_remain_filename='SUB_SG_Polygon_BREMAIN_1m.tif',  # class 12: building footprint, not shadow
        wallheight_filename='SUB_SG_Polygon_WALLS0_1m.tif',   # zero walls: only_shadow output verified bit-for-bit equivalent
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
    print(f'[city run done] {(time.perf_counter()-start)/60:.1f} min')

if __name__=="__main__":
    main()
