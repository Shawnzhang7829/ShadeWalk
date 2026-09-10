"""Test-area run (b10 configuration): 13-class shadow category with DSMremain building DSM
and the pedestrian-network covered-linkway LDSM (3 m). Only difference from b4c:
ldsm_filename = SG_LDSMpednet_1m_test.tif. Runs in the QGIS Python environment.
"""
from solweig_gpu import thermal_comfort

BASE=r"D:\Claude\SVI_FFW\output\step3_adsm\testrun_i"
MET=r"C:\Users\City Syntax Lab\Desktop\SOLWEIG_GPU\data\2-SG Test\Forcing_data\S50_Clementi Road.txt"

def main():
    thermal_comfort(
    base_path=BASE,
    selected_date_str='2026-03-01',
    building_dsm_filename='SG_DSMremain_1m_test.tif',
    dem_filename='SG_DEM_1m_test.tif',
    trees_filename='SG_CDSM_1m_test.tif',
    ldsm_filename='SG_LDSMpednet_1m_test.tif',   # new pednet vectors, uniform 3 m (replaces the old government raster with anomalous 255 m values)
    arcade_filename='SG_ADSM_top_1m_test.tif',
    arcade_base_filename='SG_ADSMB_base_1m_test.tif',
    building_remain_filename='SG_BREMAIN_1m_test.tif',
    wallheight_filename='SG_WALLS0_1m_test.tif',
    wallaspect_filename='SG_ASPECT0_1m_test.tif',
    landcover_filename=None,
    tile_size=4000,
    overlap=0,
    use_own_met=True,
    start_time='2026-03-01 00:00:00',
    end_time='2026-03-01 23:00:00',
    own_met_file=MET,
    save_tmrt=False,
    save_shadow=True,
    shadow_category=True,
    only_shadow=True,
    skip_sparse_tiles=False,
    reuse_tiles=True,        # LDSM folder does not exist -> tiled from the new file; the other layers reuse hard links
    )
    print('[step3c-b10 done]')

if __name__ == "__main__":
    main()





