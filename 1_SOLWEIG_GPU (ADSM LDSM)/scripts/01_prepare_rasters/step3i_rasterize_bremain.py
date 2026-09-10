# -*- coding: utf-8 -*-
"""Step3i: burn building_remain (building footprint minus arcade strip) into a 0/1 uint8 mask.
Grids align exactly with the test-area SG_DEM_1m_test.tif and the citywide SUB_SG_Polygon_DEM_1m.tif respectively.
Pixel-centre rule (all_touched=False), consistent with the step3e strip rasterization. pyenv environment."""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features

GPKG = r"D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg"
JOBS = [
    # (grid reference, output, BIGTIFF)
    (r"D:\Claude\SVI_FFW\output\step3_adsm\testrun\SG_DEM_1m_test.tif",
     r"D:\Claude\SVI_FFW\output\step3_adsm\testrun\SG_BREMAIN_1m_test.tif", "NO"),
    (r"D:\Claude\SVI_FFW\TIF\SUB_SG_Polygon_DEM_1m.tif",
     r"D:\Claude\SVI_FFW\TIF\SUB_SG_Polygon_BREMAIN_1m.tif", "YES"),
]

g = gpd.read_file(GPKG)
geoms = [geom for geom in g.geometry if geom is not None and not geom.is_empty]
print(f"building_remain: {len(g)} features, valid geometries {len(geoms)}, total area {g.geometry.area.sum():,.0f} m2")

for ref, out, bigtiff in JOBS:
    with rasterio.open(ref) as r:
        meta = r.meta.copy()
        shape = (r.height, r.width)
        transform = r.transform
        crs = r.crs
    assert crs.to_epsg() == 3414, f"grid CRS is not 3414: {ref}"
    mask = features.rasterize(
        ((geom, 1) for geom in geoms),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    meta.update(dtype="uint8", count=1, nodata=None,
                compress="lzw", tiled=True, BIGTIFF=bigtiff)
    with rasterio.open(out, "w", **meta) as o:
        o.write(mask, 1)
    print(f"{out}: shape={shape}, burned pixels {int(mask.sum()):,} px")
