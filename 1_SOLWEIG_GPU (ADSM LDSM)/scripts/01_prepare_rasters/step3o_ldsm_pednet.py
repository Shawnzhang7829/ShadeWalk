# -*- coding: utf-8 -*-
"""Step3o: generate the LDSM from the new covered-linkway vectors (pednet_bridged), uniform height 3.0 m.
  Test area: SG_LDSMpednet_1m_test.tif (testrun_c grid)
  Citywide:  SUB_SG_Polygon_LDSMpednet_1m.tif (written by window)
Pixel-centre rule, float32, top height 3.0 (the bottom height 2.7 is derived at run time from ldsm_bottom_height_ratio=0.9). pyenv."""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from pathlib import Path

GPKG = r"D:\Claude\SVI_FFW\Shp\SG\covered_linkway_SG_island_tv_pednet_bridged.gpkg"
H_CLW = 3.0

g = gpd.read_file(GPKG)
geoms = [geom for geom in g.geometry if geom is not None and not geom.is_empty]
print(f"covered linkway: {len(geoms)} polygons, total area {g.geometry.area.sum():,.0f} m2, burned height {H_CLW} m")

# (1) Test area (full extent)
ref = r"D:\Claude\SVI_FFW\output\step3_adsm\testrun_c\SG_DEM_1m_test.tif"
with rasterio.open(ref) as r:
    meta = r.meta.copy(); shape = (r.height, r.width); transform = r.transform
arr = features.rasterize(((geom, H_CLW) for geom in geoms), out_shape=shape,
                         transform=transform, fill=0.0, dtype="float32", all_touched=False)
meta.update(dtype="float32", count=1, nodata=None, compress="lzw", tiled=True)
for d in ("testrun_c", "testrun"):
    out = Path(r"D:\Claude\SVI_FFW\output\step3_adsm") / d / "SG_LDSMpednet_1m_test.tif"
    with rasterio.open(out, "w", **meta) as o: o.write(arr, 1)
print(f"test area: burned pixels {int((arr>0).sum()):,} px -> SG_LDSMpednet_1m_test.tif (testrun_c + testrun)")

# (2) Citywide (height grid burned once over the full extent, output written by window)
DEM = r"D:\Claude\SVI_FFW\TIF\SUB_SG_Polygon_DEM_1m.tif"
OUT = r"D:\Claude\SVI_FFW\TIF\SUB_SG_Polygon_LDSMpednet_1m.tif"
with rasterio.open(DEM) as r:
    meta2 = r.meta.copy(); H, W = r.height, r.width; tr2 = r.transform
big = features.rasterize(((geom, H_CLW) for geom in geoms), out_shape=(H, W),
                         transform=tr2, fill=0.0, dtype="float32", all_touched=False)
print(f"citywide: burned pixels {int((big>0).sum()):,} px (vector area 1,657,712 m2)")
meta2.update(dtype="float32", count=1, nodata=None, compress="lzw", tiled=True, BIGTIFF="YES")
with rasterio.open(DEM) as rd, rasterio.open(OUT, "w", **meta2) as o:
    for _, win in rd.block_windows(1):
        r0, c0 = win.row_off, win.col_off
        o.write(big[r0:r0+win.height, c0:c0+win.width], 1, window=win)
print(f"written: {OUT}")
