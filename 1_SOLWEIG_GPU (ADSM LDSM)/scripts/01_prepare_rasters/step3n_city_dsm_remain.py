# -*- coding: utf-8 -*-
"""Step3n: citywide remain-based building DSM (= DEM + building_remain.height; the arcade strip is naturally ground level).
44000x27000 float32: the height raster is burned in one pass over the full extent (~4.8 GB); DEM/ADSM/output are streamed by window. pyenv."""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features

GPKG = r"D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg"
DEM  = r"D:\Claude\SVI_FFW\TIF\SUB_SG_Polygon_DEM_1m.tif"
ADSM = r"D:\Claude\SVI_FFW\TIF\SUB_SG_Polygon_ADSM_1m.tif"
OUT  = r"D:\Claude\SVI_FFW\TIF\SUB_SG_Polygon_DSMremain_1m.tif"

g = gpd.read_file(GPKG)
h = gpd.pd.to_numeric(g["height"], errors="coerce").fillna(0).astype(float)
print(f"building_remain: {len(g)} features, height med {h.median():.2f} / max {h.max():.2f} m")

with rasterio.open(DEM) as r:
    meta = r.meta.copy(); H, W = r.height, r.width; transform = r.transform

hgrid = features.rasterize(
    ((geom, val) for geom, val in zip(g.geometry, h) if geom is not None and not geom.is_empty),
    out_shape=(H, W), transform=transform, fill=0.0, dtype="float32", all_touched=False)
n_b = int((hgrid > 0).sum())
print(f"building pixels (height>0) = {n_b:,} (BREMAIN mask should be 80,208,935)")

meta.update(dtype="float32", count=1, nodata=None, compress="lzw", tiled=True, BIGTIFF="YES")
n_overlap = 0
with rasterio.open(DEM) as rd, rasterio.open(ADSM) as ra, rasterio.open(OUT, "w", **meta) as o:
    for _, win in rd.block_windows(1):
        r0, c0 = win.row_off, win.col_off
        dem = rd.read(1, window=win).astype(np.float32)
        dem[dem < -100] = 0.0                      # nodata cleaning (same rule as step3e)
        hh = hgrid[r0:r0+win.height, c0:c0+win.width]
        top = ra.read(1, window=win)
        ovl = (hh > 0) & (top > 0)
        n_overlap += int(ovl.sum())
        out = dem + hh
        out[ovl] = dem[ovl]                        # strip safety cut (theoretically 0)
        o.write(out, 1, window=win)
print(f"pixels overlapping the strip = {n_overlap:,} (theoretically 0)")
print(f"written: {OUT}")
