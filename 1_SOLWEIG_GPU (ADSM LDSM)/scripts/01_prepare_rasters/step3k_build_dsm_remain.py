# -*- coding: utf-8 -*-
"""Step3k: rebuild the test-area building DSM from the building_remain vectors (height field, AGL).
  DSM_new = DEM + height (inside remain polygons, pixel-centre rule) ; elsewhere = DEM
  Strip safety check: pixels overlapping the ADSM strip are forced = DEM (theoretically 0 pixels)
Output: testrun\SG_DSMremain_1m_test.tif + copy to testrun_b. pyenv."""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from pathlib import Path

GPKG = r"D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg"
TR  = Path(r"D:\Claude\SVI_FFW\output\step3_adsm\testrun")
TRB = Path(r"D:\Claude\SVI_FFW\output\step3_adsm\testrun_b")

g = gpd.read_file(GPKG)
h = gpd.pd.to_numeric(g["height"], errors="coerce").fillna(0).astype(float)
print(f"building_remain: {len(g)} features, height min {h.min():.2f} / med {h.median():.2f} / max {h.max():.2f} m")

with rasterio.open(TR / "SG_DEM_1m_test.tif") as r:
    dem = r.read(1).astype(np.float32)
    meta = r.meta.copy(); shape = (r.height, r.width); transform = r.transform

hgrid = features.rasterize(
    ((geom, val) for geom, val in zip(g.geometry, h) if geom is not None and not geom.is_empty),
    out_shape=shape, transform=transform, fill=0.0, dtype="float32", all_touched=False)

with rasterio.open(TR / "SG_ADSM_top_1m_test.tif") as r:
    top = r.read(1).astype(np.float32)
with rasterio.open(TR / "SG_BREMAIN_1m_test.tif") as r:
    bmask = r.read(1).astype(bool)

n_b = int((hgrid > 0).sum())
overlap = (hgrid > 0) & (top > 0)
print(f"building pixels (height>0) = {n_b:,} | BREMAIN mask pixels = {int(bmask.sum()):,} | overlap with strip = {int(overlap.sum()):,}")
assert n_b == int(bmask.sum()), "building pixel count != mask pixel count (same vectors, grid and rule; they should be equal)"

dsm_new = dem + hgrid
dsm_new[overlap] = dem[overlap]   # safety cut (theoretically 0 pixels)

meta.update(dtype="float32", count=1, compress="lzw", tiled=True)
for out in (TR / "SG_DSMremain_1m_test.tif", TRB / "SG_DSMremain_1m_test.tif"):
    with rasterio.open(out, "w", **meta) as o:
        o.write(dsm_new, 1)
print(f"building height above DEM: min {hgrid[hgrid>0].min():.2f} / med {np.median(hgrid[hgrid>0]):.2f} / max {hgrid.max():.2f} m")
print("written: SG_DSMremain_1m_test.tif (testrun + testrun_b)")
