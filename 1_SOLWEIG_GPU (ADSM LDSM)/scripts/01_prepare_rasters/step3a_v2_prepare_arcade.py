"""Step3a-v2: v2 arcade vectors -> ADSM raster pair (top=bld_h, base=arc_h) + carved-out building DSM (model B).
Assembles a standalone test directory output/step3_adsm/testrun/ (copies the 6 input rasters) without modifying the original desktop data.
Environment: pyenv (rasterio+geopandas).
"""
from pathlib import Path
import shutil
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from shapely.geometry import box

ROOT=Path(r"D:\Claude\SVI_FFW")
SRC=Path(r"C:\Users\City Syntax Lab\Desktop\SOLWEIG_GPU\data\2-SG Test\Input_rasters")
ARC=ROOT/"output"/"detection_geomfirst"/"step2_arcade_sg.gpkg"        # v2 vectors (user-specified)
RUN=ROOT/"output"/"step3_adsm"/"testrun"; RUN.mkdir(parents=True,exist_ok=True)

with rasterio.open(SRC/"SG_DSM_1m_test.tif") as r:
    meta=r.meta.copy(); transform=r.transform; shape=r.shape; crs=r.crs; bounds=r.bounds
    dsm=r.read(1).astype("float32")
with rasterio.open(SRC/"SG_DEM_1m_test.tif") as r:
    dem=r.read(1).astype("float32")

arc=gpd.read_file(ARC).to_crs(crs)
arc=arc[arc.geometry.intersects(box(*bounds))].copy()
print(f"v2 arcade polygons inside the test area: {len(arc)}  (bld_h {arc.bld_h.min():.1f}-{arc.bld_h.max():.1f}m, arc_h {arc.arc_h.min():.1f}-{arc.arc_h.max():.1f}m)")

# top = bld_h (roof of the floors above the walkway, above ground); base = arc_h (clear headroom, above ground)
top=rasterize([(g,float(h)) for g,h in zip(arc.geometry,arc.bld_h)],out_shape=shape,transform=transform,
              fill=0.0,dtype="float32",merge_alg=MergeAlg.replace)
base=rasterize([(g,float(h)) for g,h in zip(arc.geometry,arc.arc_h)],out_shape=shape,transform=transform,
               fill=0.0,dtype="float32",merge_alg=MergeAlg.replace)
strip=top>0
# Numerical protection: the box is at least 0.5 m thick (low buildings with bld_h ~= arc_h; the roof slab should still cast shade)
base=np.where(strip, np.minimum(base, np.maximum(top-0.5, 0.5)), 0.0).astype("float32")
nthin=int((strip & (top-base<0.51)).sum())

# Carve-out (model B): arcade-strip pixels DSM := DEM (ground); the floating box is handled by the ADSM layer
dsm_carved=np.where(strip, dem, dsm).astype("float32")
drop=(dsm-dsm_carved)[strip]
print(f"strip pixels {int(strip.sum())}; carved height median {np.median(drop):.1f}m / max {drop.max():.1f}m; thin-box protection applied to {nthin}px")
print(f"top range {top[strip].min():.1f}-{top[strip].max():.1f}m; base range {base[strip].min():.1f}-{base[strip].max():.1f}m")

meta.update(dtype="float32",count=1,nodata=None)
def w(name,arr):
    with rasterio.open(RUN/name,"w",**meta) as dst: dst.write(arr,1)
    print(f"  wrote testrun/{name}")
w("SG_DSM_carved_1m_test.tif",dsm_carved)
w("SG_ADSM_top_1m_test.tif",top)
w("SG_ADSMB_base_1m_test.tif",base)

# Assemble the remaining inputs (copy; originals untouched)
for f in ["SG_DEM_1m_test.tif","SG_CDSM_1m_test.tif","SG_LDSM_1m_test.tif","SG_LC_1m_test.tif"]:
    shutil.copy2(SRC/f,RUN/f); print(f"  copied {f}")
print("[done] testrun directory ready")
