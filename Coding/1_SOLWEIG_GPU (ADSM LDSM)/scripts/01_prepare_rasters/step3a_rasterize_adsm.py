"""Step3a: Step2 arcade polygons -> ADSM raster (aligned to the test-area DSM grid, value = clearance height arc_h).
Follows the LDSM format, for use by the SOLWEIG arcade module.
"""
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

ROOT=Path(r"D:\Claude\SVI_FFW")
TESTDIR=Path(r"C:\Users\City Syntax Lab\Desktop\SOLWEIG_GPU\data\2-SG Test\Input_rasters")
REF=TESTDIR/"SG_DSM_1m_test.tif"
ARC=ROOT/"output"/"step2_projection"/"step2_arcade_sg.gpkg"
OUT_LOCAL=ROOT/"output"/"step3_adsm"/"SG_ADSM_1m_test.tif"
OUT_TEST=TESTDIR/"SG_ADSM_1m_test.tif"
OUT_LOCAL.parent.mkdir(parents=True,exist_ok=True)

with rasterio.open(REF) as ref:
    meta=ref.meta.copy(); transform=ref.transform; shape=ref.shape; crs=ref.crs; bounds=ref.bounds

arc=gpd.read_file(ARC).to_crs(crs)
# Clip to the test area
from shapely.geometry import box
arc=arc[arc.geometry.intersects(box(*bounds))].copy()
print(f"arcade polygons inside the test area: {len(arc)}")
if len(arc)==0:
    print("WARNING: no arcade inside the test area -- this 4000 px tile may not lie in an arcade-dense (five-foot way) district");
# Rasterize: value=arc_h (clearance height), burned from small to large area (larger ones overwrite smaller; clearance heights are nearly uniform here)
shapes=[(geom,float(h)) for geom,h in zip(arc.geometry,arc["arc_h"])]
adsm=rasterize(shapes,out_shape=shape,transform=transform,fill=0.0,dtype="float32",merge_alg=rasterio.enums.MergeAlg.replace) if shapes else np.zeros(shape,dtype="float32")

meta.update(dtype="float32",count=1,nodata=0.0)
for p in [OUT_LOCAL,OUT_TEST]:
    with rasterio.open(p,"w",**meta) as dst: dst.write(adsm,1)
print(f"ADSM: nonzero pixels {int(np.count_nonzero(adsm))}, clearance range {adsm[adsm>0].min() if np.any(adsm>0) else 0:.1f}-{adsm.max():.1f}m")
print(f"wrote {OUT_LOCAL}")
print(f"wrote {OUT_TEST}")
