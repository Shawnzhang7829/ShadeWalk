# -*- coding: utf-8 -*-
"""Step3t: (1) citywide CDSMclean (nodata/negative -> 0, windowed); (2) DEM health check;
(3) rebuild the full set of test inputs on a window strictly on the citywide grid -> testrun_h. pyenv."""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.windows import Window
from pathlib import Path

TIF = Path(r"D:\Claude\SVI_FFW\TIF")
TH  = Path(r"D:\Claude\SVI_FFW\output\step3_adsm\testrun_h")
TH.mkdir(exist_ok=True)

# ---------- (1) Citywide CDSMclean ----------
src = TIF/"SUB_SG_Polygon_CDSM_1m.tif"
dst = TIF/"SUB_SG_Polygon_CDSMclean_1m.tif"
if dst.exists():
    print("(1) CDSMclean already exists, skipping (generated in the previous round: canopy 206,634,406 px, max 40.0 m)")
else:
    with rasterio.open(src) as r:
        meta = r.meta.copy()
        meta.update(nodata=None, compress="lzw", tiled=True, BIGTIFF="YES")
        n_pos = 0; n_fix = 0; vmax = 0.0
        with rasterio.open(dst, "w", **meta) as o:
            for _, win in r.block_windows(1):
                a = r.read(1, window=win)
                bad = ~np.isfinite(a) | (a < 0)
                n_fix += int(bad.sum()); a[bad] = 0
                n_pos += int((a > 0).sum()); vmax = max(vmax, float(a.max()) if a.size else 0)
                o.write(a, 1, window=win)
    print(f"(1) CDSMclean: cleaned {n_fix:,} px (nodata/negative -> 0) | canopy pixels {n_pos:,} | max {vmax:.1f} m")

# ---------- (2) DEM health check ----------
with rasterio.open(TIF/"SUB_SG_Polygon_DEM_1m.tif") as r:
    dmin, dmax = 1e9, -1e9
    n_neg = 0
    for _, win in r.block_windows(1):
        a = r.read(1, window=win)
        a = a[np.isfinite(a)]
        if a.size:
            dmin = min(dmin, float(a.min())); dmax = max(dmax, float(a.max()))
            n_neg += int((a < -100).sum())
print(f"(2) DEM: min {dmin:.2f} / max {dmax:.2f} | large negative values {n_neg:,} px {'(clean, usable as is)' if n_neg==0 else '(needs cleaning!)'}")

# ---------- (3) Aligned test window (same grid as citywide) ----------
with rasterio.open(TIF/"SUB_SG_Polygon_DEM_1m.tif") as r:
    ctr = r.transform
# Old test-window origin (29042,34797), snapped to the citywide grid
col0 = int(round((29042.0 - ctr.c) / ctr.a))
row0 = int(round((34797.0 - ctr.f) / ctr.e))  # the old test transform.f=34797 is already the top-edge Y
W, H = 4000, 3974
win = Window(col0, row0, W, H)
twin = rasterio.windows.transform(win, ctr)
print(f"(3) aligned window: col0={col0}, row0={row0}, origin ({twin.c:.2f},{twin.f:.2f}) (whole-metre aligned)")

def write(name, arr, dtype="float32"):
    m = dict(driver="GTiff", height=H, width=W, count=1, dtype=dtype,
             crs="EPSG:3414", transform=twin, nodata=None, compress="lzw", tiled=True)
    with rasterio.open(TH/name, "w", **m) as o:
        o.write(arr.astype(dtype), 1)

# DEM / CDSM are clipped directly from the citywide rasters (CDSM uses the clean version)
with rasterio.open(TIF/"SUB_SG_Polygon_DEM_1m.tif") as r:
    dem = r.read(1, window=win).astype(np.float32); dem[~np.isfinite(dem) | (dem < -100)] = 0
with rasterio.open(dst) as r:
    cdsm = r.read(1, window=win).astype(np.float32)
write("SG_DEM_1m_test.tif", dem); write("SG_CDSM_1m_test.tif", cdsm)
print(f"   DEM: P50 {np.median(dem):.2f}/max {dem.max():.2f} | CDSM canopy px {int((cdsm>0).sum()):,}")

# Vector rasterization: BREMAIN / DSMremain / LDSMpednet / ADSM / ADSMB
g = gpd.read_file(r"D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg")
h = gpd.pd.to_numeric(g["height"], errors="coerce").fillna(0).astype(float)
hgrid = features.rasterize(((geom, val) for geom, val in zip(g.geometry, h) if geom is not None and not geom.is_empty),
                           out_shape=(H, W), transform=twin, fill=0.0, dtype="float32")
write("SG_BREMAIN_1m_test.tif", (hgrid > 0).astype(np.uint8), "uint8")

arc = gpd.read_file(r"D:\Claude\SVI_FFW\Shp\SG\step2_arcade_sg.gpkg")
top = features.rasterize(((geom, v) for geom, v in zip(arc.geometry, arc["bld_h"].astype(float))),
                         out_shape=(H, W), transform=twin, fill=0.0, dtype="float32")
base = features.rasterize(((geom, v) for geom, v in zip(arc.geometry, arc["arc_h"].astype(float))),
                          out_shape=(H, W), transform=twin, fill=0.0, dtype="float32")
base = np.where(top > 0, np.minimum(base, np.maximum(top - 0.5, 0.5)), 0).astype(np.float32)  # thin-box protection (same as step3e)
write("SG_ADSM_top_1m_test.tif", top); write("SG_ADSMB_base_1m_test.tif", base)

ovl = (hgrid > 0) & (top > 0)
dsmr = dem + hgrid; dsmr[ovl] = dem[ovl]
write("SG_DSMremain_1m_test.tif", dsmr)

clw = gpd.read_file(r"D:\Claude\SVI_FFW\Shp\SG\covered_linkway_SG_island_tv_pednet_bridged.gpkg")
ld = features.rasterize(((geom, 3.0) for geom in clw.geometry if geom is not None and not geom.is_empty),
                        out_shape=(H, W), transform=twin, fill=0.0, dtype="float32")
write("SG_LDSMpednet_1m_test.tif", ld)

z = np.zeros((H, W), np.float32)
write("SG_WALLS0_1m_test.tif", z); write("SG_ASPECT0_1m_test.tif", z)

print(f"   strip px {int((top>0).sum()):,} | BREMAIN px {int((hgrid>0).sum()):,} | overlap with strip {int(ovl.sum()):,} | LDSM px {int((ld>0).sum()):,}")
print("testrun_h inputs rebuilt (strictly on the citywide grid)")
