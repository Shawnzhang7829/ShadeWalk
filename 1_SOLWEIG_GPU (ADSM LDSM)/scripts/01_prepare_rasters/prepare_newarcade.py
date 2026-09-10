# -*- coding: utf-8 -*-
"""Generate all SOLWEIG input raster layers from the new arcade/remain/linkway vectors + CDSM into a new directory (does not overwrite the old TIF folder).
 Combines step3e (ADSM/ADSMB) + step3n (DSMremain) + step3i (BREMAIN) + step3o (LDSM) + CDSMclean + hard links. pyenv 3.11.9 (rasterio)."""
import os, time, shutil
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
TIF=r"D:\Claude\SVI_FFW\TIF"; NEW=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade"; os.makedirs(NEW,exist_ok=True)
DEM=f"{TIF}/SUB_SG_Polygon_DEM_1m.tif"; CDSM=f"{TIF}/SUB_SG_Polygon_CDSM_1m.tif"
ARC=r"D:\Claude\SVI_FFW\Shp\SG\step2_arcade_sg.gpkg"
REM=r"D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg"
LNK=r"D:\Claude\SVI_FFW\Shp\SG\covered_linkway_SG_island_tv_pednet_bridged.gpkg"
def P(n): return f"{NEW}/SUB_SG_Polygon_{n}_1m.tif"
t0=time.time()
with rasterio.open(DEM) as r: meta=r.meta.copy(); H,W=r.height,r.width; tr=r.transform; crs=r.crs
print(f"grid {W}x{H} crs={crs}",flush=True)
meta.update(dtype="float32",count=1,nodata=None,compress="lzw",tiled=True,BIGTIFF="YES")

# 1) ADSM (bld_h, box top)
arc=gpd.read_file(ARC).to_crs(crs); print(f"arcade {len(arc)} polys",flush=True)
arr=rasterize([(g,float(v)) for g,v in zip(arc.geometry,arc["bld_h"])],out_shape=(H,W),transform=tr,fill=0.0,dtype="float32",merge_alg=MergeAlg.replace)
with rasterio.open(P("ADSM"),"w",**meta) as o: o.write(arr,1)
print(f"[ADSM] {time.time()-t0:.0f}s",flush=True); del arr
# 2) ADSMB (arc_h, box base, raw)
arr=rasterize([(g,float(v)) for g,v in zip(arc.geometry,arc["arc_h"])],out_shape=(H,W),transform=tr,fill=0.0,dtype="float32",merge_alg=MergeAlg.replace)
with rasterio.open(P("ADSMB"),"w",**meta) as o: o.write(arr,1)
print(f"[ADSMB raw] {time.time()-t0:.0f}s",flush=True); del arr
# Thin-box protection (windowed): on the strip base=min(base, max(top-0.5,0.5)); outside the strip = 0
with rasterio.open(P("ADSM")) as rt, rasterio.open(P("ADSMB"),"r+") as rb:
    for _,win in rt.block_windows(1):
        top=rt.read(1,window=win); base=rb.read(1,window=win); strip=top>0
        base=np.where(strip,np.minimum(base,np.maximum(top-0.5,0.5)),0).astype("float32")
        rb.write(base,1,window=win)
print(f"[ADSMB protect] {time.time()-t0:.0f}s",flush=True)

# 3) BREMAIN (remain.height>0 mask) + DSMremain (DEM+height, strip cut-out)
g=gpd.read_file(REM).to_crs(crs); h=gpd.pd.to_numeric(g["height"],errors="coerce").fillna(0).astype(float)
print(f"remain {len(g)} h_med {h.median():.1f}/max {h.max():.1f}",flush=True)
hgrid=rasterize(((geom,val) for geom,val in zip(g.geometry,h) if geom is not None and not geom.is_empty),out_shape=(H,W),transform=tr,fill=0.0,dtype="float32",all_touched=False)
bmeta=meta.copy(); bmeta.update(dtype="uint8")
with rasterio.open(P("BREMAIN"),"w",**bmeta) as o: o.write((hgrid>0).astype("uint8"),1)
print(f"[BREMAIN] {int((hgrid>0).sum()):,}px {time.time()-t0:.0f}s",flush=True)
with rasterio.open(DEM) as rd, rasterio.open(P("ADSM")) as ra, rasterio.open(P("DSMremain"),"w",**meta) as o:
    for _,win in rd.block_windows(1):
        r0,c0=win.row_off,win.col_off
        dem=rd.read(1,window=win).astype("float32"); dem[dem<-100]=0.0
        hh=hgrid[r0:r0+win.height,c0:c0+win.width]; tp=ra.read(1,window=win)
        out=dem+hh; ovl=(hh>0)&(tp>0); out[ovl]=dem[ovl]
        o.write(out,1,window=win)
print(f"[DSMremain] {time.time()-t0:.0f}s",flush=True); del hgrid

# 4) LDSM (linkway top 3.0 m)
lk=gpd.read_file(LNK).to_crs(crs); geoms=[x for x in lk.geometry if x is not None and not x.is_empty]
arr=rasterize(((geom,3.0) for geom in geoms),out_shape=(H,W),transform=tr,fill=0.0,dtype="float32",all_touched=False)
with rasterio.open(P("LDSMpednet"),"w",**meta) as o: o.write(arr,1)
print(f"[LDSM] {int((arr>0).sum()):,}px {time.time()-t0:.0f}s",flush=True); del arr

# 5) CDSMclean (nodata/-3.4e38 -> 0, windowed)
with rasterio.open(CDSM) as rc, rasterio.open(P("CDSMclean"),"w",**meta) as o:
    for _,win in rc.block_windows(1):
        a=rc.read(1,window=win).astype("float32")
        a=np.where((a<-1e30)|~np.isfinite(a),0.0,a).astype("float32")
        o.write(a,1,window=win)
print(f"[CDSMclean] {time.time()-t0:.0f}s",flush=True)

# 6) Hard-link DEM/WALLS0/ASPECT0 (content unchanged)
for n in ["DEM","WALLS0","ASPECT0"]:
    src=f"{TIF}/SUB_SG_Polygon_{n}_1m.tif"; dst=P(n)
    if os.path.exists(dst): os.remove(dst)
    try: os.link(src,dst); print(f"[hardlink] {n}",flush=True)
    except Exception as e: shutil.copy2(src,dst); print(f"[copy] {n} ({e})",flush=True)
print(f"[ALL LAYERS DONE] {(time.time()-t0)/60:.1f} min -> {NEW}",flush=True)
