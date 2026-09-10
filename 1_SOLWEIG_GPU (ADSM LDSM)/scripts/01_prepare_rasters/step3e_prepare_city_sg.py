"""Step3e: citywide SG (44k x 27k) ADSM preprocessing -- rasterio implementation (pyenv).
(1) rasterize ADSM (bld_h) / ADSMB (arc_h) (one full-extent layer ~4.8 GB in memory, written with LZW);
(2) windowed: thin-box protection + carved DSM + nodata cleaning. Output to TIF/ (SUB_SG_Polygon_* naming)."""
import time
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.enums import MergeAlg

TIF=r"D:\Claude\SVI_FFW\TIF"
GPKG=r"D:\Claude\SVI_FFW\output\detection_geomfirst\step2_arcade_sg.gpkg"
DSM=f"{TIF}/SUB_SG_Polygon_DSM_1m.tif"; DEM=f"{TIF}/SUB_SG_Polygon_DEM_1m.tif"
TOP=f"{TIF}/SUB_SG_Polygon_ADSM_1m.tif"; BAS=f"{TIF}/SUB_SG_Polygon_ADSMB_1m.tif"
CRV=f"{TIF}/SUB_SG_Polygon_DSMcarved_1m.tif"

t0=time.time()
with rasterio.open(DSM) as r:
    meta=r.meta.copy(); transform=r.transform; shape=(r.height,r.width); crs=r.crs
W,H=shape[1],shape[0]; print(f"grid {W}x{H}",flush=True)

arc=gpd.read_file(GPKG).to_crs(crs)
print(f"arcade polygons: {len(arc)}",flush=True)
meta.update(dtype="float32",count=1,nodata=None,compress="lzw",tiled=True,BIGTIFF="YES")

for attr,dst in [("bld_h",TOP),("arc_h",BAS)]:
    arr=rasterize([(g,float(v)) for g,v in zip(arc.geometry,arc[attr])],
                  out_shape=shape,transform=transform,fill=0.0,dtype="float32",
                  merge_alg=MergeAlg.replace)
    with rasterio.open(dst,"w",**meta) as o: o.write(arr,1)
    print(f"rasterized {attr} -> {dst} ({time.time()-t0:.0f}s)",flush=True)
    del arr

# Windowed: thin-box protection (rewrites BAS) + carve-out (writes CRV) + nodata cleaning
with rasterio.open(DSM) as rd, rasterio.open(DEM) as re_, rasterio.open(TOP) as rt, rasterio.open(BAS,"r+") as rb:
    with rasterio.open(CRV,"w",**meta) as rc:
        nstrip=0; drops=[]
        CH=1024
        for y0 in range(0,H,CH):
            h=min(CH,H-y0); win=Window(0,y0,W,h)
            dsm=rd.read(1,window=win).astype("float32"); dem=re_.read(1,window=win).astype("float32")
            top=rt.read(1,window=win); base=rb.read(1,window=win)
            dsm=np.where((dsm<-1e30)|~np.isfinite(dsm),0,dsm)
            dem=np.where((dem<-1e30)|~np.isfinite(dem)|(dem<-100),0,dem)
            strip=top>0
            base=np.where(strip,np.minimum(base,np.maximum(top-0.5,0.5)),0).astype("float32")
            rb.write(base,1,window=win)
            carved=np.where(strip,dem,dsm).astype("float32")
            rc.write(carved,1,window=win)
            nstrip+=int(strip.sum())
            if strip.any(): drops.append(float(np.median((dsm-carved)[strip])))
            if (y0//CH)%6==0: print(f"  rows {y0+h}/{H} ({time.time()-t0:.0f}s)",flush=True)
print(f"[done] strip pixels {nstrip} | carved median (median of block medians) {np.median(drops):.1f} m | total time {time.time()-t0:.0f}s",flush=True)
print(f"wrote {TOP}\nwrote {BAS}\nwrote {CRV}",flush=True)
