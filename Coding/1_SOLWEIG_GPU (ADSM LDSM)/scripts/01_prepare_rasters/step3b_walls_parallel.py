"""Recompute walls/aspect on 12 cores in parallel (carved DSM, single 4000^2 tile).
Bit-for-bit equivalent to the in-package algorithm:
  - findwalls: same mathematics (4-neighbourhood max minus centre, threshold 3.0 m, borders zeroed), vectorised implementation;
  - aspect: calls the in-package filter1Goodwin_as_aspect_v3 directly on 12 row bands (halo=16 px >= filter radius 4+1),
    keeping only the interior rows of each band -> identical to the serial full-image result.
Outputs testrun/SG_WALLS_1m_test.tif + SG_ASPECT_1m_test.tif for direct use by thermal_comfort (skips the in-package recomputation).
Environment: QGIS python (run_qgis.bat).
"""
import sys, time
import numpy as np
from osgeo import gdal
from concurrent.futures import ProcessPoolExecutor

PKG=r"D:\GitHub\SOLWEIG-GPU"
sys.path.insert(0, PKG)
# Usage: python step3b_walls_parallel.py [DSM path] [output walls] [output aspect] [n processes]
_a=sys.argv
DSM=_a[1] if len(_a)>1 else r"D:\Claude\SVI_FFW\output\step3_adsm\testrun\SG_DSM_carved_1m_test.tif"
OUT_W=_a[2] if len(_a)>2 else r"D:\Claude\SVI_FFW\output\step3_adsm\testrun\SG_WALLS_1m_test.tif"
OUT_A=_a[3] if len(_a)>3 else r"D:\Claude\SVI_FFW\output\step3_adsm\testrun\SG_ASPECT_1m_test.tif"
NPROC=int(_a[4]) if len(_a)>4 else 12
WALLLIMIT=3.0; HALO=16

def findwalls_fast(a, walllimit):
    """Vectorised version bit-for-bit equivalent to walls_aspect.findwalls (4-neighbourhood cross)."""
    nb=np.full(a.shape,-np.inf,dtype=np.float64)
    nb[1:,:]=np.maximum(nb[1:,:],a[:-1,:]); nb[:-1,:]=np.maximum(nb[:-1,:],a[1:,:])
    nb[:,1:]=np.maximum(nb[:,1:],a[:,:-1]); nb[:,:-1]=np.maximum(nb[:,:-1],a[:,1:])
    walls=nb-a
    walls[walls<walllimit]=0
    walls[:,0]=0; walls[:,-1]=0; walls[0,:]=0; walls[-1,:]=0
    return walls

def band_aspect(args):
    k,r0,r1,e0,e1,scale=args
    import sys as _s; _s.path.insert(0,PKG)
    import numpy as _np
    from osgeo import gdal as _g
    from solweig_gpu.walls_aspect import filter1Goodwin_as_aspect_v3
    ds=_g.Open(DSM); a=ds.GetRasterBand(1).ReadAsArray().astype(_np.float32); ds=None
    w=findwalls_fast(a,WALLLIMIT)
    yb=filter1Goodwin_as_aspect_v3(w[e0:e1,:],scale,a[e0:e1,:])
    return k,r0,r1,yb[r0-e0:r1-e0,:]

def main():
    t0=time.time()
    ds=gdal.Open(DSM); gt=ds.GetGeoTransform(); proj=ds.GetProjection()
    a=ds.GetRasterBand(1).ReadAsArray().astype(np.float32); ds=None
    H,W=a.shape; scale=1/gt[1]
    print(f"DSM {W}x{H}, scale={scale}",flush=True)
    walls=findwalls_fast(a,WALLLIMIT)
    print(f"findwalls done {time.time()-t0:.0f}s, wall px={int((walls>0).sum())}",flush=True)
    jobs=[]
    for k in range(NPROC):
        r0=k*H//NPROC; r1=(k+1)*H//NPROC
        e0=max(0,r0-HALO); e1=min(H,r1+HALO)
        jobs.append((k,r0,r1,e0,e1,scale))
    y=np.zeros((H,W),dtype=np.float64)
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        for k,r0,r1,yb in ex.map(band_aspect,jobs):
            y[r0:r1,:]=yb
            print(f"  band {k+1}/{NPROC} done ({time.time()-t0:.0f}s)",flush=True)
    drv=gdal.GetDriverByName('GTiff')
    for path,arr in [(OUT_W,walls),(OUT_A,y)]:
        o=drv.Create(path,W,H,1,gdal.GDT_Float32)
        o.SetGeoTransform(gt); o.SetProjection(proj)
        o.GetRasterBand(1).WriteArray(arr); o.FlushCache(); o=None
        print(f"wrote {path}",flush=True)
    print(f"[walls parallel done] {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
