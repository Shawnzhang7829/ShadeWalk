# -*- coding: utf-8 -*-
"""Per-edge "no facility" shade fraction esn (for the topology map / scenario comparison): building + tree shadow (SUB_SG_BLDTREE) sampled every 2 m along each edge,
BREM pixels fully shaded (mirrors the step4_4b basis: full_sh[brem>0]=0, then the mean).
Row order = step4_4_edges_flow_SG comp==0 (strictly aligned with the inlined es/efac/esrc). Output uint8 0-100. pyenv."""
import geopandas as gpd, numpy as np, rasterio, shapely, time
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
BT  =f"{OUT}\\SUB_SG_BLDTREE_SHADOW_h14_1m.tif"
BREM=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\SUB_SG_Polygon_BREMAIN_1m.tif"
t0=time.time()
e=gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg"); e=e[e['comp']==0].reset_index(drop=True)
n=len(e); print(f"edges {n} | {time.time()-t0:.0f}s",flush=True)
with rasterio.open(BT) as ds:
    bt=ds.read(1); T=ds.transform; H,W=bt.shape
bt=np.where(bt==255,1,bt).astype(np.uint8)          # nodata->lit
with rasterio.open(BREM) as ds: brem=ds.read(1)
bt[brem>0]=0                                        # through building interior -> fully shaded (pixel level, same as 4b)
del brem
print(f"raster loaded {bt.shape} | {time.time()-t0:.0f}s",flush=True)
seg=e.geometry.segmentize(2.0)
xy,idx=shapely.get_coordinates(seg,return_index=True)
cc=(xy[:,0]-T.c).astype(np.int64); rr=(T.f-xy[:,1]).astype(np.int64)
ok=(cc>=0)&(cc<W)&(rr>=0)&(rr<H)
sh=np.zeros(len(xy),np.float64); sh[ok]=1.0-bt[rr[ok],cc[ok]]
sums=np.bincount(idx[ok],weights=sh[ok],minlength=n); cnts=np.bincount(idx[ok],minlength=n)
esn=np.zeros(n); m=cnts>0; esn[m]=sums[m]/cnts[m]
ESN=np.clip(np.round(esn*100),0,100).astype(np.uint8)
np.save(f"{OUT}\\edge_shade_nofac_SG.npy",ESN)
print(f"sample points {len(xy):,} (valid {int(ok.sum()):,}) | {time.time()-t0:.0f}s",flush=True)
# check: length-weighted means of es (shade_full, with facilities) vs esn (no facilities) per facility class -- arcade/linkway classes should drop sharply, open-air classes should stay roughly unchanged
EFAC=np.load(f"{OUT}\\edge_facility_SG.npy")
ES=np.clip(np.round(np.nan_to_num(e['shade_full'].values)*100),0,100)
L=e['length'].values
print("class         n     es(with fac) esn(no fac)   diff")
for cls,nm in [(3,'arcade'),(4,'linkway'),(5,'indoor'),(2,'tree'),(1,'bldg'),(0,'sun')]:
    mm=EFAC==cls
    a=np.average(ES[mm],weights=L[mm]); b=np.average(ESN[mm],weights=L[mm])
    print(f"{nm:8s} {int(mm.sum()):6d}   {a:6.1f}      {b:6.1f}    {b-a:+6.1f}",flush=True)
a=np.average(ES,weights=L); b=np.average(ESN,weights=L)
print(f"network    {n}   {a:6.1f}      {b:6.1f}    {b-a:+6.1f}",flush=True)
print("DONE",flush=True)
