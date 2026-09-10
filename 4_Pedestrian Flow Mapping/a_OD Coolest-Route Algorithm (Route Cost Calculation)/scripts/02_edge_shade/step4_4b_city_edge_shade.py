# -*- coding: utf-8 -*-
"""Step4-4b city-wide: per-edge 14:00 shade fraction (full system vs LOD1 buildings only). Both shadow rasters are reduced to uint8 in memory,
sampled every 2 m along each edge (segmentize). Output edges/nodes (with u, v, comp, shade_full, shade_bld). pyenv."""
import numpy as np, geopandas as gpd, rasterio, time
from rasterio.windows import from_bounds, Window
from shapely.geometry import Point
from scipy.spatial import cKDTree
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
FULL=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\merge_images\Shadow_2pm_h14.tif"   # new arcade shadow (2026-06)
BLD =f"{OUT}\\SUB_SG_BUILDING_SHADOW_h14_1m.tif"
BREM=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\SUB_SG_Polygon_BREMAIN_1m.tif"  # edges passing through building interiors (building_remain, same grid as FULL) are treated as fully shaded (old TIF\ archived)
t0=time.time()
net=gpd.read_file(f"{OUT}\\step4_network_final_prep.gpkg")   # new network (comp computed + src mapped to footpath/shade/bridge)
segs=list(net.geometry.values); N=len(segs)
print(f"edges {N} | read {time.time()-t0:.0f}s",flush=True)

# nodes (endpoint clustering)
ep=np.empty((2*N,2))
for i,s in enumerate(segs):
    c=s.coords; ep[2*i]=c[0][:2]; ep[2*i+1]=c[-1][:2]
tr=cKDTree(ep); par=np.arange(2*N)
def find(x):
    r=x
    while par[r]!=r: r=par[r]
    while par[x]!=r: par[x],x=r,par[x]
    return r
for i,j in tr.query_pairs(0.5):
    ri,rj=find(i),find(j)
    if ri!=rj: par[ri]=rj
root2node={}; nid=np.empty(2*N,dtype=np.int64)
for k in range(2*N): nid[k]=root2node.setdefault(find(k),len(root2node))
nn=len(root2node); node_xy=np.zeros((nn,2)); cnt=np.zeros(nn)
for k in range(2*N): node_xy[nid[k]]+=ep[k]; cnt[nid[k]]+=1
node_xy/=cnt[:,None]
uu=nid[0::2].astype(int); vv=nid[1::2].astype(int)
print(f"nodes {nn} | {time.time()-t0:.0f}s",flush=True)

# both shadow rasters -> uint8 in memory (1=lit, 0=shaded)
with rasterio.open(FULL) as ds:
    W,H=ds.width,ds.height; T=ds.transform
    full_sh=np.full((H,W),100,np.uint8)  # lit fraction x100: 0=fully shaded, 10=tree shade (0.1), 100=full sun
    blk=2000
    for r in range(0,H,blk):
        h=min(blk,H-r); a=ds.read(1,window=Window(0,r,W,h))
        full_sh[r:r+h]=np.clip(np.nan_to_num(a,nan=1.0)*100.0,0,255).astype(np.uint8)
print(f"full_sh loaded {time.time()-t0:.0f}s",flush=True)
with rasterio.open(BLD) as ds:
    bld_sh=ds.read(1)
    bld_sh=np.where(bld_sh==255,1,bld_sh).astype(np.uint8)   # nodata->lit
print(f"bld_sh loaded {time.time()-t0:.0f}s",flush=True)
with rasterio.open(BREM) as ds: brem=ds.read(1)            # building_remain mask (0/1)
full_sh[brem>0]=0; bld_sh[brem>0]=0                        # through building interior -> fully shaded (lit=0, same as under an artificial facility canopy)
del brem
print(f"building_remain merged (interior fully shaded) {time.time()-t0:.0f}s",flush=True)
left=T.c; top=T.f

# per-edge 2 m sampling
sf_full=np.full(N,np.nan); sf_bld=np.full(N,np.nan)
for i,s in enumerate(segs):
    d=s.segmentize(2.0); xy=np.asarray(d.coords)
    cc=((xy[:,0]-left)).astype(np.int64); rr=((top-xy[:,1])).astype(np.int64)
    ok=(cc>=0)&(cc<W)&(rr>=0)&(rr<H)
    if ok.sum()==0: continue
    fv=full_sh[rr[ok],cc[ok]]; bv=bld_sh[rr[ok],cc[ok]]
    sf_full[i]=float((1.0-fv/100.0).mean()); sf_bld[i]=float((1.0-bv).mean())
    if (i+1)%40000==0: print(f"  sampled {i+1}/{N} ({time.time()-t0:.0f}s)",flush=True)
del full_sh,bld_sh

net['u']=uu; net['v']=vv; net['shade_full']=sf_full; net['shade_bld']=sf_bld
net.to_file(f"{OUT}\\step4_4_edges_SG.gpkg",driver="GPKG")
nodes=gpd.GeoDataFrame({'node':np.arange(nn),'geometry':[Point(*node_xy[k]) for k in range(nn)]},crs=3414)
nodes.to_file(f"{OUT}\\step4_4_nodes_SG.gpkg",driver="GPKG")

L=net['length'].values; ok=np.isfinite(sf_full)&np.isfinite(sf_bld)
def wsh(mask):
    m=ok&mask; return np.sum(L[m]*sf_full[m])/L[m].sum(), np.sum(L[m]*sf_bld[m])/L[m].sum(), L[m].sum()/1000
Wf,Wb,Lk=wsh(np.ones(N,bool))
Wfm,Wbm,Lkm=wsh(net.comp.values==0)
print(f"\ncity-wide network 14:00 length-weighted shade fraction (all {Lk:,.0f} km):")
print(f"  full system {100*Wf:.1f}% | LOD1 buildings only {100*Wb:.1f}% | gain {Wf/max(Wb,1e-9):.1f}x (+{100*(Wf-Wb):.1f}pp)")
print(f"main component ({Lkm:,.0f} km): full {100*Wfm:.1f}% | buildings only {100*Wbm:.1f}% | {Wfm/max(Wbm,1e-9):.1f}x")
print(f"saved edges/nodes SG | {time.time()-t0:.0f}s",flush=True)
