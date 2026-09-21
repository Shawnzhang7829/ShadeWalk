# -*- coding: utf-8 -*-
"""v4 step4_network_final -> nav prep: (1) comp (endpoint clustering 0.5 m -> connected components, main component = 0, others = 1) (2) src mapping
(osm/osmlink -> footpath, arcade/linkway/facconn -> shade, fac2osm/gapfill -> bridge). Writes step5_nav_webapp\\step4_network_final_prep.gpkg (read by step4_4b). pyenv."""
import geopandas as gpd, numpy as np, collections, time
from scipy.spatial import cKDTree
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
FINAL=r"D:\Claude\SVI_FFW\output\step4_network\step4_network_final.gpkg"
t=time.time()
g=gpd.read_file(FINAL)
try: g=g.to_crs(3414)
except Exception: g=g.set_crs(3414,allow_override=True)
segs=list(g.geometry.values); N=len(segs)
print(f"read {N} segments | {time.time()-t:.0f}s",flush=True)
ep=np.empty((2*N,2))
for i,s in enumerate(segs): c=s.coords; ep[2*i]=c[0][:2]; ep[2*i+1]=c[-1][:2]
par=np.arange(2*N)
def fnd(x):
    r=x
    while par[r]!=r: r=par[r]
    while par[x]!=r: par[x],x=r,par[x]
    return r
for i,j in cKDTree(ep).query_pairs(0.5):
    a,b=fnd(i),fnd(j)
    if a!=b: par[a]=b
nid=np.array([fnd(k) for k in range(2*N)])
roots=np.unique(nid); ri={int(r):i for i,r in enumerate(roots)}; pr=np.arange(len(roots))
def fr(x):
    r=x
    while pr[r]!=r: r=pr[r]
    while pr[x]!=r: pr[x],x=r,pr[x]
    return r
for i in range(N):
    a,b=ri[int(nid[2*i])],ri[int(nid[2*i+1])]
    if a!=b and fr(a)!=fr(b): pr[fr(a)]=fr(b)
seg_block=np.array([fr(ri[int(nid[2*i])]) for i in range(N)])
clen=collections.defaultdict(float)
for i in range(N): clen[seg_block[i]]+=segs[i].length
main=max(clen,key=clen.get); comp=(seg_block!=main).astype(int)
print(f"connected components {len(clen)} | main component {100*clen[main]/sum(clen.values()):.1f}% | {time.time()-t:.0f}s",flush=True)
SRC={'osm':'footpath','osmlink':'footpath','arcade':'shade','linkway':'shade','facconn':'shade','fac2osm':'bridge','gapfill':'bridge'}
g['src']=g['src'].astype(str).map(SRC).fillna('footpath')
g['comp']=comp; g['length']=g.geometry.length
g.to_file(OUT+r"\step4_network_final_prep.gpkg",driver="GPKG")
print(f"src {g['src'].value_counts().to_dict()} | comp main {int((comp==0).sum())}/other {int((comp==1).sum())} | DONE {time.time()-t:.0f}s",flush=True)
