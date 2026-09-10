# -*- coding: utf-8 -*-
"""Stage 2a facility-internal -> whole lines [bridge 5 m gaps first, then remove linkway spurs < 3 m; arcade spurs are kept].
(1) degree-1 endpoint pairs < 5 m are bridged across connected components (re-joins breaks within a facility + end-to-end joins between different facilities; deflection angle >= 120 deg, i.e. straight, preferred);
(2) remove dead-end spurs < 3 m -- linkway only (Voronoi centrelines are irregular and spiky); arcade (runs-offset centrelines) is clean by itself and is not pruned.
   Topology check after bridging: a < 3 m segment with a degree-1 end (connected to only one segment) is deleted / a < 3 m segment connected at both ends (joined to several small segments = multi-way crossing / continuous small path) is kept (confirmation 4);
(3) each connected component = one main-axis line unit (line_id), degree-1 = exposed main-axis end (for the stage 3 connection to OSM). pyenv."""
import numpy as np, geopandas as gpd, time, collections, math
from scipy.spatial import cKDTree
from shapely.geometry import LineString
OUT=r"D:\Claude\SVI_FFW\output\step4_network"
DEL=3.0; D5=5.0; CLUST=0.5; ANG=120.0
t=time.time()
g=gpd.read_file(f"{OUT}\\step4_1_shade_centerline_SG.gpkg")
segs=list(g.geometry.values); fac=list(g['fac_id'].values); src=list(g['src'].values); N=len(segs)
print(f"Read centreline {N} segments | facilities {len(set(fac))} | total length {g.geometry.length.sum()/1000:.1f}km | {time.time()-t:.0f}s",flush=True)

def cluster(geoms):
    M=len(geoms); ep=np.empty((2*M,2))
    for i,s in enumerate(geoms): c=s.coords; ep[2*i]=c[0][:2]; ep[2*i+1]=c[-1][:2]
    par=np.arange(2*M)
    def f(x):
        r=x
        while par[r]!=r: r=par[r]
        while par[x]!=r: par[x],x=r,par[x]
        return r
    for i,j in cKDTree(ep).query_pairs(CLUST):
        a,b=f(i),f(j)
        if a!=b: par[a]=b
    root={}; nid=np.empty(2*M,np.int64)
    for k in range(2*M): nid[k]=root.setdefault(f(k),len(root))
    NN=len(root); xy=np.zeros((NN,2)); cnt=np.zeros(NN)
    for k in range(2*M): xy[nid[k]]+=ep[k]; cnt[nid[k]]+=1
    xy/=cnt[:,None]
    return NN,xy,[(int(nid[2*i]),int(nid[2*i+1])) for i in range(M)]
def tangent(geom,p):
    c=list(geom.coords); a=np.array(c[0][:2]); b=np.array(c[-1][:2]); pa=np.array(p)
    if np.hypot(*(a-pa))<=np.hypot(*(b-pa)): d=np.array(c[1][:2])-a
    else: d=np.array(c[-2][:2])-b
    n=np.hypot(*d); return d/n if n>1e-9 else np.array([1.0,0.0])
def straight(d_seg,p,pp):
    link=np.array(pp)-np.array(p); n=np.hypot(*link)
    if n<1e-9: return 0.0
    link=link/n; cv=max(-1.0,min(1.0,float(-d_seg[0]*link[0]-d_seg[1]*link[1])))
    return math.degrees(math.acos(cv))

# (1) bridge 5 m gaps first (degree-1 endpoint pairs, straight >= 120 deg preferred, across connected components) -- based on the original centreline topology
nn,xy,sn=cluster(segs)
deg=collections.Counter()
for a,b in sn:
    if a!=b: deg[a]+=1; deg[b]+=1
node_seg={}
for si,(a,b) in enumerate(sn):
    if a==b: continue
    for nd in (a,b):
        if deg[nd]==1: node_seg[nd]=si
par=np.arange(nn)
def find(x):
    r=x
    while par[r]!=r: r=par[r]
    while par[x]!=r: par[x],x=r,par[x]
    return r
for a,b in sn:
    if a!=b and find(a)!=find(b): par[find(a)]=find(b)
d1=[nd for nd in range(nn) if deg[nd]==1 and nd in node_seg]
d1xy=xy[np.array(d1)] if d1 else np.zeros((0,2))
from shapely import STRtree
seg_tree=STRtree(segs)   # original facility centrelines (before bridging), used to test whether a facconn doubles back
def overlap_fac(ln, exclude):
    """A bridging line whose overlap with OTHER facility centrelines exceeds 50% = doubles back / loops around, reject (rule 6).
    Exclude the centreline segments that own the two endpoints (the endpoints are attached to them anyway; otherwise collinear continuation joins between adjacent arcade segments would be misjudged)."""
    lb=ln.buffer(0.6); ov=0.0
    for si in seg_tree.query(lb):
        si=int(si)
        if si in exclude: continue
        try:
            ov+=segs[si].intersection(lb).length
            if ov>ln.length*0.5: return True
        except Exception: pass
    return False
cand=[]
for ia,ib in cKDTree(d1xy).query_pairs(D5):
    na,nb=d1[ia],d1[ib]
    if find(na)==find(nb): continue
    d=float(np.hypot(*(xy[na]-xy[nb])))
    if not (0.5<d<=D5): continue
    da=tangent(segs[node_seg[na]],xy[na]); db=tangent(segs[node_seg[nb]],xy[nb])
    st=min(straight(da,xy[na],xy[nb]), straight(db,xy[nb],xy[na]))
    cand.append((st<ANG, d, na, nb))   # straight (st>=120) first, then nearest
cand.sort()
links=[]; sameF=diffF=0; nbt=0
for acute,d,na,nb in cand:
    if find(na)==find(nb): continue
    ln=LineString([tuple(xy[na]),tuple(xy[nb])])
    if overlap_fac(ln, {node_seg[na],node_seg[nb]}): nbt+=1; continue   # still >50% overlap after excluding the endpoint segments = true loop-back, reject (rule 6)
    par[find(na)]=find(nb); links.append((na,nb))
    if fac[node_seg[na]]==fac[node_seg[nb]]: sameF+=1
    else: diffF+=1
print(f"(1) bridged <5m endpoint pairs: {len(links)} (same-facility re-joins {sameF} / inter-facility joins {diffF} / double-back rejected {nbt}) | {time.time()-t:.0f}s",flush=True)
# merge the bridging edges into the network (facconn), then proceed to step (2) spur removal
segs=list(segs)+[LineString([tuple(xy[na]),tuple(xy[nb])]) for na,nb in links]
src=list(src)+['facconn']*len(links)
fac=list(fac)+[fac[node_seg[na]] for na,nb in links]
N=len(segs)

# (2) remove dead-end spurs < 3 m -- linkway only (arcade = runs-offset centreline, clean, not pruned); topology check after bridging: < 3 m with a degree-1 end deleted / multi-way crossing (< 3 m connected at both ends) kept
keep=list(range(N)); it=0
while True:
    it+=1; cur=[segs[i] for i in keep]; csrc=[src[i] for i in keep]
    nn,xy,sn=cluster(cur)
    deg=collections.Counter()
    for a,b in sn:
        if a!=b: deg[a]+=1; deg[b]+=1
    drop=set(i for i,(a,b) in enumerate(sn) if a!=b and csrc[i]=='linkway' and cur[i].length<DEL and (deg[a]==1 or deg[b]==1))
    if not drop: break
    keep=[keep[i] for i in range(len(keep)) if i not in drop]
    print(f"  (2) remove linkway <3m dead ends, round {it}: -{len(drop)} -> remaining {len(keep)} | {time.time()-t:.0f}s",flush=True)
segs=[segs[i] for i in keep]; src=[src[i] for i in keep]; fac=[fac[i] for i in keep]; N=len(segs)
print(f"After linkway spur removal {N} segments (all arcade kept) | {time.time()-t:.0f}s",flush=True)

# (3) output line_id (whole-line unit) + exposed-end statistics
og=list(segs); osrc=list(src); ofac=list(fac)
nn2,xy2,sn2=cluster(og)
p2=np.arange(nn2)
def f2(x):
    r=x
    while p2[r]!=r: r=p2[r]
    while p2[x]!=r: p2[x],x=r,p2[x]
    return r
for a,b in sn2:
    if a!=b and f2(a)!=f2(b): p2[f2(a)]=f2(b)
comp=[f2(sn2[i][0]) for i in range(len(og))]; ncomp=len(set(comp))
_u={c:k for k,c in enumerate(sorted(set(comp)))}; line_id=[f"L{_u[c]}" for c in comp]
out=gpd.GeoDataFrame({'src':osrc,'fac_id':ofac,'line_id':line_id,'geometry':og},crs=3414)
out['length']=out.geometry.length
out.to_file(f"{OUT}\\step4_fac_lines.gpkg",driver="GPKG")
d2=collections.Counter()
for a,b in sn2:
    if a!=b: d2[a]+=1; d2[b]+=1
n_end=sum(1 for nd in range(nn2) if d2[nd]==1)
print(f"Done {len(out)} segments | total length {out.length.sum()/1000:.1f}km | connected components {ncomp} | exposed main-axis ends (degree-1) {n_end} | {time.time()-t:.0f}s",flush=True)
