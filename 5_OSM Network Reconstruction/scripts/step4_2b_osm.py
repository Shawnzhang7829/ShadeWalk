# -*- coding: utf-8 -*-
"""Stage 2 OSM-internal: read the OSM source (Highway_OSM.gpkg, confirmation 1; already EPSG:3414), degree-1 endpoints -> connect to the nearest segment < 15 m
(endpoint to line body, target segment split at the foot of the perpendicular), joining broken lines into continuous ones. Shortest first, no redundancy. Output step4_osm_lines.gpkg. pyenv."""
import numpy as np, geopandas as gpd, time, collections
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely import STRtree, force_2d
OUT=r"D:\Claude\SVI_FFW\output\step4_network"
OSM=r"D:\Claude\SVI_FFW\Shp\SG\Pedestrian route\OSM\Highway_OSM.gpkg"
EXCL_HW={'motorway','trunk','motorway_link','trunk_link'}   # motor-vehicle-only roads (pedestrians legally prohibited); source switched to Highway_OSM on 2026-06-24 and filtered to pedestrian-accessible ways
D15=15.0; CLUST=0.5
t=time.time()
g=gpd.read_file(OSM)
try: g=g.to_crs(3414)
except Exception: g=g.set_crs(3414,allow_override=True)
n0=len(g); hw=g['highway'].astype(str); mask=~hw.isin(EXCL_HW)
if 'foot' in g.columns: mask=mask & (g['foot'].astype(str)!='no')
g=g[mask].reset_index(drop=True)
print(f"Pedestrian-accessible filter: {n0} -> {len(g)} segments (excluding motor-vehicle-only roads + foot=no) | {time.time()-t:.0f}s",flush=True)
g['geometry']=g.geometry.apply(force_2d)
g=g.explode(index_parts=False).reset_index(drop=True)
segs=[s for s in g.geometry.values if s.geom_type=='LineString' and s.length>0.3]
N=len(segs)
print(f"Read OSM (filtered) {N} segments | total length {sum(s.length for s in segs)/1000:.0f}km | {time.time()-t:.0f}s",flush=True)

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

nn,xy,sn=cluster(segs)
deg=collections.Counter()
for a,b in sn:
    if a!=b: deg[a]+=1; deg[b]+=1
node_seg={}
for si,(a,b) in enumerate(sn):
    if a==b: continue
    for nd in (a,b):
        if deg[nd]==1: node_seg[nd]=si
d1=[nd for nd in range(nn) if deg[nd]==1 and nd in node_seg]
print(f"Nodes {nn} | degree-1 endpoints {len(d1)} | {time.time()-t:.0f}s",flush=True)
tree=STRtree(segs)
links=[]; splits=collections.defaultdict(list)
for ii,nd in enumerate(d1):
    if (ii+1)%20000==0: print(f"  connection progress {ii+1}/{len(d1)} | {time.time()-t:.0f}s",flush=True)
    p=xy[nd]; my=node_seg[nd]; pPt=Point(p); best=(1e9,None,None)
    for ci in tree.query(pPt.buffer(D15)):
        sj=int(ci)
        if sj==my: continue
        d=segs[sj].distance(pPt)
        if 0.5<d<best[0] and d<=D15:
            pp=segs[sj].interpolate(segs[sj].project(pPt)); best=(d,sj,(pp.x,pp.y))
    d,sj,pp=best
    if sj is not None: links.append((p,pp)); splits[sj].append(Point(pp))
print(f"<15m connections: {len(links)} | {time.time()-t:.0f}s",flush=True)
def split_line(line,pts):
    L=line.length; cuts=sorted(set(round(line.project(pp),2) for pp in pts)); cuts=[c for c in cuts if 0.2<c<L-0.2]
    if not cuts: return [line]
    bs=[0.0]+cuts+[L]; out=[]
    for k in range(len(bs)-1):
        if bs[k+1]-bs[k]>0.2:
            try: out.append(substring(line,bs[k],bs[k+1]))
            except Exception: pass
    return out if out else [line]
og=[]; osrc=[]
for i in range(N):
    if i in splits:
        for sub in split_line(segs[i],splits[i]): og.append(sub); osrc.append('osm')
    else: og.append(segs[i]); osrc.append('osm')
for p,pp in links: og.append(LineString([tuple(p),pp])); osrc.append('osmlink')
out=gpd.GeoDataFrame({'src':osrc,'geometry':og},crs=3414); out['length']=out.geometry.length
out.to_file(f"{OUT}\\step4_osm_lines.gpkg",driver="GPKG")
nn2,xy2,sn2=cluster(og)
p2=np.arange(nn2)
def f2(x):
    r=x
    while p2[r]!=r: r=p2[r]
    while p2[x]!=r: p2[x],x=r,p2[x]
    return r
for a,b in sn2:
    if a!=b and f2(a)!=f2(b): p2[f2(a)]=f2(b)
nc=len(set(f2(sn2[i][0]) for i in range(len(og))))
print(f"Done {len(out)} segments | total length {out.length.sum()/1000:.0f}km | connected components {nc} (fewer = more continuous) | {time.time()-t:.0f}s",flush=True)
