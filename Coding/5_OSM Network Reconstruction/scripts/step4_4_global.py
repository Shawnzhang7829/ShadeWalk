# -*- coding: utf-8 -*-
"""Stage 4 global fallback: read step4_connected, every degree-1 endpoint < 15 m connects to the nearest segment (endpoint to line body, cross-component links preferred so islands get attached,
no crossing of real buildings -- allowed only where a complete OSM line already runs under the building). Output step4_network_final.gpkg. pyenv."""
import numpy as np, geopandas as gpd, time, collections
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely import STRtree
OUT=r"D:\Claude\SVI_FFW\output\step4_network"
BLDG=r"D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg"
D15=15.0; CLUST=0.5; CTOL=2.0
t=time.time()
g=gpd.read_file(f"{OUT}\\step4_connected.gpkg")
osmf=gpd.read_file(f"{OUT}\\step4_osm_lines.gpkg")
segs=list(g.geometry.values); osrc=list(g['src'].values); N=len(segs)
print(f"Read {N} segments | {time.time()-t:.0f}s",flush=True)
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
par=np.arange(nn)
def find(x):
    r=x
    while par[r]!=r: r=par[r]
    while par[x]!=r: par[x],x=r,par[x]
    return r
for a,b in sn:
    if a!=b and find(a)!=find(b): par[find(a)]=find(b)
print(f"Nodes {nn} | degree-1 {len(d1)} | {time.time()-t:.0f}s",flush=True)
gb=gpd.read_file(BLDG)
try: gb=gb.to_crs(3414)
except Exception: gb=gb.set_crs(3414,allow_override=True)
bgeoms=list(gb.geometry.values); btree=STRtree(bgeoms)
osmf_g=list(osmf.geometry.values); osmftree=STRtree(osmf_g); _uc={}
def osm_under(bi):
    if bi in _uc: return _uc[bi]
    bg=bgeoms[bi]; tot=0.0
    for fi in osmftree.query(bg):
        try:
            tot+=osmf_g[int(fi)].intersection(bg).length
            if tot>CTOL: break
        except Exception: pass
    r=tot>CTOL; _uc[bi]=r; return r
def crosses(line):
    for bi in btree.query(line):
        bi=int(bi)
        try:
            if bgeoms[bi].intersection(line).length>CTOL:
                if osm_under(bi): continue
                return True
        except Exception: pass
    return False
tree=STRtree(segs)
links=[]; splits=collections.defaultdict(list); ncross=nsame=nblk=0
for ii,nd in enumerate(d1):
    if (ii+1)%5000==0: print(f"  fallback {ii+1}/{len(d1)} | {time.time()-t:.0f}s",flush=True)
    p=xy[nd]; my=node_seg[nd]; pPt=Point(p); cands=[]
    for ci in tree.query(pPt.buffer(D15)):
        sj=int(ci)
        if sj==my: continue
        d=segs[sj].distance(pPt)
        if 0.5<d<=D15:
            pp=segs[sj].interpolate(segs[sj].project(pPt)); xc=(find(sn[sj][0])!=find(nd))
            cands.append((not xc, d, sj, (pp.x,pp.y)))   # cross-component (xc=True -> not xc=False) first, then nearest
    if not cands: continue
    cands.sort(); ok=False
    for _,d,sj,pp in cands:
        ln=LineString([tuple(p),pp])
        if crosses(ln): continue
        links.append((p,pp)); splits[sj].append(Point(pp))
        if find(sn[sj][0])!=find(nd): ncross+=1; par[find(nd)]=find(sn[sj][0])
        else: nsame+=1
        ok=True; break
    if not ok: nblk+=1
print(f"Fallback connections: cross-component attached {ncross} | same-component filled {nsame} | building crossing rejected {nblk} | {time.time()-t:.0f}s",flush=True)
def split_line(line,pts):
    L=line.length; cuts=sorted(set(round(line.project(pp),2) for pp in pts)); cuts=[c for c in cuts if 0.2<c<L-0.2]
    if not cuts: return [line]
    bs=[0.0]+cuts+[L]; out=[]
    for k in range(len(bs)-1):
        if bs[k+1]-bs[k]>0.2:
            try: out.append(substring(line,bs[k],bs[k+1]))
            except Exception: pass
    return out if out else [line]
og=[]; ogsrc=[]
for i in range(N):
    if i in splits:
        for sub in split_line(segs[i],splits[i]): og.append(sub); ogsrc.append(osrc[i])
    else: og.append(segs[i]); ogsrc.append(osrc[i])
for p,pp in links: og.append(LineString([tuple(p),pp])); ogsrc.append('gapfill')
out=gpd.GeoDataFrame({'src':ogsrc,'geometry':og},crs=3414); out['length']=out.geometry.length
out.to_file(f"{OUT}\\step4_network_final.gpkg",driver="GPKG")
nn2,xy2,sn2=cluster(og)
p2=np.arange(nn2)
def f2(x):
    r=x
    while p2[r]!=r: r=p2[r]
    while p2[x]!=r: p2[x],x=r,p2[x]
    return r
for a,b in sn2:
    if a!=b and f2(a)!=f2(b): p2[f2(a)]=f2(b)
cl=collections.defaultdict(float)
for i in range(len(og)): cl[f2(sn2[i][0])]+=og[i].length
nc=len(cl); mainpct=100*max(cl.values())/sum(cl.values())
print(f"Done {len(out)} segments {out.length.sum()/1000:.0f}km | connected components {nc} | main component {mainpct:.1f}% | {time.time()-t:.0f}s",flush=True)
