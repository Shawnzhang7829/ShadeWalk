# -*- coding: utf-8 -*-
"""Stage 3b facility <-> OSM: (1) facility ends (degree-1) connect to the nearest OSM (< 15 m); (2) [new] every arcade corner connects to the nearest OSM (< 30 m)
-- guarantees smooth circulation between arcades and main paths (otherwise only the ends reach the main path and corners cannot step down to it). First split the arcade centrelines into nodes at corners (after simplify removes the dense Bezier
arc points, vertex angle deviating from straight by > 30 deg), then connect every arcade node (corner + end) to the nearest OSM.
Criteria: crossing a real building rejected (allowed where a complete OSM line already runs under the building) / doubling back (connection overlaps facility centrelines > 50%, excluding the origin segment) rejected / landing point under a facility polygon not connected (in_fac) / out of range not connected.
Output step4_connected.gpkg. pyenv."""
import numpy as np, geopandas as gpd, time, collections, math
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely import STRtree
OUT=r"D:\Claude\SVI_FFW\output\step4_network"
BLDG=r"D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg"
ARC=r"D:\Claude\SVI_FFW\Shp\SG\step2_arcade_sg.gpkg"
LKW=r"D:\Claude\SVI_FFW\Shp\SG\covered_linkway_SG_island_tv_pednet_bridged.gpkg"
D150=15.0; D_ARC=30.0; CLUST=0.5; CTOL=2.0; OVL=0.5     # ends 15 m / arcade corners 30 m
CORNER_ANG=150.0; CORNER_SIMP=1.2                        # corner: after simplify removes the dense arc points, vertex angle < 150 deg (deviation from straight > 30 deg)
t=time.time()
fac=gpd.read_file(f"{OUT}\\step4_fac_lines.gpkg")
osm=gpd.read_file(f"{OUT}\\step4_osm_kept.gpkg")        # connection targets (after removal)
osmf=gpd.read_file(f"{OUT}\\step4_osm_lines.gpkg")      # for the building-crossing test (complete OSM before removal)
facsegs=list(fac.geometry.values); facsrc=list(fac['src'].values)
osmsegs=list(osm.geometry.values); osmsrc=list(osm['src'].values)
print(f"Facility {len(facsegs)} | OSM kept {len(osmsegs)} | OSM complete {len(osmf)} | {time.time()-t:.0f}s",flush=True)

def _ang(p,a,b):
    v1=(a[0]-p[0],a[1]-p[1]); v2=(b[0]-p[0],b[1]-p[1]); n1=math.hypot(*v1); n2=math.hypot(*v2)
    if n1<1e-9 or n2<1e-9: return 180.0
    c=max(-1.0,min(1.0,(v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2))); return math.degrees(math.acos(c))

# === [new] split arcade centrelines at corners (corners become nodes so connections can attach) ===
def corner_cuts(geom):
    """After simplify removes the dense Bezier arc points of the arcade centreline, a vertex with angle < CORNER_ANG (deviation from straight > 30 deg) = corner; returns arc-length positions along the line."""
    s=geom.simplify(CORNER_SIMP); cc=list(s.coords); cuts=[]
    for k in range(1,len(cc)-1):
        if _ang(cc[k][:2],cc[k-1][:2],cc[k+1][:2])<CORNER_ANG:
            cuts.append(geom.project(Point(cc[k][:2])))
    return sorted(set(round(c,2) for c in cuts))
nf_s=[]; nf_c=[]; ncut=0
for geom,s in zip(facsegs,facsrc):
    if s=='arcade':
        cuts=[c for c in corner_cuts(geom) if 0.5<c<geom.length-0.5]
        if cuts:
            bs=[0.0]+cuts+[geom.length]
            for k in range(len(bs)-1):
                if bs[k+1]-bs[k]>0.3:
                    try: nf_s.append(substring(geom,bs[k],bs[k+1])); nf_c.append('arcade')
                    except Exception: pass
            ncut+=len(cuts); continue
    nf_s.append(geom); nf_c.append(s)
facsegs=nf_s; facsrc=nf_c
print(f"arcade corner split: +{ncut} break points -> facility {len(facsegs)} segments | {time.time()-t:.0f}s",flush=True)

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

nn,xy,sn=cluster(facsegs)
deg=collections.Counter()
node_segs=collections.defaultdict(list)
for si,(a,b) in enumerate(sn):
    if a!=b: deg[a]+=1; deg[b]+=1; node_segs[a].append(si); node_segs[b].append(si)
# candidates: (1) all arcade nodes (corners + ends, each connected to the nearest OSM) (2) linkway degree-1 ends (non-arcade nodes)
node_is_arc=set()
for si,(a,b) in enumerate(sn):
    if a!=b and facsrc[si]=='arcade': node_is_arc.add(a); node_is_arc.add(b)
lkw_d1=set()
for si,(a,b) in enumerate(sn):
    if a==b: continue
    for nd in (a,b):
        if deg[nd]==1 and facsrc[si]=='linkway': lkw_d1.add(nd)
arc_d1=sorted(nd for nd in node_is_arc if deg[nd]==1)        # arcade ends (degree-1)
arc_corner=sorted(nd for nd in node_is_arc if deg[nd]>=2)    # arcade corners (degree>=2: split break points / junction points)
lkw_nodes=sorted(lkw_d1-node_is_arc)                          # linkway ends
print(f"Candidates: arcade ends {len(arc_d1)} | arcade corners {len(arc_corner)} | linkway ends {len(lkw_nodes)} | {time.time()-t:.0f}s",flush=True)

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
_fp=gpd.read_file(ARC); _fl=gpd.read_file(LKW)
for _g in (_fp,_fl):
    try: _g.set_crs(3414,allow_override=True,inplace=True)
    except Exception: pass
_facpolys=[g for g in list(_fp.geometry.values)+list(_fl.geometry.values) if g is not None and not g.is_empty]
_factree=STRtree(_facpolys)
def in_fac(xy_pt):                                       # landing point under a facility polygon -> do not connect (should use the facility centreline); STRtree query of nearby polygons + contains, avoids the ~33 s union of 16224 polys
    pt=Point(xy_pt)
    for i in _factree.query(pt):
        try:
            if _facpolys[int(i)].contains(pt): return True
        except Exception: pass
    return False
print(f"Facility polygon STRtree (in_fac criterion): {len(_facpolys)} polys | {time.time()-t:.0f}s",flush=True)
factree=STRtree(facsegs)
def backtrack(line, exclude):
    """A connection overlapping OTHER facility centrelines by > 50% = doubles back / loops around, reject. Exclude the segments owning the origin node (corners/ends lie on the centreline anyway; the inherent overlap at the origin is not doubling back)."""
    lb=line.buffer(0.6); ov=0.0
    for fi in factree.query(lb):
        fi=int(fi)
        if fi in exclude: continue
        try:
            ov+=facsegs[fi].intersection(lb).length
            if ov>line.length*OVL: return True
        except Exception: pass
    return False

otree=STRtree(osmsegs)
links=[]; splits=collections.defaultdict(list); nfar=nblk=nclash=0
DCLASH=4.0
_end_tree=[None]; _corner_pp=[]
def clash(pp):
    """Landing point pp within DCLASH of any existing landing point = clashes (overlaps) with an existing connection, skip."""
    if _end_tree[0] is not None and len(_end_tree[0].query_ball_point(pp, DCLASH))>0: return True
    return any((pp[0]-q[0])**2+(pp[1]-q[1])**2 < DCLASH*DCLASH for q in _corner_pp)
def connect(nd, maxd, anti=False):
    """From nd, connect perpendicularly (preferring a foot point inside the OSM segment) to the nearest connectable OSM line; with anti=True also avoid overlapping (clashing with) existing landing points. Returns True on success."""
    global nfar,nblk,nclash
    p=xy[nd]; pPt=Point(p); cands=[]
    for ci in otree.query(pPt.buffer(maxd)):
        sj=int(ci); seg=osmsegs[sj]; L=seg.length
        proj=seg.project(pPt); pp=seg.interpolate(proj); d=pPt.distance(pp)
        if 0.5<d<=maxd:
            perp=0 if (0.5<proj<L-0.5) else 1              # foot point inside the OSM segment = truly perpendicular, sorted first; beyond an endpoint = non-perpendicular fallback
            cands.append((perp,d,sj,(pp.x,pp.y)))
    if not cands: nfar+=1; return False
    cands.sort(); exc=set(node_segs[nd])                   # perpendicular first, then nearest; take the first connectable one (no real-building crossing / no doubling back / not landing under a facility)
    blocked_clash=False
    for perp,d,sj,pp in cands:
        ln=LineString([tuple(p),pp])
        if crosses(ln) or backtrack(ln,exc): continue
        if in_fac(pp): continue
        if anti and clash(pp): blocked_clash=True; continue   # avoid clashing with existing connections (overlapping landing points)
        links.append((p,pp)); splits[sj].append(Point(pp))
        if anti: _corner_pp.append(pp)
        return True
    if blocked_clash: nclash+=1
    else: nblk+=1
    return False
# Step 1: all ends (arcade ends + linkway ends) connect to OSM -> record the nodes "already attached to a main path" (NOTE: fixed decision baseline, not updated during the corner stage)
osm_node_end=set()
for nd in arc_d1:
    if connect(nd, D_ARC): osm_node_end.add(nd)
for nd in lkw_nodes:
    if connect(nd, D150): osm_node_end.add(nd)
n_end=len(osm_node_end)
_end_tree[0]=cKDTree(np.array([pp for p,pp in links])) if links else None   # end-connection landing points -> baseline for corner clash avoidance
print(f"Ends attached to OSM: {n_end} | {time.time()-t:.0f}s",flush=True)
# Step 2: arcade corners -- NOTE: each corner is judged independently (only the fixed end state osm_node_end is consulted, unaffected by whether other corners were connected):
# when the "other end" of neither adjacent arcade segment is attached to a main path, add one perpendicular connection; also avoid clashing (overlapping landing points) with existing connections (ends + already connected corners)
def other_end(si,nd): a,b=sn[si]; return a if b==nd else b
n_corner=0
for i,nd in enumerate(arc_corner):
    arc_si=[si for si in node_segs[nd] if facsrc[si]=='arcade']     # arcade segments on both sides of the corner
    if any(other_end(si,nd) in osm_node_end for si in arc_si): continue  # the other end on one side is already attached to a main path -> do not connect (NOTE: decision uses the fixed end state only)
    if connect(nd, D_ARC, anti=True): n_corner+=1
    if (i+1)%4000==0: print(f"  corner check {i+1}/{len(arc_corner)} (connected {n_corner}) | {time.time()-t:.0f}s",flush=True)
nconn=len(links)
print(f"Connected to OSM: ends {n_end} + corners (neither side connected + clash avoided) {n_corner} = {nconn} | out of range {nfar} | clash skipped {nclash} | building crossing/double-back/in_fac rejected {nblk} | {time.time()-t:.0f}s",flush=True)

def split_line(line,pts):
    L=line.length; cuts=sorted(set(round(line.project(pp),2) for pp in pts)); cuts=[c for c in cuts if 0.2<c<L-0.2]
    if not cuts: return [line]
    bs=[0.0]+cuts+[L]; out=[]
    for k in range(len(bs)-1):
        if bs[k+1]-bs[k]>0.2:
            try: out.append(substring(line,bs[k],bs[k+1]))
            except Exception: pass
    return out if out else [line]
og=list(facsegs); osrc=list(facsrc)
for i in range(len(osmsegs)):
    if i in splits:
        for sub in split_line(osmsegs[i],splits[i]): og.append(sub); osrc.append(osmsrc[i])
    else: og.append(osmsegs[i]); osrc.append(osmsrc[i])
for p,pp in links: og.append(LineString([tuple(p),pp])); osrc.append('fac2osm')
out=gpd.GeoDataFrame({'src':osrc,'geometry':og},crs=3414); out['length']=out.geometry.length
out.to_file(f"{OUT}\\step4_connected.gpkg",driver="GPKG")
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
_ho=collections.defaultdict(bool)
for i in range(len(og)):
    if osrc[i] in ('osm','osmlink'): _ho[f2(sn2[i][0])]=True
_iso=len(set(f2(sn2[i][0]) for i in range(len(og)))-set(c for c,h in _ho.items() if h))
print(f"Done {len(out)} segments {out.length.sum()/1000:.0f}km | connected components {nc} | isolated facility components (not attached to OSM) {_iso} | {time.time()-t:.0f}s",flush=True)
