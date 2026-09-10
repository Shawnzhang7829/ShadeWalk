"""Step2b merge v6 (gap bridging along the footprint outline + 1 m outward street-side check):
 - Collinear grouping: near-parallel < 30 deg + perpendicular offset < 3 m = the same facade line (including a continuous frontage across adjacent buildings); gap < 50 m.
 - Gap bridging along the outline: within the same building -> follow that building's OUTER RING outline (short arc, never across a large corner); across adjacent buildings -> straight connection (approx. collinear outline).
 - Outward street-side check: every point of the bridged outline / connection MUST have air 1 m outward (no other building); touching another building (shared wall / seam) -> no bridging.
 - Prevents crossing streets (straight connection with buildings on both sides = inside a seam -> no bridging; no building backing -> no bridging), wrapping around corners (break at large corners) and cutting through building interiors.
 - Corner snap <= 20 m; drop runs < 30 m.
Usage: python step2_merge.py sg / bo
"""
from __future__ import annotations
import sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union, substring, nearest_points
from shapely.strtree import STRtree

ROOT=Path(r"D:\Claude\SVI_FFW"); OUT=ROOT/"output"/"step2_projection"
GAP=50.0; MIN_LEN=5.0; ANG_COS=0.866; OFFSET_MAX=3.0; CORNER_EXT=20.0; BACK_THR=2.5   # MIN_LEN 30 -> 5 (user rule 15: drop only fragments < 5 m, keep valid short arcades)
CONS_PATH=r"C:\Users\City Syntax Lab\Desktop\Covered Linkways\1-data\1-SG\ConservationArea2025\MasterPlan2025ConservationAreaBoundaryLayer.geojson"  # conservation areas: inside them, breaks along the same street are always bridged along the outline
AIR=1.0; CORNER_ANG=30.0   # 1 m outward must be air; ring corner > 30 deg = facade boundary (never crossed)
BLD={"sg":(ROOT/"Shp"/"SG"/"SG_Building"/"SG_Building_SVY21_TH.shp",3414),
     "bo":(ROOT/"Shp"/"Bologna"/"c_a944ctc_edifici_pl.geojson",32632)}

class UF:
    def __init__(s,n): s.p=list(range(n))
    def f(s,x):
        while s.p[x]!=x: s.p[x]=s.p[s.p[x]]; x=s.p[x]
        return x
    def u(s,a,b): s.p[s.f(a)]=s.f(b)

def _axis(coords):
    a=np.array(coords)[:,:2]; a0=a-a.mean(0)
    if len(a)<2: return np.array([1.,0.])
    w,v=np.linalg.eigh(np.cov(a0.T)); return v[:,int(np.argmax(w))]

def pt_in_bld(pt, btree, bgeoms):
    for idx in btree.query(pt):
        if bgeoms[idx].intersects(pt): return True
    return False

def out_normal(ring, s, RL, poly):
    p=ring.interpolate(s); p0=ring.interpolate(max(0.0,s-0.5)); p1=ring.interpolate(min(RL,s+0.5))
    dx=p1.x-p0.x; dy=p1.y-p0.y; L=math.hypot(dx,dy)
    if L==0: return p,(1.0,0.0)
    dx,dy=dx/L,dy/L; na=(dy,-dx)
    if poly.contains(Point(p.x+0.1*na[0],p.y+0.1*na[1])): na=(-na[0],-na[1])
    return p,na

def air_out(p, n, btree, bgeoms, self_idx, air=AIR):
    probe=LineString([(p.x+0.05*n[0],p.y+0.05*n[1]),(p.x+air*n[0],p.y+air*n[1])])
    for idx in btree.query(probe):
        if idx==self_idx: continue
        if probe.intersects(bgeoms[idx]): return False
    return True

def fill_same(bi, ea, eb, rings, btree, bgeoms, lenient=False):
    poly,ring,RL,corners=rings[bi]
    pa=ring.project(ea); pb=ring.project(eb); d=abs(pb-pa)
    if min(d,RL-d)>GAP: return None
    if d<=RL-d: lo,hi=min(pa,pb),max(pa,pb)
    else: lo,hi=max(pa,pb),min(pa,pb)+RL
    for cp in corners:                                       # the arc must not contain a large corner (wrapping around / switching facade) -- kept in conservation areas too (same street = no large corner crossed)
        cpw=cp if cp>=lo else cp+RL
        if lo+1e-6<cpw<hi-1e-6: return None
    if not lenient:                                          # conservation areas (lenient) skip the "1 m outward must be air" check and always bridge along the outline
        s=lo+0.5
        while s<hi:
            p,n=out_normal(ring,s%RL,RL,poly)
            if not air_out(p,n,btree,bgeoms,bi): return None
            s+=1.5
    if hi<=RL+1e-6: return substring(ring,lo,min(hi,RL))
    a=substring(ring,lo,RL); b=substring(ring,0.0,hi-RL); u=unary_union([a,b])
    return linemerge(u) if u.geom_type=="MultiLineString" else u

def fill_diff(ea, eb, btree, bgeoms, lenient=False):
    dx=eb.x-ea.x; dy=eb.y-ea.y; L=math.hypot(dx,dy)
    if L<0.3 or L>=GAP: return None
    if not lenient:                                          # conservation areas (lenient) skip the backing / seam checks; breaks along the same street are connected directly
        perp=(dy/L,-dx/L); nstep=max(2,int(L/1.5))
        for k in range(nstep+1):
            t=k/nstep; px=ea.x+dx*t; py=ea.y+dy*t; pp=Point(px,py)
            backed=any(bgeoms[idx].distance(pp)<=BACK_THR for idx in btree.query(pp.buffer(BACK_THR+0.2)))
            if not backed: return None                       # no building backing = crossing a street
            s1=Point(px+perp[0]*AIR,py+perp[1]*AIR); s2=Point(px-perp[0]*AIR,py-perp[1]*AIR)
            if pt_in_bld(s1,btree,bgeoms) and pt_in_bld(s2,btree,bgeoms): return None  # buildings on both sides = inside a seam
    return LineString([(ea.x,ea.y),(eb.x,eb.y)])

def snap_corners(run, ctree, cpts, ext):
    if run.geom_type!="LineString": return run
    cs=list(run.coords)
    if len(cs)<2: return run
    out=cs[:]
    for which in (0,1):
        P=cs[0] if which==0 else cs[-1]; Q=cs[1] if which==0 else cs[-2]
        dx=P[0]-Q[0]; dy=P[1]-Q[1]; L=math.hypot(dx,dy)
        if L==0: continue
        dx,dy=dx/L,dy/L; best=None
        for idx in ctree.query(Point(P[0],P[1]).buffer(ext)):
            V=cpts[idx]; vx=V[0]-P[0]; vy=V[1]-P[1]; dist=math.hypot(vx,vy)
            if dist<0.3 or dist>ext: continue
            if vx*dx+vy*dy<=0: continue
            if abs(vx*dy-vy*dx)>1.5: continue
            if best is None or dist<best[1]: best=(V,dist)
        if best: out=([best[0]]+out) if which==0 else (out+[best[0]])
    return LineString(out)

def assemble(mem, geoms, seg_bld, rings, btree, bgeoms, cons_prep=None):
    if len(mem)==1: return geoms[mem[0]]
    parts=[geoms[m] for m in mem]
    ax=_axis([c for p in parts for c in p.coords]); pr=lambda xy: xy[0]*ax[0]+xy[1]*ax[1]
    order=sorted(mem, key=lambda m: pr(geoms[m].interpolate(0.5,normalized=True).coords[0]))
    pieces=[geoms[order[0]]]
    for i in range(len(order)-1):
        m,m2=order[i],order[i+1]; g,g2=geoms[m],geoms[m2]
        ea,eb=nearest_points(g,g2); bi,bj=int(seg_bld[m]),int(seg_bld[m2])
        lenient=cons_prep is not None and cons_prep.contains(Point((ea.x+eb.x)/2,(ea.y+eb.y)/2))  # gap midpoint inside a conservation area -> lenient bridging
        fill=None
        if bi==bj and bi in rings: fill=fill_same(bi,ea,eb,rings,btree,bgeoms,lenient)
        else: fill=fill_diff(ea,eb,btree,bgeoms,lenient)
        if fill is not None and not fill.is_empty: pieces.append(fill)
        pieces.append(g2)
    u=unary_union(pieces); return linemerge(u) if u.geom_type=="MultiLineString" else u

def _mindist(pt, btree, bgeoms):
    c=btree.query(pt.buffer(8.0))
    if len(c)==0: c=btree.query(pt.buffer(40.0))
    return min((bgeoms[i].distance(pt) for i in c), default=99.0)

def _one_side_building(p, perp, btree, bgeoms, d=0.6):
    """A point on the connection line: at least one side within ±d perpendicular falls inside a building = one side hugs a building (not floating)."""
    for sgn in (1.0,-1.0):
        q=Point(p.x+sgn*d*perp[0], p.y+sgn*d*perp[1])
        for idx in btree.query(q):
            if bgeoms[idx].intersects(q): return True
    return False

def trim_offbuilding(run, btree, bndry, cons_prep=None, thr=0.6, step=0.5):
    """The run must lie on a building BOUNDARY (facade) line (kept only where the distance to the nearest building boundary < thr).
    Also trims away: (1) floating parts crossing a street / seam (far from the boundary = air) (2) parts cutting into the building interior (far from the boundary = solid).
    Points inside conservation areas (cons_prep) are always kept (rule 15: breaks along the same street are always bridged)."""
    parts=[run] if run.geom_type=="LineString" else [p for p in getattr(run,"geoms",[]) if p.geom_type=="LineString"]
    out=[]
    for ls in parts:
        L=ls.length
        if L<=0.5: continue
        ss=list(np.arange(0.0,L+1e-9,step))
        if ss[-1]<L-1e-9: ss.append(L)
        on=[]
        for s in ss:
            p=ls.interpolate(s); c=btree.query(p.buffer(2.0))
            d=min((bndry[i].distance(p) for i in c),default=99.0)
            on.append(d<thr or (cons_prep is not None and cons_prep.contains(p)))
        i=0
        while i<len(ss):
            if on[i]:
                j=i
                while j+1<len(ss) and on[j+1]: j+=1
                a=ss[i]; b=ss[j]
                if b-a>=1.0:
                    sub=substring(ls,a,b)
                    if (not sub.is_empty) and sub.length>=1.0: out.append(sub)
                i=j+1
            else: i+=1
    if not out: return None
    u=unary_union(out); return linemerge(u) if u.geom_type=="MultiLineString" else u

def main(city):
    bldpath,epsg=BLD[city]
    seg=gpd.read_file(OUT/f"step2a_segments_{city}.gpkg")
    print(f"[{city}] input segments: {len(seg)}")
    geoms=[LineString([(c[0],c[1]) for c in g.coords]) for g in seg.geometry.values]
    seg_bld=seg["bld_idx"].values; n=len(geoms)
    def sdir(g):
        cc=list(g.coords); dx=cc[-1][0]-cc[0][0]; dy=cc[-1][1]-cc[0][1]; L=math.hypot(dx,dy)
        return (dx/L,dy/L) if L>0 else (1.0,0.0)
    dirs=[sdir(g) for g in geoms]
    mids=[(g.interpolate(0.5,normalized=True).x,g.interpolate(0.5,normalized=True).y) for g in geoms]

    bld=gpd.read_file(bldpath).to_crs(epsg)
    bld=bld[bld.geometry.notna()&bld.geometry.is_valid].reset_index(drop=True)
    bld["geometry"]=bld.geometry.buffer(0); bgeoms=list(bld.geometry.values); btree=STRtree(bgeoms)
    bndry=[g.boundary for g in bgeoms]   # building boundary (facade) lines, used by trim
    cons_prep=None                       # conservation areas (user rule 15: inside them, breaks along the same street are always bridged along the outline)
    if Path(CONS_PATH).exists():
        from shapely.prepared import prep
        consg=gpd.read_file(CONS_PATH).to_crs(epsg)
        cons_prep=prep(unary_union(consg.geometry.values))
        print(f"[{city}] conservation areas: {len(consg)} polygons (more lenient bridging inside them)")
    cpts=[]
    for gm in bgeoms:
        for pg in ([gm] if gm.geom_type=="Polygon" else list(gm.geoms)):
            for cc in list(pg.exterior.coords)[:-1]: cpts.append((cc[0],cc[1]))
    ctree=STRtree([Point(p) for p in cpts])

    tree=STRtree(geoms); uf=UF(n)
    for i in range(n):
        di=dirs[i]; mi=mids[i]
        for j in tree.query(geoms[i].buffer(GAP)):
            if j<=i or geoms[i].distance(geoms[j])>=GAP: continue
            dj=dirs[j]
            if abs(di[0]*dj[0]+di[1]*dj[1])<ANG_COS: continue
            vx=mids[j][0]-mi[0]; vy=mids[j][1]-mi[1]
            if abs(vx*di[1]-vy*di[0])>OFFSET_MAX: continue
            uf.u(i,j)
    comp=defaultdict(list)
    for i in range(n): comp[uf.f(i)].append(i)

    rings={}
    for bi in {int(seg_bld[m]) for mem in comp.values() for m in mem}:
        if bi<0 or bi>=len(bgeoms): continue
        poly=bgeoms[bi]
        if poly.geom_type=="MultiPolygon": poly=max(poly.geoms,key=lambda g:g.area)
        ring=LineString([(c[0],c[1]) for c in poly.exterior.coords]); RL=ring.length  # force 2D (SG buildings carry Z)
        cs=list(poly.exterior.coords); corners=[]
        for k in range(1,len(cs)-1):
            a,b,cc=cs[k-1],cs[k],cs[k+1]; v1=(b[0]-a[0],b[1]-a[1]); v2=(cc[0]-b[0],cc[1]-b[1])
            L1=math.hypot(*v1); L2=math.hypot(*v2)
            if L1<1e-6 or L2<1e-6: continue
            ang=math.degrees(math.acos(max(-1,min(1,(v1[0]*v2[0]+v1[1]*v2[1])/(L1*L2)))))
            if ang>CORNER_ANG: corners.append(ring.project(Point(b)))
        rings[bi]=(poly,ring,RL,corners)

    runs=[]
    for cid,mem in comp.items():
        run=assemble(mem,geoms,seg_bld,rings,btree,bgeoms,cons_prep)
        run=snap_corners(run,ctree,cpts,CORNER_EXT)
        run=trim_offbuilding(run,btree,bndry,cons_prep)        # trim parts > 0.6 m from the building boundary; always kept inside conservation areas
        if run is None: continue
        total=float(run.length)
        if total>=MIN_LEN:
            bh=float(seg.iloc[mem]["bld_h"].median())
            runs.append({"run_id":int(cid),"length":total,"n_seg":len(mem),"bld_h_med":bh,"geometry":run})
    out=gpd.GeoDataFrame(runs,crs=epsg); out.to_file(OUT/f"step2b_runs_{city}.gpkg",driver="GPKG")
    tot=out["length"].sum() if len(out) else 0
    print(f"[{city}] merged arcade runs (>={MIN_LEN:.0f}m): {len(out)}, total length {tot/1000:.1f} km")
    print(f"   wrote step2b_runs_{city}.gpkg")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "sg")
