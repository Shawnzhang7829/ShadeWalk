# -*- coding: utf-8 -*-
"""Step2b-edge (edge-based approach, replaces step2_merge):
  (1) aggregate building footprints (union touching ones into blocks) -> (2) extract the outer and inner-hole outlines -> (3) split at corners (deflection > 30 deg; each piece = one straight facade edge)
  -> (4) for each edge: project the step2a_segments that are close to it (< NEAR) and parallel (angle < 30 deg) onto the edge, take the union length cov;
     cov/edge length > 20% or cov > 20 m -> the edge is classified as five-foot way.
  Writes step2b_runs_sg.gpkg (length / bld_h_med / geometry), compatible with the downstream step2_buffer_split (buf=2.0). pyenv.
Usage: python step2_edge_ffw.py sg
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
ROOT=Path(r"D:\Claude\SVI_FFW"); OUT=ROOT/"output"/"step2_projection"
BLD={"sg":(ROOT/"Shp"/"SG"/"SG_Building"/"SG_Building_SVY21_TH.shp",3414),
     "bo":(ROOT/"Shp"/"Bologna"/"c_a944ctc_edifici_pl.geojson",32632),
     "gz":(ROOT/"Shp"/"GZ"/"GZ_core_building"/"GZ_core_building.shp",32649)}
CORNER_ANG=30.0; NEAR=1.5; PARA_COS=0.866; COV_PCT=0.20; COV_LEN=20.0; MINEDGE=2.0

def _dir(c0,c1):
    dx,dy=c1[0]-c0[0],c1[1]-c0[1]; L=math.hypot(dx,dy)
    return (dx/L,dy/L) if L>0 else (1.0,0.0)

def split_ring(coords, ang_thr):
    """Ring vertices (closed, coords[-1]==coords[0]) -> split at corners with deflection > ang_thr -> list of straight facade edges."""
    n=len(coords)-1
    if n<2: return []
    corner=[False]*n
    for k in range(n):
        a,b,c=coords[(k-1)%n],coords[k],coords[(k+1)%n]
        v1=(b[0]-a[0],b[1]-a[1]); v2=(c[0]-b[0],c[1]-b[1])
        L1=math.hypot(*v1); L2=math.hypot(*v2)
        if L1<1e-6 or L2<1e-6: continue
        ang=math.degrees(math.acos(max(-1.0,min(1.0,(v1[0]*v2[0]+v1[1]*v2[1])/(L1*L2)))))
        if ang>ang_thr: corner[k]=True
    cs=[k for k in range(n) if corner[k]]
    out=[]
    if not cs:
        if LineString(coords).length>=MINEDGE: out.append(LineString(coords))
        return out
    for ci in range(len(cs)):
        s=cs[ci]; e=cs[(ci+1)%len(cs)]; pts=[]; k=s
        while True:
            pts.append(coords[k%n])
            if k%n==e%n: break
            k+=1
            if len(pts)>n+1: break
        if len(pts)>=2:
            ls=LineString(pts)
            if ls.length>=MINEDGE: out.append(ls)
    return out

def cov_len(intervals):
    if not intervals: return 0.0
    intervals=sorted(intervals); tot=0.0; cs,ce=intervals[0]
    for a,b in intervals[1:]:
        if a>ce: tot+=ce-cs; cs,ce=a,b
        else: ce=max(ce,b)
    tot+=ce-cs; return tot

def main(city):
    bldpath,epsg=BLD[city]
    seg=gpd.read_file(OUT/f"step2a_segments_{city}.gpkg")
    segs=list(seg.geometry.values); seg_bh=seg["bld_h"].values
    sdir=[_dir(g.coords[0],g.coords[-1]) for g in segs]
    segtree=STRtree(segs)
    print(f"[{city}] step2a projected segments: {len(segs)}",flush=True)

    bld=gpd.read_file(bldpath).to_crs(epsg)
    bld=bld[bld.geometry.notna()&bld.geometry.is_valid].reset_index(drop=True)
    bld["geometry"]=bld.geometry.buffer(0)
    print(f"[{city}] buildings {len(bld)} -> aggregating (unary_union)...",flush=True)
    blocks=unary_union(list(bld.geometry.values))
    polys=list(blocks.geoms) if blocks.geom_type=="MultiPolygon" else [blocks]
    print(f"[{city}] aggregated blocks {len(polys)} -> extracting outlines and splitting at corners...",flush=True)

    edges=[]
    for poly in polys:
        rings=[list(poly.exterior.coords)]+[list(r.coords) for r in poly.interiors]
        for rc in rings:
            edges+=split_ring(rc, CORNER_ANG)
    print(f"[{city}] facade edges {len(edges)} -> projection coverage test...",flush=True)

    runs=[]; nffw=0
    for ei,es in enumerate(edges):
        if (ei+1)%20000==0: print(f"   {ei+1}/{len(edges)}",flush=True)
        ed=_dir(es.coords[0],es.coords[-1]); intervals=[]; bhs=[]
        for si in segtree.query(es.buffer(NEAR)):
            si=int(si); s=segs[si]
            if s.distance(es)>NEAR: continue
            d=sdir[si]
            if abs(ed[0]*d[0]+ed[1]*d[1])<PARA_COS: continue   # parallel (angle < 30 deg)
            t0=es.project(Point(s.coords[0])); t1=es.project(Point(s.coords[-1]))
            intervals.append((min(t0,t1),max(t0,t1))); bhs.append(seg_bh[si])
        if not intervals: continue
        cov=cov_len(intervals); EL=es.length
        if cov/EL>COV_PCT or cov>COV_LEN:                       # coverage > 20% or covered length > 20 m
            runs.append({"run_id":nffw,"length":EL,"n_seg":len(intervals),
                         "bld_h_med":float(np.median(bhs)) if bhs else 0.0,"geometry":es})
            nffw+=1
    out=gpd.GeoDataFrame(runs,crs=epsg); out.to_file(OUT/f"step2b_runs_{city}.gpkg",driver="GPKG")
    tot=out["length"].sum() if len(out) else 0
    print(f"[{city}] five-foot way edges: {len(out)}, total length {tot/1000:.1f} km",flush=True)
    print(f"   wrote step2b_runs_{city}.gpkg (next: buffer_split produces the 2 m deep arcade strip)",flush=True)

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "sg")
