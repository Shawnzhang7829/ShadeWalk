"""Step2a projection (current, lenient / Option A): colonnade-positive view -> nearest visible STREET-FACING facade -> vector segment along the footprint edge.
Three rules (lenient version):
 (1) The facade must face a non-building street (probe clear m along the outward normal, no building hit); facades facing another wall are all discarded.
 (2) A facade counts as soon as ONE unoccluded colonnade-positive view claims it (the nearest observation need not be positive; nearby views may be occluded by cars/trees and missed).
 (3) Project along viewheading onto the nearest UNOCCLUDED wall (first hit of the fan of rays, occlusion-aware).
Also writes the street-view point layer (projected = whether the view was projected onto a street facade).
Usage: python step2_project.py sg / bo
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import substring
from shapely.strtree import STRtree

ROOT=Path(r"D:\Claude\SVI_FFW"); OUT=ROOT/"output"/"step2_projection"; OUT.mkdir(parents=True,exist_ok=True)
CFG={
 "sg":dict(bld=ROOT/"Shp"/"SG"/"SG_Building"/"SG_Building_SVY21_TH.shp",epsg=3414,
           hcol="height",pos=ROOT/"output"/"production_colonnade"/"colonnade_sg_v2_positive.csv",
           buffer=1.5,arc_h=3.6,clear=3.0),
 "bo":dict(bld=ROOT/"Shp"/"Bologna"/"c_a944ctc_edifici_pl.geojson",epsg=32632,
           hcol="altezza_gr",pos=ROOT/"output"/"production_colonnade"/"colonnade_bo_v2_positive.csv",
           buffer=3.0,arc_h=3.6,clear=3.0),
 "gz":dict(bld=ROOT/"Shp"/"GZ"/"GZ_core_building"/"GZ_core_building.shp",epsg=32649,
           hcol="Height",pos=ROOT/"output"/"production_colonnade"/"colonnade_gz_v2_positive.csv",
           buffer=1.5,arc_h=3.6,clear=3.0),
}
RAY_LEN=50.0; SEG_HALF=10.0; FAN=[-28,-16,-8,0,8,16,28]   # RAY_LEN = upper bound of the distance to the facing facade 50 m (roads <= 100 m; > 50 m = distant building at a T-junction, ignored)

def entry_pt(inter, cam):
    cands=[]; gt=inter.geom_type
    if gt=="Point": cands=[inter]
    elif gt=="LineString": cands=[Point(inter.coords[0]),Point(inter.coords[-1])]
    elif gt in ("MultiLineString","GeometryCollection","MultiPoint"):
        for gg in inter.geoms:
            if gg.geom_type=="LineString": cands+=[Point(gg.coords[0]),Point(gg.coords[-1])]
            elif gg.geom_type=="Point": cands.append(gg)
    return min(cands,key=lambda p:cam.distance(p)) if cands else None

def outward_normal(edge, poly):
    c0,c1=edge.coords[0],edge.coords[1]; x0,y0=c0[0],c0[1]; x1,y1=c1[0],c1[1]
    dx,dy=x1-x0,y1-y0; L=math.hypot(dx,dy)
    if L==0: return None
    dx,dy=dx/L,dy/L; na=(dy,-dx); mx,my=(x0+x1)/2,(y0+y1)/2
    if poly.contains(Point(mx+0.15*na[0],my+0.15*na[1])): na=(-na[0],-na[1])
    return na

def faces_street(pt, n, tree, geoms, clear):
    s=(pt.x+0.10*n[0],pt.y+0.10*n[1]); e=(pt.x+clear*n[0],pt.y+clear*n[1])
    probe=LineString([s,e])
    for idx in tree.query(probe):
        if probe.intersects(geoms[idx]): return False
    return True

def main(city):
    c=CFG[city]; CLEAR=c["clear"]
    print(f"[{city}] load buildings …")
    bld=gpd.read_file(c["bld"]).to_crs(c["epsg"])
    bld=bld[bld.geometry.notna() & bld.geometry.is_valid].reset_index(drop=True)
    bld["geometry"]=bld.geometry.buffer(0)
    h=pd.to_numeric(bld[c["hcol"]],errors="coerce").fillna(0).values
    geoms=list(bld.geometry.values); tree=STRtree(geoms)

    pos=pd.read_csv(c["pos"])
    print(f"[{city}] colonnade-positive views: {len(pos)}")
    g=gpd.GeoDataFrame(pos,geometry=gpd.points_from_xy(pos.lon,pos.lat,crs="EPSG:4326")).to_crs(c["epsg"])
    xs=g.geometry.x.values; ys=g.geometry.y.values; vhs=g["viewheading"].values

    rows=[]; pts=[]; nmiss=0; nrej_face=0; n_inbld=0; not_facing=0
    for i in range(len(g)):
        cam=Point(xs[i],ys[i]); vh=float(vhs[i])
        pid=int(g.iloc[i]["pid"]); view=int(g.iloc[i]["view"])
        # Rule (4): camera point falls inside a building footprint (GPS error / coincides with the building) -> drop it directly, neither projected nor recorded
        inb=False
        for idx in tree.query(cam):
            if geoms[idx].intersects(cam): inb=True; break
        if inb: n_inbld+=1; continue
        cands=[]
        for off in FAN:
            vr=math.radians(vh+off); dx,dy=math.sin(vr),math.cos(vr)
            ray=LineString([cam,Point(cam.x+RAY_LEN*dx,cam.y+RAY_LEN*dy)]); rh=None
            for idx in tree.query(ray):
                inter=ray.intersection(geoms[idx])
                if inter.is_empty: continue
                ep=entry_pt(inter,cam)
                if ep is None: continue
                d=cam.distance(ep)
                if d<0.5 or d>RAY_LEN: continue
                if rh is None or d<rh[0]: rh=(d,idx,ep)
            if rh: cands.append(rh)
        proj=0; dout=None
        if cands:
            cands.sort(key=lambda t:t[0]); face_seen=False; nf=False
            for d,bi,entry in cands:
                poly=geoms[bi]
                if poly.geom_type=="MultiPolygon": poly=min(poly.geoms,key=lambda gg:gg.distance(entry))  # take the sub-polygon containing the entry point
                coords=list(poly.exterior.coords); be=None
                for k in range(len(coords)-1):
                    e=LineString([coords[k],coords[k+1]]); de=e.distance(entry)
                    if be is None or de<be[0]: be=(de,e)
                edge=be[1]; n=outward_normal(edge,poly)
                if n is None: continue
                if not faces_street(entry,n,tree,geoms,CLEAR): face_seen=True; continue   # rule (1): must face the street
                # geometric facing test: keep only if the angle between viewheading and the facade direction is > 45 deg (incidence angle < 45 deg, facing); grazing views along the facade are dropped
                ec=list(edge.coords); efx=ec[1][0]-ec[0][0]; efy=ec[1][1]-ec[0][1]; efl=math.hypot(efx,efy)
                if efl>0:
                    vr2=math.radians(vh)
                    if abs((efx/efl)*math.sin(vr2)+(efy/efl)*math.cos(vr2))>0.707: nf=True; continue
                elen=edge.length; t=edge.project(entry)
                s0=max(0.0,t-SEG_HALF); s1=min(elen,t+SEG_HALF)
                ss=list(np.arange(s0,s1+1e-6,1.0))
                if not ss or ss[-1]<s1-1e-6: ss.append(s1)
                openf=[faces_street(edge.interpolate(s),n,tree,geoms,CLEAR) for s in ss]   # rule (3): clip to the unoccluded part
                ti=int(np.argmin([abs(s-t) for s in ss]))
                if not openf[ti]: continue
                lo=ti
                while lo-1>=0 and openf[lo-1]: lo-=1
                hi=ti
                while hi+1<len(ss) and openf[hi+1]: hi+=1
                a,b=ss[lo],ss[hi]
                if b-a<1.0: continue
                seg=substring(edge,a,b)
                if seg.is_empty or seg.length<1: continue
                rows.append({"bld_idx":int(bi),"bld_h":float(h[bi]),"pid":pid,"view":view,
                             "viewheading":vh,"dist":float(d),"geometry":seg})
                proj=1; dout=float(d); break
            if proj==0 and nf: not_facing+=1
            elif proj==0 and face_seen: nrej_face+=1
        if proj==0: nmiss+=1
        pts.append({"pid":pid,"view":view,"viewheading":vh,"projected":proj,
                    "dist":(dout if dout else -1.0),"geometry":cam})

    out=gpd.GeoDataFrame(rows,crs=c["epsg"]); out.to_file(OUT/f"step2a_segments_{city}.gpkg",driver="GPKG")
    pg=gpd.GeoDataFrame(pts,crs=c["epsg"]); pg.to_file(OUT/f"step2a_points_{city}.gpkg",driver="GPKG")
    print(f"[{city}] dropped by rule (4) (camera inside a building): {n_inbld}")
    print(f"[{city}] projected segments: {len(out)}, buildings involved {out.bld_idx.nunique() if len(out) else 0}")
    print(f"[{city}] missed {nmiss}/{len(g)-n_inbld} ({nmiss/max(len(g)-n_inbld,1)*100:.0f}%); dropped as [not facing / grazing along facade] {not_facing}, discarded as [facing another building] {nrej_face}")
    print(f"[{city}] street-view points: {len(pg)} (projected {int(pg.projected.sum())})")
    print(f"   wrote step2a_segments_{city}.gpkg + step2a_points_{city}.gpkg")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "sg")
