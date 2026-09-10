"""Evaluate the optimal gap and min-len thresholds for merge (data-driven).
- Street-view spacing: median distance between adjacent panos
- Gap structure: distribution of the gaps between adjacent projected segments within the same corridor (look for natural breaks)
- Run length distribution after gap bridging for different GAP values (where min-len should cut)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

ROOT=Path(r"D:\Claude\SVI_FFW"); OUT=ROOT/"output"/"step2_projection"
EPSG={"sg":3414,"bo":32632}
POS={"sg":ROOT/"output"/"production_colonnade"/"colonnade_sg_positive.csv",
     "bo":ROOT/"output"/"production_colonnade"/"colonnade_bo_positive.csv"}

class UF:
    def __init__(s,n): s.p=list(range(n))
    def f(s,x):
        while s.p[x]!=x: s.p[x]=s.p[s.p[x]]; x=s.p[x]
        return x
    def u(s,a,b): s.p[s.f(a)]=s.f(b)

def comps(geoms,GAP):
    n=len(geoms); tree=STRtree(geoms); uf=UF(n)
    for i in range(n):
        for j in tree.query(geoms[i].buffer(GAP)):
            if j>i and geoms[i].distance(geoms[j])<GAP: uf.u(i,j)
    cc={}
    for i in range(n): cc.setdefault(uf.f(i),[]).append(i)
    return list(cc.values())

def axis(coords):
    a=np.array(coords)[:,:2]; a0=a-a.mean(0)
    if len(a)<2: return np.array([1.,0.])
    w,v=np.linalg.eigh(np.cov(a0.T)); return v[:,int(np.argmax(w))]

def fill_len(members,geoms,GAP):
    """Continuous run length after bridging gaps < GAP (and the list of internal gaps)"""
    lines=[geoms[m] for m in members]
    u=unary_union(lines); m=linemerge(u) if u.geom_type=="MultiLineString" else u
    parts=list(m.geoms) if m.geom_type=="MultiLineString" else [m]
    allc=[c for p in parts for c in p.coords]
    ax=axis(allc)
    def proj(pt): return pt[0]*ax[0]+pt[1]*ax[1]
    info=[]
    for p in parts:
        e0,e1=p.coords[0],p.coords[-1]
        lo,hi=(e0,e1) if proj(e0)<=proj(e1) else (e1,e0)
        info.append(dict(lo=lo,hi=hi,plo=min(proj(e0),proj(e1)),part=p))
    info.sort(key=lambda d:d["plo"])
    newl=list(parts); gaps=[]
    for i in range(len(info)-1):
        hi=info[i]["hi"]; lo=info[i+1]["lo"]
        gp=((hi[0]-lo[0])**2+(hi[1]-lo[1])**2)**0.5
        gaps.append(gp)
        if 1e-6<gp<GAP: newl.append(LineString([hi,lo]))
    uu=unary_union(newl)
    mm=linemerge(uu) if uu.geom_type=="MultiLineString" else uu
    return float(mm.length),gaps

def main(city):
    seg=gpd.read_file(OUT/f"step2a_segments_{city}.gpkg")
    geoms=list(seg.geometry.values)
    print(f"\n########## {city.upper()} ##########")
    print(f"projected segments: {len(geoms)}  (median segment length {np.median([g.length for g in geoms]):.1f}m)")
    # street-view spacing
    pos=pd.read_csv(POS[city]).drop_duplicates("pid")
    g=gpd.GeoDataFrame(pos,geometry=gpd.points_from_xy(pos.lon,pos.lat,crs="EPSG:4326")).to_crs(EPSG[city])
    xy=np.c_[g.geometry.x.values,g.geometry.y.values]
    kt=cKDTree(xy); dd,_=kt.query(xy,k=2); nn=dd[:,1]
    print(f"adjacent pano spacing: median {np.median(nn):.1f}m (p25 {np.percentile(nn,25):.1f} / p75 {np.percentile(nn,75):.1f})")

    # gap structure (group at GAP=80, inspect the gap distribution between adjacent segments inside a group)
    allg=[]
    for mem in comps(geoms,80.0):
        if len(mem)<2: continue
        _,gaps=fill_len(mem,geoms,80.0); allg+=[x for x in gaps if x>1e-6]
    allg=np.array(allg)
    print(f"\ngap distribution between adjacent segments within the same corridor (n={len(allg)}):")
    bins=[0,10,20,30,40,50,60,80,120,1e9]; lab=["<10","10-20","20-30","30-40","40-50","50-60","60-80","80-120",">120"]
    h,_=np.histogram(allg,bins=bins)
    for l,c in zip(lab,h): print(f"   {l:>7}m: {c:5d}  ({c/max(len(allg),1)*100:4.1f}%)  cumulative <= upper bound {np.sum(allg<=bins[lab.index(l)+1])/max(len(allg),1)*100:4.1f}%")

    # run length distribution (after gap bridging) for different GAP values
    print(f"\nrun statistics (after gap bridging) for different GAP values:")
    print(f"  {'GAP':>4} {'#run':>6} {'tot_km':>7}  length distribution [<30 / 30-50 / 50-75 / 75-100 / >=100]")
    for GAP in [20,30,40,50,60]:
        Ls=[]
        for mem in comps(geoms,float(GAP)):
            L,_=fill_len(mem,geoms,float(GAP)); Ls.append(L)
        Ls=np.array(Ls)
        b=[(Ls<30).sum(),((Ls>=30)&(Ls<50)).sum(),((Ls>=50)&(Ls<75)).sum(),((Ls>=75)&(Ls<100)).sum(),(Ls>=100).sum()]
        print(f"  {GAP:>4} {len(Ls):>6} {Ls.sum()/1000:>7.1f}  [{b[0]:4d} /{b[1]:4d} /{b[2]:4d} /{b[3]:4d} /{b[4]:4d}]")

if __name__=="__main__":
    for c in (sys.argv[1:] if len(sys.argv)>1 else ["sg","bo"]): main(c)
