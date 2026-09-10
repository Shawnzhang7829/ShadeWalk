# -*- coding: utf-8 -*-
"""Stage 3a remove OSM lines that hug the shade centrelines: facility centrelines (arcade+linkway+facconn) merged, buffered 4 m on each side = 8 m band.
Criterion (updated 2026-06-25): an OSM segment with >= 80% of its length inside the band (overlap) AND parallel to the nearest centreline segment (angle < 30 deg) -> removed (the centreline already represents that walkway);
parallel lines ONLY -- crossing/intersecting lines (angle >= 30 deg) are kept even if partly inside the band (they are real crossing paths and must not be deleted).
Output step4_osm_kept.gpkg. pyenv."""
import geopandas as gpd, numpy as np, time
from shapely.ops import unary_union
from shapely import STRtree
OUT=r"D:\Claude\SVI_FFW\output\step4_network"
BUF=4.0; OVL=0.8; PARA_COS=0.866   # removed only if overlap > 80% AND parallel (angle < 30 deg, cos >= 0.866); intersecting/crossing kept
t=time.time()
fac=gpd.read_file(f"{OUT}\\step4_fac_lines.gpkg")
osm=gpd.read_file(f"{OUT}\\step4_osm_lines.gpkg")
mid=fac[fac['src'].isin(['arcade','linkway','facconn'])]
mid_geoms=list(mid.geometry.values)
band=unary_union(mid_geoms).buffer(BUF)
bgeoms=list(band.geoms) if band.geom_type=='MultiPolygon' else [band]
btree=STRtree(bgeoms); mid_tree=STRtree(mid_geoms)
print(f"Facility centrelines {len(mid)} segments -> 8m band ({len(bgeoms)} parts) | OSM {len(osm)} segments | {time.time()-t:.0f}s",flush=True)
def _dir(g):
    c=g.coords; a=c[0]; b=c[-1]; dx=b[0]-a[0]; dy=b[1]-a[1]; L=(dx*dx+dy*dy)**0.5
    return (dx/L,dy/L) if L>1e-9 else (1.0,0.0)
def parallel_to_mid(s):
    """Whether s is parallel to the nearest facility centreline segment (angle < 30 deg): only parallel lines are removed, crossing/intersecting ones are kept."""
    sd=_dir(s); nd=1e9; md=None
    for mi in mid_tree.query(s.buffer(BUF)):
        m=mid_geoms[int(mi)]; dist=m.distance(s)
        if dist<nd: nd=dist; md=_dir(m)
    return md is not None and abs(sd[0]*md[0]+sd[1]*md[1])>=PARA_COS
osm_geoms=list(osm.geometry.values); osm_src=list(osm['src'].values)
keep=[]; ksrc=[]; ncut=0; nkeep_cross=0
for i,s in enumerate(osm_geoms):
    if (i+1)%40000==0: print(f"  {i+1}/{len(osm_geoms)} | {time.time()-t:.0f}s",flush=True)
    ov=0.0; L=s.length
    for bi in btree.query(s):
        try:
            ov+=bgeoms[int(bi)].intersection(s).length
            if ov>L*OVL: break
        except Exception: pass
    if ov>L*OVL:
        if parallel_to_mid(s): ncut+=1; continue      # overlap > 80% AND parallel -> remove
        else: nkeep_cross+=1                            # overlap > 80% but crossing -> keep
    keep.append(s); ksrc.append(osm_src[i])
print(f"Removed (overlap >80% and parallel) {ncut} | crossing kept (high overlap but intersecting) {nkeep_cross} | kept total {len(keep)} | {time.time()-t:.0f}s",flush=True)
out=gpd.GeoDataFrame({'src':ksrc,'geometry':keep},crs=3414); out['length']=out.geometry.length
out.to_file(f"{OUT}\\step4_osm_kept.gpkg",driver="GPKG")
print(f"Done kept OSM {len(out)} segments {out.length.sum()/1000:.0f}km (before removal {sum(s.length for s in osm_geoms)/1000:.0f}km) | {time.time()-t:.0f}s",flush=True)
