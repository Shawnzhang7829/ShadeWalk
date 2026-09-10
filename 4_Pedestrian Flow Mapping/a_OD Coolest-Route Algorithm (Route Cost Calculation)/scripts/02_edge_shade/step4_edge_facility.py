# -*- coding: utf-8 -*-
"""Assign a "dominant shade facility" to every network edge (for the background segment colouring of the route shade-profile panel).
Sample 3 points along each edge from the 14:00 shadow category raster (Category band15), take the dominant class,
and map by priority arcade > linkway (LDSM) > tree > building to 0=sun / 1=building / 2=tree / 3=arcade / 4=linkway.
Output edge_facility_SG.npy (row order = step4_4_edges_flow_SG comp==0). pyenv."""
import geopandas as gpd, numpy as np, rasterio, time
from shapely import STRtree
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
CAT=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\merge_images\Category_2pm_h14.tif"   # new arcade category (single frame, band1 = 14:00)
BREM=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\SUB_SG_Polygon_BREMAIN_1m.tif"  # through building interior -> indoor class (5) (old TIF\ archived as TIF[old]; the newarcade version is on the same grid)
t0=time.time()
e=gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg"); e=e[e['comp']==0].reset_index(drop=True)
try: e=e.to_crs(3414)
except Exception: e=e.set_crs(3414,allow_override=True)
n=len(e); print(f"edges {n} | {time.time()-t0:.0f}s",flush=True)
ds=rasterio.open(CAT); cat=ds.read(1); inv=~ds.transform   # the new Category is already a single 14:00 frame
bds=rasterio.open(BREM); brem=bds.read(1); binv=~bds.transform   # building_remain mask (0/1)
print(f"category raster band15 loaded {cat.shape} {cat.dtype} | {time.time()-t0:.0f}s",flush=True)
def cat2fac(c):  # pixel class -> facility major class (priority arcade > linkway > tree > building)
    f=np.zeros(c.shape,np.uint8)
    f[np.isin(c,[8,9,10,11])]=3                      # arcade classes
    m=(f==0)&np.isin(c,[4,5,6,7]); f[m]=4            # linkway / low-canopy LDSM classes
    m=(f==0)&np.isin(c,[2,3]);     f[m]=2            # tree (veg) classes
    m=(f==0)&(c==1);               f[m]=1            # building
    return f
ARC=r"D:\Claude\SVI_FFW\Shp\SG\step2_arcade_sg.gpkg"   # arcade vector (polygons): hard overlay criterion
arc=gpd.read_file(ARC).to_crs(3414); atree=STRtree(arc.geometry.buffer(1.5).values); arc_hit=np.zeros(n,bool)  # 1.5 m buffer tolerance (arcade/building vector alignment + web-side loading error)
print(f"arcade vector {len(arc)} polygons loaded (buffer 1.5 m) | {time.time()-t0:.0f}s",flush=True)
H,W=cat.shape; BH,BW=brem.shape; SAMP=[0.05,0.2,0.35,0.5,0.65,0.8,0.95]; allf=np.zeros((n,len(SAMP)),np.uint8); allb=np.zeros((n,len(SAMP)),bool)
for j,tt in enumerate(SAMP):
    pts=e.geometry.interpolate(tt,normalized=True); xs=pts.x.values; ys=pts.y.values
    cols=np.round(inv.a*xs+inv.b*ys+inv.c).astype(int); rows=np.round(inv.d*xs+inv.e*ys+inv.f).astype(int)
    ok=(rows>=0)&(rows<H)&(cols>=0)&(cols<W); cv=np.zeros(n,np.uint8); cv[ok]=cat[rows[ok],cols[ok]]
    allf[:,j]=cat2fac(cv)
    ah=atree.query(pts.values,predicate='intersects'); arc_hit[ah[0]]=True   # sample point inside an arcade polygon (buffer) -> covered by the arcade vector
    bcols=np.round(binv.a*xs+binv.b*ys+binv.c).astype(int); brows=np.round(binv.d*xs+binv.e*ys+binv.f).astype(int)
    bok=(brows>=0)&(brows<BH)&(bcols>=0)&(bcols<BW); bv=np.zeros(n,np.uint8); bv[bok]=brem[brows[bok],bcols[bok]]; allb[:,j]=bv>0
    print(f"  sample position {tt} | {time.time()-t0:.0f}s",flush=True)
# per-edge dominant priority (any occurrence on the edge counts, higher priority overwrites later): arcade(3) > linkway(4) > indoor(5) > tree(2) > building(1) > sun(0)
# fix: edges under an arcade/linkway corridor are classed as arcade/linkway even if their geometry falls inside the building_remain outline (elevated colonnade / canopy is walkable), no longer swallowed by Indoor
EFAC=np.zeros(n,np.uint8)
EFAC[(allf==1).any(axis=1)]=1                          # building (cast shadow)
EFAC[(allf==2).any(axis=1)]=2                          # tree
EFAC[allb.any(axis=1)]=5                               # inside building_remain -> indoor (laid first, overwritten below by arcade/linkway)
EFAC[(allf==4).any(axis=1)]=4                          # linkway LDSM: overrides indoor (under a linkway canopy, not through the building body)
EFAC[(allf==3).any(axis=1)]=3                          # arcade (Category walkway): overrides Indoor
EFAC[arc_hit]=3                                        # arcade vector overlay (hard criterion): every edge covered by an arcade polygon is Arcade, tolerating building-vector error, never Indoor
np.save(f"{OUT}\\edge_facility_SG.npy",EFAC)
u,c=np.unique(EFAC,return_counts=True)
print("distribution 0 sun/1 building/2 tree/3 arcade/4 linkway/5 indoor:",dict(zip(u.tolist(),c.tolist())),f"| {time.time()-t0:.0f}s",flush=True)
print("DONE",flush=True)
