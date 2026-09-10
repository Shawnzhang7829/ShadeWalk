"""Where are the NEW-only arcade runs? Overlay map: grey = runs shared with production,
red = stretches only in the precise geometry-first output (broad-screen recoveries).
Top: full city (SG, BO). Bottom: zoom on the densest added cluster per city."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.ops import unary_union
plt.rcParams["font.sans-serif"]=["Arial","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
import sys; sys.path.insert(0, r"D:\Claude\SVI_FFW\output\step2_projection")
from mapviz_util import add_scalebar

ROOT=Path(r"D:\Claude\SVI_FFW"); OLD=ROOT/"output"/"step2_projection"
VEC=ROOT/"output"/"detection_geomfirst"/"precise"/"vectors"
CFG={"sg":dict(epsg=3414,bld=ROOT/"Shp"/"SG"/"SG_Building"/"SG_Building_SVY21_TH.shp",
               base=ROOT/"Shp"/"SG"/"SG_Subzone"/"SG_subzone boundary 2019_SVY21.shp",name="Singapore — five-foot ways"),
     "bo":dict(epsg=32632,bld=ROOT/"Shp"/"Bologna"/"c_a944ctc_edifici_pl.geojson",
               base=ROOT/"Shp"/"Bologna"/"utm_etrf"/"zone.shp",name="Bologna — porticoes")}

fig,axes=plt.subplots(2,2,figsize=(20,17))
res={}
for j,city in enumerate(["sg","bo"]):
    c=CFG[city]
    old=gpd.read_file(OLD/f"step2b_runs_{city}.gpkg").to_crs(c["epsg"])
    new=gpd.read_file(VEC/f"step2b_runs_{city}.gpkg").to_crs(c["epsg"])
    oldu=unary_union(old.geometry.values).buffer(10)
    added=[]
    for g in new.geometry.values:
        d=g.difference(oldu)
        if d.is_empty: continue
        for gg in ([d] if d.geom_type=="LineString" else list(getattr(d,"geoms",[]))):
            if gg.geom_type=="LineString" and gg.length>=15: added.append(gg)
    add=gpd.GeoDataFrame(geometry=added,crs=c["epsg"]); akm=add.length.sum()/1000
    res[city]=(len(add),akm)
    print(f"[{city}] added pieces {len(add)}, {akm:.1f} km")
    # ---- full city ----
    ax=axes[0,j]
    base=gpd.read_file(c["base"]).to_crs(c["epsg"]); base.plot(ax=ax,fc="#f5f5f5",ec="#ccc",lw=0.3)
    old.plot(ax=ax,color="#9aa6b2",lw=0.45)
    add.plot(ax=ax,color="#d40000",lw=1.1)
    ax.set_title(f"{c['name']}\ngrey = shared with production ({len(old)} runs)   ·   "
                 f"red = ADDED by geometry-first (+{akm:.1f} km, {len(add)} stretches)",fontsize=12.5)
    ax.set_aspect("equal"); ax.axis("off"); add_scalebar(ax)
    # ---- zoom on densest added cluster (800 m grid cell with max added length) ----
    cx=np.array([g.centroid.x for g in added]); cy=np.array([g.centroid.y for g in added])
    ln=np.array([g.length for g in added])
    gx=np.floor(cx/800); gy=np.floor(cy/800)
    keys={}
    for k in range(len(ln)): keys[(gx[k],gy[k])]=keys.get((gx[k],gy[k]),0)+ln[k]
    (bx,by),_=max(keys.items(),key=lambda kv:kv[1])
    x0,y0=bx*800+400,by*800+400; W=700
    axz=axes[1,j]; bbox=(x0-W,y0-W,x0+W,y0+W)
    bld=gpd.read_file(c["bld"],bbox=bbox).to_crs(c["epsg"])
    bld.plot(ax=axz,fc="#e2e2e2",ec="#bbb",lw=0.3)
    old.cx[x0-W:x0+W,y0-W:y0+W].plot(ax=axz,color="#5a7ba6",lw=2.0)
    add.cx[x0-W:x0+W,y0-W:y0+W].plot(ax=axz,color="#d40000",lw=2.6)
    axz.set_xlim(x0-W,x0+W); axz.set_ylim(y0-W,y0+W)
    axz.set_title(f"zoom — densest ADDED cluster   (blue = production runs, red = added)",fontsize=11.5)
    axz.set_aspect("equal"); axz.axis("off"); add_scalebar(axz)

fig.suptitle("Geometry-first vs production — where the ADDED arcade runs are\n"
             "(additions = broad-screen misses recovered; same probe & same vector rules)",fontsize=15.5,y=0.995)
plt.tight_layout(rect=[0,0,1,0.965])
for p in [ROOT/"output"/"detection_geomfirst"/"precise"/"vectors_added_map.png", ROOT/"output"/"vectors_added_map.png"]:
    plt.savefig(p,dpi=130,bbox_inches="tight")
print("wrote vectors_added_map.png")
