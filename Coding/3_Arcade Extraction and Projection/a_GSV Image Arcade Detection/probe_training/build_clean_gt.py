"""Build a clean SG GT (pyenv/GIS):
  positives: CLW∩building with bearing filter (only views facing the arcade) → clean five-foot way
  negatives: far from CLW (≥80m) + far from conservation areas (≥50m) → clean open street
Sample both classes and tile a montage for visual checking by Claude.
"""
from __future__ import annotations
import math, re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(r"D:\Claude\SVI_FFW")
OUT = ROOT / "output" / "sg_improve"
OUT.mkdir(parents=True, exist_ok=True)
EPSG = 3414
FOV_HALF = 50.0
N_POS, N_NEG, SEED = 2000, 1500, 7   # positives are automatically capped at the available maximum (~960)

NAME_RE = re.compile(
    r"^(?P<pid>\d+)_(?P<lon>-?\d+\.\d+)_(?P<lat>-?\d+\.\d+)"
    r"_(?P<date>\d{6})_baseheading(?P<bh>-?\d+\.\d+)"
    r"_viewheading(?P<vh>-?\d+\.\d+)_(?P<view>\d)\.jpg$")

def index_svi():
    rows=[]
    for p in (ROOT/"SVI"/"Singapore"/"Singapore SVI").iterdir():
        m=NAME_RE.match(p.name)
        if m: rows.append({"pid":int(m["pid"]),"lon":float(m["lon"]),"lat":float(m["lat"]),
                           "viewheading":float(m["vh"]),"view_idx":int(m["view"]),"path":str(p)})
    return pd.DataFrame(rows)

print("[1] CLW∩building overlap …")
clw=gpd.read_file(ROOT/"Shp"/"SG"/"CoveredLinkWay_Mar2026"/"CoveredLinkWay.shp").to_crs(EPSG)
bld=gpd.read_file(ROOT/"Shp"/"SG"/"SG_Building"/"SG_Building_SVY21_TH.shp").to_crs(EPSG)
clw["geometry"]=clw.geometry.buffer(0); bld["geometry"]=bld.geometry.buffer(0)
inter=gpd.overlay(clw[["OBJECTID","geometry"]],bld[["geometry"]],how="intersection",keep_geom_type=True)
inter=inter[inter.geometry.area>=3.0]
ov_union=inter.geometry.union_all(); clw_union=clw.geometry.union_all()
cons=gpd.read_file(ROOT/"Shp"/"SG"/"SG ConservationArea2025"/"MasterPlan2025ConservationAreaBoundaryLayer.geojson").to_crs(EPSG)
cons_union=cons.geometry.union_all()
ov_cent=np.array([(g.x,g.y) for g in inter.geometry.centroid])

print("[2] index SVI + distances …")
df=index_svi()
g=gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lon,df.lat,crs="EPSG:4326")).to_crs(EPSG)
uniq=g.drop_duplicates("pid")[["pid","geometry"]].copy()
uniq["d_ov"]=uniq.geometry.distance(ov_union)
uniq["d_clw"]=uniq.geometry.distance(clw_union)
uniq["d_cons"]=uniq.geometry.distance(cons_union)
for c in ["d_ov","d_clw","d_cons"]:
    g[c]=g["pid"].map(uniq.set_index("pid")[c])

# positives: ≤12m to overlap + bearing facing it
pos=g[g.d_ov<=12].copy()
pts=np.array([(geom.x,geom.y) for geom in pos.geometry])
bear=[]
for (x,y) in pts:
    dx=ov_cent[:,0]-x; dy=ov_cent[:,1]-y; k=int(np.argmin(dx*dx+dy*dy))
    bear.append(math.degrees(math.atan2(dx[k],dy[k]))%360)
pos["bearing"]=bear
dvh=(pos.viewheading-pos.bearing).abs()%360; dvh=dvh.where(dvh<=180,360-dvh)
pos=pos[dvh<FOV_HALF].copy()
print(f"    clean positives (facing, <=12m): {len(pos)}")

# negatives: far from CLW and far from conservation areas
neg=g[(g.d_clw>=80)&(g.d_cons>=50)].copy()
print(f"    candidate negatives: {len(neg)}")

pos_s=pos.sample(n=min(N_POS,len(pos)),random_state=SEED)
neg_s=neg.sample(n=min(N_NEG,len(neg)),random_state=SEED)
pos_s["cand_label"]=1; neg_s["cand_label"]=0
cand=pd.concat([pos_s,neg_s]).reset_index(drop=True)
cand["cell_id"]=range(len(cand))
cand.drop(columns="geometry").to_csv(OUT/"gt_candidates.csv",index=False)
print(f"[done] candidates: {len(cand)} ({cand.cand_label.sum()} pos / {(cand.cand_label==0).sum()} neg)")
print(f"       wrote {OUT/'gt_candidates.csv'}")
