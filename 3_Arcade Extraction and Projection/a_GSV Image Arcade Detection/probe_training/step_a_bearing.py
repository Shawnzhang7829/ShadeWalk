"""Step A (pyenv python): compute for each positive sample the bearing to the nearest CLW∩Building overlap
and flag faces_clw. Also samples pos/neg. The output is used by Step B (CLIP)."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(r"D:\Claude\SVI_FFW")
OUT = ROOT / "output" / "phase1c_sg_val"
EPSG = 3414
FOV_HALF = 55.0
N_POS, N_NEG, SEED = 1500, 1500, 42

pos = pd.read_csv(OUT/"sg_val_positive.csv").sample(n=1500, random_state=SEED).reset_index(drop=True)
neg = pd.read_csv(OUT/"sg_val_negative.csv").sample(n=1500, random_state=SEED).reset_index(drop=True)

clw = gpd.read_file(ROOT/"Shp"/"SG"/"CoveredLinkWay_Mar2026"/"CoveredLinkWay.shp").to_crs(EPSG)
bld = gpd.read_file(ROOT/"Shp"/"SG"/"SG_Building"/"SG_Building_SVY21_TH.shp").to_crs(EPSG)
clw["geometry"]=clw.geometry.buffer(0); bld["geometry"]=bld.geometry.buffer(0)
inter = gpd.overlay(clw[["OBJECTID","geometry"]], bld[["geometry"]], how="intersection", keep_geom_type=True)
inter = inter[inter.geometry.area>=3.0]
cent = inter.geometry.centroid
ov = np.array([(g.x,g.y) for g in cent])

pts = gpd.GeoDataFrame(pos.copy(), geometry=gpd.points_from_xy(pos.lon,pos.lat,crs="EPSG:4326")).to_crs(EPSG)
bearings=[]
for geom in pts.geometry:
    dx=ov[:,0]-geom.x; dy=ov[:,1]-geom.y
    k=int(np.argmin(dx*dx+dy*dy))
    bearings.append(math.degrees(math.atan2(dx[k],dy[k]))%360)
pos["bearing_to_clw"]=bearings
dvh=(pos["viewheading"]-pos["bearing_to_clw"]).abs()%360
dvh=dvh.where(dvh<=180,360-dvh)
pos["faces_clw"]=dvh<FOV_HALF
print(f"positives facing CLW (|d|<{FOV_HALF}): {int(pos.faces_clw.sum())}/{len(pos)}")

pos.to_csv(OUT/"eval_pos_prepped.csv",index=False)
neg.to_csv(OUT/"eval_neg_prepped.csv",index=False)
print("wrote eval_pos_prepped.csv / eval_neg_prepped.csv")
