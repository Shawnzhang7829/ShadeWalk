"""Step2c: arcade run -> buffer towards the building interior -> arcade height field -> split the arcade out of the building.
  arcade strip = building footprint ∩ run.buffer(SG 1.5 / BO 3 m)  (the arcade occupies the ground floor of the building)
  arcade height = min(default arcade height, building height)
  building_remain = footprint - arcade strip
Usage: python step2_buffer_split.py sg / bo
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

ROOT=Path(r"D:\Claude\SVI_FFW"); OUT=ROOT/"output"/"step2_projection"
CFG={"sg":dict(bld=ROOT/"Shp"/"SG"/"SG_Building"/"SG_Building_SVY21_TH.shp",epsg=3414,hcol="height",buf=2.0,arc_h=3.6),  # buf 1.5 -> 2.0 (user rule 15: arcade depth 2 m towards the building interior)
     "bo":dict(bld=ROOT/"Shp"/"Bologna"/"c_a944ctc_edifici_pl.geojson",epsg=32632,hcol="altezza_gr",buf=3.0,arc_h=3.6),
     "gz":dict(bld=ROOT/"Shp"/"GZ"/"GZ_core_building"/"GZ_core_building.shp",epsg=32649,hcol="Height",buf=2.0,arc_h=3.6)}  # Guangzhou: depth 2 m, arcade height 3.6, same as SG

def main(city):
    c=CFG[city]
    runs=gpd.read_file(OUT/f"step2b_runs_{city}.gpkg")
    print(f"[{city}] arcade runs: {len(runs)}")
    bld=gpd.read_file(c["bld"]).to_crs(c["epsg"])
    bld=bld[bld.geometry.notna()&bld.geometry.is_valid].reset_index(drop=True); bld["geometry"]=bld.geometry.buffer(0)
    bld["_bh"]=pd.to_numeric(bld[c["hcol"]],errors="coerce").fillna(0)

    # arcade buffer strip (buffer the run by buf on both sides, then intersect with the buildings -> the part inside the building = ground-floor arcade)
    # cap_style=2 flat caps (no rounded ends); join_style=2 mitre joins (no rounded arcs bulging out at corners)
    arc_buf=unary_union(runs.geometry.values).buffer(c["buf"],cap_style=2,join_style=2)
    arc_gdf=gpd.GeoDataFrame(geometry=[arc_buf],crs=c["epsg"])

    print(f"[{city}] splitting buildings ...")
    # intersect with the buildings -> arcade strip (carries the building height)
    arcade=gpd.overlay(bld[["_bh","geometry"]],arc_gdf,how="intersection",keep_geom_type=True)
    arcade=arcade[arcade.geometry.area>=2.0].reset_index(drop=True)
    arcade["arc_h"]=np.minimum(c["arc_h"], arcade["_bh"].where(arcade["_bh"]>0, c["arc_h"]))  # arcade height <= building height
    arcade=arcade.rename(columns={"_bh":"bld_h"})
    # building_remain = original building - arcade strip
    remain=gpd.overlay(bld,arc_gdf,how="difference",keep_geom_type=True)

    arcade.to_file(OUT/f"step2_arcade_{city}.gpkg",driver="GPKG")
    remain.to_file(OUT/f"step2_building_remain_{city}.gpkg",driver="GPKG")
    print(f"[{city}] arcade strip polygons: {len(arcade)}, total area {arcade.geometry.area.sum():.0f} m², mean arcade height {arcade['arc_h'].mean():.2f}m")
    print(f"[{city}] building_remain: {len(remain)}")
    print(f"   wrote step2_arcade_{city}.gpkg + step2_building_remain_{city}.gpkg")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "sg")
