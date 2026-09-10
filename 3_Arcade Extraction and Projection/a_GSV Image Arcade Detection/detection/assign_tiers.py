"""Unified detection - step0: assign a region_tier to every pano. (geopandas / pyenv environment)

Replaces the geometry part of build_manifest.py. The only difference from the old version: no more
2/4-view subsampling for Tier3 (all 4 views go to detect_arcade.py), so the output is a pano-level
tier table rather than a per-view filtered manifest.

Tier definition (same as the old version):
  Tier1 near official arcades    SG: CoveredLinkWay∩Building ≤12m   BO: UNESCO portici lines ≤10m
  Tier2 conservation area (+30m) SG: MasterPlan2025 Conservation    BO: Bologna old-town zone
  Tier3 everything else

Output: detection/pano_tier_{city}.csv  (pid, lon, lat, region_tier)
Usage: python assign_tiers.py sg   /   python assign_tiers.py bo
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path(r"D:\Claude\SVI_FFW"); OUT = ROOT/"output"/"detection"; OUT.mkdir(parents=True, exist_ok=True)
CONS_BUFFER = 30.0; TIER1_DIST = {"sg": 12.0, "bo": 10.0}
CFG = {
 "sg": dict(svi=ROOT/"SVI"/"Singapore"/"Singapore SVI", epsg=3414,
            cons=ROOT/"Shp"/"SG"/"SG ConservationArea2025"/"MasterPlan2025ConservationAreaBoundaryLayer.geojson", cons_layer=None, tier1="clw"),
 "bo": dict(svi=ROOT/"SVI"/"Bologna"/"Bologna SVI", epsg=32632,
            cons=ROOT/"Shp"/"Bologna"/"Bologna_Arcade.gpkg", cons_layer="zone", tier1="portici"),
}
NAME_RE = re.compile(r"^(?P<pid>\d+)_(?P<lon>-?\d+\.\d+)_(?P<lat>-?\d+\.\d+)_\d{6}_baseheading-?\d+\.\d+_viewheading-?\d+\.\d+_\d\.jpg$")

def unique_panos(svi):
    seen={}
    for p in svi.iterdir():
        m=NAME_RE.match(p.name)
        if m and int(m["pid"]) not in seen: seen[int(m["pid"])]=(float(m["lon"]),float(m["lat"]))
    return pd.DataFrame([(k,v[0],v[1]) for k,v in seen.items()], columns=["pid","lon","lat"])

def tier1_geom(city, epsg):
    if city=="sg":
        clw=gpd.read_file(ROOT/"Shp"/"SG"/"CoveredLinkWay_Mar2026"/"CoveredLinkWay.shp").to_crs(epsg)
        bld=gpd.read_file(ROOT/"Shp"/"SG"/"SG_Building"/"SG_Building_SVY21_TH.shp").to_crs(epsg)
        clw["geometry"]=clw.geometry.buffer(0); bld["geometry"]=bld.geometry.buffer(0)
        inter=gpd.overlay(clw[["OBJECTID","geometry"]], bld[["geometry"]], how="intersection", keep_geom_type=True)
        return inter[inter.geometry.area>=3.0].geometry.union_all()
    por=gpd.read_file(ROOT/"Shp"/"Bologna"/"origini-di-bologna-portici"/"origini-di-bologna-portici.shp").to_crs(epsg)
    return por.geometry.union_all()

def main(city):
    c=CFG[city]; epsg=c["epsg"]
    print(f"=== assign tiers: {city.upper()} ===")
    up=unique_panos(c["svi"]); print(f"  panos: {len(up)}")
    g=gpd.GeoDataFrame(up, geometry=gpd.points_from_xy(up.lon,up.lat,crs="EPSG:4326")).to_crs(epsg)
    cons=gpd.read_file(c["cons"], layer=c["cons_layer"]) if c["cons_layer"] else gpd.read_file(c["cons"])
    cons_buf=cons.to_crs(epsg).geometry.buffer(CONS_BUFFER).union_all()
    g["in_cons"]=g.geometry.within(cons_buf)
    t1=tier1_geom(city, epsg).buffer(TIER1_DIST[city])
    g["in_t1"]=g.geometry.within(t1)
    g["region_tier"]=g.apply(lambda r: 1 if r["in_t1"] else (2 if r["in_cons"] else 3), axis=1)
    out=g[["pid","lon","lat","region_tier"]].sort_values("pid")
    out.to_csv(OUT/f"pano_tier_{city}.csv", index=False)
    for t in (1,2,3): print(f"  Tier{t}: {int((out.region_tier==t).sum())} panos")
    print(f"  wrote pano_tier_{city}.csv")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "sg")
