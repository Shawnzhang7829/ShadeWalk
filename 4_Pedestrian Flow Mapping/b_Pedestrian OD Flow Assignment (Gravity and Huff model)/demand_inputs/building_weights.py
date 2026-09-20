# -*- coding: utf-8 -*-
"""building_hourly_weight_gfa.gpkg: the hourly building-weight table of the demand model (the destination weights Module 4a reads).

For each building b and slot s = (day type, hour): weight_b(s) = GFA_b x occupancy_density(archetype_b, s), with the gross floor
area GFA_b = `gfa_corr` of the released building dataset (data record, 5_base_data/SG_buildings_footprint_height_function.shp:
storeys x footprint), joined on the OSM id, and the occupancy-density table lookup/occupancy_density.parquet (people per m2 by
archetype and slot, see build_occupancy_table.py).  Geometry, row order and archetypes come from the building file of
Constants.BUILDING_GEOJSON (reprojected to EPSG:3414).

Inputs (Constants.ROOT): building/sg_buildings_v5.geojson (id, gross_floor_area, building_archetype, geometry);
        5_base_data/SG_buildings_footprint_height_function/SG_buildings_footprint_height_function.shp (id, gfa_orig, gfa_corr);
        demand_inputs/lookup/occupancy_density.parquet.
Output: output/building_hourly_weight_gfa.gpkg (+ _report.csv: buildings, gross floor area and 14:00 weight per archetype)
  geometry              : building footprint POLYGON (SVY21)
  building_archetype    : 21 archetypes
  gross_floor_area      : GFA (m2) = gfa_corr
  weight_<daytype>_<HH> : 48 columns of weight_b(slot) (people-equivalent)
  weight_weekday_peak / _total / _peak_hour, weight_weekends_peak : convenience columns
Run:    python demand_inputs/building_weights.py [--out PATH] [--overwrite]
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, pandas as pd, geopandas as gpd, pyogrio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # module root (demand_inputs package)
from demand_inputs import Constants as C

ap = argparse.ArgumentParser(description="hourly building weights, GFA = gfa_corr of the released building dataset")
ap.add_argument("--out", default=str(C.OUTPUT_DIR / "building_hourly_weight_gfa.gpkg"))
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
T0 = time.time()
ZEN = C.BUILDING_RELEASED_SHP
OUT = args.out; GFA, ARCH = C.BUILDING_GFA_COL, C.BUILDING_ARCHETYPE_COL
assert args.overwrite or not os.path.exists(OUT), f"{OUT} exists (pass --overwrite or --out)"

gdf = gpd.read_file(C.BUILDING_GEOJSON, columns=["id", GFA, ARCH, "geometry"]).to_crs(C.TARGET_CRS)
n0 = len(gdf); gdf = gdf.dropna(subset=[GFA]); gdf = gdf[gdf[GFA] > 0]; gdf[ARCH] = gdf[ARCH].fillna("__missing__").astype(str)
print(f"buildings {n0:,} -> after GFA filter {len(gdf):,} | {time.time()-T0:.0f}s", flush=True)
z = pyogrio.read_dataframe(str(ZEN), columns=["id", "gfa_orig", "gfa_corr"], read_geometry=False)
assert z["id"].is_unique and gdf["id"].is_unique
m = gdf[["id", GFA]].merge(z, on="id", how="left", validate="one_to_one"); assert m["gfa_corr"].notna().all(), "unmatched ids"
assert np.allclose(m[GFA].values, m["gfa_orig"].values, rtol=1e-5), "building file gross_floor_area != released gfa_orig (id join check)"
gfa = m["gfa_corr"].astype(np.float64).values; assert (gfa > 0).all()
print(f"join on id: {len(m):,} buildings | gross floor area (gfa_corr) {gfa.sum()/1e6:.2f} M m2", flush=True)
occ = pd.read_parquet(C.OCCUPANCY_DENSITY_PARQUET); archetypes = gdf[ARCH].values

# weight_<daytype>_<HH> = GFA x density(archetype, slot) in float32
W = {}; wd, we = [], []; gf = gfa.astype(np.float32)
for (daytype, hour) in occ.columns:
    tag = f"{daytype.split('/')[0].lower()}_{hour:02d}"
    dens = pd.Series(archetypes).map(occ[(daytype, hour)]).fillna(C.OCCUPANCY_DEFAULT_DENSITY).astype(np.float32).values
    col = f"weight_{tag}"; W[col] = gf * dens; (wd if "weekday" in tag else we).append(col)
W = pd.DataFrame(W, index=gdf.index); wd, we = sorted(wd), sorted(we)
out = gdf[[ARCH]].copy(); out[GFA] = gfa
out["weight_weekday_peak"] = W[wd].max(axis=1).astype(np.float32); out["weight_weekday_total"] = W[wd].sum(axis=1).astype(np.float32)
out["weight_weekday_peak_hour"] = W[wd].values.argmax(axis=1).astype(np.int8); out["weight_weekends_peak"] = W[we].max(axis=1).astype(np.float32)
for c in wd + we: out[c] = W[c].values
out["geometry"] = gdf.geometry.values; out = gpd.GeoDataFrame(out, geometry="geometry", crs=C.TARGET_CRS)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True); out.to_file(OUT, driver="GPKG")
print(f"written {OUT} | {len(out):,} buildings, {len(out.columns)-1} attributes | {time.time()-T0:.0f}s", flush=True)
rep = pd.DataFrame({"archetype": archetypes, "n": 1, "gfa_Mm2": gfa / 1e6, "A14": out["weight_weekday_14"].values}).groupby("archetype").sum()
rep["A14_share_pct"] = 100 * rep["A14"] / rep["A14"].sum(); rep = rep.sort_values("A14", ascending=False)
rep.to_csv(OUT[:-5] + "_report.csv", float_format="%.4f"); print(rep.round(2).to_string())
print(f"14:00 attraction total: {rep['A14'].sum()/1e6:.2f} M persons")
