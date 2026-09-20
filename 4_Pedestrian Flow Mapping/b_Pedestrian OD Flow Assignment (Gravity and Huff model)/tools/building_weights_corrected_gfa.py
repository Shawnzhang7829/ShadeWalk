# -*- coding: utf-8 -*-
"""building_hourly_weight_gfacorr.gpkg: the building-weight table of the demand model, gross floor area = `gfa_corr` of the
released building dataset (data record, 5_base_data/SG_buildings_footprint_height_function.shp: resolved storeys x footprint).

Same procedure as export_building_hourly.py: the building file of Constants.BUILDING_GEOJSON (geometry, row order, archetypes,
reprojected to EPSG:3414), the occupancy-density table lookup/occupancy_density.parquet, weight_<daytype>_<HH> = GFA x
density(archetype, slot), the four convenience columns and the same column order; the GFA of every building is taken from the
released dataset by the OSM id.  Consistency check: run with the building file's own gross_floor_area, the code reproduces the
export's building_hourly_weight.gpkg to float32 precision (skipped when that file is absent).

Inputs (Constants.ROOT): building/sg_buildings_v5.geojson (id, gross_floor_area, building_archetype, geometry);
        5_base_data/SG_buildings_footprint_height_function/SG_buildings_footprint_height_function.shp (id, gfa_orig, gfa_corr, bf_area);
        Patronage_Flow/lookup/occupancy_density.parquet.
Output: output/building_hourly_weight_gfacorr.gpkg (+ _report.csv: per-archetype GFA and 14:00 weights, building-file value and
        gfa_corr side by side), the table Module 4a reads.
Run:    python tools/building_weights_corrected_gfa.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd, geopandas as gpd, pyogrio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # module root (Patronage_Flow package)
from Patronage_Flow import Constants as C

T0 = time.time()
ZEN = C.ROOT / "5_base_data" / "SG_buildings_footprint_height_function" / "SG_buildings_footprint_height_function.shp"
OLD = C.OUTPUT_DIR / "building_hourly_weight.gpkg"; OUT = C.OUTPUT_DIR / "building_hourly_weight_gfacorr.gpkg"
GFA, ARCH = C.BUILDING_GFA_COL, C.BUILDING_ARCHETYPE_COL
assert not OUT.exists(), f"refusing to overwrite {OUT}"

gdf = gpd.read_file(C.BUILDING_GEOJSON, columns=["id", GFA, ARCH, "geometry"]).to_crs(C.TARGET_CRS)
n0 = len(gdf); gdf = gdf.dropna(subset=[GFA]); gdf = gdf[gdf[GFA] > 0]; gdf[ARCH] = gdf[ARCH].fillna("__missing__").astype(str)
print(f"buildings {n0:,} -> after GFA filter {len(gdf):,} | {time.time()-T0:.0f}s", flush=True)
z = pyogrio.read_dataframe(str(ZEN), columns=["id", "gfa_orig", "gfa_corr", "bf_area"], read_geometry=False)
assert z["id"].is_unique and gdf["id"].is_unique
m = gdf[["id", GFA]].merge(z, on="id", how="left", validate="one_to_one"); assert m["gfa_corr"].notna().all(), "unmatched ids"
assert np.allclose(m[GFA].values, m["gfa_orig"].values, rtol=1e-5), "building file gross_floor_area != released gfa_orig (id join check)"
print(f"join on id: {len(m):,} buildings | gfa_corr total {m['gfa_corr'].sum()/1e6:.2f} M m2", flush=True)
gfa_new = m["gfa_corr"].astype(np.float64).values; assert (gfa_new > 0).all()
occ = pd.read_parquet(C.OCCUPANCY_DENSITY_PARQUET); archetypes = gdf[ARCH].values


def weights(gfas):
    """export_building_hourly.py arithmetic: weight_<daytype>_<HH> = GFA x density(archetype, slot) in float32."""
    out = {}; wd, we = [], []; gf = gfas.astype(np.float32)
    for (daytype, hour) in occ.columns:
        tag = f"{daytype.split('/')[0].lower()}_{hour:02d}"
        dens = pd.Series(archetypes).map(occ[(daytype, hour)]).fillna(C.OCCUPANCY_DEFAULT_DENSITY).astype(np.float32).values
        col = f"weight_{tag}"; out[col] = gf * dens; (wd if "weekday" in tag else we).append(col)
    W = pd.DataFrame(out, index=gdf.index)
    W["weight_weekday_peak"] = W[wd].max(axis=1).astype(np.float32); W["weight_weekday_total"] = W[wd].sum(axis=1).astype(np.float32)
    W["weight_weekday_peak_hour"] = W[wd].values.argmax(axis=1).astype(np.int8); W["weight_weekends_peak"] = W[we].max(axis=1).astype(np.float32)
    return W, sorted(wd), sorted(we)


W0, wd, we = weights(gdf[GFA].values)
if OLD.exists():
    old = pyogrio.read_dataframe(str(OLD), read_geometry=False); assert len(old) == len(W0)
    for c in ("weight_weekday_14", "weight_weekday_peak", "weight_weekday_total", "weight_weekends_peak"):
        assert np.allclose(old[c].values, W0[c].values, rtol=1e-5, atol=1e-4), c
    assert (old["weight_weekday_peak_hour"].values == W0["weight_weekday_peak_hour"].values).all()
    print("consistency check: the building file's gross_floor_area reproduces building_hourly_weight.gpkg", flush=True)
W1, _, _ = weights(gfa_new)
out = gdf[[ARCH]].copy(); out[GFA] = gfa_new
for c in ["weight_weekday_peak", "weight_weekday_total", "weight_weekday_peak_hour", "weight_weekends_peak"] + wd + we: out[c] = W1[c].values
out["geometry"] = gdf.geometry.values; out = gpd.GeoDataFrame(out, geometry="geometry", crs=C.TARGET_CRS)
C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True); out.to_file(OUT, driver="GPKG")
print(f"written {OUT} | {len(out):,} buildings, {len(out.columns)-1} attributes | {time.time()-T0:.0f}s", flush=True)
rep = pd.DataFrame({"archetype": archetypes, "n": 1, "gfa_old_Mm2": gdf[GFA].values / 1e6, "gfa_new_Mm2": gfa_new / 1e6,
                    "A14_old": W0["weight_weekday_14"].values, "A14_new": W1["weight_weekday_14"].values}).groupby("archetype").sum()
rep["gfa_ratio"] = rep["gfa_new_Mm2"] / rep["gfa_old_Mm2"]; rep["A14_share_old_pct"] = 100 * rep["A14_old"] / rep["A14_old"].sum(); rep["A14_share_new_pct"] = 100 * rep["A14_new"] / rep["A14_new"].sum()
rep.sort_values("A14_new", ascending=False).to_csv(str(OUT)[:-5] + "_report.csv", float_format="%.4f")
print(rep.sort_values("A14_new", ascending=False).round(2).to_string())
print(f"14:00 attraction total: {rep['A14_old'].sum()/1e6:.2f} M -> {rep['A14_new'].sum()/1e6:.2f} M persons")
