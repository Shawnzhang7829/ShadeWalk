"""
Export per-building hourly destination weight for QA / mapping.

For each building b and each slot s = (DAY_TYPE, HOUR):
    W_b(s) = GFA_b * People_per_m2(archetype_b) * Schedule(archetype_b, s)
           = GFA_b * occupancy_density(archetype_b, s)
This is exactly the destination weight madina sees in the betweenness call.

Output: Patronage_Flow/output/building_hourly_weight.gpkg
  geometry              : building footprint POLYGON (SVY21)
  building_archetype    : 21 SGP archetypes
  gross_floor_area      : GFA (m^2)
  weight_<daytype>_<HH> : 48 columns of W_b(slot) (people-equivalent)
  weight_weekday_peak   : max across 24h weekday  (good single-frame map)
  weight_weekday_total  : sum across 24h weekday  (volume measure)

Lightweight: only reads geojson + occupancy parquet. Safe to run while a
betweenness pass is in progress (no madina, no network build).

Run:
    python -m Patronage_Flow.export_building_hourly
"""
from __future__ import annotations
import warnings
from shapely.geometry import box
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

from Patronage_Flow import Constants as C
from Patronage_Flow.Network import load_island_border, _filter_to_border


def main():
    bbox = tuple(C.SMOKE_BBOX) if C.SMOKE_BBOX is not None else None
    if bbox:
        bb = tuple((gpd.GeoSeries([box(*bbox)], crs=C.TARGET_CRS)
                       .to_crs("EPSG:4326").total_bounds).tolist())
        print(f"[bldg-export] SMOKE BBOX = {bbox}")
    else:
        bb = None

    print(f"[bldg-export] reading {C.BUILDING_GEOJSON.name} (full polygons) ...")
    cols = [C.BUILDING_GFA_COL, C.BUILDING_ARCHETYPE_COL, "geometry"]
    gdf = gpd.read_file(C.BUILDING_GEOJSON, bbox=bb, columns=cols).to_crs(C.TARGET_CRS)
    print(f"[bldg-export]   raw rows: {len(gdf):,}")

    gdf = gdf.dropna(subset=[C.BUILDING_GFA_COL])
    gdf = gdf[gdf[C.BUILDING_GFA_COL] > 0]
    gdf[C.BUILDING_ARCHETYPE_COL] = (gdf[C.BUILDING_ARCHETYPE_COL]
                                       .fillna("__missing__").astype(str))
    print(f"[bldg-export]   after GFA filter: {len(gdf):,}")

    # Note: do NOT clip to Island_boarder so polygons stay continuous;
    # buildings outside the boundary will simply be inert (won't show up
    # near any origin) but the polygon is still shown for context.

    print(f"[bldg-export] loading occupancy density table ...")
    occ = pd.read_parquet(C.OCCUPANCY_DENSITY_PARQUET)

    archetypes = gdf[C.BUILDING_ARCHETYPE_COL].values
    gfas = gdf[C.BUILDING_GFA_COL].astype(np.float32).values

    weekday_cols, weekend_cols = [], []
    for (daytype, hour) in occ.columns:
        slot_tag = f"{daytype.split('/')[0].lower()}_{hour:02d}"
        density_per_arch = occ[(daytype, hour)]
        densities = (pd.Series(archetypes).map(density_per_arch)
                                          .fillna(0.0).astype(np.float32).values)
        col = f"weight_{slot_tag}"
        gdf[col] = gfas * densities
        if "weekday" in slot_tag:
            weekday_cols.append(col)
        else:
            weekend_cols.append(col)

    # convenience aggregates
    gdf["weight_weekday_peak"]  = gdf[weekday_cols].max(axis=1).astype(np.float32)
    gdf["weight_weekday_total"] = gdf[weekday_cols].sum(axis=1).astype(np.float32)
    gdf["weight_weekday_peak_hour"] = (gdf[weekday_cols].values.argmax(axis=1)
                                                                .astype(np.int8))
    gdf["weight_weekends_peak"]  = gdf[weekend_cols].max(axis=1).astype(np.float32)

    keep = (["building_archetype", "gross_floor_area",
             "weight_weekday_peak", "weight_weekday_total",
             "weight_weekday_peak_hour", "weight_weekends_peak",
             "geometry"]
            + sorted(weekday_cols) + sorted(weekend_cols))
    out = gdf[keep].copy()
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=C.TARGET_CRS)

    out_path = C.OUTPUT_DIR / "building_hourly_weight.gpkg"
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[bldg-export] writing {out_path.name} ...")
    out.to_file(out_path, driver="GPKG")
    sz = out_path.stat().st_size / 1e6
    print(f"[bldg-export] done. {len(out):,} buildings, {len(out.columns)-1} attrs, "
          f"{sz:.1f} MB")

    # console report
    print()
    print("--- top 10 buildings by weekday peak weight ---")
    top = (out.sort_values("weight_weekday_peak", ascending=False)
              .head(10)[["building_archetype", "gross_floor_area",
                          "weight_weekday_peak_hour", "weight_weekday_peak",
                          "weight_weekday_total"]])
    print(top.to_string())
    print()
    print("--- archetype share of weekday total weight ---")
    by_arch = (out.groupby("building_archetype")
                  .agg(n=("gross_floor_area", "size"),
                       total_W=("weight_weekday_total", "sum"))
                  .sort_values("total_W", ascending=False))
    by_arch["pct"] = (by_arch["total_W"] / by_arch["total_W"].sum() * 100).round(1)
    print(by_arch.head(12).to_string())


if __name__ == "__main__":
    main()
