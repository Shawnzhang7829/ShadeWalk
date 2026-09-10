"""
Patronage_Flow / Main.py    (v2 — madina + buildings)

End-to-end orchestrator. Reads everything in Constants.py, builds the
madina Zonal, computes hourly betweenness flows, and writes outputs.

Run:
    python -m Patronage_Flow.Main           # full Singapore (slow)
    set SMOKE_BBOX in Constants.py for a small bbox first.

Outputs in Patronage_Flow/output/:
    edges.gpkg               -- madina edges with parent_street_id + geometry
    station_snap.gpkg        -- stations as origin nodes after snap (QC layer)
    flow_long.parquet        -- long table: edge_id x DAY_TYPE x HOUR -> flow
    flow_<daytype>_<HH>.gpkg -- snapshot per representative hour for QGIS
"""
from __future__ import annotations
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from Patronage_Flow import Constants as C
from Patronage_Flow.Network import (
    load_pedestrian_highway, load_stations, load_buildings, build_zonal,
)
from Patronage_Flow.Flow_Computation import (
    load_ridership_table, compute_hourly_flow,
)


def _bbox_or_none():
    if C.SMOKE_BBOX is None:
        return None
    return tuple(C.SMOKE_BBOX)


def main():
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bbox = _bbox_or_none()
    if bbox:
        print(f"[main] SMOKE BBOX active = {bbox}")

    # --- 1. data ---
    hw       = load_pedestrian_highway(bbox=bbox)
    stations = load_stations(bbox=bbox)
    buildings = load_buildings(bbox=bbox)

    # stations need 0..N-1 integer index so madina's source_id maps cleanly
    stations = stations.reset_index(drop=True)
    buildings = buildings.reset_index(drop=True)

    # export the final pedestrian network (vehicle-only roads removed,
    # clipped to the island boundary) for QA in QGIS
    hw_export = hw.copy()
    hw_export["length_m"] = hw_export.geometry.length
    hw_export_path = C.OUTPUT_DIR / "pedestrian_network_filtered.gpkg"
    hw_export.to_file(hw_export_path, driver="GPKG")
    print(f"[main] wrote {hw_export_path.name} ({len(hw_export):,} edges, "
          f"total length = {hw_export['length_m'].sum()/1000:.1f} km)")

    # --- 2. madina zonal ---
    z = build_zonal(hw, stations, buildings)

    # snapshot of the canvas
    edges_gdf = z.network.edges.copy()
    edges_gdf.to_file(C.OUTPUT_DIR / "edges.gpkg", driver="GPKG")
    origin_nodes = z.network.nodes[z.network.nodes["type"] == "origin"].copy()
    origin_nodes.to_file(C.OUTPUT_DIR / "station_snap.gpkg", driver="GPKG")

    # --- 3. ridership + hourly betweenness ---
    # output_dir: write one GPKG per slot immediately after it completes,
    # so results are visible in QGIS before all slots finish.
    ridership = load_ridership_table()
    flow_long = compute_hourly_flow(
        z, stations, buildings, ridership,
        output_dir=C.OUTPUT_DIR,
    )

    # --- 4. write parquet summary ---
    flow_long.to_parquet(C.OUTPUT_DIR / "flow_long.parquet", index=False)
    print(f"[main] wrote flow_long.parquet ({len(flow_long):,} rows)")


if __name__ == "__main__":
    main()
