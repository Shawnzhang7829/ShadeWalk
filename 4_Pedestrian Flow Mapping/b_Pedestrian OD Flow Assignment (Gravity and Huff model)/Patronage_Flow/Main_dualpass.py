"""
Patronage_Flow / Main_dualpass.py   (v6 — true dual-pass)

Full-island production script for the TRUE dual-pass betweenness model.
Designed to run on the 100-core server:

    python -m Patronage_Flow.Main_dualpass

Architecture (see _smoke_v6_dualpass.py for the proof-of-concept):

  PASS A  (forward zonal, stations → buildings)
    origin  = stations, weight = tap_out ridership / n_exits
    dest    = buildings, weight = GFA * occupancy_density(archetype, slot)
    → models post-alighting walks: people leaving transit and walking to destinations

  PASS B  (reverse zonal, buildings → stations)
    origin  = buildings, weight = GFA * occupancy_density(archetype, slot)
    dest    = stations,  weight = tap_in ridership / n_exits
    → models pre-boarding walks: people leaving origins and walking to transit

  Per slot, 4 sub-passes:
    A1: bus  origins  @ R = BUS_RADIUS_M  (400 m)  — forward zonal
    A2: rail origins  @ R = RAIL_RADIUS_M (800 m)  — forward zonal
    B1: bus  dest     @ R = BUS_RADIUS_M  (400 m)  — reverse zonal
    B2: rail dest     @ R = RAIL_RADIUS_M (800 m)  — reverse zonal

  flow_dual = flow_A + flow_B  (bi-directional walking flow)

Outputs in Patronage_Flow/output/:
  flow_long_dual.parquet       -- long table: edge_id × DAY_TYPE × HOUR →
                                   flow_A, flow_B, flow_dual
  flow_dual_<daytype>_<HH>.gpkg  -- per-slot edge geometries + flow_dual (QGIS)
  flow_dual_weekday_peak.gpkg  -- max flow_dual across all weekday hours (quick map)
  edges.gpkg                   -- network edges (if not already written by Main.py)
  pedestrian_network_filtered.gpkg  -- same (QA layer)

Set N_WORKERS = 100 in Constants.py before running on the server.
"""
from __future__ import annotations
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

from madina.una.tools import betweenness

from Patronage_Flow import Constants as C
from Patronage_Flow.Network import (
    load_pedestrian_highway, load_stations, load_buildings,
    build_zonal, build_zonal_reverse,
)
from Patronage_Flow.Flow_Computation import (
    load_ridership_split, load_occupancy_density,
    _set_origin_weights, _set_destination_weights,
    _activate_origin_subset, _restore_all_origins,
    _set_origin_weights_buildings, _set_destination_weights_stations,
    _activate_destination_subset, _restore_all_destinations,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run_betweenness(zonal, R: float, label: str) -> tuple[np.ndarray, float]:
    """Single betweenness call; returns (flow_array, elapsed_s)."""
    t0 = time.perf_counter()
    betweenness(
        zonal=zonal,
        search_radius=R,
        detour_ratio=C.DETOUR_RATIO,
        decay=True,
        decay_method="exponent",
        beta=C.BETA,
        num_cores=C.N_WORKERS,
        closest_destination=C.CLOSEST_DESTINATION,
        save_betweenness_as=None,
    )
    dt = time.perf_counter() - t0
    flow = zonal.network.edges["betweenness"].astype(np.float64).values.copy()
    nz = (flow > 0).sum()
    print(f"    {label:<22}  {dt:>5.1f}s   max={flow.max():>9.1f}   nz={nz:>7,}")
    return flow, dt


def _slot_stats(label: str, flow: np.ndarray) -> None:
    """Print one-line percentile summary for a flow array."""
    nz = (flow > 0).sum()
    if nz == 0:
        print(f"  {label:<16} all zero")
        return
    nzv = flow[flow > 0]
    top10_share = np.sort(flow)[::-1][:10].sum() / flow.sum() * 100
    print(f"  {label:<16}  max={flow.max():>9.1f}  "
          f"p99={np.percentile(nzv, 99):>7.1f}  "
          f"p50={np.percentile(nzv, 50):>5.1f}  "
          f"nz={nz:>7,}  top10={top10_share:.2f}%")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("[dual-pass] Patronage_Flow dual-pass full-island run")
    print(f"[dual-pass] N_WORKERS={C.N_WORKERS}  "
          f"beta={C.BETA}  detour={C.DETOUR_RATIO}  "
          f"bus_R={C.BUS_RADIUS_M:.0f}m  rail_R={C.RAIL_RADIUS_M:.0f}m")
    print("=" * 68)

    # -----------------------------------------------------------------------
    # 1. load data (full island, no bbox)
    # -----------------------------------------------------------------------
    hw        = load_pedestrian_highway(bbox=None)
    stations  = load_stations(bbox=None).reset_index(drop=True)
    buildings = load_buildings(bbox=None).reset_index(drop=True)

    # export filtered pedestrian network once for QA
    hw_export = hw.copy()
    hw_export["length_m"] = hw_export.geometry.length
    hw_path = C.OUTPUT_DIR / "pedestrian_network_filtered.gpkg"
    if not hw_path.exists():
        hw_export.to_file(hw_path, driver="GPKG")
        print(f"[dual-pass] wrote {hw_path.name}  "
              f"({len(hw_export):,} edges, "
              f"{hw_export['length_m'].sum()/1000:.1f} km)")

    # -----------------------------------------------------------------------
    # 2. build two zonals
    # -----------------------------------------------------------------------
    print("\n[dual-pass] building FORWARD zonal (stations -> buildings) ...")
    zA = build_zonal(hw, stations, buildings)
    n_edges = len(zA.network.edges)
    print(f"[dual-pass] forward zonal: {n_edges:,} edges")

    # export network snapshot
    edges_path = C.OUTPUT_DIR / "edges.gpkg"
    if not edges_path.exists():
        zA.network.edges.to_file(edges_path, driver="GPKG")
        print(f"[dual-pass] wrote {edges_path.name}")

    print("\n[dual-pass] building REVERSE zonal (buildings -> stations) ...")
    zB = build_zonal_reverse(hw, stations, buildings)
    print(f"[dual-pass] reverse zonal: {len(zB.network.edges):,} edges")

    # -----------------------------------------------------------------------
    # 3. ridership (split by direction) + occupancy
    # -----------------------------------------------------------------------
    print()
    ride_out  = load_ridership_split("tap_out")   # Pass A origins
    ride_in   = load_ridership_split("tap_in")    # Pass B destinations
    occupancy = load_occupancy_density()

    # -----------------------------------------------------------------------
    # 4. slot schedule
    # -----------------------------------------------------------------------
    schedule = (
        [("WEEKDAY", h)           for h in C.HOURS_WEEKDAY] +
        [("WEEKENDS/HOLIDAY", h)  for h in C.HOURS_WEEKEND]
    )
    n_slots = len(schedule)
    print(f"\n[dual-pass] {n_slots} slots × 4 sub-passes = {n_slots*4} betweenness calls")

    # -----------------------------------------------------------------------
    # 5. main loop
    # -----------------------------------------------------------------------
    out_long: list[pd.DataFrame] = []
    # accumulators for weekday peak map
    weekday_peak  = np.zeros(n_edges, dtype=np.float64)

    for i, (daytype, hour) in enumerate(schedule, 1):
        t_slot = time.perf_counter()
        tag = f"{daytype.split('/')[0].lower()}_{hour:02d}"
        print(f"\n[{i:>2}/{n_slots}]  {daytype}  hour={hour:>2}  ({tag})")

        # --- ridership + occupancy for this slot ---
        rs_out = (ride_out[(daytype, hour)]
                  if (daytype, hour) in ride_out.columns
                  else pd.Series(0.0, index=ride_out.index))
        rs_in  = (ride_in[(daytype, hour)]
                  if (daytype, hour) in ride_in.columns
                  else pd.Series(0.0, index=ride_in.index))
        den    = (occupancy[(daytype, hour)]
                  if (daytype, hour) in occupancy.columns
                  else pd.Series(0.0, index=occupancy.index))

        # =================================================================
        # PASS A  forward: stations (tap_out) → buildings (occupancy)
        # =================================================================
        _set_destination_weights(zA, buildings, den)
        flow_A = np.zeros(n_edges, dtype=np.float64)

        # A1 — bus origins @ 400 m
        _set_origin_weights(zA, stations, rs_out, only_source="BUS")
        _activate_origin_subset(zA, stations, keep_source="BUS")
        f, _ = _run_betweenness(zA, C.BUS_RADIUS_M, "A1 bus  R=400")
        flow_A += f

        # A2 — rail (MRT/LRT) origins @ 800 m
        _set_origin_weights(zA, stations, rs_out, only_source="MRT")
        _activate_origin_subset(zA, stations, keep_source="MRT")
        f, _ = _run_betweenness(zA, C.RAIL_RADIUS_M, "A2 rail R=800")
        flow_A += f

        _restore_all_origins(zA)

        # =================================================================
        # PASS B  reverse: buildings (occupancy) → stations (tap_in)
        # =================================================================
        _set_origin_weights_buildings(zB, buildings, den)
        flow_B = np.zeros(n_edges, dtype=np.float64)

        # B1 — bus destinations @ 400 m
        _set_destination_weights_stations(zB, stations, rs_in, only_source="BUS")
        _activate_destination_subset(zB, stations, keep_source="BUS")
        f, _ = _run_betweenness(zB, C.BUS_RADIUS_M, "B1 -> bus  R=400")
        flow_B += f

        # B2 — rail (MRT/LRT) destinations @ 800 m
        _set_destination_weights_stations(zB, stations, rs_in, only_source="MRT")
        _activate_destination_subset(zB, stations, keep_source="MRT")
        f, _ = _run_betweenness(zB, C.RAIL_RADIUS_M, "B2 -> rail R=800")
        flow_B += f

        _restore_all_destinations(zB)

        # =================================================================
        # aggregate + log
        # =================================================================
        flow_dual = flow_A + flow_B
        dt_slot = time.perf_counter() - t_slot

        _slot_stats("Pass A",    flow_A)
        _slot_stats("Pass B",    flow_B)
        _slot_stats("DUAL(A+B)", flow_dual)
        print(f"  slot wall-time: {dt_slot:.1f}s  "
              f"(elapsed total: {(time.perf_counter()-t_start)/60:.1f} min)")

        # --- accumulate long table ---
        slot_df = pd.DataFrame({
            "edge_id":   zA.network.edges.index.values,
            "DAY_TYPE":  daytype,
            "HOUR":      hour,
            "flow_A":    flow_A,
            "flow_B":    flow_B,
            "flow_dual": flow_dual,
        })
        out_long.append(slot_df)

        # --- per-slot GPKG ---
        gpkg_out = zA.network.edges.copy()
        gpkg_out["flow_A"]    = flow_A
        gpkg_out["flow_B"]    = flow_B
        gpkg_out["flow_dual"] = flow_dual
        gpkg_path = C.OUTPUT_DIR / f"flow_dual_{tag}.gpkg"
        gpkg_out.to_file(gpkg_path, driver="GPKG")
        print(f"  wrote {gpkg_path.name}")

        # --- weekday peak accumulator ---
        if "weekday" in tag:
            weekday_peak = np.maximum(weekday_peak, flow_dual)

    # -----------------------------------------------------------------------
    # 6. write outputs
    # -----------------------------------------------------------------------
    print("\n[dual-pass] writing flow_long_dual.parquet ...")
    flow_long = pd.concat(out_long, ignore_index=True)
    flow_long.to_parquet(C.OUTPUT_DIR / "flow_long_dual.parquet", index=False)
    print(f"[dual-pass] wrote flow_long_dual.parquet  ({len(flow_long):,} rows)")

    # weekday peak map (single GPKG — quickest map in QGIS)
    peak_out = zA.network.edges.copy()
    peak_out["flow_dual_weekday_peak"] = weekday_peak
    peak_path = C.OUTPUT_DIR / "flow_dual_weekday_peak.gpkg"
    peak_out.to_file(peak_path, driver="GPKG")
    print(f"[dual-pass] wrote {peak_path.name}")

    # -----------------------------------------------------------------------
    # 7. final summary
    # -----------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 68)
    print(f"[dual-pass] DONE — total wall-time: {elapsed/3600:.2f} h  "
          f"({elapsed/60:.1f} min)")
    print(f"  {n_slots} slots × 4 sub-passes  "
          f"avg {elapsed/n_slots/4:.1f}s per sub-pass")
    print(f"  output dir: {C.OUTPUT_DIR}")

    # network-wide final stats on weekday peak
    wdn = weekday_peak[weekday_peak > 0]
    if len(wdn):
        print(f"\n  weekday peak (max across 24h, all edges):")
        print(f"    max   = {weekday_peak.max():,.1f}")
        print(f"    p99   = {np.percentile(wdn, 99):,.1f}")
        print(f"    p95   = {np.percentile(wdn, 95):,.1f}")
        print(f"    nz    = {len(wdn):,} / {n_edges:,} edges")
    print("=" * 68)


if __name__ == "__main__":
    main()
