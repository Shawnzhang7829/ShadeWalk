"""
Patronage_Flow / run_2025_quarterly.py

Quarterly snapshots: island-wide weekday pedestrian flow distribution at 12:00 and 14:00 for
March, June, September and December 2025, plus the merged annual average.

Output directory layout (all under Patronage_Flow/output/):
  2025_03/
    flow_weekday_12.gpkg
    flow_weekday_14.gpkg
    flow_long.parquet
  2025_06/  (same as above)
  2025_09/  (same as above)
  2025_12/  (same as above)
  2025_annual/
    flow_weekday_12_annual_avg.gpkg   -- average of the 4 months
    flow_weekday_14_annual_avg.gpkg
    flow_long_annual_avg.parquet

Actual number of weekdays per month (used to convert monthly totals to daily averages):
  2025-03: 21 weekdays
  2025-06: 21 weekdays
  2025-09: 22 weekdays
  2025-12: 23 weekdays

Run:
    python -m Patronage_Flow.run_2025_quarterly
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from Patronage_Flow import Constants as C
from Patronage_Flow.Network import (
    load_pedestrian_highway, load_stations, load_buildings, build_zonal,
)
from Patronage_Flow.Flow_Computation import (
    load_ridership_table, compute_hourly_flow,
)

# ---------------------------------------------------------------------------
# Quarterly configuration: (month tag, bus csv, train csv, weekday count, weekend count)
# ---------------------------------------------------------------------------
NODE_ROOT = C.ROOT / "Station flow" / "2025" / "node"

QUARTERS = [
    ("2025_03", NODE_ROOT / "transport_node_bus_202503.csv",
               NODE_ROOT / "transport_node_train_202503.csv",
               21, 9),
    ("2025_06", NODE_ROOT / "transport_node_bus_202506.csv",
               NODE_ROOT / "transport_node_train_202506.csv",
               21, 9),
    ("2025_09", NODE_ROOT / "transport_node_bus_202509.csv",
               NODE_ROOT / "transport_node_train_202509.csv",
               22, 8),
    ("2025_12", NODE_ROOT / "transport_node_bus_202512.csv",
               NODE_ROOT / "transport_node_train_202512.csv",
               23, 8),
]

HOURS = [12, 14]   # only these two hours are run


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 60)
    print("[2025-quarterly] loading network (shared across all months)...")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load the network once (shared by all months)
    # -----------------------------------------------------------------------
    hw        = load_pedestrian_highway(bbox=None)
    stations  = load_stations(bbox=None).reset_index(drop=True)
    buildings = load_buildings(bbox=None).reset_index(drop=True)
    zonal     = build_zonal(hw, stations, buildings)

    n_edges   = len(zonal.network.edges)
    print(f"[2025-quarterly] network ready: {n_edges:,} edges")

    # export the network (if not present)
    edges_path = C.OUTPUT_DIR / "edges.gpkg"
    if not edges_path.exists():
        C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        zonal.network.edges.to_file(edges_path, driver="GPKG")
        print(f"[2025-quarterly] wrote {edges_path.name}")

    # -----------------------------------------------------------------------
    # 2. Quarterly computation
    # -----------------------------------------------------------------------
    # for the final merge: { hour -> list of flow arrays }
    annual_flows: dict[int, list[np.ndarray]] = {h: [] for h in HOURS}

    for tag, bus_csv, train_csv, wd_days, we_days in QUARTERS:
        print(f"\n{'='*60}")
        print(f"[2025-quarterly] === {tag}  weekdays={wd_days} ===")
        print(f"{'='*60}")

        out_dir = C.OUTPUT_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        ridership = load_ridership_table(
            bus_csv=bus_csv,
            train_csv=train_csv,
            weekdays_per_month=wd_days,
            weekend_holidays_per_month=we_days,
        )

        flow_long = compute_hourly_flow(
            zonal, stations, buildings, ridership,
            hours_weekday=HOURS,
            hours_weekend=[],
            output_dir=out_dir,
        )

        # save parquet
        pq_path = out_dir / "flow_long.parquet"
        flow_long.to_parquet(pq_path, index=False)
        print(f"[2025-quarterly] wrote {pq_path.relative_to(C.OUTPUT_DIR)}")

        # collect the hourly flows for the annual average
        for h in HOURS:
            sub = flow_long[flow_long["HOUR"] == h]
            if len(sub):
                arr = sub.set_index("edge_id")["flow"].reindex(
                    zonal.network.edges.index).fillna(0.0).values
                annual_flows[h].append(arr)

    # -----------------------------------------------------------------------
    # 3. Merge the annual average
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[2025-quarterly] computing annual average ...")
    annual_dir = C.OUTPUT_DIR / "2025_annual"
    annual_dir.mkdir(parents=True, exist_ok=True)

    annual_long_parts = []
    for h in HOURS:
        arrs = annual_flows[h]
        if not arrs:
            continue
        avg_flow = np.mean(np.stack(arrs, axis=0), axis=0)

        # GPKG
        gpkg_out = zonal.network.edges.copy()
        gpkg_out["flow"] = avg_flow
        gpkg_path = annual_dir / f"flow_weekday_{h:02d}_annual_avg.gpkg"
        gpkg_out.to_file(gpkg_path, driver="GPKG")
        nz = (avg_flow > 0).sum()
        print(f"[2025-quarterly] wrote {gpkg_path.name}  "
              f"max={avg_flow.max():.1f}  p99={np.percentile(avg_flow[avg_flow>0],99):.1f}"
              f"  nz={nz:,}")

        annual_long_parts.append(pd.DataFrame({
            "edge_id": zonal.network.edges.index.values,
            "DAY_TYPE": "WEEKDAY",
            "HOUR": h,
            "flow_annual_avg": avg_flow,
        }))

    if annual_long_parts:
        annual_pq = pd.concat(annual_long_parts, ignore_index=True)
        annual_pq.to_parquet(annual_dir / "flow_long_annual_avg.parquet", index=False)
        print(f"[2025-quarterly] wrote flow_long_annual_avg.parquet")

    # -----------------------------------------------------------------------
    # 4. Completion report
    # -----------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"[2025-quarterly] DONE  total: {elapsed/3600:.2f}h ({elapsed/60:.1f}min)")
    print(f"  output dir : {C.OUTPUT_DIR}")
    print(f"  subdirs    : 2025_03/ 2025_06/ 2025_09/ 2025_12/ 2025_annual/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
