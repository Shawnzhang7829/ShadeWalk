"""
Patronage_Flow / run_2025_annual_total.py
==========================================
2025 annual TOTAL pedestrian flow computation.

Principle
---------
madina patronage betweenness is linear in the origin weight, therefore:
  flow_annual_total = betweenness(sum_12months(ridership_m))
                    = sum_12months(betweenness(ridership_m))

Sum the raw monthly station ridership of the 12 months first, then run betweenness once;
the result is exactly equivalent to running 12 times and summing, but needs only 1/12 of the compute time.

Input
-----
  Station flow/2025/node/transport_node_bus_2025{01..12}.csv
  Station flow/2025/node/transport_node_train_2025{01..12}.csv

Output  (Patronage_Flow/output/2025_total/)
------
  annual_ridership_table.parquet     -- station ridership table summed over the 12 months
  flow_weekday_{HH}.gpkg             -- annual total flow GPKG for each hour
  flow_long_annual_total.parquet     -- long-format summary

Run:
    python -m Patronage_Flow.run_2025_annual_total
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
    compute_hourly_flow,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NODE_ROOT = C.ROOT / "Station flow" / "2025" / "node"
MONTHS    = list(range(1, 13))        # all months 1-12
HOURS     = [12, 14]                  # hours to compute (weekday only)
OUT_DIR   = C.OUTPUT_DIR / "2025_total"


# ---------------------------------------------------------------------------
# Core: merge the 12 monthly node CSVs into an annual total table
# ---------------------------------------------------------------------------
def build_annual_ridership_table() -> pd.DataFrame:
    """
    Read the bus + train node CSVs of all 12 months of 2025,
    sum the raw monthly totals directly for each (PT_CODE, DAY_TYPE, HOUR),
    with no "divide by number of weekdays" normalisation (keeps the annual-total scale).

    Slash interchange codes (e.g. NS24/NE6/CC1) are split equally by number of lines and exploded,
    conserving the total (consistent with load_ridership_table).

    Returns a wide table in the same format as load_ridership_table():
      Index : PT_CODE
      Cols  : MultiIndex(DAY_TYPE, HOUR)
      Values: annual total tap_in + tap_out ("both" mode)
    """
    t0 = time.perf_counter()
    print(f"[annual] loading {len(MONTHS)} months of node data ...")
    frames = []
    for m in MONTHS:
        tag      = f"2025{m:02d}"
        bus_csv  = NODE_ROOT / f"transport_node_bus_{tag}.csv"
        trn_csv  = NODE_ROOT / f"transport_node_train_{tag}.csv"

        bus  = pd.read_csv(bus_csv)
        rail = pd.read_csv(trn_csv)
        df   = pd.concat([bus, rail], ignore_index=True)
        df   = df.dropna(subset=["TIME_PER_HOUR", "PT_CODE", "DAY_TYPE"])
        df["PT_CODE"] = df["PT_CODE"].astype(str)
        df["HOUR"]    = df["TIME_PER_HOUR"].astype(int)

        # slash interchange code fix: split equally by number of lines, then explode
        has_slash = df["PT_CODE"].str.contains("/", na=False)
        if has_slash.any():
            split = df[has_slash].copy()
            n_parts = split["PT_CODE"].str.count("/") + 1
            split["TOTAL_TAP_IN_VOLUME"]  = split["TOTAL_TAP_IN_VOLUME"]  / n_parts
            split["TOTAL_TAP_OUT_VOLUME"] = split["TOTAL_TAP_OUT_VOLUME"] / n_parts
            split["PT_CODE"] = split["PT_CODE"].str.split("/")
            split = split.explode("PT_CODE")
            df = pd.concat([df[~has_slash], split], ignore_index=True)

        frames.append(df[["PT_CODE", "DAY_TYPE", "HOUR",
                           "TOTAL_TAP_IN_VOLUME", "TOTAL_TAP_OUT_VOLUME"]])
        print(f"  month {m:02d}: {len(df):,} rows loaded")

    print(f"[annual] concatenating {len(frames)} months ...")
    combined = pd.concat(frames, ignore_index=True)

    # sum by (PT_CODE, DAY_TYPE, HOUR) -> annual total
    annual = (combined
              .groupby(["PT_CODE", "DAY_TYPE", "HOUR"], as_index=False)
              [["TOTAL_TAP_IN_VOLUME", "TOTAL_TAP_OUT_VOLUME"]]
              .sum())

    # W = both (tap_in + tap_out)
    annual["W"] = annual["TOTAL_TAP_IN_VOLUME"] + annual["TOTAL_TAP_OUT_VOLUME"]

    # convert to a wide table (same output format as load_ridership_table)
    wide = (annual.groupby(["PT_CODE", "DAY_TYPE", "HOUR"])["W"].sum()
                  .unstack(["DAY_TYPE", "HOUR"])
                  .fillna(0.0))

    elapsed = time.perf_counter() - t0
    print(f"[annual] ridership table: {wide.shape[0]:,} stations x "
          f"{wide.shape[1]} (daytype, hour) bins  [{elapsed:.1f}s]")
    return wide, annual   # also return the long format for archiving


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    print("=" * 60)
    print("[annual-total] 2025 annual total pedestrian flow")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build the network (identical to Main.py)
    # ------------------------------------------------------------------
    print("\n[annual-total] loading network ...")
    hw        = load_pedestrian_highway(bbox=None)
    stations  = load_stations(bbox=None).reset_index(drop=True)
    buildings = load_buildings(bbox=None).reset_index(drop=True)
    zonal     = build_zonal(hw, stations, buildings)
    n_edges   = len(zonal.network.edges)
    print(f"[annual-total] network ready: {n_edges:,} edges")

    # ------------------------------------------------------------------
    # 2. Build the station ridership table summed over the 12 months
    # ------------------------------------------------------------------
    print("\n[annual-total] building annual ridership table ...")
    ridership_wide, ridership_long = build_annual_ridership_table()

    # save the table (for later inspection)
    pq_path = OUT_DIR / "annual_ridership_table.parquet"
    ridership_long.to_parquet(pq_path, index=False)
    print(f"[annual-total] wrote {pq_path.name}  "
          f"({len(ridership_long):,} rows)")

    # quick check: print the annual totals of a few representative stations
    for stn in ["NS24", "EW14", "84009"]:
        if stn in ridership_wide.index:
            wd_col = [c for c in ridership_wide.columns if c[0] == "WEEKDAY"]
            tot = ridership_wide.loc[stn, wd_col].sum()
            print(f"  {stn}  weekday annual total W = {tot:,.0f}")

    # ------------------------------------------------------------------
    # 3. Run betweenness (WEEKDAY only, each hour)
    # ------------------------------------------------------------------
    print(f"\n[annual-total] running betweenness for hours {HOURS} ...")
    flow_long = compute_hourly_flow(
        zonal, stations, buildings, ridership_wide,
        hours_weekday=HOURS,
        hours_weekend=[],
        output_dir=OUT_DIR,
    )

    # save the long format
    pq_out = OUT_DIR / "flow_long_annual_total.parquet"
    flow_long.to_parquet(pq_out, index=False)
    print(f"[annual-total] wrote {pq_out.name}")

    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"[annual-total] DONE  total: {elapsed/3600:.2f}h ({elapsed/60:.1f}min)")
    print(f"  output dir: {OUT_DIR}")   # output/2025_total/
    for h in HOURS:
        gpkg = OUT_DIR / f"flow_weekday_{h:02d}.gpkg"
        if gpkg.exists():
            print(f"    {gpkg.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
