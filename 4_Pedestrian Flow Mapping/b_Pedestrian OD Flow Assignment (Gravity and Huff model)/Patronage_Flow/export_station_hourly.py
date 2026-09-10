"""
Export per-station hourly ridership snapshots for QA / verification.

Outputs tap_in, tap_out, and total (tap_in + tap_out) separately per slot.

Output: Patronage_Flow/output/station_hourly_ridership.gpkg
  geometry          : station / MRT-exit point (SVY21)
  PT_CODE           : station code
  source            : BUS / MRT
  n_exits           : weight_divisor (1 for bus; N exits for MRT)
  in_<daytype>_<HH> : raw tap_in at that station-row for that slot
  out_<daytype>_<HH>: raw tap_out
  tot_<daytype>_<HH>: tap_in + tap_out
  inj_<daytype>_<HH>: injected weight per exit = tot / n_exits
                      (what madina "both" mode actually uses)

  in_weekday_total  : sum of tap_in across all weekday hours
  out_weekday_total : sum of tap_out across all weekday hours
  tot_weekday_total : sum of total across all weekday hours

Also writes station_hourly_summary.gpkg (one row per PT_CODE).

Run:
    python -m Patronage_Flow.export_station_hourly
"""
from __future__ import annotations
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

from Patronage_Flow import Constants as C
from Patronage_Flow.Network import load_stations
from Patronage_Flow.Flow_Computation import load_ridership_split


def main():
    bbox = tuple(C.SMOKE_BBOX) if C.SMOKE_BBOX is not None else None
    if bbox:
        print(f"[export] SMOKE BBOX active = {bbox}")

    print("[export] loading stations ...")
    stations = load_stations(bbox=bbox).reset_index(drop=True)
    print(f"[export] {len(stations):,} origin rows "
          f"(BUS={(stations['source']=='BUS').sum()}, "
          f"MRT={(stations['source']=='MRT').sum()})")

    print("[export] loading ridership (tap_in + tap_out separately) ...")
    ride_in  = load_ridership_split("tap_in")
    ride_out = load_ridership_split("tap_out")

    # all slots in the ridership table (24 weekday + 24 weekend = 48)
    all_slots = sorted(set(ride_in.columns) | set(ride_out.columns))

    out = stations[["PT_CODE", "source", "weight_divisor", "geometry"]].copy()
    out = out.rename(columns={"weight_divisor": "n_exits"})

    in_cols, out_cols, tot_cols, inj_cols = [], [], [], []

    for (daytype, hour) in all_slots:
        slot_tag = f"{daytype.split('/')[0].lower()}_{hour:02d}"

        # raw per-station values (same value shared by all exits of same PT_CODE)
        raw_in  = (ride_in.get((daytype, hour),
                               pd.Series(0.0, index=ride_in.index))
                          .reindex(out["PT_CODE"]).fillna(0.0)
                          .astype(np.float32).values)
        raw_out = (ride_out.get((daytype, hour),
                                pd.Series(0.0, index=ride_out.index))
                           .reindex(out["PT_CODE"]).fillna(0.0)
                           .astype(np.float32).values)
        raw_tot = raw_in + raw_out

        out[f"in_{slot_tag}"]  = raw_in
        out[f"out_{slot_tag}"] = raw_out
        out[f"tot_{slot_tag}"] = raw_tot
        # injected weight per exit = what madina "both" mode uses
        out[f"inj_{slot_tag}"] = (raw_tot /
                                   out["n_exits"].astype(np.float32).values
                                   ).astype(np.float32)

        in_cols.append(f"in_{slot_tag}")
        out_cols.append(f"out_{slot_tag}")
        tot_cols.append(f"tot_{slot_tag}")
        inj_cols.append(f"inj_{slot_tag}")

    # all-day totals
    wd_in  = [c for c in in_cols  if "weekday" in c]
    wd_out = [c for c in out_cols if "weekday" in c]
    wd_tot = [c for c in tot_cols if "weekday" in c]
    we_tot = [c for c in tot_cols if "weekends" in c]

    out["in_weekday_total"]   = out[wd_in].sum(axis=1).astype(np.float32)
    out["out_weekday_total"]  = out[wd_out].sum(axis=1).astype(np.float32)
    out["tot_weekday_total"]  = out[wd_tot].sum(axis=1).astype(np.float32)
    out["tot_weekends_total"] = out[we_tot].sum(axis=1).astype(np.float32) if we_tot else 0.0

    out = gpd.GeoDataFrame(out, geometry="geometry", crs=C.TARGET_CRS)

    # --- per-exit export (one row per PT_CODE × exit) -----------------------
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.OUTPUT_DIR / "station_hourly_ridership.gpkg"
    out.to_file(out_path, driver="GPKG")
    print(f"[export] wrote {out_path.name}  "
          f"({len(out):,} rows, {len(out.columns)-1} cols + geom)")

    # --- per-station summary (one row per PT_CODE) --------------------------
    agg_dict = {
        "source":   "first",
        "n_exits":  "first",
        "geometry": "first",
        "in_weekday_total":   "first",
        "out_weekday_total":  "first",
        "tot_weekday_total":  "first",
        "tot_weekends_total": "first",
    }
    # raw cols: same value across exits -> take first
    for c in in_cols + out_cols + tot_cols:
        agg_dict[c] = "first"
    # injected cols: sum across exits = raw total (by construction)
    for c in inj_cols:
        agg_dict[c] = "sum"

    grp = out.groupby("PT_CODE", as_index=False).agg(agg_dict)
    grp = gpd.GeoDataFrame(grp, geometry="geometry", crs=C.TARGET_CRS)
    grp_path = C.OUTPUT_DIR / "station_hourly_summary.gpkg"
    grp.to_file(grp_path, driver="GPKG")
    print(f"[export] wrote {grp_path.name}  ({len(grp):,} unique PT_CODEs)")

    # --- console summary ----------------------------------------------------
    print()
    print("--- top 10 stations by weekday all-day total (tap_in + tap_out) ---")
    top = grp.sort_values("tot_weekday_total", ascending=False).head(10)
    print(top[["PT_CODE", "source", "n_exits",
               "in_weekday_total", "out_weekday_total",
               "tot_weekday_total"]].to_string(index=False))

    print()
    print("--- slot totals (weekday) ---")
    for c_in, c_out in zip(sorted(wd_in), sorted(wd_out)):
        hour = c_in.split("_")[-1]
        print(f"  weekday {hour}h  "
              f"tap_in={out[c_in].sum():>10,.0f}  "
              f"tap_out={out[c_out].sum():>10,.0f}  "
              f"total={out[c_in].sum()+out[c_out].sum():>10,.0f}")


if __name__ == "__main__":
    main()
