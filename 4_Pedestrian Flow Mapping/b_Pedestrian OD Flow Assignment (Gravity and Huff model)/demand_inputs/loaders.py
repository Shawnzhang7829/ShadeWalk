"""
demand_inputs / loaders.py

Readers of the demand inputs:
  - load_pedestrian_highway : the pedestrian-passable subset of the OSM highway layer (Highway_OSM.gpkg), clipped to the island
  - load_stations           : bus stops (BusStop.shp) and MRT / LRT exits (Train_Station_Exit_Layer.shp) as origin candidates
  - load_ridership_split    : the LTA DataMall passenger volumes by node (tap-in or tap-out) as a per-day table indexed by PT_CODE

The functions are the loaders of the earlier `Patronage_Flow` package (Network.py, Flow_Computation.py), unchanged.
"""
from __future__ import annotations
import warnings

import geopandas as gpd
import pandas as pd

from demand_inputs import Constants as C


# ---------------------------------------------------------------------------
# Singapore island boundary (mainland + offshore islands)
# ---------------------------------------------------------------------------
_BORDER_CACHE: gpd.GeoDataFrame | None = None

def load_island_border() -> gpd.GeoDataFrame:
    """Return the island boundary as a single-row GeoDataFrame in TARGET_CRS.
    Cached so we don't re-read the shapefile for each loader."""
    global _BORDER_CACHE
    if _BORDER_CACHE is None:
        g = gpd.read_file(C.ISLAND_BORDER_SHP).to_crs(C.TARGET_CRS)
        _BORDER_CACHE = gpd.GeoDataFrame(
            geometry=[g.geometry.union_all()], crs=C.TARGET_CRS,
        )
        print(f"[network] island border loaded: "
              f"{len(g)} polygons, area = "
              f"{_BORDER_CACHE.area.iloc[0]/1e6:.1f} km^2")
    return _BORDER_CACHE


def _filter_to_border(gdf: gpd.GeoDataFrame, predicate: str = "intersects") -> gpd.GeoDataFrame:
    """Spatially filter `gdf` to features that satisfy `predicate` against
    the island boundary. Features fully outside Singapore are dropped."""
    border = load_island_border()
    out = gpd.sjoin(gdf, border, predicate=predicate, how="inner")
    return out.drop(columns="index_right")


# ---------------------------------------------------------------------------
# pedestrian highway loader
# ---------------------------------------------------------------------------
def load_pedestrian_highway(bbox=None) -> gpd.GeoDataFrame:
    """
    Read the OSM highway gpkg, keep only pedestrian-passable features, and
    return a clean LineString GeoDataFrame.

    Filters applied (in order):
      1. `highway` tag must be in PEDESTRIAN_HIGHWAYS
      2. drop foot=no/private and access=no/private
      3. drop service=parking_aisle/driveway/drive-through/emergency_access
         -- these are vehicle-only despite having `highway=service`
      4. drop expressway=yes -- limited-access roads mis-tagged below motorway
    """
    print(f"[network] reading {C.HIGHWAY_GPKG.name} (layer={C.HIGHWAY_LAYER}) ...")
    cols = ["highway", "foot", "access", "service", "expressway", "geometry"]
    gdf = gpd.read_file(C.HIGHWAY_GPKG, layer=C.HIGHWAY_LAYER,
                        bbox=bbox, columns=cols)
    print(f"[network]   raw features in scope     : {len(gdf):,}")

    gdf = gdf[gdf["highway"].isin(C.PEDESTRIAN_HIGHWAYS)].copy()
    print(f"[network]   pedestrian-tag matches    : {len(gdf):,}")

    if "foot" in gdf.columns:
        gdf = gdf[~gdf["foot"].isin(C.EXCLUDE_FOOT_VALUES)]
    if "access" in gdf.columns:
        gdf = gdf[~gdf["access"].isin(C.EXCLUDE_ACCESS_VALUES)]
    print(f"[network]   after foot/access filter : {len(gdf):,}")

    if "service" in gdf.columns:
        gdf = gdf[~gdf["service"].isin(C.EXCLUDE_SERVICE_VALUES)]
        print(f"[network]   after service filter      : {len(gdf):,}")
    if "expressway" in gdf.columns:
        gdf = gdf[~gdf["expressway"].isin(C.EXCLUDE_EXPRESSWAY_VALUES)]
        print(f"[network]   after expressway filter  : {len(gdf):,}")

    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.length > 1e-6]
    gdf = gdf[["geometry"]].reset_index(drop=True)
    gdf = gdf.set_crs(C.TARGET_CRS, allow_override=True)

    gdf = _filter_to_border(gdf, predicate="intersects")
    gdf = gdf[["geometry"]].reset_index(drop=True)
    print(f"[network]   inside Island_boarder    : {len(gdf):,}")
    return gdf


# ---------------------------------------------------------------------------
# stations
# ---------------------------------------------------------------------------
def load_bus_stations(bbox=None) -> gpd.GeoDataFrame:
    """Bus stops with PT_CODE matching the LTA CSV. One stop = one origin."""
    gdf = gpd.read_file(C.BUS_SHP, bbox=bbox).to_crs(C.TARGET_CRS)
    gdf = gdf[["BUS_STOP_N", "geometry"]].rename(columns={"BUS_STOP_N": "PT_CODE"})
    gdf["PT_CODE"] = gdf["PT_CODE"].astype(str)
    gdf["source"] = "BUS"
    # fix 2026-09-16: a few bus-stop codes occur twice in the shapefile; the stop's volume is shared over its rows
    gdf["weight_divisor"] = gdf.groupby("PT_CODE")["PT_CODE"].transform("size").astype(float)
    n0 = len(gdf)
    gdf = _filter_to_border(gdf, predicate="within").reset_index(drop=True)
    print(f"[network] bus stops: {n0:,} -> {len(gdf):,} after border filter")
    return gdf


def load_mrt_stations(bbox=None) -> gpd.GeoDataFrame:
    """
    MRT/LRT *exits* as origin points (v3).

    Each station's hourly ridership is split equally across its N exits:
        per-exit weight at hour h = ridership[PT_CODE](h) / N_exits

    For interchange stations (e.g. Dhoby Ghaut = NS24 / NE6 / CC1) the same
    set of exits is replicated once per PT_CODE so each line's ridership
    is independently distributed across the shared exits.

    Join chain:
        exit.stn_name (UPPER)  ->  lookup.STN_NAM_DE  ->  PT_CODE
    """
    if not C.MRT_LOOKUP_CSV.exists():
        warnings.warn(f"MRT lookup not at {C.MRT_LOOKUP_CSV}; MRT excluded.")
        return gpd.GeoDataFrame(
            columns=["PT_CODE", "weight_divisor", "geometry", "source"],
            crs=C.TARGET_CRS,
        )

    exits = gpd.read_file(C.MRT_EXIT_SHP, bbox=bbox).to_crs(C.TARGET_CRS)
    exits["stn_name"] = exits["stn_name"].astype(str).str.upper().str.strip()
    n_exits = exits.groupby("stn_name").size().rename("weight_divisor")
    print(f"[network] MRT/LRT exits read: {len(exits):,} across "
          f"{exits['stn_name'].nunique()} stations")

    lk = pd.read_csv(C.MRT_LOOKUP_CSV, dtype=str)
    lk["STN_NAM_DE"] = lk["STN_NAM_DE"].str.upper().str.strip()

    # one row per (PT_CODE, exit) — interchanges fan out automatically
    merged = exits.merge(
        lk.rename(columns={"STN_NAM_DE": "stn_name"}),
        on="stn_name", how="inner",
    )
    merged = merged.merge(n_exits, on="stn_name", how="left")
    dropped_names = set(exits["stn_name"]) - set(merged["stn_name"])
    if dropped_names:
        print(f"[network]   dropped {len(dropped_names)} exit-stations with no "
              f"PT_CODE in lookup: {sorted(dropped_names)}")

    gdf = gpd.GeoDataFrame(
        merged[["PT_CODE", "weight_divisor", "geometry"]],
        crs=C.TARGET_CRS,
    )
    gdf["source"] = "MRT"
    n0 = len(gdf)
    gdf = _filter_to_border(gdf, predicate="within").reset_index(drop=True)
    print(f"[network]   MRT origin rows (exit x PT_CODE): "
          f"{n0:,} -> {len(gdf):,} after border filter")
    return gdf


def load_stations(bbox=None) -> gpd.GeoDataFrame:
    """Bus stops + MRT exits as a single GeoDataFrame of origin candidates."""
    bus = load_bus_stations(bbox)
    mrt = load_mrt_stations(bbox)
    out = pd.concat([bus, mrt], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=C.TARGET_CRS)
    out["W_init"] = 1.0   # placeholder weight; rewritten per-hour later
    print(f"[network] stations loaded: {len(out):,}  "
          f"(BUS={(out['source']=='BUS').sum()}, "
          f"MRT={(out['source']=='MRT').sum()})")
    return out


# ---------------------------------------------------------------------------
# ridership (monthly sums -> per-day means, indexed by PT_CODE)
# ---------------------------------------------------------------------------
def load_ridership_split(mode: str) -> pd.DataFrame:
    """Per-day mean tap-in or tap-out volume of every station: DataFrame indexed by PT_CODE with
    columns = MultiIndex(DAY_TYPE, HOUR).  `mode` is one of: 'tap_in', 'tap_out'."""
    assert mode in ("tap_in", "tap_out"), f"unknown mode {mode!r}"
    bus  = pd.read_csv(C.BUS_CSV, dtype={'PT_CODE': str})   # fix 2026-09-17: bus-stop codes keep their leading zero ('05013'); parsed as int they never joined the stop layer and the 225 central-area stops exported as 0
    rail = pd.read_csv(C.TRAIN_CSV, dtype={'PT_CODE': str})
    df = pd.concat([bus, rail], ignore_index=True)
    df = df.dropna(subset=["TIME_PER_HOUR", "PT_CODE", "DAY_TYPE"])
    df["PT_CODE"] = df["PT_CODE"].astype(str)
    df["HOUR"]    = df["TIME_PER_HOUR"].astype(int)

    # interchange codes "NS24/NE6/CC1" -> one row per line code,
    # ridership divided equally so total is preserved.
    has_slash = df["PT_CODE"].str.contains("/", na=False)
    if has_slash.any():
        split = df[has_slash].copy()
        n_parts = split["PT_CODE"].str.count("/") + 1
        split["TOTAL_TAP_IN_VOLUME"]  = split["TOTAL_TAP_IN_VOLUME"]  / n_parts
        split["TOTAL_TAP_OUT_VOLUME"] = split["TOTAL_TAP_OUT_VOLUME"] / n_parts
        split["PT_CODE"] = split["PT_CODE"].str.split("/")
        split = split.explode("PT_CODE")
        df = pd.concat([df[~has_slash], split], ignore_index=True)

    if mode == "tap_in":
        df["W"] = df["TOTAL_TAP_IN_VOLUME"]
    else:
        df["W"] = df["TOTAL_TAP_OUT_VOLUME"]

    days = {"WEEKDAY": C.WEEKDAYS_PER_MONTH,
            "WEEKENDS/HOLIDAY": C.WEEKEND_HOLIDAYS_PER_MONTH}
    df["W"] = df["W"] / df["DAY_TYPE"].map(days)

    wide = (df.groupby(["PT_CODE", "DAY_TYPE", "HOUR"])["W"].sum()
              .unstack(["DAY_TYPE", "HOUR"])
              .fillna(0.0))
    print(f"[flow] ridership table ({mode}): {wide.shape[0]:,} stations x "
          f"{wide.shape[1]} bins")
    return wide
