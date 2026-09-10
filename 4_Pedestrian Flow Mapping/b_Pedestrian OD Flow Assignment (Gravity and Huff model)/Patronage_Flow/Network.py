"""
Patronage_Flow / Network.py    (v2 — madina-based)

Step 1 of the pipeline.

Input layers come from three OSM-derived / official sources:
  - Highway_OSM.gpkg            -- the pedestrian-passable subset of OSM highways
  - BusStop.shp + RapidTransitSystemStation.shp + LTA ridership CSVs
                                -- transit stations as ORIGINS (weighted by ridership)
  - SG_Building_SVY21_TH.shp    -- building footprints as DESTINATIONS (weighted
                                   by floor area as a proxy for attractiveness)

The function `build_zonal()` assembles these into a `madina.Zonal` workspace
with origins and destinations inserted onto the network -- ready for
`madina.una.tools.betweenness()` to run.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

from madina.zonal import Zonal

from Patronage_Flow import Constants as C


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
    gdf["weight_divisor"] = 1.0   # bus stops are single-point origins
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
# buildings (destinations)
# ---------------------------------------------------------------------------
def load_buildings(bbox=None) -> gpd.GeoDataFrame:
    """
    Building polygons -> centroid points with archetype + GFA columns.

    v5 change: do NOT pre-compute a static destination weight W. Instead,
    retain `gross_floor_area` and `building_archetype` so that the per-slot
    weight = GFA * occupancy_density(archetype, day_type, hour) can be
    written into Zonal node weights at run time.

    bbox is interpreted in TARGET_CRS (EPSG:3414) for consistency with the
    rest of the pipeline. The geojson is in EPSG:4326, so we convert the bbox
    to lon/lat before reading and re-project the result.
    """
    print(f"[network] reading {C.BUILDING_GEOJSON.name} ...")
    cols = [C.BUILDING_GFA_COL, C.BUILDING_ARCHETYPE_COL, "geometry"]
    if bbox is not None:
        from shapely.geometry import box
        bb = tuple((gpd.GeoSeries([box(*bbox)], crs=C.TARGET_CRS)
                       .to_crs("EPSG:4326").total_bounds).tolist())
    else:
        bb = None
    gdf = gpd.read_file(C.BUILDING_GEOJSON, bbox=bb, columns=cols).to_crs(C.TARGET_CRS)

    gdf = gdf.dropna(subset=[C.BUILDING_GFA_COL])
    gdf = gdf[gdf[C.BUILDING_GFA_COL] > 0]
    gdf[C.BUILDING_ARCHETYPE_COL] = (gdf[C.BUILDING_ARCHETYPE_COL]
                                       .fillna("__missing__").astype(str))

    gdf["geometry"] = gdf.geometry.centroid
    # placeholder W -- madina's insert_node needs *some* weight column on
    # the layer. Real per-slot weights are written in _set_destination_weights.
    gdf["W"] = 1.0
    keep = ["W", C.BUILDING_ARCHETYPE_COL, C.BUILDING_GFA_COL, "geometry"]
    gdf = gdf[keep].reset_index(drop=True)
    gdf = gdf.set_crs(C.TARGET_CRS, allow_override=True)
    print(f"[network] buildings (GFA > 0): {len(gdf):,}  "
          f"total GFA = {gdf[C.BUILDING_GFA_COL].sum()/1e6:.1f}M m^2")

    # archetype-share summary (by raw GFA only -- no static multiplier in v5)
    by_arch = (gdf.groupby(C.BUILDING_ARCHETYPE_COL)
                  .agg(n=("W", "size"),
                       gfa_sum=(C.BUILDING_GFA_COL, "sum"))
                  .sort_values("gfa_sum", ascending=False))
    by_arch["gfa_pct"] = (by_arch["gfa_sum"] / by_arch["gfa_sum"].sum() * 100).round(1)
    print(f"[network]   archetype GFA distribution (top 8):")
    print(by_arch.head(8).to_string())
    return gdf


# ---------------------------------------------------------------------------
# build the madina Zonal workspace
# ---------------------------------------------------------------------------
def build_zonal(highway_gdf: gpd.GeoDataFrame,
                stations_gdf: gpd.GeoDataFrame,
                buildings_gdf: gpd.GeoDataFrame) -> Zonal:
    """
    Assemble a madina Zonal:
      1. street network from highway_gdf
      2. origins = stations (with placeholder weight 'W_init', overwritten later)
      3. destinations = buildings (weight = floor area)
      4. internal NetworkX graph
    """
    z = Zonal()
    z.load_layer("streets", highway_gdf)
    z.create_street_network(
        source_layer="streets",
        node_snapping_tolerance=C.NODE_SNAPPING_TOLERANCE_M,
        redundant_edge_treatment=C.REDUNDANT_EDGE_TREATMENT,
        turn_threshold_degree=C.TURN_THRESHOLD_DEG,
        turn_penalty_amount=C.TURN_PENALTY_M,
    )
    print(f"[network] madina graph: "
          f"{len(z.network.nodes):,} nodes  "
          f"{len(z.network.edges):,} edges")

    z.load_layer("stations", stations_gdf)
    z.insert_node("stations", label="origin", weight_attribute="W_init")

    z.load_layer("buildings", buildings_gdf)
    z.insert_node("buildings", label="destination", weight_attribute="W")

    z.create_graph(light_graph=True, d_graph=True)

    counts = z.network.nodes["type"].value_counts().to_dict()
    print(f"[network] node types: {counts}")
    return z


def build_zonal_reverse(highway_gdf: gpd.GeoDataFrame,
                        stations_gdf: gpd.GeoDataFrame,
                        buildings_gdf: gpd.GeoDataFrame) -> Zonal:
    """
    Reverse-direction Zonal for true dual-pass (Pass B):
      origins = buildings (weight = GFA * occupancy at slot)
      destinations = stations (weight = tap_in at slot, divided by exits)

    Models pre-boarding walks: people walking FROM buildings TO transit stations
    to begin a ride. Combined with normal Zonal (Pass A: post-alighting walks
    FROM stations TO buildings), gives the full bi-directional walking flow.
    """
    z = Zonal()
    z.load_layer("streets", highway_gdf)
    z.create_street_network(
        source_layer="streets",
        node_snapping_tolerance=C.NODE_SNAPPING_TOLERANCE_M,
        redundant_edge_treatment=C.REDUNDANT_EDGE_TREATMENT,
        turn_threshold_degree=C.TURN_THRESHOLD_DEG,
        turn_penalty_amount=C.TURN_PENALTY_M,
    )
    print(f"[network] reverse madina graph: "
          f"{len(z.network.nodes):,} nodes  "
          f"{len(z.network.edges):,} edges")

    # buildings -> origin (in reverse direction)
    z.load_layer("buildings", buildings_gdf)
    z.insert_node("buildings", label="origin", weight_attribute="W")

    # stations -> destination (in reverse direction)
    z.load_layer("stations", stations_gdf)
    z.insert_node("stations", label="destination", weight_attribute="W_init")

    z.create_graph(light_graph=True, d_graph=True)

    counts = z.network.nodes["type"].value_counts().to_dict()
    print(f"[network] reverse node types: {counts}")
    return z
