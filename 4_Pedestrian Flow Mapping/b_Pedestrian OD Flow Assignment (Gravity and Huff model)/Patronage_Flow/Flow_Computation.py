"""
Patronage_Flow / Flow_Computation.py    (v2 — madina-based)

Step 2 of the pipeline.  THIS IS THE CORE.

What changed from v1
--------------------
v1 (NKDE radial decay) distributed each station's ridership outward onto edges
with an exponential kernel, ignoring destinations. It was a stand-in for not
having building data.

v2 uses **patronage betweenness** -- the canonical pedestrian-flow model that
Sevtsuk's group (madina authors) recommend for transit-oriented analysis:

    For each station s with weight W_s = ridership(s, hour, daytype):
      For each building b within search_radius along the network with
      weight W_b = floor_area(b):
        share trip volume from s to b according to a Huff competition,
        with kernel exp(-beta * d(s,b)),
        and add the share to every edge along the (allowed-detour) path s->b.

Madina's `madina.una.tools.betweenness()` does this end-to-end.  Our job here
is to:
  1. write the right station ridership weights into the Zonal's origin nodes,
  2. call betweenness once per (DAY_TYPE, HOUR) we want,
  3. collect the per-edge results into a tidy long DataFrame.

For 24*2 = 48 hours, this is 48 betweenness calls. Each smoke-bbox call took
~50s single-core; the full island will be hours, not minutes. Use HOURS_WEEKDAY
and HOURS_WEEKEND in Constants to subset.
"""
from __future__ import annotations
import time

import numpy as np
import pandas as pd
import geopandas as gpd

from madina.una.tools import betweenness

from Patronage_Flow import Constants as C


# ---------------------------------------------------------------------------
# ridership table (monthly sums -> per-day means, indexed by PT_CODE)
# ---------------------------------------------------------------------------
def load_ridership_table(bus_csv=None, train_csv=None,
                         weekdays_per_month=None,
                         weekend_holidays_per_month=None) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by PT_CODE with columns = MultiIndex(DAY_TYPE, HOUR),
    values = MEAN people emitted from that station that hour on a typical day.

    LTA's CSV is a SUM across every weekday (or weekend/holiday) in the month --
    so we divide by the number of such days per month to get per-day means.

    Optional overrides:
      bus_csv / train_csv              -- alternate CSV paths (e.g. 2025 months)
      weekdays_per_month               -- actual weekday count for that month
      weekend_holidays_per_month       -- actual weekend/holiday count
    """
    if bus_csv is None:
        bus_csv = C.BUS_CSV
    if train_csv is None:
        train_csv = C.TRAIN_CSV
    if weekdays_per_month is None:
        weekdays_per_month = C.WEEKDAYS_PER_MONTH
    if weekend_holidays_per_month is None:
        weekend_holidays_per_month = C.WEEKEND_HOLIDAYS_PER_MONTH

    bus  = pd.read_csv(bus_csv)
    rail = pd.read_csv(train_csv)
    df = pd.concat([bus, rail], ignore_index=True)
    df = df.dropna(subset=["TIME_PER_HOUR", "PT_CODE", "DAY_TYPE"])
    df["PT_CODE"] = df["PT_CODE"].astype(str)
    df["HOUR"]    = df["TIME_PER_HOUR"].astype(int)

    # interchange codes "NS24/NE6/CC1" -> one row per line code,
    # ridership divided equally so total is preserved.
    # e.g. NS24/NE6/CC1 tap_out=61822 -> NS24=20607, NE6=20607, CC1=20607
    has_slash = df["PT_CODE"].str.contains("/", na=False)
    if has_slash.any():
        split = df[has_slash].copy()
        n_parts = split["PT_CODE"].str.count("/") + 1
        split["TOTAL_TAP_IN_VOLUME"]  = split["TOTAL_TAP_IN_VOLUME"]  / n_parts
        split["TOTAL_TAP_OUT_VOLUME"] = split["TOTAL_TAP_OUT_VOLUME"] / n_parts
        split["PT_CODE"] = split["PT_CODE"].str.split("/")
        split = split.explode("PT_CODE")
        df = pd.concat([df[~has_slash], split], ignore_index=True)

    if C.RIDERSHIP_MODE == "tap_out":
        df["W"] = df["TOTAL_TAP_OUT_VOLUME"]
    elif C.RIDERSHIP_MODE == "tap_in":
        df["W"] = df["TOTAL_TAP_IN_VOLUME"]
    else:
        df["W"] = df["TOTAL_TAP_IN_VOLUME"] + df["TOTAL_TAP_OUT_VOLUME"]

    days = {"WEEKDAY": weekdays_per_month,
            "WEEKENDS/HOLIDAY": weekend_holidays_per_month}
    df["W"] = df["W"] / df["DAY_TYPE"].map(days)

    wide = (df.groupby(["PT_CODE", "DAY_TYPE", "HOUR"])["W"].sum()
              .unstack(["DAY_TYPE", "HOUR"])
              .fillna(0.0))
    print(f"[flow] ridership table: {wide.shape[0]:,} stations x "
          f"{wide.shape[1]} (daytype, hour) bins")
    return wide


# ---------------------------------------------------------------------------
# write per-station ridership weights into the Zonal origin nodes
# ---------------------------------------------------------------------------
def _set_origin_weights(zonal,
                        stations_gdf: gpd.GeoDataFrame,
                        ridership_per_station: pd.Series,
                        only_source: str | None = None) -> None:
    """
    Update the 'weight' column on origin/origin_inactive nodes in
    zonal.network.nodes according to ridership for the chosen (DAY_TYPE, HOUR).

    Each row's effective weight = ridership[PT_CODE] / weight_divisor.

    Note: this *only* writes weight values. To make madina actually skip
    non-target origins (so it doesn't waste BFS work on them), use
    `_activate_origin_subset()` immediately before the betweenness call.
    """
    nodes = zonal.network.nodes
    # accept both 'origin' and 'origin_inactive' so this is a no-op-safe
    # operation regardless of the previous activation state.
    is_origin = nodes["type"].isin(["origin", "origin_inactive"])
    origin_rows = nodes[is_origin]

    src_idx   = origin_rows["source_id"].values
    pt_codes  = stations_gdf.loc[src_idx, "PT_CODE"].values
    sources   = stations_gdf.loc[src_idx, "source"].values
    divisors  = (stations_gdf.loc[src_idx, "weight_divisor"]
                              .astype(np.float32).values)
    raw_w = (ridership_per_station.reindex(pt_codes)
                                  .fillna(0.0)
                                  .astype(np.float32)
                                  .values)
    weights = raw_w / divisors
    if only_source is not None:
        weights = np.where(sources == only_source, weights, 0.0).astype(np.float32)
    nodes.loc[is_origin, "weight"] = weights


def _activate_origin_subset(zonal,
                            stations_gdf: gpd.GeoDataFrame,
                            keep_source: str) -> None:
    """
    Toggle node `type` so madina only iterates origins matching `keep_source`.

    madina's betweenness loop uses `node_gdf[node_gdf['type'] == 'origin']`
    -- it does NOT skip zero-weight origins, so we have to actually remove
    them from the origin set. We rename non-target origins to
    'origin_inactive' (an unused type label), and restore them later via
    `_restore_all_origins()`.
    """
    nodes = zonal.network.nodes
    # first: any previously inactive origin becomes active again so we can
    # then re-mask. This makes the function idempotent.
    nodes.loc[nodes["type"] == "origin_inactive", "type"] = "origin"

    is_origin = nodes["type"] == "origin"
    src_idx   = nodes.loc[is_origin, "source_id"].values
    sources   = stations_gdf.loc[src_idx, "source"].values

    # mark non-target origins as inactive
    inactive_idx = nodes.index[is_origin][sources != keep_source]
    nodes.loc[inactive_idx, "type"] = "origin_inactive"


def _restore_all_origins(zonal) -> None:
    """Restore any 'origin_inactive' nodes back to 'origin'."""
    nodes = zonal.network.nodes
    nodes.loc[nodes["type"] == "origin_inactive", "type"] = "origin"


# ---------------------------------------------------------------------------
# REVERSE DIRECTION (Pass B): buildings as origin, stations as destination
# ---------------------------------------------------------------------------
def _activate_destination_subset(zonal, stations_gdf, keep_source: str) -> None:
    """In a reverse-direction Zonal (stations are destinations),
    mask non-target stations by toggling their type to 'destination_inactive'.
    Idempotent — restores any previous masking first."""
    nodes = zonal.network.nodes
    nodes.loc[nodes["type"] == "destination_inactive", "type"] = "destination"

    is_dest = nodes["type"] == "destination"
    src_idx = nodes.loc[is_dest, "source_id"].values

    # only station nodes have a `source` mapping in stations_gdf;
    # building destinations (forward zonal) won't apply here -- guard via try
    try:
        sources = stations_gdf.loc[src_idx, "source"].values
    except KeyError:
        return  # not a station-destination zonal

    inactive_idx = nodes.index[is_dest][sources != keep_source]
    nodes.loc[inactive_idx, "type"] = "destination_inactive"


def _restore_all_destinations(zonal) -> None:
    """Restore any 'destination_inactive' back to 'destination'."""
    nodes = zonal.network.nodes
    nodes.loc[nodes["type"] == "destination_inactive", "type"] = "destination"


def _set_origin_weights_buildings(zonal_reverse,
                                   buildings_gdf: gpd.GeoDataFrame,
                                   density_per_archetype: pd.Series) -> None:
    """
    Pass B: set per-building origin weight = GFA * occupancy_density(arch, slot).
    Used in the reverse-direction zonal.
    """
    nodes = zonal_reverse.network.nodes
    is_origin = nodes["type"].isin(["origin", "origin_inactive"])
    src_idx = nodes.loc[is_origin, "source_id"].values
    archetypes = buildings_gdf.loc[src_idx, C.BUILDING_ARCHETYPE_COL].values
    gfas = (buildings_gdf.loc[src_idx, C.BUILDING_GFA_COL]
                          .astype(np.float32).values)
    densities = (pd.Series(archetypes).map(density_per_archetype)
                                      .fillna(C.OCCUPANCY_DEFAULT_DENSITY)
                                      .astype(np.float32).values)
    nodes.loc[is_origin, "weight"] = gfas * densities


def _set_destination_weights_stations(zonal_reverse,
                                       stations_gdf: gpd.GeoDataFrame,
                                       ridership_per_station: pd.Series,
                                       only_source: str | None = None) -> None:
    """
    Pass B: set per-station destination weight = tap_in / weight_divisor.
    `only_source` masks the OTHER kind to weight=0 (used as a defensive layer
    in addition to type masking).
    """
    nodes = zonal_reverse.network.nodes
    is_dest = nodes["type"].isin(["destination", "destination_inactive"])
    dest_rows = nodes[is_dest]

    src_idx   = dest_rows["source_id"].values
    pt_codes  = stations_gdf.loc[src_idx, "PT_CODE"].values
    sources   = stations_gdf.loc[src_idx, "source"].values
    divisors  = (stations_gdf.loc[src_idx, "weight_divisor"]
                              .astype(np.float32).values)
    raw_w = (ridership_per_station.reindex(pt_codes)
                                  .fillna(0.0)
                                  .astype(np.float32)
                                  .values)
    weights = raw_w / divisors
    if only_source is not None:
        weights = np.where(sources == only_source, weights, 0.0).astype(np.float32)
    nodes.loc[is_dest, "weight"] = weights


# ---------------------------------------------------------------------------
# split ridership loader: tap_in only / tap_out only
# ---------------------------------------------------------------------------
def load_ridership_split(mode: str) -> pd.DataFrame:
    """Same as load_ridership_table but force-uses one of: 'tap_in', 'tap_out'.
    Used for true dual-pass where we need each direction separately."""
    assert mode in ("tap_in", "tap_out"), f"unknown mode {mode!r}"
    bus  = pd.read_csv(C.BUS_CSV)
    rail = pd.read_csv(C.TRAIN_CSV)
    df = pd.concat([bus, rail], ignore_index=True)
    df = df.dropna(subset=["TIME_PER_HOUR", "PT_CODE", "DAY_TYPE"])
    df["PT_CODE"] = df["PT_CODE"].astype(str)
    df["HOUR"]    = df["TIME_PER_HOUR"].astype(int)

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


def load_occupancy_density() -> pd.DataFrame:
    """Load the per-archetype, per-slot occupancy density table.
    Returns DataFrame indexed by archetype with MultiIndex(DAY_TYPE, HOUR) cols,
    values = people per m^2 at that hour."""
    df = pd.read_parquet(C.OCCUPANCY_DENSITY_PARQUET)
    print(f"[flow] occupancy density table: {df.shape[0]} archetypes x "
          f"{df.shape[1]} (day_type, hour) bins")
    return df


def _set_destination_weights(zonal,
                             buildings_gdf: gpd.GeoDataFrame,
                             density_per_archetype: pd.Series) -> None:
    """
    Set per-building destination weight = GFA * occupancy_density(archetype, slot).

    `density_per_archetype` is a Series indexed by archetype name, values =
    people/m^2 at the current (DAY_TYPE, HOUR).
    """
    nodes = zonal.network.nodes
    is_dest = nodes["type"] == "destination"
    dest_rows = nodes[is_dest]
    src_idx = dest_rows["source_id"].values
    archetypes = buildings_gdf.loc[src_idx, C.BUILDING_ARCHETYPE_COL].values
    gfas = (buildings_gdf.loc[src_idx, C.BUILDING_GFA_COL]
                          .astype(np.float32).values)
    densities = (pd.Series(archetypes).map(density_per_archetype)
                                      .fillna(C.OCCUPANCY_DEFAULT_DENSITY)
                                      .astype(np.float32).values)
    nodes.loc[is_dest, "weight"] = gfas * densities


# ---------------------------------------------------------------------------
# main hourly flow loop
# ---------------------------------------------------------------------------
def compute_hourly_flow(zonal,
                        stations_gdf: gpd.GeoDataFrame,
                        buildings_gdf: gpd.GeoDataFrame,
                        ridership: pd.DataFrame,
                        hours_weekday=None,
                        hours_weekend=None,
                        beta: float = C.BETA,
                        n_workers: int = C.N_WORKERS,
                        output_dir=None,
                        ) -> pd.DataFrame:
    """
    For each (DAY_TYPE, HOUR):
      1. update destination weights = GFA * occupancy_density(archetype, slot)
      2. run TWO sub-passes of betweenness:
         - bus-only origins at R = BUS_RADIUS_M  (default 400 m)
         - rail-only (MRT+LRT) origins at R = RAIL_RADIUS_M  (default 800 m)
      3. sum the two flow columns -> per-slot flow

    If output_dir is provided, writes flow_<tag>.gpkg immediately after each
    slot completes (so results are available in QGIS before all slots finish).

    Returns a long DataFrame: edge_id, DAY_TYPE, HOUR, flow.
    """
    from pathlib import Path

    if hours_weekday is None:
        hours_weekday = C.HOURS_WEEKDAY
    if hours_weekend is None:
        hours_weekend = C.HOURS_WEEKEND

    schedule = []
    for h in hours_weekday:
        schedule.append(("WEEKDAY", h))
    for h in hours_weekend:
        schedule.append(("WEEKENDS/HOLIDAY", h))

    occupancy = load_occupancy_density()

    print(f"[flow] running {len(schedule)} slots x 2 sub-passes "
          f"(bus R={C.BUS_RADIUS_M:.0f}m, rail R={C.RAIL_RADIUS_M:.0f}m, "
          f"detour={C.DETOUR_RATIO}, beta={beta:.4f}, cores={n_workers})")
    if output_dir is not None:
        print(f"[flow] instant per-slot GPKG output -> {output_dir}")

    out_long = []
    for i, (daytype, hour) in enumerate(schedule, 1):
        # ridership for this slot
        if (daytype, hour) in ridership.columns:
            ridership_slot = ridership[(daytype, hour)]
        else:
            ridership_slot = pd.Series(0.0, index=ridership.index)

        # destination weights for this slot
        if (daytype, hour) in occupancy.columns:
            density_slot = occupancy[(daytype, hour)]
        else:
            density_slot = pd.Series(0.0, index=occupancy.index)
        _set_destination_weights(zonal, buildings_gdf, density_slot)

        ed = zonal.network.edges
        flow_total = np.zeros(len(ed), dtype=np.float64)
        elapsed_total = 0.0

        # --- sub-pass 1: bus origins, R=400 -----------------------------------
        _set_origin_weights(zonal, stations_gdf, ridership_slot, only_source="BUS")
        _activate_origin_subset(zonal, stations_gdf, keep_source="BUS")
        t0 = time.perf_counter()
        betweenness(
            zonal=zonal,
            search_radius=C.BUS_RADIUS_M,
            detour_ratio=C.DETOUR_RATIO,
            decay=True,
            decay_method="exponent",
            beta=beta,
            num_cores=n_workers,
            closest_destination=C.CLOSEST_DESTINATION,
            save_betweenness_as=None,
        )
        elapsed_total += time.perf_counter() - t0
        flow_total += zonal.network.edges["betweenness"].astype(np.float64).values

        # --- sub-pass 2: rail (MRT+LRT) origins, R=800 ------------------------
        _set_origin_weights(zonal, stations_gdf, ridership_slot, only_source="MRT")
        _activate_origin_subset(zonal, stations_gdf, keep_source="MRT")
        t0 = time.perf_counter()
        betweenness(
            zonal=zonal,
            search_radius=C.RAIL_RADIUS_M,
            detour_ratio=C.DETOUR_RATIO,
            decay=True,
            decay_method="exponent",
            beta=beta,
            num_cores=n_workers,
            closest_destination=C.CLOSEST_DESTINATION,
            save_betweenness_as=None,
        )
        elapsed_total += time.perf_counter() - t0
        flow_total += zonal.network.edges["betweenness"].astype(np.float64).values

        # restore origins so subsequent slots (or external calls) work cleanly
        _restore_all_origins(zonal)

        nz = (flow_total > 0).sum()
        print(f"  [{i:>2}/{len(schedule)}] {daytype:<18} hour={hour:>2}  "
              f"{elapsed_total:>5.1f}s   non-zero={nz:>6,}   "
              f"max={flow_total.max():>9.1f}")

        slot = pd.DataFrame({
            "edge_id":   ed.index.values,
            "DAY_TYPE":  daytype,
            "HOUR":      hour,
            "flow":      flow_total,
        })
        out_long.append(slot)

        # --- instant GPKG output per slot ------------------------------------
        if output_dir is not None:
            tag = f"{daytype.split('/')[0].lower()}_{hour:02d}"
            gpkg_out = ed.copy()
            gpkg_out["flow"] = flow_total
            gpkg_path = Path(output_dir) / f"flow_{tag}.gpkg"
            gpkg_out.to_file(gpkg_path, driver="GPKG")
            print(f"  [flow] wrote {gpkg_path.name}  "
                  f"(p99={gpkg_out['flow'].quantile(0.99):.1f}  "
                  f"max={gpkg_out['flow'].max():.1f})")

    return pd.concat(out_long, ignore_index=True)
