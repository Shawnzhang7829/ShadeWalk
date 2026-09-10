"""Conservation-area-aware view selection rule (shared by Phase 0 sampling and city-wide inference).

Rule:
    point ∈ conservation area (30m outward buffer):  check all 4 headings (view 1/2/3/4)
    point ∉ conservation area:                       check only the left/right sides (view 2/4)

Conservation areas:
    SG  — MasterPlan2025ConservationAreaBoundaryLayer.geojson  (EPSG:4326, 295 polys)
    BO  — Bologna_Arcade.gpkg, layer 'zone'                     (EPSG:25832, 4 zones)

Usage:
    from view_selection import tag_views
    df = tag_views(df, city="sg")   # df must contain lon/lat/view_idx columns
    keep = df[df["keep"]]           # rows with keep=True go to inference
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd

ROOT = Path(r"D:\Claude\SVI_FFW")

SG_CONS = ROOT / "Shp" / "SG" / "SG ConservationArea2025" / "MasterPlan2025ConservationAreaBoundaryLayer.geojson"
BO_CONS = ROOT / "Shp" / "Bologna" / "Bologna_Arcade.gpkg"   # layer 'zone'

EPSG_SG = 3414      # SVY21 (meter)
EPSG_BO = 32632     # UTM 32N (meter)

CONS_BUFFER_M = 30.0          # conservation area 30m outward buffer
PERP_VIEWS = {2, 4}           # outside: left/right sides
ALL_VIEWS = {1, 2, 3, 4}      # inside: all 4 headings


def _load_conservation_union(city: Literal["sg", "bo"]):
    """Read the conservation areas → project to a metric CRS → outward buffer → return the union geometry + EPSG."""
    if city == "sg":
        g = gpd.read_file(SG_CONS).to_crs(epsg=EPSG_SG)
        epsg = EPSG_SG
    elif city == "bo":
        g = gpd.read_file(BO_CONS, layer="zone").to_crs(epsg=EPSG_BO)
        epsg = EPSG_BO
    else:
        raise ValueError(f"city must be 'sg' or 'bo', got {city!r}")
    union = g.geometry.buffer(CONS_BUFFER_M).union_all()
    return union, epsg


def tag_views(df: pd.DataFrame, city: Literal["sg", "bo"]) -> pd.DataFrame:
    """Add two columns to df:
        in_conservation  — whether the point falls inside a conservation area (+30m buffer)
        keep             — whether to include it in inference (inside: keep all; outside: keep only 2/4)

    df must contain: lon, lat, view_idx
    """
    union, epsg = _load_conservation_union(city)

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"),
    ).to_crs(epsg=epsg)

    # compute within once per unique location (dedup for speed)
    uniq = gdf[["lon", "lat", "geometry"]].drop_duplicates(subset=["lon", "lat"]).copy()
    uniq["in_conservation"] = uniq.geometry.within(union)
    lut = uniq.set_index(["lon", "lat"])["in_conservation"].to_dict()

    df = df.copy()
    df["in_conservation"] = [lut.get((lo, la), False)
                             for lo, la in zip(df["lon"], df["lat"])]
    df["keep"] = df.apply(
        lambda r: (r["view_idx"] in ALL_VIEWS) if r["in_conservation"]
        else (r["view_idx"] in PERP_VIEWS),
        axis=1,
    )
    return df


if __name__ == "__main__":
    # self-check: count points inside/outside the conservation areas and the final kept amount for both cities
    import re

    NAME_RE = re.compile(
        r"^(?P<pid>\d+)_(?P<lon>-?\d+\.\d+)_(?P<lat>-?\d+\.\d+)"
        r"_(?P<date>\d{6})_baseheading(?P<bh>-?\d+\.\d+)"
        r"_viewheading(?P<vh>-?\d+\.\d+)_(?P<view>\d)\.jpg$")

    for city, svi_dir in [
        ("sg", ROOT / "SVI" / "Singapore" / "Singapore SVI"),
        ("bo", ROOT / "SVI" / "Bologna" / "Bologna SVI"),
    ]:
        print(f"\n=== {city.upper()} ===")
        rows = []
        for p in svi_dir.iterdir():
            m = NAME_RE.match(p.name)
            if m:
                rows.append({"lon": float(m["lon"]), "lat": float(m["lat"]),
                             "view_idx": int(m["view"])})
        d = pd.DataFrame(rows)
        d = tag_views(d, city=city)
        n_total = len(d)
        n_incons = int(d["in_conservation"].sum())
        n_keep = int(d["keep"].sum())
        n_keep_cons = int((d["keep"] & d["in_conservation"]).sum())
        n_keep_out = n_keep - n_keep_cons
        print(f"  total images        : {n_total}")
        print(f"  in conservation(+30): {n_incons}")
        print(f"  KEEP total          : {n_keep}  ({n_keep/n_total*100:.1f}%)")
        print(f"    - from conservation (4-view): {n_keep_cons}")
        print(f"    - from outside (2/4 only)   : {n_keep_out}")
