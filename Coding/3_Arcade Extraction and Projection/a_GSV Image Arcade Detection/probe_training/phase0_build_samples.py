"""Phase 0: before the 4-model smoke test, pick out the test sample subset.

Outputs:
    output/phase0_samples/sg_test_pts.csv      (SVI points inside the SG 4000pixel area + left/right view paths)
    output/phase0_samples/bo_positive.csv      (within ≤10m of Bologna portici, used as pseudo-label positives)
    output/phase0_samples/bo_negative.csv      (beyond ≥50m from Bologna portici, used as control negatives)

Only the views to the left/right of the driving direction are taken (view_idx == 2 or 4), halving the sample size.
"""
from __future__ import annotations
import re
import os
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

ROOT = Path(r"D:\Claude\SVI_FFW")
SVI_SG_DIR = ROOT / "SVI" / "Singapore" / "Singapore SVI"
SVI_BO_DIR = ROOT / "SVI" / "Bologna" / "Bologna SVI"

SG_PTS_SHP   = ROOT / "Shp" / "SG" / "Singapore_street view" / "Singapore_sample_points.shp"
SG_TEST_POLY = ROOT / "Shp" / "SG" / "4000pixel_polygon_SVY21" / "4000pixel_polygon_SVY21.shp"

BO_PTS_SHP   = ROOT / "Shp" / "Bologna" / "Bologna_street view" / "Italy_City_sample_points.shp"
BO_PORTICI   = ROOT / "Shp" / "Bologna" / "origini-di-bologna-portici" / "origini-di-bologna-portici.shp"

OUT_DIR = ROOT / "output" / "phase0_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CRS used: SVY21 (EPSG:3414) for SG, UTM-32N (EPSG:32632) for Bologna
EPSG_SG = 3414
EPSG_BO = 32632

# file name format: id_lon_lat_yyyymm_baseheadingXXX_viewheadingYYY_<1-4>.jpg
NAME_RE = re.compile(
    r"^(?P<pid>\d+)"
    r"_(?P<lon>-?\d+\.\d+)"
    r"_(?P<lat>-?\d+\.\d+)"
    r"_(?P<date>\d{6})"
    r"_baseheading(?P<bh>-?\d+\.\d+)"
    r"_viewheading(?P<vh>-?\d+\.\d+)"
    r"_(?P<view>\d)\.jpg$"
)


def index_svi_dir(svi_dir: Path) -> pd.DataFrame:
    """Scan the whole SVI directory and split file names into structured fields."""
    rows = []
    n_skipped = 0
    for p in svi_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".jpg":
            continue
        m = NAME_RE.match(p.name)
        if not m:
            n_skipped += 1
            continue
        d = m.groupdict()
        rows.append({
            "pid": int(d["pid"]),
            "lon": float(d["lon"]),
            "lat": float(d["lat"]),
            "date": d["date"],
            "baseheading": float(d["bh"]),
            "viewheading": float(d["vh"]),
            "view_idx": int(d["view"]),
            "path": str(p),
        })
    print(f"  indexed {len(rows)} jpgs (skipped {n_skipped} not-matching)")
    return pd.DataFrame(rows)


def filter_perpendicular(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the views to the left/right of the driving direction (view_idx 2/4)."""
    return df[df["view_idx"].isin([2, 4])].reset_index(drop=True)


def build_sg_sample():
    print("[SG] indexing SVI dir …")
    df = index_svi_dir(SVI_SG_DIR)
    df = filter_perpendicular(df)
    print(f"  perpendicular-only (views 2/4): {len(df)}")

    print("[SG] loading test polygon …")
    poly = gpd.read_file(SG_TEST_POLY).to_crs(epsg=EPSG_SG)
    test_geom = poly.unary_union

    print("[SG] spatial filter to 4000pixel polygon …")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"),
    ).to_crs(epsg=EPSG_SG)
    inside = gdf[gdf.geometry.within(test_geom)].copy()
    print(f"  in-polygon images: {len(inside)} "
          f"(unique pano ids: {inside['pid'].nunique()})")

    out = OUT_DIR / "sg_test_pts.csv"
    inside.drop(columns="geometry").to_csv(out, index=False)
    print(f"  wrote {out}")


def build_bo_samples():
    print("[BO] indexing SVI dir …")
    df = index_svi_dir(SVI_BO_DIR)
    df = filter_perpendicular(df)
    print(f"  perpendicular-only (views 2/4): {len(df)}")

    print("[BO] loading portici lines …")
    pol = gpd.read_file(BO_PORTICI).to_crs(epsg=EPSG_BO)
    pol_geom = pol.unary_union

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"),
    ).to_crs(epsg=EPSG_BO)

    # shortest distance to the portici lines
    print("[BO] computing distance to portici lines …")
    gdf["dist_m"] = gdf.geometry.distance(pol_geom)

    pos = gdf[gdf["dist_m"] <= 10].copy()
    neg = gdf[gdf["dist_m"] >= 50].copy().sample(n=min(len(pos) * 2, 4000), random_state=42)
    print(f"  positives (<=10m): {len(pos)}")
    print(f"  negatives (>=50m, sampled): {len(neg)}")

    out_p = OUT_DIR / "bo_positive.csv"
    out_n = OUT_DIR / "bo_negative.csv"
    pos.drop(columns="geometry").to_csv(out_p, index=False)
    neg.drop(columns="geometry").to_csv(out_n, index=False)
    print(f"  wrote {out_p}")
    print(f"  wrote {out_n}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0 — build smoke-test sample set")
    print("=" * 60)
    build_sg_sample()
    print()
    build_bo_samples()
    print("\nDONE.")
