"""Build the SG official validation set from CoveredLinkWay ∩ Building footprint (the equivalent of Bologna portici).

Logic (specified by the user):
    the part of CoveredLinkWay overlapping a building footprint = where a five-foot way is
    → street view images there should contain a five-foot way → used as SG positive validation

Outputs:
    output/phase1c_sg_val/sg_val_positive.csv   (SVI points within ≤12m of the overlap + left/right or all headings)
    output/phase1c_sg_val/sg_val_negative.csv   (SVI points ≥60m from any CoveredLinkWay)
    output/phase1c_sg_val/overlap_stats.txt
"""
from __future__ import annotations
import re
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

ROOT = Path(r"D:\Claude\SVI_FFW")
OUT = ROOT / "output" / "phase1c_sg_val"
OUT.mkdir(parents=True, exist_ok=True)

CLW = ROOT / "Shp" / "SG" / "CoveredLinkWay_Mar2026" / "CoveredLinkWay.shp"
BLD = ROOT / "Shp" / "SG" / "SG_Building" / "SG_Building_SVY21_TH.shp"
SVI = ROOT / "SVI" / "Singapore" / "Singapore SVI"
EPSG = 3414

POS_DIST = 12.0      # SVI point ≤12m from the overlap → positive
NEG_DIST = 60.0      # ≥60m from any CLW → negative
MIN_OVERLAP_M2 = 3.0 # overlap-area filter to drop slivers

NAME_RE = re.compile(
    r"^(?P<pid>\d+)_(?P<lon>-?\d+\.\d+)_(?P<lat>-?\d+\.\d+)"
    r"_(?P<date>\d{6})_baseheading(?P<bh>-?\d+\.\d+)"
    r"_viewheading(?P<vh>-?\d+\.\d+)_(?P<view>\d)\.jpg$")


def index_svi():
    rows = []
    for p in SVI.iterdir():
        m = NAME_RE.match(p.name)
        if m:
            rows.append({"pid": int(m["pid"]), "lon": float(m["lon"]),
                         "lat": float(m["lat"]), "baseheading": float(m["bh"]),
                         "viewheading": float(m["vh"]), "view_idx": int(m["view"]),
                         "path": str(p)})
    return pd.DataFrame(rows)


def main():
    print("[1] loading CoveredLinkWay + buildings …")
    clw = gpd.read_file(CLW).to_crs(EPSG)
    bld = gpd.read_file(BLD).to_crs(EPSG)
    print(f"    CLW polys: {len(clw)}, buildings: {len(bld)}")

    print("[2] computing CLW ∩ building overlap …")
    # overlay intersection with the building union
    clw_valid = clw[clw.geometry.notna() & clw.geometry.is_valid].copy()
    clw_valid["geometry"] = clw_valid.geometry.buffer(0)   # fix self-intersections
    bld_valid = bld[bld.geometry.notna()].copy()
    bld_valid["geometry"] = bld_valid.geometry.buffer(0)

    inter = gpd.overlay(clw_valid[["OBJECTID", "SDSM_h", "geometry"]],
                        bld_valid[["height", "geometry"]],
                        how="intersection", keep_geom_type=True)
    inter["ov_area"] = inter.geometry.area
    inter = inter[inter["ov_area"] >= MIN_OVERLAP_M2].copy()
    total_ov = inter["ov_area"].sum()
    print(f"    overlap polygons (>= {MIN_OVERLAP_M2} m2): {len(inter)}")
    print(f"    total overlap area: {total_ov/1e4:.2f} ha")
    print(f"    SDSM_h (linkway height) mean/median: "
          f"{inter['SDSM_h'].mean():.2f} / {inter['SDSM_h'].median():.2f} m")

    overlap_union = inter.geometry.union_all()
    clw_union = clw_valid.geometry.union_all()

    print("[3] indexing SVI …")
    df = index_svi()
    print(f"    SVI images: {len(df)}")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat, crs="EPSG:4326")).to_crs(EPSG)

    print("[4] distance to overlap / clw …")
    # compute distances for unique locations only (faster)
    uniq = gdf.drop_duplicates("pid")[["pid", "geometry"]].copy()
    uniq["d_overlap"] = uniq.geometry.distance(overlap_union)
    uniq["d_clw"] = uniq.geometry.distance(clw_union)
    dmap_ov = uniq.set_index("pid")["d_overlap"].to_dict()
    dmap_clw = uniq.set_index("pid")["d_clw"].to_dict()
    gdf["d_overlap"] = gdf["pid"].map(dmap_ov)
    gdf["d_clw"] = gdf["pid"].map(dmap_clw)

    pos = gdf[gdf["d_overlap"] <= POS_DIST].copy()
    neg = gdf[gdf["d_clw"] >= NEG_DIST].copy()
    print(f"    positive points (<= {POS_DIST}m to overlap): {len(pos)} imgs, {pos.pid.nunique()} panos")
    print(f"    negative points (>= {NEG_DIST}m to any CLW): {len(neg)} imgs, {neg.pid.nunique()} panos")

    pos.drop(columns="geometry").to_csv(OUT / "sg_val_positive.csv", index=False)
    # too many negatives; randomly sample 2x the positives
    neg_s = neg.sample(n=min(len(neg), len(pos) * 2), random_state=42)
    neg_s.drop(columns="geometry").to_csv(OUT / "sg_val_negative.csv", index=False)

    with open(OUT / "overlap_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"CLW polys: {len(clw)}\n")
        f.write(f"overlap polygons (>= {MIN_OVERLAP_M2} m2): {len(inter)}\n")
        f.write(f"total overlap area ha: {total_ov/1e4:.2f}\n")
        f.write(f"SDSM_h mean/median m: {inter['SDSM_h'].mean():.2f} / {inter['SDSM_h'].median():.2f}\n")
        f.write(f"positive imgs: {len(pos)} ({pos.pid.nunique()} panos)\n")
        f.write(f"negative imgs (sampled): {len(neg_s)}\n")
    print(f"[done] wrote validation CSVs to {OUT}")


if __name__ == "__main__":
    main()
