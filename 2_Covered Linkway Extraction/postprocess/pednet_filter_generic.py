"""
Generic pedestrian-network soft constraint.

Keep predicted polygons that intersect a BUFFER_M buffer of the pedestrian
network; attach pednet_overlap_ratio. Originals untouched (new outputs).

Usage:
  python pednet_filter_generic.py --src_gpkg ... --src_tif ... \
         --out_gpkg ... --out_tif ... [--buffer 5]
"""
from __future__ import annotations
import os, argparse, time
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.ops import unary_union

PEDNET = r"D:\Claude\GeoSAM-TopoLoRA\shp\footpath\pedestrian_network_filtered.gpkg"


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--src_gpkg", required=True)
    P.add_argument("--src_tif",  required=True)
    P.add_argument("--out_gpkg", required=True)
    P.add_argument("--out_tif",  required=True)
    P.add_argument("--buffer",   type=float, default=5.0)
    P.add_argument("--pednet",   default=PEDNET)
    args = P.parse_args()

    t0 = time.time()
    print(f"=== pednet soft constraint  buffer={args.buffer} m ===")
    gdf = gpd.read_file(args.src_gpkg)
    print(f"src polygons: {len(gdf):,}  area={gdf['area_m2'].sum()/1e6:.3f} km^2  CRS={gdf.crs}")

    ped = gpd.read_file(args.pednet)
    if ped.crs != gdf.crs:
        ped = ped.to_crs(gdf.crs)
    print(f"pednet lines: {len(ped):,}  buffering {args.buffer} m + dissolve...")
    t1 = time.time()
    buf = unary_union(ped.geometry.buffer(args.buffer).values)
    print(f"  buffer area={buf.area/1e6:.2f} km^2  ({(time.time()-t1)/60:.1f} min)")

    inter = gdf.geometry.intersects(buf)
    print(f"intersect: {inter.sum():,}/{len(gdf):,} ({100*inter.mean():.1f}%)")
    kept = gdf[inter].copy().reset_index(drop=True)
    t2 = time.time()
    kept["overlap_area_m2"] = kept.geometry.intersection(buf).area
    kept["pednet_overlap_ratio"] = (kept["overlap_area_m2"] / kept["area_m2"]).round(3)
    print(f"  overlap-ratio computed ({(time.time()-t2)/60:.1f} min)")
    final = kept[kept["pednet_overlap_ratio"] > 0.0].sort_values(
        "area_m2", ascending=False).reset_index(drop=True)

    print(f"FINAL: {len(final):,} polygons "
          f"({100*len(final)/len(gdf):.1f}% of src)  "
          f"area={final['area_m2'].sum()/1e6:.3f} km^2 "
          f"(kept {100*final['area_m2'].sum()/gdf['area_m2'].sum():.1f}%)")
    print("overlap_ratio:", final["pednet_overlap_ratio"].describe().to_dict())

    final.to_file(args.out_gpkg, driver="GPKG")
    print(f"saved GPKG: {args.out_gpkg}  ({os.path.getsize(args.out_gpkg)/1e6:.1f} MB)")

    # rasterise filtered polygons onto src_tif grid
    with rasterio.open(args.src_tif) as s:
        H, W = s.height, s.width
        tf = s.transform; prof = s.profile.copy()
    rast = rasterize([(g, 1) for g in final.geometry],
                     out_shape=(H, W), transform=tf, fill=0, dtype="uint8")
    prof.update(count=1, dtype="uint8", compress="lzw",
                tiled=True, blockxsize=512, blockysize=512,
                nodata=0, BIGTIFF="YES")
    with rasterio.open(args.out_tif, "w", **prof) as d:
        d.write(rast, 1)
    print(f"saved TIF : {args.out_tif}  pos px={int(rast.sum()):,}")
    print(f"total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
