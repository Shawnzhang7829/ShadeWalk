"""
Generic streaming vectorise + min_area filter for a large binary GeoTIFF.
Usage:
  python vectorise_generic.py --src_tif ... --out_gpkg ... [--min_area_m2 25]
"""
from __future__ import annotations
import argparse, time, numpy as np, rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shapely_shape
import geopandas as gpd

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--src_tif", required=True)
    P.add_argument("--out_gpkg", required=True)
    P.add_argument("--min_area_m2", type=float, default=25.0)
    a = P.parse_args()

    t0 = time.time()
    with rasterio.open(a.src_tif) as s:
        W, H = s.width, s.height
        tf = s.transform; crs = s.crs
        print(f"src {W:,}x{H:,} @ {abs(tf.a)} m  min_area={a.min_area_m2} m^2")
        arr = s.read(1)
    rows = []
    seen = 0
    for geom, val in rio_shapes(arr, mask=(arr > 0), transform=tf):
        seen += 1
        if val != 1:
            continue
        g = shapely_shape(geom)
        if g.area >= a.min_area_m2:
            rows.append({"geometry": g, "area_m2": round(g.area, 2)})
        if seen % 50000 == 0:
            print(f"  seen {seen:,}  kept {len(rows):,}  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    gdf = gpd.GeoDataFrame(rows, crs=crs).sort_values(
        "area_m2", ascending=False).reset_index(drop=True)
    gdf.to_file(a.out_gpkg, driver="GPKG")
    print(f"saved {a.out_gpkg}  ({len(gdf):,} polygons, "
          f"{gdf['area_m2'].sum()/1e6:.3f} km^2)  total {(time.time()-t0)/60:.1f} min")
