"""
Post-process the autonomous inference output:
  1. Morphological closing (fill small gaps in linkway segments)
  2. Connected-component filter (drop small noise blobs)
  3. Vectorize cleaned mask

Outputs:
  covered_linkway_autonomous_post.tif   (cleaned binary GeoTIFF)
  covered_linkway_autonomous_post.gpkg  (cleaned polygons)
  postprocess_comparison.png            (before/after visualisation)
"""

from __future__ import annotations
import os, sys, numpy as np, rasterio
from rasterio.features import shapes as rio_shapes
from scipy.ndimage import binary_closing, generate_binary_structure, label as scipy_label
from skimage.morphology import disk as sk_disk
import geopandas as gpd
from shapely.geometry import shape as shapely_shape


INPUT_TIF   = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_autonomous.tif"
OUTPUT_TIF  = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_autonomous_post.tif"
OUTPUT_GPKG = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_autonomous_post.gpkg"

CLOSING_RADIUS_PX = 7      # ~2.1 m at 0.3 m/px
MIN_AREA_M2       = 25.0   # drop blobs smaller than this


def main():
    print(f"=== Post-processing {os.path.basename(INPUT_TIF)} ===\n")

    with rasterio.open(INPUT_TIF) as src:
        mask = src.read(1).astype(bool)
        transform = src.transform
        crs       = src.crs
        profile   = src.profile.copy()
        pix       = abs(transform.a)

    n0 = mask.sum()
    print(f"Step 0: original mask  = {n0:,} px  ({100*mask.mean():.3f}% of raster)")

    # ── Step 1: Morphological closing ──
    print(f"\nStep 1: morphological closing  radius={CLOSING_RADIUS_PX} px "
          f"({CLOSING_RADIUS_PX*pix:.2f} m)")
    struct = sk_disk(CLOSING_RADIUS_PX).astype(bool)
    mask_closed = binary_closing(mask, structure=struct)
    n1 = mask_closed.sum()
    print(f"        after closing = {n1:,} px  ({100*mask_closed.mean():.3f}%)  "
          f"+{100*(n1-n0)/n0:+.1f}% area")

    # ── Step 2: Connected-component filter ──
    min_area_px = int(MIN_AREA_M2 / (pix * pix))
    print(f"\nStep 2: drop blobs with area < {MIN_AREA_M2} m^2 "
          f"({min_area_px} px)")
    labeled, n_lbl = scipy_label(mask_closed)
    sizes = np.bincount(labeled.ravel())
    keep_ids = np.where(sizes >= min_area_px)[0]
    keep_ids = keep_ids[keep_ids != 0]      # drop background
    mask_final = np.isin(labeled, keep_ids).astype(np.uint8)
    n2 = mask_final.sum()
    print(f"        components before     = {n_lbl}")
    print(f"        components kept       = {len(keep_ids)}")
    print(f"        components dropped    = {n_lbl - len(keep_ids)}")
    print(f"        positive px           = {n2:,}  ({100*mask_final.mean():.3f}%)")

    # ── Step 3: Save GeoTIFF ──
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
        dst.write(mask_final, 1)
    print(f"\nSaved raster: {OUTPUT_TIF}  ({os.path.getsize(OUTPUT_TIF)/1024:.0f} KB)")

    # ── Step 4: Vectorise ──
    rows = []
    for geom_dict, val in rio_shapes(mask_final, transform=transform):
        if val != 1: continue
        s = shapely_shape(geom_dict)
        if s.area >= MIN_AREA_M2:
            rows.append({"geometry": s, "area_m2": round(s.area, 2)})
    if rows:
        gdf = gpd.GeoDataFrame(rows, crs=crs).sort_values("area_m2", ascending=False).reset_index(drop=True)
        gdf.to_file(OUTPUT_GPKG, driver="GPKG")
        print(f"Saved vector: {OUTPUT_GPKG}  ({len(gdf):,} polygons)")
        print(f"\nFinal polygon stats:")
        print(f"  count     : {len(gdf)}")
        print(f"  min area  : {gdf['area_m2'].min():.1f} m^2")
        print(f"  median    : {gdf['area_m2'].median():.1f} m^2")
        print(f"  mean      : {gdf['area_m2'].mean():.1f} m^2")
        print(f"  max       : {gdf['area_m2'].max():.1f} m^2")
        print(f"  total area: {gdf['area_m2'].sum():,.0f} m^2")
    else:
        print("No polygons remain after filtering!")

    return n0, n1, n2, n_lbl, len(keep_ids), len(rows)


if __name__ == "__main__":
    main()
