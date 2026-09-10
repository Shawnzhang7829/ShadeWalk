"""
Post-process full-island autonomous mask:
  1. Vectorize the GeoTIFF directly (streaming, low memory)
  2. Filter polygons by min_area_m2
  3. Save GPKG
  4. Report stats + write downsampled overview PNG for quick inspection

For 20+ GB virtual mask, we use streaming rio_shapes and avoid loading
the whole array into Python.  Visualisation uses a 1/64 overview.
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shapely_shape
import geopandas as gpd

INPUT_TIF   = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island.tif"
OUTPUT_GPKG = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island.gpkg"
OVERVIEW_PNG = r"C:\GeoSAM-backup\runs\autonomous\SG_island_overview.png"

MIN_AREA_M2 = 25.0


def main():
    t0 = time.time()
    with rasterio.open(INPUT_TIF) as src:
        H, W = src.height, src.width
        transform = src.transform
        crs       = src.crs
        pix       = abs(transform.a)
    print(f"Input:  {W:,} x {H:,} px @ {pix} m   CRS={crs}")
    print(f"Vector min_area_m2 = {MIN_AREA_M2}")
    print()

    # ── 1. Streaming vectorise ──
    print("Step 1: streaming vectorise (rio_shapes)...")
    t1 = time.time()
    rows = []
    total_seen = 0
    with rasterio.open(INPUT_TIF) as src:
        # mask out 0 before polygonising so background isn't included
        for geom_dict, val in rio_shapes(src.read(1, masked=False),
                                         mask=(src.read(1) > 0),
                                         transform=transform):
            total_seen += 1
            if val != 1:
                continue
            s = shapely_shape(geom_dict)
            a = s.area
            if a >= MIN_AREA_M2:
                rows.append({"geometry": s, "area_m2": round(a, 2)})
            if total_seen % 50000 == 0:
                print(f"  seen {total_seen:,} shapes, kept {len(rows):,}  "
                      f"elapsed={(time.time()-t1)/60:.1f}min", flush=True)
    print(f"  scan complete: {total_seen:,} shapes seen, {len(rows):,} kept "
          f"in {(time.time()-t1)/60:.1f} min")

    if not rows:
        print("No polygons above threshold!")
        return

    gdf = gpd.GeoDataFrame(rows, crs=crs).sort_values("area_m2",
            ascending=False).reset_index(drop=True)
    gdf.to_file(OUTPUT_GPKG, driver="GPKG")
    print(f"\nSaved {OUTPUT_GPKG}  ({len(gdf):,} polygons, "
          f"{os.path.getsize(OUTPUT_GPKG)/1e6:.1f} MB)")

    # ── 2. Stats ──
    print(f"\nPolygon size distribution (m^2):")
    print(gdf['area_m2'].describe().to_string())
    print()
    print('Size buckets:')
    for lo, hi in [(25, 100), (100, 500), (500, 2000), (2000, 10000), (10000, 1e9)]:
        sel = gdf[(gdf['area_m2'] >= lo) & (gdf['area_m2'] < hi)]
        pct = 100 * len(sel) / max(len(gdf), 1)
        ar = sel['area_m2'].sum()
        print(f'  {lo:>6}-{hi if hi < 1e9 else "inf":>5} m^2 : '
              f'{len(sel):>6,} polygons ({pct:>5.1f}%)  total {ar:>12,.0f} m^2')
    print(f"\nTotal predicted linkway area: {gdf['area_m2'].sum():,.0f} m^2 "
          f"({gdf['area_m2'].sum()/1e6:.2f} km^2)")

    # ── 3. Overview PNG (downsample 1/64) ──
    print(f"\nStep 3: generating overview PNG (1/64 downsampled)...")
    OVR = 64
    with rasterio.open(INPUT_TIF) as src:
        mask_small = src.read(1, out_shape=(H // OVR, W // OVR))
    print(f"  overview shape: {mask_small.shape}, positive px: {mask_small.sum():,}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(mask_small, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f'Singapore Covered Linkway (autonomous GeoSAM+TopoLoRA)\n'
                 f'{len(gdf):,} polygons, total {gdf["area_m2"].sum()/1e6:.2f} km^2',
                 fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OVERVIEW_PNG, dpi=200, bbox_inches='tight')
    print(f"  Saved: {OVERVIEW_PNG}")

    print(f"\nTotal post-process time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
