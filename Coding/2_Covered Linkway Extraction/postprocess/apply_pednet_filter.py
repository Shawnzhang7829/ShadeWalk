"""
Apply pedestrian-network soft constraint to the full-island autonomous output.

Soft constraint:
  - A real covered linkway must lie along / near a pedestrian path.
  - We buffer the pedestrian network by BUFFER_M metres and KEEP only the
    predicted polygons whose centroid OR ANY interior point falls inside
    the buffer.
  - Polygons that intersect get an attribute `pednet_overlap_ratio` showing
    how much of the polygon lies within the buffer (0-1).

This does NOT modify the original GPKG / TIF.  New outputs:
  covered_linkway_SG_island_pednet.gpkg     filtered polygons
  covered_linkway_SG_island_pednet.tif      rasterised filtered mask
  pednet_filter_comparison.png              before/after overview
"""

from __future__ import annotations

import os, time
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.ops import unary_union

# Inputs
SRC_GPKG     = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island.gpkg"
SRC_TIF      = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island.tif"
PEDNET_GPKG  = r"D:\Claude\GeoSAM-TopoLoRA\shp\footpath\pedestrian_network_filtered.gpkg"

# Outputs (new files; originals untouched)
OUT_GPKG = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island_pednet.gpkg"
OUT_TIF  = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island_pednet.tif"
OUT_PNG  = r"C:\GeoSAM-backup\runs\autonomous\pednet_filter_comparison.png"

# Parameters
BUFFER_M = 5.0   # pedestrian network buffer radius
KEEP_MIN_OVERLAP = 0.0  # >0 means strictly overlap; 0 = any touch (soft)


def main():
    t0 = time.time()
    print(f"=== Pedestrian-network soft constraint ===")
    print(f"buffer = {BUFFER_M} m   keep if overlap_ratio > {KEEP_MIN_OVERLAP}")
    print()

    # ── 1. Load predicted polygons ──
    print(f"Loading predicted polygons: {SRC_GPKG}")
    gdf = gpd.read_file(SRC_GPKG)
    print(f"  {len(gdf):,} polygons, CRS={gdf.crs}, total {gdf['area_m2'].sum()/1e6:.2f} km^2")

    # ── 2. Load + buffer pedestrian network ──
    print(f"\nLoading pedestrian network: {PEDNET_GPKG}")
    ped = gpd.read_file(PEDNET_GPKG)
    if ped.crs != gdf.crs:
        ped = ped.to_crs(gdf.crs)
    print(f"  {len(ped):,} line features")
    print(f"\nBuffering by {BUFFER_M} m + dissolving (may take 1-2 min)...")
    t1 = time.time()
    ped_buf = ped.geometry.buffer(BUFFER_M)
    buf_union = unary_union(ped_buf.values)
    print(f"  buffer ready  area={buf_union.area/1e6:.2f} km^2  ({(time.time()-t1)/60:.1f} min)")

    # ── 3. Spatial filter: keep polygons intersecting the buffer ──
    print(f"\nFiltering polygons against buffer...")
    t2 = time.time()
    intersects_mask = gdf.geometry.intersects(buf_union)
    print(f"  {intersects_mask.sum():,} / {len(gdf):,} polygons intersect buffer "
          f"({100*intersects_mask.mean():.1f}%)  in {time.time()-t2:.1f}s")

    # ── 4. Compute overlap ratio for kept polygons ──
    print(f"\nComputing overlap ratio (vector intersection)...")
    t3 = time.time()
    kept = gdf[intersects_mask].copy().reset_index(drop=True)
    # vectorised: intersection area / polygon area
    kept["overlap_area_m2"] = kept.geometry.intersection(buf_union).area
    kept["pednet_overlap_ratio"] = (kept["overlap_area_m2"] / kept["area_m2"]).round(3)
    print(f"  done in {(time.time()-t3)/60:.1f} min")

    # Apply final threshold (soft: 0 by default = keep all that touch)
    final = kept[kept["pednet_overlap_ratio"] > KEEP_MIN_OVERLAP].copy()
    print(f"\nFinal kept: {len(final):,} polygons  ({100*len(final)/len(gdf):.1f}% of original)")
    print(f"  total area: {final['area_m2'].sum()/1e6:.2f} km^2 "
          f"(was {gdf['area_m2'].sum()/1e6:.2f}, kept {100*final['area_m2'].sum()/gdf['area_m2'].sum():.1f}%)")
    print()
    print("overlap_ratio distribution among kept:")
    print(final["pednet_overlap_ratio"].describe().to_string())

    # ── 5. Save filtered GPKG ──
    final = final.sort_values("area_m2", ascending=False).reset_index(drop=True)
    final.to_file(OUT_GPKG, driver="GPKG")
    print(f"\nSaved GPKG: {OUT_GPKG}  ({os.path.getsize(OUT_GPKG)/1e6:.1f} MB)")

    # ── 6. Rasterise filtered polygons (same geometry as SRC_TIF) ──
    print(f"\nRasterising filtered polygons (same geotransform as {SRC_TIF})...")
    with rasterio.open(SRC_TIF) as src:
        H, W = src.height, src.width
        tf   = src.transform
        crs  = src.crs
        profile = src.profile.copy()
    # Stream rasterise (one polygon at a time is slow; use rasterize bulk)
    raster = rasterize([(g, 1) for g in final.geometry],
                       out_shape=(H, W), transform=tf,
                       fill=0, dtype="uint8")
    profile.update(count=1, dtype="uint8", compress="lzw",
                   tiled=True, blockxsize=512, blockysize=512,
                   nodata=0, BIGTIFF="YES")
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(raster, 1)
    print(f"Saved TIF: {OUT_TIF}  ({os.path.getsize(OUT_TIF)/1e6:.1f} MB)")
    print(f"  positive px in filtered raster: {int(raster.sum()):,} "
          f"({100*raster.mean():.3f}% of raster)")

    # ── 7. Side-by-side overview PNG ──
    print(f"\nGenerating comparison overview PNG...")
    import matplotlib.pyplot as plt
    OVR = 64
    INPUT_TIF = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\SG_google_map_03m_SVY21.tif"
    from scipy.ndimage import binary_dilation
    with rasterio.open(INPUT_TIF) as src:
        rgb = src.read([1,2,3], out_shape=(3, H//OVR, W//OVR)).transpose(1,2,0).astype(np.float32)
        rgb = (rgb - rgb.min()) / (rgb.ptp() + 1e-6)
    with rasterio.open(SRC_TIF) as src:
        m_before = src.read(1, out_shape=(H//OVR, W//OVR))
    m_after = raster[::OVR, ::OVR].copy() if raster.shape == (H, W) else None
    if m_after is None:
        # fallback: downsample written TIF
        with rasterio.open(OUT_TIF) as src:
            m_after = src.read(1, out_shape=(H//OVR, W//OVR))
    m_before_vis = binary_dilation(m_before > 0, iterations=2)
    m_after_vis  = binary_dilation(m_after  > 0, iterations=2)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    axes[0].imshow(rgb); axes[0].set_title("Singapore RGB (1/64)", fontsize=11); axes[0].axis("off")
    o1 = rgb.copy(); o1[m_before_vis] = [1, 0.15, 0.15]
    axes[1].imshow(o1)
    axes[1].set_title(f"Before pednet filter\n{len(gdf):,} polygons / {gdf['area_m2'].sum()/1e6:.2f} km^2",
                     fontsize=11, color="#C0392B"); axes[1].axis("off")
    o2 = rgb.copy(); o2[m_after_vis] = [1, 0.15, 0.15]
    axes[2].imshow(o2)
    axes[2].set_title(f"After pednet soft constraint ({BUFFER_M} m buffer)\n"
                     f"{len(final):,} polygons / {final['area_m2'].sum()/1e6:.2f} km^2",
                     fontsize=11, color="#27AE60"); axes[2].axis("off")
    plt.suptitle("GeoSAM+TopoLoRA autonomous: pedestrian-network soft constraint",
                fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"Saved overview: {OUT_PNG}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
