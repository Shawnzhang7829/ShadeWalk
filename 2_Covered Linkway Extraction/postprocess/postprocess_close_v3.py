"""
Option A: enlarged morphological closing on v3 outputs to bridge short gaps
(tree-shadow / bus-stop / roof-material discontinuities).

- 4 km raster  : in-memory closing (fits RAM)
- full island  : windowed streaming closing with halo (handles 20 GB virtual)

Closing radius R px fills gaps up to ~2R px. Default R=17 px (~5 m at 0.3 m).
Outputs are NEW files (originals untouched).
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shapely_shape
from scipy.ndimage import binary_closing
from skimage.morphology import disk
import geopandas as gpd


def vectorise(mask_tif, out_gpkg, min_area_m2, streaming=False):
    with rasterio.open(mask_tif) as src:
        tf = src.transform; crs = src.crs
        rows = []
        if streaming:
            arr = src.read(1)
            it = rio_shapes(arr, mask=(arr > 0), transform=tf)
        else:
            arr = src.read(1)
            it = rio_shapes(arr, mask=(arr > 0), transform=tf)
        for geom, val in it:
            if val != 1:
                continue
            s = shapely_shape(geom)
            if s.area >= min_area_m2:
                rows.append({"geometry": s, "area_m2": round(s.area, 2)})
    if rows:
        g = gpd.GeoDataFrame(rows, crs=crs).sort_values(
            "area_m2", ascending=False).reset_index(drop=True)
        g.to_file(out_gpkg, driver="GPKG")
        return len(g), g["area_m2"].sum()
    return 0, 0.0


def close_small(input_tif, out_tif, out_gpkg, radius, min_area_m2):
    """In-memory closing for the 4 km raster."""
    print(f"[4km] closing radius={radius}px (~{radius*0.3:.1f} m)")
    with rasterio.open(input_tif) as src:
        mask = src.read(1).astype(bool)
        profile = src.profile.copy()
        pix = abs(src.transform.a)
    n0 = mask.sum()
    st = disk(radius).astype(bool)
    closed = binary_closing(mask, structure=st)
    n1 = closed.sum()
    out = closed.astype(np.uint8)
    profile.update(count=1, dtype="uint8", compress="lzw", nodata=0)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out, 1)
    npoly, area = vectorise(out_tif, out_gpkg, min_area_m2)
    print(f"[4km] px {n0:,} -> {n1:,} (+{100*(n1-n0)/max(n0,1):.1f}%)  "
          f"polygons={npoly}  area={area/1e6:.3f} km^2")
    print(f"[4km] saved: {out_tif}")
    print(f"[4km] saved: {out_gpkg}")


def close_island(input_tif, out_tif, radius, block=4096):
    """Windowed streaming closing with halo for the full island."""
    halo = radius * 2 + 2
    st = disk(radius).astype(bool)
    with rasterio.open(input_tif) as src:
        H, W = src.height, src.width
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", compress="lzw",
                   tiled=True, blockxsize=512, blockysize=512,
                   nodata=0, BIGTIFF="YES")
    nbx = (W + block - 1) // block
    nby = (H + block - 1) // block
    total = nbx * nby
    print(f"[island] closing radius={radius}px halo={halo}px  "
          f"{nby}x{nbx}={total} blocks")
    t0 = time.time()
    pos = 0
    done = 0
    with rasterio.open(input_tif) as src, \
         rasterio.open(out_tif, "w", **profile) as dst:
        for by in range(0, H, block):
            for bx in range(0, W, block):
                r0 = max(0, by - halo); r1 = min(H, by + block + halo)
                c0 = max(0, bx - halo); c1 = min(W, bx + block + halo)
                win = Window(c0, r0, c1 - c0, r1 - r0)
                sub = src.read(1, window=win).astype(bool)
                done += 1
                if sub.any():
                    sub = binary_closing(sub, structure=st)
                    # crop back to the core block region
                    cy0 = by - r0; cx0 = bx - c0
                    cy1 = cy0 + min(block, H - by)
                    cx1 = cx0 + min(block, W - bx)
                    core = sub[cy0:cy1, cx0:cx1].astype(np.uint8)
                else:
                    core = np.zeros((min(block, H - by), min(block, W - bx)),
                                    np.uint8)
                dst.write(core, 1, window=Window(bx, by,
                          core.shape[1], core.shape[0]))
                pos += int(core.sum())
                if done % 100 == 0:
                    el = time.time() - t0
                    print(f"  block {done}/{total}  pos={pos:,}  "
                          f"{el/60:.1f}min", flush=True)
    print(f"[island] done {(time.time()-t0)/60:.1f} min  pos px={pos:,}")
    print(f"[island] saved: {out_tif}")
    return pos


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--radius", type=int, default=17, help="closing radius px")
    P.add_argument("--min_area_m2", type=float, default=25.0)
    P.add_argument("--do_4km", action="store_true")
    P.add_argument("--do_island", action="store_true")
    args = P.parse_args()

    R = args.radius
    BASE = r"C:\GeoSAM-backup\runs\autonomous"

    if args.do_4km:
        close_small(
            input_tif = rf"{BASE}\covered_linkway_autonomous.tif",
            out_tif   = rf"{BASE}\covered_linkway_autonomous_closeR{R}.tif",
            out_gpkg  = rf"{BASE}\covered_linkway_autonomous_closeR{R}.gpkg",
            radius=R, min_area_m2=args.min_area_m2)

    if args.do_island:
        out_tif = rf"{BASE}\covered_linkway_SG_island_closeR{R}.tif"
        close_island(
            input_tif = rf"{BASE}\covered_linkway_SG_island.tif",
            out_tif   = out_tif, radius=R)
        print("[island] vectorising...")
        npoly, area = vectorise(
            out_tif, rf"{BASE}\covered_linkway_SG_island_closeR{R}.gpkg",
            args.min_area_m2, streaming=True)
        print(f"[island] polygons={npoly:,}  area={area/1e6:.3f} km^2")
        print(f"[island] saved: {BASE}\\covered_linkway_SG_island_closeR{R}.gpkg")
