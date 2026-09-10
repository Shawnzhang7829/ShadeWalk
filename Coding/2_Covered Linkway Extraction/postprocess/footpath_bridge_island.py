"""
Deploy footpath-constrained closing to the FULL ISLAND (val-proven F1-safe:
ΔF1 = +0.0000, ΔIoU = +0.0000 on 303 val tiles).

bridge = ( closing(M, disk(G)) AND LTA-footpath-corridor ) AND NOT M
M_new  = M OR bridge          (G=15 px, corridor 3 m)

Windowed/streaming over the 179k x 115k island with halo so block edges
are correct.  Reads the Tversky pednet deliverable, writes a new bridged
TIF + GPKG (originals untouched).
"""
from __future__ import annotations
import os, time, numpy as np, rasterio
from rasterio.windows import Window, transform as win_tf
from rasterio.features import rasterize, shapes as rio_shapes
from shapely.geometry import box, shape as shp_shape
import geopandas as gpd
from skimage.morphology import binary_closing, disk

SRC_TIF  = r"C:\GeoSAM-backup\runs\autonomous_tversky\covered_linkway_SG_island_tv_pednet.tif"
PEDN     = r"C:\Users\City Syntax Lab\Desktop\Covered Linkways\1-data\1-SG\Footpath_Mar2026\Footpath.gpkg"
OUT_TIF  = r"C:\GeoSAM-backup\runs\autonomous_tversky\covered_linkway_SG_island_tv_pednet_bridged.tif"
OUT_GPKG = r"C:\GeoSAM-backup\runs\autonomous_tversky\covered_linkway_SG_island_tv_pednet_bridged.gpkg"

G_PX       = 15
CORRIDOR_M = 1.5
BLOCK      = 4096
MIN_AREA   = 25.0


def main():
    t0 = time.time()
    src = rasterio.open(SRC_TIF)
    H, W = src.height, src.width
    tf   = src.transform; crs = src.crs
    prof = src.profile.copy()
    pix  = abs(tf.a)
    print(f"src {W:,}x{H:,} @ {pix} m  G={G_PX}px(~{G_PX*pix:.1f}m, "
          f"bridge<= {2*G_PX*pix:.0f}m)  corridor={2*CORRIDOR_M:.0f}m")

    ped = gpd.read_file(PEDN)
    if ped.crs is None: ped.set_crs(crs, inplace=True)
    elif ped.crs != crs: ped = ped.to_crs(crs)
    sidx = ped.sindex
    st = disk(G_PX)
    halo = G_PX * 2 + 4

    prof.update(count=1, dtype="uint8", compress="lzw", tiled=True,
                blockxsize=512, blockysize=512, nodata=0, BIGTIFF="YES")
    nbx = (W + BLOCK - 1) // BLOCK
    nby = (H + BLOCK - 1) // BLOCK
    total = nbx * nby
    print(f"blocks {nby}x{nbx}={total}  halo={halo}px")
    done = 0; pos = 0; added = 0

    with rasterio.open(OUT_TIF, "w", **prof) as dst:
        for by in range(0, H, BLOCK):
            for bx in range(0, W, BLOCK):
                done += 1
                r0 = max(0, by - halo); r1 = min(H, by + BLOCK + halo)
                c0 = max(0, bx - halo); c1 = min(W, bx + BLOCK + halo)
                win = Window(c0, r0, c1 - c0, r1 - r0)
                M = (src.read(1, window=win) > 0)
                bh, bw = M.shape
                cy0 = by - r0; cx0 = bx - c0
                cy1 = cy0 + min(BLOCK, H - by)
                cx1 = cx0 + min(BLOCK, W - bx)
                if not M.any():
                    core = np.zeros((cy1-cy0, cx1-cx0), np.uint8)
                    dst.write(core, 1, window=Window(bx, by, core.shape[1], core.shape[0]))
                    continue
                wtf = win_tf(win, tf)
                x0, y0 = wtf.c, wtf.f
                x1 = x0 + bw*wtf.a; y1 = y0 + bh*wtf.e
                bb = box(min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1))
                cand = list(sidx.intersection(bb.bounds))
                if cand:
                    geoms = [ped.geometry.iloc[i].buffer(CORRIDOR_M) for i in cand]
                    corr = rasterize([(g,1) for g in geoms], out_shape=(bh,bw),
                                     transform=wtf, fill=0, dtype="uint8").astype(bool)
                    closed = binary_closing(M, footprint=st)
                    Mn = M | (closed & corr & (~M))
                else:
                    Mn = M
                core = Mn[cy0:cy1, cx0:cx1].astype(np.uint8)
                added += int(core.sum()) - int(M[cy0:cy1, cx0:cx1].sum())
                pos += int(core.sum())
                dst.write(core, 1, window=Window(bx, by, core.shape[1], core.shape[0]))
                if done % 100 == 0:
                    print(f"  block {done}/{total}  +{added:,}px  "
                          f"{(time.time()-t0)/60:.1f}min", flush=True)
    src.close()
    print(f"bridged px added: {added:,}  total pos: {pos:,}  "
          f"({(time.time()-t0)/60:.1f} min)")
    print(f"saved TIF: {OUT_TIF}")

    # vectorise
    print("vectorising...")
    rows = []
    with rasterio.open(OUT_TIF) as s:
        arr = s.read(1); vtf = s.transform; vcrs = s.crs
    for geom, val in rio_shapes(arr, mask=(arr > 0), transform=vtf):
        if val != 1: continue
        g = shp_shape(geom)
        if g.area >= MIN_AREA:
            rows.append({"geometry": g, "area_m2": round(g.area, 2)})
    gdf = gpd.GeoDataFrame(rows, crs=vcrs).sort_values(
        "area_m2", ascending=False).reset_index(drop=True)
    gdf.to_file(OUT_GPKG, driver="GPKG")
    print(f"saved GPKG: {OUT_GPKG}  ({len(gdf):,} polygons, "
          f"{gdf['area_m2'].sum()/1e6:.3f} km^2)  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
