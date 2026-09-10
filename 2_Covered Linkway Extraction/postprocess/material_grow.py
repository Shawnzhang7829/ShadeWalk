"""
Material-guided region growing on top of the post-processed v3 mask.

Idea: a covered linkway has a consistent roof material/colour. Take the
high-confidence post-processed prediction as seeds, sample each segment's
roof colour signature (median CIELAB), then iteratively grow the mask only
into neighbouring pixels whose colour matches that signature within deltaE.
Growth stops automatically at material boundaries (roof ends / ground).

This extends linkways along their true continuation WITHOUT the lateral
fattening / blob artefacts of isotropic morphological closing.

Base (seed)  : covered_linkway_autonomous_post.tif   (do NOT overwrite)
Output       : covered_linkway_autonomous_post_grow.tif / .gpkg
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import cv2
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shapely_shape
from scipy.ndimage import label as sci_label, grey_dilation
import geopandas as gpd


def grow(input_rgb_tif, seed_tif, out_tif, out_gpkg,
         delta_e=14.0, step_px=3, max_iter=10, min_area_m2=25.0,
         rgb_bands=(1, 2, 3), verbose=True):
    t0 = time.time()
    with rasterio.open(seed_tif) as s:
        seed = (s.read(1) > 0)
        tf = s.transform; crs = s.crs; profile = s.profile.copy()
        H, W = s.height, s.width
        pix = abs(tf.a)
    with rasterio.open(input_rgb_tif) as s:
        # read same window/extent (input is the 4km RGB, same grid as seed)
        rgb = s.read(list(rgb_bands)).transpose(1, 2, 0).astype(np.uint8)
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    n0 = int(seed.sum())
    if verbose:
        print(f"seed px = {n0:,}  ({100*seed.mean():.3f}%)  deltaE={delta_e} "
              f"step={step_px}px iter={max_iter} (max reach ~{step_px*max_iter*pix:.1f} m)")

    # Per-component roof colour signature (median Lab of seed pixels)
    lbl, n_comp = sci_label(seed)
    if verbose:
        print(f"components = {n_comp}")
    # Build a signature image: each seed pixel carries its component's median Lab
    sig = np.zeros((H, W, 3), np.float32)
    comp_sig = {}
    for cid in range(1, n_comp + 1):
        m = (lbl == cid)
        if m.sum() < 5:
            comp_sig[cid] = None
            continue
        med = np.median(lab[m], axis=0)
        comp_sig[cid] = med

    grown = seed.copy()
    grown_lbl = lbl.copy()                        # track which comp owns a px
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * step_px + 1, 2 * step_px + 1))

    for it in range(max_iter):
        # candidate ring = dilate(grown) minus grown
        dil = cv2.dilate(grown.astype(np.uint8), se).astype(bool)
        ring = dil & (~grown)
        if not ring.any():
            break
        # for each ring pixel, find an adjacent grown pixel's component,
        # then test Lab distance against that component's signature
        # (vectorised: propagate owning component label by dilation)
        own = grey_dilation(grown_lbl, footprint=(se > 0))
        ry, rx = np.where(ring)
        cids = own[ry, rx]
        add = np.zeros_like(grown)
        # group by component for speed
        for cid in np.unique(cids):
            if cid == 0:
                continue
            s = comp_sig.get(int(cid))
            if s is None:
                continue
            sel = (cids == cid)
            yy = ry[sel]; xx = rx[sel]
            d = np.linalg.norm(lab[yy, xx] - s[None, :], axis=1)
            ok = d < delta_e
            add[yy[ok], xx[ok]] = True
            grown_lbl[yy[ok], xx[ok]] = cid
        if not add.any():
            break
        grown |= add
        if verbose:
            print(f"  iter {it+1}: +{int(add.sum()):,} px  total={int(grown.sum()):,}",
                  flush=True)

    n1 = int(grown.sum())
    if verbose:
        print(f"grown px = {n1:,}  (+{100*(n1-n0)/max(n0,1):.1f}% vs seed)")

    profile.update(count=1, dtype="uint8", compress="lzw", nodata=0)
    out = grown.astype(np.uint8)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out, 1)
    if verbose:
        print(f"saved raster: {out_tif}")

    # vectorise
    rows = []
    for geom, val in rio_shapes(out, mask=(out > 0), transform=tf):
        if val != 1:
            continue
        g = shapely_shape(geom)
        if g.area >= min_area_m2:
            rows.append({"geometry": g, "area_m2": round(g.area, 2)})
    if rows:
        gdf = gpd.GeoDataFrame(rows, crs=crs).sort_values(
            "area_m2", ascending=False).reset_index(drop=True)
        gdf.to_file(out_gpkg, driver="GPKG")
        print(f"saved vector: {out_gpkg}  ({len(gdf)} polygons, "
              f"{gdf['area_m2'].sum()/1e6:.3f} km^2)")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--input_rgb",
        default=r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\SG_google_map_4000_03m.tif")
    P.add_argument("--seed_tif",
        default=r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_autonomous_post.tif")
    P.add_argument("--out_tif",
        default=r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_autonomous_post_grow.tif")
    P.add_argument("--out_gpkg",
        default=r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_autonomous_post_grow.gpkg")
    P.add_argument("--delta_e", type=float, default=14.0)
    P.add_argument("--step_px", type=int, default=3)
    P.add_argument("--max_iter", type=int, default=10)
    P.add_argument("--min_area_m2", type=float, default=25.0)
    args = P.parse_args()
    grow(args.input_rgb, args.seed_tif, args.out_tif, args.out_gpkg,
         delta_e=args.delta_e, step_px=args.step_px, max_iter=args.max_iter,
         min_area_m2=args.min_area_m2)
