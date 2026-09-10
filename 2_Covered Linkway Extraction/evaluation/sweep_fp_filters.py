"""
Zero-cost FP-reduction sweep on the FINAL pednet product.

Levers (no retraining):
  - pednet_overlap_ratio threshold : 0.0 (current) / 0.2 / 0.3 / 0.5
  - shape aspect-ratio threshold   : 0 (off) / 2.0 / 3.0
        aspect = long/short side of each polygon's min rotated rectangle
        (linkway = elongated ribbon -> high aspect;
         carpark canopy / blob FP   -> low aspect)

For every (overlap x aspect) combo, rasterise the kept polygons inside each
of the 303 v3 val-tile windows and report micro P/R/F1/IoU + mean clDice +
mean Betti-0.  Goal: find a combo that raises F1 without dropping clDice
(i.e. removes FP without fragmenting the network).
"""
from __future__ import annotations
import os, glob, time, numpy as np, rasterio
from rasterio.windows import Window, transform as win_transform
from rasterio.features import rasterize
from PIL import Image
import geopandas as gpd
from shapely.geometry import box
from skimage.morphology import skeletonize
from scipy.ndimage import label as cc_label

VAL   = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks\val"
SRC   = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\SG_google_map_03m_SVY21.tif"
PEDN  = r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island_pednet.gpkg"
TILE  = 1024
OVL   = [0.0, 0.2, 0.3, 0.5]
ASP   = [0.0, 2.0, 3.0]


def offs(p):
    s = os.path.splitext(os.path.basename(p))[0]; _, x, y = s.split("_")
    return int(x[1:]), int(y[1:])


def aspect(geom):
    try:
        r = geom.minimum_rotated_rectangle
        xs, ys = r.exterior.coords.xy
        e = [((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2) ** 0.5 for i in range(4)]
        L = max(e[0], e[1]); S = max(min(e[0], e[1]), 1e-6)
        return L / S
    except Exception:
        return 1.0


def cldice(p, g, eps=1e-7):
    if p.sum() == 0 and g.sum() == 0: return 1.0
    if p.sum() == 0 or g.sum() == 0:  return 0.0
    sp, sg = skeletonize(p > 0), skeletonize(g > 0)
    tp = (sp & (g > 0)).sum() / (sp.sum() + eps)
    ts = (sg & (p > 0)).sum() / (sg.sum() + eps)
    return 0.0 if tp + ts < eps else float(2*tp*ts/(tp+ts+eps))


def main():
    t0 = time.time()
    masks = sorted(glob.glob(os.path.join(VAL, "*.png")))
    with rasterio.open(SRC) as s:
        base_tf = s.transform; crs = s.crs
    g = gpd.read_file(PEDN)
    if g.crs is None: g.set_crs(crs, inplace=True)
    g["aspect"] = g.geometry.apply(aspect)
    if "pednet_overlap_ratio" not in g.columns:
        g["pednet_overlap_ratio"] = 1.0
    sidx = g.sindex
    print(f"pednet polygons={len(g)}  val tiles={len(masks)}  "
          f"combos={len(OVL)*len(ASP)}")

    # Pre-extract per-tile GT + candidate polygon indices
    tiles = []
    for mp in masks:
        c, r = offs(mp)
        gt = (np.array(Image.open(mp).convert("L")) > 128).astype(np.uint8)
        wtf = win_transform(Window(c, r, TILE, TILE), base_tf)
        # tile bbox in CRS
        x0, y0 = wtf.c, wtf.f
        x1 = x0 + TILE * wtf.a
        y1 = y0 + TILE * wtf.e
        bb = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        cand = list(sidx.intersection(bb.bounds))
        tiles.append((gt, wtf, cand))

    print(f"\n{'ovl':>4}{'asp':>5}{'#poly':>8}{'P':>8}{'R':>8}{'F1':>8}"
          f"{'IoU':>8}{'clDice':>8}{'Betti0':>8}")
    print("-" * 64)

    best = None
    for ov in OVL:
        for asp in ASP:
            sub = g[(g["pednet_overlap_ratio"] >= ov) & (g["aspect"] >= asp)]
            keep_idx = set(sub.index)
            geoms_by_idx = {i: g.geometry.iloc[i] for i in keep_idx}
            TP = FP = FN = 0
            clds, betti = [], []
            for gt, wtf, cand in tiles:
                use = [geoms_by_idx[i] for i in cand if i in keep_idx]
                if use:
                    pr = rasterize([(gm, 1) for gm in use],
                                   out_shape=(TILE, TILE), transform=wtf,
                                   fill=0, dtype="uint8")
                else:
                    pr = np.zeros((TILE, TILE), np.uint8)
                th, tw = gt.shape
                pr = pr[:th, :tw]
                tp = int((pr & gt).sum()); fp = int((pr & (1-gt)).sum())
                fn = int(((1-pr) & gt).sum())
                TP += tp; FP += fp; FN += fn
                clds.append(cldice(pr, gt))
                betti.append(abs(cc_label(pr)[1] - cc_label(gt)[1]))
            P = TP/max(TP+FP, 1); R = TP/max(TP+FN, 1)
            F1 = 2*P*R/max(P+R, 1e-9); IoU = TP/max(TP+FP+FN, 1)
            cl = float(np.mean(clds)); bt = float(np.mean(betti))
            mark = ""
            if best is None or F1 > best[0]:
                best = (F1, ov, asp, P, R, IoU, cl, bt); mark = "  <-- best F1"
            print(f"{ov:>4.1f}{asp:>5.1f}{len(sub):>8}{P:>8.3f}{R:>8.3f}"
                  f"{F1:>8.3f}{IoU:>8.3f}{cl:>8.3f}{bt:>8.2f}{mark}")
    print()
    F1, ov, asp, P, R, IoU, cl, bt = best
    print(f"BEST: overlap>={ov} aspect>={asp}  "
          f"F1={F1:.4f} (was 0.5251)  P={P:.3f} R={R:.3f} "
          f"IoU={IoU:.3f} clDice={cl:.3f} (was 0.640) Betti0={bt:.2f} (was 2.78)")
    print(f"\ntotal {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
