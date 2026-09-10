"""
Paper-grade evaluation on the 303 v3 val tiles.

For each product (island RAW vs island+pednet) reports:
  - pixel  : micro Precision / Recall / F1 / IoU
  - clDice : centerline-Dice (topology / connectivity)
  - relaxF1: boundary-relaxed F1 at tolerance 1 / 2 / 3 px
  - Betti0 : mean |#CC(pred) - #CC(GT)|  (fragmentation error)

Val tiles are 1024 crops of SG_google_map_03m_SVY21.tif; the full-island
prediction GeoTIFFs share that grid, windowed at (col,row) from the name.
"""
from __future__ import annotations
import os, glob, numpy as np, rasterio
from rasterio.windows import Window
from PIL import Image
from skimage.morphology import skeletonize, binary_dilation, disk
from scipy.ndimage import label as cc_label

VAL = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks\val"
PROD = {
    "island RAW (v3, min_area25)":
        r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island.tif",
    "island + pednet 5m (FINAL)":
        r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island_pednet.tif",
}


def offs(p):
    s = os.path.splitext(os.path.basename(p))[0]
    _, x, y = s.split("_")
    return int(x[1:]), int(y[1:])


def cldice(pred, gt, eps=1e-7):
    if pred.sum() == 0 or gt.sum() == 0:
        return 1.0 if (pred.sum() == 0 and gt.sum() == 0) else 0.0
    sp = skeletonize(pred > 0)
    sg = skeletonize(gt > 0)
    tprec = (sp & (gt > 0)).sum() / (sp.sum() + eps)
    tsens = (sg & (pred > 0)).sum() / (sg.sum() + eps)
    if tprec + tsens < eps:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens + eps))


def relaxed_f1(pred, gt, tol):
    if tol > 0:
        se = disk(tol)
        gd = binary_dilation(gt > 0, se)
        pd = binary_dilation(pred > 0, se)
    else:
        gd = gt > 0
        pd = pred > 0
    p = pred > 0
    g = gt > 0
    tp_p = (p & gd).sum()
    prec = tp_p / max(p.sum(), 1)
    tp_r = (g & pd).sum()
    rec = tp_r / max(g.sum(), 1)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


if __name__ == "__main__":
    masks = sorted(glob.glob(os.path.join(VAL, "*.png")))
    print(f"val tiles: {len(masks)}\n")
    for label, tif in PROD.items():
        TP = FP = FN = 0
        cld = []
        rf = {1: [], 2: [], 3: []}
        betti = []
        with rasterio.open(tif) as src:
            W, H = src.width, src.height
            for mp in masks:
                c, r = offs(mp)
                if r >= H or c >= W:
                    continue
                g = (np.array(Image.open(mp).convert("L")) > 128).astype(np.uint8)
                th, tw = g.shape
                p = src.read(1, window=Window(c, r, min(tw, W - c), min(th, H - r)))
                if p.shape != g.shape:
                    q = np.zeros_like(g); q[:p.shape[0], :p.shape[1]] = p[:g.shape[0], :g.shape[1]]; p = q
                p = (p > 0).astype(np.uint8)
                tp = int((p & g).sum()); fp = int((p & (1 - g)).sum()); fn = int(((1 - p) & g).sum())
                TP += tp; FP += fp; FN += fn
                cld.append(cldice(p, g))
                for t in (1, 2, 3):
                    rf[t].append(relaxed_f1(p, g, t))
                ng = cc_label(g)[1]; npd = cc_label(p)[1]
                betti.append(abs(npd - ng))
        prec = TP / max(TP + FP, 1)
        rec = TP / max(TP + FN, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        iou = TP / max(TP + FP + FN, 1)
        print(f"==== {label} ====")
        print(f"  pixel    P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  IoU={iou:.4f}")
        print(f"  clDice                              {np.mean(cld):.4f}")
        print(f"  relaxF1  tol1={np.mean(rf[1]):.4f}  tol2={np.mean(rf[2]):.4f}  tol3={np.mean(rf[3]):.4f}")
        print(f"  Betti0   mean|#CC_pred-#CC_gt| = {np.mean(betti):.2f}")
        print()
