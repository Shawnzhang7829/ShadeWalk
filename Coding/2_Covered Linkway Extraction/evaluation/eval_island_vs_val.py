"""
Evaluate the FINAL full-island products against the v3 validation set.

The validation tiles in
    D:\\Claude\\GeoSAM-TopoLoRA\\covered Linkway\\masks\\val\\tile_x{col}_y{row}.png
are 1024x1024 crops of SG_google_map_03m_SVY21.tif (179096 x 115025 @ 0.3 m).
The full-island prediction GeoTIFFs share that exact grid, so we extract the
window at (row, col) and compare to the GT mask.

Computes mean Dice / IoU over all 303 val tiles for each product.
"""
from __future__ import annotations
import os, glob
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

VAL_MASK_DIR = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks\val"
TILE = 1024

PRODUCTS = {
    "island RAW (v3, min_area25)":
        r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island.tif",
    "island + pednet 5m (FINAL/best)":
        r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island_pednet.tif",
}


def parse_offsets(name):
    # tile_x{col}_y{row}.png
    stem = os.path.splitext(name)[0]
    _, xs, ys = stem.split("_")
    return int(xs[1:]), int(ys[1:])   # col_off, row_off


def eval_product(tif_path, masks):
    n = 0
    dice_sum = iou_sum = 0.0
    # also tile-restricted: only tiles whose GT has any fg (the meaningful ones)
    n_fg = 0
    dice_fg = iou_fg = 0.0
    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        for mp in masks:
            name = os.path.basename(mp)
            col, row = parse_offsets(name)
            if row >= H or col >= W:
                continue
            g = np.array(Image.open(mp).convert("L"))
            g = (g > 128).astype(np.uint8)
            th, tw = g.shape
            win = Window(col, row, min(tw, W - col), min(th, H - row))
            p = src.read(1, window=win)
            if p.shape != g.shape:
                pp = np.zeros_like(g)
                pp[:p.shape[0], :p.shape[1]] = p[:g.shape[0], :g.shape[1]]
                p = pp
            p = (p > 0).astype(np.uint8)
            tp = int((p & g).sum())
            fp = int((p & (1 - g)).sum())
            fn = int(((1 - p) & g).sum())
            dice = 2*tp / max(2*tp + fp + fn, 1)
            iou  = tp / max(tp + fp + fn, 1)
            dice_sum += dice; iou_sum += iou; n += 1
            if g.sum() > 0:
                dice_fg += dice; iou_fg += iou; n_fg += 1
    return dict(n=n, dice=dice_sum/max(n,1), iou=iou_sum/max(n,1),
                n_fg=n_fg, dice_fg=dice_fg/max(n_fg,1), iou_fg=iou_fg/max(n_fg,1))


if __name__ == "__main__":
    masks = sorted(glob.glob(os.path.join(VAL_MASK_DIR, "*.png")))
    print(f"val tiles: {len(masks)}")
    print()
    print(f"{'product':<38}{'n':>5}{'Dice':>9}{'IoU':>9}"
          f"{'Dice(fg)':>10}{'IoU(fg)':>9}")
    print("-" * 80)
    for label, path in PRODUCTS.items():
        if not os.path.exists(path):
            print(f"{label:<38}  MISSING: {path}")
            continue
        r = eval_product(path, masks)
        print(f"{label:<38}{r['n']:>5}{r['dice']:>9.4f}{r['iou']:>9.4f}"
              f"{r['dice_fg']:>10.4f}{r['iou_fg']:>9.4f}")
    print()
    print("Note: Dice/IoU = mean over all 303 val tiles.")
    print("      Dice(fg)/IoU(fg) = mean over only tiles whose GT has linkway")
    print("      (excludes empty-GT tiles where any FP tanks the score).")
