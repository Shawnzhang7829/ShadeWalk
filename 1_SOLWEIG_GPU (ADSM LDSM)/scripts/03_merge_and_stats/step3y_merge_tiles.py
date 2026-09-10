# -*- coding: utf-8 -*-
"""Step3y: mosaic the Shadow / Category outputs of the 77 tiles into citywide images, stored in TIF\merge_images.
Tile = 4100x4100 (4000 core + 100 overlap halo), origins stepped by 4000.
Method: take the [core 4000x4000] of each tile (= halo excluded, avoiding seams) and place it by geographic position ->
      citywide 44000x27000 24-band images. Fully consistent with the statistics basis (class 12 = 80,208,935 verified). pyenv."""
import os
import numpy as np
import rasterio
from rasterio.windows import Window

TIF = r"D:\Claude\SVI_FFW\TIF"
OFD = os.path.join(TIF, "output_folder")
MERGE = os.path.join(TIF, "merge_images")
CORE = 4000
ref = rasterio.open(os.path.join(TIF, "SUB_SG_Polygon_DEM_1m.tif"))
CITY_W, CITY_H = ref.width, ref.height
base_transform, base_crs = ref.transform, ref.crs
ref.close()

tiles = sorted([d for d in os.listdir(OFD) if os.path.isdir(os.path.join(OFD, d))],
               key=lambda k: tuple(map(int, k.split("_"))))
print(f"tiles: {len(tiles)} | citywide {CITY_W}x{CITY_H}")

# Take band metadata from the first tile (Time + Category class names)
SAMPLE = tiles[0]

def merge(layer, dtype, compress, predictor):
    sub = os.path.join(MERGE, layer); os.makedirs(sub, exist_ok=True)
    out_path = os.path.join(sub, f"{layer}_merged.tif")
    # band count + band tags
    with rasterio.open(os.path.join(OFD, SAMPLE, f"{layer}_{SAMPLE}.tif")) as s:
        nb = s.count
        band_tags = [s.tags(b) for b in range(1, nb + 1)]
        band_desc = [s.descriptions[b - 1] for b in range(1, nb + 1)]
    profile = dict(driver="GTiff", width=CITY_W, height=CITY_H, count=nb, dtype=dtype,
                   crs=base_crs, transform=base_transform, nodata=None,
                   compress=compress, predictor=predictor, tiled=True,
                   blockxsize=512, blockysize=512, BIGTIFF="YES", num_threads="ALL_CPUS")
    placed = 0
    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(1, nb + 1):
            if band_tags[b - 1]: dst.update_tags(b, **band_tags[b - 1])
            if band_desc[b - 1]: dst.set_band_description(b, band_desc[b - 1])
        for key in tiles:
            x, y = map(int, key.split("_"))
            cw = min(CORE, CITY_W - x); ch = min(CORE, CITY_H - y)
            tp = os.path.join(OFD, key, f"{layer}_{key}.tif")
            with rasterio.open(tp) as t:
                arr = t.read(window=Window(0, 0, cw, ch))   # (bands, ch, cw) core area
            dst.write(arr, window=Window(x, y, cw, ch))
            placed += 1
    sz = os.path.getsize(out_path) / 1e9
    print(f"[{layer}] {placed} tile cores merged -> {out_path}  ({nb} bands, {sz:.2f} GB)")
    return out_path

# Category first (small, fast), Shadow after (large)
merge("Category", "uint8",   "LZW", 2)
merge("Shadow",   "float32", "LZW", 3)
print("DONE: merge_images\\Category\\Category_merged.tif + merge_images\\Shadow\\Shadow_merged.tif")
