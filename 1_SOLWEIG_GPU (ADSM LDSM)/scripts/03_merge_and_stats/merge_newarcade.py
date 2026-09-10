# -*- coding: utf-8 -*-
"""Mosaic the core areas of band 15 (14:00) of the 77 new-arcade tiles into single-band citywide images, replicating merge_images:
  TIF_shadow_newarcade\merge_images\Shadow_2pm_h14.tif (float32, 0=shadow/1=sunlit)
  TIF_shadow_newarcade\merge_images\Category_2pm_h14.tif (uint8, 13 classes + colour palette)
Each tile contributes its 4000x4000 core (100 px halo removed to avoid seams), placed by geographic position. pyenv 3.11.9."""
import os
import rasterio
from rasterio.windows import Window
NEW = r"D:\Claude\SVI_FFW\TIF_shadow_newarcade"
OFD = os.path.join(NEW, "output_folder")
MERGE = os.path.join(NEW, "merge_images"); os.makedirs(MERGE, exist_ok=True)
CORE = 4000; BAND = 15  # 14:00 = hour14 -> band15
PAL = {0:(255,255,204,255),1:(154,154,154,255),2:(116,196,118,255),3:(74,127,74,255),
       4:(253,174,107,255),5:(176,112,64,255),6:(143,174,90,255),7:(107,91,58,255),
       8:(212,0,0,255),9:(192,101,192,255),10:(224,128,32,255),11:(123,63,160,255),12:(90,90,90,255)}
with rasterio.open(os.path.join(NEW, "SUB_SG_Polygon_DEM_1m.tif")) as ref:
    CITY_W, CITY_H = ref.width, ref.height; base_transform, base_crs = ref.transform, ref.crs
tiles = sorted([d for d in os.listdir(OFD) if os.path.isdir(os.path.join(OFD, d))],
               key=lambda k: tuple(map(int, k.split("_"))))
print(f"tiles {len(tiles)} | citywide {CITY_W}x{CITY_H}", flush=True)
def build(layer, dtype, predictor, palette=None):
    out = os.path.join(MERGE, f"{layer}_2pm_h14.tif")
    prof = dict(driver="GTiff", width=CITY_W, height=CITY_H, count=1, dtype=dtype,
                crs=base_crs, transform=base_transform, nodata=None, compress="LZW",
                predictor=predictor, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
    tag = None
    with rasterio.open(out, "w", **prof) as dst:
        for key in tiles:
            x, y = map(int, key.split("_")); cw = min(CORE, CITY_W - x); ch = min(CORE, CITY_H - y)
            with rasterio.open(os.path.join(OFD, key, f"{layer}_{key}.tif")) as t:
                arr = t.read(BAND, window=Window(0, 0, cw, ch))
                if tag is None: tag = t.tags(BAND).get("Time")
            dst.write(arr, 1, window=Window(x, y, cw, ch))
        if palette: dst.write_colormap(1, palette)
        if tag: dst.update_tags(1, Time=tag)
    print(f"[{layer}] band{BAND} ({tag}) -> {out} ({os.path.getsize(out)/1e6:.1f}MB)", flush=True)
build("Category", "uint8", 2, PAL)
build("Shadow", "float32", 3)
print("DONE -> " + MERGE, flush=True)
