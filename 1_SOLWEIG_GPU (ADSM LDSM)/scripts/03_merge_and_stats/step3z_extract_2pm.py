# -*- coding: utf-8 -*-
"""Extract band 15 (14:00, 2 pm) from the citywide 24-band mosaics as single-band full-resolution images.
Category embeds the 13-class colour palette (renders in colour directly in QGIS); Shadow is the numeric version (1=sunlit, 0=shadow).
Windowed copy, memory safe. pyenv."""
import rasterio
from rasterio.windows import Window

MERGE = r"D:\Claude\SVI_FFW\TIF\merge_images"
BAND = 15  # 14:00 = hour 14 -> band 15
PAL = {0:(255,255,204,255),1:(154,154,154,255),2:(116,196,118,255),3:(74,127,74,255),
       4:(253,174,107,255),5:(176,112,64,255),6:(143,174,90,255),7:(107,91,58,255),
       8:(212,0,0,255),9:(192,101,192,255),10:(224,128,32,255),11:(123,63,160,255),
       12:(90,90,90,255)}

def extract(layer, dtype, palette=None):
    src = rf"{MERGE}\{layer}\{layer}_merged.tif"
    out = rf"{MERGE}\{layer}_2pm_h14.tif"
    with rasterio.open(src) as r:
        prof = r.profile.copy()
        prof.update(count=1, dtype=dtype, compress="LZW",
                    predictor=(2 if dtype == "uint8" else 3),
                    tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
        tag = r.tags(BAND).get("Time")
        with rasterio.open(out, "w", **prof) as o:
            for _, win in r.block_windows(1):
                o.write(r.read(BAND, window=win), 1, window=win)
            if palette:
                o.write_colormap(1, palette)
            o.update_tags(1, Time=tag)
    import os
    print(f"[{layer}] band{BAND} ({tag}) -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

extract("Category", "uint8", PAL)
extract("Shadow", "float32")
print("DONE")
