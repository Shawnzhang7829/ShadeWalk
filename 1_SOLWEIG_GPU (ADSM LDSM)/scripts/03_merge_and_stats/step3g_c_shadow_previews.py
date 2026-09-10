# -*- coding: utf-8 -*-
"""testrun_b: extract h09/h14/h18 single-band previews from the 24-band Shadow_0_0.tif (float32, 1=sunlit, 0=shadow). pyenv."""
import rasterio
from pathlib import Path

O = Path(r"D:\Claude\SVI_FFW\output\step3_adsm\testrun_b\output_folder\0_0")
with rasterio.open(O/"Shadow_0_0.tif") as r:
    meta = r.meta.copy()
    for hr in (9, 14, 18):
        arr = r.read(hr+1)          # band = hour+1
        m = meta.copy(); m.update(count=1)
        with rasterio.open(O/f"Shadow_0_0_h{hr:02d}.tif", "w", **m) as o:
            o.write(arr, 1)
        print(f"Shadow_0_0_h{hr:02d}.tif  range {arr.min():.2f}-{arr.max():.2f}, fully shaded px {(arr<0.5).sum():,}")
print("done ->", O)
