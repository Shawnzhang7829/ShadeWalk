"""Turn the test outputs into single-band files that open directly in QGIS (embedded colour palette):
 Category_0_0_h09/h14/h18.tif (uint8 + colour table) + Shadow_0_0_h14.tif. Written back to testrun/output_folder/0_0."""
import numpy as np, rasterio
from pathlib import Path
O=Path(r"D:\Claude\SVI_FFW\output\step3_adsm\testrun\output_folder\0_0")
PAL={0:(255,255,204,255),1:(154,154,154,255),2:(116,196,118,255),3:(74,127,74,255),
     4:(253,174,107,255),5:(176,112,64,255),6:(143,174,90,255),7:(107,91,58,255),
     8:(212,0,0,255),9:(192,101,192,255),10:(224,128,32,255),11:(123,63,160,255),
     12:(90,90,90,255)}  # 12 = building footprint (building_remain), not shadow
with rasterio.open(O/"Category_0_0.tif") as r:
    meta=r.meta.copy()
    for hr in (9,14,18):
        arr=r.read(hr+1)
        m=meta.copy(); m.update(count=1,dtype="uint8",nodata=None)
        with rasterio.open(O/f"Category_0_0_h{hr:02d}.tif","w",**m) as o:
            o.write(arr,1); o.write_colormap(1,PAL)
        u,c=np.unique(arr,return_counts=True)
        print(f"h{hr:02d}: classes={dict(zip(u.tolist(),c.tolist()))}")
with rasterio.open(O/"Shadow_0_0.tif") as r:
    m=r.meta.copy(); m.update(count=1)
    arr=r.read(15)
    with rasterio.open(O/"Shadow_0_0_h14.tif","w",**m) as o: o.write(arr,1)
print("wrote Category_0_0_h09/h14/h18.tif (with colour palette, opens directly in QGIS) + Shadow_0_0_h14.tif")
