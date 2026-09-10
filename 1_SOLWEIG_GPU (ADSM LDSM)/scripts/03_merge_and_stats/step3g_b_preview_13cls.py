# -*- coding: utf-8 -*-
"""Verification material for the testrun_b 13-class version:
(1) Category_0_0_h09/h14/h18.tif (uint8 + embedded colour palette, opens directly in QGIS, 12 = dark grey)
(2) hourly 13-class count table for the whole day -> step3_class_counts_13cls.csv + console print
"""
import numpy as np, rasterio, csv
from pathlib import Path

O = Path(r"D:\Claude\SVI_FFW\output\step3_adsm\testrun_b\output_folder\0_0")
OUT_CSV = Path(r"D:\Claude\SVI_FFW\output\step3_adsm\step3_class_counts_13cls.csv")
PAL={0:(255,255,204,255),1:(154,154,154,255),2:(116,196,118,255),3:(74,127,74,255),
     4:(253,174,107,255),5:(176,112,64,255),6:(143,174,90,255),7:(107,91,58,255),
     8:(212,0,0,255),9:(192,101,192,255),10:(224,128,32,255),11:(123,63,160,255),
     12:(90,90,90,255)}
NAMES={0:"sunlit",1:"building",2:"vegetation",3:"bldg+veg",4:"ldsm",5:"bldg+ldsm",
       6:"veg+ldsm",7:"bldg+veg+ldsm",8:"arcade(incl bldg overlap)",9:"veg+arcade",
       10:"ldsm+arcade",11:"veg+ldsm+arcade",12:"building footprint(not shadow)"}

with rasterio.open(O/"Category_0_0.tif") as r:
    meta=r.meta.copy()
    # (1) Previews
    for hr in (9,14,18):
        arr=r.read(hr+1)
        m=meta.copy(); m.update(count=1,dtype="uint8",nodata=None)
        with rasterio.open(O/f"Category_0_0_h{hr:02d}.tif","w",**m) as o:
            o.write(arr,1); o.write_colormap(1,PAL)
    print("previews written: Category_0_0_h09/h14/h18.tif (testrun_b\\output_folder\\0_0, opens directly in QGIS)")
    # (2) Hourly count table
    rows=[]
    for b in range(1, r.count+1):
        arr=r.read(b)
        u,c=np.unique(arr,return_counts=True); d=dict(zip(u.tolist(),c.tolist()))
        rows.append([f"{b-1:02d}:00"]+[d.get(k,0) for k in range(13)])
    with open(OUT_CSV,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["hour"]+[f"{k}:{NAMES[k]}" for k in range(13)]); w.writerows(rows)
    print(f"count table written: {OUT_CSV}")
    # Console summary (key daytime hours)
    hdr=["hour"]+[str(k) for k in range(13)]
    print("  ".join(f"{h:>9}" for h in ["hour","0 sunlit","1 bldg","2 veg","8 arcade","12 fpt"]))
    for row in rows:
        h=int(row[0][:2])
        if 7<=h<=19:
            print(f"{row[0]:>9}  {row[1]:>9,}  {row[2]:>9,}  {row[3]:>9,}  {row[9]:>9,}  {row[13]:>9,}")
