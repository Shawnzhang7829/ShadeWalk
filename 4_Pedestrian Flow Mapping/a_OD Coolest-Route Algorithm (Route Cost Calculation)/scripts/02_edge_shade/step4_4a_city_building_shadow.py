# -*- coding: utf-8 -*-
"""Step4-4a city-wide: LOD1-building-only 14:00 shadow, tiled numpy port (verified 0 violations);
each tile uses the sun position at the tile centre (13:30 half-hour step), 120 px margin captures shadows cast from outside the tile. Output uint8 (1=lit, 0=shaded). pyenv."""
import sys, numpy as np, rasterio, time
from rasterio.windows import Window, from_bounds
sys.path.insert(0, r'D:\Claude\SVI_FFW\Module\SOLWEIG-GPU')
from solweig_gpu.sun_position import sun_position
from pyproj import Transformer
TIF=r"D:\Claude\SVI_FFW\TIF"; DSMF=f"{TIF}\\SUB_SG_Polygon_DSMremain_1m.tif"
SHF=f"{TIF}\\merge_images\\Shadow_2pm_h14.tif"
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
OUTF=f"{OUT}\\SUB_SG_BUILDING_SHADOW_h14_1m.tif"
TILE=4000; MARGIN=120
to_ll=Transformer.from_crs(3414,4326,always_xy=True)

def shadow_13(a,az_deg,al_deg,scale=1.0):
    a=a.astype(np.float32); sx,sy=a.shape
    az=np.float64(az_deg)*np.pi/180; al=np.float64(al_deg)*np.pi/180
    f=a.copy(); temp=np.zeros_like(a)
    pq=np.pi/4; p3=3*pq; p5=5*pq; p7=7*pq
    sinaz,cosaz,tanaz=np.sin(az),np.cos(az),np.tan(az)
    ssin,scos=np.sign(sinaz),np.sign(cosaz); dssin,dscos=abs(1/sinaz),abs(1/cosaz)
    tanalt=np.tan(al)/scale; amax=float(a.max()); index=1; dz=dx=dy=0.0
    while (amax>=dz) and (abs(dx)<sx) and (abs(dy)<sy):
        if (pq<=az<p3) or (p5<=az<p7): dy=ssin*index; dx=-scos*abs(round(index/tanaz)); ds=dssin
        else: dy=ssin*abs(round(index*tanaz)); dx=-scos*index; ds=dscos
        dz=ds*index*tanalt; temp[:]=0.0
        absdx,absdy=abs(dx),abs(dy)
        xc1=int((dx+absdx)/2); xc2=int(sx+(dx-absdx)/2); yc1=int((dy+absdy)/2); yc2=int(sy+(dy-absdy)/2)
        xp1=int(-((dx-absdx)/2)); xp2=int(sx-(dx+absdx)/2); yp1=int(-((dy-absdy)/2)); yp2=int(sy-(dy+absdy)/2)
        temp[xp1:xp2,yp1:yp2]=a[xc1:xc2,yc1:yc2]-dz; f=np.maximum(f,temp); index+=1
    sh=(f-a); sh=(sh!=0).astype(np.uint8); sh=(1-sh).astype(np.uint8)   # 1=lit,0=shaded
    return sh

with rasterio.open(DSMF) as ds:
    W,H=ds.width,ds.height; T=ds.transform; crs=ds.crs
prof=dict(driver='GTiff',height=H,width=W,count=1,dtype='uint8',crs=crs,transform=T,
          compress='LZW',tiled=True,blockxsize=512,blockysize=512,BIGTIFF='YES',nodata=255)
t0=time.time(); ntile=0
viol_tot=0; bsh_tot=0
with rasterio.open(DSMF) as dsm_ds, rasterio.open(SHF) as full_ds, rasterio.open(OUTF,'w',**prof) as out:
    for r0 in range(0,H,TILE):
        for c0 in range(0,W,TILE):
            th=min(TILE,H-r0); tw=min(TILE,W-c0)
            rc0=max(0,r0-MARGIN); cc0=max(0,c0-MARGIN)
            rc1=min(H,r0+th+MARGIN); cc1=min(W,c0+tw+MARGIN)
            win=Window(cc0,rc0,cc1-cc0,rc1-rc0)
            dsm=dsm_ds.read(1,window=win).astype(np.float32)
            dsm=np.nan_to_num(dsm,nan=0.0,posinf=0.0,neginf=0.0); dsm[dsm<0]=0.0
            if dsm.max()<=0.01:   # no buildings -> all lit
                out.write(np.ones((th,tw),np.uint8),1,window=Window(c0,r0,tw,th)); ntile+=1; continue
            # sun position at tile centre (13:30 local UTC+8)
            cx=T.c+(c0+tw/2); cy=T.f-(r0+th/2)
            lon,lat=to_ll.transform(cx,cy)
            s=sun_position({'year':2026,'month':3,'day':1,'hour':13,'min':30,'sec':0,'UTC':8},
                           {'longitude':lon,'latitude':lat,'altitude':0.0})
            alt=90.0-float(np.asarray(s['zenith']).ravel()[0]); azi=float(np.asarray(s['azimuth']).ravel()[0])
            sh=shadow_13(dsm,azi,alt)
            # crop the margin
            roff=r0-rc0; coff=c0-cc0
            sh_c=sh[roff:roff+th,coff:coff+tw]
            out.write(sh_c,1,window=Window(c0,r0,tw,th))
            # sampled verification subset (every few tiles)
            if ntile%15==0:
                full=full_ds.read(1,window=Window(c0,r0,tw,th))
                bsh=sh_c<0.5; fsh=full<0.5; v=int((bsh&~fsh&np.isfinite(full)).sum())
                viol_tot+=v; bsh_tot+=int(bsh.sum())
            ntile+=1
        print(f"  row {r0}/{H} ({time.time()-t0:.0f}s, {ntile} tiles)",flush=True)
print(f"finished {ntile} tiles {time.time()-t0:.0f}s | sampled-subset violations {viol_tot}/{bsh_tot} ({100*viol_tot/max(bsh_tot,1):.3f}%)",flush=True)
print("saved",OUTF,flush=True)
