# -*- coding: utf-8 -*-
"""Building + tree (no arcade / linkway facilities) 14:00 shadow = shade fallback layer for the topology-map "facilities removed" scenario.
NOTE: vegetation must not be merged into a single surface (max(DSM, DEM+CDSM) computes "canopy top lit", but pedestrians walk on the ground under the canopy!) --
mirrors UMEP shadowingfunction_20: building-surface shadow as usual, vegetation treated as a separate occluder
(shifted vegtop - dz > building surface => canopy blocks light; canopy treated as opaque, SOLWEIG transmissivity is only ~3%, negligible on a binary basis;
canopy-bottom transparency ignored (at the 13:30 solar altitude of 80 deg a ray passing the canopy almost always crosses the crown body)).
Tiled, tile-centre sun at 13:30 half-hour step, 120 px margin. Output uint8 1=lit/0=shaded. pyenv."""
import sys, numpy as np, rasterio, time
from rasterio.windows import Window
sys.path.insert(0, r'D:\Claude\SVI_FFW\Module\SOLWEIG-GPU')
from solweig_gpu.sun_position import sun_position
from pyproj import Transformer
NA=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade"
DSMF=f"{NA}\\SUB_SG_Polygon_DSMremain_1m.tif"
DEMF=f"{NA}\\SUB_SG_Polygon_DEM_1m.tif"
CDSF=f"{NA}\\SUB_SG_Polygon_CDSMclean_1m.tif"
SHF =f"{NA}\\merge_images\\Shadow_2pm_h14.tif"     # full-system shadow (spot-check reference: building+tree shade should be a subset of full-system shade)
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
OUTF=f"{OUT}\\SUB_SG_BLDTREE_SHADOW_h14_1m.tif"
TILE=4000; MARGIN=120
to_ll=Transformer.from_crs(3414,4326,always_xy=True)

def shadow_bldveg(a,vg,az_deg,al_deg,scale=1.0):
    """a = building surface (absolute elevation, no trees); vg = vegetation canopy top (absolute elevation, 0 where there is no tree = never occludes).
    Buildings: f = max(a_shift-dz) surface shadow; vegetation: vg_shift-dz > a means the canopy blocks light (including the ground under the canopy)."""
    a=a.astype(np.float32); vg=vg.astype(np.float32); sx,sy=a.shape
    az=np.float64(az_deg)*np.pi/180; al=np.float64(al_deg)*np.pi/180
    f=a.copy(); temp=np.zeros_like(a); tv=np.zeros_like(a); vsh=np.zeros(a.shape,bool)
    pq=np.pi/4; p3=3*pq; p5=5*pq; p7=7*pq
    sinaz,cosaz,tanaz=np.sin(az),np.cos(az),np.tan(az)
    ssin,scos=np.sign(sinaz),np.sign(cosaz); dssin,dscos=abs(1/sinaz),abs(1/cosaz)
    tanalt=np.tan(al)/scale; amax=float(max(a.max(),vg.max())); index=1; dz=dx=dy=0.0
    while (amax>=dz) and (abs(dx)<sx) and (abs(dy)<sy):
        if (pq<=az<p3) or (p5<=az<p7): dy=ssin*index; dx=-scos*abs(round(index/tanaz)); ds=dssin
        else: dy=ssin*abs(round(index*tanaz)); dx=-scos*index; ds=dscos
        dz=ds*index*tanalt; temp[:]=0.0; tv[:]=0.0
        absdx,absdy=abs(dx),abs(dy)
        xc1=int((dx+absdx)/2); xc2=int(sx+(dx-absdx)/2); yc1=int((dy+absdy)/2); yc2=int(sy+(dy-absdy)/2)
        xp1=int(-((dx-absdx)/2)); xp2=int(sx-(dx+absdx)/2); yp1=int(-((dy-absdy)/2)); yp2=int(sy-(dy+absdy)/2)
        temp[xp1:xp2,yp1:yp2]=a[xc1:xc2,yc1:yc2]-dz; f=np.maximum(f,temp)
        tv[xp1:xp2,yp1:yp2]=vg[xc1:xc2,yc1:yc2]-dz; vsh|=(tv>(a+0.01)); index+=1
    shaded=((f-a)!=0)|vsh
    return (~shaded).astype(np.uint8)   # 1=lit,0=shaded

with rasterio.open(DSMF) as ds:
    W,H=ds.width,ds.height; T=ds.transform; crs=ds.crs
prof=dict(driver='GTiff',height=H,width=W,count=1,dtype='uint8',crs=crs,transform=T,
          compress='LZW',tiled=True,blockxsize=512,blockysize=512,BIGTIFF='YES',nodata=255)
t0=time.time(); ntile=0; viol_tot=0; bsh_tot=0
with rasterio.open(DSMF) as dsm_ds, rasterio.open(DEMF) as dem_ds, rasterio.open(CDSF) as cd_ds, \
     rasterio.open(SHF) as full_ds, rasterio.open(OUTF,'w',**prof) as out:
    for r0 in range(0,H,TILE):
        for c0 in range(0,W,TILE):
            th=min(TILE,H-r0); tw=min(TILE,W-c0)
            rc0=max(0,r0-MARGIN); cc0=max(0,c0-MARGIN)
            rc1=min(H,r0+th+MARGIN); cc1=min(W,c0+tw+MARGIN)
            win=Window(cc0,rc0,cc1-cc0,rc1-rc0)
            dsm=np.nan_to_num(dsm_ds.read(1,window=win).astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
            dem=np.nan_to_num(dem_ds.read(1,window=win).astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
            cd =np.nan_to_num(cd_ds.read(1,window=win).astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
            cd[cd<0]=0.0; dsm[dsm<0]=0.0; dem[dem<0]=0.0
            a=dsm                              # building surface (absolute elevation, no trees / facilities)
            vg=np.where(cd>0.5,dem+cd,0.0).astype(np.float32)   # canopy top (absolute); no tree = 0, never occludes
            if max(a.max(),vg.max())<=0.01:    # pure sea tile -> all lit
                out.write(np.ones((th,tw),np.uint8),1,window=Window(c0,r0,tw,th)); ntile+=1; continue
            cx=T.c+(c0+tw/2); cy=T.f-(r0+th/2)
            lon,lat=to_ll.transform(cx,cy)
            s=sun_position({'year':2026,'month':3,'day':1,'hour':13,'min':30,'sec':0,'UTC':8},
                           {'longitude':lon,'latitude':lat,'altitude':0.0})
            alt=90.0-float(np.asarray(s['zenith']).ravel()[0]); azi=float(np.asarray(s['azimuth']).ravel()[0])
            sh=shadow_bldveg(a,vg,azi,alt)
            roff=r0-rc0; coff=c0-cc0
            sh_c=sh[roff:roff+th,coff:coff+tw]
            out.write(sh_c,1,window=Window(c0,r0,tw,th))
            if ntile%15==0:                   # spot check: building+tree shaded but full system lit = violation (should be ~0)
                full=full_ds.read(1,window=Window(c0,r0,tw,th))
                bsh=sh_c<0.5; fsh=full<0.5; v=int((bsh&~fsh&np.isfinite(full)).sum())
                viol_tot+=v; bsh_tot+=int(bsh.sum())
            ntile+=1
        print(f"  row {r0}/{H} ({time.time()-t0:.0f}s, {ntile} tiles)",flush=True)
print(f"finished {ntile} tiles {time.time()-t0:.0f}s | spot-check violations {viol_tot}/{bsh_tot} ({100*viol_tot/max(bsh_tot,1):.3f}%)",flush=True)
print("saved",OUTF,flush=True)
print("DONE",flush=True)
