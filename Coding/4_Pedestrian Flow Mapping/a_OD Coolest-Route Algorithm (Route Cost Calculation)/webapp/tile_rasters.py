# -*- coding: utf-8 -*-
"""1 m source rasters -> 3857 XYZ tile pyramid (z11-18; z18 ~ 0.6 m renders the 1 m source losslessly).
Continuous values are coloured: canopy-height gradient / sunlit-fraction gradient (darker = more shade) / category palette.
The base level z18 reprojects the source VALUES tile by tile and then colours them; the pyramid is downsampled from the base tiles (continuous = BILINEAR, categorical = NEAREST). pyenv."""
import rasterio, numpy as np, os, time, shutil, sys
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
from affine import Affine
from PIL import Image
TIF=r"D:\Claude\SVI_FFW\TIF"; TILES=r"D:\Claude\SVI_FFW\output\step4_network\webapp\tiles"
MERC=20037508.342789244; ZMIN=11; ZMAX=18
def tb(z,x,y):
    n=2**z; s=2*MERC/n; L=-MERC+x*s; T=MERC-y*s; return (L,T-s,L+s,T)
def trange(z,b):
    n=2**z; s=2*MERC/n
    return (int((b[0]+MERC)//s),int((b[2]+MERC)//s),int((MERC-b[3])//s),int((MERC-b[1])//s))
def ramp(t,lo,hi):
    return (np.array(lo,float)+(np.array(hi,float)-np.array(lo,float))*t[...,None]).astype(np.uint8)
def col_tree(a):       # canopy height (m) -> green from light to dark; <1 m transparent
    r=np.zeros(a.shape+(4,),np.uint8); t=np.clip((a-1.0)/24.0,0,1)
    r[...,:3]=ramp(t,[199,233,192],[0,68,27]); r[...,3]=np.where(a>1.0,200,0).astype(np.uint8); return r
def col_shadow(a):     # sunlit fraction s -> shade intensity t=1-s; solid and sharp: fixed high alpha (no semi-transparent gradient), t<0.05 transparent
    a=np.nan_to_num(a,nan=1.0); t=np.clip(1.0-a,0,1); r=np.zeros(a.shape+(4,),np.uint8)
    r[...,:3]=ramp(t,[158,202,235],[8,48,107]); r[...,3]=np.where(t>=0.05,205,0).astype(np.uint8); return r
CATCOL={1:(120,120,132),2:(46,139,61),3:(95,150,82),4:(120,170,210),5:(110,140,165),6:(80,160,140),7:(95,125,115),8:(212,50,44),9:(225,115,60),10:(150,95,165),11:(185,85,120),12:(70,70,78)}
def col_cat(a):
    a=np.nan_to_num(a,nan=0).astype(int); r=np.zeros(a.shape+(4,),np.uint8)
    for c,(R,G,B) in CATCOL.items(): r[a==c]=[R,G,B,190]
    return r
# (name, path, band, colorize, base_resampling, fill, pyramid_PIL_filter)
LAYERS=[("tree",f"{TIF}\\SUB_SG_Polygon_CDSMclean_1m.tif",1,col_tree,Resampling.bilinear,0.0,Image.BILINEAR),
        ("shadow14",f"{TIF}\\merge_images\\Shadow\\Shadow_merged.tif",15,col_shadow,Resampling.nearest,1.0,Image.NEAREST),
        ("cat14",f"{TIF}\\merge_images\\Category\\Category_merged.tif",15,col_cat,Resampling.nearest,0.0,Image.NEAREST)]
if 'fresh' in sys.argv and os.path.isdir(TILES): shutil.rmtree(TILES,ignore_errors=True)  # resumes by default (skips existing tiles); only the fresh argument clears and rebuilds
for name,path,band,colf,rs,fill,pf in LAYERS:
    if os.path.exists(f"{TILES}\\{name}\\.done"): print(f"[{name}] already done, skipping",flush=True); continue
    t0=time.time(); src=rasterio.open(path)
    b=transform_bounds(src.crs,'EPSG:3857',*src.bounds); sb=rasterio.band(src,band)
    # base level z18: reproject the source VALUES to 256x256 tile by tile, then colour (true 1 m, no resampling in colour space)
    x0,x1,yy0,yy1=trange(ZMAX,b); nb=0; ncol=x1-x0
    for x in range(x0,x1+1):
        for y in range(yy0,yy1+1):
            op=f"{TILES}\\{name}\\{ZMAX}\\{x}\\{y}.png"
            if os.path.exists(op): nb+=1; continue  # resume: skip if it already exists
            w,s,e,n=tb(ZMAX,x,y)
            dst=np.full((256,256),fill,np.float32)
            daff=Affine((e-w)/256.0,0,w,0,-(n-s)/256.0,n)
            try: reproject(source=sb,destination=dst,dst_transform=daff,dst_crs='EPSG:3857',resampling=rs,num_threads=2)
            except Exception: continue
            rgba=colf(dst)
            if rgba[...,3].max()==0: continue
            os.makedirs(f"{TILES}\\{name}\\{ZMAX}\\{x}",exist_ok=True)
            Image.fromarray(rgba,'RGBA').save(op); nb+=1
        if (x-x0)%30==0: print(f"  {name} z{ZMAX} col {x-x0}/{ncol} tiles {nb} ({time.time()-t0:.0f}s)",flush=True)
    print(f"  {name} z{ZMAX}: {nb} tiles ({time.time()-t0:.0f}s)",flush=True)
    # pyramid: downsample from the level above (continuous layers BILINEAR / categorical layers NEAREST, avoiding colour bleeding between classes)
    for z in range(ZMAX-1,ZMIN-1,-1):
        x0,x1,yy0,yy1=trange(z,b); m=0
        for x in range(x0,x1+1):
            for y in range(yy0,yy1+1):
                op=f"{TILES}\\{name}\\{z}\\{x}\\{y}.png"
                if os.path.exists(op): m+=1; continue  # resume: skip if it already exists
                im=Image.new('RGBA',(256,256),(0,0,0,0)); has=False
                for dx in (0,1):
                    for dy in (0,1):
                        cp=f"{TILES}\\{name}\\{z+1}\\{2*x+dx}\\{2*y+dy}.png"
                        if os.path.exists(cp):
                            ch=Image.open(cp).resize((128,128),pf); im.paste(ch,(dx*128,dy*128)); has=True
                if has:
                    os.makedirs(f"{TILES}\\{name}\\{z}\\{x}",exist_ok=True); im.save(op); m+=1
        print(f"  {name} z{z}: {m} tiles ({time.time()-t0:.0f}s)",flush=True)
    src.close(); open(f"{TILES}\\{name}\\.done",'w').close(); print(f"[{name}] done {time.time()-t0:.0f}s",flush=True)
print("ALL TILES DONE",flush=True)
