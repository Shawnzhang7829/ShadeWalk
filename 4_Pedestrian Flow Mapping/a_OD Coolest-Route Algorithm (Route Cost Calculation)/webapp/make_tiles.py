# -*- coding: utf-8 -*-
"""Block C: new source (TIF_shadow_newarcade) -> 3857 XYZ tile pyramid (z11-18) under step5 webapp\\tiles.
3 layers: tree (new CDSM) / shadow14 (new Shadow_merged band15) / cat14 (new Category_merged band15).
Resumes by default (skips existing tiles); add the fresh argument to clear and rebuild. pyenv."""
import rasterio, numpy as np, os, time, shutil, sys
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
from affine import Affine
from PIL import Image
NEW=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade"; TILES=r"D:\Claude\SVI_FFW\output\step5_nav_webapp\webapp\tiles"
MERC=20037508.342789244; ZMIN=11; ZMAX=18
def tb(z,x,y):
    n=2**z; s=2*MERC/n; L=-MERC+x*s; T=MERC-y*s; return (L,T-s,L+s,T)
def trange(z,b):
    n=2**z; s=2*MERC/n
    return (int((b[0]+MERC)//s),int((b[2]+MERC)//s),int((MERC-b[3])//s),int((MERC-b[1])//s))
def ramp(t,lo,hi):
    return (np.array(lo,float)+(np.array(hi,float)-np.array(lo,float))*t[...,None]).astype(np.uint8)
def col_tree(a):
    r=np.zeros(a.shape+(4,),np.uint8); t=np.clip((a-1.0)/24.0,0,1)
    r[...,:3]=ramp(t,[199,233,192],[0,68,27]); r[...,3]=np.where(a>1.0,200,0).astype(np.uint8); return r
def col_shadow(a):
    a=np.nan_to_num(a,nan=1.0); t=np.clip(1.0-a,0,1); r=np.zeros(a.shape+(4,),np.uint8)
    r[...,:3]=ramp(t,[158,202,235],[8,48,107]); r[...,3]=np.where(t>=0.05,205,0).astype(np.uint8); return r
CATCOL={1:(120,120,132),2:(46,139,61),3:(95,150,82),4:(120,170,210),5:(110,140,165),6:(80,160,140),7:(95,125,115),8:(212,50,44),9:(225,115,60),10:(150,95,165),11:(185,85,120),12:(70,70,78)}
def col_cat(a):
    a=np.nan_to_num(a,nan=0).astype(int); r=np.zeros(a.shape+(4,),np.uint8)
    for c,(R,G,B) in CATCOL.items(): r[a==c]=[R,G,B,190]
    return r
LAYERS=[("tree",f"{NEW}\\SUB_SG_Polygon_CDSMclean_1m.tif",1,col_tree,Resampling.bilinear,0.0,Image.BILINEAR),
        ("shadow14",f"{NEW}\\merge_images\\Shadow\\Shadow_merged.tif",15,col_shadow,Resampling.nearest,1.0,Image.NEAREST),
        ("cat14",f"{NEW}\\merge_images\\Category\\Category_merged.tif",15,col_cat,Resampling.nearest,0.0,Image.NEAREST)]
if 'fresh' in sys.argv and os.path.isdir(TILES): shutil.rmtree(TILES,ignore_errors=True)
for name,path,band,colf,rs,fill,pf in LAYERS:
    if os.path.exists(f"{TILES}\\{name}\\.done"): print(f"[{name}] already done, skipping",flush=True); continue
    t0=time.time(); src=rasterio.open(path)
    b=transform_bounds(src.crs,'EPSG:3857',*src.bounds); sb=rasterio.band(src,band)
    x0,x1,yy0,yy1=trange(ZMAX,b); nb=0; ncol=x1-x0
    for x in range(x0,x1+1):
        for y in range(yy0,yy1+1):
            op=f"{TILES}\\{name}\\{ZMAX}\\{x}\\{y}.png"
            if os.path.exists(op): nb+=1; continue
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
    for z in range(ZMAX-1,ZMIN-1,-1):
        x0,x1,yy0,yy1=trange(z,b); m=0
        for x in range(x0,x1+1):
            for y in range(yy0,yy1+1):
                op=f"{TILES}\\{name}\\{z}\\{x}\\{y}.png"
                if os.path.exists(op): m+=1; continue
                im=Image.new('RGBA',(256,256),(0,0,0,0)); has=False
                for dx in (0,1):
                    for dy in (0,1):
                        cp=f"{TILES}\\{name}\\{z+1}\\{2*x+dx}\\{2*y+dy}.png"
                        if os.path.exists(cp):
                            try: ch=Image.open(cp).resize((128,128),pf); im.paste(ch,(dx*128,dy*128)); has=True
                            except Exception: os.remove(cp)   # corrupted tile (half-written when stopped) -> delete and skip
                if has:
                    os.makedirs(f"{TILES}\\{name}\\{z}\\{x}",exist_ok=True); im.save(op); m+=1
        print(f"  {name} z{z}: {m} tiles ({time.time()-t0:.0f}s)",flush=True)
    src.close(); open(f"{TILES}\\{name}\\.done",'w').close(); print(f"[{name}] done {time.time()-t0:.0f}s",flush=True)
print("ALL TILES DONE",flush=True)
