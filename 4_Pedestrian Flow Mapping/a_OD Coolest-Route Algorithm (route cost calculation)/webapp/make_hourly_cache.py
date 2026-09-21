# -*- coding: utf-8 -*-
"""Generate the hourly @10 m shadow / category cache in parallel -> _hourly_layers.json.
On an NVMe SSD multiple processes reading the same large raster do not contend; each process reads one band (peak ~5 GB), NPROC is bounded by memory (64 GB total / ~28 GB free -> 6 processes ~30 GB).
After each band completes the main process immediately writes the progress increment to _hourly_prog.json (typ+hour -> base64), so a rerun after being killed / crashing resumes exactly without redoing work.
Compatible with the old _hourly_shad.json (list of 5 frames). pyenv, exclusive."""
import rasterio, numpy as np, io, base64
from rasterio.enums import Resampling
from multiprocessing import Pool
SHADOW=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\merge_images\Shadow\Shadow_merged.tif"     # new arcade, 24 bands
CATEG =r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\merge_images\Category\Category_merged.tif" # new arcade, 24 bands
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"; HOURS=list(range(8,19)); DEC=10; NPROC=6
CATCOL={1:(120,120,132),2:(46,139,61),3:(95,150,82),4:(120,170,210),5:(110,140,165),6:(80,160,140),
        7:(95,125,115),8:(212,50,44),9:(225,115,60),10:(150,95,165),11:(185,85,120),12:(70,70,78)}
# class-priority downsampling (fixes nearest dropping the 1-3 px wide arcade/linkway thin classes): take the highest-priority class within each cell
# arcade 8-11 > linkway 4-7 > building footprint 12 > building+tree 3 > tree 2 > building 1 > sun 0
_PRIORDER=[8,9,10,11,4,5,6,7,12,3,2,1,0]
CATPRI=np.zeros(256,np.int16)
for _r,_c in enumerate(_PRIORDER): CATPRI[_c]=len(_PRIORDER)-_r
def png_uri(rgba):
    from PIL import Image; bb=io.BytesIO(); Image.fromarray(rgba,'RGBA').save(bb,'PNG')
    return 'data:image/png;base64,'+base64.b64encode(bb.getvalue()).decode()
def work(arg):
    typ,h=arg
    if typ=='s':
        with rasterio.open(SHADOW) as ds:
            sw,sh=ds.width//DEC,ds.height//DEC
            a=np.nan_to_num(ds.read(h+1,out_shape=(sh,sw),resampling=Resampling.nearest),nan=1.0)
        rg=np.zeros((sh,sw,4),np.uint8); rg[a<0.5]=[44,127,184,140]; return (typ,h,png_uri(rg))
    else:
        with rasterio.open(CATEG) as ds:
            cw,ch=ds.width//DEC,ds.height//DEC
            full=ds.read(h+1)[:ch*DEC,:cw*DEC]              # full 1 m band, cropped to a multiple of DEC
        bp=np.zeros((ch,cw),np.int16); bc=np.zeros((ch,cw),np.uint8)   # priority reduction: highest-priority class per 10x10 cell
        for i in range(DEC):
            for j in range(DEC):
                sub=full[i::DEC,j::DEC][:ch,:cw]; p=CATPRI[sub]; m=p>bp; bc[m]=sub[m]; bp[m]=p[m]
        del full; c=bc.astype(np.int16)
        rg=np.zeros((ch,cw,4),np.uint8)
        for cls,(r_,g_,bb_) in CATCOL.items(): rg[c==cls]=[r_,g_,bb_,160]
        return (typ,h,png_uri(rg))
if __name__=='__main__':
    import json,time,os
    t0=time.time(); PROG=f"{OUT}\\_hourly_prog.json"
    with rasterio.open(SHADOW) as ds: b=ds.bounds; extS=[b.left,b.bottom,b.right,b.top]
    with rasterio.open(CATEG) as ds: b=ds.bounds; extC=[b.left,b.bottom,b.right,b.top]
    prog={}
    if os.path.exists(PROG):
        prog=json.load(open(PROG)); print(f"resuming: existing progress {len(prog)}/22 | {time.time()-t0:.0f}s",flush=True)
    elif os.path.exists(f"{OUT}\\_hourly_shad.json"):
        for i,u in enumerate(json.load(open(f"{OUT}\\_hourly_shad.json"))['shad']): prog['s%d'%HOURS[i]]=u
        print(f"resuming from the old shad shard: existing {len(prog)}/22 | {time.time()-t0:.0f}s",flush=True)
    todo=[('s',h) for h in HOURS if ('s%d'%h) not in prog]+[('c',h) for h in HOURS if ('c%d'%h) not in prog]
    print(f"parallel {NPROC} processes, {len(todo)} bands to read | {time.time()-t0:.0f}s",flush=True)
    if todo:
        with Pool(NPROC) as p:
            for typ,h,uri in p.imap_unordered(work, todo):
                prog['%s%d'%(typ,h)]=uri; json.dump(prog,open(PROG,'w'))
                print(f"  {typ} h{h} done ({len(prog)}/22) | {time.time()-t0:.0f}s",flush=True)
    shad=[prog['s%d'%h] for h in HOURS]; cat=[prog['c%d'%h] for h in HOURS]
    json.dump(dict(hours=HOURS,shad=shad,cat=cat,extS_abs=extS,extC_abs=extC),open(f"{OUT}\\_hourly_layers.json","w"))
    print(f"cached _hourly_layers.json ({len(HOURS)} hours @10m, shad{len(shad)}/cat{len(cat)}) | {time.time()-t0:.0f}s",flush=True)
