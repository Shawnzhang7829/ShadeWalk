"""
Footpath-constrained closing — PROVE it doesn't harm F1 on the 303 val tiles
before deploying to the full island.

bridge = ( binary_closing(M, disk(G)) AND footpath_corridor ) AND NOT M
M_new  = M OR bridge

Only fills gaps that (a) the morphological closing would bridge AND
(b) lie inside the thin pedestrian-network corridor. This eliminates the
lateral-blob redundancy of plain closing (corridor caps the width) and
only connects linkway across genuine path gaps.

Reports micro P/R/F1/IoU + mean clDice + Betti-0  BEFORE vs AFTER, on the
final Tversky deliverable, windowed at the 303 val-tile locations.
"""
from __future__ import annotations
import os, glob, time, numpy as np, rasterio
from rasterio.windows import Window, transform as win_tf
from rasterio.features import rasterize
from PIL import Image
import geopandas as gpd
from shapely.geometry import box
from skimage.morphology import skeletonize, binary_closing, disk
from scipy.ndimage import label as cc

VAL   = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks\val"
SRC   = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\SG_google_map_03m_SVY21.tif"
PRED  = r"C:\GeoSAM-backup\runs\autonomous_tversky\covered_linkway_SG_island_tv_pednet.tif"
PEDN  = r"D:\Claude\GeoSAM-TopoLoRA\shp\footpath\pedestrian_network_filtered.gpkg"
TILE  = 1024
G_PX        = 30      # closing radius px (~9 m -> bridges gaps up to ~18 m)
CORRIDOR_M  = 2.5     # footpath half-width buffer (m) -> ~5 m wide corridor


def offs(p):
    s = os.path.splitext(os.path.basename(p))[0]; _, x, y = s.split("_")
    return int(x[1:]), int(y[1:])


def cld(p, g, e=1e-7):
    if p.sum()==0 and g.sum()==0: return 1.0
    if p.sum()==0 or g.sum()==0:  return 0.0
    sp,sg=skeletonize(p>0),skeletonize(g>0)
    a=(sp&(g>0)).sum()/(sp.sum()+e); b=(sg&(p>0)).sum()/(sg.sum()+e)
    return 0.0 if a+b<e else float(2*a*b/(a+b+e))


def metrics(getpred, masks, src):
    TP=FP=FN=0; C=[]; B=[]
    W,H = src.width, src.height
    for mp in masks:
        c,r = offs(mp)
        if r>=H or c>=W: continue
        g=(np.array(Image.open(mp).convert("L"))>128).astype(np.uint8)
        th,tw=g.shape
        p = getpred(r,c,th,tw)
        TP+=int((p&g).sum()); FP+=int((p&(1-g)).sum()); FN+=int(((1-p)&g).sum())
        C.append(cld(p,g)); B.append(abs(cc(p)[1]-cc(g)[1]))
    P=TP/max(TP+FP,1); R=TP/max(TP+FN,1)
    return dict(P=P,R=R,F1=2*P*R/max(P+R,1e-9),IoU=TP/max(TP+FP+FN,1),
                clD=float(np.mean(C)),B0=float(np.mean(B)))


def main():
    t0=time.time()
    masks=sorted(glob.glob(os.path.join(VAL,"*.png")))
    src=rasterio.open(SRC); base_tf=src.transform; crs=src.crs
    pred=rasterio.open(PRED)
    ped=gpd.read_file(PEDN)
    if ped.crs is None: ped.set_crs(crs,inplace=True)
    elif ped.crs!=crs: ped=ped.to_crs(crs)
    sidx=ped.sindex
    st=disk(G_PX)

    def raw(r,c,th,tw):
        a=pred.read(1,window=Window(c,r,min(tw,pred.width-c),min(th,pred.height-r)))
        if a.shape!=(th,tw):
            q=np.zeros((th,tw),a.dtype); q[:a.shape[0],:a.shape[1]]=a; a=q
        return (a>0).astype(np.uint8)

    def bridged(r,c,th,tw):
        M=raw(r,c,th,tw)
        if M.sum()==0: return M
        wtf=win_tf(Window(c,r,tw,th),base_tf)
        x0,y0=wtf.c,wtf.f; x1=x0+tw*wtf.a; y1=y0+th*wtf.e
        bb=box(min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1))
        cand=list(sidx.intersection(bb.bounds))
        if not cand:
            return M
        geoms=[ped.geometry.iloc[i].buffer(CORRIDOR_M) for i in cand]
        corridor=rasterize([(gm,1) for gm in geoms],out_shape=(th,tw),
                           transform=wtf,fill=0,dtype="uint8").astype(bool)
        closed=binary_closing(M>0,footprint=st)
        bridge=closed & corridor & (~(M>0))
        return (M.astype(bool) | bridge).astype(np.uint8)

    print(f"G={G_PX}px(~{G_PX*0.3:.1f}m bridge<= {2*G_PX*0.3:.0f}m)  "
          f"corridor={2*CORRIDOR_M:.0f}m  val={len(masks)}\n")
    print(f"{'variant':<30}{'P':>8}{'R':>8}{'F1':>8}{'IoU':>8}{'clDice':>8}{'Betti0':>8}")
    print("-"*70)
    b=metrics(raw,masks,src)
    print(f"{'BEFORE (tv+pednet)':<30}{b['P']:>8.4f}{b['R']:>8.4f}{b['F1']:>8.4f}"
          f"{b['IoU']:>8.4f}{b['clD']:>8.4f}{b['B0']:>8.2f}")
    a=metrics(bridged,masks,src)
    print(f"{'AFTER (+footpath bridge)':<30}{a['P']:>8.4f}{a['R']:>8.4f}{a['F1']:>8.4f}"
          f"{a['IoU']:>8.4f}{a['clD']:>8.4f}{a['B0']:>8.2f}")
    print()
    print(f"delta  P {a['P']-b['P']:+.4f}  R {a['R']-b['R']:+.4f}  "
          f"F1 {a['F1']-b['F1']:+.4f}  IoU {a['IoU']-b['IoU']:+.4f}  "
          f"clDice {a['clD']-b['clD']:+.4f}  Betti0 {a['B0']-b['B0']:+.2f}")
    verdict = ("SAFE: F1 not harmed" if a['F1']>=b['F1']-0.002
               else "HARMS F1 -- do NOT deploy")
    print(f"\nVERDICT: {verdict}   (total {(time.time()-t0)/60:.1f} min)")
    src.close(); pred.close()


if __name__=="__main__":
    main()
