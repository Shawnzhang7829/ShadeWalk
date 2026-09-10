"""
Checkpoint-level paper metrics on the 303 v3 val tiles.

Runs forward_autonomous directly on each val image (no island inference),
so it is an apples-to-apples comparison of trained models.

Reports micro Precision/Recall/F1/IoU + mean clDice + mean Betti-0 for
each --ckpt given.
"""
from __future__ import annotations
import os, sys, glob, argparse, numpy as np, torch, torch.nn as nn
from PIL import Image
from skimage.morphology import skeletonize
from scipy.ndimage import label as cc_label
from segment_anything import sam_model_registry

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))  # repo layout: training modules

from train_geosam_topolora_linkway import inject_lora_into_qkv, ensure_text_embedding
from train_autonomous_v2 import forward_autonomous, _MEAN, _STD

IMG = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\images\val"
MSK = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks\val"
SAM = r"C:\GeoSAM-backup\sam_vit_h_4b8939.pth"
TXT = r"C:\GeoSAM-backup\clip_linkway_emb.pth"


def load(ck, dev="cuda"):
    sam = sam_model_registry["vit_h"](checkpoint=SAM).to(dev)
    for p in sam.parameters(): p.requires_grad_(False)
    st = torch.load(ck, map_location=dev, weights_only=False)
    inject_lora_into_qkv(sam.image_encoder, rank=st.get("lora_rank",16),
                         alpha=st.get("lora_alpha",32.0))
    sam.image_encoder.to(dev)
    sam.load_state_dict(st["sam_trainable"], strict=False)
    pj = nn.Linear(512,256).to(dev); pj.load_state_dict(st["projection"])
    sam.eval(); pj.eval()
    return sam, pj


def cldice(p,g,e=1e-7):
    if p.sum()==0 and g.sum()==0: return 1.0
    if p.sum()==0 or g.sum()==0:  return 0.0
    sp,sg=skeletonize(p>0),skeletonize(g>0)
    tp=(sp&(g>0)).sum()/(sp.sum()+e); ts=(sg&(p>0)).sum()/(sg.sum()+e)
    return 0.0 if tp+ts<e else float(2*tp*ts/(tp+ts+e))


@torch.no_grad()
def evaluate(ck, thr, dev="cuda"):
    sam,pj = load(ck, dev)
    temb = ensure_text_embedding(TXT,"Covered Linkway",dev)
    names = sorted(f for f in os.listdir(IMG) if f.endswith(".png"))
    names = [n for n in names if os.path.exists(os.path.join(MSK,n))]
    TP=FP=FN=0; clds=[]; betti=[]
    for n in names:
        img=np.array(Image.open(os.path.join(IMG,n)).convert("RGB"),np.float32)/255.0
        g=(np.array(Image.open(os.path.join(MSK,n)).convert("L"))>128).astype(np.uint8)
        x=((img-_MEAN)/_STD).transpose(2,0,1)
        xt=torch.from_numpy(x)[None].to(dev)
        with torch.amp.autocast("cuda"):
            lo=forward_autonomous(sam,pj,temb,xt,dev,use_text=True)
        p=(torch.sigmoid(lo).squeeze().float().cpu().numpy()>thr).astype(np.uint8)
        tp=int((p&g).sum()); fp=int((p&(1-g)).sum()); fn=int(((1-p)&g).sum())
        TP+=tp;FP+=fp;FN+=fn
        clds.append(cldice(p,g)); betti.append(abs(cc_label(p)[1]-cc_label(g)[1]))
    P=TP/max(TP+FP,1); R=TP/max(TP+FN,1)
    F1=2*P*R/max(P+R,1e-9); IoU=TP/max(TP+FP+FN,1)
    return dict(P=P,R=R,F1=F1,IoU=IoU,clD=float(np.mean(clds)),
                B0=float(np.mean(betti)),n=len(names))


if __name__=="__main__":
    Ap=argparse.ArgumentParser()
    Ap.add_argument("--ckpts", nargs="+", required=True)
    Ap.add_argument("--labels", nargs="+", required=True)
    Ap.add_argument("--thr", type=float, default=0.5)
    a=Ap.parse_args()
    print(f"thr={a.thr}\n{'model':<34}{'P':>8}{'R':>8}{'F1':>8}{'IoU':>8}"
          f"{'clDice':>8}{'Betti0':>8}")
    print("-"*82)
    for ck,lb in zip(a.ckpts,a.labels):
        r=evaluate(ck,a.thr)
        print(f"{lb:<34}{r['P']:>8.4f}{r['R']:>8.4f}{r['F1']:>8.4f}"
              f"{r['IoU']:>8.4f}{r['clD']:>8.4f}{r['B0']:>8.2f}")
