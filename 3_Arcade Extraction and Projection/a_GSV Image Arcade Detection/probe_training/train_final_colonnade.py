"""Final colonnade probe: all human-verified labels + clean facade-facing hard negatives (neg4).
  Positives: shophouse100 + colonnade (neg_back66 + neg2_back57 + neg4_back18) = 241
  Negatives: no_colonnade (neg_100 34 + neg2_100 43 + neg4_100 82) = 159 [+ open-scene samples]
  Compare CLIP-only vs CLIP+Depth with GroupKFold CV; check the rejection rate of no_colonnade (especially the clean facing neg4).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold

ROOT=Path(r"D:\Claude\SVI_FFW"); OUT=ROOT/"output"/"strict_gt"
device="cuda" if torch.cuda.is_available() else "cpu"; L14="openai/clip-vit-large-patch14"
N_OPEN=80; SEED=7
def ranks(folder,pre): return {int(m.group(1)) for f in Path(folder).glob("*.jpg") if (m:=re.match(pre+r"(\d+)_",f.name))}

p500=pd.read_csv(OUT/"prelabel_500.csv"); pn=pd.read_csv(OUT/"prelabel_neg.csv"); pn2=pd.read_csv(OUT/"prelabel_neg2.csv"); pn4=pd.read_csv(OUT/"prelabel_neg4.csv")
pos=pd.concat([
  p500[p500["rank"]<=100][["path","pid"]].assign(src="shophouse"),
  pn[pn["rank"].isin(ranks(OUT/"candidates_neg_back","n"))][["path","pid"]].assign(src="colonnade"),
  pn2[pn2["rank"].isin(ranks(OUT/"candidates_neg2_back","m"))][["path","pid"]].assign(src="colonnade"),
  pn4[pn4["rank"].isin(ranks(OUT/"candidates_neg4_back","j"))][["path","pid"]].assign(src="colonnade_facing"),
]); pos["y"]=1
neg=pd.concat([
  pn[pn["rank"].isin(ranks(OUT/"candidates_neg_100","n"))][["path","pid"]].assign(src="nocol"),
  pn2[pn2["rank"].isin(ranks(OUT/"candidates_neg2_100","m"))][["path","pid"]].assign(src="nocol"),
  pn4[pn4["rank"].isin(ranks(OUT/"candidates_neg4_100","j"))][["path","pid"]].assign(src="nocol_facing"),
]); neg["y"]=0
r=pd.read_csv(ROOT/"output"/"stageA_l14"/"results_sg_l14.csv")
used=set(pos.pid)|set(neg.pid)
op=r[(r.is_positive==0)&(~r.pid.isin(used))].sample(N_OPEN,random_state=SEED)[["path","pid"]].assign(src="open",y=0)
df=pd.concat([pos,neg,op]).reset_index(drop=True)
print(f"pos={len(pos)} (shop+colonnade)  neg={len(neg)+len(op)} (nocol{len(neg)}+open{len(op)})")
print(f"  of which clean facing: colonnade_facing={ (pos.src=='colonnade_facing').sum() }, nocol_facing={ (neg.src=='nocol_facing').sum() }")

from transformers import CLIPProcessor, CLIPModel, pipeline as hfpipe
clip=CLIPModel.from_pretrained(L14).to(device).eval(); proc=CLIPProcessor.from_pretrained(L14)
dpipe=hfpipe("depth-estimation",model="depth-anything/Depth-Anything-V2-Small-hf",device=0 if device=="cuda" else -1)
def load(p):
    try:return Image.open(p).convert("RGB")
    except:return None
def dfeat(img):
    d=np.asarray(dpipe(img)["predicted_depth"],dtype=np.float32);W,H=img.size
    d=np.array(Image.fromarray(d).resize((W,H)));dn=(d-d.min())/max(1e-6,d.max()-d.min())
    band=dn[int(H*0.35):int(H*0.90),int(W*0.10):int(W*0.90)]
    g=np.array(Image.fromarray((band*255).astype(np.uint8)).resize((12,8)),dtype=np.float32)/255
    rg=np.abs(np.diff(g,axis=1)).mean(axis=1)
    st=np.array([band.mean(),band.std(),(band>0.6).mean(),np.percentile(band,10),np.percentile(band,90)])
    return np.concatenate([g.flatten(),rg,st]).astype(np.float32)
Xc=[];Xd=[];paths=df["path"].tolist()
for i in range(0,len(paths),16):
    ims=[load(p) for p in paths[i:i+16]]; ims=[im for im in ims if im is not None]
    inp=proc(images=ims,return_tensors="pt").to(device)
    with torch.no_grad():
        vo=clip.vision_model(pixel_values=inp["pixel_values"]);f=clip.visual_projection(vo.pooler_output)
    Xc.append((f/f.norm(dim=-1,keepdim=True)).cpu().numpy())
    for im in ims: Xd.append(dfeat(im))
Xc=np.concatenate(Xc); Xd=np.array(Xd); Xd=(Xd-Xd.mean(0))/(Xd.std(0)+1e-6); df=df.iloc[:len(Xc)].copy()
y=df["y"].values; g=df["pid"].values; srcs=df["src"].values
def run(X,tag,save=None):
    clf=LogisticRegression(max_iter=4000,C=0.5,class_weight="balanced")
    pr=cross_val_predict(clf,X,y,cv=GroupKFold(5),groups=g,method="predict_proba")[:,1]
    best=(0.5,0,0,0)
    for t in np.linspace(0.2,0.8,25):
        p=(pr>=t).astype(int);tp=((p==1)&(y==1)).sum();fp=((p==1)&(y==0)).sum();fn=((p==0)&(y==1)).sum()
        P=tp/max(1,tp+fp);R=tp/max(1,tp+fn);F=2*P*R/max(1e-9,P+R)
        if F>best[1]:best=(t,F,P,R)
    t,f1,P,R=best;p=(pr>=t).astype(int)
    print(f"\n=== {tag} === t={t:.2f} F1={f1:.3f} (P={P:.2f} R={R:.2f})")
    for s in ["shophouse","colonnade","colonnade_facing","nocol","nocol_facing","open"]:
        m=srcs==s
        if m.sum()==0: continue
        if "col" in s and "nocol" not in s: print(f"    pos-{s}: recall {(p[m]==1).mean():.2f} (n={m.sum()})")
        else: print(f"    neg-{s}: rejected {(p[m]==0).mean():.2f} (n={m.sum()})")
    if save: clf.fit(X,y); import joblib; joblib.dump(clf,OUT/save); print(f"  saved {save}")
    return pr
run(Xc,"CLIP only", save="sg_colonnade_final.joblib")   # CLIP-only is better; used as the final probe
run(np.hstack([Xc,Xd]),"CLIP + Depth")
