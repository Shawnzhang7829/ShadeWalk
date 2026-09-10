"""(1) Train the L/14 linear probes for both cities.
  SG: CoveredLinkWay bearing-filtered positives + distant negatives (gt_candidates.csv)
  BO: portici positives/negatives (phase0_samples)
  For each: L/14 embeddings → LogisticRegression → save probe + per-tier thresholds.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

ROOT = Path(r"D:\Claude\SVI_FFW")
OUT = ROOT/"output"/"stageA_l14"; OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
L14 = "openai/clip-vit-large-patch14"

from transformers import CLIPProcessor, CLIPModel
print("[load] CLIP-L/14 …")
clip = CLIPModel.from_pretrained(L14).to(device).eval()
proc = CLIPProcessor.from_pretrained(L14)

def embed(paths, bs=16):
    feats=[]
    for i in range(0,len(paths),bs):
        imgs=[Image.open(p).convert("RGB") for p in paths[i:i+bs]]
        inp=proc(images=imgs,return_tensors="pt").to(device)
        with torch.no_grad():
            vo=clip.vision_model(pixel_values=inp["pixel_values"])
            f=clip.visual_projection(vo.pooler_output)
        f=f/f.norm(dim=-1,keepdim=True)
        feats.append(f.cpu().numpy())
        if (i//bs)%30==0: print(f"    {i+len(imgs)}/{len(paths)}",flush=True)
    return np.concatenate(feats)

def best_threshold(proba,y):
    best=(0.5,0)
    for t in np.linspace(0.1,0.9,33):
        pred=(proba>=t).astype(int)
        tp=((pred==1)&(y==1)).sum();fp=((pred==1)&(y==0)).sum();fn=((pred==0)&(y==1)).sum()
        p=tp/max(1,tp+fp);r=tp/max(1,tp+fn);f=2*p*r/max(1e-9,p+r)
        if f>best[1]: best=(float(t),float(f),float(p),float(r))
    return best

def train_city(name, df):
    print(f"\n=== {name} === ({df.y.sum()} pos / {(df.y==0).sum()} neg)")
    X=embed(df["path"].tolist()); y=df["y"].values
    clf=LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced")
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    proba=cross_val_predict(clf,X,y,cv=cv,method="predict_proba")[:,1]
    t,f1,p,r=best_threshold(proba,y)
    print(f"  5-fold CV best: t={t:.3f} F1={f1:.3f} (P={p:.2f} R={r:.2f})")
    clf.fit(X,y)
    joblib.dump(clf, OUT/f"{name}_probe.joblib")
    print(f"  saved {name}_probe.joblib")
    return t

# SG
sg=pd.read_csv(ROOT/"output"/"sg_improve"/"gt_candidates.csv")[["path","cand_label"]].rename(columns={"cand_label":"y"})
t_sg=train_city("sg", sg)
# BO
bopos=pd.read_csv(ROOT/"output"/"phase0_samples"/"bo_positive.csv").sample(1500,random_state=1).assign(y=1)
boneg=pd.read_csv(ROOT/"output"/"phase0_samples"/"bo_negative.csv").sample(1500,random_state=1).assign(y=0)
bo=pd.concat([bopos,boneg])[["path","y"]].reset_index(drop=True)
t_bo=train_city("bo", bo)

# per-tier thresholds: base±offset (tiers with a higher prior are more lenient)
thr={
  "sg": {"base":t_sg, 1:max(0.15,t_sg-0.15), 2:max(0.2,t_sg-0.08), 3:min(0.9,t_sg+0.08)},
  "bo": {"base":t_bo, 1:max(0.15,t_bo-0.15), 2:max(0.2,t_bo-0.08), 3:min(0.9,t_bo+0.08)},
}
(OUT/"thresholds.json").write_text(json.dumps(thr,indent=2))
print("\nthresholds:",json.dumps(thr,indent=2))
print(f"\nwrote {OUT/'thresholds.json'}")
