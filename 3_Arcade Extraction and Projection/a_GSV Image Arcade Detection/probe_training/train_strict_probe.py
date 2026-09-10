"""Train the strict shophouse five-foot way probe (L/14).
  Positives: 100 human-confirmed images (prelabel_500 ranks 1-100)
  Negatives: (1) HDB linkway hard negatives (Tier1 CoveredLinkWay positives, covered but not shophouse)
             (2) open streets (is_positive==0)
  Key point: the hard negatives make the probe learn "shophouse vs other covered" rather than "covered vs open".
  Evaluation: GroupKFold by pid (reduces near-duplicate leakage).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold

ROOT=Path(r"D:\Claude\SVI_FFW")
OUT=ROOT/"output"/"strict_gt"
device="cuda" if torch.cuda.is_available() else "cpu"
L14="openai/clip-vit-large-patch14"
N_HARD_NEG=120   # HDB linkway hard negatives
N_OPEN_NEG=100   # open-scene negatives
SEED=7

# positives: the confirmed 100
pre=pd.read_csv(OUT/"prelabel_500.csv")
pos=pre[pre["rank"]<=100][["path","pid"]].copy(); pos["y"]=1; pos["src"]="shophouse"
print(f"positives (confirmed shophouse): {len(pos)}")

# negatives come from the city-wide results
r=pd.read_csv(ROOT/"output"/"stageA_l14"/"results_sg_l14.csv")
pos_pids=set(pos.pid)
# hard negatives: Tier1 (CoveredLinkWay, mostly HDB linkways) positives, excluding pids shared with positives
hard=r[(r.is_positive==1)&(r.region_tier==1)&(~r.pid.isin(pos_pids))].sample(N_HARD_NEG,random_state=SEED)
hard=hard[["path","pid"]].copy(); hard["y"]=0; hard["src"]="hdb_linkway"
# open negatives: is_positive==0, excluding shared pids
openn=r[(r.is_positive==0)&(~r.pid.isin(pos_pids))].sample(N_OPEN_NEG,random_state=SEED)
openn=openn[["path","pid"]].copy(); openn["y"]=0; openn["src"]="open"
print(f"negatives: hard(HDB linkway)={len(hard)}, open={len(openn)}")

df=pd.concat([pos,hard,openn]).reset_index(drop=True)

# L/14 embeddings
from transformers import CLIPProcessor, CLIPModel
clip=CLIPModel.from_pretrained(L14).to(device).eval(); proc=CLIPProcessor.from_pretrained(L14)
def load(p):
    try:return p,Image.open(p).convert("RGB")
    except:return p,None
feats=[]; paths=df["path"].tolist()
with ThreadPoolExecutor(max_workers=8) as pool:
    for i in range(0,len(paths),16):
        loaded=list(pool.map(load,paths[i:i+16])); imgs=[im for _,im in loaded if im is not None]
        inp=proc(images=imgs,return_tensors="pt").to(device)
        with torch.no_grad():
            vo=clip.vision_model(pixel_values=inp["pixel_values"]); f=clip.visual_projection(vo.pooler_output)
        f=f/f.norm(dim=-1,keepdim=True); feats.append(f.cpu().numpy())
X=np.concatenate(feats); y=df["y"].values; groups=df["pid"].values
print(f"embedded {X.shape}")

def f1_at(proba,y,ths=np.linspace(0.2,0.8,25)):
    best=(0.5,0,0,0)
    for t in ths:
        pred=(proba>=t).astype(int)
        tp=((pred==1)&(y==1)).sum();fp=((pred==1)&(y==0)).sum();fn=((pred==0)&(y==1)).sum()
        p=tp/max(1,tp+fp);rr=tp/max(1,tp+fn);ff=2*p*rr/max(1e-9,p+rr)
        if ff>best[1]:best=(float(t),ff,p,rr)
    return best

clf=LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced")
gkf=GroupKFold(n_splits=5)
proba=cross_val_predict(clf,X,y,cv=gkf,groups=groups,method="predict_proba")[:,1]
t,f1,p,rr=f1_at(proba,y)
print(f"\n=== strict probe GroupKFold (by pid) CV ===")
print(f"  best t={t:.3f} F1={f1:.3f} (P={p:.2f} R={rr:.2f})")
# hard-negative performance by negative source
pred=(proba>=t).astype(int)
for src in ["shophouse","hdb_linkway","open"]:
    mask=df.src.values==src
    if src=="shophouse":
        acc=(pred[mask]==1).mean(); print(f"  pos-shophouse recall: {acc:.2f}")
    else:
        acc=(pred[mask]==0).mean(); print(f"  neg-{src} correctly rejected: {acc:.2f}")

clf.fit(X,y); joblib.dump(clf,OUT/"sg_strict_probe.joblib")
df["cv_proba"]=proba; df.to_csv(OUT/"strict_train_scored.csv",index=False)
print(f"\nsaved sg_strict_probe.joblib + strict_train_scored.csv")
