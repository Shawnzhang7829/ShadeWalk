"""SG F1 improvement benchmark (QGIS/torch):
  Methods: (1) zero-shot B/32  (2) zero-shot L/14  (3) prompt ensemble L/14
           (4) linear probe B/32  (5) linear probe L/14
  Evaluation: weak GT (800 CLW candidates, 5-fold CV) + clean GT (32 hand-labelled, gold cross-check)
  Also: confirm no degradation on BO portici.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

ROOT = Path(r"D:\Claude\SVI_FFW")
SGI = ROOT/"output"/"sg_improve"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- GT ----------
weak = pd.read_csv(SGI/"gt_candidates.csv")[["path","cand_label"]].rename(columns={"cand_label":"y"})
# clean 32 (hand-labelled)
gl = pd.read_csv(ROOT/"output"/"phase1b_sg"/"gt_labels.csv")
gi = pd.read_csv(ROOT/"output"/"phase1b_sg"/"montage_hi_index.csv")
clean = gl.merge(gi,on="cell_id")[["path","label"]].rename(columns={"label":"y"})
print(f"weak GT: {len(weak)} ({weak.y.sum()} pos)  | clean GT: {len(clean)} ({clean.y.sum()} pos)")

# BO portici GT
bopos = pd.read_csv(ROOT/"output"/"phase0_samples"/"bo_positive.csv").sample(150,random_state=1)
boneg = pd.read_csv(ROOT/"output"/"phase0_samples"/"bo_negative.csv").sample(150,random_state=1)
bo = pd.concat([bopos.assign(y=1), boneg.assign(y=0)])[["path","y"]].reset_index(drop=True)
print(f"BO GT: {len(bo)} ({bo.y.sum()} pos)")

# ---------- backbones ----------
from transformers import CLIPProcessor, CLIPModel
BACKBONES = {"B32":"openai/clip-vit-base-patch32", "L14":"openai/clip-vit-large-patch14"}

PROMPTS3 = ["a street with an arcade or covered walkway under the building",
            "a street with open sidewalk, no covered walkway",
            "an industrial warehouse or factory with a metal canopy or loading bay shelter"]
POS_ENS = ["a covered five-foot way walkway under a shophouse",
           "an arcade or colonnade with columns and a sheltered walkway",
           "a covered pedestrian linkway under the building"]
NEG_ENS = ["an open road with no shelter","a bare building facade with open sky",
           "trees, grass, or an open car park","an industrial loading bay canopy"]

def f1_sweep(s,y,ths=(0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5)):
    best=(0,0,0,0)
    for t in ths:
        pred=(np.asarray(s)>=t).astype(int);y=np.asarray(y)
        tp=((pred==1)&(y==1)).sum();fp=((pred==1)&(y==0)).sum();fn=((pred==0)&(y==1)).sum()
        p=tp/max(1,tp+fp);r=tp/max(1,tp+fn);f=2*p*r/max(1e-9,p+r)
        if f>best[1]: best=(t,f,p,r)
    return best  # t,f1,P,R

def embed_and_score(model_id, df):
    clip=CLIPModel.from_pretrained(model_id).to(device).eval()
    proc=CLIPProcessor.from_pretrained(model_id)
    feats=[]; s3=[]; sens=[]
    paths=df["path"].tolist(); BS=16
    # text features (low-level text_model + projection)
    def txt_feat(prompts):
        t=proc(text=prompts,return_tensors="pt",padding=True).to(device)
        with torch.no_grad():
            to=clip.text_model(input_ids=t["input_ids"],attention_mask=t.get("attention_mask"))
            tf=clip.text_projection(to.pooler_output)
        return (tf/tf.norm(dim=-1,keepdim=True))
    tf3=txt_feat(PROMPTS3); tfp=txt_feat(POS_ENS); tfn=txt_feat(NEG_ENS)
    for i in range(0,len(paths),BS):
        imgs=[Image.open(p).convert("RGB") for p in paths[i:i+BS]]
        inp=proc(images=imgs,return_tensors="pt").to(device)
        with torch.no_grad():
            vo=clip.vision_model(pixel_values=inp["pixel_values"])
            imf=clip.visual_projection(vo.pooler_output)
        imf=imf/imf.norm(dim=-1,keepdim=True)
        feats.append(imf.cpu().numpy())
        # 3-prompt softmax P(prompt0)
        logits3=(100.0*imf@tf3.t()).softmax(-1).cpu().numpy()[:,0]; s3.extend(logits3)
        # ensemble: max pos sim vs max neg sim → 2-class softmax
        sp=(imf@tfp.t()).max(dim=1).values; sn=(imf@tfn.t()).max(dim=1).values
        ens=torch.stack([sp,sn],dim=1).mul(100.0).softmax(-1).cpu().numpy()[:,0]; sens.extend(ens)
    del clip; torch.cuda.empty_cache()
    return np.concatenate(feats), np.array(s3), np.array(sens)

results=[]
emb={}
for name,mid in BACKBONES.items():
    print(f"\n[embed] {name} …")
    fw,s3w,sew = embed_and_score(mid, weak)
    fc,s3c,sec = embed_and_score(mid, clean)
    emb[name]=(fw,fc)
    # zero-shot 3-prompt
    bw=f1_sweep(s3w,weak.y.values); bc=f1_sweep(s3c,clean.y.values)
    results.append((f"zeroshot-{name} 3prompt", bw, bc))
    if name=="L14":
        # ensemble (shown for L14 only)
        bw=f1_sweep(sew,weak.y.values); bc=f1_sweep(sec,clean.y.values)
        results.append((f"ensemble-{name}", bw, bc))

# linear probe: weak GT 5-fold CV + train on weak → test on clean
for name in BACKBONES:
    fw,fc=emb[name]
    clf=LogisticRegression(max_iter=2000,C=1.0,class_weight="balanced")
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    proba_cv=cross_val_predict(clf,fw,weak.y.values,cv=cv,method="predict_proba")[:,1]
    bw=f1_sweep(proba_cv,weak.y.values)
    clf.fit(fw,weak.y.values)
    proba_c=clf.predict_proba(fc)[:,1]
    bc=f1_sweep(proba_c,clean.y.values)
    results.append((f"probe-{name}", bw, bc))

print("\n"+"="*78)
print(f"{'method':<24}{'weakGT F1(P/R)':<26}{'cleanGT F1(P/R)':<26}")
print("="*78)
for name,bw,bc in results:
    print(f"{name:<24}{bw[1]:.3f} ({bw[2]:.2f}/{bw[3]:.2f}) @{bw[0]:<6}{bc[1]:.3f} ({bc[2]:.2f}/{bc[3]:.2f}) @{bc[0]}")

# ---------- BO check ----------
print("\n[BO check] embedding portici GT …")
bo_res=[]
for name,mid in BACKBONES.items():
    fb,s3b,seb=embed_and_score(mid,bo)
    bb=f1_sweep(s3b,bo.y.values)
    bo_res.append((f"zeroshot-{name}",bb))
    # probe trained on SG weak → apply to BO? Not sensible. BO trains its probe on its own portici (5-fold CV)
    clf=LogisticRegression(max_iter=2000,class_weight="balanced")
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    pcv=cross_val_predict(clf,fb,bo.y.values,cv=cv,method="predict_proba")[:,1]
    bo_res.append((f"probe-{name}(BO-self)",f1_sweep(pcv,bo.y.values)))
print("\n--- BO portici GT ---")
for name,bb in bo_res:
    print(f"  {name:<22} F1={bb[1]:.3f} (P={bb[2]:.2f} R={bb[3]:.2f}) @t={bb[0]}")
