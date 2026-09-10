"""Step B (QGIS python): run CLIP on the prepped pos/neg; PR under the two protocols RAW + BEARING-AWARE."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(r"D:\Claude\SVI_FFW")
OUT = ROOT / "output" / "phase1c_sg_val"

def metrics(s,y,t):
    pred=(np.asarray(s)>=t).astype(int);y=np.asarray(y)
    tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum())
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn);f=2*p*r/max(1e-9,p+r);return p,r,f

def sweep(s,y,name):
    print(f"\n--- {name}  (n={len(y)}, pos={int(np.sum(y))}) ---")
    best=None
    for t in [0.07,0.12,0.20,0.25,0.30,0.40]:
        p,r,f=metrics(s,y,t); print(f"  t={t:.2f}  P={p:.2f} R={r:.2f} F1={f:.3f}")
        if best is None or f>best[1]: best=(t,f,p,r)
    print(f"  BEST t={best[0]:.2f} F1={best[1]:.3f} (P={best[2]:.2f} R={best[3]:.2f})")
    return best

pos=pd.read_csv(OUT/"eval_pos_prepped.csv")
neg=pd.read_csv(OUT/"eval_neg_prepped.csv")
device="cuda" if torch.cuda.is_available() else "cpu"
from transformers import CLIPProcessor, CLIPModel
cid="openai/clip-vit-base-patch32"
clip=CLIPModel.from_pretrained(cid).to(device).eval()
proc=CLIPProcessor.from_pretrained(cid)
PROMPTS=["a street with an arcade or covered walkway under the building",
         "a street with open sidewalk, no covered walkway"]
def score(paths):
    out=[];BS=16
    for i in range(0,len(paths),BS):
        imgs=[Image.open(p).convert("RGB") for p in paths[i:i+BS]]
        inp=proc(text=PROMPTS,images=imgs,return_tensors="pt",padding=True)
        inp={k:v.to(device) for k,v in inp.items()}
        with torch.no_grad(): o=clip(**inp)
        out.extend(o.logits_per_image.softmax(-1).cpu().numpy()[:,0].tolist())
    return np.array(out)

print("scoring positives …"); pos["clip"]=score(pos.path.tolist())
print("scoring negatives …"); neg["clip"]=score(neg.path.tolist())

s_raw=np.concatenate([pos["clip"].values,neg["clip"].values])
y_raw=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
b_raw=sweep(s_raw,y_raw,"RAW (all views of pos points)")

posf=pos[pos["faces_clw"]==True]
s_ba=np.concatenate([posf["clip"].values,neg["clip"].values])
y_ba=np.concatenate([np.ones(len(posf)),np.zeros(len(neg))])
b_ba=sweep(s_ba,y_ba,"BEARING-AWARE (pos views facing CLW)")

pos.to_csv(OUT/"eval_pos_scored.csv",index=False)
neg.to_csv(OUT/"eval_neg_scored.csv",index=False)
print("\n=== SUMMARY (official CoveredLinkWay validation) ===")
print(f"RAW            best F1={b_raw[1]:.3f} @ t={b_raw[0]:.2f} (P={b_raw[2]:.2f} R={b_raw[3]:.2f})")
print(f"BEARING-AWARE  best F1={b_ba[1]:.3f} @ t={b_ba[0]:.2f} (P={b_ba[2]:.2f} R={b_ba[3]:.2f})")
