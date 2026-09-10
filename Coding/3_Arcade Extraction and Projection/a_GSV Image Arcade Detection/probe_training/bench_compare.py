"""Compare detection methods on the SAME per-city GT used for the production probes:
  zeroshot-B/32, zeroshot-L/14, probe-B/32, probe-L/14.
SG = curated 480 (GroupKFold-5 by pano); BO = 3000 portici (StratifiedKFold-5).
Writes _bench_results.csv  (method, city, F1, P, R, acc, thr)."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold

ROOT=Path(r"D:\Claude\SVI_FFW"); OUT=ROOT/"output"; SG=OUT/"strict_gt"
device="cuda" if torch.cuda.is_available() else "cpu"
from transformers import CLIPProcessor, CLIPModel
BB={"B/32":"openai/clip-vit-base-patch32","L/14":"openai/clip-vit-large-patch14"}
PROMPTS3=["a street with an arcade or covered walkway under the building",
          "a street with open sidewalk, no covered walkway",
          "an industrial warehouse or factory with a metal canopy or loading bay shelter"]

def metrics(s,y,fine=True):
    y=np.asarray(y); best=None
    for t in (np.linspace(0.05,0.95,91) if fine else [0.5]):
        p=(np.asarray(s)>=t).astype(int)
        tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum());tn=int(((p==0)&(y==0)).sum())
        P=tp/max(1,tp+fp);R=tp/max(1,tp+fn);F=2*P*R/max(1e-9,P+R);A=(tp+tn)/len(y)
        if best is None or F>best["F1"]: best=dict(thr=float(t),F1=F,P=P,R=R,acc=A)
    return best

def embed(model_id,paths,bs=32):
    clip=CLIPModel.from_pretrained(model_id).to(device).eval(); proc=CLIPProcessor.from_pretrained(model_id)
    def txt(pr):
        t=proc(text=pr,return_tensors="pt",padding=True).to(device)
        with torch.no_grad():
            to=clip.text_model(input_ids=t["input_ids"],attention_mask=t.get("attention_mask")); tf=clip.text_projection(to.pooler_output)
        return tf/tf.norm(dim=-1,keepdim=True)
    tf3=txt(PROMPTS3); feats=[]; s3=[]
    for i in range(0,len(paths),bs):
        ims=[]
        for p in paths[i:i+bs]:
            try: ims.append(Image.open(p).convert("RGB"))
            except: pass
        inp=proc(images=ims,return_tensors="pt").to(device)
        with torch.no_grad():
            vo=clip.vision_model(pixel_values=inp["pixel_values"]); imf=clip.visual_projection(vo.pooler_output)
        imf=imf/imf.norm(dim=-1,keepdim=True); feats.append(imf.cpu().numpy())
        s3.extend((100.0*imf@tf3.t()).softmax(-1).cpu().numpy()[:,0])
        if (i//bs)%20==0: print(f"    {model_id.split('-')[-1]} {i+len(ims)}/{len(paths)}",flush=True)
    del clip; torch.cuda.empty_cache()
    return np.concatenate(feats), np.array(s3)

# ---- SG curated 480 ----
def ranks(folder,pre): return {int(m.group(1)) for f in Path(folder).glob("*.jpg") if (m:=re.match(pre+r"(\d+)_",f.name))}
p500=pd.read_csv(SG/"prelabel_500.csv");pn=pd.read_csv(SG/"prelabel_neg.csv");pn2=pd.read_csv(SG/"prelabel_neg2.csv");pn4=pd.read_csv(SG/"prelabel_neg4.csv")
pos=pd.concat([p500[p500["rank"]<=100][["path","pid"]],pn[pn["rank"].isin(ranks(SG/"candidates_neg_back","n"))][["path","pid"]],
  pn2[pn2["rank"].isin(ranks(SG/"candidates_neg2_back","m"))][["path","pid"]],pn4[pn4["rank"].isin(ranks(SG/"candidates_neg4_back","j"))][["path","pid"]]]);pos["y"]=1
neg=pd.concat([pn[pn["rank"].isin(ranks(SG/"candidates_neg_100","n"))][["path","pid"]],pn2[pn2["rank"].isin(ranks(SG/"candidates_neg2_100","m"))][["path","pid"]],
  pn4[pn4["rank"].isin(ranks(SG/"candidates_neg4_100","j"))][["path","pid"]]]);neg["y"]=0
r=pd.read_csv(OUT/"stageA_l14"/"results_sg_l14.csv");used=set(pos.pid)|set(neg.pid)
op=r[(r.is_positive==0)&(~r.pid.isin(used))].sample(80,random_state=7)[["path","pid"]].assign(y=0)
sgdf=pd.concat([pos,neg,op]).reset_index(drop=True)
# ---- BO 3000 ----
bo=pd.concat([pd.read_csv(OUT/"phase0_samples"/"bo_positive.csv").sample(1500,random_state=1).assign(y=1),
              pd.read_csv(OUT/"phase0_samples"/"bo_negative.csv").sample(1500,random_state=1).assign(y=0)])[["path","y"]].reset_index(drop=True)

rows=[]
for bb,mid in BB.items():
    print(f"[embed SG] {bb}",flush=True); Xs,s3s=embed(mid,sgdf["path"].tolist()); ys=sgdf["y"].values[:len(Xs)]; gs=sgdf["pid"].values[:len(Xs)]
    z=metrics(s3s[:len(Xs)],ys); rows.append(dict(method=f"zero-shot {bb}",city="SG",**z))
    clf=LogisticRegression(max_iter=4000,C=0.5,class_weight="balanced")
    pr=cross_val_predict(clf,Xs,ys,cv=GroupKFold(5),groups=gs,method="predict_proba")[:,1]
    rows.append(dict(method=f"probe {bb}",city="SG",**metrics(pr,ys)))
    print(f"[embed BO] {bb}",flush=True); Xb,s3b=embed(mid,bo["path"].tolist()); yb=bo["y"].values[:len(Xb)]
    rows.append(dict(method=f"zero-shot {bb}",city="BO",**metrics(s3b[:len(Xb)],yb)))
    clfb=LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced")
    prb=cross_val_predict(clfb,Xb,yb,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
    rows.append(dict(method=f"probe {bb}",city="BO",**metrics(prb,yb)))

res=pd.DataFrame(rows); res.to_csv(OUT/"_bench_results.csv",index=False)
print("\n"+res.to_string(index=False)); print("\n[wrote _bench_results.csv]")
