"""Use the depth tunnel to select truly facade-facing views + CLIP retrieval of no_colonnade → clean facing negatives.
  Pool: sample 2200 covered positives → compute depth tunnel → keep low tunnel (facade-facing, bottom 40%)
  → CLIP retrieval of the images most similar to no_colonnade → top100.
"""
from __future__ import annotations
import re
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(r"D:\Claude\SVI_FFW"); OUT=ROOT/"output"/"strict_gt"
device="cuda" if torch.cuda.is_available() else "cpu"; L14="openai/clip-vit-large-patch14"
TUNNEL_KEEP_Q=0.40   # keep the lowest 40% of tunnel (most facade-facing)
PRESENT_N=100
def ranks_in(folder,pre): return {int(m.group(1)) for f in Path(folder).glob("*.jpg") if (m:=re.match(pre+r"(\d+)_",f.name))}

pre_pos=pd.read_csv(OUT/"prelabel_500.csv"); pre_neg=pd.read_csv(OUT/"prelabel_neg.csv"); pre_neg2=pd.read_csv(OUT/"prelabel_neg2.csv")
labeled=set(pre_pos[pre_pos["rank"]<=100].pid)|set(pre_neg["pid"])|set(pre_neg2["pid"])
q1=pre_neg[pre_neg["rank"].isin(ranks_in(OUT/"candidates_neg_100","n"))]
q2=pre_neg2[pre_neg2["rank"].isin(ranks_in(OUT/"candidates_neg2_100","m"))]
query=pd.concat([q1,q2])[["path","pid"]]

r=pd.read_csv(ROOT/"output"/"stageA_l14"/"results_sg_l14.csv")
pool=r[(r.is_positive==1)&(~r.pid.isin(labeled))].sample(2200,random_state=17).reset_index(drop=True)
print(f"pool {len(pool)}, query {len(query)}")

from transformers import CLIPProcessor, CLIPModel, pipeline as hfpipe
clip=CLIPModel.from_pretrained(L14).to(device).eval(); proc=CLIPProcessor.from_pretrained(L14)
dpipe=hfpipe("depth-estimation",model="depth-anything/Depth-Anything-V2-Small-hf",device=0 if device=="cuda" else -1)
def load(p):
    try:return Image.open(p).convert("RGB")
    except:return None
def tunnel(img):
    d=np.asarray(dpipe(img)["predicted_depth"],dtype=np.float32);W,H=img.size
    d=np.array(Image.fromarray(d).resize((W,H)));dn=(d-d.min())/max(1e-6,d.max()-d.min());far=1-dn
    cb=far[int(H*0.30):int(H*0.62),int(W*0.30):int(W*0.70)]
    lb=far[int(H*0.30):int(H*0.62),int(W*0.05):int(W*0.25)];rb=far[int(H*0.30):int(H*0.62),int(W*0.75):int(W*0.95)]
    return float(np.percentile(cb,75)-np.percentile(np.concatenate([lb.ravel(),rb.ravel()]),75))

print("[1] depth tunnel: selecting facade-facing views …")
kept_paths=[];tn=[];feats=[]
paths=pool["path"].tolist()
for i in range(0,len(paths),16):
    loaded=[(p,load(p)) for p in paths[i:i+16]]
    valid=[(p,im) for p,im in loaded if im is not None]
    if not valid: continue
    ims=[im for _,im in valid]
    inp=proc(images=ims,return_tensors="pt").to(device)
    with torch.no_grad():
        vo=clip.vision_model(pixel_values=inp["pixel_values"]);f=clip.visual_projection(vo.pooler_output)
    f=(f/f.norm(dim=-1,keepdim=True)).cpu().numpy()
    for j,(p,im) in enumerate(valid):
        kept_paths.append(p); feats.append(f[j]); tn.append(tunnel(im))
    if i%320==0: print(f"   {i+len(valid)}/{len(paths)}",flush=True)
pool=pool.set_index("path").loc[kept_paths].reset_index()   # align strictly to kept_paths
pool["tunnel"]=tn; Xpool=np.array(feats)
print("[2] directly take the 100 most negative tunnel values (most facade-facing) …")
cand=pool.sort_values("tunnel").head(PRESENT_N).reset_index(drop=True); cand["rank"]=range(1,len(cand)+1)
print(f"   selected 100, tunnel range: {cand.tunnel.min():.2f} ~ {cand.tunnel.max():.2f}")
cand[["rank","path","pid","tunnel"]].to_csv(OUT/"prelabel_neg4.csv",index=False)
cdir=OUT/"candidates_neg4_100"; cdir.mkdir(exist_ok=True)
for f in cdir.glob("*.jpg"): f.unlink()
for _,row in cand.iterrows():
    src=Path(row["path"]); shutil.copy2(src,cdir/f"j{int(row['rank']):03d}_tn{row['tunnel']:+.2f}_{src.name}")
CELL=300;COLS=5;ROWS=4;PER=20
for s in range(0,len(cand),PER):
    ch=cand.iloc[s:s+PER];canvas=Image.new("RGB",(COLS*CELL,ROWS*CELL),(20,20,20));d=ImageDraw.Draw(canvas)
    for i,(_,row) in enumerate(ch.iterrows()):
        cx=(i%COLS)*CELL;cy=(i//COLS)*CELL
        try:
            im=Image.open(row["path"]).convert("RGB");im.thumbnail((CELL-4,CELL-16));canvas.paste(im,(cx+2,cy+14))
        except: pass
        d.rectangle([cx,cy,cx+CELL,cy+12],fill=(0,0,0));d.text((cx+2,cy+2),f"j{int(row['rank'])} tn{row['tunnel']:+.2f}",fill=(255,200,0))
    canvas.save(OUT/f"neg4_sheet_{s//PER}.jpg",quality=88)
print(f"copied {len(cand)} -> candidates_neg4_100/ (depth-selected facade-facing)")
