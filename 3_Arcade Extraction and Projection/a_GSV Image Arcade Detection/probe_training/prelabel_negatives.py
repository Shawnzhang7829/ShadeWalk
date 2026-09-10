"""Pre-label hard-negative candidates for the user to judge.
  Pool: Tier1 (CoveredLinkWay = HDB linkways, covered but not shophouse) + Tier3 covered positives
  Score with the strict probe and pick "covered but low strict score = not shophouse" → hard-negative candidates
  Copy 100 to candidates_neg_100/ + contact sheets. The user deletes those that are actually shophouses; what remains = clean negatives.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor
import joblib

ROOT=Path(r"D:\Claude\SVI_FFW")
OUT=ROOT/"output"/"strict_gt"
device="cuda" if torch.cuda.is_available() else "cpu"
L14="openai/clip-vit-large-patch14"
POOL_N=1200; PRESENT_N=100; SEED=11

print("[1] candidate pool: Tier1 (HDB linkways) + Tier3 covered positives …")
r=pd.read_csv(ROOT/"output"/"stageA_l14"/"results_sg_l14.csv")
pos100_pids=set(pd.read_csv(OUT/"prelabel_500.csv").query("rank<=100").pid)
# Tier1 covered (mostly HDB linkways) + Tier3 covered, excluding positive pids
t1=r[(r.is_positive==1)&(r.region_tier==1)&(~r.pid.isin(pos100_pids))]
t3=r[(r.is_positive==1)&(r.region_tier==3)&(~r.pid.isin(pos100_pids))]
pool=pd.concat([t1.sample(min(800,len(t1)),random_state=SEED),
                t3.sample(min(400,len(t3)),random_state=SEED)]).reset_index(drop=True)
print(f"    pool {len(pool)} (T1={len(t1)} T3={len(t3)})")

print("[2] scoring with the strict probe …")
from transformers import CLIPProcessor, CLIPModel
clip=CLIPModel.from_pretrained(L14).to(device).eval(); proc=CLIPProcessor.from_pretrained(L14)
probe=joblib.load(OUT/"sg_strict_probe.joblib")
def load(p):
    try:return p,Image.open(p).convert("RGB")
    except:return p,None
scores=[]; paths=pool["path"].tolist()
with ThreadPoolExecutor(max_workers=8) as pool_ex:
    for i in range(0,len(paths),16):
        loaded=list(pool_ex.map(load,paths[i:i+16])); imgs=[im for _,im in loaded if im is not None]
        inp=proc(images=imgs,return_tensors="pt").to(device)
        with torch.no_grad():
            vo=clip.vision_model(pixel_values=inp["pixel_values"]); f=clip.visual_projection(vo.pooler_output)
        f=f/f.norm(dim=-1,keepdim=True); pr=probe.predict_proba(f.cpu().numpy())[:,1]
        pi=0
        for _,im in loaded:
            scores.append(float(pr[pi]) if im is not None else -1.0); pi+=1 if im is not None else 0
pool["strict_score"]=scores

print("[3] picking hard-negative candidates: clearly covered (high broad score) but not shophouse (low strict score) = HDB linkway hard negatives …")
# hard negative = clearly covered (high broad probe_score) and low strict score (not shophouse)
pool["hardness"]=pool["probe_score"]*(1-pool["strict_score"])  # clearly covered and not shophouse
cand=pool[pool["strict_score"]<0.5].sort_values("hardness",ascending=False).head(PRESENT_N).reset_index(drop=True)
cand["rank"]=range(1,len(cand)+1)
cand[["rank","path","pid","region_tier","strict_score"]].to_csv(OUT/"prelabel_neg.csv",index=False)

cdir=OUT/"candidates_neg_100"; cdir.mkdir(exist_ok=True)
for f in cdir.glob("*.jpg"): f.unlink()
for _,row in cand.iterrows():
    src=Path(row["path"]); tier=int(row["region_tier"])
    dst=cdir/f"n{int(row['rank']):03d}_t{tier}_s{row['strict_score']:.2f}_{src.name}"
    shutil.copy2(src,dst)
CELL=300; COLS=5; ROWS=4; PER=20
for s in range(0,len(cand),PER):
    chunk=cand.iloc[s:s+PER]
    canvas=Image.new("RGB",(COLS*CELL,ROWS*CELL),(20,20,20)); d=ImageDraw.Draw(canvas)
    for i,(_,row) in enumerate(chunk.iterrows()):
        cx=(i%COLS)*CELL; cy=(i//COLS)*CELL
        try:
            im=Image.open(row["path"]).convert("RGB"); im.thumbnail((CELL-4,CELL-20)); canvas.paste(im,(cx+2,cy+18))
        except: pass
        d.rectangle([cx,cy,cx+CELL,cy+16],fill=(0,0,0))
        d.text((cx+3,cy+3),f"n{int(row['rank'])} t{int(row['region_tier'])} s{row['strict_score']:.2f}",fill=(255,200,0))
    canvas.save(OUT/f"neg_sheet_{s//PER}.jpg",quality=88)
print(f"    copied {len(cand)} to candidates_neg_100/, 5 sheets")
print(f"\n[done] Your action: browse candidates_neg_100/; these should all be 'non-shophouse covered walkways or open scenes'.")
print(f"       If one is actually a shophouse five-foot way, delete it (a probe error); what remains = clean negatives.")
