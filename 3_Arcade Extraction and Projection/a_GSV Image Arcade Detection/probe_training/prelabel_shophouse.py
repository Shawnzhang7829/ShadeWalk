"""Pre-label shophouse five-foot way candidates (strict definition).
  Pool: SG Tier2 (conservation areas, dense in shophouses) L/14 positives, top N by probe_score
  Scoring: L/14 + shophouse-specific prompts → P(shophouse five-foot way)
  Pre-label 500 → copy the top 100 to candidates_100/ + contact sheets for the user to pick from
"""
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(r"D:\Claude\SVI_FFW")
OUT=ROOT/"output"/"strict_gt"; OUT.mkdir(parents=True,exist_ok=True)
device="cuda" if torch.cuda.is_available() else "cpu"
L14="openai/clip-vit-large-patch14"

POOL_N=1500   # the top 1500 Tier2 positives by probe_score enter the pool
PRELABEL_N=500
PRESENT_N=100

# shophouse-specific prompts (0 = target)
PROMPTS=["a traditional old shophouse building with a covered five-foot way arcade walkway and columns at the ground floor",
         "a modern HDB apartment block or high-rise residential tower",
         "an open street, road, or open car park with no covered walkway",
         "a glass office building or modern commercial mall"]

print("[1] build pool from SG Tier2 positives …")
r=pd.read_csv(ROOT/"output"/"stageA_l14"/"results_sg_l14.csv")
pool=r[(r.is_positive==1)&(r.region_tier==2)].sort_values("probe_score",ascending=False).head(POOL_N).reset_index(drop=True)
print(f"    pool: {len(pool)} (Tier2 top by probe_score)")

print("[2] L/14 embed + shophouse score …")
from transformers import CLIPProcessor, CLIPModel
clip=CLIPModel.from_pretrained(L14).to(device).eval()
proc=CLIPProcessor.from_pretrained(L14)
def txt(prompts):
    t=proc(text=prompts,return_tensors="pt",padding=True).to(device)
    with torch.no_grad():
        to=clip.text_model(input_ids=t["input_ids"],attention_mask=t.get("attention_mask"))
        f=clip.text_projection(to.pooler_output)
    return f/f.norm(dim=-1,keepdim=True)
tf=txt(PROMPTS)
def load(p):
    try: return p,Image.open(p).convert("RGB")
    except: return p,None
scores=[]; paths=pool["path"].tolist()
with ThreadPoolExecutor(max_workers=8) as pool_ex:
    for i in range(0,len(paths),16):
        loaded=list(pool_ex.map(load,paths[i:i+16]))
        imgs=[im for _,im in loaded if im is not None]
        if imgs:
            inp=proc(images=imgs,return_tensors="pt").to(device)
            with torch.no_grad():
                vo=clip.vision_model(pixel_values=inp["pixel_values"])
                imf=clip.visual_projection(vo.pooler_output)
            imf=imf/imf.norm(dim=-1,keepdim=True)
            sm=(100.0*imf@tf.t()).softmax(-1).cpu().numpy()[:,0]
        else: sm=[]
        pi=0
        for _,im in loaded:
            scores.append(float(sm[pi]) if im is not None else -1.0); pi+= 1 if im is not None else 0
        if i%320==0: print(f"    {i+len(loaded)}/{len(paths)}",flush=True)
pool["shophouse_score"]=scores

print("[3] pre-labelling 500 (sorted by shophouse_score) …")
pre=pool.sort_values("shophouse_score",ascending=False).head(PRELABEL_N).reset_index(drop=True)
pre["rank"]=range(1,len(pre)+1)
pre[["rank","path","pid","viewheading","probe_score","shophouse_score"]].to_csv(OUT/"prelabel_500.csv",index=False)
print(f"    wrote prelabel_500.csv (shophouse_score {pre.shophouse_score.min():.2f}-{pre.shophouse_score.max():.2f})")

print(f"[4] copying the top {PRESENT_N} + contact sheets …")
cand=pre.head(PRESENT_N).reset_index(drop=True)
cdir=OUT/"candidates_100"; cdir.mkdir(exist_ok=True)
# clear old files
for f in cdir.glob("*.jpg"): f.unlink()
for _,row in cand.iterrows():
    src=Path(row["path"])
    dst=cdir/f"r{int(row['rank']):03d}_sh{row['shophouse_score']:.2f}_{src.name}"
    shutil.copy2(src,dst)
# contact sheets (5col×4row = 20 per sheet, 5 sheets in total = 100)
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
        d.text((cx+3,cy+3),f"r{int(row['rank'])} sh{row['shophouse_score']:.2f}",fill=(255,255,0))
    canvas.save(OUT/f"sheet_{s//PER}.jpg",quality=88)
print(f"    copied {len(cand)} to candidates_100/, wrote 5 sheets")
print(f"\n[done] Your action: open candidates_100/, delete those that are not shophouse five-foot ways; what remains = strict GT positives")
