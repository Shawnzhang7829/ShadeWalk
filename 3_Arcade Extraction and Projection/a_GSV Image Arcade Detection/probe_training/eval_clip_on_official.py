"""Evaluate CLIP (the selected main model) on the official CoveredLinkWay validation set.

Two protocols:
  RAW         : every heading of a positive point counts as positive (includes views facing away from the linkway → label noise)
  BEARING-AWARE: only views whose viewheading faces the overlap (|Δbearing|<FOV) are kept as positives

Outputs PR + best threshold, for comparison with the earlier hand labels / Bologna.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from PIL import Image

ROOT = Path(r"D:\Claude\SVI_FFW")
OUT = ROOT / "output" / "phase1c_sg_val"
EPSG = 3414
FOV_HALF = 55.0     # half field of view: |viewheading - bearing| < 55° counts as "facing" the linkway

N_POS = 1500
N_NEG = 1500
SEED = 42

# ---------- recompute the bearing from each positive point to the nearest overlap ----------
def add_bearing(pos: pd.DataFrame) -> pd.DataFrame:
    clw = gpd.read_file(ROOT/"Shp"/"SG"/"CoveredLinkWay_Mar2026"/"CoveredLinkWay.shp").to_crs(EPSG)
    bld = gpd.read_file(ROOT/"Shp"/"SG"/"SG_Building"/"SG_Building_SVY21_TH.shp").to_crs(EPSG)
    clw["geometry"]=clw.geometry.buffer(0); bld["geometry"]=bld.geometry.buffer(0)
    inter = gpd.overlay(clw[["OBJECTID","geometry"]], bld[["geometry"]],
                        how="intersection", keep_geom_type=True)
    inter = inter[inter.geometry.area>=3.0]
    # point → bearing to the nearest overlap centroid
    pts = gpd.GeoDataFrame(pos.copy(),
        geometry=gpd.points_from_xy(pos.lon,pos.lat,crs="EPSG:4326")).to_crs(EPSG)
    ov_cent = inter.geometry.centroid
    ov_union_pts = np.array([(g.x,g.y) for g in ov_cent])
    bearings=[]
    for geom in pts.geometry:
        dx = ov_union_pts[:,0]-geom.x; dy = ov_union_pts[:,1]-geom.y
        d2 = dx*dx+dy*dy
        k = int(np.argmin(d2))
        # bearing: 0=N, 90=E (clockwise), same convention as viewheading
        ang = (math.degrees(math.atan2(dx[k], dy[k]))) % 360
        bearings.append(ang)
    pos = pos.copy(); pos["bearing_to_clw"]=bearings
    dvh = (pos["viewheading"]-pos["bearing_to_clw"]).abs()%360
    dvh = dvh.where(dvh<=180, 360-dvh)
    pos["faces_clw"] = dvh < FOV_HALF
    return pos

def metrics(scores,labels,t):
    pred=(np.asarray(scores)>=t).astype(int); labels=np.asarray(labels)
    tp=int(((pred==1)&(labels==1)).sum());fp=int(((pred==1)&(labels==0)).sum());fn=int(((pred==0)&(labels==1)).sum())
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn);f=2*p*r/max(1e-9,p+r);return p,r,f

def sweep(scores,labels,name):
    print(f"\n--- {name}  (n={len(labels)}, pos={int(np.sum(labels))}) ---")
    best=None
    for t in [0.07,0.12,0.20,0.25,0.30,0.40]:
        p,r,f=metrics(scores,labels,t)
        print(f"  t={t:.2f}  P={p:.2f} R={r:.2f} F1={f:.3f}")
        if best is None or f>best[1]: best=(t,f,p,r)
    print(f"  BEST t={best[0]:.2f} F1={best[1]:.3f} (P={best[2]:.2f} R={best[3]:.2f})")
    return best

def main():
    pos=pd.read_csv(OUT/"sg_val_positive.csv")
    neg=pd.read_csv(OUT/"sg_val_negative.csv")
    pos=pos.sample(n=min(N_POS,len(pos)),random_state=SEED).reset_index(drop=True)
    neg=neg.sample(n=min(N_NEG,len(neg)),random_state=SEED).reset_index(drop=True)
    print(f"sampled pos={len(pos)} neg={len(neg)}")

    print("computing bearing-to-CLW for positives …")
    pos=add_bearing(pos)
    print(f"  positives facing CLW (|Δ|<{FOV_HALF}°): {int(pos.faces_clw.sum())}/{len(pos)}")

    # ---- CLIP ----
    device="cuda" if torch.cuda.is_available() else "cpu"
    from transformers import CLIPProcessor, CLIPModel
    cid="openai/clip-vit-base-patch32"
    clip=CLIPModel.from_pretrained(cid).to(device).eval()
    proc=CLIPProcessor.from_pretrained(cid)
    PROMPTS=["a street with an arcade or covered walkway under the building",
             "a street with open sidewalk, no covered walkway"]
    def clip_scores(paths):
        out=[]
        BS=16
        for i in range(0,len(paths),BS):
            imgs=[Image.open(p).convert("RGB") for p in paths[i:i+BS]]
            inp=proc(text=PROMPTS,images=imgs,return_tensors="pt",padding=True)
            inp={k:v.to(device) for k,v in inp.items()}
            with torch.no_grad(): o=clip(**inp)
            out.extend(o.logits_per_image.softmax(-1).cpu().numpy()[:,0].tolist())
        return np.array(out)

    print("scoring positives …"); pos["clip"]=clip_scores(pos.path.tolist())
    print("scoring negatives …"); neg["clip"]=clip_scores(neg.path.tolist())

    # RAW protocol
    s_raw=np.concatenate([pos.clip.values, neg.clip.values])
    y_raw=np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    b_raw=sweep(s_raw,y_raw,"RAW (all views of pos pts)")

    # BEARING-AWARE protocol: positives are only the views facing the linkway
    posf=pos[pos.faces_clw]
    s_ba=np.concatenate([posf.clip.values, neg.clip.values])
    y_ba=np.concatenate([np.ones(len(posf)), np.zeros(len(neg))])
    b_ba=sweep(s_ba,y_ba,"BEARING-AWARE (pos views facing CLW)")

    pos.to_csv(OUT/"eval_pos_scored.csv",index=False)
    neg.to_csv(OUT/"eval_neg_scored.csv",index=False)
    print("\n=== SUMMARY (official CoveredLinkWay validation) ===")
    print(f"RAW            best F1={b_raw[1]:.3f} @ t={b_raw[0]:.2f} (P={b_raw[2]:.2f} R={b_raw[3]:.2f})")
    print(f"BEARING-AWARE  best F1={b_ba[1]:.3f} @ t={b_ba[0]:.2f} (P={b_ba[2]:.2f} R={b_ba[3]:.2f})")

if __name__=="__main__":
    main()
