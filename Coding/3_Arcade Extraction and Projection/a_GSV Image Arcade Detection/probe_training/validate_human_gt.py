"""Validate the L/14 probe vs B32 with the user's human GT (human_gt.csv, 15 pos / 17 neg).
All 32 images are in the SG manifest, so their decisions are taken directly from the two result sets and compared with the human labels.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Claude\SVI_FFW")
hg = pd.read_csv(ROOT/"output"/"phase1b_sg"/"human_gt.csv")   # cell_id,human_label,...,path
rl = pd.read_csv(ROOT/"output"/"stageA_l14"/"results_sg_l14.csv")
rb = pd.read_csv(ROOT/"output"/"stageA"/"results_sg.csv")

# path-normalised matching
def norm(s): return str(s).strip().lower()
rl["k"]=rl["path"].map(norm); rb["k"]=rb["path"].map(norm); hg["k"]=hg["path"].map(norm)

m = hg.merge(rl[["k","probe_score","is_positive"]].rename(columns={"is_positive":"l14_pos"}), on="k", how="left")
m = m.merge(rb[["k","clip_score","is_positive"]].rename(columns={"is_positive":"b32_pos"}), on="k", how="left")

n_match = m["l14_pos"].notna().sum()
print(f"human GT: {len(hg)} ({hg.human_label.sum()} pos / {(hg.human_label==0).sum()} neg)")
print(f"matched in L14 results: {n_match}/{len(hg)}")

def metrics(pred, y):
    pred=np.asarray(pred); y=np.asarray(y)
    tp=((pred==1)&(y==1)).sum();fp=((pred==1)&(y==0)).sum()
    fn=((pred==0)&(y==1)).sum();tn=((pred==0)&(y==0)).sum()
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn);f=2*p*r/max(1e-9,p+r);acc=(tp+tn)/len(y)
    return dict(F1=f,P=p,R=r,Acc=acc,TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn))

mm = m.dropna(subset=["l14_pos","b32_pos"]).copy()
y = mm.human_label.values
print("\n=== performance on the human GT ===")
mb = metrics(mm.b32_pos.values, y)
ml = metrics(mm.l14_pos.values, y)
print(f"B32  : F1={mb['F1']:.3f} P={mb['P']:.2f} R={mb['R']:.2f} Acc={mb['Acc']:.2f}  (TP{mb['TP']} FP{mb['FP']} FN{mb['FN']} TN{mb['TN']})")
print(f"L14  : F1={ml['F1']:.3f} P={ml['P']:.2f} R={ml['R']:.2f} Acc={ml['Acc']:.2f}  (TP{ml['TP']} FP{ml['FP']} FN{ml['FN']} TN{ml['TN']})")

# per-cell disagreements
print("\n=== cells misclassified by the probe ===")
for _,r in mm.iterrows():
    if int(r.l14_pos)!=int(r.human_label):
        kind="miss FN" if r.human_label==1 else "false alarm FP"
        print(f"  cell{int(r.cell_id)} [{kind}] human={int(r.human_label)} L14={int(r.l14_pos)}(prob{r.probe_score:.2f}) '{r.note}'")

mm.to_csv(ROOT/"output"/"stageA_l14"/"human_gt_validation.csv",index=False)
print(f"\nwrote human_gt_validation.csv")
