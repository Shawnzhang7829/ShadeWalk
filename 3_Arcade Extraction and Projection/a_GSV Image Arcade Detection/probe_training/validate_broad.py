"""Re-validate the L/14 probe vs B32 with the broad-definition GT (22 pos / 10 neg; only obvious/close-range counts)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(r"D:\Claude\SVI_FFW")
BASE=ROOT/"output"/"phase1b_sg"
gb=pd.read_csv(BASE/"human_gt_broad.csv")
# the 7 LOW images (edge/distant) are all negative per the user's rule "only obvious/close-range counts" (already 0); freeze it
gb["broad_label_final"]=gb["broad_label"]  # LOW is already labelled 0
gb.loc[gb.confidence=="LOW","broad_label_final"]=0
gb.to_csv(BASE/"human_gt_broad.csv",index=False)

rl=pd.read_csv(ROOT/"output"/"stageA_l14"/"results_sg_l14.csv")
rb=pd.read_csv(ROOT/"output"/"stageA"/"results_sg.csv")
norm=lambda s:str(s).strip().lower()
for d in (rl,rb,gb): d["k"]=d["path"].map(norm)
m=gb.merge(rl[["k","probe_score","is_positive"]].rename(columns={"is_positive":"l14"}),on="k",how="left")
m=m.merge(rb[["k","clip_score","is_positive"]].rename(columns={"is_positive":"b32"}),on="k",how="left")

def met(pred,y):
    pred=np.asarray(pred);y=np.asarray(y)
    tp=((pred==1)&(y==1)).sum();fp=((pred==1)&(y==0)).sum();fn=((pred==0)&(y==1)).sum();tn=((pred==0)&(y==0)).sum()
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn);f=2*p*r/max(1e-9,p+r)
    return dict(F1=f,P=p,R=r,Acc=(tp+tn)/len(y),TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn))

y=m.broad_label_final.values
print(f"broad-definition GT: {int(y.sum())} pos / {int((y==0).sum())} neg")
print("\n=== performance on the broad-definition GT ===")
for name,col in [("B32","b32"),("L14probe","l14")]:
    r=met(m[col].values,y)
    print(f"{name:<8}: F1={r['F1']:.3f} P={r['P']:.2f} R={r['R']:.2f} Acc={r['Acc']:.2f}  (TP{r['TP']} FP{r['FP']} FN{r['FN']} TN{r['TN']})")

print("\n=== compared with the strict GT (earlier) ===")
ys=m.human_label.values
for name,col in [("B32","b32"),("L14probe","l14")]:
    r=met(m[col].values,ys)
    print(f"{name:<8} strict: F1={r['F1']:.3f} (P={r['P']:.2f} R={r['R']:.2f})")

# still misclassified under the broad definition
print("\n=== L14 still misclassified under the broad definition ===")
for _,r in m.iterrows():
    if int(r.l14)!=int(r.broad_label_final):
        k="miss" if r.broad_label_final==1 else "false alarm"
        print(f"  cell{int(r.cell_id)} [{k}] broadGT={int(r.broad_label_final)} L14={int(r.l14)}(p{r.probe_score:.2f}) {r.note}")
