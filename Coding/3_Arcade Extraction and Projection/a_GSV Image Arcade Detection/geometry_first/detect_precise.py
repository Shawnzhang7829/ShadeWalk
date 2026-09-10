"""Precise geometry-first - step3 (torch/QGIS): run the CLIP colonnade probe only on the precise facade-facing views (same probe, same threshold).
v2: incremental writing (flush every batch) + resume from checkpoint (per pid,view) - no progress is lost if the process is killed.
All scores are written to detect_scores_{city}.csv; positives are derived into arcade_precise_{city}_positive.csv.
Usage: run_qgis.bat detect_precise.py [sg] [bo]
"""
import sys, re, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib
from PIL import Image
ROOT=Path(r"D:\Claude\SVI_FFW"); PRE=ROOT/"output"/"detection_geomfirst"/"precise"; L=ROOT/"output"/"stageA_l14"
device="cuda" if torch.cuda.is_available() else "cpu"; L14="openai/clip-vit-large-patch14"
CFG={"sg":dict(svi=ROOT/"SVI"/"Singapore"/"Singapore SVI",arc=ROOT/"output"/"strict_gt"/"sg_colonnade_final.joblib",thr=0.45),
     "bo":dict(svi=ROOT/"SVI"/"Bologna"/"Bologna SVI",arc=L/"bo_probe.joblib",thr=0.40)}
NAME_RE=re.compile(r"^(\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_\d{6}_baseheading[-\d.]+_viewheading([-\d.]+)_(\d)\.jpg$")
BATCH=64

def main(city):
    c=CFG[city]; arcp=joblib.load(c["arc"]); thr=c["thr"]
    fv=pd.read_csv(PRE/f"facing_views_{city}.csv")
    want={(int(r.pid),int(r.view)) for r in fv.itertuples()}
    sc_p=PRE/f"detect_scores_{city}.csv"; done=set()
    if sc_p.exists():
        d0=pd.read_csv(sc_p,usecols=["pid","view"])
        done=set(map(tuple,d0.values.tolist())); print(f"[{city}] resume: {len(done)} already scored",flush=True)
    todo_keys=want-done
    print(f"[{city}] facing views {len(fv)}, to score {len(todo_keys)}, indexing paths …",flush=True)
    idx={}
    if todo_keys:
        for f in os.scandir(c["svi"]):
            m=NAME_RE.match(f.name)
            if m and (int(m.group(1)),int(m.group(5))) in todo_keys:
                idx[(int(m.group(1)),int(m.group(5)))]=(f.path,float(m.group(4)),float(m.group(2)),float(m.group(3)))
    rows=[(pid,vw,*idx[(pid,vw)]) for (pid,vw) in todo_keys if (pid,vw) in idx]
    print(f"[{city}] images found {len(rows)}",flush=True)
    if rows:
        from transformers import CLIPProcessor, CLIPModel
        clip=CLIPModel.from_pretrained(L14).to(device).eval(); proc=CLIPProcessor.from_pretrained(L14)
        def embed(ims):
            inp=proc(images=ims,return_tensors="pt").to(device)
            with torch.no_grad():
                vo=clip.vision_model(pixel_values=inp["pixel_values"]); f=clip.visual_projection(vo.pooler_output)
            return (f/f.norm(dim=-1,keepdim=True)).cpu().numpy()
        new=not sc_p.exists()
        fout=open(sc_p,"a",encoding="utf-8",newline="")
        if new: fout.write("pid,view,viewheading,lon,lat,probe_prob\n")
        t0=time.time(); npos=0
        for i in range(0,len(rows),BATCH):
            bt=rows[i:i+BATCH]; ims=[]; meta=[]
            for pid,vw,path,vh,lon,lat in bt:
                try: ims.append(Image.open(path).convert("RGB")); meta.append((pid,vw,vh,lon,lat))
                except: pass
            if not ims: continue
            pr=arcp.predict_proba(embed(ims))[:,1]
            for (pid,vw,vh,lon,lat),p in zip(meta,pr):
                fout.write(f"{pid},{vw},{vh},{lon},{lat},{p:.4f}\n"); npos+=int(p>=thr)
            fout.flush()                                   # flush every batch: nothing is lost if killed
            if (i//BATCH)%50==0:
                rate=(i+len(bt))/(time.time()-t0)
                print(f"  [{city}] {i+len(bt)}/{len(rows)} newpos={npos} {rate:.0f}img/s ETA{(len(rows)-i)/max(rate,1)/60:.0f}min",flush=True)
        fout.close()
        print(f"[{city}] scoring done {(time.time()-t0)/60:.1f} min",flush=True)
    # finalize (idempotent)
    d=pd.read_csv(sc_p).drop_duplicates(subset=["pid","view"],keep="last")
    pos=d[d.probe_prob>=thr].copy(); pos["is_colonnade"]=1
    pos.to_csv(PRE/f"arcade_precise_{city}_positive.csv",index=False)
    print(f"[{city}] === FINAL ===  scored {len(d)} → colonnade positives {len(pos)} (panos involved {pos.pid.nunique()})",flush=True)

if __name__=="__main__":
    for city in (sys.argv[1:] or ["sg","bo"]): main(city)
