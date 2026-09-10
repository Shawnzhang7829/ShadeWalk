"""Unified colonnade/arcade detection - a single script runs the city-wide detection. (torch / QGIS-python environment)

Replaces the 4 old scripts: run_clip_inference (B/32, superseded by L/14, deprecated), rescore_l14,
extend_broad_13 (re-scoring) and run_colonnade_v2. Logically equivalent to the "existing output"
(which, after re-scoring, already covered all 4 views).

Pipeline (streamed pano by pano; each image is L/14-encoded only once):
  for pano in whole city (all 4 views):
     feats = L/14( view1..4 )
     (1) broad screen: broad_score = broad_probe(feats);  threshold by region_tier → any view above it ⇒ the pano is a candidate
     (2) colonnade: (candidate panos only) arcade_prob = arcade_probe(feats);  ≥ arc_thr ⇒ that view is positive
  The facade-facing judgement is not done here (left to the Step2 geometry).

Thresholds: broad screen per Tier (read from stageA_l14/thresholds.json); colonnade SG 0.45 / BO 0.40.
SG broad screen = sg_probe, colonnade = sg_colonnade_final (two different probes); BO uses the same bo_probe for both stages, only the thresholds differ.

Incremental writing + resume from checkpoint (per pano).
Usage: run_qgis.bat detect_arcade.py sg [smoke_N]      (with smoke_N only N Tier1 panos are run as a check)
Output: detection/detect_{city}.csv           all views (broad/arcade scores + decisions)
        detection/arcade_{city}_positive.csv  positive views (same schema as colonnade_v2_positive, for Step2)
"""
from __future__ import annotations
import sys, re, os, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib
from PIL import Image

ROOT=Path(r"D:\Claude\SVI_FFW"); DET=ROOT/"output"/"detection"; L=ROOT/"output"/"stageA_l14"
device="cuda" if torch.cuda.is_available() else "cpu"; L14="openai/clip-vit-large-patch14"
CFG={
 "sg":dict(svi=ROOT/"SVI"/"Singapore"/"Singapore SVI",
           broad=L/"sg_probe.joblib", arc=ROOT/"output"/"strict_gt"/"sg_colonnade_final.joblib", arc_thr=0.45),
 "bo":dict(svi=ROOT/"SVI"/"Bologna"/"Bologna SVI",
           broad=L/"bo_probe.joblib", arc=L/"bo_probe.joblib", arc_thr=0.40),
}
NAME_RE=re.compile(r"^(\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_\d{6}_baseheading[-\d.]+_viewheading([-\d.]+)_(\d)\.jpg$")
BATCH_PANOS=64; COLS="pid,view,viewheading,lon,lat,region_tier,broad_score,is_candidate,arcade_prob,is_arcade\n"

def build_index(svi, pids):
    idx={}
    for f in os.scandir(svi):
        m=NAME_RE.match(f.name)
        if not m: continue
        pid=int(m.group(1))
        if pid not in pids: continue
        idx.setdefault(pid,{})[int(m.group(5))]=(f.path,float(m.group(4)),float(m.group(2)),float(m.group(3)))
    return idx

def main(city):
    c=CFG[city]
    thr_b=json.loads((L/"thresholds.json").read_text())[city]
    thr_b={int(k) if str(k).isdigit() else k:v for k,v in thr_b.items()}   # tier->thr
    broad=joblib.load(c["broad"]); arcp=joblib.load(c["arc"]); arc_thr=c["arc_thr"]
    tier=pd.read_csv(DET/f"pano_tier_{city}.csv").set_index("pid")["region_tier"].to_dict()

    todo=sorted(tier.keys())
    if len(sys.argv)>2:    # smoke: take only the first N Tier1 panos (always candidates; checks the arcade path)
        n=int(sys.argv[2]); todo=[p for p in todo if tier.get(p)==1][:n]; print(f"[{city}] SMOKE {len(todo)} Tier1 panos")
    out_p=DET/f"detect_{city}.csv"; done=set()
    if out_p.exists() and len(sys.argv)<=2:
        done=set(pd.read_csv(out_p,usecols=["pid"])["pid"].unique()); print(f"[{city}] resume {len(done)}")
    todo=[p for p in todo if p not in done]
    if not todo: print("nothing to do"); return finalize(city)
    print(f"[{city}] building 4-view index …"); idx=build_index(c["svi"], set(todo))
    todo=[p for p in todo if len(idx.get(p,{}))==4]
    print(f"[{city}] panos with all 4 views: {len(todo)}")

    from transformers import CLIPProcessor, CLIPModel
    clip=CLIPModel.from_pretrained(L14).to(device).eval(); proc=CLIPProcessor.from_pretrained(L14)
    def embed(imgs):
        inp=proc(images=imgs,return_tensors="pt").to(device)
        with torch.no_grad():
            vo=clip.vision_model(pixel_values=inp["pixel_values"]); f=clip.visual_projection(vo.pooler_output)
        return (f/f.norm(dim=-1,keepdim=True)).cpu().numpy()

    fout=open(out_p,"a",encoding="utf-8",newline="");
    if not done: fout.write(COLS)
    t0=time.time(); ncand=0; narc=0; nproc=0
    for bi in range(0,len(todo),BATCH_PANOS):
        bp=todo[bi:bi+BATCH_PANOS]; imgs=[]; meta=[]
        for pid in bp:
            v=idx[pid]; ims={k:_safe(v[k][0]) for k in (1,2,3,4)}
            if any(im is None for im in ims.values()): continue
            for k in (1,2,3,4):
                imgs.append(ims[k]); meta.append((pid,k,v[k][1],v[k][2],v[k][3]))
        if not imgs: nproc+=len(bp); continue
        feats=[]
        for j in range(0,len(imgs),64): feats.append(embed(imgs[j:j+64]))
        feats=np.concatenate(feats)
        bs=broad.predict_proba(feats)[:,1]; ap=arcp.predict_proba(feats)[:,1]
        # group by pid → candidate (any view passes the Tier broad-screen threshold)
        byp={}
        for i,(pid,k,vh,lon,lat) in enumerate(meta): byp.setdefault(pid,[]).append(i)
        for pid,ii in byp.items():
            t=tier.get(pid,3); cand=any(bs[i]>=thr_b[t] for i in ii); ncand+=int(cand)
            for i in ii:
                pidv,k,vh,lon,lat=meta[i]
                isarc=int(cand and ap[i]>=arc_thr); narc+=isarc
                fout.write(f"{pidv},{k},{vh},{lon},{lat},{t},{bs[i]:.4f},{int(cand)},{ap[i]:.4f},{isarc}\n")
        fout.flush(); nproc+=len(bp)
        rate=nproc/(time.time()-t0); eta=(len(todo)-nproc)/max(rate,.1)/60
        print(f"  [{city}] {nproc}/{len(todo)} panos  cand={ncand} arc_views={narc}  {rate:.1f}pano/s ETA{eta:.0f}min",flush=True)
    fout.close(); print(f"[{city}] done {(time.time()-t0)/60:.1f}min"); finalize(city)

def _safe(p):
    try: return Image.open(p).convert("RGB")
    except: return None

def finalize(city):
    d=pd.read_csv(DET/f"detect_{city}.csv").drop_duplicates(subset=["pid","view"],keep="last")
    d.to_csv(DET/f"detect_{city}.csv",index=False)
    pos=d[d.is_arcade==1][["pid","view","viewheading","lon","lat","arcade_prob"]].copy()
    pos.columns=["pid","view","viewheading","lon","lat","probe_prob"]; pos["is_colonnade"]=1
    pos.to_csv(DET/f"arcade_{city}_positive.csv",index=False)
    cand_panos=d[d.is_candidate==1].pid.nunique()
    print(f"\n[{city}] === FINAL ===  views {len(d)}  candidate panos {cand_panos}  positive views {len(pos)} (panos involved {pos.pid.nunique()})")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "sg")
