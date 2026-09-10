"""Precise geometry-first - controlled-variable version of step1+2 (pyenv).
Facing judgement = call step2_project.main directly (the same sector-ray code, zero changes);
only the input is switched from "colonnade-positive views" to "all views" → the projected flag of step2a_points = precise facing.
Outputs precise/facing_views_{city}.csv + timing.
"""
import sys, re, os, time
from pathlib import Path
import pandas as pd

ROOT=Path(r"D:\Claude\SVI_FFW"); PRE=ROOT/"output"/"detection_geomfirst"/"precise"; PRE.mkdir(parents=True,exist_ok=True)
sys.path.insert(0, str(ROOT/"output"/"step2_projection"))
import step2_project as P1
import geopandas as gpd

NAME_RE=re.compile(r"^(\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_\d{6}_baseheading[-\d.]+_viewheading([-\d.]+)_(\d)\.jpg$")
SVI={"sg":ROOT/"SVI"/"Singapore"/"Singapore SVI","bo":ROOT/"SVI"/"Bologna"/"Bologna SVI"}

P1.OUT=PRE   # redirect output; existing outputs are untouched
timing={}
for city in ["sg","bo"]:
    av=PRE/f"allviews_{city}.csv"
    if not av.exists():
        rows=[]
        for f in os.scandir(SVI[city]):
            m=NAME_RE.match(f.name)
            if m: rows.append((int(m.group(1)),int(m.group(5)),float(m.group(4)),float(m.group(2)),float(m.group(3))))
        pd.DataFrame(rows,columns=["pid","view","viewheading","lon","lat"]).to_csv(av,index=False)
        print(f"[{city}] allviews: {len(rows)}",flush=True)
    P1.CFG[city]["pos"]=av
    t0=time.time(); print(f"\n===== {city.upper()} precise facing (same code as Step2) =====",flush=True)
    P1.main(city)
    timing[city]=time.time()-t0
    pts=gpd.read_file(PRE/f"step2a_points_{city}.gpkg")
    fac=pts[pts.projected==1][["pid","view","viewheading"]].copy()
    allv=pd.read_csv(av)
    fac=fac.merge(allv[["pid","view","lon","lat"]],on=["pid","view"])
    fac.to_csv(PRE/f"facing_views_{city}.csv",index=False)
    print(f"[{city}] precise facing views: {len(fac)}/{len(allv)} ({len(fac)/len(allv)*100:.1f}%)  geometry time {timing[city]/60:.1f} min",flush=True)
pd.Series(timing).to_csv(PRE/"_geom_timing.csv")
print("\n[done facing]",flush=True)
