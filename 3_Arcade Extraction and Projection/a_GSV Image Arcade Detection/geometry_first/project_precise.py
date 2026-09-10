"""Precise geometry-first version - vector generation: run the existing Step2 chain (project -> merge -> buffer_split)
on arcade_precise_{city}_positive.csv, writing vectors to precise/vectors/ (no existing output is overwritten).
Exactly the same code as production; only the input and output directories differ."""
import sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Claude\SVI_FFW\output\step2_projection")
import step2_project as P1, step2_merge as P2, step2_buffer_split as P3
PRE=Path(r"D:\Claude\SVI_FFW\output\detection_geomfirst\precise")
VEC=PRE/"vectors"; VEC.mkdir(parents=True,exist_ok=True)
P1.OUT=VEC; P2.OUT=VEC; P3.OUT=VEC
for city in ["sg","bo"]:
    P1.CFG[city]["pos"]=PRE/f"arcade_precise_{city}_positive.csv"
    t0=time.time(); print(f"\n########## {city.upper()} ##########",flush=True)
    print("--- project ---",flush=True); P1.main(city)
    print("--- merge ---",flush=True);   P2.main(city)
    print("--- buffer ---",flush=True);  P3.main(city)
    print(f"[{city}] chain done {(time.time()-t0)/60:.1f} min",flush=True)
print("\n[done] precise vectors written to precise/vectors/",flush=True)
