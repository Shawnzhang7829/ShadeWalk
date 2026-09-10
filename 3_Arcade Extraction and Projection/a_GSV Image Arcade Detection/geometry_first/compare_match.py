"""Match verdict: precise geometry-first vs current CLIP-first (per pid,view).
Universe note: OLD = broad-candidates ∩ arcade ∩ facing; NEW = facing ∩ arcade (no broad gate).
So compare on the SAME universe (candidate panos) -> expect ~identical; extras outside = broad-gate recoveries."""
import geopandas as gpd, pandas as pd
from pathlib import Path
ROOT=Path(r"D:\Claude\SVI_FFW\output"); PRE=ROOT/"detection_geomfirst"/"precise"
print(f"{'':4}{'NEW':>8}{'OLD':>8}{'∩agree':>8}{'onlyNEW':>9}{'onlyOLD':>9}{'Jaccard':>9}  (restricted to candidate panos)")
for c in ["sg","bo"]:
    new=pd.read_csv(PRE/f"arcade_precise_{c}_positive.csv")
    new_all=set(map(tuple,new[["pid","view"]].astype(int).values.tolist()))
    old=gpd.read_file(ROOT/"step2_projection"/f"step2a_segments_{c}.gpkg")
    old_set=set(map(tuple,old[["pid","view"]].astype(int).values.tolist()))
    cand=set(pd.read_csv(ROOT/"production_colonnade"/f"colonnade_{c}_v2.csv",usecols=["pid"])["pid"].unique())
    new_c={t for t in new_all if t[0] in cand}
    inter=new_c&old_set; uni=new_c|old_set
    print(f"{c.upper():4}{len(new_c):>8}{len(old_set):>8}{len(inter):>8}{len(new_c-old_set):>9}{len(old_set-new_c):>9}{len(inter)/len(uni):>9.4f}")
    extras=new_all-new_c
    print(f"     full NEW={len(new_all)}; outside candidate panos (= broad-screen misses recovered): {len(extras)}")
    oo=old_set-new_c
    if oo:
        # diagnose only-OLD: e.g. because v2 required all 4 views to be present
        sample=list(oo)[:6]; print(f"     only-OLD sample: {sample}")
