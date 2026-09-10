"""Distribution of colonnade-positive views by Tier (the region_tier of their pano)."""
import pandas as pd
from pathlib import Path
ROOT=Path(r"D:\Claude\SVI_FFW")
for c in ["sg","bo"]:
    pos=pd.read_csv(ROOT/"output"/"production_colonnade"/f"colonnade_{c}_positive.csv")
    res=pd.read_csv(ROOT/"output"/"stageA_l14"/f"results_{c}_l14.csv",usecols=["pid","region_tier"])
    pid_tier=res.drop_duplicates("pid").set_index("pid")["region_tier"].to_dict()
    pos["tier"]=pos["pid"].map(pid_tier).fillna(3).astype(int)  # views from the 1/3 re-scoring are assigned to Tier3
    print(f"\n=== {c.upper()} colonnade-positive views {len(pos)} ===")
    vc=pos["tier"].value_counts().sort_index()
    for t in [1,2,3]:
        n=int(vc.get(t,0)); print(f"  Tier{t}: {n} ({n/len(pos)*100:.1f}%)")
    # panos involved, by tier
    print(f"  (panos involved: {pos.pid.nunique()})")
