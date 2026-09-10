# -*- coding: utf-8 -*-
"""Step3x: citywide shadow share on a land basis (mask = DEM>0), plus the arcade-family (8-11) total.
Re-reads the 77 Category tiles (clipped to the 4000 core) + the matching DEM windows. pyenv."""
import numpy as np, rasterio, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OF = Path(r"D:\Claude\SVI_FFW\TIF\output_folder")
DEM = r"D:\Claude\SVI_FFW\TIF\SUB_SG_Polygon_DEM_1m.tif"
OUT = Path(r"D:\Claude\SVI_FFW\output\step3_adsm")
TILE = 4000; CITY_W, CITY_H = 44000, 27000
NAMES={0:"sunlit",1:"building",2:"vegetation",3:"bldg+veg",4:"ldsm",5:"bldg+ldsm",
       6:"veg+ldsm",7:"bldg+veg+ldsm",8:"arcade(strip sheltered)",9:"veg+arcade",
       10:"ldsm+arcade",11:"veg+ldsm+arcade",12:"building footprint(not shadow)"}

tiles = sorted([d for d in OF.iterdir() if d.is_dir() and (d/f"Category_{d.name}.tif").exists()],
               key=lambda d: tuple(map(int, d.name.split("_"))))
counts = None
n_land = 0
with rasterio.open(DEM) as rd:
    for d in tiles:
        x, y = map(int, d.name.split("_"))
        cw = min(TILE, CITY_W - x); ch = min(TILE, CITY_H - y)
        dem = rd.read(1, window=rasterio.windows.Window(x, y, cw, ch))
        land = (dem > 0) & np.isfinite(dem)
        n_land += int(land.sum())
        with rasterio.open(d/f"Category_{d.name}.tif") as r:
            nb = r.count
            if counts is None: counts = np.zeros((nb, 13), dtype=np.int64)
            for b in range(1, nb+1):
                a = r.read(b)[:ch, :cw][land]
                u, c = np.unique(a, return_counts=True)
                for v, n in zip(u.tolist(), c.tolist()):
                    if v < 13: counts[b-1, v] += n
print(f"land pixels (DEM>0) = {n_land:,} ({n_land/1e6:.1f} km2, full raster 1,188 km2)")

with open(OUT/"step3_city_class_counts_land.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["hour"] + [f"{k}:{NAMES[k]}" for k in range(13)] + ["arcade_family(8-11)", "shadow_pct_land(1-11)/(0-11)"])
    for b in range(counts.shape[0]):
        row = counts[b]
        denom = int(row[0:12].sum()); shade = int(row[1:12].sum()); fam = int(row[8:12].sum())
        pct = (shade/denom*100) if denom else 0.0
        w.writerow([f"{b:02d}:00"] + [int(v) for v in row] + [fam, f"{pct:.2f}"])
print("CSV: step3_city_class_counts_land.csv")

print(f"{'hour':>6} {'land shd%':>9} {'0 sunlit':>12} {'1 bldg':>12} {'2 veg':>12} {'4 linkway':>9} {'fam 8-11':>9} {'12 footprint':>12}")
for b in range(counts.shape[0]):
    if not (7 <= b <= 19): continue
    row = counts[b]; denom = int(row[0:12].sum()); shade = int(row[1:12].sum()); fam = int(row[8:12].sum())
    pct = (shade/denom*100) if denom else 0.0
    print(f"{b:>4}:00 {pct:>8.2f} {row[0]:>12,} {row[1]:>12,} {row[2]:>12,} {row[4]:>9,} {fam:>9,} {row[12]:>12,}")
print(f"class 12 (within land) = {int(counts[12,12]):,}")

hours = list(range(counts.shape[0]))
denoms = counts[:, 0:12].sum(axis=1).astype(float)
with np.errstate(divide="ignore", invalid="ignore"):
    total = np.where(denoms > 0, counts[:, 1:12].sum(axis=1)/denoms*100, 0)
    veg   = np.where(denoms > 0, (counts[:,2]+counts[:,3]+counts[:,6]+counts[:,7]+counts[:,9]+counts[:,11])/denoms*100, 0)
    bld   = np.where(denoms > 0, (counts[:,1]+counts[:,3]+counts[:,5]+counts[:,7])/denoms*100, 0)
    arc   = np.where(denoms > 0, counts[:, 8:12].sum(axis=1)/denoms*100, 0)
    ld    = np.where(denoms > 0, (counts[:,4]+counts[:,5]+counts[:,6]+counts[:,7]+counts[:,10]+counts[:,11])/denoms*100, 0)
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(hours, total, "k-", lw=2.5, label="Total shaded (classes 1-11)")
ax.plot(hours, bld, color="#9a9a9a", lw=1.8, label="Building involved (1/3/5/7)")
ax.plot(hours, veg, color="#74c476", lw=1.8, label="Vegetation involved (2/3/6/7/9/11)")
ax.plot(hours, ld, color="#fdae6b", lw=1.5, label="Linkway involved (4/5/6/7/10/11)")
ax.plot(hours, arc, color="#d40000", lw=1.5, label="Arcade family (8-11)")
ax.set_xlabel("Hour of day (2026-03-01, SGT)")
ax.set_ylabel("Share of LAND area excl. building footprint (%)")
ax.set_title("Singapore citywide shadow share by hour — land-mask denominator (DEM>0)")
ax.set_xticks(range(0, 24, 2)); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(OUT/"step3_city_shadow_share.png", dpi=150)
print("step3_city_shadow_share.png (land-based) updated")
