# -*- coding: utf-8 -*-
"""Step3w: citywide verification + shadow share + overview maps. pyenv. Run after the citywide run has finished.
  (1) hourly 13-class pixel counts (each tile clipped to its 4000x4000 core, 100 px overlap band removed) -> CSV
  (2) shadow share: shadow (1-11) / (0-11) (denominator excludes class 12 building footprint); shares of the main classes too
  (3) class 12 reconciliation (within the computed tile coverage) + magnitude of strip class 8
  (4) figures: citywide category overview h09/h14/h18 (1/16 downsampled, colour palette) + shadow share curves
"""
import numpy as np, rasterio, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OF = Path(r"D:\Claude\SVI_FFW\TIF\output_folder")
OUT = Path(r"D:\Claude\SVI_FFW\output\step3_adsm")
TILE, F = 4000, 16          # core size / downsampling factor
CITY_W, CITY_H = 44000, 27000
NAMES={0:"sunlit",1:"building",2:"vegetation",3:"bldg+veg",4:"ldsm",5:"bldg+ldsm",
       6:"veg+ldsm",7:"bldg+veg+ldsm",8:"arcade(strip sheltered)",9:"veg+arcade",
       10:"ldsm+arcade",11:"veg+ldsm+arcade",12:"building footprint(not shadow)"}
PALRGB = {0:(255,255,204),1:(154,154,154),2:(116,196,118),3:(74,127,74),
          4:(253,174,107),5:(176,112,64),6:(143,174,90),7:(107,91,58),
          8:(212,0,0),9:(192,101,192),10:(224,128,32),11:(123,63,160),12:(90,90,90)}

tiles = sorted([d for d in OF.iterdir() if d.is_dir() and (d/f"Category_{d.name}.tif").exists()],
               key=lambda d: tuple(map(int, d.name.split("_"))))
print(f"tiles with finished category output: {len(tiles)} tile(s)")

counts = None  # (24, 13)
mosaic = {hb: np.zeros((CITY_H//F, CITY_W//F), dtype=np.uint8) for hb in (10, 15, 19)}
covered = np.zeros((CITY_H//F, CITY_W//F), dtype=bool)

for d in tiles:
    x, y = map(int, d.name.split("_"))
    with rasterio.open(d/f"Category_{d.name}.tif") as r:
        nb = r.count
        if counts is None: counts = np.zeros((nb, 13), dtype=np.int64)
        cw = min(TILE, CITY_W - x); ch = min(TILE, CITY_H - y)
        for b in range(1, nb+1):
            a = r.read(b)[:ch, :cw]
            u, c = np.unique(a, return_counts=True)
            for v, n in zip(u.tolist(), c.tolist()):
                if v < 13: counts[b-1, v] += n
            if b in mosaic:
                sm = a[::F, ::F]
                hh = min(sm.shape[0], mosaic[b].shape[0] - y//F)
                ww = min(sm.shape[1], mosaic[b].shape[1] - x//F)
                mosaic[b][y//F : y//F + hh, x//F : x//F + ww] = sm[:hh, :ww]
        a0 = np.ones((ch, cw), dtype=bool)[::F, ::F]
        hh = min(a0.shape[0], covered.shape[0] - y//F)
        ww = min(a0.shape[1], covered.shape[1] - x//F)
        covered[y//F : y//F + hh, x//F : x//F + ww] = True

# ---- CSV + shares ----
with open(OUT/"step3_city_class_counts.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["hour"] + [f"{k}:{NAMES[k]}" for k in range(13)] + ["shadow_pct(1-11)/(0-11)"])
    for b in range(counts.shape[0]):
        row = counts[b]
        denom = int(row[0:12].sum()); shade = int(row[1:12].sum())
        pct = (shade/denom*100) if denom else 0.0
        w.writerow([f"{b:02d}:00"] + [int(v) for v in row] + [f"{pct:.2f}"])
print("CSV: step3_city_class_counts.csv")

print(f"{'hour':>6} {'shade%':>9} {'0 sunlit':>12} {'1 bldg':>12} {'2 veg':>12} {'4 linkway':>9} {'8 arcade':>9} {'12 footprint':>12}")
for b in range(counts.shape[0]):
    h = b
    if not (7 <= h <= 19): continue
    row = counts[b]; denom = int(row[0:12].sum()); shade = int(row[1:12].sum())
    pct = (shade/denom*100) if denom else 0.0
    print(f"{h:>4}:00 {pct:>8.2f} {row[0]:>12,} {row[1]:>12,} {row[2]:>12,} {row[4]:>9,} {row[8]:>9,} {row[12]:>12,}")

n12 = int(counts[12, 12]) if counts.shape[0] > 12 else int(counts[10, 12])
print(f"class 12 (within computed tile coverage) = {n12:,} / citywide mask 80,208,935 (difference = footprint inside sparse skipped tiles)")

# ---- Overview maps ----
cmap = ListedColormap([tuple(v/255 for v in PALRGB[k]) for k in range(13)])
for hb, tag in ((10, "h09"), (15, "h14"), (19, "h18")):
    img = mosaic[hb].copy()
    fig, ax = plt.subplots(figsize=(16, 10))
    m = np.ma.masked_where(~covered, img)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=12, interpolation="nearest")
    ax.set_title(f"Singapore citywide shadow classification — {tag[1:]}:00 (13 classes, 1m, downsampled x{F})", fontsize=13)
    ax.axis("off")
    import matplotlib.patches as mp
    handles = [mp.Patch(color=tuple(v/255 for v in PALRGB[k]), label=f"{k} {NAMES[k]}") for k in range(13)]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUT/f"step3_city_category_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"step3_city_category_{tag}.png")

# ---- Shadow share curves ----
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
ax.set_xlabel("Hour of day (2026-03-01, SGT)"); ax.set_ylabel("Share of computed area excl. building footprint (%)")
ax.set_title("Singapore citywide shadow share by hour — final 13-class run")
ax.set_xticks(range(0, 24, 2)); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(OUT/"step3_city_shadow_share.png", dpi=150)
print("step3_city_shadow_share.png")
