# -*- coding: utf-8 -*-
"""Add key-hour pie charts below the "citywide shadow share" line chart. Reads the precomputed land count table (does not rerun the tiles).
Denominator = land excl. building footprint (classes 0-11). Output overwrites step3_city_shadow_share.png. pyenv."""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CSV = r"D:\Claude\SVI_FFW\output\step3_adsm\step3_city_class_counts_land.csv"
PAL = {1:'#9a9a9a',2:'#74c476',3:'#4a7f4a',4:'#fdae6b',8:'#d40000'}

hours, C = [], {k: [] for k in range(13)}
with open(CSV, encoding="utf-8-sig") as f:
    rd = csv.reader(f); next(rd)
    for row in rd:
        hours.append(int(row[0][:2]))
        for k in range(13): C[k].append(int(row[1+k]))
hours = np.array(hours); cnt = {k: np.array(C[k], float) for k in range(13)}
denom = sum(cnt[k] for k in range(12)); denom = np.where(denom > 0, denom, 1)
pct = {k: cnt[k]/denom*100 for k in range(13)}
total = sum(pct[k] for k in range(1, 12))
bld = pct[1]+pct[3]+pct[5]+pct[7]
veg = pct[2]+pct[3]+pct[6]+pct[7]+pct[9]+pct[11]
lkw = pct[4]+pct[5]+pct[6]+pct[7]+pct[10]+pct[11]
arc = pct[8]+pct[9]+pct[10]+pct[11]

fig = plt.figure(figsize=(13, 11))
gs = GridSpec(2, 4, figure=fig, height_ratios=[2.5, 1.1], hspace=0.34, wspace=0.12,
              top=0.93, bottom=0.13, left=0.075, right=0.975)
ax = fig.add_subplot(gs[0, :])
ax.plot(hours, total, 'k-', lw=2.6, label="Total shaded (classes 1-11)")
ax.plot(hours, bld, color='#9a9a9a', lw=1.9, label="Building involved (1/3/5/7)")
ax.plot(hours, veg, color='#74c476', lw=1.9, label="Vegetation involved (2/3/6/7/9/11)")
ax.plot(hours, lkw, color='#fdae6b', lw=1.6, label="Linkway involved (4/5/6/7/10/11)")
ax.plot(hours, arc, color='#d40000', lw=1.6, label="Arcade family (8-11)")
ax.set_xlim(0,23); ax.set_xticks(range(0,24,2)); ax.set_ylim(bottom=-2)
ax.set_xlabel("Hour of day (2026-03-01, SGT)"); ax.set_ylabel("Share of LAND area excl. building footprint (%)")
ax.set_title("Singapore citywide shadow share by hour — land-mask denominator (DEM>0)", fontsize=13, fontweight='bold')
ax.grid(alpha=0.25); ax.legend(fontsize=9, loc='upper right')

# ---- Key-hour pie charts (composition of the 5 mutually exclusive groups within the shaded area, total = 100%) ----
G_COL=[PAL[1],PAL[2],PAL[3],PAL[4],PAL[8]]
G_LAB=['1 building','2 vegetation','3 bldg+veg','4-7 linkway-inv.','8-11 arcade fam.']
def groups(h):
    return np.array([pct[1][h],pct[2][h],pct[3][h],
                     pct[4][h]+pct[5][h]+pct[6][h]+pct[7][h],
                     pct[8][h]+pct[9][h]+pct[10][h]+pct[11][h]])
for i,h in enumerate([8,12,14,18]):
    axp=fig.add_subplot(gs[1,i]); g=groups(h); g=g/g.sum()*100
    wed,_=axp.pie(g,colors=G_COL,startangle=90,counterclock=False,wedgeprops=dict(edgecolor='white',linewidth=0.6))
    for w,val in zip(wed,g):
        if val>=5:
            a=np.deg2rad((w.theta1+w.theta2)/2)
            axp.text(0.6*np.cos(a),0.6*np.sin(a),f"{val:.0f}%",ha='center',va='center',fontsize=9.5,fontweight='bold',color='white' if val>12 else '#222')
    axp.set_title(f"{h:02d}:00\nshaded {total[h]:.0f}% of land",fontsize=10)
    cap=(f"bldg {g[0]:.1f}%  veg {g[1]:.1f}%  b+v {g[2]:.1f}%\nlinkway {g[3]:.2f}%  arcade {g[4]:.2f}%")
    axp.text(0,-1.32,cap,ha='center',va='top',fontsize=8.0,color='#222')
fig.legend(handles=[plt.matplotlib.patches.Patch(fc=G_COL[k],label=G_LAB[k]) for k in range(5)],
           loc='lower center',ncol=5,fontsize=9,frameon=True,bbox_to_anchor=(0.5,0.01),
           title="Shadow-type composition within the shaded area at key hours  (each pie = 100% of shade)")
fig.legends[-1].get_title().set_fontsize(10.5); fig.legends[-1].get_title().set_fontweight('bold')

plt.savefig(r"D:\Claude\SVI_FFW\output\step3_adsm\step3_city_shadow_share.png", dpi=150)
print("wrote step3_city_shadow_share.png")
