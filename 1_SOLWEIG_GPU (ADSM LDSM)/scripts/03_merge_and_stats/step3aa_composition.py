# -*- coding: utf-8 -*-
"""Citywide "shadow composition by hour" figure: top = stacked area of the 11 mutually exclusive shadow classes (true share of each class; the stack sums to the total shaded fraction);
bottom = zoom on the small-share linkway and arcade families. Denominator = land area excl. building footprint (classes 0-11). pyenv."""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

CSV = r"D:\Claude\SVI_FFW\output\step3_adsm\step3_city_class_counts_land.csv"
PAL = {0:'#ffffcc',1:'#9a9a9a',2:'#74c476',3:'#4a7f4a',4:'#fdae6b',5:'#b07040',
       6:'#8fae5a',7:'#6b5b3a',8:'#d40000',9:'#c065c0',10:'#e08020',11:'#7b3fa0',12:'#5a5a5a'}
NM = {0:'sunlit',1:'building',2:'vegetation',3:'bldg+veg',4:'ldsm/linkway',5:'bldg+linkway',
      6:'veg+linkway',7:'bldg+veg+linkway',8:'arcade',9:'veg+arcade',10:'linkway+arcade',11:'veg+linkway+arcade'}

hours, C = [], {k: [] for k in range(13)}
with open(CSV, encoding="utf-8-sig") as f:
    rd = csv.reader(f); next(rd)
    for row in rd:
        hours.append(int(row[0][:2]))
        for k in range(13): C[k].append(int(row[1+k]))
hours = np.array(hours)
cnt = {k: np.array(C[k], float) for k in range(13)}
denom = sum(cnt[k] for k in range(12))          # land excl. footprint (= classes 0..11), constant
denom = np.where(denom > 0, denom, 1)
pct = {k: cnt[k] / denom * 100 for k in range(13)}
total = sum(pct[k] for k in range(1, 12))       # total shaded fraction

fig = plt.figure(figsize=(13, 16))
gs = GridSpec(3, 4, figure=fig, height_ratios=[2.3, 1.0, 1.6],
              top=0.88, bottom=0.11, left=0.075, right=0.975, hspace=0.55, wspace=0.18)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])

# ---- Top: stacked area of the 11 mutually exclusive classes ----
order = list(range(1, 12))                       # 1..11 stacked bottom to top
ax1.stackplot(hours, *[pct[k] for k in order],
              colors=[PAL[k] for k in order], labels=[f"{k} {NM[k]}" for k in order],
              edgecolor='white', linewidth=0.25)
ax1.plot(hours, total, color='k', lw=1.6, ls='--', label='total shaded (1-11)')
ax1.set_xlim(0,23); ax1.set_ylim(0,96); ax1.set_xticks(range(0,24,2))
ax1.set_ylabel("Share of land area excl. footprint (%)")
ax1.grid(alpha=0.25, axis='y')
# Legend moved above the plotting area (horizontal, does not cover the curves)
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=8, framealpha=0.95)
# Peak/trough annotations at key hours (offset to avoid overlapping curves/borders)
for h, dy in ((8, 8), (14, 8), (19, 8)):
    ax1.annotate(f"{total[h]:.0f}%", (h, total[h]), textcoords="offset points", xytext=(0, dy),
                 ha='center', fontsize=9, fontweight='bold')

# ---- Bottom: zoom on the small-share classes (linkway family / arcade family) ----
linkway = pct[4]+pct[5]+pct[6]+pct[7]
arcade  = pct[8]+pct[9]+pct[10]+pct[11]
ax2.plot(hours, linkway, color=PAL[4], lw=2.0, marker='o', ms=3, label='linkway-involved (4+5+6+7)')
ax2.plot(hours, arcade,  color=PAL[8], lw=2.0, marker='s', ms=3, label='arcade family (8+9+10+11)')
ax2.plot(hours, pct[8],  color=PAL[8], lw=1.0, ls=':', label='  of which arcade strip only (8)')
ax2.set_xlim(0,23); ax2.set_xticks(range(0,24,2)); ax2.set_ylim(0, max(linkway.max(),arcade.max())*1.25+0.01)
ax2.set_xlabel("Hour of day (2026-03-01, SGT)"); ax2.set_ylabel("Share of land\nexcl. footprint (%)")
ax2.set_title("zoom — small classes (covered linkway & arcade five-foot way)", fontsize=10.5, pad=8)
ax2.grid(alpha=0.25); ax2.legend(loc='upper right', ncol=1, fontsize=8, framealpha=0.95)

# ---- Third row: pie charts of the shadow-type composition within the shaded area at key hours (5 mutually exclusive groups) ----
G_LAB = ['1 building', '2 vegetation', '3 bldg+veg', '4-7 linkway-inv.', '8-11 arcade fam.']
G_COL = [PAL[1], PAL[2], PAL[3], PAL[4], PAL[8]]
def groups(h):
    return np.array([pct[1][h], pct[2][h], pct[3][h],
                     pct[4][h]+pct[5][h]+pct[6][h]+pct[7][h],
                     pct[8][h]+pct[9][h]+pct[10][h]+pct[11][h]])
SHORT = ['bldg', 'veg', 'b+v', 'linkway', 'arcade']
for i, h in enumerate([8, 12, 14, 18]):
    axp = fig.add_subplot(gs[2, i])
    g = groups(h); g = g / g.sum() * 100
    wed, _ = axp.pie(g, colors=G_COL, startangle=90, counterclock=False,
                     wedgeprops=dict(edgecolor='white', linewidth=0.6))
    # Label large wedges (>=5%) with their percentage inside the pie
    for w, val in zip(wed, g):
        if val >= 5:
            ang = np.deg2rad((w.theta1 + w.theta2) / 2)
            axp.text(0.6*np.cos(ang), 0.6*np.sin(ang), f"{val:.0f}%",
                     ha='center', va='center', fontsize=9.5, fontweight='bold',
                     color='white' if val > 12 else '#222')
    axp.set_title(f"{h:02d}:00\nshaded {total[h]:.0f}% of land", fontsize=10)
    # Below the pie: list the shares of all 5 parts (tiny wedges included; two decimals down to 0.01%)
    cap = (f"bldg {g[0]:.1f}%   veg {g[1]:.1f}%   b+v {g[2]:.1f}%\n"
           f"linkway {g[3]:.2f}%   arcade {g[4]:.2f}%")
    axp.text(0, -1.32, cap, ha='center', va='top', fontsize=8.0, color='#222')
leg = fig.legend(handles=[Patch(fc=G_COL[k], label=G_LAB[k]) for k in range(5)],
                 loc='lower center', ncol=5, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.005),
                 title="Shadow-type composition WITHIN the shaded area at key hours  (each pie = 100% of shade)")
leg.get_title().set_fontsize(11); leg.get_title().set_fontweight('bold')
fig.suptitle("Singapore citywide shadow composition by hour (mutually-exclusive classes, 2026-03-01)",
             fontsize=13.5, fontweight='bold', y=0.965)

plt.savefig(r"D:\Claude\SVI_FFW\output\step3_adsm\step3_city_shadow_composition.png", dpi=150)
print("wrote step3_city_shadow_composition.png")
# Also print the per-class share table at key hours
print(f"\n{'hr':>3} {'total':>6} " + " ".join(f"{NM[k][:6]:>7}" for k in range(1,12)))
for h in (8,10,12,14,16,18):
    print(f"{h:>3} {total[h]:>5.1f}% " + " ".join(f"{pct[k][h]:>6.2f}%" for k in range(1,12)))
