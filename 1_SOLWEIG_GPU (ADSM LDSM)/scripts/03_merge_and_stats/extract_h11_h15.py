# -*- coding: utf-8 -*-
"""Extract 12:00 (band 13) and 13:00 (band 14) from the 24-band merged
Category / Shadow rasters into single-band files beside Category_2pm_h14.tif,
same profile (LZW, tiled 512, predictor 2/3).  Band k = hour k-1 (verified:
band 15 == Category_2pm_h14.tif pixel-for-pixel).  Also counts pixels per
Category class (area_px) and Shadow < 0.5 pixels per hour, and re-counts the
14:00 files to check against tab1_stats (area_px / shadow_lt05_m2)."""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, rasterio
from rasterio.windows import Window
T0 = time.time()
BASE = 'D:/Claude/SVI_FFW/TIF_shadow_newarcade/merge_images'
HOURS = {12: ('11am', 'h11'), 16: ('3pm', 'h15')}      # band -> name parts (band = hour + 1)
out = {}
for nm, pred in [('Category', 2), ('Shadow', 3)]:
    src = rasterio.open(f'{BASE}/{nm}/{nm}_merged.tif')
    ref = rasterio.open(f'{BASE}/{nm}_2pm_h14.tif')
    prof = ref.profile.copy()
    prof.update(count=1, compress='lzw', tiled=True, blockxsize=512, blockysize=512, predictor=pred, BIGTIFF='IF_SAFER', interleave='band')
    dsts = {b: rasterio.open(f'{BASE}/{nm}_{HOURS[b][0]}_{HOURS[b][1]}.tif', 'w', **prof) for b in HOURS}
    cnt = {b: np.zeros(256, np.int64) for b in HOURS}; lt05 = {b: 0 for b in HOURS}
    cnt14 = np.zeros(256, np.int64); lt05_14 = 0
    H, W = src.height, src.width
    for r0 in range(0, H, 512):
        win = Window(0, r0, W, min(512, H - r0))
        arr = src.read(list(HOURS.keys()), window=win)
        for i, b in enumerate(HOURS):
            dsts[b].write(arr[i], 1, window=win)
            if nm == 'Category':
                cnt[b] += np.bincount(arr[i].ravel(), minlength=256)
            else:
                lt05[b] += int((arr[i] < 0.5).sum())
        a14 = ref.read(1, window=win)
        if nm == 'Category':
            cnt14 += np.bincount(a14.ravel(), minlength=256)
        else:
            lt05_14 += int((a14 < 0.5).sum())
        if (r0 // 512) % 10 == 0:
            print(f'  {nm} rows {r0}/{H} | {time.time()-T0:.0f}s', flush=True)
    for b in HOURS:
        dsts[b].close()
    if nm == 'Category':
        for b in HOURS:
            out[f'area_px_{HOURS[b][1]}'] = {str(k): int(v) for k, v in enumerate(cnt[b]) if v > 0}
        out['area_px_h14_recount'] = {str(k): int(v) for k, v in enumerate(cnt14) if v > 0}
    else:
        for b in HOURS:
            out[f'shadow_lt05_m2_{HOURS[b][1]}'] = lt05[b]
        out['shadow_lt05_m2_h14_recount'] = lt05_14
    print(f'{nm} done | {time.time()-T0:.0f}s', flush=True)
OLD = json.load(open('hourly_counts.json')); OLD.update(out); json.dump(OLD, open('hourly_counts.json', 'w'), indent=1)
# (release note) the original script cross-checked the 14:00 recount against the statistics JSON of the paper
# figure folder here; that workstation-specific consistency check was removed from the released version.
for h in ['h11', 'h15', 'h14_recount']:
    c = out[f'area_px_{h}']; g = lambda ks: sum(c.get(str(k), 0) for k in ks)
    print(f"{h}: building {g([1]):,} veg {g([2,3]):,} ldsm {g([4,5,6,7]):,} arcade {g([8,9,10,11]):,} sun {g([0]):,} footprint {g([12]):,}")
print(f'DONE | {time.time()-T0:.0f}s')
