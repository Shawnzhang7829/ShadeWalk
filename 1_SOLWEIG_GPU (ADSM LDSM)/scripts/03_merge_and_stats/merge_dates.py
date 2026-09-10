# -*- coding: utf-8 -*-
"""Per-date merge + hourly extraction for the solstice/equinox SOLWEIG runs.
usage: python merge_dates.py <tile_output_folder> <merge_out_folder>
 1) 77 tiles (core 4000x4000, 100 px halo dropped) -> 24-band Shadow_merged.tif
    + Category_merged.tif in <merge_out_folder>  (same profile as
    merge_images/Shadow|Category/*_merged.tif)
 2) bands 12..16 (11:00..15:00) -> single-band Category_11am_h11.tif ...
    Shadow_3pm_h15.tif (same names/profile as the 14:00 extracts)
 3) writes _MERGE_OK with the band-count / tag checks; the orchestrator deletes
    the tile folder only when this marker exists.  pyenv 3.11.9 (rasterio)."""
import os, sys, time, json
import numpy as np, rasterio
from rasterio.windows import Window
OFD, MERGE = sys.argv[1], sys.argv[2]
NEW = 'D:/Claude/SVI_FFW/TIF_shadow_newarcade'
CORE, NB, NT = 4000, 24, 77
HOURS = {11: '11am_h11', 12: '12pm_h12', 13: '1pm_h13', 14: '2pm_h14', 15: '3pm_h15'}
PAL = {0: (255, 255, 204, 255), 1: (154, 154, 154, 255), 2: (116, 196, 118, 255), 3: (74, 127, 74, 255),
       4: (253, 174, 107, 255), 5: (176, 112, 64, 255), 6: (143, 174, 90, 255), 7: (107, 91, 58, 255),
       8: (212, 0, 0, 255), 9: (192, 101, 192, 255), 10: (224, 128, 32, 255), 11: (123, 63, 160, 255), 12: (90, 90, 90, 255)}
t0 = time.time()
os.makedirs(MERGE, exist_ok=True)
with rasterio.open(f'{NEW}/SUB_SG_Polygon_DEM_1m.tif') as ref:
    CW, CH = ref.width, ref.height; bt, bc = ref.transform, ref.crs
tiles = sorted([d for d in os.listdir(OFD) if os.path.isdir(os.path.join(OFD, d))], key=lambda k: tuple(map(int, k.split('_'))))
missing = [k for k in tiles if not (os.path.exists(f'{OFD}/{k}/Shadow_{k}.tif') and os.path.exists(f'{OFD}/{k}/Category_{k}.tif'))]
assert len(tiles) == NT and not missing, f'tiles {len(tiles)} missing {missing}'
print(f'tiles {len(tiles)} | city {CW}x{CH} | out {MERGE}', flush=True)
checks = {}


def build(layer, dtype, predictor, palette=None):
    out = f'{MERGE}/{layer}_merged.tif'
    prof = dict(driver='GTiff', width=CW, height=CH, count=NB, dtype=dtype, crs=bc, transform=bt, nodata=None,
                compress='LZW', predictor=predictor, tiled=True, blockxsize=512, blockysize=512, BIGTIFF='YES')
    tags = {}
    with rasterio.open(out, 'w', **prof) as dst:
        for ti, key in enumerate(tiles):
            x, y = map(int, key.split('_')); cw = min(CORE, CW - x); ch = min(CORE, CH - y)
            with rasterio.open(f'{OFD}/{key}/{layer}_{key}.tif') as t:
                assert t.count == NB, f'{key} has {t.count} bands'
                arr = t.read(window=Window(0, 0, cw, ch))
                if not tags:
                    for b in range(1, NB + 1):
                        tags[b] = t.tags(b).get('Time')
            dst.write(arr, window=Window(x, y, cw, ch))
            if (ti + 1) % 20 == 0:
                print(f'  {layer} {ti + 1}/{len(tiles)} ({time.time() - t0:.0f}s)', flush=True)
        for b in range(1, NB + 1):
            if palette:
                dst.write_colormap(b, palette)
            if tags.get(b):
                dst.update_tags(b, Time=tags[b])
    checks[layer] = {'bands': NB, 'time_band15': tags.get(15), 'size_GB': round(os.path.getsize(out) / 1e9, 2)}
    print(f'[{layer}] {NB} bands -> {out} ({os.path.getsize(out) / 1e9:.2f} GB) ({time.time() - t0:.0f}s)', flush=True)
    # hourly extracts 11:00-15:00 (band = hour + 1)
    with rasterio.open(out) as r:
        for h, nm in HOURS.items():
            b = h + 1
            o = f'{MERGE}/{layer}_{nm}.tif'
            p = r.profile.copy(); p.update(count=1, compress='LZW', predictor=predictor, tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
            s = 0.0; n = 0
            with rasterio.open(o, 'w', **p) as w:
                for _, win in r.block_windows(1):
                    a = r.read(b, window=win); w.write(a, 1, window=win)
                    if layer == 'Shadow':
                        s += float((a < 0.5).sum()); n += a.size
                if palette:
                    w.write_colormap(1, palette)
                w.update_tags(1, Time=tags.get(b))
            checks[f'{layer}_{nm}'] = {'band': b, 'time': tags.get(b), 'MB': round(os.path.getsize(o) / 1e6, 1), **({'shadow_lt05_frac': round(s / n, 4)} if layer == 'Shadow' else {})}
            print(f'  [{layer}] band{b} ({tags.get(b)}) -> {o}', flush=True)


build('Category', 'uint8', 2, PAL)
build('Shadow', 'float32', 3)
json.dump(checks, open(f'{MERGE}/_MERGE_OK', 'w'), indent=1)
print(f'DONE {MERGE} | {time.time() - t0:.0f}s', flush=True)
