# -*- coding: utf-8 -*-
"""Step4-4b hourly (0917): per-edge shade fraction for every hour 08:00-18:00, same method as step4_4b_city_edge_shade.py (shade_full):
2-m segmentize sampling along every edge of step4_network_final_prep.gpkg (row order = step4_4_edges_SG.gpkg), Shadow raster value =
lit fraction (0 = fully shaded, 0.1 = under tree canopy, 1 = sunlit; nodata -> lit), pixels inside building_remain -> fully shaded,
edge shade = mean(1 - lit).  The 24-band Shadow_merged.tif carries one band per hour (band = hour + 1; band 15 = 14:00 =
Shadow_2pm_h14.tif, verified identical on the same grid).  Pixels are sampled once, then every hour's band is read and indexed.
Output: edge_shade_hourly_SG.npz with keys h08 ... h18 (float64, full row order, nan = edge without sample) + hours + meta.
Consistency: the 14:00 layer is compared with the shade_full column of step4_4_edges_SG.gpkg.  The mask given by --brem must be
the one that column was computed with (0 differing edges); otherwise the script exits with code 2 (the nav app builder refuses
inconsistent inputs) unless --allow-mismatch is given (analysis runs).
Usage: python step4_4b_city_edge_shade_hourly.py [--brem MASK.tif] [--out edge_shade_hourly_SG.npz] [--hours 8-18] [--allow-mismatch]
"""
import sys, io, os, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, geopandas as gpd, rasterio, pyogrio
from rasterio.windows import Window
OUT = r'D:\Claude\SVI_FFW\output\step5_nav_webapp'
SHADOW = r'D:\Claude\SVI_FFW\TIF_shadow_newarcade\merge_images\Shadow\Shadow_merged.tif'      # 24 bands, band = hour + 1 (new-arcade SOLWEIG run)
BREM_DEFAULT = r'D:\Claude\SVI_FFW\TIF_shadow_newarcade\SUB_SG_Polygon_BREMAIN_1m.tif'        # building_remain mask of step4_4b_city_edge_shade.py
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--brem', default=BREM_DEFAULT, help='building_remain mask (1 m, same grid as the Shadow raster)')
ap.add_argument('--out', default=os.path.join(OUT, 'edge_shade_hourly_SG.npz'))
ap.add_argument('--hours', default='8-18', help='first-last hour, inclusive')
ap.add_argument('--allow-mismatch', action='store_true', help='do not exit 2 when the 14:00 layer differs from the gpkg shade_full column')
args = ap.parse_args()
h0, h1 = [int(x) for x in args.hours.split('-')]; HOURS = list(range(h0, h1 + 1))
T0 = time.time()
net = gpd.read_file(os.path.join(OUT, 'step4_network_final_prep.gpkg'), columns=['length'])
segs = net.geometry.values; N = len(segs)
with rasterio.open(SHADOW) as ds:
    W, H = ds.width, ds.height; T = ds.transform; NB = ds.count
assert NB >= max(HOURS) + 1, (NB, HOURS)
left, top = T.c, T.f
RR, CC, EID = [], [], []
for i, s in enumerate(segs):
    xy = np.asarray(s.segmentize(2.0).coords)
    cc = (xy[:, 0] - left).astype(np.int64); rr = (top - xy[:, 1]).astype(np.int64)
    ok = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
    if ok.sum() == 0:
        continue
    RR.append(rr[ok]); CC.append(cc[ok]); EID.append(np.full(int(ok.sum()), i, np.int64))
RR = np.concatenate(RR); CC = np.concatenate(CC); EID = np.concatenate(EID)
cnt = np.bincount(EID, minlength=N)
print(f'edges {N:,} | sample pixels {len(RR):,} | {time.time()-T0:.0f}s', flush=True)
with rasterio.open(args.brem) as ds:
    assert (ds.width, ds.height) == (W, H) and abs(ds.transform.c - left) < 1e-6 and abs(ds.transform.f - top) < 1e-6, 'mask grid differs from the Shadow grid'
    brem = ds.read(1)
brem_pts = brem[RR, CC] > 0; del brem
print(f'building_remain mask {args.brem} ({brem_pts.mean()*100:.2f}% of samples indoor) | {time.time()-T0:.0f}s', flush=True)
ref = pyogrio.read_dataframe(os.path.join(OUT, 'step4_4_edges_SG.gpkg'), read_geometry=False, columns=['shade_full', 'length', 'comp'])
assert len(ref) == N, (len(ref), N)
L = ref['length'].values; c0 = ref['comp'].values == 0
out = {}; mismatch = None
for h in HOURS:
    full_sh = np.full((H, W), 100, np.uint8)
    with rasterio.open(SHADOW) as ds:
        for r in range(0, H, 2000):
            hh = min(2000, H - r); a = ds.read(h + 1, window=Window(0, r, W, hh))
            full_sh[r:r + hh] = np.clip(np.nan_to_num(a, nan=1.0) * 100.0, 0, 255).astype(np.uint8)
    fv = full_sh[RR, CC].astype(np.float64); del full_sh
    fv[brem_pts] = 0.0
    sums = np.bincount(EID, weights=1.0 - fv / 100.0, minlength=N)
    sf = np.full(N, np.nan); sf[cnt > 0] = sums[cnt > 0] / cnt[cnt > 0]
    out[f'h{h:02d}'] = sf; ok = np.isfinite(sf)
    print(f'h{h:02d}: length-weighted shade all {100*np.sum(L[ok]*sf[ok])/L[ok].sum():.2f}% | comp0 {100*np.sum(L[ok&c0]*sf[ok&c0])/L[ok&c0].sum():.2f}% | {time.time()-T0:.0f}s', flush=True)
    if h == 14:
        g = ref['shade_full'].values; both = ok & np.isfinite(g); d = np.abs(sf[both] - g[both])
        mismatch = int((d > 1e-6).sum()) + int((ok != np.isfinite(g)).sum())
        print(f'  CHECK 14:00 vs gpkg shade_full: nan pattern equal {bool((ok == np.isfinite(g)).all())} | max|diff| {d.max():.2e} | edges differing {mismatch}', flush=True)
meta = dict(shadow=SHADOW, brem=args.brem, network='step4_network_final_prep.gpkg', method='2 m samples, mean(1 - lit), building_remain -> shaded; band = hour + 1',
            mismatch_vs_gpkg_shade_full_h14=mismatch, created=time.strftime('%Y-%m-%d %H:%M'))
np.savez_compressed(args.out, hours=np.array(HOURS), meta=np.array([str(meta)]), **out)
if mismatch:
    print(f'MISMATCH: the 14:00 layer differs from the gpkg shade_full column on {mismatch} edges (mask {args.brem})', flush=True)
    if not args.allow_mismatch:
        print('exit 2 (use the mask the gpkg column was computed with, or --allow-mismatch for analysis runs)'); sys.exit(2)
print(f'SAVED {args.out} | {time.time()-T0:.0f}s')
