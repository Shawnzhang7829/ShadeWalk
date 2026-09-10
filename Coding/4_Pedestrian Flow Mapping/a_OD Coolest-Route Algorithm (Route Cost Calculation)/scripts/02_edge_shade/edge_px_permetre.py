# -*- coding: utf-8 -*-
"""Per-metre shade-source attribution for every comp-0 edge (Fig3-1b basis):
sample Category (14:00) + BREMAIN at 1 m steps along every edge:
  shade_px = metres of "outdoor shaded" length on the edge      (Category 1..11, not indoor)
  fac_px   = metres of that shade that come from facilities    (Category 4..11, non-open-air classes; not indoor)
Indoor pixels (BREMAIN>0) are removed -- they count neither as shade nor as fac (through-building excluded).
Output edge_px_SG.npz (shade_px, fac_px; row order = step4_4_edges_flow_SG comp==0)."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, geopandas as gpd, rasterio
OUT = r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
CAT = r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\merge_images\Category_2pm_h14.tif"
BREM = r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\SUB_SG_Polygon_BREMAIN_1m.tif"
t0 = time.time()

e = gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg", columns=['comp'])
e = e[e['comp'] == 0].reset_index(drop=True)
n = len(e)
print(f"edges {n} | {time.time()-t0:.0f}s", flush=True)

ds = rasterio.open(CAT); cat = ds.read(1); inv = ~ds.transform
bds = rasterio.open(BREM); brem = bds.read(1); binv = ~bds.transform
H, W = cat.shape
print(f"rasters loaded | {time.time()-t0:.0f}s", flush=True)

# per-edge 1m points: build via linear referencing on coordinate arrays
shade_px = np.zeros(n, np.float32)
fac_px = np.zeros(n, np.float32)
BATCHPTS_X = []; BATCHPTS_Y = []; BATCH_EDGE = []


def flush_batch():
    global BATCHPTS_X, BATCHPTS_Y, BATCH_EDGE
    if not BATCH_EDGE:
        return
    xs = np.concatenate(BATCHPTS_X); ys = np.concatenate(BATCHPTS_Y)
    ei = np.concatenate(BATCH_EDGE)
    cols = np.round(inv.a*xs + inv.b*ys + inv.c).astype(np.int64)
    rows = np.round(inv.d*xs + inv.e*ys + inv.f).astype(np.int64)
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    cv = np.zeros(len(xs), np.uint8); cv[ok] = cat[rows[ok], cols[ok]]
    bv = np.zeros(len(xs), bool); bv[ok] = brem[rows[ok], cols[ok]] > 0
    outdoor = ~bv
    sh = outdoor & (cv >= 1) & (cv <= 11)
    fc = outdoor & (cv >= 4) & (cv <= 11)
    np.add.at(shade_px, ei[sh], 1.0)
    np.add.at(fac_px, ei[fc], 1.0)
    BATCHPTS_X = []; BATCHPTS_Y = []; BATCH_EDGE = []


cnt = 0
for i, g in enumerate(e.geometry.values):
    co = np.asarray(g.coords)
    seg = np.hypot(np.diff(co[:, 0]), np.diff(co[:, 1]))
    L = seg.sum()
    m = max(1, int(round(L)))                     # 1 point per metre
    d = (np.arange(m) + 0.5) * (L / m)            # midpoints of 1m bins
    cs = np.r_[0.0, np.cumsum(seg)]
    k = np.searchsorted(cs, d, side='right') - 1
    k = np.clip(k, 0, len(seg) - 1)
    tloc = (d - cs[k]) / np.maximum(seg[k], 1e-9)
    px = co[k, 0] + tloc * (co[k + 1, 0] - co[k, 0])
    py = co[k, 1] + tloc * (co[k + 1, 1] - co[k, 1])
    BATCHPTS_X.append(px); BATCHPTS_Y.append(py)
    BATCH_EDGE.append(np.full(m, i, np.int64))
    cnt += m
    if cnt > 2_000_000:
        flush_batch(); cnt = 0
        if i % 50000 < 1:
            print(f"  {i}/{n} | {time.time()-t0:.0f}s", flush=True)
flush_batch()

np.savez(f"{OUT}\\edge_px_SG.npz", shade_px=shade_px, fac_px=fac_px)
print(f"saved edge_px_SG.npz | total shade km {shade_px.sum()/1000:.0f} "
      f"fac km {fac_px.sum()/1000:.0f} | {time.time()-t0:.0f}s", flush=True)
