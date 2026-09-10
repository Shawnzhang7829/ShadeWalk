# -*- coding: utf-8 -*-
"""Per-metre attribution sensitivity: sample the Category raster every 1 m along each edge and count 1 m per pixel class,
giving network class lengths "split strictly by raster class" (exact to the 1 m raster resolution).
Compared against the second-column shares of edge_facility (whole-edge priority classification) to quantify the bias of the lenient rule.
The pixel class -> major class mapping is the same as in step4_edge_facility.py; the arcade vector overlay and BREMAIN
are not used (pure raster basis), the difference is explained in the report. Output: printout + edge_class_1m_totals.csv."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, geopandas as gpd, rasterio, shapely, time, pandas as pd
OUT = r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
CAT = r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\merge_images\Category_2pm_h14.tif"
t0 = time.time()
e = gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg")
e = e[e['comp'] == 0].reset_index(drop=True)
try: e = e.to_crs(3414)
except Exception: e = e.set_crs(3414, allow_override=True)
ln = e.geometry.length.values
n = len(e)
print(f"edges {n:,} total length {ln.sum()/1e3:,.0f} km | {time.time()-t0:.0f}s", flush=True)
ds = rasterio.open(CAT); cat = ds.read(1); inv = ~ds.transform
H, W = cat.shape
print(f"raster {cat.shape} | {time.time()-t0:.0f}s", flush=True)

# sampling distances at 1 m steps per edge (at least 1 point), vectorised with shapely 2
npts = np.maximum(1, np.ceil(ln).astype(int))
geoms = np.repeat(e.geometry.values, npts)
dist = np.concatenate([np.arange(k) + 0.5 for k in npts])   # midpoint of each 1 m bin
w = np.concatenate([np.full(k, l / k) for k, l in zip(npts, ln)])  # length carried by each point
print(f"sample points {len(dist):,} | {time.time()-t0:.0f}s", flush=True)
pts = shapely.line_interpolate_point(geoms, dist)
xs, ys = shapely.get_x(pts), shapely.get_y(pts)
print(f"interpolate done | {time.time()-t0:.0f}s", flush=True)
cols = np.round(inv.a * xs + inv.b * ys + inv.c).astype(np.int64)
rows = np.round(inv.d * xs + inv.e * ys + inv.f).astype(np.int64)
ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
cv = np.zeros(len(dist), np.uint8); cv[ok] = cat[rows[ok], cols[ok]]
# pixel class -> major class (same priority as cat2fac; pixel classes are mutually exclusive anyway)
fac = np.zeros(len(dist), np.uint8)
fac[np.isin(cv, [8, 9, 10, 11])] = 3
m = (fac == 0) & np.isin(cv, [4, 5, 6, 7]); fac[m] = 4
m = (fac == 0) & np.isin(cv, [2, 3]);       fac[m] = 2
m = (fac == 0) & (cv == 1);                 fac[m] = 1
tot = {k: float(w[fac == k].sum()) for k in range(5)}
names = {0: 'sun', 1: 'building', 2: 'tree', 3: 'arcade', 4: 'ldsm'}
print(f"\n== 1 m per-point attribution (km) | {time.time()-t0:.0f}s")
for k, nm in names.items():
    print(f"  {nm:9s} {tot[k]/1e3:9.1f}")
four = tot[1] + tot[2] + tot[3] + tot[4]
print(f"\nfour-class denominator {four/1e3:,.0f} km")
for k in (3, 4, 2, 1):
    print(f"  {names[k]:9s} {100*tot[k]/four:5.1f}% of shaded")
print(f"  facilities (arcade+ldsm) {100*(tot[3]+tot[4])/four:5.1f}%")
pd.DataFrame([{'cls': names[k], 'len_m': tot[k]} for k in range(5)]
             ).to_csv(f"{OUT}\\edge_class_1m_totals.csv", index=False)
print("SAVED edge_class_1m_totals.csv")
