# -*- coding: utf-8 -*-
"""City-wide: split the modified network (step4_network_final main component) by the 1 m Category raster class
with true geometry -- sample the class code every metre along each edge, merge consecutive runs of the same class, and cut sub-segments with substring.
Class codes as in step4_edge_class_1m (pixel-level priority arcade > ldsm > tree > building, remainder sun).
Output step4_network_split_by_class.gpkg (eid, cls, len_m, geometry). pyenv."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, geopandas as gpd, rasterio, shapely, time
from shapely.ops import substring
OUT = r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
CAT = r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\merge_images\Category_2pm_h14.tif"
t0 = time.time()
e = gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg")
e = e[e['comp'] == 0].reset_index(drop=True)
try: e = e.to_crs(3414)
except Exception: e = e.set_crs(3414, allow_override=True)
geo = e.geometry.values
ln = e.geometry.length.values
n = len(e)
ds = rasterio.open(CAT); cat = ds.read(1); inv = ~ds.transform
H, W = cat.shape
print(f"edges {n:,} | raster {cat.shape} | {time.time()-t0:.0f}s", flush=True)
npts = np.maximum(1, np.ceil(ln).astype(int))
off = np.concatenate([[0], np.cumsum(npts)])
geoms_r = np.repeat(geo, npts)
dist = np.concatenate([np.arange(k) + 0.5 for k in npts])
pts = shapely.line_interpolate_point(geoms_r, dist)
xs, ys = shapely.get_x(pts), shapely.get_y(pts)
cols = np.round(inv.a * xs + inv.b * ys + inv.c).astype(np.int64)
rows = np.round(inv.d * xs + inv.e * ys + inv.f).astype(np.int64)
ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
cv = np.zeros(len(dist), np.uint8); cv[ok] = cat[rows[ok], cols[ok]]
fac = np.zeros(len(dist), np.uint8)
fac[np.isin(cv, [8, 9, 10, 11])] = 3
m = (fac == 0) & np.isin(cv, [4, 5, 6, 7]); fac[m] = 4
m = (fac == 0) & np.isin(cv, [2, 3]);       fac[m] = 2
m = (fac == 0) & (cv == 1);                 fac[m] = 1
print(f"per-metre class codes ready | {time.time()-t0:.0f}s", flush=True)

out_g, out_c, out_e, out_l = [], [], [], []
for i in range(n):
    f = fac[off[i]:off[i + 1]]
    L = ln[i]
    if len(f) == 1 or (f == f[0]).all():
        out_g.append(geo[i]); out_c.append(int(f[0])); out_e.append(i)
        out_l.append(L); continue
    cut = np.flatnonzero(f[1:] != f[:-1]) + 1          # run starts
    bounds = np.concatenate([[0], cut, [len(f)]]).astype(float)
    bounds = bounds / len(f) * L                        # metre positions
    for a, b, k in zip(bounds[:-1], bounds[1:], f[np.concatenate([[0], cut])]):
        out_g.append(substring(geo[i], a, b))
        out_c.append(int(k)); out_e.append(i); out_l.append(b - a)
    if i % 60000 == 0:
        print(f"  {i:,}/{n:,} | {time.time()-t0:.0f}s", flush=True)
g = gpd.GeoDataFrame({'eid': out_e, 'cls': out_c, 'len_m': out_l},
                     geometry=out_g, crs=3414)
g.to_file(f"{OUT}\\step4_network_split_by_class.gpkg", driver='GPKG')
print(f"\nsub-segments {len(g):,} total length {g.len_m.sum()/1e3:,.0f} km | {time.time()-t0:.0f}s")
for k, nm in [(0, 'sun'), (1, 'building'), (2, 'tree'), (3, 'arcade'), (4, 'ldsm')]:
    print(f"  {nm:9s} {g.loc[g.cls==k,'len_m'].sum()/1e3:9.1f} km")
print("SAVED step4_network_split_by_class.gpkg")
