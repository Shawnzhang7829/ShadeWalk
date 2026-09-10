# -*- coding: utf-8 -*-
"""On the ORIGINAL network (footpath only, src=footpath) compute the lambda=0.15 coolest per-edge flow
flow_cool_orig -- exactly the same basis as step4_4c_orig_flow.py (station ridership x building weight x
exp(-d/350) distribution, catchment membership judged by shortest-path distance), only the path is changed to the w_cool optimum
(w_cool = len x ((1 - shade) + 0.15), no detour cap). Together with the existing flow_cool in the gpkg
(coolest on the modified network) this forms the before/after comparison on the coolest basis; flow_orig/flow_short
is the before/after comparison on the shortest basis.
Output flow_cool_orig_SG.npy (row order of step4_4_edges_flow_SG.gpkg); the gpkg is not modified."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, geopandas as gpd, time, collections
from scipy.spatial import cKDTree
import networkx as nx
OUT = r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
BW = r"D:\Claude\UNA\Patronage_Flow\output\building_hourly_weight.gpkg"
ST = r"D:\Claude\SVI_FFW\Shp\SG\POI\station_hourly_ridership.gpkg"
LAM = 0.15
BETA = 350.0; SNAP_MAX = 120.0; D_MRT = 800.0; D_BUS = 400.0; t0 = time.time()
edges = gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg")
nodes = gpd.read_file(f"{OUT}\\step4_4_nodes_SG.gpkg")
node_xy = np.column_stack([nodes.geometry.x.values, nodes.geometry.y.values])
uu = edges.u.values.astype(int); vv = edges.v.values.astype(int)
ln = edges.length.values; src = edges['src'].astype(str).values
sf = np.nan_to_num(edges.shade_full.values, nan=0.0)
G = nx.Graph()
for i in range(len(edges)):
    if src[i] != 'footpath': continue
    u, v = int(uu[i]), int(vv[i])
    if u == v: continue
    w = ln[i]
    if G.has_edge(u, v) and G[u][v]['w_len'] <= w: continue
    G.add_edge(u, v, w_len=w, w_cool=w * ((1.0 - sf[i]) + LAM))
print(f"original (footpath) graph nodes {G.number_of_nodes()} edges {G.number_of_edges()} | {time.time()-t0:.0f}s", flush=True)
ntree = cKDTree(node_xy)
def snap_many(gs):
    P = np.column_stack([gs.x.values, gs.y.values]); d, i = ntree.query(P); return i, d
bw = gpd.read_file(BW).to_crs(3414); rep = bw.representative_point()
bi, bd = snap_many(rep); bw_w = bw.weight_weekday_14.values
node_blds = collections.defaultdict(list); nb = 0
for k in range(len(bw)):
    if bd[k] <= SNAP_MAX and np.isfinite(bw_w[k]) and bw_w[k] > 0 and int(bi[k]) in G:
        node_blds[int(bi[k])].append(float(bw_w[k])); nb += 1
st = gpd.read_file(ST).to_crs(3414); si, sd = snap_many(st.geometry)
stas = []
for k in range(len(st)):
    if sd[k] <= SNAP_MAX and np.isfinite(st.tot_weekday_14.values[k]) and st.tot_weekday_14.values[k] > 0 and int(si[k]) in G:
        stas.append((int(si[k]), float(st.tot_weekday_14.values[k]), st.source.values[k]))
print(f"snapped (original network): buildings {nb} | stations {len(stas)} | {time.time()-t0:.0f}s", flush=True)
flow_c = collections.defaultdict(float)
def pe(p): return [(p[i], p[i+1]) if p[i] < p[i+1] else (p[i+1], p[i]) for i in range(len(p)-1)]
tot_R = sum(s[1] for s in stas); routed = 0.0
# upper bound: an access destination has shortest-path length <= D, so its cool cost is <= (1+LAM)*D,
# hence the coolest path's cool cost is also <= (1+LAM)*D -- using this as the cutoff means no truncation
for qi, (sn, R, mode) in enumerate(stas):
    D = D_MRT if mode == 'MRT' else D_BUS
    dl, pl = nx.single_source_dijkstra(G, sn, cutoff=D, weight='w_len')
    cat = [(n, w, d) for n, d in dl.items() if n in node_blds for w in node_blds[n]]
    if not cat: continue
    denom = sum(w * np.exp(-d / BETA) for (n, w, d) in cat)
    if denom <= 0: continue
    dc, pc = nx.single_source_dijkstra(G, sn, cutoff=(1.0 + LAM) * D, weight='w_cool')
    for (n, w, d) in cat:
        vol = R * w * np.exp(-d / BETA) / denom
        if vol <= 0 or n == sn: continue
        cp = pc.get(n, pl[n])
        if len(cp) < 2: continue
        for a, b in pe(cp): flow_c[(a, b)] += vol
        routed += vol
    if (qi + 1) % 800 == 0:
        print(f"  stations {qi+1}/{len(stas)} ({time.time()-t0:.0f}s)", flush=True)
out = np.array([flow_c.get((min(int(u), int(v)), max(int(u), int(v))), 0.0)
                for u, v in zip(uu, vv)])
np.save(f"{OUT}\\flow_cool_orig_SG.npy", out)
print(f"\nstation 14:00 total ridership {tot_R:,.0f} | original-network coolest assigned {routed:,.0f} ({100*routed/max(tot_R,1):.0f}%)", flush=True)
print(f"flow_cool_orig max {out.max():,.0f} | total person-edges {out.sum():,.0f} | {time.time()-t0:.0f}s", flush=True)
print("SAVED flow_cool_orig_SG.npy")
