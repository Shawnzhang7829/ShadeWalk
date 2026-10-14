# -*- coding: utf-8 -*-
# Inputs: building weights building_hourly_weight_gfa.gpkg (Module 4b demand_inputs/building_weights.py) and the station table
# station_hourly_ridership_v4.gpkg (one record per real exit point / bus stop, Module 4b demand_inputs/station_table_real_exits.py);
# stations and demand buildings snap to the nearest node of the main connected component (<= 120 m).
"""OD table of the 14:00 demand model for the web tool's 'OD flow' layer, at building level: the line ends coincide
with the web tool's 'Building centroids' layer.
Demand exactly as Module 4b routing_flow/step4_4c_city_routing.py: graph = step4_4_edges_SG.gpkg (all comps,
u != v, parallel edges keep the shortest); demand buildings = building_hourly_weight_gfa.gpkg weight_weekday_14 > 0, representative point
snapped <= 120 m to a network node (weights summed per node); stations = station_hourly_ridership_v4.gpkg (one row per bus stop / MRT
exit) with the per-row origin weight inj_weekday_14, snapped <= 120 m; catchment = length Dijkstra <= 800 m (MRT) / 400 m (bus);
Huff share of a node = W_node exp(-d/350) / sum over the catchment (incl. the station's own node); a node pair is kept when
bnode != snode and d >= 20 m.  Each node pair is split among the demand buildings snapped to that node in proportion to their weights
(= a building-level Huff share, all buildings of a node having the same distance).
Line end = centroid of the web tool's building layer (step2_building_remain_sg.gpkg, the 'Building centroids' points D.bcx/bcy):
each demand building is mapped to the app building with the largest footprint overlap, else to the nearest app centroid within 30 m,
else it keeps its own centroid (flag ab = -1); pairs sharing (station, target) are summed.
Out: od_pairs_app.npz  (per pair: station row index s, origin node sn, app building index ab (-1 = own centroid), demand building
     row index b (largest-volume member when several were merged), destination node n (idem), station xy, line-end xy (EPSG:3414),
     volume v, volume-weighted shortest length L0) + od_pairs_app_summary.json."""
import sys, io, time, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, geopandas as gpd, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
T0 = time.time()
OUT = r'D:\Claude\SVI_FFW\output\step5_nav_webapp'
BW = r'D:\Claude\SVI_FFW\Shp\SG\POI+station\building_hourly_weight_gfa.gpkg'   # Module 4b building weights
ST = r'D:\Claude\SVI_FFW\Shp\SG\POI+station\station_hourly_ridership_v4.gpkg'
BLDG = r'D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg'          # the web tool's building layer (BLDG in make_nav_app_maplibre.py)
BETA, SNAP_MAX, D_MRT, D_BUS, BS, MATCH_MAX = 350.0, 120.0, 800.0, 400.0, 32, 30.0
edges = gpd.read_file(f'{OUT}\\step4_4_edges_SG.gpkg', columns=['u', 'v', 'length'])
uu0 = edges['u'].values.astype(np.int64); vv0 = edges['v'].values.astype(np.int64); ln0 = edges['length'].values.astype(float)
nodes = gpd.read_file(f'{OUT}\\step4_4_nodes_SG.gpkg'); node_xy = np.column_stack([nodes.geometry.x.values, nodes.geometry.y.values]); N = len(nodes)
ok = uu0 != vv0; uu, vv, ln = uu0[ok], vv0[ok], ln0[ok]
kk = np.minimum(uu, vv) * np.int64(N) + np.maximum(uu, vv); o = np.lexsort((ln, kk)); sel = o[np.r_[True, kk[o][1:] != kk[o][:-1]]]
uu, vv, ln = uu[sel], vv[sel], ln[sel]; deg = np.zeros(N, bool); deg[uu] = True; deg[vv] = True
Alen = csr_matrix((np.r_[ln, ln], (np.r_[uu, vv], np.r_[vv, uu])), shape=(N, N)); print(f'graph N={N:,} E={len(ln):,} | {time.time()-T0:.0f}s', flush=True)
from scipy.sparse.csgraph import connected_components   # snapping to the main connected component
_lab = connected_components(Alen, directed=False)[1]; MAIN = np.flatnonzero(_lab == np.argmax(np.bincount(_lab))); print(f'main component {len(MAIN):,} / {N:,} nodes', flush=True)
tree = cKDTree(node_xy[MAIN])
bw = gpd.read_file(BW, columns=['weight_weekday_14']).to_crs(3414); rp = bw.representative_point(); cb = bw.geometry.centroid
bd, bi = tree.query(np.c_[rp.x, rp.y]); bi = MAIN[bi]; bwv = np.nan_to_num(bw['weight_weekday_14'].values.astype(float))
bkeep = (bd <= SNAP_MAX) & (bwv > 0) & deg[bi]; brow = np.nonzero(bkeep)[0]; bnode_of = bi[bkeep].astype(int); bw_k = bwv[bkeep]
node_w = np.zeros(N); np.add.at(node_w, bnode_of, bw_k); bshare = bw_k / node_w[bnode_of]
order = np.argsort(bnode_of, kind='stable'); bnode_sorted = bnode_of[order]
indptr = np.zeros(N + 1, np.int64); np.add.at(indptr, bnode_sorted + 1, 1); indptr = np.cumsum(indptr)
bnodes = np.unique(bnode_of); bwt = node_w[bnodes]
# ---- map demand buildings -> app buildings (largest overlap, else nearest centroid <= 30 m, else own centroid) ----
gb = gpd.read_file(BLDG, columns=[]); gb = gb.to_crs(3414) if gb.crs else gb.set_crs(3414, allow_override=True); ca = gb.geometry.centroid
sj = gpd.sjoin(gpd.GeoDataFrame(geometry=bw.geometry, crs=3414), gpd.GeoDataFrame(geometry=gb.geometry, crs=3414), how='inner', predicate='intersects')
import shapely
ov = shapely.area(shapely.intersection(shapely.make_valid(np.asarray(sj.geometry.values)), shapely.make_valid(np.asarray(gb.geometry.values[sj.index_right.values]))))   # make_valid: a few footprints are self-touching (TopologyException otherwise)
best = pd.DataFrame({'d': sj.index.values, 'a': sj.index_right.values, 'ov': ov}); best = best[best.ov > 0].sort_values('ov', ascending=False).drop_duplicates('d')
amap = np.full(len(bw), -1, np.int64); amap[best.d.values] = best.a.values
miss = np.nonzero(amap < 0)[0]; dd, jj = cKDTree(np.c_[ca.x.values, ca.y.values]).query(np.c_[cb.x.values[miss], cb.y.values[miss]]); near = dd <= MATCH_MAX; amap[miss[near]] = jj[near]
print(f'demand buildings {len(bw):,}: overlap-matched {len(best):,}, nearest<=30 m {int(near.sum()):,}, own centroid {int((amap < 0).sum()):,} | app buildings {len(gb):,} | {time.time()-T0:.0f}s', flush=True)
end_x = np.where(amap >= 0, ca.x.values[np.maximum(amap, 0)], cb.x.values); end_y = np.where(amap >= 0, ca.y.values[np.maximum(amap, 0)], cb.y.values)
st = gpd.read_file(ST).to_crs(3414); sxy = np.c_[st.geometry.x.values, st.geometry.y.values]; sd, si = tree.query(sxy); si = MAIN[si]
sv = st['inj_weekday_14'].values.astype(float); skeep = (sd <= SNAP_MAX) & np.isfinite(sv) & (sv > 0) & deg[si]
srow = np.nonzero(skeep)[0]; snodes = si[skeep].astype(int); R = sv[skeep]; MODE = st['source'].astype(str).values[skeep]
print(f'demand buildings kept {len(brow):,} on {len(bnodes):,} nodes | stations {len(snodes):,} (weight sum {R.sum():,.0f}) | {time.time()-T0:.0f}s', flush=True)
S_, SN_, B_, NODE_, V_, L_ = [], [], [], [], [], []
for s0 in range(0, len(snodes), BS):
    idx = snodes[s0:s0 + BS]; dl = dijkstra(Alen, indices=idx, limit=D_MRT * 1.000001)
    for r in range(len(idx)):
        k = s0 + r; D = D_MRT if MODE[k] == 'MRT' else D_BUS; d0 = dl[r, bnodes]; cm = d0 <= D
        if not cm.any(): continue
        ws = bwt[cm] * np.exp(-d0[cm] / BETA); denom = ws.sum()
        if denom <= 0: continue
        vol = R[k] * ws / denom; bn = bnodes[cm]; dd = d0[cm]; rm = (bn != idx[r]) & (dd >= 20.0) & (vol > 0)
        bn, vol, dd = bn[rm], vol[rm], dd[rm]
        if not len(bn): continue
        starts, ends = indptr[bn], indptr[bn + 1]; cnt = ends - starts; tot = int(cnt.sum())
        pos = np.repeat(starts - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(tot); kb = order[pos]
        S_.append(np.full(tot, srow[k])); SN_.append(np.full(tot, idx[r])); B_.append(brow[kb]); NODE_.append(np.repeat(bn, cnt)); V_.append(np.repeat(vol, cnt) * bshare[kb]); L_.append(np.repeat(dd, cnt))
S_, SN_, B_, NODE_, V_, L_ = (np.concatenate(x) for x in (S_, SN_, B_, NODE_, V_, L_)); n_raw = len(V_)
# ---- merge pairs that share (station, line end) ----
AB = amap[B_]; target = np.where(AB >= 0, AB, len(gb) + B_); key = S_.astype(np.int64) * np.int64(10 ** 7) + target
df = pd.DataFrame({'key': key, 's': S_, 'sn': SN_, 'ab': AB, 'b': B_, 'n': NODE_, 'v': V_, 'vl': V_ * L_}); df = df.sort_values(['key', 'v'], ascending=[True, False])
g = df.groupby('key', sort=False); first = g.first(); v = g['v'].sum().values; L0 = g['vl'].sum().values / v
S_, SN_, AB_, B_, NODE_ = first.s.values, first.sn.values, first.ab.values, first.b.values, first.n.values
np.savez_compressed(f'{OUT}\\od_pairs_app.npz', s=S_, sn=SN_, ab=AB_, b=B_, n=NODE_, v=v, L0=L0, sx=sxy[S_, 0], sy=sxy[S_, 1], cx=end_x[B_], cy=end_y[B_])
summ = dict(pairs=int(len(v)), pairs_before_merge=int(n_raw), stations=int(len(np.unique(S_))), line_ends=int(len(np.unique(target))), line_ends_on_app_centroids=int(len(np.unique(AB_[AB_ >= 0]))),
            pairs_on_own_centroid=int((AB_ < 0).sum()), volume_on_own_centroid_pct=float(100 * v[AB_ < 0].sum() / v.sum()), building_nodes=int(len(np.unique(NODE_))), volume_total=float(v.sum()), station_weight_total=float(R.sum()),
            volume_percentiles={p: float(np.percentile(v, p)) for p in (50, 90, 99)}, volume_max=float(v.max()), pairs_ge_100=int((v >= 100).sum()), share_ge_100_pct=float(100 * v[v >= 100].sum() / v.sum()),
            L0_percentiles_m={p: float(np.percentile(L0, p)) for p in (10, 50, 90)}, station_table=ST, destination='centroid of the app building layer (step2_building_remain_sg.gpkg) matched by overlap / nearest <= 30 m, own centroid otherwise; node share split by demand-building weight')
json.dump(summ, open(f'{OUT}\\od_pairs_app_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1)); print(f'SAVED od_pairs_app.npz | {time.time()-T0:.0f}s')
