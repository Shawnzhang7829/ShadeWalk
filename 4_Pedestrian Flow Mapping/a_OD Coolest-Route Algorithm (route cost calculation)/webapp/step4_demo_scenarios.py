# -*- coding: utf-8 -*-
"""Demonstration OD pairs of the web tool's "Demo scenarios" panel: origin type x destination type dropdowns
(HDB / Office / Transit / Dining / Store, 5 x 5 = 25 combinations), up to 5 pairs per combination, written to _demo_scenarios_od25.json.

The graph replicates the web tool exactly: main-component edges (comp == 0), the builder's node numbering, EL = integer metres,
ES = round(shade_full * 100), 14:00, all facilities present; coordinates are written with the page's ll(NX, NY) formula, so the
page's snapPt() lands on the same node.

Selection (routes within one mile whose shortest and coolest routes differ):
  - shortest route 400-1609 m;
  - paper-page Coolest = min sum l * ((1 - sigma) + 0.2): length <= 1609 m and <= 1.5 x the shortest length (the default detour limit of the main page);
  - main-page Coolest = coolDetour() (detour limit 1.5 x, a literal port of the page's bisection): length <= 1609 m;
  - both Coolest routes share <= 60 % of the shortest length with the shortest route (at least 40 % on other streets);
  - paper Coolest shade at least 10 percentage points above the shortest route;
ranked by the shade gain of the paper Coolest, origins de-duplicated on an 800 m grid, 5 pairs per combination.
Types: HDB = hdb; Office = office + business_park; Dining = restaurant + hawker_centre; Store = supermarket + retail
(archetype of building_hourly_weight_gfa.gpkg, footprint centroid); Transit = exits / bus stops of station_hourly_ridership_v4.gpkg.
Parallel, up to 10 processes."""
import os, sys, io, json, time, math, heapq, random
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
OUT = r"D:\Claude\SVI_FFW\output\step5_nav_webapp"; SHP = r"D:\Claude\SVI_FFW\Shp\SG\POI+station"
CACHE = os.path.join(OUT, '_demo_od25_graph.npz')
MILE = 1609.0; LMIN = 400.0; OVMAX = 0.60; GAINMIN = 0.10; PLAM = 0.2; DETOUR = 1.5
NORIG = 220; NPICK = 5; NVERIFY = 14; GRID = 800.0
CATS = ['hdb', 'office', 'transit', 'dining', 'store']


# ---------------- graph (main process builds, workers load the cache) ----------------
def build_cache():
    import geopandas as gpd
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    t0 = time.time()
    e = gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg", ignore_geometry=True)
    e = e[(e['comp'] == 0).values].reset_index(drop=True)
    nd = gpd.read_file(f"{OUT}\\step4_4_nodes_SG.gpkg"); maxid = int(nd['node'].max())
    CX = np.full(maxid + 1, np.nan); CY = np.full(maxid + 1, np.nan)
    CX[nd['node'].values] = nd.geometry.x.values; CY[nd['node'].values] = nd.geometry.y.values
    u = e['u'].values.astype(np.int64); v = e['v'].values.astype(np.int64)
    used = np.unique(np.concatenate([u, v])); remap = np.full(maxid + 1, -1, np.int64); remap[used] = np.arange(len(used))
    U = remap[u].astype(np.int64); V = remap[v].astype(np.int64)
    TO = Transformer.from_crs(3414, 3857, always_xy=True)
    nmx, nmy = TO.transform(CX[used], CY[used]); nmx = np.asarray(nmx); nmy = np.asarray(nmy)
    ox = float(np.floor(nmx.min())); oy = float(np.floor(nmy.min()))
    NX = np.round(nmx - ox).astype(np.int64); NY = np.round(nmy - oy).astype(np.int64)
    EL = np.round(e['length'].values).astype(np.int64)
    ES = np.clip(np.round(np.nan_to_num(e['shade_full'].values) * 100), 0, 100).astype(np.int64)
    sx, sy = CX[used], CY[used]; tree = cKDTree(np.c_[sx, sy])
    # categories -> snapped node indices (unique), in EPSG:3414 for the 800 m grid
    bw = gpd.read_file(f"{SHP}\\building_hourly_weight_gfa.gpkg", columns=['building_archetype', 'geometry'])
    try: bw = bw.to_crs(3414)
    except Exception: bw = bw.set_crs(3414, allow_override=True)
    cc = bw.geometry.centroid; arch = bw['building_archetype'].values; bxy = np.c_[cc.x.values, cc.y.values]
    st = gpd.read_file(f"{SHP}\\station_hourly_ridership_v4.gpkg")
    try: st = st.to_crs(3414)
    except Exception: st = st.set_crs(3414, allow_override=True)
    sxy = np.c_[st.geometry.x.values, st.geometry.y.values]
    sel = {'hdb': bxy[arch == 'hdb'], 'office': bxy[np.isin(arch, ['office', 'business_park'])],
           'dining': bxy[np.isin(arch, ['restaurant', 'hawker_centre'])], 'store': bxy[np.isin(arch, ['supermarket', 'retail'])],
           'transit': sxy}
    out = dict(U=U, V=V, EL=EL, ES=ES, NX=NX, NY=NY, ox=ox, oy=oy, sx=sx, sy=sy)
    for k, xy in sel.items():
        d, n = tree.query(xy); n = np.unique(n[d <= 60.0])   # point must lie within 60 m of the main-component network
        out['cat_' + k] = n.astype(np.int64)
        print(f"  {k:8s} points {len(xy):6d} -> nodes {len(n):6d}", flush=True)
    np.savez(CACHE, **out)
    print(f"graph N{len(used)} E{len(U)} cached | {time.time()-t0:.0f}s", flush=True)


G = {}
def load():
    if G: return G
    z = np.load(CACHE)
    for k in z.files: G[k] = z[k]
    U, V = G['U'], G['V']; N = len(G['NX']); NE = len(U)
    deg = np.zeros(N + 1, np.int64)
    np.add.at(deg, U + 1, 1); np.add.at(deg, V + 1, 1)          # page adjacency: both directions for every edge (incl. loops)
    off = np.cumsum(deg); aN = np.zeros(off[-1], np.int64); aE = np.zeros(off[-1], np.int64); pos = off[:-1].copy()
    for i in range(NE):
        a, b = int(U[i]), int(V[i])
        aN[pos[a]] = b; aE[pos[a]] = i; pos[a] += 1
        aN[pos[b]] = a; aE[pos[b]] = i; pos[b] += 1
    G['off'] = off.tolist(); G['aN'] = aN.tolist(); G['aE'] = aE.tolist()
    G['ELf'] = G['EL'].astype(float).tolist(); G['SHf'] = (G['ES'] / 100.0).tolist(); G['N'] = N
    return G


def sssp(s, lam, limit):
    """single source to all within cost limit; lam=None -> length cost. returns dist, pred-edge dicts"""
    off, aN, aE, EL, SH = G['off'], G['aN'], G['aE'], G['ELf'], G['SHf']
    dist = {s: 0.0}; pe = {s: -1}; h = [(0.0, s)]
    while h:
        d, x = heapq.heappop(h)
        if d > dist[x]: continue
        for p in range(off[x], off[x + 1]):
            y = aN[p]; ei = aE[p]
            c = EL[ei] if lam is None else EL[ei] * ((1 - SH[ei]) + lam)
            n2 = d + c
            if n2 > limit: continue
            if n2 < dist.get(y, 1e18): dist[y] = n2; pe[y] = ei; heapq.heappush(h, (n2, y))
    return dist, pe


def p2p(s, t, lam):
    """page dij(): point-to-point, stops when t is popped; lam=None -> length. returns (edges list) or None"""
    off, aN, aE, EL, SH = G['off'], G['aN'], G['aE'], G['ELf'], G['SHf']
    dist = {s: 0.0}; pe = {s: -1}; pn = {s: -1}; h = [(0.0, s)]
    while h:
        d, x = heapq.heappop(h)
        if d > dist[x]: continue
        if x == t: break
        for p in range(off[x], off[x + 1]):
            y = aN[p]; ei = aE[p]
            c = EL[ei] if lam is None else EL[ei] * ((1 - SH[ei]) + lam)
            n2 = d + c
            if n2 < dist.get(y, 1e18): dist[y] = n2; pe[y] = ei; pn[y] = x; heapq.heappush(h, (n2, y))
    if t not in dist: return None
    es = []; x = t
    while x != s: es.append(pe[x]); x = pn[x]
    return es


def walk(pe, t, s):
    U, V = G['U'], G['V']; es = []; x = t
    while x != s:
        ei = pe[x]; es.append(ei); a, b = int(U[ei]), int(V[ei]); x = a if b == x else b
    return es


def stats(es):
    EL, SH = G['ELf'], G['SHf']
    L = sum(EL[i] for i in es); S = sum(EL[i] * SH[i] for i in es)
    return L, (S / L if L > 0 else 0.0)


def overlap(a, b):
    EL = G['ELf']; sb = set(b); La = sum(EL[i] for i in a)
    return sum(EL[i] for i in a if i in sb) / La if La > 0 else 1.0


def cool_detour(s, t, short_es):
    """verbatim port of the page coolDetour() (DETOUR=1.5, 22 bisection steps)"""
    Ls = stats(short_es)[0]; cap = Ls * DETOUR + 1
    c0 = p2p(s, t, 0.0)
    if c0 is not None and stats(c0)[0] <= cap: return c0
    lo, hi = 0.0, 3.0; ch = p2p(s, t, hi)
    while (ch is None or stats(ch)[0] > cap) and hi < 3000: lo = hi; hi *= 4; ch = p2p(s, t, hi)
    if ch is None or stats(ch)[0] > cap: return short_es
    best = ch
    for _ in range(22):
        mid = (lo + hi) / 2; c = p2p(s, t, mid)
        if c is not None and stats(c)[0] <= cap: hi = mid; best = c
        else: lo = mid
    return best


# ---------------- stage 1: screen every sampled origin against all 5 destination types ----------------
def screen(args):
    oc, s = args; load()
    tgt = {k: set(G['cat_' + k].tolist()) for k in CATS}
    ds, ps = sssp(s, None, MILE)
    dc, pc = sssp(s, PLAM, MILE * (1 + PLAM))
    res = {}
    for dcat in CATS:
        best = None
        for t in tgt[dcat]:
            L = ds.get(t)
            if L is None or L < LMIN or L > MILE or t == s or t not in dc: continue
            se = walk(ps, t, s); ce = walk(pc, t, s)
            Ls, ss = stats(se); Lc, sc = stats(ce)
            if Lc > MILE or Lc > Ls * DETOUR + 1 or sc - ss < GAINMIN: continue
            ov = overlap(se, ce)
            if ov > OVMAX: continue
            cand = (sc - ss, t, Ls, ss, Lc, sc, ov)
            if best is None or cand[0] > best[0]: best = cand
        if best: res[dcat] = best
    return oc, s, res


# ---------------- stage 2: verify the page-default (non-paper) Coolest for the top candidates ----------------
def verify(args):
    key, s, t = args; load()
    se = p2p(s, t, None); ce = p2p(s, t, PLAM); be = cool_detour(s, t, se)
    Ls, ss = stats(se); Lc, sc = stats(ce); Lb, sb = stats(be)
    ok = (LMIN <= Ls <= MILE and Lc <= MILE and Lc <= Ls * DETOUR + 1 and Lb <= MILE and overlap(se, ce) <= OVMAX and overlap(se, be) <= OVMAX and sc - ss >= GAINMIN)
    return key, s, t, ok, dict(m=int(round(Ls)), sShort=int(round(ss * 100)), mCool=int(round(Lc)), sCool=int(round(sc * 100)),
                               mCoolT=int(round(Lb)), sCoolT=int(round(sb * 100)), ov=int(round(overlap(se, ce) * 100)), ovT=int(round(overlap(se, be) * 100)))


def lnglat(n):
    Rm = 6378137.0; X = G['NX'][n] + G['ox']; Y = G['NY'][n] + G['oy']
    return [round(X / Rm * 180 / math.pi, 7), round((2 * math.atan(math.exp(Y / Rm)) - math.pi / 2) * 180 / math.pi, 7)]


if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time()
    if not os.path.exists(CACHE): build_cache()
    load(); rnd = random.Random(923)
    jobs = []
    for oc in CATS:                          # spread origins: one random node per 500 m cell, then sample NORIG cells
        nodes = G['cat_' + oc].tolist(); cells = {}
        for n in nodes: cells.setdefault((int(G['sx'][n] // 500), int(G['sy'][n] // 500)), []).append(n)
        picks = [rnd.choice(v) for v in cells.values()]; rnd.shuffle(picks)
        jobs += [(oc, s) for s in picks[:NORIG]]
        print(f"origin {oc:8s}: {len(nodes)} nodes, {len(cells)} cells -> {min(NORIG, len(picks))} sampled", flush=True)
    with Pool(10) as pool:
        scr = pool.map(screen, jobs, chunksize=4)
    print(f"stage 1 screened {len(jobs)} origins | {time.time()-t0:.0f}s", flush=True)
    cand = {}
    for oc, s, res in scr:
        for dc, b in res.items(): cand.setdefault(f"{oc}_{dc}", []).append((b[0], s, b[1]))
    vjobs = []
    for key, arr in cand.items():
        arr.sort(key=lambda r: -r[0]); seen = set(); k = 0
        for g, s, t in arr:
            cell = (int(G['sx'][s] // GRID), int(G['sy'][s] // GRID))
            if cell in seen: continue
            seen.add(cell); vjobs.append((key, s, t)); k += 1
            if k >= NVERIFY: break
    with Pool(10) as pool:
        ver = pool.map(verify, vjobs, chunksize=2)
    print(f"stage 2 verified {len(vjobs)} pairs | {time.time()-t0:.0f}s", flush=True)
    outj = {}; rank = {(k, s, t): i for i, (k, s, t) in enumerate(vjobs)}
    for key, s, t, ok, info in sorted(ver, key=lambda r: rank[(r[0], r[1], r[2])]):
        if not ok: continue
        arr = outj.setdefault(key, [])
        if len(arr) < NPICK: arr.append(lnglat(s) + lnglat(t) + [info])
    print(f"{'combo':16s} n  shortest m   paper-coolest m (shade pp)   page-coolest m   overlap%")
    for oc in CATS:
        for dc in CATS:
            key = f"{oc}_{dc}"; arr = outj.get(key, [])
            if not arr: print(f"{key:16s} 0  -- none --"); continue
            i = [a[4] for a in arr]
            print(f"{key:16s} {len(arr)}  {min(x['m'] for x in i)}-{max(x['m'] for x in i):<5d}  {min(x['mCool'] for x in i)}-{max(x['mCool'] for x in i)} "
                  f"(+{min(x['sCool']-x['sShort'] for x in i)}..+{max(x['sCool']-x['sShort'] for x in i)})   "
                  f"{min(x['mCoolT'] for x in i)}-{max(x['mCoolT'] for x in i)}   {max(max(x['ov'],x['ovT']) for x in i)}")
    json.dump(outj, open(os.path.join(OUT, '_demo_scenarios_od25.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    print(f"wrote _demo_scenarios_od25.json: {len(outj)} combos, {sum(len(v) for v in outj.values())} pairs | {time.time()-t0:.0f}s", flush=True)
