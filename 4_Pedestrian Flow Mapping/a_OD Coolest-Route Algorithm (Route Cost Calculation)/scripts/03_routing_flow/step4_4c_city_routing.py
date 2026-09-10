# -*- coding: utf-8 -*-
"""Step4-4c city-wide: station access routing + pedestrian flow. Each path has a station (MRT/BUS) at one end and any building at the other.
Station 14:00 ridership is distributed by surrounding building weight x distance decay (MRT 800 m / BUS 400 m catchment, beta = 350 m, local-cutoff Dijkstra).
Shortest route + coolest route; ridership accumulated along the path = per-edge pedestrian flow. Summarised by building type. pyenv."""
import numpy as np, geopandas as gpd, pandas as pd, time, collections
from scipy.spatial import cKDTree
import networkx as nx
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
BW=r"D:\Claude\UNA\Patronage_Flow\output\building_hourly_weight.gpkg"
ST=r"D:\Claude\SVI_FFW\Shp\SG\POI\station_hourly_ridership.gpkg"
LAM=0.15; BETA=350.0; SNAP_MAX=120.0; D_MRT=800.0; D_BUS=400.0
t0=time.time()
edges=gpd.read_file(f"{OUT}\\step4_4_edges_SG.gpkg")
nodes=gpd.read_file(f"{OUT}\\step4_4_nodes_SG.gpkg")
node_xy=np.column_stack([nodes.geometry.x.values,nodes.geometry.y.values])
G=nx.Graph()
uu=edges.u.values.astype(int); vv=edges.v.values.astype(int)
ln=edges.length.values; sf=np.nan_to_num(edges.shade_full.values,nan=0.0)
for i in range(len(edges)):
    u,v=int(uu[i]),int(vv[i])
    if u==v: continue
    w=ln[i]
    if G.has_edge(u,v) and G[u][v]['w_len']<=w: continue
    G.add_edge(u,v,w_len=w,w_cool=w*((1.0-sf[i])+LAM),length=w,shade=float(sf[i]))
print(f"graph nodes {G.number_of_nodes()} edges {G.number_of_edges()} | {time.time()-t0:.0f}s",flush=True)

ntree=cKDTree(node_xy)
def snap_many(gs):
    P=np.column_stack([gs.x.values,gs.y.values]); d,i=ntree.query(P); return i,d
bw=gpd.read_file(BW).to_crs(3414); rep=bw.representative_point()
bi,bd=snap_many(rep); bw_w=bw.weight_weekday_14.values; bw_t=bw.building_archetype.values
node_blds=collections.defaultdict(list)   # node -> [(w,btype)]
nb=0
for k in range(len(bw)):
    if bd[k]<=SNAP_MAX and np.isfinite(bw_w[k]) and bw_w[k]>0 and int(bi[k]) in G:
        node_blds[int(bi[k])].append((float(bw_w[k]),bw_t[k])); nb+=1
st=gpd.read_file(ST).to_crs(3414); si,sd=snap_many(st.geometry)
stas=[]
for k in range(len(st)):
    if sd[k]<=SNAP_MAX and np.isfinite(st.tot_weekday_14.values[k]) and st.tot_weekday_14.values[k]>0 and int(si[k]) in G:
        stas.append((int(si[k]),float(st.tot_weekday_14.values[k]),st.source.values[k]))
print(f"snapped: buildings {nb} | stations {len(stas)} | {time.time()-t0:.0f}s",flush=True)

flow_s=collections.defaultdict(float); flow_c=collections.defaultdict(float)
def pe(p): return [(p[i],p[i+1]) if p[i]<p[i+1] else (p[i+1],p[i]) for i in range(len(p)-1)]
def path_metrics(p):
    L=0.0;S=0.0
    for a,b in pe(p):
        e=G[a][b]; L+=e['length']; S+=e['length']*e['shade']
    return L,(S/L if L>0 else np.nan)
agg=collections.defaultdict(lambda: np.zeros(6))  # btype -> [vol, vol*sl, vol*ss, vol*cl, vol*cs, vol*det]
tot_R=sum(s[1] for s in stas); routed=0.0
for qi,(sn,R,mode) in enumerate(stas):
    D=D_MRT if mode=='MRT' else D_BUS
    dl,pl=nx.single_source_dijkstra(G,sn,cutoff=D,weight='w_len')
    cat=[]   # (bnode, w, bt, d)
    for n,d in dl.items():
        if n in node_blds:
            for (w,bt) in node_blds[n]: cat.append((n,w,bt,d))
    if not cat: continue
    denom=sum(w*np.exp(-d/BETA) for (n,w,bt,d) in cat)
    if denom<=0: continue
    dc,pc=nx.single_source_dijkstra(G,sn,cutoff=(1+LAM)*D*1.6,weight='w_cool')
    for (n,w,bt,d) in cat:
        vol=R*w*np.exp(-d/BETA)/denom
        if vol<=0 or n==sn: continue
        sp=pl[n]; Ls,ss=path_metrics(sp)
        if Ls<20: continue
        cp=pc.get(n,sp); Lc,sc=path_metrics(cp)
        for a,b in pe(sp): flow_s[(a,b)]+=vol
        for a,b in pe(cp): flow_c[(a,b)]+=vol
        routed+=vol
        det=Lc/Ls if Ls>0 else 1.0
        for key in (bt,'__all__'):
            agg[key]+=[vol,vol*Ls,vol*ss,vol*Lc,vol*sc,vol*det]
    if (qi+1)%800==0: print(f"  stations {qi+1}/{len(stas)} ({time.time()-t0:.0f}s)",flush=True)

edges['flow_short']=[flow_s.get((min(int(u),int(v)),max(int(u),int(v))),0.0) for u,v in zip(uu,vv)]
edges['flow_cool'] =[flow_c.get((min(int(u),int(v)),max(int(u),int(v))),0.0) for u,v in zip(uu,vv)]
edges.to_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg",driver="GPKG")
rows=[]
for bt,a in sorted(agg.items(),key=lambda kv:-kv[1][0]):
    v=a[0]
    rows.append(dict(btype=bt,people=v,short_len=a[1]/v,short_shade=a[2]/v,cool_len=a[3]/v,cool_shade=a[4]/v,detour=a[5]/v))
R=pd.DataFrame(rows); R.to_csv(f"{OUT}\\step4_4_od_metrics_SG.csv",index=False,encoding='utf-8-sig')
print(f"\nstation 14:00 total ridership {tot_R:,.0f} | assigned to access routes {routed:,.0f} ({100*routed/max(tot_R,1):.0f}%)",flush=True)
print("=== city-wide access metrics (flow-weighted) ===",flush=True)
for r in rows[:8]:
    print(f"  {r['btype']:<12} people {r['people']:>10,.0f} | shortest {r['short_len']:.0f}m/{100*r['short_shade']:.0f}% | coolest {r['cool_len']:.0f}m/{100*r['cool_shade']:.0f}% (detour {r['detour']:.2f}, +{100*(r['cool_shade']-r['short_shade']):.0f}pp)",flush=True)
print(f"per-edge pedestrian flow (shortest): max {edges.flow_short.max():,.0f} | {time.time()-t0:.0f}s",flush=True)
