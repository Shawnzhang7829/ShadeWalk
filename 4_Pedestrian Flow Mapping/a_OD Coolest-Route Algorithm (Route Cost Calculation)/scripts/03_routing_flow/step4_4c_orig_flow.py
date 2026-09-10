# -*- coding: utf-8 -*-
"""On the ORIGINAL network (footpath only, src=footpath, no shade-facility shortcuts) recompute the per-edge flow flow_orig on the same basis,
for the nav_app flow mode "original only" comparison of the flow distribution before/after the modification. Method as in step4_4c (station ridership distributed by surrounding building weight x distance decay,
accumulated along shortest routes), but the graph is built from footpath edges only. Writes the column flow_orig back into step4_4_edges_flow_SG.gpkg. pyenv, exclusive."""
import numpy as np, geopandas as gpd, time, collections
from scipy.spatial import cKDTree
import networkx as nx
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
BW=r"D:\Claude\UNA\Patronage_Flow\output\building_hourly_weight.gpkg"
ST=r"D:\Claude\SVI_FFW\Shp\SG\POI\station_hourly_ridership.gpkg"
BETA=350.0; SNAP_MAX=120.0; D_MRT=800.0; D_BUS=400.0; t0=time.time()
edges=gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg")
nodes=gpd.read_file(f"{OUT}\\step4_4_nodes_SG.gpkg")
node_xy=np.column_stack([nodes.geometry.x.values,nodes.geometry.y.values])
uu=edges.u.values.astype(int); vv=edges.v.values.astype(int)
ln=edges.length.values; src=edges['src'].astype(str).values
G=nx.Graph()
for i in range(len(edges)):
    if src[i]!='footpath': continue        # original footpath edges only (no facility shortcuts)
    u,v=int(uu[i]),int(vv[i])
    if u==v: continue
    w=ln[i]
    if G.has_edge(u,v) and G[u][v]['w_len']<=w: continue
    G.add_edge(u,v,w_len=w)
print(f"original (footpath) graph nodes {G.number_of_nodes()} edges {G.number_of_edges()} | {time.time()-t0:.0f}s",flush=True)
ntree=cKDTree(node_xy)
def snap_many(gs):
    P=np.column_stack([gs.x.values,gs.y.values]); d,i=ntree.query(P); return i,d
bw=gpd.read_file(BW).to_crs(3414); rep=bw.representative_point()
bi,bd=snap_many(rep); bw_w=bw.weight_weekday_14.values
node_blds=collections.defaultdict(list)
nb=0
for k in range(len(bw)):
    if bd[k]<=SNAP_MAX and np.isfinite(bw_w[k]) and bw_w[k]>0 and int(bi[k]) in G:
        node_blds[int(bi[k])].append(float(bw_w[k])); nb+=1
st=gpd.read_file(ST).to_crs(3414); si,sd=snap_many(st.geometry)
stas=[]
for k in range(len(st)):
    if sd[k]<=SNAP_MAX and np.isfinite(st.tot_weekday_14.values[k]) and st.tot_weekday_14.values[k]>0 and int(si[k]) in G:
        stas.append((int(si[k]),float(st.tot_weekday_14.values[k]),st.source.values[k]))
print(f"snapped (original network): buildings {nb} | stations {len(stas)} | {time.time()-t0:.0f}s",flush=True)
flow_s=collections.defaultdict(float)
def pe(p): return [(p[i],p[i+1]) if p[i]<p[i+1] else (p[i+1],p[i]) for i in range(len(p)-1)]
tot_R=sum(s[1] for s in stas); routed=0.0
for qi,(sn,R,mode) in enumerate(stas):
    D=D_MRT if mode=='MRT' else D_BUS
    dl,pl=nx.single_source_dijkstra(G,sn,cutoff=D,weight='w_len')
    cat=[(n,w,d) for n,d in dl.items() if n in node_blds for w in node_blds[n]]
    if not cat: continue
    denom=sum(w*np.exp(-d/BETA) for (n,w,d) in cat)
    if denom<=0: continue
    for (n,w,d) in cat:
        vol=R*w*np.exp(-d/BETA)/denom
        if vol<=0 or n==sn: continue
        sp=pl[n]
        if len(sp)<2: continue
        for a,b in pe(sp): flow_s[(a,b)]+=vol
        routed+=vol
    if (qi+1)%800==0: print(f"  stations {qi+1}/{len(stas)} ({time.time()-t0:.0f}s)",flush=True)
edges['flow_orig']=[flow_s.get((min(int(u),int(v)),max(int(u),int(v))),0.0) for u,v in zip(uu,vv)]
edges.to_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg",driver="GPKG")
print(f"\nstation 14:00 total ridership {tot_R:,.0f} | original network assigned {routed:,.0f} ({100*routed/max(tot_R,1):.0f}%)",flush=True)
print(f"original-network per-edge flow max {edges.flow_orig.max():,.0f} (modified network flow_short max {edges.flow_short.max():,.0f}) | {time.time()-t0:.0f}s",flush=True)
