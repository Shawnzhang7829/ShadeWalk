# -*- coding: utf-8 -*-
"""step4_network_final connected-component composition analysis + visualisation: main component / isolated facility components (facility without OSM) / pure OSM fragments. pyenv."""
import geopandas as gpd, numpy as np, collections, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
OUT=r"D:\Claude\SVI_FFW\output\step4_network"
g=gpd.read_file(OUT+r"\step4_network_final.gpkg")
segs=list(g.geometry.values); src=np.array(g['src'].values); L=g.length.values; N=len(segs)
print(f"Segments {N} | total length {L.sum()/1000:.0f}km",flush=True)
print("src composition:")
for s in sorted(set(src)):
    m=src==s; print(f"  {s:9s}: {m.sum():>7} segments {L[m].sum()/1000:7.1f}km")
# connected components (endpoints clustered at 0.5 m)
ep=np.empty((2*N,2))
for i,sg in enumerate(segs): c=sg.coords; ep[2*i]=c[0][:2]; ep[2*i+1]=c[-1][:2]
par=np.arange(2*N)
def f(x):
    r=x
    while par[r]!=r: r=par[r]
    while par[x]!=r: par[x],x=r,par[x]
    return r
for i,j in cKDTree(ep).query_pairs(0.5):
    a,b=f(i),f(j)
    if a!=b: par[a]=b
# cluster endpoints into nodes, then union start <-> end of each segment as an edge (crucial: both ends of a segment must be connected; skipping this step shatters everything)
nid=np.array([f(k) for k in range(2*N)])
roots=sorted(set(nid.tolist())); rmap={r:k for k,r in enumerate(roots)}; NN=len(roots)
p2=np.arange(NN)
def f2(x):
    r=x
    while p2[r]!=r: r=p2[r]
    while p2[x]!=r: p2[x],x=r,p2[x]
    return r
for i in range(N):
    a=rmap[nid[2*i]]; b=rmap[nid[2*i+1]]
    if f2(a)!=f2(b): p2[f2(a)]=f2(b)
comp=np.array([f2(rmap[nid[2*i]]) for i in range(N)])
FAC={'arcade','linkway','facconn'}; OSMS={'osm','osmlink'}
blk_len=collections.defaultdict(float); has_osm=collections.defaultdict(bool); has_fac=collections.defaultdict(bool)
for i in range(N):
    c=comp[i]; blk_len[c]+=L[i]
    if src[i] in OSMS: has_osm[c]=True
    if src[i] in FAC: has_fac[c]=True
blks=sorted(blk_len.items(),key=lambda x:-x[1]); tot=sum(blk_len.values())
main_c=blks[0][0]
iso_fac=[c for c in blk_len if has_fac[c] and not has_osm[c]]      # isolated facility components (facility dangling, not attached to OSM)
iso_osm=[c for c in blk_len if has_osm[c] and not has_fac[c]]      # pure OSM fragments (breaks within the OSM network itself)
both=[c for c in blk_len if has_fac[c] and has_osm[c]]
print(f"\nConnected components {len(blks)} | main component {100*blks[0][1]/tot:.1f}% ({blks[0][1]/1000:.0f}km)")
print(f"Isolated facility components (facility without OSM): {len(iso_fac)} components {sum(blk_len[c] for c in iso_fac)/1000:.1f}km")
print(f"Pure OSM fragments (OSM broken by itself): {len(iso_osm)} components {sum(blk_len[c] for c in iso_osm)/1000:.1f}km")
print(f"Mixed facility+OSM components: {len(both)} components {sum(blk_len[c] for c in both)/1000:.1f}km")
print(f"top10 components (km): {[round(b[1]/1000,1) for b in blks[:10]]}")
# visualisation
idx_main=comp==main_c; idx_isofac=np.isin(comp,iso_fac); idx_other=~idx_main&~idx_isofac
fig,ax=plt.subplots(figsize=(16,11))
g[idx_other].plot(ax=ax,color='#cccccc',lw=0.2)
g[idx_main].plot(ax=ax,color='#0066cc',lw=0.3)
g[idx_isofac].plot(ax=ax,color='#dd0000',lw=0.6)
ax.set_title(f"step4 final connectivity: main component (blue {100*blks[0][1]/tot:.1f}%) - isolated facility components (red {len(iso_fac)} components) - other OSM fragments (grey)",fontsize=13)
ax.set_aspect('equal'); ax.axis('off')
plt.savefig(OUT+r"\_step4_final_components.png",dpi=140,bbox_inches='tight'); print("saved _step4_final_components.png",flush=True)
