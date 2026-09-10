# -*- coding: utf-8 -*-
"""Merge the demo_saved.json ([[olng,olat,dlng,dlat],...]) that the user downloaded via the nav_app "Export" button into
the 'my_saved' scene of _demo_scenarios.json: for every OD pair compute the shade fraction and length of the SHORTEST / COOLEST route on the real city-wide network,
producing an info record on the same basis as the four main scenes. Afterwards rerun make_nav_app and the "My favourites" button cycles through these
built-in demos together with the browser's local favourites (dmMine reads localStorage union D.demoOD.my_saved).
pyenv, exclusive (reads only edges/nodes, no large rasters). Usage: python step4_merge_saved_demo.py [demo_saved.json]"""
import geopandas as gpd, numpy as np, json, heapq, time, sys, os
from scipy.spatial import cKDTree
from pyproj import Transformer
OUT=r"D:\Claude\SVI_FFW\output\step4_network"; LAM=0.15; t0=time.time()
LL2T=Transformer.from_crs(4326,3414,always_xy=True)
SAVED=sys.argv[1] if len(sys.argv)>1 else f"{OUT}\\demo_saved.json"
if not os.path.exists(SAVED):
    print(f"cannot find {SAVED}\n  first click 'Export' in nav_app to download demo_saved.json and put it under {OUT} (or pass the path as an argument)."); sys.exit(1)
pairs=json.load(open(SAVED,encoding='utf-8')); print(f"read {len(pairs)} saved OD pairs | {time.time()-t0:.0f}s",flush=True)
# network (same basis as the demo / access scripts)
e=gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg"); nd=gpd.read_file(f"{OUT}\\step4_4_nodes_SG.gpkg")
nid=nd['node'].values.astype(np.int64); nxy=np.c_[nd.geometry.x.values,nd.geometry.y.values]
id2i={int(k):i for i,k in enumerate(nid)}; N=len(nid)
U=np.array([id2i[int(u)] for u in e['u'].values]); V=np.array([id2i[int(v)] for v in e['v'].values])
EL=e['length'].values.astype(float); SH=np.clip(np.nan_to_num(e['shade_full'].values),0,1); NE=len(U)
deg=np.zeros(N+1,np.int64)
for i in range(NE):
    if U[i]!=V[i]: deg[U[i]+1]+=1; deg[V[i]+1]+=1
off=np.cumsum(deg); aN=np.zeros(off[-1],np.int64); aE=np.zeros(off[-1],np.int64); pos=off[:-1].copy()
for i in range(NE):
    a,b=int(U[i]),int(V[i])
    if a==b: continue
    aN[pos[a]]=b;aE[pos[a]]=i;pos[a]+=1; aN[pos[b]]=a;aE[pos[b]]=i;pos[b]+=1
tree=cKDTree(nxy); WCOOL=EL*((1.0-SH)+LAM)
print(f"graph N{N} E{NE} | {time.time()-t0:.0f}s",flush=True)
def dij(s,t,w):  # single-source single-target Dijkstra; backtrack the actual length and shaded length (SH*EL) along the path
    if s==t: return (0.0,0.0)
    dist=np.full(N,np.inf); dist[s]=0.0; pe=np.full(N,-1,np.int64); pn=np.full(N,-1,np.int64); h=[(0.0,s)]
    while h:
        d,x=heapq.heappop(h)
        if d>dist[x]: continue
        if x==t: break
        for p in range(off[x],off[x+1]):
            y=aN[p]; ei=aE[p]; nw=d+w[ei]
            if nw<dist[y]: dist[y]=nw; pe[y]=ei; pn[y]=int(x); heapq.heappush(h,(nw,int(y)))
    if not np.isfinite(dist[t]): return None
    L=0.0; S=0.0; cur=t
    while cur!=s:
        ei=pe[cur]
        if ei<0: return None
        L+=EL[ei]; S+=SH[ei]*EL[ei]; cur=int(pn[cur])
    return (L,S)
res=[]
for q in pairs:
    try: olng,olat,dlng,dlat=float(q[0]),float(q[1]),float(q[2]),float(q[3])
    except Exception: print(f"  skipped (bad format) {q}",flush=True); continue
    ox,oy=LL2T.transform(olng,olat); dx,dy=LL2T.transform(dlng,dlat)
    _,os_=tree.query([ox,oy]); _,ds_=tree.query([dx,dy])
    sp=dij(int(os_),int(ds_),EL); cp=dij(int(os_),int(ds_),WCOOL)
    if not sp or not cp or sp[0]<=0: print(f"  skipped (not connected) {olng:.5f},{olat:.5f}",flush=True); continue
    Ls,Ss=sp; Lc,Sc=cp; sShort=Ss/Ls; sCool=(Sc/Lc) if Lc>0 else 0.0
    res.append([round(olng,6),round(olat,6),round(dlng,6),round(dlat,6),
                {'m':int(Ls),'sShort':int(sShort*100),'sCool':int(sCool*100),'dShade':int((sCool-sShort)*100)}])
    print(f"  {int(Ls)}m  shortest {int(sShort*100)}% -> coolest {int(sCool*100)}% (+{int((sCool-sShort)*100)}pp)",flush=True)
ds=json.load(open(f"{OUT}\\_demo_scenarios.json",encoding='utf-8'))
ds['my_saved']=res; json.dump(ds,open(f"{OUT}\\_demo_scenarios.json","w"),ensure_ascii=False)
print(f"merged {len(res)}/{len(pairs)} pairs -> _demo_scenarios.json['my_saved']",flush=True)
print(f"now rerun: python make_nav_app_maplibre.py && python _make_en_navapp.py  to build them in ('My favourites' carousel) | {time.time()-t0:.0f}s",flush=True)
