# -*- coding: utf-8 -*-
"""Precompute representative origin-destination pairs for the nav_app Demo scenario bar. 4 scenes: HDB -> transit, office -> transit, office -> food, HDB -> mart.
Per scene: source building centroid -> nearest target (15 min walking range, network distance <= 1200 m); pick the 5 pairs where the shortest route is sun-exposed and the heat-avoiding route is clearly better shaded (large delta shade)
and which are geographically dispersed (800 m grid dedup). Output _demo_scenarios.json {scene:[[olng,olat,dlng,dlat,info],...]}. pyenv, exclusive."""
import geopandas as gpd, numpy as np, json, heapq, time
from collections import defaultdict
from scipy.spatial import cKDTree
from pyproj import Transformer
OUT=r"D:\Claude\SVI_FFW\output\step4_network"; SHP=r"D:\Claude\SVI_FFW\Shp\SG"
T2LL=Transformer.from_crs(3414,4326,always_xy=True); t0=time.time(); MAXW=1200.0; LAM=0.15
def to3414(g):
    try: return g.to_crs(3414)
    except Exception: return g.set_crs(3414,allow_override=True)
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
tree=cKDTree(nxy); print(f"graph N{N} E{NE} | {time.time()-t0:.0f}s",flush=True)
bw=to3414(gpd.read_file(f"{SHP}\\POI\\building_hourly_weight.gpkg", columns=['building_archetype','geometry']))
cc=bw.geometry.centroid; bx=cc.x.values; by=cc.y.values; arch=bw['building_archetype'].values
def cxy(m): return np.c_[bx[m],by[m]]
hdb=cxy(arch=='hdb'); office=cxy(arch=='office')
food=cxy(np.isin(arch,['restaurant','hawker_centre'])); mart=cxy(np.isin(arch,['supermarket','retail']))
st=to3414(gpd.read_file(f"{SHP}\\POI\\station_hourly_ridership.gpkg")); sta=np.c_[st.geometry.x.values,st.geometry.y.values]
print(f"sources hdb{len(hdb)} office{len(office)} | targets sta{len(sta)} food{len(food)} mart{len(mart)} | {time.time()-t0:.0f}s",flush=True)
def msd(txy):
    _,tn=tree.query(txy); tn=tn.astype(int)
    dist=np.full(N,np.inf);L=np.zeros(N);S=np.zeros(N);srcn=np.full(N,-1,np.int64);h=[]
    for n in np.unique(tn):
        if dist[n]>0: dist[n]=0.0;srcn[n]=n;heapq.heappush(h,(0.0,int(n)))
    while h:
        d,x=heapq.heappop(h)
        if d>dist[x]:continue
        for p in range(off[x],off[x+1]):
            y=aN[p];ei=aE[p];w=d+EL[ei]
            if w<dist[y]:dist[y]=w;L[y]=L[x]+EL[ei];S[y]=S[x]+SH[ei]*EL[ei];srcn[y]=srcn[x];heapq.heappush(h,(w,int(y)))
    return L,S,srcn
WCOOL=EL*((1.0-SH)+LAM)
DIST=np.full(N,np.inf);LC=np.zeros(N);SC=np.zeros(N);STMP=np.zeros(N,np.int64);stamp=[0]
def cool1(snode,cutoff):
    stamp[0]+=1;c=stamp[0];DIST[snode]=0.0;LC[snode]=0.0;SC[snode]=0.0;STMP[snode]=c;h=[(0.0,snode)]
    while h:
        d,x=heapq.heappop(h)
        if d>DIST[x]:continue
        for p in range(off[x],off[x+1]):
            y=aN[p];ei=aE[p];w=d+WCOOL[ei]
            if w>cutoff:continue
            if STMP[y]!=c or w<DIST[y]:DIST[y]=w;STMP[y]=c;LC[y]=LC[x]+EL[ei];SC[y]=SC[x]+SH[ei]*EL[ei];heapq.heappush(h,(w,int(y)))
    return c
def pick(sxy,txy,ntop=5):
    L,S,srcn=msd(txy); _,sn=tree.query(sxy); sn=sn.astype(int)
    cand=[]
    for i,node in enumerate(sn):
        if 500<=L[node]<=MAXW and srcn[node]>=0:
            cand.append((i,node,int(srcn[node]),L[node],S[node]/L[node]))
    bytgt=defaultdict(list)
    for ci,(i,node,tgt,Lsh,ssh) in enumerate(cand): bytgt[tgt].append(ci)
    res=[]
    for tgt,cis in bytgt.items():
        cutoff=max(cand[ci][3] for ci in cis)*(1+LAM)*1.4+40; c=cool1(tgt,cutoff)
        for ci in cis:
            i,node,_,Lsh,ssh=cand[ci]
            if STMP[node]==c and LC[node]>0:
                scool=SC[node]/LC[node]; res.append((i,tgt,Lsh,ssh,scool,scool-ssh))
    res.sort(key=lambda r:-r[5])
    out=[]; seen=set()
    for i,tgt,Lsh,ssh,scool,dsh in res:
        if ssh>0.45: continue
        g=(int(sxy[i,0]/800),int(sxy[i,1]/800))
        if g in seen: continue
        seen.add(g)
        o=T2LL.transform(sxy[i,0],sxy[i,1]); d=T2LL.transform(nxy[tgt,0],nxy[tgt,1])
        out.append([round(o[0],6),round(o[1],6),round(d[0],6),round(d[1],6),
                    {'m':int(Lsh),'sShort':int(ssh*100),'sCool':int(scool*100),'dShade':int(dsh*100)}])
        if len(out)>=ntop: break
    return out
SCENES={'hdb_transit':(hdb,sta),'office_transit':(office,sta),'office_food':(office,food),'hdb_mart':(hdb,mart)}
res={}
for k,(sxy,txy) in SCENES.items():
    res[k]=pick(sxy,txy); print(f"[{k}] {len(res[k])} pairs:",flush=True)
    for p in res[k]: print(f"    {p[4]['m']}m  shortest shade {p[4]['sShort']}% -> coolest {p[4]['sCool']}% (+{p[4]['dShade']}pp)",flush=True)
json.dump(res,open(f"{OUT}\\_demo_scenarios.json","w"),ensure_ascii=False)
print(f"wrote _demo_scenarios.json | {time.time()-t0:.0f}s",flush=True)
