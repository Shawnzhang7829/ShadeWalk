# -*- coding: utf-8 -*-
"""Paper basis (fixed lambda) city-wide flow assignment: coolest = min omega = sum of length x ((1 - sigma) + lambda), no detour cap.
OD demand exactly mirrors step4_4c/4e (station 14:00 ridership x building weight x exp(-d/350), MRT 800 m / BUS 400 m, SNAP 120 m, Ls<20 dropped).
lambda takes the sweep levels of the main-text SI sensitivity analysis (lambda_sweep_4c_strict.py):
  [3.0,1.5,1.0,0.7,0.5,0.35,0.25,0.18,0.15,0.12,0.09,0.06,0.04,0.02,0.01,0.005], 16 levels in total
  (lambda=1e9 is equivalent to shortest; flow_short already exists and is not recomputed)
For each lambda one single-source Dijkstra per station (cost = len x ((1 - sigma) + lambda)), plen/pshade accumulated on the tree; output flow_lam_<id>.npy (comp0 row order)
+ summary CSV (shade fraction / detour ratio / facility person-metre share vs lambda). Validation: lambda=1e9 recomputation ~= 4c stored flow_short. pyenv."""
import numpy as np, geopandas as gpd, pandas as pd, time, collections, heapq
from scipy.spatial import cKDTree
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
BW=r"D:\Claude\UNA\Patronage_Flow\output\building_hourly_weight.gpkg"
ST=r"D:\Claude\SVI_FFW\Shp\SG\POI\station_hourly_ridership.gpkg"
BETA=350.0; SNAP_MAX=120.0; D_MRT=800.0; D_BUS=400.0
LAMS_ALL=[3.0,1.5,1.0,0.7,0.5,0.35,0.25,0.2,0.18,0.15,0.12,0.09,0.06,0.04,0.02,0.01,0.005,0.001]  # main-text sweep 16 levels + paper calibration 0.2 + limit 0.001 = 18 levels
def lid(l): return str(l).replace('.','p')
import os
LAMS=[l for l in LAMS_ALL if not os.path.exists(rf"{OUT}\flow_lam_{lid(l)}.npy")]   # incremental: compute only the missing levels
print(f"lambda levels to compute {len(LAMS)}/{len(LAMS_ALL)}: {LAMS}",flush=True)
if not LAMS:
    print("all levels already exist, nothing to recompute"); raise SystemExit
t0=time.time()
edges=gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg")
nodes=gpd.read_file(f"{OUT}\\step4_4_nodes_SG.gpkg")
node_xy=np.column_stack([nodes.geometry.x.values,nodes.geometry.y.values])
uu=edges.u.values.astype(int); vv=edges.v.values.astype(int)
ln=edges.length.values.astype(float); sf=np.nan_to_num(edges.shade_full.values,nan=0.0)
comp0=np.where(edges.comp.values==0)[0]
EFAC=np.load(f"{OUT}\\edge_facility_SG.npy")
EF_full=np.zeros(len(edges),np.uint8); EF_full[comp0]=EFAC
pairs={}
for i in range(len(edges)):
    u,v=uu[i],vv[i]
    if u==v: continue
    key=(u,v) if u<v else (v,u)
    if key in pairs and pairs[key][0]<=ln[i]: continue
    pairs[key]=(ln[i],sf[i],i)
npair=len(pairs); pk=np.array(list(pairs.keys())); pval=list(pairs.values())
plen2=np.array([x[0] for x in pval]); psh2=np.array([x[1] for x in pval]); prow=np.array([x[2] for x in pval])
pshm2=plen2*psh2; pfac=np.isin(EF_full[prow],[3,4])
nN=int(pk.max())+1
deg=np.zeros(nN+1,np.int64); np.add.at(deg,pk[:,0]+1,1); np.add.at(deg,pk[:,1]+1,1)
off=np.cumsum(deg); adjN=np.zeros(off[-1],np.int64); adjE=np.zeros(off[-1],np.int64); cur=off[:-1].copy()
for j in range(npair):
    a,b=pk[j,0],pk[j,1]
    adjN[cur[a]]=b; adjE[cur[a]]=j; cur[a]+=1
    adjN[cur[b]]=a; adjE[cur[b]]=j; cur[b]+=1
print(f"graph nodes {nN} edge pairs {npair} | {time.time()-t0:.0f}s",flush=True)
COSTS={l:plen2*((1.0-psh2)+l) for l in LAMS}
def dij(sn,cost,cutoff):
    dist={sn:0.0}; plen={sn:0.0}; psh={sn:0.0}; pe={}; pn={}
    h=[(0.0,sn)]
    while h:
        d,x=heapq.heappop(h)
        if d>cutoff: break
        if d>dist.get(x,1e18)+1e-9: continue
        for k in range(off[x],off[x+1]):
            y=int(adjN[k]); ei=int(adjE[k]); nd=d+cost[ei]
            if nd<dist.get(y,1e18)-1e-9:
                dist[y]=nd; plen[y]=plen[x]+plen2[ei]; psh[y]=psh[x]+pshm2[ei]; pe[y]=ei; pn[y]=x
                heapq.heappush(h,(nd,y))
    return dist,plen,psh,pe,pn
def walk(pe,pn,n,sn):
    eis=[]; x=n
    while x!=sn: ei=pe[x]; eis.append(ei); x=pn[x]
    return eis
ntree=cKDTree(node_xy)
bw=gpd.read_file(BW).to_crs(3414); rep=bw.representative_point()
bd,bi=ntree.query(np.column_stack([rep.x.values,rep.y.values]))
bw_w=bw.weight_weekday_14.values
node_blds=collections.defaultdict(float); nb=0
for k in range(len(bw)):
    if bd[k]<=SNAP_MAX and np.isfinite(bw_w[k]) and bw_w[k]>0 and int(bi[k])<nN:
        node_blds[int(bi[k])]+=float(bw_w[k]); nb+=1
st=gpd.read_file(ST).to_crs(3414)
sd,si=ntree.query(np.column_stack([st.geometry.x.values,st.geometry.y.values]))
stas=[(int(si[k]),float(st.tot_weekday_14.values[k]),st.source.values[k]) for k in range(len(st))
      if sd[k]<=SNAP_MAX and np.isfinite(st.tot_weekday_14.values[k]) and st.tot_weekday_14.values[k]>0 and int(si[k])<nN]
print(f"snapped: buildings {nb} | stations {len(stas)} | {time.time()-t0:.0f}s",flush=True)
flowL={l:collections.defaultdict(float) for l in LAMS}
AG={l:np.zeros(5) for l in LAMS}          # [vol, vol*L, vol*shadeM, vol*det, vol*facM]
AGS=np.zeros(5); flowS=collections.defaultdict(float); routed=0.0
tot_R=sum(s[1] for s in stas)
for qi,(sn,R,mode) in enumerate(stas):
    Dc=D_MRT if mode=='MRT' else D_BUS
    dS,plS,psS,peS,pnS=dij(sn,plen2,Dc)
    cat=[(n,node_blds[n],d) for n,d in dS.items() if n in node_blds and n!=sn and d>=20]
    if not cat: continue
    denom=sum(w*np.exp(-d/BETA) for n,w,d in cat)
    if denom<=0: continue
    runs={l:dij(sn,COSTS[l],(1.0+l)*Dc*3.0) for l in LAMS}   # loose cutoff guarantees reachability within the catchment
    for n,w,d in cat:
        vol=R*w*np.exp(-d/BETA)/denom
        if vol<=0: continue
        Ls=plS[n]; eS=walk(peS,pnS,n,sn)
        for ei in eS: flowS[ei]+=vol
        AGS+=[vol,vol*Ls,vol*psS[n],vol*1.0,vol*sum(plen2[ei] for ei in eS if pfac[ei])]
        routed+=vol
        for l in LAMS:
            _,plC,psC,peC,pnC=runs[l]
            if n not in plC: eis=eS; Lp=Ls; shm=psS[n]
            else: eis=walk(peC,pnC,n,sn); Lp=plC[n]; shm=psC[n]
            fd=flowL[l]; fm=0.0
            for ei in eis:
                fd[ei]+=vol
                if pfac[ei]: fm+=plen2[ei]
            AG[l]+=[vol,vol*Lp,vol*shm,vol*(Lp/Ls if Ls>0 else 1.0),vol*fm]
    if (qi+1)%500==0: print(f"  stations {qi+1}/{len(stas)} ({time.time()-t0:.0f}s)",flush=True)
print(f"assigned {routed:,.0f}/{tot_R:,.0f} ({100*routed/max(tot_R,1):.0f}%) | {time.time()-t0:.0f}s",flush=True)
# validation: recomputed shortest vs 4c stored values
fs_rep=np.zeros(len(edges))
for ei,v in flowS.items(): fs_rep[prow[ei]]=v
key=np.where(uu<vv,uu.astype(np.int64)*10**7+vv,vv.astype(np.int64)*10**7+uu)
lib=pd.DataFrame({'k':key,'f':edges.flow_short.values}).groupby('k')['f'].max()
repv=pd.DataFrame({'k':key,'f':fs_rep}).groupby('k')['f'].max()
J=pd.concat([lib,repv],axis=1,keys=['lib','rep']).fillna(0)
print(f"validation recomputed shortest vs 4c: r={np.corrcoef(J.lib,J.rep)[0,1]:.4f}",flush=True)
# output
pidx={(int(pk[j,0]),int(pk[j,1])):j for j in range(npair)}
rowpair=np.full(len(edges),-1,np.int64)
for i in range(len(edges)):
    u,v=uu[i],vv[i]
    if u==v: continue
    j=pidx.get((u,v) if u<v else (v,u))
    if j is not None: rowpair[i]=j
rows=[]
b=AGS; rows.append(dict(lam='shortest',mean_len=b[1]/b[0],mean_shade=b[2]/b[1],mean_detour=b[3]/b[0],fac_share=b[4]/b[1]))
for l in LAMS:
    pf=np.zeros(npair,np.float32)
    for ei,v in flowL[l].items(): pf[ei]=v
    full=np.where(rowpair>=0,pf[np.maximum(rowpair,0)],0).astype(np.float32)
    np.save(f"{OUT}\\flow_lam_{lid(l)}.npy",full[comp0])
    a=AG[l]; rows.append(dict(lam=l,mean_len=a[1]/a[0],mean_shade=a[2]/a[1],mean_detour=a[3]/a[0],fac_share=a[4]/a[1]))
    print(f"  flow_lam_{lid(l)}.npy max {full.max():,.0f} | shade {100*a[2]/a[1]:.1f}% detour {a[3]/a[0]:.3f} facility {100*a[4]/a[1]:.1f}%",flush=True)
S=pd.DataFrame(rows)
_csv=f"{OUT}\\step4_4f_flow_lam_summary.csv"
if os.path.exists(_csv):                      # incremental: merge with the existing levels, sorted by lambda descending (shortest on top)
    _old=pd.read_csv(_csv)
    S=pd.concat([_old[~_old.lam.astype(str).isin(S.lam.astype(str))],S],ignore_index=True)
    _k=S.lam.astype(str).map(lambda x: 1e9 if x=='shortest' else float(x))
    S=S.assign(_k=_k).sort_values('_k',ascending=False).drop(columns='_k')
S.to_csv(_csv,index=False,encoding='utf-8-sig')
print(S.to_string(index=False),flush=True)
print(f"DONE | {time.time()-t0:.0f}s",flush=True)
