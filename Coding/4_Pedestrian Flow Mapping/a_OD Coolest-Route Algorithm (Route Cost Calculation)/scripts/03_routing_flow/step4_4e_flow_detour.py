# -*- coding: utf-8 -*-
"""tau-capped heat-avoiding flow reassignment: OD demand exactly mirrors step4_4c (station 14:00 ridership x building weight x exp(-d/350),
MRT 800 m / BUS 400 m, SNAP 120 m, Ls<20 dropped); only the path rule is changed to "the coolest route with detour <= tau"
(mirrors the web coolDetour: scan lambda in ascending order, take the first that satisfies len <= tau*Ls+1 = coolest within the cap; fall back to the shortest route).
tau in {1, 1.2, 1.5, 2}. Validation: recompute flow_short in the same framework and compare with the 4c stored values. Output: 4 per-edge npy files (comp0 row order)
+ summary CSV + curve / difference figures. All-facility (All) network. pyenv."""
import numpy as np, geopandas as gpd, pandas as pd, time, collections, heapq
from scipy.spatial import cKDTree
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
BW=r"D:\Claude\UNA\Patronage_Flow\output\building_hourly_weight.gpkg"
ST=r"D:\Claude\SVI_FFW\Shp\SG\POI\station_hourly_ridership.gpkg"
BETA=350.0; SNAP_MAX=120.0; D_MRT=800.0; D_BUS=400.0
LAMS=[0,0.03,0.08,0.15,0.35,0.7,1.4,3,8,20,60,200]   # lambda ascending = shade priority descending
TAUS=[1.0,1.2,1.5,2.0]
t0=time.time()
edges=gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg")
nodes=gpd.read_file(f"{OUT}\\step4_4_nodes_SG.gpkg")
node_xy=np.column_stack([nodes.geometry.x.values,nodes.geometry.y.values])
uu=edges.u.values.astype(int); vv=edges.v.values.astype(int)
ln=edges.length.values.astype(float); sf=np.nan_to_num(edges.shade_full.values,nan=0.0)
comp0=np.where(edges.comp.values==0)[0]
EFAC=np.load(f"{OUT}\\edge_facility_SG.npy")
EF_full=np.zeros(len(edges),np.uint8); EF_full[comp0]=EFAC
# parallel edges: keep the shortest (mirrors the 4c replace-if-shorter), record the kept row index
pairs={}
for i in range(len(edges)):
    u,v=uu[i],vv[i]
    if u==v: continue
    key=(u,v) if u<v else (v,u)
    if key in pairs and pairs[key][0]<=ln[i]: continue
    pairs[key]=(ln[i],sf[i],i)
npair=len(pairs)
pk=np.array(list(pairs.keys())); pval=list(pairs.values())
plen2=np.array([x[0] for x in pval]); psh2=np.array([x[1] for x in pval]); prow=np.array([x[2] for x in pval])
pshm2=plen2*psh2                                     # shaded metres per edge pair
pfac=np.isin(EF_full[prow],[3,4])                    # facility micro-network (arcade/linkway) pairs
pfarc=EF_full[prow]==3; pflkw=EF_full[prow]==4
nN=int(pk.max())+1
deg=np.zeros(nN+1,np.int64); np.add.at(deg,pk[:,0]+1,1); np.add.at(deg,pk[:,1]+1,1)
off=np.cumsum(deg); adjN=np.zeros(off[-1],np.int64); adjE=np.zeros(off[-1],np.int64); cur=off[:-1].copy()
for j in range(npair):
    a,b=pk[j,0],pk[j,1]
    adjN[cur[a]]=b; adjE[cur[a]]=j; cur[a]+=1
    adjN[cur[b]]=a; adjE[cur[b]]=j; cur[b]+=1
print(f"graph nodes {nN} edge pairs {npair} (mirrors 4c, shortest kept) | {time.time()-t0:.0f}s",flush=True)
COSTS=[plen2*((1.0-psh2)+l) for l in LAMS]
def dij(sn,cost,cutoff):
    dist={sn:0.0}; plen={sn:0.0}; psh={sn:0.0}; pe={}; pn={}
    h=[(0.0,sn)]
    while h:
        d,x=heapq.heappop(h)
        if d>dist.get(x,1e18)+1e-9 or d>cutoff:
            if d>cutoff: break
            continue
        for k in range(off[x],off[x+1]):
            y=int(adjN[k]); ei=int(adjE[k]); nd=d+cost[ei]
            if nd<dist.get(y,1e18)-1e-9:
                dist[y]=nd; plen[y]=plen[x]+plen2[ei]; psh[y]=psh[x]+pshm2[ei]; pe[y]=ei; pn[y]=x
                heapq.heappush(h,(nd,y))
    return dist,plen,psh,pe,pn
def walk(pe,pn,n,sn):
    eis=[]; x=n
    while x!=sn:
        ei=pe[x]; eis.append(ei); x=pn[x]
    return eis
ntree=cKDTree(node_xy)
bw=gpd.read_file(BW).to_crs(3414); rep=bw.representative_point()
bd,bi=ntree.query(np.column_stack([rep.x.values,rep.y.values]))
bw_w=bw.weight_weekday_14.values
node_blds=collections.defaultdict(float)             # node -> sum of w (building-type summary not needed in this run)
nb=0
for k in range(len(bw)):
    if bd[k]<=SNAP_MAX and np.isfinite(bw_w[k]) and bw_w[k]>0 and int(bi[k])<nN:
        node_blds[int(bi[k])]+=float(bw_w[k]); nb+=1
st=gpd.read_file(ST).to_crs(3414)
sd,si=ntree.query(np.column_stack([st.geometry.x.values,st.geometry.y.values]))
stas=[(int(si[k]),float(st.tot_weekday_14.values[k]),st.source.values[k]) for k in range(len(st))
      if sd[k]<=SNAP_MAX and np.isfinite(st.tot_weekday_14.values[k]) and st.tot_weekday_14.values[k]>0 and int(si[k])<nN]
print(f"snapped: buildings {nb} | stations {len(stas)} | {time.time()-t0:.0f}s",flush=True)

flowS=collections.defaultdict(float)
flowT={t:collections.defaultdict(float) for t in TAUS}
AG={t:np.zeros(5) for t in TAUS}                     # [vol, vol*L, vol*shadeM, vol*det, vol*facM]
AGS=np.zeros(5)
routed=0.0; tot_R=sum(s[1] for s in stas)
for qi,(sn,R,mode) in enumerate(stas):
    Dc=D_MRT if mode=='MRT' else D_BUS
    dS,plS,psS,peS,pnS=dij(sn,COSTS[0]*0+plen2,Dc)   # shortest (cost=len)
    cat=[(n,node_blds[n],d) for n,d in dS.items() if n in node_blds and n!=sn and d>=20]
    if not cat: continue
    denom=sum(w*np.exp(-d/BETA) for n,w,d in cat)
    if denom<=0: continue
    runs=[]
    for li,lam in enumerate(LAMS):
        runs.append(dij(sn,COSTS[li],(1+lam)*Dc*2.2))
    for n,w,d in cat:
        vol=R*w*np.exp(-d/BETA)/denom
        if vol<=0: continue
        Ls=plS[n]
        eS=walk(peS,pnS,n,sn)
        for ei in eS: flowS[ei]+=vol
        AGS+=[vol,vol*Ls,vol*psS[n],vol*1.0,vol*sum(plen2[ei] for ei in eS if pfac[ei])]
        routed+=vol
        for t in TAUS:
            cap=t*Ls+1.0; ch=-1
            for li in range(len(LAMS)):
                pl=runs[li][1].get(n)
                if pl is not None and pl<=cap: ch=li; break
            if ch<0:
                eis=eS; Lp=Ls; shm=psS[n]
            else:
                _,plC,psC,peC,pnC=runs[ch]
                eis=walk(peC,pnC,n,sn); Lp=plC[n]; shm=psC[n]
            fd=flowT[t]
            fm=0.0
            for ei in eis:
                fd[ei]+=vol
                if pfac[ei]: fm+=plen2[ei]
            AG[t]+=[vol,vol*Lp,vol*shm,vol*(Lp/Ls),vol*fm]
    if (qi+1)%800==0: print(f"  stations {qi+1}/{len(stas)} ({time.time()-t0:.0f}s)",flush=True)
print(f"assigned {routed:,.0f}/{tot_R:,.0f} ({100*routed/max(tot_R,1):.0f}%) | {time.time()-t0:.0f}s",flush=True)

# validation: recomputed flow_short vs 4c stored values
fs_rep=np.zeros(len(edges));
for ei,v in flowS.items(): fs_rep[prow[ei]]=v
# 4c writes the pair flow to all rows with the same (u,v); the recomputation writes only the kept row, so align the basis by comparing at pair level
key=np.where(uu<vv,uu.astype(np.int64)*10**7+vv,vv.astype(np.int64)*10**7+uu)
lib=pd.DataFrame({'k':key,'f':edges.flow_short.values}).groupby('k')['f'].max()
repv=pd.DataFrame({'k':key,'f':fs_rep}).groupby('k')['f'].max()
J=pd.concat([lib,repv],axis=1,keys=['lib','rep']).fillna(0)
cc=np.corrcoef(J.lib,J.rep)[0,1]
print(f"validation recomputed flow_short vs 4c stored: pair correlation r={cc:.4f} | total {J.rep.sum():,.0f} vs {J.lib.sum():,.0f}",flush=True)

# summary
rows=[]
b=AGS; rows.append(dict(tau='shortest',mean_len=b[1]/b[0],mean_shade=b[2]/b[1],mean_detour=b[3]/b[0],fac_share=b[4]/b[1]))
for t in TAUS:
    a=AG[t]; rows.append(dict(tau=t,mean_len=a[1]/a[0],mean_shade=a[2]/a[1],mean_detour=a[3]/a[0],fac_share=a[4]/a[1]))
S=pd.DataFrame(rows); S.to_csv(f"{OUT}\\step4_4e_flow_detour_summary.csv",index=False,encoding='utf-8-sig')
print(S.to_string(index=False),flush=True)
# per-edge npy (comp0 row order; the pair flow is written back to all comp0 rows of that pair, same basis as 4c)
pairflow_col={}
for t in TAUS:
    full=np.zeros(len(edges),np.float32)
    fd=flowT[t]
    pf=np.zeros(npair,np.float32)
    for ei,v in fd.items(): pf[ei]=v
    # map: look up the pair value for each row
    rowpair=np.full(len(edges),-1,np.int64)
    pidx={ (int(pk[j,0]),int(pk[j,1])):j for j in range(npair) }
    for i in range(len(edges)):
        u,v=uu[i],vv[i]
        if u==v: continue
        j=pidx.get((u,v) if u<v else (v,u))
        if j is not None: full[i]=pf[j]
    np.save(f"{OUT}\\flow_cooltau_{int(t*100)}.npy",full[comp0])
    pairflow_col[t]=full
    print(f"npy flow_cooltau_{int(t*100)} max {full.max():,.0f}",flush=True)

# figures: curves + difference maps (tau=1.2 and 2.0 vs shortest)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(1,3,figsize=(15,4.4))
xt=[1.0,1.2,1.5,2.0]
axs[0].plot(xt,[AG[t][4]/AG[t][1]*100 for t in TAUS],'o-',color='#d4322c',label='Facility micro-network (arcade + linkway)')
axs[0].axhline(AGS[4]/AGS[1]*100,ls='--',c='#888',label='Shortest baseline')
axs[0].set_xlabel('Detour cap tau');axs[0].set_ylabel('Person-metre share %');axs[0].set_title('Person-metre share carried by the facility micro-network');axs[0].legend(fontsize=8)
axs[1].plot(xt,[AG[t][2]/AG[t][1]*100 for t in TAUS],'o-',color='#0F6E56')
axs[1].axhline(AGS[2]/AGS[1]*100,ls='--',c='#888')
axs[1].set_xlabel('Detour cap tau');axs[1].set_ylabel('Shade fraction %');axs[1].set_title('Flow-weighted shade fraction (shaded share of walked distance)')
axs[2].plot(xt,[AG[t][3]/AG[t][0] for t in TAUS],'o-',color='#c0651a')
axs[2].set_xlabel('Detour cap tau');axs[2].set_ylabel('Mean detour x');axs[2].set_title('Flow-weighted mean detour ratio')
plt.suptitle('Effect of tau-capped heat-avoiding route choice on city-wide access flow (All-facility network, 14:00)',fontsize=12)
plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig(f"{OUT}\\step4_4e_flow_detour_curves.png",dpi=160); plt.close()
gx=edges.geometry.values
fig,axs=plt.subplots(1,2,figsize=(17,7.5))
for ax,t in zip(axs,[1.2,2.0]):
    dfl=pairflow_col[t]-np.array([edges.flow_short.values[i] for i in range(len(edges))])
    idx=np.where(np.abs(dfl)>3)[0]
    o=np.argsort(np.abs(dfl[idx])); idx=idx[o]
    for i in idx[::max(1,len(idx)//12000)]:
        xs,ys=gx[i].xy
        ax.plot(xs,ys,color=('#c0392b' if dfl[i]>0 else '#2166ac'),lw=min(2.8,0.3+abs(dfl[i])/150),alpha=0.7)
    ax.set_title(f'tau={t}x heat-avoiding - Shortest flow difference (red = increase, blue = decrease, |delta| > 3 ppl)')
    ax.set_aspect('equal'); ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}\\step4_4e_flow_detour_diffmap.png",dpi=150); plt.close()
print(f"2 figures saved | {time.time()-t0:.0f}s",flush=True)
print("DONE",flush=True)
