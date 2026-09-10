# -*- coding: utf-8 -*-
"""Step4 city-wide visualisation: (1) network shade (full vs LOD1 only) (2) pedestrian flow (3) metric summary. pyenv."""
import numpy as np, geopandas as gpd, pandas as pd, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
SZ=r"D:\Claude\SVI_FFW\Shp\SG\SG_Subzone\SG_subzone boundary 2019_SVY21.shp"
t0=time.time()
e=gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg")
sz=gpd.read_file(SZ).to_crs(3414)
print(f"read {len(e)} edges {time.time()-t0:.0f}s",flush=True)
L=e.length.values; ok=np.isfinite(e.shade_full)&np.isfinite(e.shade_bld)
Wf=np.sum(L[ok]*e.shade_full.values[ok])/L[ok].sum(); Wb=np.sum(L[ok]*e.shade_bld.values[ok])/L[ok].sum()

# (1) shade comparison (two panels)
fig,ax=plt.subplots(1,2,figsize=(26,9))
for a,col,ttl in [(ax[0],'shade_full',f"Full urban shade system  network shade {100*Wf:.1f}%"),(ax[1],'shade_bld',f"LOD1 buildings only  network shade {100*Wb:.1f}%")]:
    sz.boundary.plot(ax=a,color='#eee',lw=0.3)
    e.plot(ax=a,column=col,cmap='RdYlBu',vmin=0,vmax=1,lw=0.25)
    a.set_title(ttl,fontsize=15); a.set_aspect('equal'); a.axis('off')
sm=plt.cm.ScalarMappable(cmap='RdYlBu',norm=plt.Normalize(0,1)); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,fraction=0.02,pad=0.01); cb.set_label("14:00 edge shade fraction",fontsize=12)
fig.suptitle(f"City-wide heatwave walkable network 14:00 shade coverage (full system vs LOD1 buildings only, gain {Wf/max(Wb,1e-9):.0f}x) - Singapore island-wide",fontsize=17,y=0.98)
plt.savefig(f"{OUT}\\step4_4_shade_SG.png",dpi=130,bbox_inches='tight'); plt.close(); print(f"(1) shade map {time.time()-t0:.0f}s",flush=True)

# (2) pedestrian flow
fig,ax=plt.subplots(figsize=(15,13))
sz.boundary.plot(ax=ax,color='#eee',lw=0.3)
f=e.flow_short.values; vmax=np.percentile(f[f>0],99)
order=np.argsort(f); e2=e.iloc[order]
lw=0.15+2.6*np.sqrt(np.clip(e2.flow_short.values,0,vmax)/max(vmax,1))
e2.plot(ax=ax,column='flow_short',cmap='inferno_r',vmin=0,vmax=vmax,lw=lw)
ax.set_aspect('equal'); ax.axis('off')
sm=plt.cm.ScalarMappable(cmap='inferno_r',norm=plt.Normalize(0,vmax)); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,fraction=0.03,pad=0.01); cb.set_label("Per-segment pedestrian flow (station access patronage betweenness, 14:00)",fontsize=12)
ax.set_title("City-wide heatwave pedestrian flow - station (MRT/BUS) to building access, station ridership x building weight\nSingapore island-wide, line width/colour = flow (shortest-route assignment)",fontsize=14)
plt.savefig(f"{OUT}\\step4_4_flow_SG.png",dpi=135,bbox_inches='tight'); plt.close(); print(f"(2) flow map {time.time()-t0:.0f}s",flush=True)

# (3) metric summary
od=pd.read_csv(f"{OUT}\\step4_4_od_metrics_SG.csv"); allr=od[od.btype=='__all__'].iloc[0]
fig,axs=plt.subplots(1,3,figsize=(18,5.5))
b=axs[0].bar(['Full urban\nshade system','LOD1\nbuildings only'],[100*Wf,100*Wb],color=['#1565c0','#bdbdbd'],width=0.6)
for r,v in zip(b,[Wf,Wb]): axs[0].text(r.get_x()+r.get_width()/2,100*v+1,f"{100*v:.1f}%",ha='center',fontsize=14,fontweight='bold')
axs[0].set_title(f"(1) City-wide network 14:00 shade coverage\nfull system = {Wf/max(Wb,1e-9):.0f}x buildings only",fontsize=13); axs[0].set_ylabel("Length-weighted shade fraction %"); axs[0].set_ylim(0,60)
ss=100*allr.short_shade; cs=100*allr.cool_shade
b=axs[1].bar(['Shortest','Coolest'],[ss,cs],color=['#ef6c00','#2e7d32'],width=0.6)
for r,v in zip(b,[ss,cs]): axs[1].text(r.get_x()+r.get_width()/2,v+0.6,f"{v:.1f}%",ha='center',fontsize=14,fontweight='bold')
axs[1].set_title(f"(2) Access-route shade fraction (flow-weighted)\nheat-avoiding routing +{cs-ss:.1f}pp (city-wide mean, limited by low network redundancy)",fontsize=12); axs[1].set_ylabel("Route shade fraction %"); axs[1].set_ylim(0,70)
det=allr.detour; sl=allr.short_len; cl=allr.cool_len
b=axs[2].bar(['Shortest','Coolest'],[sl,cl],color=['#ef6c00','#2e7d32'],width=0.6)
for r,v in zip(b,[sl,cl]): axs[2].text(r.get_x()+r.get_width()/2,v+3,f"{v:.0f}m",ha='center',fontsize=13,fontweight='bold')
axs[2].set_title(f"(3) Access walking length (flow-weighted)\ndetour {det:.2f}x (first/last mile, mean {sl:.0f}m)",fontsize=12); axs[2].set_ylabel("Mean path length m")
fig.suptitle("Step4 city-wide heat-avoiding walkable network metric summary - Singapore island-wide",fontsize=16,y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}\\step4_4_summary_SG.png",dpi=135,bbox_inches='tight'); plt.close(); print(f"(3) summary figure {time.time()-t0:.0f}s",flush=True)
print("city-wide figures saved",flush=True)
