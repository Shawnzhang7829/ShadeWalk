# -*- coding: utf-8 -*-
"""ShadeWalk mobile version (real functionality, not a demo): the same city-wide network + the same Dijkstra / adaptive coolDetour,
UI styled like a phone navigation app (header / OD bar / time chips / map / Coolest and Shortest cards / route shade bar / Start walking animation).
Slimmed down: routing core only (egeom/el/es/efac/eu/ev/nx/ny) + 3 shade raster frames (08/14/17 h, switched by chips) + demoOD,
~31 MB; desktop features (3D / trees / building rings / flow / scenarios etc.) are not included in the mobile version. Routing and shade % use the fixed 14:00 basis (same as desktop).
Output webapp/nav_app_mobile.html (EN). pyenv."""
import geopandas as gpd, numpy as np, json, time
from pyproj import Transformer
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"
WEB=f"{OUT}\\webapp"
t0=time.time()
TO=Transformer.from_crs(3414,3857,always_xy=True)
def to_m(xs,ys):
    a,b=TO.transform(np.asarray(xs,float),np.asarray(ys,float)); return np.asarray(a),np.asarray(b)
e=gpd.read_file(f"{OUT}\\step4_4_edges_flow_SG.gpkg"); e=e[e['comp']==0].reset_index(drop=True)
nd=gpd.read_file(f"{OUT}\\step4_4_nodes_SG.gpkg"); maxid=int(nd['node'].max())
CX=np.full(maxid+1,np.nan); CY=np.full(maxid+1,np.nan)
CX[nd['node'].values]=nd.geometry.x.values; CY[nd['node'].values]=nd.geometry.y.values
u=e['u'].values.astype(np.int64); v=e['v'].values.astype(np.int64)
used=np.unique(np.concatenate([u,v])); remap=np.full(maxid+1,-1,np.int64); remap[used]=np.arange(len(used))
U=remap[u].astype(int); V=remap[v].astype(int)
nmx,nmy=to_m(CX[used],CY[used]); ox=float(np.floor(nmx.min())); oy=float(np.floor(nmy.min()))
NX=np.round(nmx-ox).astype(int); NY=np.round(nmy-oy).astype(int)
EL=np.round(e['length'].values).astype(int)
ES=np.clip(np.round(np.nan_to_num(e['shade_full'].values)*100),0,100).astype(int)
EFAC=np.load(f"{OUT}\\edge_facility_SG.npy")
gs=gpd.GeoSeries(e.geometry.values,crs=3414).simplify(2.0)
cdf=gs.get_coordinates(); amx,amy=to_m(cdf['x'].values,cdf['y'].values)
AX=np.round(amx-ox).astype(int); AY=np.round(amy-oy).astype(int); iv=cdf.index.values
st_=np.searchsorted(iv,np.arange(len(e)),'left'); en_=np.searchsorted(iv,np.arange(len(e)),'right')
EGEOM=[]
for i in range(len(e)):
    s,t=st_[i],en_[i]; flat=np.empty(2*(t-s),int); flat[0::2]=AX[s:t]; flat[1::2]=AY[s:t]; EGEOM.append(flat.tolist())
print(f"network edges {len(e)} | {time.time()-t0:.0f}s",flush=True)
# --- POV massing data (same basis as the desktop version: building / arcade / linkway rings + heights, city-wide tree points) ---
from shapely import force_2d
ARC=r"D:\Claude\SVI_FFW\Shp\SG\step2_arcade_sg.gpkg"
LKW=r"D:\Claude\SVI_FFW\Shp\SG\covered_linkway_SG_island_tv_pednet_bridged.gpkg"
BLDG=r"D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg"
def rings_g(g,simp,hcols=None):
    try: g=g.to_crs(3414)
    except Exception: g=g.set_crs(3414,allow_override=True)
    out=[]; hh=[[] for _ in (hcols or [])]; hv=[np.asarray(g[c].values,float) for c in (hcols or [])]
    for gi,geom in enumerate(g.geometry):
        if geom is None: continue
        geom=force_2d(geom)
        def _flat(rr):
            c=np.asarray(rr.coords)
            if len(c)<4: return None
            mx,my=to_m(c[:,0],c[:,1]); xs=np.round(mx-ox).astype(int); ys=np.round(my-oy).astype(int)
            f=np.empty(2*len(xs),int); f[0::2]=xs; f[1::2]=ys; return f.tolist()
        for p in (geom.geoms if geom.geom_type=='MultiPolygon' else [geom]):
            if p.geom_type!='Polygon': continue
            ext=_flat(p.exterior.simplify(simp))
            if ext is None: continue
            rl=[ext]
            for hole in p.interiors:
                hf=_flat(hole.simplify(simp))
                if hf is not None: rl.append(hf)
            out.append(rl)
            for _k in range(len(hcols or [])): hh[_k].append(hv[_k][gi])
    if hcols: return out,hh
    return out
arcR,arcHH=rings_g(gpd.read_file(ARC),0.5,hcols=['bld_h']); arcBH=[int(round(x)) if (x==x and x>4) else 6 for x in arcHH[0]]
lkwR=rings_g(gpd.read_file(LKW),1.0)
bldR,bldHH=rings_g(gpd.read_file(BLDG),3.0,hcols=['height']); bldH=[int(round(x)) if (x==x and x>0) else 10 for x in bldHH[0]]
_gt=gpd.read_file(r"D:\Claude\SVI_FFW\Shp\SG\SG point tree\Point tree.shp")
try: _gt=_gt.to_crs(3414)
except Exception: pass
_tmx,_tmy=to_m(_gt.geometry.x.values,_gt.geometry.y.values)
_th=np.clip(np.round(np.nan_to_num(_gt['height_est'].values.astype(float),nan=8)),3,30).astype(int)
_tg=_gt['girth_size'].map({'XS':0,'S':1,'M':2,'L':3}).fillna(1).astype(int).values
tree3d=dict(x=np.round(_tmx-ox).astype(int).tolist(),y=np.round(_tmy-oy).astype(int).tolist(),h=_th.tolist(),g=_tg.tolist())
print(f"massing: arcade rings {len(arcR)} linkway rings {len(lkwR)} building rings {len(bldR)} trees {len(_th)} | {time.time()-t0:.0f}s",flush=True)
J=json.load(open(f"{OUT}\\_hourly_layers.json"))
HSEL=[0,6,9]                       # 8:00 / 14:00 / 17:00
shadH=[J['shad'][i] for i in HSEL]
ex=J['extS_abs']                   # 3414 absolute [xmin,ymin,xmax,ymax]
c1=to_m([ex[0],ex[2]],[ex[1],ex[3]])
extS=[int(round(c1[0][0]-ox)),int(round(c1[1][0]-oy)),int(round(c1[0][1]-ox)),int(round(c1[1][1]-oy))]
demoOD=json.load(open(f"{OUT}\\_demo_scenarios.json",encoding='utf-8'))
data=dict(ox=ox,oy=oy,egeom=EGEOM,el=EL.tolist(),es=ES.tolist(),efac=EFAC.tolist(),
          eu=U.tolist(),ev=V.tolist(),nx=NX.tolist(),ny=NY.tolist(),
          shadH=shadH,extS=extS,hrs=[8,14,17],demoOD=demoOD,
          bld=bldR,bldH=bldH,arc=arcR,arcBH=arcBH,lkw=lkwR,tree3d=tree3d)
dj=json.dumps(data,separators=(',',':'))
print(f"JSON {len(dj)/1e6:.1f} MB | {time.time()-t0:.0f}s",flush=True)

HTML='''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>ShadeWalk · Heat-Avoid Nav</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><rect x='1' y='1' width='22' height='22' rx='5.5' fill='%232e8b46'/><rect x='4.5' y='5' width='15' height='2.1' rx='0.6' fill='%23ffd23f'/><line x1='6.3' y1='6.9' x2='6.3' y2='17.6' stroke='%23ffd23f' stroke-width='1.4' stroke-linecap='round'/><line x1='17.7' y1='6.9' x2='17.7' y2='17.6' stroke='%23ffd23f' stroke-width='1.4' stroke-linecap='round'/><circle cx='12' cy='10.2' r='2.05' fill='%23fff'/><rect x='10.75' y='12.7' width='2.5' height='6.5' rx='1.25' fill='%23fff'/></svg>">
<link href="https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html,body{height:100%;overflow:hidden}
body{font-family:-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;background:#f6f5f1;color:#1d1c1a;display:flex;flex-direction:column}
#hdr{display:flex;align-items:center;gap:9px;padding:10px 12px 7px;background:#fff;border-bottom:1px solid #eceae3}
#logo{width:34px;height:34px;flex:0 0 auto;display:flex;align-items:center;justify-content:center}
#hdr h1{font-size:15px;line-height:1.15}
#hdr .sub{font-size:10.5px;color:#8a887f}
#badge{margin-left:auto;background:#fdeceb;color:#b3372c;font-size:10px;font-weight:600;padding:5px 8px;border-radius:8px;text-align:center;line-height:1.25;flex:0 0 auto}
#odbox{background:#fff;padding:7px 12px 6px;border-bottom:1px solid #eceae3}
.odrow{display:flex;align-items:center;gap:8px;border:1px solid #e2e0d7;border-radius:10px;padding:7px 10px;font-size:12.5px;color:#555;background:#fbfaf7}
.odrow+.odrow{margin-top:6px}
.odrow b{color:#1d1c1a;font-weight:600}
.dot{width:10px;height:10px;border-radius:50%;flex:0 0 auto}
#odbtns{display:flex;gap:6px;margin-top:7px}
.chip{font-size:11.5px;padding:6px 11px;border-radius:9px;border:1px solid #d8d6cc;background:#fff;color:#444;font-weight:500}
.chip.on{border-color:#0F6E56;color:#0F6E56;background:#eef6f2;font-weight:700}
.chip:active{transform:scale(.96)}
#hrrow{display:flex;align-items:center;gap:6px;padding:7px 12px;background:#fff;border-bottom:1px solid #eceae3}
#hrrow .lbl{margin-left:auto;font-size:11.5px;color:#444;display:flex;align-items:center;gap:5px}
#map{flex:1;min-height:34vh;position:relative}
#foot{background:#fff;border-top:1px solid #e5e3da;padding:8px 12px calc(9px + env(safe-area-inset-bottom));box-shadow:0 -3px 14px rgba(0,0,0,.06)}
#cards{display:flex;gap:8px}
.card{flex:1;border:1.6px solid #dcdad0;border-radius:12px;padding:8px 10px;background:#fff}
.card.sel{border-color:#0F6E56;box-shadow:0 0 0 2px #0f6e5622}
.card.sh.sel{border-color:#b06a1a;box-shadow:0 0 0 2px #b06a1a22}
.card .top{display:flex;align-items:center;gap:5px;font-size:11.5px;font-weight:700}
.card .tag{font-size:8.5px;font-weight:600;padding:2px 5px;border-radius:6px;margin-left:auto}
.card .big{font-size:21px;font-weight:700;margin:2px 0 1px}
.card .big span{font-size:10.5px;font-weight:500;color:#8a887f}
.card .mut{font-size:10px;color:#77756c;line-height:1.35}
#profwrap{margin-top:7px;background:#f6f4ee;border-radius:10px;padding:7px 9px 8px}
#profhead{display:flex;justify-content:space-between;font-size:10.5px;color:#55534b;margin-bottom:4px}
#profbar{display:flex;height:9px;border-radius:5px;overflow:hidden;background:#eae8df}
#insight{font-size:10.3px;color:#55534b;margin-top:5px;line-height:1.45}
#startBtn{width:100%;margin-top:8px;border:none;border-radius:11px;padding:12px;font-size:14.5px;font-weight:700;color:#fff;background:#0F6E56;letter-spacing:.2px}
#startBtn.sh{background:#9a5c14}
#startBtn:disabled{background:#c9c7bd}
#startBtn:active{transform:scale(.985)}
#tip{position:absolute;left:50%;top:8px;transform:translateX(-50%);background:#1d1c1a;color:#fff;font-size:11px;padding:6px 12px;border-radius:16px;z-index:9;display:none;max-width:86%;text-align:center}
#bPov{display:none;position:absolute;left:50%;bottom:12px;transform:translateX(-50%);z-index:7;border:none;border-radius:18px;padding:9px 15px;font-size:12.5px;font-weight:700;color:#fff;background:rgba(29,28,26,.88)}
#bPov:active{transform:translateX(-50%) scale(.96)}
#pov{display:none;position:absolute;inset:0;z-index:8;background:#c2d8ea}
#pov canvas{width:100%;height:100%;display:block}
#povX{position:absolute;top:10px;right:10px;z-index:9;border:none;border-radius:16px;padding:8px 13px;font-size:12px;font-weight:700;color:#fff;background:rgba(29,28,26,.88)}
#povHint{position:absolute;left:8px;bottom:8px;z-index:9;font-size:9.5px;color:#fff;background:rgba(29,28,26,.6);padding:4px 9px;border-radius:8px}
.mut2{color:#8a887f}
</style></head><body>
<div id="hdr">
  <div id="logo"><svg viewBox="0 0 24 24" width="34" height="34" aria-hidden="true"><rect x="1" y="1" width="22" height="22" rx="5.5" fill="#2e8b46"/><rect x="4.5" y="5" width="15" height="2.1" rx="0.6" fill="#ffd23f"/><line x1="6.3" y1="6.9" x2="6.3" y2="17.6" stroke="#ffd23f" stroke-width="1.4" stroke-linecap="round"/><line x1="17.7" y1="6.9" x2="17.7" y2="17.6" stroke="#ffd23f" stroke-width="1.4" stroke-linecap="round"/><circle cx="12" cy="9.7" r="1.62" fill="#fff"/><path d="M11.5 12.5C9.9 14 9.85 16.3 11.5 19.4Q12 20.1 12.5 19.4C14.15 16.3 14.1 14 12.5 12.5Q12 11.95 11.5 12.5Z" fill="#fff"/></svg></div>
  <div><h1>ShadeWalk · Heat-Avoid Nav</h1><div class="sub">Singapore · transit access on foot</div></div>
  <div id="badge">feels 41°C · UV<br>extreme</div>
</div>
<div id="odbox">
  <div class="odrow"><span class="dot" style="background:#0F6E56"></span><span id="oTxt">Tap map to set <b>origin</b></span></div>
  <div class="odrow"><span class="dot" style="background:#c0392b"></span><span id="dTxt">then tap again to set <b>destination</b></span></div>
  <div id="odbtns">
    <button class="chip" id="bDemo">▶ Demo</button>
    <button class="chip" id="bSwap">⇄ Swap</button>
    <button class="chip" id="bClear">✕ Clear</button>
  </div>
</div>
<div id="hrrow">
  <button class="chip hb on" data-i="1">14:00</button><button class="chip hb" data-i="0">08:00</button><button class="chip hb" data-i="2">17:00</button>
  <label class="lbl"><input type="checkbox" id="cbShade" checked> Shade</label>
</div>
<div id="map"><div id="tip"></div>
 <button id="bPov">👁 First-person</button>
 <div id="pov"><canvas id="povCv"></canvas><button id="povX">🗺 Map view</button><div id="povHint">Pedestrian view · gray buildings · green trees · rust arcades · blue linkways</div></div>
</div>
<div id="foot">
  <div id="cards">
    <div class="card sel" id="cardC"><div class="top" style="color:#0F6E56">☂ Coolest<span class="tag" style="background:#e3f2ec;color:#0F6E56">recommended</span></div><div class="big" id="cPct">–<span> shaded</span></div><div class="mut" id="cSub">tap map to route</div></div>
    <div class="card sh" id="cardS"><div class="top" style="color:#b06a1a">⚡ Shortest<span class="tag" style="background:#f7ecdd;color:#b06a1a">direct</span></div><div class="big" id="sPct">–<span> shaded</span></div><div class="mut" id="sSub">&nbsp;</div></div>
  </div>
  <div id="profwrap"><div id="profhead"><span>Shade profile along route</span><span id="profR">–</span></div><div id="profbar"></div><div id="insight">Routing &amp; shade % use the 14:00 SOLWEIG layer (same basis as the desktop app). Time chips switch the shade overlay only.</div></div>
  <button id="startBtn" disabled>▷ Start · coolest route</button>
</div>
<script src="https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script>
const D=__DATA__;
const Rm=6378137,OX=D.ox,OY=D.oy,LAM=0.2,DETOUR=1.2;
function ll(x,y){let X=x+OX,Y=y+OY;return [X/Rm*180/Math.PI,(2*Math.atan(Math.exp(Y/Rm))-Math.PI/2)*180/Math.PI];}
function edgeLL(g){let a=[];for(let k=0;k<g.length;k+=2)a.push(ll(g[k],g[k+1]));return a;}
const NE=D.eu.length,NN=D.nx.length,EL=D.el,ES=D.es,U=D.eu,V=D.ev;
const deg=new Int32Array(NN);for(let i=0;i<NE;i++){deg[U[i]]++;deg[V[i]]++;}
const off=new Int32Array(NN+1);for(let i=0;i<NN;i++)off[i+1]=off[i]+deg[i];
const aN=new Int32Array(NE*2),aE=new Int32Array(NE*2),cu=off.slice();
for(let i=0;i<NE;i++){let a=U[i],b=V[i];aN[cu[a]]=b;aE[cu[a]++]=i;aN[cu[b]]=a;aE[cu[b]++]=i;}
const GC=300,grid=new Map();for(let i=0;i<NN;i++){let k=((D.nx[i]/GC)|0)+'_'+((D.ny[i]/GC)|0);(grid.get(k)||grid.set(k,[]).get(k)).push(i);}
function snapEdge(mx,my){var cx=(mx/GC)|0,cy=(my/GC)|0,seen=new Set(),best=-1,bd=1e18,bstart=-1,bcum=0,bel=0,bfx=0,bfy=0;
 for(var dx=-2;dx<=2;dx++)for(var dy=-2;dy<=2;dy++){var a=grid.get((cx+dx)+'_'+(cy+dy));if(!a)continue;
  for(const ni of a){for(var q=off[ni];q<off[ni+1];q++){var ei=aE[q];if(seen.has(ei))continue;seen.add(ei);
   var g=D.egeom[ei];var su=(g[0]-D.nx[U[ei]])**2+(g[1]-D.ny[U[ei]])**2,sv=(g[0]-D.nx[V[ei]])**2+(g[1]-D.ny[V[ei]])**2;var sN=su<=sv?U[ei]:V[ei];var cum=0;
   for(var k=0;k+3<g.length;k+=2){var ax=g[k],ay=g[k+1],vx=g[k+2]-ax,vy=g[k+3]-ay,L2=vx*vx+vy*vy;var t=L2>0?((mx-ax)*vx+(my-ay)*vy)/L2:0;t=t<0?0:t>1?1:t;var px=ax+t*vx,py=ay+t*vy,d=(mx-px)**2+(my-py)**2,sl=Math.sqrt(L2);if(d<bd){bd=d;best=ei;bcum=cum+t*sl;bstart=sN;bel=EL[ei];bfx=px;bfy=py;}cum+=sl;}}}}
 if(best<0)return null;
 var g2=D.egeom[best],tot=0;for(var k2=0;k2+3<g2.length;k2+=2)tot+=Math.hypot(g2[k2+2]-g2[k2],g2[k2+3]-g2[k2+1]);
 var duS=tot>0?(bcum/tot)*bel:0,duE=bel-duS,node=(duS<=duE)?bstart:(bstart===U[best]?V[best]:U[best]);
 return {node:node,fx:bfx,fy:bfy};}
function dij(s,t,mode,lam){const LM=(lam===undefined?LAM:lam);const dist=new Float64Array(NN).fill(1e18),pe=new Int32Array(NN).fill(-1),pn=new Int32Array(NN).fill(-1);dist[s]=0;const hd=[s],hk=[0];
 function up(i){while(i>0){let p=(i-1)>>1;if(hk[p]<=hk[i])break;[hk[p],hk[i]]=[hk[i],hk[p]];[hd[p],hd[i]]=[hd[i],hd[p]];i=p;}}
 function dn(i){let n=hd.length;for(;;){let l=2*i+1,r=l+1,m=i;if(l<n&&hk[l]<hk[m])m=l;if(r<n&&hk[r]<hk[m])m=r;if(m===i)break;[hk[m],hk[i]]=[hk[i],hk[m]];[hd[m],hd[i]]=[hd[i],hd[m]];i=m;}}
 while(hd.length){let d=hk[0],x=hd[0],L=hd.length-1;hd[0]=hd[L];hk[0]=hk[L];hd.pop();hk.pop();if(hd.length)dn(0);if(d>dist[x])continue;if(x===t)break;
  for(let p=off[x];p<off[x+1];p++){let y=aN[p],ei=aE[p];let c=mode==='cool'?EL[ei]*((1-ES[ei]/100)+LM):EL[ei];let n2=d+c;if(n2<dist[y]){dist[y]=n2;pe[y]=ei;pn[y]=x;hd.push(y);hk.push(n2);up(hd.length-1);}}}
 if(dist[t]>=1e18)return null;let coords=[],eis=[],len=0,sh=0,x=t;
 while(pn[x]>=0){let ei=pe[x];eis.push(ei);len+=EL[ei];sh+=EL[ei]*ES[ei]/100;coords.push(edgeLL(D.egeom[ei]));x=pn[x];}
 eis.reverse();return {coords:coords,len:len,shade:len>0?sh/len:0,eis:eis};}
function coolDetour(O,Dst,s){if(!s)return null;let c0=dij(O,Dst,'cool',0);
 if(c0&&c0.len<=s.len*DETOUR+1)return c0;
 let lo=0,hi=3.0,ch=dij(O,Dst,'cool',hi);
 while((!ch||ch.len>s.len*DETOUR+1)&&hi<3000){lo=hi;hi*=4;ch=dij(O,Dst,'cool',hi);}
 if(!ch||ch.len>s.len*DETOUR+1)return s;
 let best=ch;
 for(let it=0;it<22;it++){let mid=(lo+hi)/2,c=dij(O,Dst,'cool',mid);
  if(c&&c.len<=s.len*DETOUR+1){hi=mid;best=c;}else{lo=mid;}}
 return best;}
const FACCOL={0:'#f4e7b0',1:'#8a8a8a',2:'#2e8b3d',3:'#d4322c',4:'#2c7fb8',5:'#5b4a6a'};
// ---- map ----
const map=new maplibregl.Map({container:'map',style:{version:8,glyphs:'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',sources:{},layers:[{id:'bg',type:'background',paint:{'background-color':'#e9edf0'}}]},center:[103.85,1.30],zoom:11,attributionControl:false,dragRotate:false,pitchWithRotate:false});
map.addControl(new maplibregl.AttributionControl({compact:true,customAttribution:'© OSM / CARTO · ShadeWalk'}));
let hIdx=1,shadeOn=true;
function imgCoords(ext){let a=ll(ext[0],ext[3]),b=ll(ext[2],ext[3]),c=ll(ext[2],ext[1]),d=ll(ext[0],ext[1]);return [a,b,c,d];}
map.on('load',()=>{
 map.addSource('base',{type:'raster',tiles:['https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png','https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'],tileSize:256});
 map.addLayer({id:'base',type:'raster',source:'base',paint:{'raster-opacity':0.9}});
 map.addSource('ish',{type:'image',url:D.shadH[hIdx],coordinates:imgCoords(D.extS)});
 map.addLayer({id:'ish',type:'raster',source:'ish',paint:{'raster-opacity':0.62,'raster-resampling':'nearest'}});
 map.addSource('rs',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
 map.addLayer({id:'rs',type:'line',source:'rs',layout:{'line-cap':'round'},paint:{'line-color':'#E0791E','line-width':4.5,'line-opacity':0.5}});
 map.addSource('rc',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
 map.addLayer({id:'rc',type:'line',source:'rc',layout:{'line-cap':'round'},paint:{'line-color':'#0F9E75','line-width':5}});
 map.addSource('od',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
 map.addLayer({id:'od',type:'circle',source:'od',paint:{'circle-radius':10,'circle-color':['get','c'],'circle-stroke-width':2.5,'circle-stroke-color':'#fff'}});
 map.addLayer({id:'odlab',type:'symbol',source:'od',layout:{'text-field':['get','lab'],'text-size':12,'text-font':['Open Sans Semibold'],'text-allow-overlap':true},paint:{'text-color':'#fff'}});
 map.addSource('walk',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
 map.addLayer({id:'walk',type:'fill',source:'walk',paint:{'fill-color':['get','c'],'fill-opacity':['get','o']}});
});
document.querySelectorAll('.hb').forEach(b=>{b.onclick=function(){hIdx=+this.dataset.i;document.querySelectorAll('.hb').forEach(x=>x.classList.remove('on'));this.classList.add('on');if(map.getSource('ish'))map.getSource('ish').updateImage({url:D.shadH[hIdx]});};});
document.getElementById('cbShade').onchange=function(){shadeOn=this.checked;if(map.getLayer('ish'))map.setLayoutProperty('ish','visibility',shadeOn?'visible':'none');};
function tip(t){let e=document.getElementById('tip');if(!t){e.style.display='none';return;}e.textContent=t;e.style.display='block';clearTimeout(tip._t);tip._t=setTimeout(()=>e.style.display='none',3200);}
// ---- OD & routing ----
let O=-1,Dst=-1,C=null,S=null,SEL='C',demoI=0;
function llInv(lng,lat){return [lng*Math.PI/180*Rm-OX,Rm*Math.log(Math.tan(Math.PI/4+lat*Math.PI/360))-OY];}
function odUpd(oll,dll){let f=[];if(oll)f.push({type:'Feature',properties:{c:'#0F6E56',lab:'A'},geometry:{type:'Point',coordinates:oll}});
 if(dll)f.push({type:'Feature',properties:{c:'#c0392b',lab:'B'},geometry:{type:'Point',coordinates:dll}});
 if(map.getSource('od'))map.getSource('od').setData({type:'FeatureCollection',features:f});}
let OLL=null,DLL=null;
map&&map.on('click',e=>{if(walkOn)return;let m=llInv(e.lngLat.lng,e.lngLat.lat),sp=snapEdge(m[0],m[1]);if(!sp){tip('No walkable path nearby');return;}
 if(O<0||(O>=0&&Dst>=0)){O=sp.node;OLL=ll(sp.fx,sp.fy);Dst=-1;DLL=null;C=S=null;clearRoutes();document.getElementById('oTxt').innerHTML='Origin · <b>tapped point</b>';document.getElementById('dTxt').innerHTML='then tap again to set <b>destination</b>';}
 else{Dst=sp.node;DLL=ll(sp.fx,sp.fy);document.getElementById('dTxt').innerHTML='Destination · <b>tapped point</b>';route(true);}
 odUpd(OLL,DLL);});
function clearRoutes(){['rc','rs'].forEach(id=>{if(map.getSource(id))map.getSource(id).setData({type:'FeatureCollection',features:[]});});
 document.getElementById('cPct').innerHTML='–<span> shaded</span>';document.getElementById('sPct').innerHTML='–<span> shaded</span>';
 document.getElementById('cSub').textContent='tap map to route';document.getElementById('sSub').innerHTML='&nbsp;';
 document.getElementById('profbar').innerHTML='';document.getElementById('profR').textContent='–';
 document.getElementById('startBtn').disabled=true;
 document.getElementById('bPov').style.display='none';document.getElementById('pov').style.display='none';
 if(typeof PV!=='undefined'){PV.on=false;PV.key=null;}
 stopWalk();}
function route(fit){if(O<0||Dst<0)return;tip('Routing…');
 setTimeout(()=>{let s=dij(O,Dst,'short'),c=s?coolDetour(O,Dst,s):null;
  if(!c||!s){tip('The two points are not connected');return;}
  C=c;S=s;
  map.getSource('rc').setData({type:'Feature',geometry:{type:'MultiLineString',coordinates:c.coords},properties:{}});
  map.getSource('rs').setData({type:'Feature',geometry:{type:'MultiLineString',coordinates:s.coords},properties:{}});
  let mn=(x)=>Math.max(1,Math.round(x/80));
  document.getElementById('cPct').innerHTML=Math.round(c.shade*100)+'%<span> shaded</span>';
  document.getElementById('cSub').textContent='🚶 '+Math.round(c.len)+' m · '+mn(c.len)+' min · sun '+(c.len*(1-c.shade)/80).toFixed(1)+' min';
  document.getElementById('sPct').innerHTML=Math.round(s.shade*100)+'%<span> shaded</span>';
  document.getElementById('sSub').textContent='🚶 '+Math.round(s.len)+' m · '+mn(s.len)+' min · sun '+(s.len*(1-s.shade)/80).toFixed(1)+' min';
  document.getElementById('startBtn').disabled=false;
  selCard(SEL);
  if(fit){let bb=[[1e9,1e9],[-1e9,-1e9]];[c,s].forEach(r=>r.coords.forEach(seg=>seg.forEach(p=>{bb[0][0]=Math.min(bb[0][0],p[0]);bb[0][1]=Math.min(bb[0][1],p[1]);bb[1][0]=Math.max(bb[1][0],p[0]);bb[1][1]=Math.max(bb[1][1],p[1]);})));
   map.fitBounds(bb,{padding:{top:40,bottom:40,left:34,right:34},duration:650});}
  tip(0);},15);}
function selCard(k){SEL=k;let c=document.getElementById('cardC'),s=document.getElementById('cardS');
 c.classList.toggle('sel',k==='C');s.classList.toggle('sel',k==='S');
 if(map.getLayer('rc')){map.setPaintProperty('rc','line-opacity',k==='C'?1:0.38);map.setPaintProperty('rs','line-opacity',k==='S'?0.95:0.38);}
 let b=document.getElementById('startBtn');b.classList.toggle('sh',k==='S');
 b.textContent=walkOn?'⏸ Stop walking':'▷ Start · '+(k==='C'?'coolest':'shortest')+' route';
 if((walkOn||(typeof PV!=='undefined'&&PV.on))&&C&&S){wd=0;wHdg=null;ensurePath();if(PV.on&&wPts){pvBuild();pvFrame();}}
 profRender();}
document.getElementById('cardC').onclick=()=>{if(C)selCard('C');};
document.getElementById('cardS').onclick=()=>{if(S)selCard('S');};
function profRender(){let r=SEL==='C'?C:S;let bar=document.getElementById('profbar');if(!r){bar.innerHTML='';return;}
 let h='';r.eis.forEach(ei=>{h+='<span style="display:inline-block;height:100%;width:'+(EL[ei]/r.len*100).toFixed(3)+'%;background:'+FACCOL[D.efac[ei]]+'"></span>';});
 bar.innerHTML=h;
 document.getElementById('profR').textContent='shaded '+Math.round(r.shade*100)+'% · sun '+Math.round((1-r.shade)*100)+'%';
 if(C&&S){let dpp=Math.round((C.shade-S.shade)*100),dl=Math.round((C.len/S.len-1)*100);
  let am=0,lm=0;C.eis.forEach(ei=>{let sm=EL[ei]*ES[ei]/100;if(D.efac[ei]===3)am+=sm;else if(D.efac[ei]===4)lm+=sm;});
  let tot=C.shade*C.len||1,fs=Math.round((am+lm)/tot*100);
  document.getElementById('insight').innerHTML='☀ Coolest is ~'+dl+'% longer but shade rises '+Math.round(S.shade*100)+'%→'+Math.round(C.shade*100)+'% (+'+dpp+'pp); '+fs+'% of its shade comes from arcades / covered links. Colors: <span style="color:#d4322c">arcade</span> · <span style="color:#2c7fb8">linkway</span> · <span style="color:#2e8b3d">trees</span> · <span style="color:#8a8a8a">buildings</span> · <span style="color:#5b4a6a">indoor</span> · <span style="color:#b8a03c">sun</span>. Basis 14:00 SOLWEIG.';}}
// ---- demo ----
document.getElementById('bDemo').onclick=()=>{let pool=[];['hdb_transit','office_transit','office_food','hdb_mart'].forEach(k=>{(D.demoOD[k]||[]).forEach(p=>pool.push([k,p]));});
 if(!pool.length){tip('No demo scenarios');return;}
 let it=pool[demoI%pool.length];demoI++;let p=it[1];
 let m1=llInv(p[0],p[1]),m2=llInv(p[2],p[3]),s1=snapEdge(m1[0],m1[1]),s2=snapEdge(m2[0],m2[1]);
 if(!s1||!s2){tip('Demo point off-network, next…');return;}
 O=s1.node;Dst=s2.node;OLL=ll(s1.fx,s1.fy);DLL=ll(s2.fx,s2.fy);
 let nm={hdb_transit:'HDB → transit',office_transit:'Office → transit',office_food:'Office → food',hdb_mart:'HDB → mart'}[it[0]];
 document.getElementById('oTxt').innerHTML='Origin · <b>'+nm+' demo</b>';
 document.getElementById('dTxt').innerHTML='Destination · <b>'+nm.split('→')[1]+'</b>';
 odUpd(OLL,DLL);route(true);};
document.getElementById('bSwap').onclick=()=>{if(O<0||Dst<0)return;let t=O;O=Dst;Dst=t;let tl=OLL;OLL=DLL;DLL=tl;odUpd(OLL,DLL);route(false);};
document.getElementById('bClear').onclick=()=>{O=-1;Dst=-1;OLL=DLL=null;C=S=null;odUpd(null,null);clearRoutes();
 document.getElementById('oTxt').innerHTML='Tap map to set <b>origin</b>';document.getElementById('dTxt').innerHTML='then tap again to set <b>destination</b>';};
// ---- walk animation (arrow follows the selected route; camera follows) ----
let walkOn=false,wd=0,wPts=null,wCum=null,wTot=1,wHdg=null,wT=0;
function flat(r){let oPos=OLL?[OLL[0],OLL[1]]:null,chunks=r.coords.slice().reverse(),pts=[];
 chunks.forEach(seg=>{let s=seg.slice(),ref=pts.length?pts[pts.length-1]:oPos;
  if(ref){let d0=Math.hypot(s[0][0]-ref[0],s[0][1]-ref[1]),d1=Math.hypot(s[s.length-1][0]-ref[0],s[s.length-1][1]-ref[1]);if(d1<d0)s.reverse();}
  s.forEach(L=>{if(!pts.length||Math.abs(L[0]-pts[pts.length-1][0])>1e-9||Math.abs(L[1]-pts[pts.length-1][1])>1e-9)pts.push(L);});});
 return pts;}
function startWalk(){let r=SEL==='C'?C:S;if(!r)return;
 wPts=flat(r);if(!wPts||wPts.length<2)return;
 let mLat=111320,mLng=111320*Math.cos(wPts[0][1]*Math.PI/180);
 wCum=[0];for(let i=1;i<wPts.length;i++)wCum.push(wCum[i-1]+Math.hypot((wPts[i][0]-wPts[i-1][0])*mLng,(wPts[i][1]-wPts[i-1][1])*mLat));
 wTot=wCum[wCum.length-1]||1;wd=0;wHdg=null;walkOn=true;wT=performance.now();
 map.easeTo({center:wPts[0],zoom:Math.max(map.getZoom(),16.2),duration:500});
 document.getElementById('startBtn').textContent='⏸ Stop walking';
 document.getElementById('bPov').style.display='block';
 requestAnimationFrame(walkFrame);}
function ptAt(a){let n=wCum.length;if(a<=0)return wPts[0];if(a>=wTot)return wPts[n-1];
 let i=1;while(i<n&&wCum[i]<a)i++;let s=wCum[i]-wCum[i-1],f=s>0?(a-wCum[i-1])/s:0;
 return [wPts[i-1][0]+(wPts[i][0]-wPts[i-1][0])*f,wPts[i-1][1]+(wPts[i][1]-wPts[i-1][1])*f];}
function walkFrame(ts){if(!walkOn)return;
 let dt=Math.min((ts-wT)/1000,0.1);wT=ts;wd+=1.4*6*dt;   // 6x walking speed
 if(wd>=wTot){wd=wTot;walkOn=false;}
 let P=ptAt(wd),T=ptAt(Math.min(wd+10,wTot));
 let mLat=111320,mLng=111320*Math.cos(P[1]*Math.PI/180);
 let tx=(T[0]-P[0])*mLng,tz=(T[1]-P[1])*mLat,TL=Math.hypot(tx,tz);
 if(TL>0.05){tx/=TL;tz/=TL;}else if(wHdg){tx=wHdg[0];tz=wHdg[1];}else{tx=1;tz=0;}
 if(!wHdg)wHdg=[tx,tz];else{let aa=1-Math.exp(-dt*3.5),hx=wHdg[0]+(tx-wHdg[0])*aa,hz=wHdg[1]+(tz-wHdg[1])*aa,HL=Math.hypot(hx,hz)||1;wHdg=[hx/HL,hz/HL];}
 let hx=wHdg[0],hz=wHdg[1];
 function pt(fw,sd){let ex=hx*fw-hz*sd,ny=hz*fw+hx*sd;return [P[0]+ex/mLng,P[1]+ny/mLat];}
 let col=SEL==='C'?'#0F9E75':'#E0791E',ph=(performance.now()%1300)/1300,R=7+9*ph,ring=[];
 for(let i2=0;i2<=14;i2++){let a2=i2/14*2*Math.PI;ring.push([P[0]+R*Math.cos(a2)/mLng,P[1]+R*Math.sin(a2)/mLat]);}
 function dart(sc){return [pt(11*sc,0),pt(-5*sc,4.6*sc),pt(-2*sc,0),pt(-5*sc,-4.6*sc),pt(11*sc,0)];}
 map.getSource('walk').setData({type:'FeatureCollection',features:[
  {type:'Feature',properties:{c:col,o:0.35*(1-ph)},geometry:{type:'Polygon',coordinates:[ring]}},
  {type:'Feature',properties:{c:'#ffffff',o:0.95},geometry:{type:'Polygon',coordinates:[dart(1.35)]}},
  {type:'Feature',properties:{c:col,o:1},geometry:{type:'Polygon',coordinates:[dart(1)]}}]});
 map.setCenter(P);
 if(typeof PV!=='undefined'&&PV.on)pvFrame();
 let r=SEL==='C'?C:S;
 document.getElementById('profR').textContent=Math.round(wd/wTot*r.len)+' / '+Math.round(r.len)+' m';
 if(walkOn)requestAnimationFrame(walkFrame);
 else{document.getElementById('startBtn').textContent='▷ Start · '+(SEL==='C'?'coolest':'shortest')+' route';profRender();}}
function stopWalk(){walkOn=false;if(map.getSource&&map.getSource('walk'))map.getSource('walk').setData({type:'FeatureCollection',features:[]});}
document.getElementById('startBtn').onclick=function(){if(walkOn){walkOn=false;this.textContent='▷ Start · '+(SEL==='C'?'coolest':'shortest')+' route';profRender();return;}startWalk();};
// ---- first-person view (three.js; same massing + sun model as the desktop Pedestrian view) ----
function sunAltAz(hour){var DOY=60,lat=1.35*Math.PI/180,lon=103.8,tz=8,hh=hour-0.5;  // NOAA; SOLWEIG date 2026-03-01, half-step
 var g=2*Math.PI/365*(DOY-1+(hh-12)/24);
 var decl=0.006918-0.399912*Math.cos(g)+0.070257*Math.sin(g)-0.006758*Math.cos(2*g)+0.000907*Math.sin(2*g)-0.002697*Math.cos(3*g)+0.00148*Math.sin(3*g);
 var eot=229.18*(0.000075+0.001868*Math.cos(g)-0.032077*Math.sin(g)-0.014615*Math.cos(2*g)-0.040849*Math.sin(2*g));
 var tst=hh*60+eot+4*lon-60*tz,H=(tst/4-180)*Math.PI/180;
 var sa=Math.sin(lat)*Math.sin(decl)+Math.cos(lat)*Math.cos(decl)*Math.cos(H),alt=Math.asin(Math.max(-1,Math.min(1,sa)));
 var ca=(Math.sin(decl)-Math.sin(lat)*sa)/(Math.cos(lat)*Math.cos(alt)+1e-9),az=Math.acos(Math.max(-1,Math.min(1,ca)));if(H>0)az=2*Math.PI-az;
 return {alt:alt,az:az};}
var PV={on:false,ready:false,key:null};
function ensurePath(){let r=SEL==='C'?C:S;if(!r)return false;wPts=flat(r);if(!wPts||wPts.length<2){wPts=null;return false;}
 let mLat=111320,mLng=111320*Math.cos(wPts[0][1]*Math.PI/180);
 wCum=[0];for(let i=1;i<wPts.length;i++)wCum.push(wCum[i-1]+Math.hypot((wPts[i][0]-wPts[i-1][0])*mLng,(wPts[i][1]-wPts[i-1][1])*mLat));
 wTot=wCum[wCum.length-1]||1;if(wd>wTot)wd=wTot;return true;}
function pvInit(){if(PV.r)return;var cv=document.getElementById('povCv');
 PV.r=new THREE.WebGLRenderer({canvas:cv,antialias:true});PV.r.shadowMap.enabled=true;PV.r.shadowMap.type=THREE.PCFShadowMap;
 PV.scene=new THREE.Scene();PV.scene.background=new THREE.Color(0xc2d8ea);PV.scene.fog=new THREE.Fog(0xc2d8ea,140,430);
 PV.cam=new THREE.PerspectiveCamera(60,1,0.4,1500);
 PV.hemi=new THREE.HemisphereLight(0xffffff,0x9aa488,0.8);PV.scene.add(PV.hemi);
 PV.sun=new THREE.DirectionalLight(0xfff2d6,1.0);PV.sun.castShadow=true;PV.sun.shadow.mapSize.set(1024,1024);PV.sun.shadow.bias=-0.0005;
 var sc=PV.sun.shadow.camera;sc.near=1;sc.far=900;sc.left=-200;sc.right=200;sc.top=200;sc.bottom=-200;sc.updateProjectionMatrix();
 PV.scene.add(PV.sun);PV.scene.add(PV.sun.target);PV.world=new THREE.Group();PV.scene.add(PV.world);}
function pvClear(){if(!PV.world)return;for(var i=PV.world.children.length-1;i>=0;i--){var c=PV.world.children[i];if(c.geometry)c.geometry.dispose();PV.world.remove(c);}}
function pvSize(){var el=document.getElementById('map'),w=el.clientWidth,h=el.clientHeight;PV.r.setSize(w,h,false);PV.cam.aspect=w/h;PV.cam.updateProjectionMatrix();}
function pvBuild(){var r=SEL==='C'?C:S;if(!r||!wPts)return;pvInit();pvClear();
 var o=wPts[Math.floor(wPts.length/2)],mLat=111320,mLng=111320*Math.cos(o[1]*Math.PI/180);PV.o=o;PV.mLng=mLng;PV.mLat=mLat;
 function E(c){return (c[0]-o[0])*mLng;} function Nf(c){return (c[1]-o[1])*mLat;}
 var xs=wPts.map(function(p){return E(p);}),zs=wPts.map(function(p){return Nf(p);});
 var bx0=Math.min.apply(null,xs)-100,bx1=Math.max.apply(null,xs)+100,bz0=Math.min.apply(null,zs)-100,bz1=Math.max.apply(null,zs)+100;
 var gnd=new THREE.Mesh(new THREE.PlaneGeometry(1800,1800),new THREE.MeshLambertMaterial({color:0xe9e5db}));gnd.rotation.x=-Math.PI/2;gnd.receiveShadow=true;PV.world.add(gnd);
 function inB(ext){if(!ext||ext.length<2)return false;var c=ll(ext[0],ext[1]),e2=E(c),n=Nf(c);return e2>=bx0&&e2<=bx1&&n>=bz0&&n<=bz1;}
 function addPoly(ext,base,top,color){if(!ext||ext.length<6||top<=base)return;var shp=new THREE.Shape();for(var m=0;m<ext.length;m+=2){var c=ll(ext[m],ext[m+1]);if(m===0)shp.moveTo(E(c),Nf(c));else shp.lineTo(E(c),Nf(c));}
  var g=new THREE.ExtrudeGeometry(shp,{depth:top-base,bevelEnabled:false});g.rotateX(-Math.PI/2);g.translate(0,base,0);var mm=new THREE.Mesh(g,new THREE.MeshLambertMaterial({color:color}));mm.castShadow=true;mm.receiveShadow=true;PV.world.add(mm);}
 var nb=0;for(var bi=0;bi<D.bld.length&&nb<450;bi++){if(inB(D.bld[bi][0])){addPoly(D.bld[bi][0],0,D.bldH[bi]||10,0xcfccc4);nb++;}}
 for(var ai=0;ai<D.arc.length;ai++){if(inB(D.arc[ai][0]))addPoly(D.arc[ai][0],3.6,Math.max(D.arcBH[ai]||6,4.2),0xc0603a);}
 for(var li=0;li<D.lkw.length;li++){if(inB(D.lkw[li][0]))addPoly(D.lkw[li][0],3.0,3.4,0x3b7fb0);}
 var T=D.tree3d,GH={0:5,1:9,2:14,3:18},GR={0:1.7,1:2.8,2:4.5,3:6.5},tc=0;
 for(var ti=0;ti<T.x.length&&tc<320;ti++){var c2=ll(T.x[ti],T.y[ti]),e3=E(c2),n3=Nf(c2);if(e3<bx0||e3>bx1||n3<bz0||n3>bz1)continue;
  var h=T.h[ti]||GH[T.g[ti]]||8,rr=GR[T.g[ti]]||3,th=h*0.45;
  var tk=new THREE.Mesh(new THREE.CylinderGeometry(Math.max(0.12,rr*0.09),Math.max(0.15,rr*0.11),th,5),new THREE.MeshLambertMaterial({color:0x6e4d30}));tk.position.set(e3,th/2,-n3);tk.castShadow=true;PV.world.add(tk);
  var cn=new THREE.Mesh(new THREE.SphereGeometry(rr,7,5),new THREE.MeshLambertMaterial({color:0x5aa54a}));cn.position.set(e3,th+rr*0.6,-n3);cn.scale.y=1.1;cn.castShadow=true;PV.world.add(cn);tc++;}
 function addRib(P,color,y,op){if(!P||P.length<2)return;var pos=[],lc=P.map(function(c){return [E(c),Nf(c)];});
  for(var ri=0;ri<lc.length;ri++){var a=lc[Math.max(0,ri-1)],b=lc[Math.min(lc.length-1,ri+1)],dx=b[0]-a[0],dz=b[1]-a[1],L2=Math.hypot(dx,dz)||1,px=-dz/L2*0.6,pz=dx/L2*0.6;
   pos.push(lc[ri][0]+px,y,-(lc[ri][1]+pz),lc[ri][0]-px,y,-(lc[ri][1]-pz));}
  var idx=[];for(var qi=0;qi<lc.length-1;qi++){var a2=2*qi;idx.push(a2,a2+1,a2+2,a2+1,a2+3,a2+2);}
  var g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setIndex(idx);
  var mm=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:color,transparent:true,opacity:op,depthWrite:false}));mm.renderOrder=2;PV.world.add(mm);}
 addRib(wPts,SEL==='C'?0x0F9E75:0xE0791E,0.07,0.95);
 var other=SEL==='C'?S:C;if(other){var op2=flat(other);if(op2&&op2.length>1)addRib(op2,SEL==='C'?0xE0791E:0x0F9E75,0.05,0.6);}
 PV.key=SEL+'_'+(C?C.len:0)+'_'+(S?S.len:0);PV.ready=true;}
function pvFrame(){if(!PV.on||!PV.ready||!wPts)return;
 var P=ptAt(wd);
 var x=(P[0]-PV.o[0])*PV.mLng,z=(P[1]-PV.o[1])*PV.mLat;
 var hx=wHdg?wHdg[0]:1,hz=wHdg?wHdg[1]:0;
 PV.cam.position.set(x,1.6,-z);PV.cam.lookAt(x+hx*14,1.45,-(z+hz*14));
 var s=sunAltAz(D.hrs[hIdx]),a=Math.max(0.05,s.alt);
 var se=Math.sin(s.az)*Math.cos(a),su=Math.sin(a),sn=Math.cos(s.az)*Math.cos(a);
 PV.sun.position.set(x+se*280,su*280+40,-z-sn*280);PV.sun.target.position.set(x,0,-z);PV.sun.target.updateMatrixWorld();
 PV.hemi.intensity=0.45+0.42*su;PV.sun.intensity=0.35+0.85*su;
 PV.r.render(PV.scene,PV.cam);}
document.getElementById('bPov').onclick=function(){if(typeof THREE==='undefined'){tip('three.js failed to load (needs internet)');return;}
 var r=SEL==='C'?C:S;if(!r)return;
 if(!wPts&&!ensurePath())return;
 var key=SEL+'_'+(C?C.len:0)+'_'+(S?S.len:0);
 pvInit();if(PV.key!==key)pvBuild();
 document.getElementById('pov').style.display='block';PV.on=true;pvSize();
 if(!wHdg){var P0=ptAt(wd),T0=ptAt(Math.min(wd+10,wTot));var mLng2=111320*Math.cos(P0[1]*Math.PI/180);var tx=(T0[0]-P0[0])*mLng2,tz=(T0[1]-P0[1])*111320,TL=Math.hypot(tx,tz)||1;wHdg=[tx/TL,tz/TL];}
 pvFrame();
 tip(walkOn?'First-person view · walking':'First-person view · tap Start to walk');};
document.getElementById('povX').onclick=function(){document.getElementById('pov').style.display='none';PV.on=false;};
document.querySelectorAll('.hb').forEach(function(b){var old=b.onclick;b.onclick=function(){old.call(this);if(PV.on)pvFrame();};});
</script></body></html>'''
HTML=HTML.replace('__DATA__',dj)
open(f"{WEB}\\nav_app_mobile.html","w",encoding="utf-8").write(HTML)
print(f"wrote nav_app_mobile.html ({len(HTML)/1e6:.1f} MB) | {time.time()-t0:.0f}s",flush=True)
