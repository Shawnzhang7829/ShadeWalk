# -*- coding: utf-8 -*-
"""MapLibre GL build of the heat-avoidance navigation app (GPU true vectors, crisp at any zoom).
True network polylines (binned by shade fraction / pedestrian flow into a few MultiLineStrings) + arcade / covered linkway / building footprint vector polygons + tree / shadow / class raster layers
+ OneMap/OSM tile basemap + JS Dijkstra live routing. Coordinates are 3857-relative (JS converts back to lng/lat); data is inlined so the page opens by double-click. pyenv."""
import geopandas as gpd, numpy as np, json, time, io, base64, os
from scipy.spatial import cKDTree
from rasterio.enums import Resampling
from shapely import force_2d, STRtree
from pyproj import Transformer
import rasterio
OUT=r"D:\Claude\SVI_FFW\output\step5_nav_webapp"; WEB=f"{OUT}\\webapp"; os.makedirs(WEB,exist_ok=True)
# Before each build, back up the previous html files (timestamped, keep the latest 15) for rollback / comparison with earlier versions
import shutil, glob as _g
_BK=f"{WEB}\\backup"; os.makedirs(_BK,exist_ok=True); _ts=time.strftime('%Y%m%d_%H%M%S'); _nbk=0
for _fn in ('nav_app.html','nav_app_en.html'):
    _s=f"{WEB}\\{_fn}"
    if os.path.exists(_s):
        shutil.copy2(_s,f"{_BK}\\{_fn[:-5]}_{_ts}.html"); _nbk+=1
        _hist=sorted(_g.glob(f"{_BK}\\{_fn[:-5]}_*.html"))
        for _o in _hist[:-15]: os.remove(_o)
if _nbk: print(f"已备份上一版 {_nbk} 个 html → backup/ ({_ts})",flush=True)
ARC=r"D:\Claude\SVI_FFW\Shp\SG\step2_arcade_sg.gpkg"
LKW=r"D:\Claude\SVI_FFW\Shp\SG\covered_linkway_SG_island_tv_pednet_bridged.gpkg"
BLDG=r"D:\Claude\SVI_FFW\Shp\SG\step2_building_remain_sg.gpkg"
CDSM=r"D:\Claude\SVI_FFW\TIF_shadow_newarcade\SUB_SG_Polygon_CDSMclean_1m.tif"   # tree CDSM for the new arcade dataset
TO=Transformer.from_crs(3414,3857,always_xy=True)
def to_m(xs,ys):
    a,b=TO.transform(np.asarray(xs,float),np.asarray(ys,float)); return np.asarray(a),np.asarray(b)
t0=time.time()
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
EF=np.round(np.nan_to_num(e['flow_short'].values)).astype(int)
# true polylines (simplified to 2 m, 3857-relative)
gs=gpd.GeoSeries(e.geometry.values,crs=3414).simplify(2.0)
cdf=gs.get_coordinates(); amx,amy=to_m(cdf['x'].values,cdf['y'].values)
AX=np.round(amx-ox).astype(int); AY=np.round(amy-oy).astype(int); iv=cdf.index.values
st_=np.searchsorted(iv,np.arange(len(e)),'left'); en_=np.searchsorted(iv,np.arange(len(e)),'right')
EGEOM=[]
for i in range(len(e)):
    s,t=st_[i],en_[i]; flat=np.empty(2*(t-s),int); flat[0::2]=AX[s:t]; flat[1::2]=AY[s:t]; EGEOM.append(flat.tolist())
print(f"网络 边{len(e)} 顶点{len(AX):,} | {time.time()-t0:.0f}s",flush=True)
EFAC=np.load(f"{OUT}\\edge_facility_SG.npy")  # dominant shade facility per edge (profile background): 0 sun / 1 building / 2 tree / 3 arcade / 4 linkway
ESN=np.load(f"{OUT}\\edge_shade_nofac_SG.npy")  # per-edge "no-facility" shade (buildings + trees only, 0-100): fallback value for the arcade/linkway-removed scenarios (step4_4b_esn)
FT_={t:np.round(np.load(f"{OUT}\\flow_cooltau_{t}.npy")).astype(int) for t in (100,120,150,200)}  # tau-capped heat-avoiding pedestrian flow (step4_4e): the Detour limit slider drives the Pedestrian flow layer
# fixed-lambda flow fields (step4_4f, SI sweep with 16 levels): used by the paper build's Pedestrian flow; skipped when the file is missing (the main build does not depend on them)
_LAMS=[3.0,1.5,1.0,0.7,0.5,0.35,0.25,0.2,0.18,0.15,0.12,0.09,0.06,0.04,0.02,0.01,0.005,0.001]  # 17 levels of the SI table (incl. the calibrated 0.2) + the extreme 0.001
FLAM={}
for _l in _LAMS:
    _p=f"{OUT}\\flow_lam_{str(_l).replace('.','p')}.npy"
    if os.path.exists(_p): FLAM[str(_l).replace('.','p')]=np.round(np.load(_p)).astype(int).tolist()
print(f"固定λ人流场 {len(FLAM)}/{len(_LAMS)} 档内联 | {time.time()-t0:.0f}s",flush=True)
ESRC=e['src'].astype(str).map({'footpath':0,'shade':1,'bridge':2}).fillna(0).astype(int).values  # edge source: 0 original footpath (pedestrian_network) / 1 facility centerline (linkway, arcade) / 2 access bridge
EFO=(np.round(np.nan_to_num(e['flow_orig'].values)).astype(int) if 'flow_orig' in e.columns else EF)  # per-edge flow on the original network (footpaths only, no facility shortcuts), used by the "Original" flow mode
# === By type, 5 classes: arcade 0 / linkway 1 / access bridge 2 / covered bridge 3 / inter-building corridor 4 ===
# linkway sub-classification (same definition as SVI_FFW-3 fig2): linkway polygon crosses a road centerline -> covered bridge; intersects a building and is not a bridge -> inter-building corridor; otherwise ordinary linkway.
# edge-level class = midpoints of the facility centerline edges with EFAC==4 within the classified linkway polygons.
_lkwp=gpd.read_file(LKW)
try: _lkwp=_lkwp.to_crs(3414)
except Exception: _lkwp=_lkwp.set_crs(3414,allow_override=True)
_roads=gpd.read_file(r"D:\Claude\SVI_FFW\Shp\SG\RoadSectionLine_Mar2026\RoadSectionLine.shp")
if _roads.crs and _roads.crs.to_epsg()!=3414: _roads=_roads.to_crs(3414)
_bldg4=gpd.read_file(BLDG)
try: _bldg4=_bldg4.to_crs(3414)
except Exception: _bldg4=_bldg4.set_crs(3414,allow_override=True)
_pb=np.zeros(len(_lkwp),bool); _pb[np.unique(STRtree(_roads.geometry.values).query(_lkwp.geometry.values,predicate='crosses')[0])]=True
_pc=np.zeros(len(_lkwp),bool); _pc[np.unique(STRtree(_bldg4.geometry.values).query(_lkwp.geometry.values,predicate='intersects')[0])]=True
_pc&=~_pb
_brT=STRtree(_lkwp.geometry.values[_pb]); _coT=STRtree(_lkwp.geometry.values[_pc])
_l4=np.where((ESRC==1)&(EFAC==4))[0]
_m4=gpd.GeoSeries(e.geometry.values[_l4],crs=3414).interpolate(0.5,normalized=True).values
_g4=np.full(len(_l4),1,np.uint8)                       # default: ordinary linkway g=1
if _pb.sum(): _g4[np.unique(_brT.query(_m4,predicate='within')[0])]=3
if _pc.sum():
    for _k in np.unique(_coT.query(_m4,predicate='within')[0]):
        if _g4[_k]==1: _g4[_k]=4
G4=np.full(len(e),1,np.uint8); G4[_l4]=_g4
print(f"连廊面分类: 天桥{int(_pb.sum())} 楼宇连廊{int(_pc.sum())} 普通{int((~_pb&~_pc).sum())} | 连廊边: 天桥{int((_g4==3).sum())} 楼宇{int((_g4==4).sum())} 普通{int((_g4==1).sum())}",flush=True)
del _lkwp,_roads,_bldg4,_pb,_pc,_brT,_coT,_m4
def _segLL(gm):
    cc=np.asarray(gm.simplify(2.0).coords)
    if len(cc)<2: return None
    mx,my=to_m(cc[:,0],cc[:,1]); fa=np.empty(2*len(cc),int); fa[0::2]=np.round(mx-ox); fa[1::2]=np.round(my-oy); return fa.tolist()
FACNET=[]; _egs=list(e.geometry.values)
for _i in range(len(e)):
    if ESRC[_i]==0: continue           # footpaths are not part of the facility micro-network
    if ESRC[_i]==1:                    # facility centerline: arcade -> 0; linkway -> 1 / 3 covered bridge / 4 inter-building corridor
        _g=int(G4[_i]) if EFAC[_i]==4 else 0
    else:                              # access bridge (adjusted network minus original minus facility centerlines) -> 2
        _g=2
    _c=_segLL(_egs[_i])
    if _c: FACNET.append([_c,_g])
print(f"By type 5类(骑楼/连廊/接入桥/过街天桥/楼宇连廊): {len(FACNET)} 段 | {time.time()-t0:.0f}s",flush=True)

# edge midpoints + flow (to aggregate, per arcade/linkway, the pedestrian flow served by the covered segments)
EMID=e.geometry.interpolate(0.5,normalized=True); EFLOW=np.nan_to_num(e['flow_short'].values); ETREE=STRtree(EMID.values)
def rings_g(g,simp,withflow=False,hcols=None):
    try: g=g.to_crs(3414)
    except Exception: g=g.set_crs(3414,allow_override=True)
    out=[]; fl=[]; hh=[[] for _ in (hcols or [])]; hv=[np.asarray(g[c].values,float) for c in (hcols or [])]
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
            rl=[ext]                                  # exterior ring
            for hole in p.interiors:                  # keep interior rings (holes): an arcade polygon wrapping a whole block stays hollow instead of being filled solid
                hf=_flat(hole.simplify(simp))
                if hf is not None: rl.append(hf)
            out.append(rl)                            # [exterior ring, hole1, hole2...]
            for _k in range(len(hcols or [])): hh[_k].append(hv[_k][gi])  # height per sub-polygon (strictly aligned with out)
            if withflow:
                idx=ETREE.query(p.buffer(6),predicate='intersects'); idx=idx[ESRC[idx]==1] if len(idx) else idx; fl.append(int(round(float(EFLOW[idx].mean()))) if len(idx) else 0)  # mean flow of the facility shade centerline edges (average over segments, not the total)
    if withflow and hcols: return out,fl,hh
    if withflow: return out,fl
    if hcols: return out,hh
    return out
def rings(path,simp): return rings_g(gpd.read_file(path),simp)
arcR,arcF,arcHH=rings_g(gpd.read_file(ARC),0.5,withflow=True,hcols=['bld_h']); arcBH=[int(round(x)) if (x==x and x>4) else 6 for x in arcHH[0]]  # roof height of the building the arcade belongs to, aligned with arcR; missing / too low -> 6 m
lkwR,lkwF=rings_g(gpd.read_file(LKW),1.0,withflow=True)
# buildings: footprint rings + centroids (centroids serve as optional O/D points)
gb=gpd.read_file(BLDG)
try: gbp=gb.to_crs(3414)
except Exception: gbp=gb.set_crs(3414,allow_override=True)
bldR,bldHH=rings_g(gbp,3.0,hcols=['height']); bldH=[int(round(x)) if (x==x and x>0) else 10 for x in bldHH[0]]  # building height, aligned with bldR; missing / invalid -> 10 m
ctr=gbp.geometry.centroid; bmx,bmy=to_m(ctr.x.values,ctr.y.values)
BCX=np.round(bmx-ox).astype(int); BCY=np.round(bmy-oy).astype(int)
print(f"骑楼环{len(arcR)} 连廊环{len(lkwR)} 建筑环{len(bldR)} 建筑中心{len(BCX)} | {time.time()-t0:.0f}s",flush=True)
# HDB void decks: approximated by HDB building footprints (building_hourly_weight archetype='hdb', same source as the pedestrian flow / POI data)
_bw4=gpd.read_file(r"D:\Claude\UNA\Patronage_Flow\output\building_hourly_weight.gpkg")
hdbR=rings_g(_bw4[_bw4['building_archetype']=='hdb'],2.0)
print(f"HDB 架空层环{len(hdbR)} | {time.time()-t0:.0f}s",flush=True); del _bw4
# === real tree points of the focus area (1 km around Detour_demo): full set for the 3D aerial / pedestrian views (no thinning) ===
TREE3=r"D:\Claude\SVI_FFW\Shp\SG\SG point tree\Point tree.shp"
_gt=gpd.read_file(TREE3)  # 697k trees citywide; three.js clips and renders them dynamically by the current view (not clipped here); shown only when zoomed to street level
_gt3=_gt.to_crs(3414); _tmx,_tmy=to_m(_gt3.geometry.x.values,_gt3.geometry.y.values)
_th=np.clip(np.round(np.nan_to_num(_gt['height_est'].values.astype(float),nan=8)),3,30).astype(int)
_tg=_gt['girth_size'].map({'XS':0,'S':1,'M':2,'L':3}).fillna(1).astype(int).values
tree3d=dict(x=np.round(_tmx-ox).astype(int).tolist(),y=np.round(_tmy-oy).astype(int).tolist(),h=_th.tolist(),g=_tg.tolist())
print(f"重点区树点 {len(tree3d['x'])} | 对齐 bldH{len(bldH)}/bld{len(bldR)} arcBH{len(arcBH)}/arc{len(arcR)} | {time.time()-t0:.0f}s",flush=True)
# stations (MRT / bus) as optional O/D points
STA=r"D:\Claude\SVI_FFW\Shp\SG\POI\station_hourly_ridership.gpkg"
gs2=gpd.read_file(STA)
try: gs2=gs2.to_crs(3414)
except Exception: gs2=gs2.set_crs(3414,allow_override=True)
smx,smy=to_m(gs2.geometry.x.values,gs2.geometry.y.values)
SX=np.round(smx-ox).astype(int); SY=np.round(smy-oy).astype(int)
SSRC=(gs2['source'].astype(str).str.upper()=='MRT').astype(int).tolist()
SCODE=gs2['PT_CODE'].astype(str).tolist()
SRID=np.round(np.nan_to_num(gs2['tot_weekday_total'].values)).astype(int).tolist()
print(f"站点{len(SX)} (MRT {sum(SSRC)}) | {time.time()-t0:.0f}s",flush=True)

# all rasters inlined (self-contained, no server needed): tree 10 m PNG + shadow/class 40 m hourly frames (reusing the cache)
HL=json.load(open(f"{OUT}\\_hourly_layers.json"))
def ext3857(b):
    mx,my=to_m([b[0],b[2]],[b[1],b[3]]); return [int(round(mx[0]-ox)),int(round(my[0]-oy)),int(round(mx[1]-ox)),int(round(my[1]-oy))]
EXT_S=ext3857(HL['extS_abs']); EXT_C=ext3857(HL['extC_abs'])
def png_uri(rgba):
    from PIL import Image; bb=io.BytesIO(); Image.fromarray(rgba,'RGBA').save(bb,'PNG'); return 'data:image/png;base64,'+base64.b64encode(bb.getvalue()).decode()
_RCACHE=f"{OUT}\\_navrast_cache.json"  # cache trees only (shadow/class now use the hourly frames from _hourly_layers.json)
if os.path.exists(_RCACHE):
    _r=json.load(open(_RCACHE,encoding='utf-8')); treeImg=_r['t']; EXT_T=_r['et']
    print(f"复用树木栅格缓存 | {time.time()-t0:.0f}s",flush=True)
else:
    with rasterio.open(CDSM) as ds:
        cb=ds.bounds; cd=ds.read(1,out_shape=(ds.height//10,ds.width//10),resampling=Resampling.average)
    cd=np.nan_to_num(cd); trg=np.zeros((cd.shape[0],cd.shape[1],4),np.uint8); trg[cd>1.0]=[33,115,46,238]
    treeImg=png_uri(trg); EXT_T=ext3857([cb.left,cb.bottom,cb.right,cb.top])
    json.dump({'t':treeImg,'et':EXT_T},open(_RCACHE,'w'))
shadH=HL['shad']; catH=HL['cat']  # hourly shadow/class frames (08-18h, 11 frames @10 m)
print(f"内联栅格: 树木 @10m + 阴影/分类逐时 @10m({len(shadH)}帧) + 设施逐时比例 | {time.time()-t0:.0f}s",flush=True)
# hourly scaling ratios for facility-served flow (citywide hourly station ridership relative to 14:00)
def _sumtot(h):
    c=f'tot_weekday_{h:02d}'; return float(np.nan_to_num(gs2[c].values).sum()) if c in gs2.columns else 0.0
_b14=_sumtot(14) or 1.0; FACR=[round(_sumtot(h)/_b14,3) for h in HL['hours']]
print(f"设施逐时比例(相对14:00) {FACR} | {time.time()-t0:.0f}s",flush=True)

_demoOD=json.load(open(f"{OUT}\\_demo_scenarios.json",encoding='utf-8')) if os.path.exists(f"{OUT}\\_demo_scenarios.json") else {}
data=dict(ox=ox,oy=oy,egeom=EGEOM,es=ES.tolist(),ef=EF.tolist(),efo=EFO.tolist(),eu=U.tolist(),ev=V.tolist(),el=EL.tolist(),efac=EFAC.tolist(),esrc=ESRC.tolist(),esn=ESN.tolist(),ft100=FT_[100].tolist(),ft120=FT_[120].tolist(),ft150=FT_[150].tolist(),ft200=FT_[200].tolist(),flam=FLAM,
          nx=NX.tolist(),ny=NY.tolist(),arc=arcR,lkw=lkwR,bld=bldR,hdb=hdbR,bldH=bldH,arcBH=arcBH,tree3d=tree3d,arcF=arcF,lkwF=lkwF,facR=FACR,facnet=FACNET,demoOD=_demoOD,
          bcx=BCX.tolist(),bcy=BCY.tolist(),stx=SX.tolist(),sty=SY.tolist(),sts=SSRC,stc=SCODE,srid=SRID,
          treeImg=treeImg,extT=EXT_T,shadH=shadH,catH=catH,extS=EXT_S,extC=EXT_C,hours=HL['hours'],
          xmin=int(NX.min()),xmax=int(NX.max()),ymin=int(NY.min()),ymax=int(NY.max()))
dj=json.dumps(data,separators=(',',':')); print(f"JSON {len(dj)/1e6:.1f} MB | {time.time()-t0:.0f}s",flush=True)

HTML=r'''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ShadeWalk</title><link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><rect x='1' y='1' width='22' height='22' rx='5.5' fill='%232e8b46'/><rect x='4.5' y='5' width='15' height='2.1' rx='0.6' fill='%23ffd23f'/><line x1='6.3' y1='6.9' x2='6.3' y2='17.6' stroke='%23ffd23f' stroke-width='1.4' stroke-linecap='round'/><line x1='17.7' y1='6.9' x2='17.7' y2='17.6' stroke='%23ffd23f' stroke-width='1.4' stroke-linecap='round'/><circle cx='12' cy='10.2' r='2.05' fill='%23fff'/><rect x='10.75' y='12.7' width='2.5' height='6.5' rx='1.25' fill='%23fff'/></svg>">
<link href="https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.24.0/dist/tabler-icons.min.css">
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:#1d1c1a}
#app{display:flex;height:100vh;width:100vw}
#panel{width:374px;flex:none;background:#fafaf7;border-right:1px solid #e2e0d7;padding:15px 17px;overflow-y:auto}
#map{flex:1;position:relative}
.h{font-size:18px;font-weight:500;display:flex;align-items:center;gap:8px;margin:0 0 4px}
.sub{font-size:12px;color:#8a8980;margin:0 0 13px;line-height:1.5}
.row{display:flex;align-items:center;gap:8px;font-size:13px;padding:8px 10px;border:1px solid #e2e0d7;border-radius:8px;background:#fff;margin-bottom:8px}
.btn{font-size:13px;padding:7px 10px;border:1px solid #cfcdc3;border-radius:8px;background:#fff;cursor:pointer;color:#444}
.btn:hover{background:#f0efe9}.btn.on{border:2px solid #444;color:#fff;background:#444;padding:6px 9px}
.card{border:1px solid #e2e0d7;border-radius:11px;background:#fff;padding:11px 13px;margin-bottom:9px}
.card.cool{border-left:4px solid #0F9E75}.card.short{border-left:4px solid #E0791E}
.big{font-size:22px;font-weight:500}.mut{font-size:11px;color:#8a8980}
.sec{font-size:13px;font-weight:600;color:#33312c;letter-spacing:.01em}
.lay{display:flex;align-items:center;gap:7px;font-size:13px;padding:5px 0;cursor:pointer}
.sw2{width:14px;height:14px;border-radius:3px;display:inline-block;border:1px solid #0003}
hr{border:none;border-top:1px solid #e2e0d7;margin:13px 0}
#legwrap{position:absolute;left:10px;bottom:10px;z-index:2;display:flex;flex-direction:column-reverse;gap:6px;align-items:flex-start;max-width:46%}#catleg{background:rgba(255,255,255,.94);border:1px solid #e2e0d7;border-radius:8px;padding:7px 9px;font-size:11px;line-height:1.55}.leg{background:rgba(255,255,255,.92);border:1px solid #e2e0d7;border-radius:8px;padding:8px 10px;font-size:12px;z-index:2}
.sw{display:inline-block;width:20px;height:4px;border-radius:2px;vertical-align:middle;margin-right:5px}
#tip{position:absolute;left:50%;top:10px;transform:translateX(-50%);background:#1d1c1a;color:#fff;font-size:12px;padding:6px 13px;border-radius:18px;z-index:3;display:none}
</style></head><body>
<div id="app">
<aside id="panel">
  <div class="h"><svg viewBox="0 0 24 24" width="30" height="30" style="flex:none" aria-hidden="true"><rect x="1" y="1" width="22" height="22" rx="5.5" fill="#2e8b46"/><rect x="4.5" y="5" width="15" height="2.1" rx="0.6" fill="#ffd23f"/><line x1="6.3" y1="6.9" x2="6.3" y2="17.6" stroke="#ffd23f" stroke-width="1.4" stroke-linecap="round"/><line x1="17.7" y1="6.9" x2="17.7" y2="17.6" stroke="#ffd23f" stroke-width="1.4" stroke-linecap="round"/><circle cx="12" cy="9.7" r="1.62" fill="#fff"/><path d="M11.5 12.5C9.9 14 9.85 16.3 11.5 19.4Q12 20.1 12.5 19.4C14.15 16.3 14.1 14 12.5 12.5Q12 11.95 11.5 12.5Z" fill="#fff"/></svg>ShadeWalk</div>
  <p class="sub">MapLibre 矢量 · 新加坡全城真实步行网络 · 14:00 · 实时路由</p>
  <div class="row"><i class="ti ti-current-location" style="color:#0F6E56;font-size:16px"></i><span id="oTxt">起点:点击地图选择</span></div>
  <div class="row"><i class="ti ti-map-pin" style="color:#c0392b;font-size:16px"></i><span id="dTxt">终点:再次点击地图</span></div>
  <div style="display:flex;gap:7px;margin-bottom:12px">
    <button class="btn" id="bDemo"><i class="ti ti-sparkles"></i> 示例</button>
    <button class="btn" id="bSwap"><i class="ti ti-arrows-up-down"></i> 互换</button>
    <button class="btn" id="bClear"><i class="ti ti-x"></i> 清除</button></div>
  <div class="sec" style="margin:7px 0 3px">Demo 场景(点击轮换 · 15 分钟步行圈)</div>
  <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:4px">
    <button class="btn" id="dmHt" style="font-size:12px;padding:6px 8px"><i class="ti ti-building-community"></i> 组屋→公交</button>
    <button class="btn" id="dmOt" style="font-size:12px;padding:6px 8px"><i class="ti ti-building-skyscraper"></i> 办公→公交</button>
    <button class="btn" id="dmOf" style="font-size:12px;padding:6px 8px"><i class="ti ti-tools-kitchen-2"></i> 办公→餐饮</button>
    <button class="btn" id="dmHm" style="font-size:12px;padding:6px 8px"><i class="ti ti-shopping-cart"></i> 组屋→商超</button>
  </div>
  <div class="sec" style="margin:9px 0 4px">绕行阈值 Detour limit</div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:9px;font-size:12px"><input type="range" id="detTh" min="0" max="3" step="1" value="2" style="flex:1"><span id="detThV" style="min-width:40px;text-align:right;color:#0F6E56;font-weight:600">1.5×</span></div>
  <div class="sec" style="margin:9px 0 4px">我的 Demo(收藏 / 导入导出)</div>
  <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px">
    <button class="btn" id="dmSave" style="font-size:12px;padding:6px 8px"><i class="ti ti-device-floppy"></i> 保存当前</button>
    <button class="btn" id="dmMine" style="font-size:12px;padding:6px 8px"><i class="ti ti-star"></i> 我的收藏(<span id="svN">0</span>)</button>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px">
    <button class="btn" id="dmImport" style="font-size:12px;padding:6px 8px"><i class="ti ti-upload"></i> 导入</button>
    <button class="btn" id="dmExport" style="font-size:12px;padding:6px 8px"><i class="ti ti-download"></i> 导出</button>
    <button class="btn" id="dmClear" style="font-size:12px;padding:6px 8px"><i class="ti ti-trash"></i> 清空</button>
    <input type="file" id="dmFile" accept=".json,application/json" style="display:none">
  </div>
  <div id="results"></div>
  <hr>
  <div class="sec" style="margin-bottom:3px">可选起讫点(点击设起/终点)</div>
  <label class="lay"><input type="checkbox" id="lyStaPt"><span class="sw2" style="background:#f39c12;border-radius:50%"></span>站点 MRT/巴士</label>
  <label class="lay"><input type="checkbox" id="lyBldPt"><span class="sw2" style="background:#6a51a3;border-radius:50%"></span>建筑中心点</label>
  <hr>
  <div class="sec" style="margin-bottom:5px">Basemap</div>
  <div style="display:flex;gap:7px;margin-bottom:11px"><button class="btn on" id="bmGray">Simple</button><button class="btn" id="bmOsm">OSM</button><button class="btn" id="bmOne">OneMap</button><button class="btn" id="bmGsat">卫星</button><button class="btn" id="bmNone">无</button></div>
  <div class="sec" style="margin-bottom:5px">视图 View</div>
  <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px"><button class="btn on" id="v2d">2D 平面</button><button class="btn" id="v3d">3D 鸟瞰</button><button class="btn" id="bPov"><i class="ti ti-walk"></i> 行人视角</button></div>
  <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:11px"><button class="btn" id="bSun" title="逐时太阳遮荫(随时刻滑杆)">太阳阴影</button><button class="btn" id="aCol" title="骑楼柱廊柱子(仅3D)">骑楼柱子</button><button class="btn" id="bTopo" title="OD 间可行路径拓扑图(绕行率上限=My Demos 的 Detour limit)">拓扑图</button></div>
  <div class="sec" style="margin-bottom:5px">网络视图</div>
  <div style="display:flex;flex-wrap:wrap;gap:7px;margin-bottom:11px"><button class="btn" id="cShade">遮荫率</button><button class="btn" id="cFlow">人流量</button><button class="btn on" id="cPlain">无色</button><button class="btn" id="cNone">隐藏</button></div>
  <div class="sec" style="margin-bottom:5px">人工遮荫设施(连廊/骑楼)</div>
  <div style="display:flex;flex-wrap:wrap;gap:7px;margin-bottom:11px"><button class="btn on" id="fAll">全部</button><button class="btn" id="fOrig">仅原始</button><button class="btn" id="fHi">分类高亮</button><button class="btn" id="fHiM">统一高亮</button></div>
  <div class="sec" style="margin-bottom:5px">显示路线</div>
  <div style="display:flex;flex-wrap:wrap;gap:7px;margin-bottom:11px"><button class="btn on" id="rBoth">两者</button><button class="btn" id="rCool">最遮荫</button><button class="btn" id="rShort">最短</button><button class="btn" id="rHide">隐藏</button></div>
  <div class="sec" style="margin-bottom:3px">遮荫设施图层</div>
  <label class="lay"><input type="checkbox" id="lyBld"><span class="sw2" style="background:#8a8a8a"></span>建筑</label>
  <label class="lay"><input type="checkbox" id="lyTree"><span class="sw2" style="background:#2e8b3d"></span>树木 (10m)</label>
  <label class="lay"><input type="checkbox" id="ly3DTree"><span class="sw2" style="background:#4a9d3f;border-radius:50%"></span>3D 树木</label>
  <label class="lay"><input type="checkbox" id="lyArc"><span class="sw2" style="background:#d4322c"></span>骑楼 arcade</label>
  <label class="lay"><input type="checkbox" id="lyLkw"><span class="sw2" style="background:#2c7fb8"></span>有盖连廊</label>
  <label class="lay"><input type="checkbox" id="lyHdb"><span class="sw2" style="background:#5b4a6a"></span>HDB 架空层</label>
  <hr>
  <div class="sec" style="margin-bottom:5px">遮荫栅格 (10m)</div>
  <label class="lay"><input type="checkbox" id="lyShad"><span class="sw2" style="background:#2c7fb8"></span>阴影 Shadow(随滑杆)</label>
  <label class="lay"><input type="checkbox" id="lyCat"><span class="sw2" style="background:#d4322c"></span>分类 Category(随滑杆)</label>
  <div class="sec" style="margin-top:8px;margin-bottom:4px">逐时 8–18点 · 阴影/分类栅格 + 设施人流 <b id="hrLbl" style="color:#1d1c1a">14:00</b></div>
  <input type="range" id="hr" min="0" max="10" step="1" value="6" style="width:100%;margin-bottom:6px">
  <p class="sub" style="margin-top:13px">点击地图设起点/终点。最遮荫路=最小日晒里程;最短路=最短距离。栅格已内联(树木/阴影/分类均 ≈10m 单帧),双击即可打开,无需服务器。</p>
</aside>
<main id="map"><div id="legwrap"><div class="leg" id="leg"></div><div id="catleg" style="display:none"></div></div><div id="tip"></div>
 <div id="prof" style="display:none;position:absolute;left:12px;top:12px;width:567px;background:rgba(255,255,255,.96);border:1px solid #d8d6cc;border-radius:10px;padding:7px 10px 10px;box-shadow:0 2px 12px rgba(0,0,0,.12);z-index:5;resize:horizontal;overflow:hidden;min-width:250px;max-width:900px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1px;cursor:move" title="按住标题拖动 · 拖右下角缩放"><b style="font-size:12px">⠿ 时序路径阴影</b><span id="profF" style="cursor:pointer;color:#999;font-size:14px;line-height:1;margin-right:8px" title="折叠/展开">▾</span><span id="profX" style="cursor:pointer;color:#999;font-size:15px;line-height:1">×</span></div>
  <div id="profBody"></div>
  <span style="position:absolute;right:3px;bottom:1px;color:#aaa;font-size:12px;pointer-events:none">⤡</span></div>
 <div id="pov" style="display:none;position:absolute;right:12px;bottom:64px;width:460px;background:rgba(255,255,255,.97);border:1px solid #d8d6cc;border-radius:10px;padding:7px 10px 10px;box-shadow:0 2px 12px rgba(0,0,0,.14);z-index:6;overflow:hidden">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;cursor:move"><b style="font-size:12px">🚶 行人视角 · Coolest/Shortest 对比</b><span id="povF" style="cursor:pointer;color:#999;font-size:14px;line-height:1;margin-right:8px" title="折叠/展开">▾</span><span id="povX" style="cursor:pointer;color:#999;font-size:15px;line-height:1">×</span></div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin:1px 0"><span style="color:#0F6E56;font-weight:600;font-size:10px">最遮荫 Coolest</span><button class="btn" id="povPlayC" style="padding:1px 9px;font-size:12px">▶</button></div>
  <canvas id="povCvC" width="440" height="156" style="width:440px;height:156px;border-radius:6px;background:#c2d8ea;display:block"></canvas>
  <div style="display:flex;align-items:center;gap:6px;margin:2px 0"><input type="range" id="povSeekC" min="0" max="1000" value="0" style="flex:1"><span id="povTxtC" style="font-size:10px;color:#555;min-width:60px;text-align:right">0/0m</span></div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin:3px 0 1px"><span style="color:#985316;font-weight:600;font-size:10px">最短 Shortest</span><button class="btn" id="povPlayS" style="padding:1px 9px;font-size:12px">▶</button></div>
  <canvas id="povCvS" width="440" height="156" style="width:440px;height:156px;border-radius:6px;background:#c2d8ea;display:block"></canvas>
  <div style="display:flex;align-items:center;gap:6px;margin:2px 0"><input type="range" id="povSeekS" min="0" max="1000" value="0" style="flex:1"><span id="povTxtS" style="font-size:10px;color:#555;min-width:60px;text-align:right">0/0m</span></div>
  <div style="display:flex;align-items:center;gap:6px;margin-top:5px"><button class="btn" id="povBoth" style="padding:4px 10px;font-size:12px">⏯ 同步播放</button><span style="font-size:10px;color:#555">速度(×步速)</span><button class="btn on" id="povSp4" style="padding:2px 7px;font-size:11px">4×</button><button class="btn" id="povSp9" style="padding:2px 7px;font-size:11px">9×</button><button class="btn" id="povSp18" style="padding:2px 7px;font-size:11px">18×</button><button class="btn" id="povSp36" style="padding:2px 7px;font-size:11px">36×</button></div>
  <div style="font-size:10px;color:#888;margin-top:3px">或各自 ▶ 单独播 · 灰建筑 绿树 红棕骑楼 蓝连廊</div>
 </div>
 <div id="topo" style="display:none;position:absolute;right:12px;top:12px;width:522px;max-height:calc(100% - 24px);overflow:auto;background:rgba(255,255,255,.97);border:1px solid #d8d6cc;border-radius:10px;padding:7px 10px 10px;box-shadow:0 2px 12px rgba(0,0,0,.14);z-index:7">
  <div id="topoHdr" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;cursor:move"><b style="font-size:12px">🕸 拓扑图 · OD 可行路径</b><span id="topoF" style="cursor:pointer;color:#999;font-size:14px;line-height:1;margin-right:8px" title="折叠/展开">▾</span><span id="topoX" style="cursor:pointer;color:#999;font-size:15px;line-height:1">×</span></div>
  <div id="topoBody"></div>
 </div>
</main></div>
<script src="https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script>
const D=__DATA__;
const Rm=6378137, OX=D.ox, OY=D.oy;
function ll(x,y){let X=(x+OX),Y=(y+OY);return [X/Rm*180/Math.PI,(2*Math.atan(Math.exp(Y/Rm))-Math.PI/2)*180/Math.PI];}
function edgeLL(g){let a=[];for(let k=0;k<g.length;k+=2)a.push(ll(g[k],g[k+1]));return a;}
function ringLL(g){let a=[];for(let k=0;k<g.length;k+=2)a.push(ll(g[k],g[k+1]));a.push(a[0]);return a;}
// 路由图
const NE=D.eu.length,NN=D.nx.length,LAM=0.2,EL=D.el,ES=D.es,EF=D.ef,EFO=D.efo,U=D.eu,V=D.ev,ESRCC=D.esrc,EFACC=D.efac,ESNN=D.esn;
let SCN={arc:true,lkw:true};  // facility scenario: unchecking Arcades / Covered linkways layer boxes removes that facility from routing (its centerline edges banned) and falls back covered edges' shade to esn (buildings+trees only)
function esf(ei){return ((EFACC[ei]===3&&!SCN.arc)||(EFACC[ei]===4&&!SCN.lkw))?ESNN[ei]:ES[ei];}
function eban(ei){return ESRCC[ei]===1&&((EFACC[ei]===3&&!SCN.arc)||(EFACC[ei]===4&&!SCN.lkw));}
const FT={1:D.ft100,1.2:D.ft120,1.5:D.ft150,2:D.ft200};  // per-Detour-limit heat-avoiding flow (step4_4e tau-capped assignment)
function flowCur(){return FT[DETOUR]||EF;}  // pedestrian-flow field matching the current Detour limit; falls back to flow_short
const deg=new Int32Array(NN);for(let i=0;i<NE;i++){deg[U[i]]++;deg[V[i]]++;}
const off=new Int32Array(NN+1);for(let i=0;i<NN;i++)off[i+1]=off[i]+deg[i];
const aN=new Int32Array(NE*2),aE=new Int32Array(NE*2),cu=off.slice();
for(let i=0;i<NE;i++){let a=U[i],b=V[i];aN[cu[a]]=b;aE[cu[a]++]=i;aN[cu[b]]=a;aE[cu[b]++]=i;}
const GC=300,grid=new Map();for(let i=0;i<NN;i++){let k=((D.nx[i]/GC)|0)+'_'+((D.ny[i]/GC)|0);(grid.get(k)||grid.set(k,[]).get(k)).push(i);}
function nearest(x,y){let best=-1,bd=1e18;for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++){let a=grid.get(((x/GC|0)+dx)+'_'+((y/GC|0)+dy));if(!a)continue;for(const i of a){let d=(D.nx[i]-x)**2+(D.ny[i]-y)**2;if(d<bd){bd=d;best=i;}}}return best;}
function haversine(a,b){var R=6371000,p=Math.PI/180,dLa=(b[1]-a[1])*p,dLo=(b[0]-a[0])*p,h=Math.sin(dLa/2)**2+Math.cos(a[1]*p)*Math.cos(b[1]*p)*Math.sin(dLo/2)**2;return 2*R*Math.asin(Math.sqrt(h));}
// 任意点 → 最近路段垂足:返回接入端点(较近交叉口)、垂足经纬度、接入米数(点击→垂足 + 垂足→端点沿边)
function snapEdge(mx,my){var cx=(mx/GC)|0,cy=(my/GC)|0,seen=new Set(),best=-1,bd=1e18,bfx=0,bfy=0,bcum=0,bstart=-1,bel=0;
 for(var dx=-2;dx<=2;dx++)for(var dy=-2;dy<=2;dy++){var a=grid.get((cx+dx)+'_'+(cy+dy));if(!a)continue;
  for(const ni of a){for(var q=off[ni];q<off[ni+1];q++){var ei=aE[q];if(seen.has(ei))continue;seen.add(ei);
   var g=D.egeom[ei];var su=(g[0]-D.nx[U[ei]])**2+(g[1]-D.ny[U[ei]])**2,sv=(g[0]-D.nx[V[ei]])**2+(g[1]-D.ny[V[ei]])**2;var sN=su<=sv?U[ei]:V[ei];var cum=0;
   for(var k=0;k+3<g.length;k+=2){var axx=g[k],ayy=g[k+1],vx=g[k+2]-axx,vy=g[k+3]-ayy,L2=vx*vx+vy*vy;var t=L2>0?((mx-axx)*vx+(my-ayy)*vy)/L2:0;t=t<0?0:t>1?1:t;var px=axx+t*vx,py=ayy+t*vy,d=(mx-px)**2+(my-py)**2,sl=Math.sqrt(L2);if(d<bd){bd=d;best=ei;bfx=px;bfy=py;bcum=cum+t*sl;bstart=sN;bel=EL[ei];}cum+=sl;}
  }}}
 if(best<0)return null;
 var g=D.egeom[best],tot=0;for(var k=0;k+3<g.length;k+=2)tot+=Math.sqrt((g[k+2]-g[k])**2+(g[k+3]-g[k+1])**2);
 var duS=tot>0?(bcum/tot)*bel:0,duE=bel-duS,node,dn;if(duS<=duE){node=bstart;dn=duS;}else{node=(bstart===U[best]?V[best]:U[best]);dn=duE;}
 var fll=ll(bfx,bfy),pll=ll(mx,my);return {node:node,flng:fll[0],flat:fll[1],acc:haversine(pll,fll)+dn};}
function dij(s,t,mode,lam){const LM=(lam===undefined?LAM:lam);const dist=new Float64Array(NN).fill(1e18),pe=new Int32Array(NN).fill(-1),pn=new Int32Array(NN).fill(-1);dist[s]=0;const hd=[s],hk=[0];
 function up(i){while(i>0){let p=(i-1)>>1;if(hk[p]<=hk[i])break;[hk[p],hk[i]]=[hk[i],hk[p]];[hd[p],hd[i]]=[hd[i],hd[p]];i=p;}}
 function dn(i){let n=hd.length;for(;;){let l=2*i+1,r=l+1,m=i;if(l<n&&hk[l]<hk[m])m=l;if(r<n&&hk[r]<hk[m])m=r;if(m===i)break;[hk[m],hk[i]]=[hk[i],hk[m]];[hd[m],hd[i]]=[hd[i],hd[m]];i=m;}}
 while(hd.length){let d=hk[0],x=hd[0],L=hd.length-1;hd[0]=hd[L];hk[0]=hk[L];hd.pop();hk.pop();if(hd.length)dn(0);if(d>dist[x])continue;if(x===t)break;
  for(let p=off[x];p<off[x+1];p++){let y=aN[p],ei=aE[p];if(eban(ei))continue;let c=mode==='cool'?EL[ei]*((1-esf(ei)/100)+LM):EL[ei];let n2=d+c;if(n2<dist[y]){dist[y]=n2;pe[y]=ei;pn[y]=x;hd.push(y);hk.push(n2);up(hd.length-1);}}}
 if(dist[t]>=1e18)return null;let coords=[],eis=[],len=0,sh=0,x=t;
 while(pn[x]>=0){let ei=pe[x];eis.push(ei);len+=EL[ei];sh+=EL[ei]*esf(ei)/100;coords.push(edgeLL(D.egeom[ei]));x=pn[x];}
 eis.reverse(); return {coords:coords,len:len,shade:len>0?sh/len:0,eis:eis};}
function pathFlow(r){if(!r||!r.eis||!r.eis.length)return 0;let s=0,F=flowCur();for(let k=0;k<r.eis.length;k++)s+=F[r.eis[k]];return Math.round(s/r.eis.length);}  // avg edge flow along path under the current Detour limit = served pedestrian flow (ppl)
function facShade(r){if(!r||!r.eis||!r.eis.length)return{arc:0,lkw:0,arcM:0,lkwM:0,totM:0};let arcM=0,lkwM=0,totM=0;for(let k=0;k<r.eis.length;k++){let ei=r.eis[k],sm=EL[ei]*esf(ei)/100;totM+=sm;if(D.efac[ei]===3&&SCN.arc)arcM+=sm;else if(D.efac[ei]===4&&SCN.lkw)lkwM+=sm;}return{arcM:arcM,lkwM:lkwM,totM:totM,arc:totM>0?arcM/totM:0,lkw:totM>0?lkwM/totM:0};}  // 骑楼(3)/连廊(4)对路径总遮荫的贡献占比,口径同剖面:每边遮荫米数 EL×ES/100,分母=路径总遮荫米数(=shade×len)
function facLine(r){let f=facShade(r);return '<div class="mut" style="font-size:11px;margin-top:2px">人工设施遮荫贡献 <b style="color:#d4322c" title="骑楼段沿路提供遮荫约 '+Math.round(f.arcM)+' m">骑楼 '+Math.round(f.arc*100)+'%</b> · <b style="color:#2c7fb8" title="有盖连廊段沿路提供遮荫约 '+Math.round(f.lkwM)+' m">连廊 '+Math.round(f.lkw*100)+'%</b> <span style="opacity:.65">(合计 '+Math.round((f.arc+f.lkw)*100)+'%)</span></div>';}
let DETOUR=1.5;   // detour limit (slider): cool path length <= shortest*DETOUR, pick the max shade-coverage path
function coolDetour(O,Dst,s){  // smaller lam = more shaded (more detour); within detour<=DETOUR take smallest lam = max shade coverage
 if(!s)return null;let c0=dij(O,Dst,'cool',0);            // lam=0 = pure most-shaded (longest detour)
 if(c0&&c0.len<=s.len*DETOUR+1)return c0;                 // pure-shaded already within detour limit -> use it (max coverage)
 let lo=0,hi=3.0,ch=dij(O,Dst,'cool',hi);                 // else binary-search smallest lam keeping detour <= limit
 while((!ch||ch.len>s.len*DETOUR+1)&&hi<3000){lo=hi;hi*=4;ch=dij(O,Dst,'cool',hi);}  // adaptive upper bound: grow lam until the cap is met (old fixed hi=3 could return an over-cap path at tight limits, e.g. 1x)
 if(!ch||ch.len>s.len*DETOUR+1)return s;                  // cap unreachable even at near-pure-length weighting -> Shortest itself
 let best=ch;
 for(let it=0;it<22;it++){let mid=(lo+hi)/2,c=dij(O,Dst,'cool',mid);
  if(c&&c.len<=s.len*DETOUR+1){hi=mid;best=c;}            // detour within limit -> lam can shrink (more shaded)
  else{lo=mid;}}                                          // detour over limit -> lam must grow (less detour)
 return best;}
const FACCOL={0:'#f4e7b0',1:'#8a8a8a',2:'#2e8b3d',3:'#d4322c',4:'#2c7fb8',5:'#5b4a6a'};
const FACNM={0:'日晒',1:'建筑',2:'树木',3:'骑楼',4:'连廊',5:'建筑内部'};
function profile(c,s){let P=document.getElementById('prof');
 if(!c||!c.eis||!c.eis.length){P.style.display='none';return;}
 let TOT=Math.max(c.len,(s&&s.len)||0)||1;  // shared x-axis = actual distance (Coolest, the longer one, as full scale)
 let W=567,H=126,pad=5,gx=W-2*pad,bnd=7,gp=2,ty=pad,gy0=ty+2*bnd+gp+8,gyH=H-gy0-13;window.PROF={TOT:TOT,pad:pad,gx:gx};
 function band(res,y){if(!res||!res.eis)return '';let cum=0,sv='';for(let k=0;k<res.eis.length;k++){let ei=res.eis[k],w=EL[ei]/TOT*gx,x=pad+cum/TOT*gx;let fc=D.efac[ei];if((fc===3&&!SCN.arc)||(fc===4&&!SCN.lkw))fc=1;sv+='<rect x="'+x.toFixed(1)+'" y="'+y+'" width="'+Math.max(0.4,w).toFixed(1)+'" height="'+bnd+'" fill="'+FACCOL[fc]+'" opacity="0.85"/>';cum+=EL[ei];}return sv;}
 function curve(res,col){if(!res||!res.eis)return '';let cum=0,d='';for(let k=0;k<res.eis.length;k++){let ei=res.eis[k],x0=pad+cum/TOT*gx,x1=pad+(cum+EL[ei])/TOT*gx,y=gy0+gyH-esf(ei)/100*gyH;d+=(k?'L':'M')+x0.toFixed(1)+' '+y.toFixed(1)+'L'+x1.toFixed(1)+' '+y.toFixed(1);cum+=EL[ei];}return '<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="1.6"/>';}
 let svg='<svg viewBox="0 0 '+W+' '+H+'" width="100%" height="'+H+'" preserveAspectRatio="none" style="display:block">';
 svg+=band(c,ty)+band(s,ty+bnd+gp);
 svg+='<line x1="'+pad+'" y1="'+(gy0+gyH)+'" x2="'+(W-pad)+'" y2="'+(gy0+gyH)+'" stroke="#ccc"/>'+curve(c,'#10406e')+curve(s,'#c0651a');
 if(s&&s.len<TOT){let xs=(pad+s.len/TOT*gx).toFixed(1);svg+='<line x1="'+xs+'" y1="'+ty+'" x2="'+xs+'" y2="'+(gy0+gyH)+'" stroke="#c0651a" stroke-width="0.7" stroke-dasharray="2 2" opacity="0.75"/><text x="'+xs+'" y="'+(ty-1)+'" font-size="6" fill="#c0651a" text-anchor="middle">Shortest end</text>';}
 svg+='<text x="1" y="'+(gy0+6)+'" font-size="7" fill="#777">100%</text><text x="1" y="'+(gy0+gyH)+'" font-size="7" fill="#777">0</text>';
 svg+='<text x="'+pad+'" y="'+(H-2)+'" font-size="7" fill="#777">0</text><text x="'+(W-pad)+'" y="'+(H-2)+'" font-size="7" fill="#777" text-anchor="end">'+Math.round(TOT)+' m</text>';
 svg+='<line id="pcurC" x1="-10" y1="'+ty+'" x2="-10" y2="'+(gy0+gyH)+'" stroke="#10406e" stroke-width="1.4" opacity="0.95"/><line id="pcurS" x1="-10" y1="'+ty+'" x2="-10" y2="'+(gy0+gyH)+'" stroke="#c0651a" stroke-width="1.4" opacity="0.95"/></svg>';
 let leg='';[2,4,3,1,5,0].forEach(f=>{leg+='<span style="display:inline-block;width:9px;height:9px;background:'+FACCOL[f]+';opacity:.85;margin:0 2px 0 6px;vertical-align:-1px"></span>'+FACNM[f];});
 document.getElementById('profBody').innerHTML='<div class="sec" style="margin:1px 0 2px"><b style="color:#10406e">━</b>最遮荫路 '+Math.round(c.len)+'m <b style="color:#c0651a">━</b>最短路 '+Math.round(s.len)+'m · 遮荫率(纵)×实际距离(横)</div>'+svg+'<div style="font-size:9px;color:#555">上带=最遮荫路设施,中带=最短路设施 &nbsp;'+leg+'</div>';
 P.style.display='block';}
// 网络分箱 MultiLineString
const SHC=['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac'];
const FLC=['#eee6d5','#f4cf86','#ec9f4e','#d97b2a','#b85a20','#7a3d18'];
const LWFULL=['interpolate',['linear'],['zoom'],11,0.6,14,1.4,17,3.2];
const LWHALF=['interpolate',['linear'],['zoom'],11,0.3,14,0.7,17,1.6];
function binFC(mode,srcset){let nb=mode==='shade'?8:6,bins=Array.from({length:nb},(_,b)=>({type:'Feature',properties:{b:b},geometry:{type:'MultiLineString',coordinates:[]}}));
 let EFx=(mode==='flow'&&facMode==='orig')?EFO:(mode==='flow'?flowCur():EF);
 let fmax=1;for(let i=0;i<NE;i++)if(EFx[i]>fmax)fmax=EFx[i];let lf=Math.log(fmax+1);
 for(let i=0;i<NE;i++){if(srcset&&srcset.indexOf(ESRCC[i])<0)continue;let b=mode==='shade'?Math.min(7,Math.floor(esf(i)/100*8)):Math.min(5,Math.floor(Math.log(EFx[i]+1)/lf*6));bins[b].geometry.coordinates.push(edgeLL(D.egeom[i]));}
 return {type:'FeatureCollection',features:bins};}
function facNetFC(){var co=[[],[],[],[],[]];for(let k=0;k<D.facnet.length;k++){co[D.facnet[k][1]].push(edgeLL(D.facnet[k][0]));}return {type:'FeatureCollection',features:co.map((c,g)=>({type:'Feature',properties:{g:g},geometry:{type:'MultiLineString',coordinates:c}}))};}
let facMode='all';
function ensureFac(){if(!map.getLayer('netfac')){map.addSource('netfac',{type:'geojson',data:facNetFC()});map.addLayer({id:'netfac',type:'line',source:'netfac',layout:{'line-cap':'round','line-join':'round'},paint:{'line-color':['match',['get','g'],0,'#d6336c',1,'#2b6cb0',2,'#9b59b6',3,'#12a5b0',4,'#e07b2a','#888'],'line-width':['interpolate',['linear'],['zoom'],11,1.8,14,3.4,17,5.5],'line-opacity':0.96}},'rs');}}
function redrawNet(){if(colorMode==='none'){map.setLayoutProperty('net','visibility','none');if(map.getLayer('netfac'))map.setLayoutProperty('netfac','visibility','none');updLeg();return;}
 map.setLayoutProperty('net','visibility','visible');var ss=facMode==='all'?null:[0];var hi=(facMode==='hi'||facMode==='hiM');var hiM=facMode==='hiM';
 if(colorMode==='flow'){map.getSource('net').setData(binFC('flow',ss));map.setPaintProperty('net','line-color',hi?'#b3bcc4':['match',['get','b'],0,FLC[0],1,FLC[1],2,FLC[2],3,FLC[3],4,FLC[4],5,FLC[5],'#888']);map.setPaintProperty('net','line-width',hi?LWHALF:LWFULL);}
 else if(colorMode==='plain'){map.getSource('net').setData(binFC('shade',ss));map.setPaintProperty('net','line-color',hi?'#b3bcc4':'#8c8c8c');map.setPaintProperty('net','line-width',LWHALF);}
 else{map.getSource('net').setData(binFC('shade',ss));map.setPaintProperty('net','line-color',hi?'#b3bcc4':['match',['get','b'],0,SHC[0],1,SHC[1],2,SHC[2],3,SHC[3],4,SHC[4],5,SHC[5],6,SHC[6],7,SHC[7],'#888']);map.setPaintProperty('net','line-width',hi?LWHALF:LWFULL);}
 if(hi){ensureFac();map.setLayoutProperty('netfac','visibility','visible');map.setPaintProperty('netfac','line-color',hiM?'#d6336c':['match',['get','g'],0,'#d6336c',1,'#2b6cb0',2,'#9b59b6',3,'#12a5b0',4,'#e07b2a','#888']);}else if(map.getLayer('netfac'))map.setLayoutProperty('netfac','visibility','none');
 updLeg();
 if(facMode==='orig')tip('已切到「仅原始人行道」,隐藏了设施小微网');else if(hiM)tip('已切到「统一高亮」,设施统一洋红');else if(facMode==='hi')tip('已切到「分类高亮」:红=骑楼 蓝=连廊 青=过街天桥 橙=楼宇连廊 紫=接入桥');else tip('已切到「全部路网」');}
function polyFC(rings){return {type:'FeatureCollection',features:rings.map(r=>({type:'Feature',properties:{},geometry:{type:'Polygon',coordinates:r.map(ringLL)}}))};}
// 3D bird-view: height FeatureCollections + tree octagon-prisms + view toggle/layers
function polyHFC(rings,H){return {type:'FeatureCollection',features:rings.map((r,i)=>({type:'Feature',properties:{h:(H&&H[i])||10},geometry:{type:'Polygon',coordinates:r.map(ringLL)}}))};}
function arcHFC(rings,BH){return {type:'FeatureCollection',features:rings.map((r,i)=>({type:'Feature',properties:{bh:(BH&&BH[i])||6},geometry:{type:'Polygon',coordinates:r.map(ringLL)}}))};}
function arcColFC(){var f=[],R=0.15,STEP=4.5,N=8;  // arcade colonnade: 0.3m-dia cylinder (octagon) every 4.5m along ring edges (0-3.6m)
 (D.arc||[]).forEach(function(rings){var ext=rings[0];if(!ext||ext.length<6)return;var acc=0;
   for(var j=0;j<ext.length-2;j+=2){var x1=ext[j],y1=ext[j+1],x2=ext[j+2],y2=ext[j+3],dx=x2-x1,dy=y2-y1,seg=Math.hypot(dx,dy);if(seg<0.1)continue;
     for(var d=(STEP-acc%STEP)%STEP;d<seg;d+=STEP){var t=d/seg,c=ll(x1+dx*t,y1+dy*t),mLat=111320,mLng=111320*Math.cos(c[1]*Math.PI/180),ring=[];
       for(var k=0;k<=N;k++){var a=k/N*2*Math.PI;ring.push([c[0]+R*Math.cos(a)/mLng,c[1]+R*Math.sin(a)/mLat]);}
       f.push({type:'Feature',properties:{},geometry:{type:'Polygon',coordinates:[ring]}});}
     acc+=seg;}});
 return {type:'FeatureCollection',features:f};}
function treePolyFC(){var GH={0:5,1:9,2:14,3:18},GR={0:2.0,1:3.2,2:5.0,3:7.0},N=8,T=D.tree3d,f=[];if(!T||!T.x)return{type:'FeatureCollection',features:f};
 for(var i=0;i<T.x.length;i++){var c=ll(T.x[i],T.y[i]),h=T.h[i]||GH[T.g[i]],r=GR[T.g[i]]||3,mLat=111320,mLng=111320*Math.cos(c[1]*Math.PI/180),ring=[];
  for(var k=0;k<=N;k++){var a=k/N*2*Math.PI;ring.push([c[0]+r*Math.cos(a)/mLng,c[1]+r*Math.sin(a)/mLat]);}
  f.push({type:'Feature',properties:{h:h,b:h*0.5},geometry:{type:'Polygon',coordinates:[ring]}});}
 return {type:'FeatureCollection',features:f};}
let view3d=false,l3dReady=false,arcCol=false,sunShadow=false,tree3dVis=false,tree3jsAdded=false;
function ensure3DLayers(){if(l3dReady)return;l3dReady=true;
 map.addSource('b3d',{type:'geojson',data:polyHFC(D.bld,D.bldH)});
 map.addLayer({id:'b3d',type:'fill-extrusion',source:'b3d',minzoom:12,paint:{'fill-extrusion-color':['interpolate',['linear'],['get','h'],0,'#cbc9c2',20,'#adaba4',60,'#8f8d88'],'fill-extrusion-height':['get','h'],'fill-extrusion-base':0,'fill-extrusion-opacity':0.32}},'net');
 map.addSource('a3d',{type:'geojson',data:arcHFC(D.arc,D.arcBH)});
 map.addLayer({id:'a3dTop',type:'fill-extrusion',source:'a3d',minzoom:12,paint:{'fill-extrusion-color':'#c0603a','fill-extrusion-base':3.6,'fill-extrusion-height':['max',['get','bh'],4.2],'fill-extrusion-opacity':0.92}},'net');
 map.addSource('a3dcol',{type:'geojson',data:arcColFC()});
 map.addLayer({id:'a3dCol',type:'fill-extrusion',source:'a3dcol',minzoom:14,layout:{visibility:'none'},paint:{'fill-extrusion-color':'#9a5238','fill-extrusion-height':3.6,'fill-extrusion-base':0,'fill-extrusion-opacity':1}},'net');
 map.addSource('l3d',{type:'geojson',data:polyFC(D.lkw)});
 map.addLayer({id:'l3d',type:'fill-extrusion',source:'l3d',minzoom:12,paint:{'fill-extrusion-color':'#3b7fb0','fill-extrusion-base':3.0,'fill-extrusion-height':3.3,'fill-extrusion-opacity':0.8}},'net');}
function set3DVis(v){['b3d','a3dTop','l3d'].forEach(id=>{if(map.getLayer(id))map.setLayoutProperty(id,'visibility',v?'visible':'none');});if(map.getLayer('a3dCol'))map.setLayoutProperty('a3dCol','visibility',(v&&arcCol)?'visible':'none');}
function toggle3D(){if(view3d){ensure3DLayers();set3DVis(true);map.easeTo({pitch:55,duration:600});map.dragRotate.enable();map.touchZoomRotate.enableRotation();
  ['vbld','varc','vlkw'].forEach(id=>{if(map.getLayer(id))map.setLayoutProperty(id,'visibility','none');});
  if(map.getLayer('itree'))map.setLayoutProperty('itree','visibility','none');
  if(document.getElementById('ly3DTree').checked){ensureTree3js();tree3dVis=true;map.triggerRepaint();}
  tip('3D 鸟瞰:拖拽旋转、右键调俯仰;半透明楼 + 骑楼架空体 + 连廊挑板(勾「3D 树木」看真树)');}
 else{set3DVis(false);tree3dVis=false;map.triggerRepaint();map.easeTo({pitch:0,bearing:0,duration:600});map.dragRotate.disable();
  [['vbld','lyBld'],['varc','lyArc'],['vlkw','lyLkw']].forEach(a=>{if(map.getLayer(a[0]))map.setLayoutProperty(a[0],'visibility',document.getElementById(a[1]).checked?'visible':'none');});
  if(map.getLayer('itree'))map.setLayoutProperty('itree','visibility',document.getElementById('lyTree').checked?'visible':'none');
  tip('已切回 2D 平面');}}
function llInv(lng,lat){return [lng*Math.PI/180*Rm-OX, Rm*Math.log(Math.tan(Math.PI/4+lat*Math.PI/360))-OY];}  // lnglat -> 3857-rel (inverse of ll)
// three.js scene (viewport-following): all-city trees + real sun shadow. On each moveend it rebuilds the in-view trees (<=TCAP) and building/arcade shadow-casters (<=CCAP), and sizes the shadow camera to the current view. Scene is re-centered on the view (mt = view-center mercator).
const _treeLayer={id:'tree3js',type:'custom',renderingMode:'3d',TCAP:40000,CCAP:3500,
 onAdd:function(map,gl){this.map=map;this.cam=new THREE.Camera();this.scene=new THREE.Scene();
  this.hemi=new THREE.HemisphereLight(0xffffff,0x9a9a88,0.95);this.scene.add(this.hemi);
  this.sun=new THREE.DirectionalLight(0xfff2d6,0.0);this.sun.castShadow=true;this.sun.shadow.mapSize.set(2048,2048);this.sun.shadow.bias=-0.0004;this.scene.add(this.sun);this.scene.add(this.sun.target);
  this.ground=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.ShadowMaterial({opacity:0.34}));this.ground.rotation.x=-Math.PI/2;this.ground.position.y=0.02;this.ground.receiveShadow=true;this.ground.visible=false;this.scene.add(this.ground);
  var cGeo=new THREE.SphereGeometry(1,8,6),cMat=new THREE.MeshLambertMaterial({color:0x63ad4a}),tGeo=new THREE.CylinderGeometry(1,1.3,1,6),tMat=new THREE.MeshLambertMaterial({color:0x6e4d30});
  this.canopy=new THREE.InstancedMesh(cGeo,cMat,this.TCAP);this.trunk=new THREE.InstancedMesh(tGeo,tMat,this.TCAP);this.canopy.castShadow=this.trunk.castShadow=true;this.canopy.count=0;this.trunk.count=0;this.canopy.visible=false;this.trunk.visible=false;this.scene.add(this.trunk);this.scene.add(this.canopy);
  this.casters=new THREE.Group();this.scene.add(this.casters);this.castMat=new THREE.MeshBasicMaterial({colorWrite:false,depthWrite:false});
  this.mt={x:0,y:0,z:0,s:1};
  this.renderer=new THREE.WebGLRenderer({canvas:map.getCanvas(),context:gl,antialias:true});this.renderer.autoClear=false;this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=THREE.PCFSoftShadowMap;this.renderer.shadowMap.autoUpdate=false;this.renderer.shadowMap.needsUpdate=true;
  this.rebuildView();},
 rebuildView:function(){if(!this.scene||this.map.getZoom()<13)return;var map=this.map,ctr=map.getCenter(),ref=maplibregl.MercatorCoordinate.fromLngLat(ctr,0),s=ref.meterInMercatorCoordinateUnits();this.mt={x:ref.x,y:ref.y,z:ref.z,s:s};
  var b=map.getBounds(),dw=(b.getEast()-b.getWest())*0.2,dh=(b.getNorth()-b.getSouth())*0.2,p0=llInv(b.getWest()-dw,b.getSouth()-dh),p1=llInv(b.getEast()+dw,b.getNorth()+dh),X0=Math.min(p0[0],p1[0]),X1=Math.max(p0[0],p1[0]),Y0=Math.min(p0[1],p1[1]),Y1=Math.max(p0[1],p1[1]),self=this;
  var GH={0:5,1:9,2:14,3:18},GR={0:1.7,1:2.8,2:4.5,3:6.5},dm=new THREE.Object3D(),T=D.tree3d||{x:[]},nn=0;
  for(var i=0;i<T.x.length&&nn<this.TCAP;i++){var tx=T.x[i],ty=T.y[i];if(tx<X0||tx>X1||ty<Y0||ty>Y1)continue;
    var c=ll(tx,ty),mc=maplibregl.MercatorCoordinate.fromLngLat({lng:c[0],lat:c[1]},0),east=(mc.x-ref.x)/s,south=(mc.y-ref.y)/s,h=T.h[i]||GH[T.g[i]]||8,cr=GR[T.g[i]]||3,tr=Math.max(0.12,cr*0.09),th=h*0.45;
    dm.position.set(east,th/2,south);dm.scale.set(tr,th,tr);dm.updateMatrix();this.trunk.setMatrixAt(nn,dm.matrix);
    dm.position.set(east,th+cr*0.6,south);dm.scale.set(cr,cr*1.05,cr);dm.updateMatrix();this.canopy.setMatrixAt(nn,dm.matrix);nn++;}
  this.canopy.count=nn;this.trunk.count=nn;this.canopy.instanceMatrix.needsUpdate=true;this.trunk.instanceMatrix.needsUpdate=true;
  while(this.casters.children.length){var ch=this.casters.children.pop();if(ch.geometry)ch.geometry.dispose();this.casters.remove(ch);}
  function addCaster(ext,base,top){if(!ext||ext.length<6||top<=base)return;var shp=new THREE.Shape();for(var k=0;k<ext.length;k+=2){var cc=ll(ext[k],ext[k+1]),m2=maplibregl.MercatorCoordinate.fromLngLat({lng:cc[0],lat:cc[1]},0),e=(m2.x-ref.x)/s,so=(m2.y-ref.y)/s;if(k===0)shp.moveTo(e,-so);else shp.lineTo(e,-so);}
    var g=new THREE.ExtrudeGeometry(shp,{depth:top-base,bevelEnabled:false});g.rotateX(-Math.PI/2);g.translate(0,base,0);var mm=new THREE.Mesh(g,self.castMat);mm.castShadow=true;self.casters.add(mm);}
  var cc=0;for(var bi=0;bi<D.bld.length&&cc<this.CCAP;bi++){var be=D.bld[bi][0];if(!be||be[0]<X0||be[0]>X1||be[1]<Y0||be[1]>Y1)continue;addCaster(be,0,D.bldH[bi]||10);cc++;}
  for(var ai=0;ai<D.arc.length&&cc<this.CCAP+1500;ai++){var ae=D.arc[ai][0];if(!ae||ae[0]<X0||ae[0]>X1||ae[1]<Y0||ae[1]>Y1)continue;addCaster(ae,3.6,Math.max(D.arcBH[ai]||6,4.2));cc++;}
  for(var li=0;li<D.lkw.length&&cc<this.CCAP+3000;li++){var le=D.lkw[li][0];if(!le||le[0]<X0||le[0]>X1||le[1]<Y0||le[1]>Y1)continue;addCaster(le,3.0,3.35);cc++;}
  var spanM=Math.max(200,(b.getEast()-b.getWest())*111320*Math.cos(ctr.lat*Math.PI/180));this.ground.scale.set(spanM*2,spanM*2,1);
  var scam=this.sun.shadow.camera,hw=spanM*0.8;scam.left=-hw;scam.right=hw;scam.top=hw;scam.bottom=-hw;scam.near=1;scam.far=Math.max(2000,spanM*4);scam.updateProjectionMatrix();
  this.setSun(D.hours[hidx()]);this.renderer.shadowMap.needsUpdate=true;},
 setSun:function(hour){if(!this.sun)return;var s=sunAltAz(hour),az=s.az,alt=Math.max(0.04,s.alt),dist=Math.max(500,this.ground?this.ground.scale.x:800);
  var ex=Math.sin(az)*Math.cos(alt),up=Math.max(0.08,Math.sin(alt)),so=-Math.cos(az)*Math.cos(alt);this.sun.position.set(ex*dist,up*dist,so*dist);this.sun.target.position.set(0,0,0);this.sun.target.updateMatrixWorld();if(this.renderer)this.renderer.shadowMap.needsUpdate=true;},
 applyShadow:function(){if(this.ground)this.ground.visible=sunShadow;if(this.sun)this.sun.intensity=sunShadow?1.05:0.0;if(this.renderer)this.renderer.shadowMap.needsUpdate=true;},
 applyTrees:function(){if(this.canopy)this.canopy.visible=tree3dVis;if(this.trunk)this.trunk.visible=tree3dVis;if(this.renderer)this.renderer.shadowMap.needsUpdate=true;},
 render:function(gl,matrix){if((!tree3dVis&&!sunShadow)||this.map.getZoom()<13)return;var mt=this.mt,rx=new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(1,0,0),Math.PI/2);
  var l=new THREE.Matrix4().makeTranslation(mt.x,mt.y,mt.z).multiply(new THREE.Matrix4().makeScale(mt.s,-mt.s,mt.s)).multiply(rx);
  this.cam.projectionMatrix=new THREE.Matrix4().fromArray(matrix).multiply(l);this.renderer.resetState();this.renderer.render(this.scene,this.cam);}};
function ensureTree3js(){if(tree3jsAdded)return;tree3jsAdded=true;if(typeof THREE==='undefined'){tip('three.js 未加载(需联网),3D 树木不可用');return;}if(!map.getLayer('tree3js'))map.addLayer(_treeLayer);map.on('moveend',function(){if((tree3dVis||sunShadow)&&map.getZoom()>=13){_treeLayer.rebuildView();map.triggerRepaint();}});}
document.getElementById('ly3DTree').onchange=function(){ensureTree3js();tree3dVis=this.checked;if(tree3dVis&&_treeLayer.rebuildView)_treeLayer.rebuildView();if(_treeLayer.applyTrees)_treeLayer.applyTrees();map.triggerRepaint();tip(tree3dVis?'3D 真树已开:缩放到街区(zoom≥13,2D/3D 视图都可)显示视野内真树':'3D 树木已关');};
function imgCoords(ext){let a=ll(ext[0],ext[3]),b=ll(ext[2],ext[3]),c=ll(ext[2],ext[1]),d=ll(ext[0],ext[1]);return [a,b,c,d];}
const TSRC={onemap:['https://www.onemap.gov.sg/maps/tiles/Grey/{z}/{x}/{y}.png'],osm:['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],gray:['https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png','https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png','https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png','https://d.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'],gsat:['https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}','https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}','https://mt2.google.com/vt/lyrs=s&x={x}&y={y}&z={z}','https://mt3.google.com/vt/lyrs=s&x={x}&y={y}&z={z}']};
let baseMap='gray';
const ctr=ll((D.xmin+D.xmax)/2,(D.ymin+D.ymax)/2);
const map=new maplibregl.Map({container:'map',style:{version:8,glyphs:'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',sources:{},layers:[{id:'bg',type:'background',paint:{'background-color':'#e9edf0'}}]},center:ctr,zoom:11,attributionControl:false,dragRotate:false,pitchWithRotate:false,maxPitch:70});
map.addControl(new maplibregl.AttributionControl({customAttribution:'© OpenStreetMap / © CARTO / OneMap / Google · ShadeWalk'}));
map.addControl(new maplibregl.ScaleControl({maxWidth:130,unit:'metric'}),'bottom-right');
map.addControl(new maplibregl.NavigationControl({showZoom:false,showCompass:true,visualizePitch:true}),'bottom-right');
let O=-1,Dst=-1,OA=null,DA=null,colorMode='plain',routeMode='both';
function setBase(){['onemap','osm','gray','gsat'].forEach(k=>{if(map.getLayer('base_'+k))map.removeLayer('base_'+k);if(map.getSource('base_'+k))map.removeSource('base_'+k);});
 if(baseMap==='none')return;
 map.addSource('base_'+baseMap,{type:'raster',tiles:TSRC[baseMap],tileSize:256});
 var _ls=map.getStyle().layers,_bid;for(var _i=0;_i<_ls.length;_i++){if(_ls[_i].id.indexOf('base_')!==0&&_ls[_i].type!=='background'){_bid=_ls[_i].id;break;}}
 map.addLayer({id:'base_'+baseMap,type:'raster',source:'base_'+baseMap,paint:{'raster-opacity':baseMap==='gsat'?1.0:0.85}},_bid);}
map.on('load',()=>{
 setBase();
 map.addSource('net',{type:'geojson',data:binFC('shade')});
 map.addLayer({id:'net',type:'line',source:'net',layout:{'line-cap':'round','line-join':'round'},
   paint:{'line-color':['match',['get','b'],0,SHC[0],1,SHC[1],2,SHC[2],3,SHC[3],4,SHC[4],5,SHC[5],6,SHC[6],7,SHC[7],'#888'],
          'line-width':LWFULL}});
 map.addSource('rc',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
 map.addSource('rs',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
 map.addLayer({id:'rs',type:'line',source:'rs',layout:{'line-cap':'round'},paint:{'line-color':'#E0791E','line-width':5}});
 map.addLayer({id:'rc',type:'line',source:'rc',layout:{'line-cap':'round'},paint:{'line-color':'#0F9E75','line-width':5}});
 map.addSource('acc',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
 map.addLayer({id:'acc',type:'line',source:'acc',layout:{'line-cap':'round'},paint:{'line-color':'#3a3a3a','line-width':2,'line-dasharray':[1.4,1.4],'line-opacity':0.85}});
 map.addSource('od',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
 map.addLayer({id:'od',type:'circle',source:'od',paint:{'circle-radius':9,'circle-color':['get','c'],'circle-stroke-width':2,'circle-stroke-color':'#fff'}});
 map.addLayer({id:'odlab',type:'symbol',source:'od',layout:{'text-field':['get','lab'],'text-size':13,'text-font':['Open Sans Semibold'],'text-allow-overlap':true,'text-ignore-placement':true},paint:{'text-color':'#fff'}});
 updLeg();
});
function lazyVec(id,rings,color,op){if(map.getLayer(id))return;map.addSource(id,{type:'geojson',data:polyFC(rings)});
 map.addLayer({id:id,type:'fill',source:id,paint:{'fill-color':color,'fill-opacity':op,'fill-outline-color':color}},'net');}
function facFC(rings,flows){return {type:'FeatureCollection',features:rings.map((r,i)=>({type:'Feature',properties:{f:flows[i]},geometry:{type:'Polygon',coordinates:r.map(ringLL)}}))};}
function lazyVecF(id,rings,flows,color,op){if(map.getLayer(id))return;map.addSource(id,{type:'geojson',data:facFC(rings,flows)});
 map.addLayer({id:id,type:'fill',source:id,paint:{'fill-color':color,'fill-opacity':op,'fill-outline-color':color}},'net');}
function lazyImg(id,url,ext,op){if(map.getLayer(id))return;map.addSource(id,{type:'image',url:url,coordinates:imgCoords(ext)});
 map.addLayer({id:id,type:'raster',source:id,paint:{'raster-opacity':op,'raster-resampling':'nearest'},layout:{visibility:'none'}},'net');}
function toggle(cb,id,mk){let on=document.getElementById(cb).checked;if(on&&!map.getLayer(id))mk();if(map.getLayer(id))map.setLayoutProperty(id,'visibility',on?'visible':'none');}
document.getElementById('lyBld').onchange=function(){toggle('lyBld','vbld',()=>lazyVec('vbld',D.bld,'#8a8a8a',0.45));};
let scnBoot=true;  // suppress scenario tips during initial default-on
function scnApply(){redrawNet();if(O>=0&&Dst>=0)route();
 if(scnBoot)return;
 tip((SCN.arc&&SCN.lkw)?'设施场景:全设施(骑楼+连廊计入寻路与遮荫)':('设施场景:已移除 '+(!SCN.arc&&!SCN.lkw?'骑楼+连廊':(!SCN.arc?'骑楼':'连廊'))+' — 其中线边禁行,其下路段遮荫回退为仅建筑+树(Coolest/卡片/剖面/拓扑图同步重算)'));}
document.getElementById('lyArc').onchange=function(){toggle('lyArc','varc',()=>lazyVecF('varc',D.arc,D.arcF,'#d4322c',0.7));SCN.arc=this.checked;scnApply();};
document.getElementById('lyLkw').onchange=function(){toggle('lyLkw','vlkw',()=>lazyVecF('vlkw',D.lkw,D.lkwF,'#2c7fb8',0.55));SCN.lkw=this.checked;scnApply();};
document.getElementById('lyHdb').onchange=function(){toggle('lyHdb','vhdb',()=>lazyVec('vhdb',D.hdb,'#5b4a6a',0.5));};  // HDB stilt floor (void deck), display-only layer
map.on('load',function(){['lyArc','lyLkw'].forEach(function(id){var el2=document.getElementById(id);el2.checked=true;el2.onchange();});scnBoot=false;});  // default ON: facilities shown + counted; unchecking = removal scenario
['varc','vlkw'].forEach(L=>{let nm=L==='varc'?'骑楼':'有盖连廊';
 map.on('mousemove',L,e=>{let n=Math.round(e.features[0].properties.f*D.facR[hidx()]);tip(nm+' · 服务人流 ≈ '+n.toLocaleString()+' ppl ('+D.hours[hidx()]+':00)');map.getCanvas().style.cursor='help';});
 map.on('mouseleave',L,()=>{tip(0);map.getCanvas().style.cursor='';});});
// all rasters inlined as image sources (self-contained, no server): tree 10m PNG + shadow/category 40m hourly frames
document.getElementById('lyTree').onchange=function(){toggle('lyTree','itree',()=>lazyImg('itree',D.treeImg,D.extT,0.92));};
document.getElementById('lyShad').onchange=function(){toggle('lyShad','ish',()=>lazyImg('ish',D.shadH[hidx()],D.extS,0.7));};
document.getElementById('lyCat').onchange=function(){toggle('lyCat','icat',()=>lazyImg('icat',D.catH[hidx()],D.extC,0.72));catLegUpd();};
const CATPAL={1:'#787884',2:'#2e8b3d',3:'#5f9652',4:'#78aad2',5:'#6e8ca5',6:'#50a08c',7:'#5f7d73',8:'#d4322c',9:'#e1733c',10:'#965fa5',11:'#b95578',12:'#46464e'};
const CATNM2=['','建筑阴影','树荫','建筑+树','连廊荫','建筑+连廊','树+连廊','建筑+树+连廊','骑楼走道','树+骑楼','连廊+骑楼','树+连廊+骑楼','建筑底座'];
function catLegUpd(){var cl=document.getElementById('catleg');if(!document.getElementById('lyCat')||!document.getElementById('lyCat').checked){if(cl)cl.style.display='none';return;}
 var h='<b style="font-weight:600">'+D.hours[hidx()]+':00 遮荫来源分类</b><div style="display:grid;grid-template-columns:1fr 1fr;gap:0 10px;margin-top:3px">';
 for(var k=1;k<=12;k++)h+='<span style="white-space:nowrap"><span class="sw" style="background:'+CATPAL[k]+'"></span>'+CATNM2[k]+'</span>';
 cl.innerHTML=h+'</div><div style="font-size:9px;color:#888;margin-top:2px">日晒像素透明;组合=被多种同时遮挡</div>';cl.style.display='block';}
function hidx(){return +document.getElementById('hr').value;}
document.getElementById('hr').oninput=function(){let h=hidx();document.getElementById('hrLbl').textContent=D.hours[h]+':00';if(map.getSource('ish'))map.getSource('ish').updateImage({url:D.shadH[h]});if(map.getSource('icat'))map.getSource('icat').updateImage({url:D.catH[h]});updateSunLight();catLegUpd();};
// selectable O/D points: stations + building centroids (click sets origin/dest; cached FC)
let staFC=null,bldPtFC=null;
function staGeo(){if(!staFC){let f=[];for(let i=0;i<D.stx.length;i++)f.push({type:'Feature',properties:{c:D.stc[i],m:D.sts[i],r:D.srid[i]},geometry:{type:'Point',coordinates:ll(D.stx[i],D.sty[i])}});staFC={type:'FeatureCollection',features:f};}return staFC;}
function bldPtGeo(){if(!bldPtFC){let f=[];for(let i=0;i<D.bcx.length;i++)f.push({type:'Feature',properties:{},geometry:{type:'Point',coordinates:ll(D.bcx[i],D.bcy[i])}});bldPtFC={type:'FeatureCollection',features:f};}return bldPtFC;}
document.getElementById('lyStaPt').onchange=function(){toggle('lyStaPt','odSta',()=>{map.addSource('odSta',{type:'geojson',data:staGeo()});
 map.addLayer({id:'odSta',type:'circle',source:'odSta',layout:{visibility:'none'},paint:{'circle-radius':['interpolate',['linear'],['zoom'],11,['case',['==',['get','m'],1],3,1.2],14,['case',['==',['get','m'],1],6,2.6],17,['case',['==',['get','m'],1],9,5]],'circle-color':['case',['==',['get','m'],1],'#f39c12','#1f78b4'],'circle-stroke-width':1,'circle-stroke-color':'#fff','circle-opacity':0.92}},'rs');});};
document.getElementById('lyBldPt').onchange=function(){toggle('lyBldPt','odBld',()=>{map.addSource('odBld',{type:'geojson',data:bldPtGeo()});
 map.addLayer({id:'odBld',type:'circle',source:'odBld',minzoom:14,layout:{visibility:'none'},paint:{'circle-radius':['interpolate',['linear'],['zoom'],14,1.5,17,3.2],'circle-color':'#6a51a3','circle-opacity':0.8,'circle-stroke-width':0.5,'circle-stroke-color':'#fff'}},'rs');});};
['odSta','odBld'].forEach(L=>{map.on('mouseenter',L,()=>map.getCanvas().style.cursor='pointer');map.on('mouseleave',L,()=>{map.getCanvas().style.cursor='';});});
map.on('mousemove','odSta',e=>{let p=e.features[0].properties;tip((p.m==1?'MRT ':'巴士 ')+p.c+' · 工作日客流 '+p.r);});
map.on('mouseleave','odSta',()=>tip(0));
function pick(g,ids){ids.forEach(i=>document.getElementById(i).classList.remove('on'));g.classList.add('on');}
document.getElementById('v3d').onclick=function(){view3d=true;pick(this,['v2d','v3d']);toggle3D();};
document.getElementById('v2d').onclick=function(){view3d=false;pick(this,['v2d','v3d']);toggle3D();};
document.getElementById('aCol').onclick=function(){arcCol=!arcCol;this.classList.toggle('on',arcCol);if(view3d){ensure3DLayers();map.setLayoutProperty('a3dCol','visibility',arcCol?'visible':'none');}if(typeof POV!=='undefined'&&POV.C&&POV.C.path&&document.getElementById('pov').style.display!=='none')povBuild();tip(arcCol?'骑楼柱子:开(鸟瞰 + 行人视角骑楼下都显柱)':'骑楼柱子:关(悬浮体,底层透空)');};
// sun shadow: hourly SOLWEIG shade raster on ground + directional light on extrusions, follows hr slider
function sunAltAz(hour){var DOY=60,lat=1.35*Math.PI/180,lon=103.8,tz=8,hh=hour-0.5;  // NOAA solar position; SOLWEIG date 2026-03-01(DOY 60), Singapore 1.35N/103.8E UTC+8; hh=hour-0.5 = SOLWEIG half-step (band15=14:00 -> sun at 13:30)
 var g=2*Math.PI/365*(DOY-1+(hh-12)/24);
 var decl=0.006918-0.399912*Math.cos(g)+0.070257*Math.sin(g)-0.006758*Math.cos(2*g)+0.000907*Math.sin(2*g)-0.002697*Math.cos(3*g)+0.00148*Math.sin(3*g);
 var eot=229.18*(0.000075+0.001868*Math.cos(g)-0.032077*Math.sin(g)-0.014615*Math.cos(2*g)-0.040849*Math.sin(2*g));
 var tst=hh*60+eot+4*lon-60*tz,H=(tst/4-180)*Math.PI/180;
 var sa=Math.sin(lat)*Math.sin(decl)+Math.cos(lat)*Math.cos(decl)*Math.cos(H),alt=Math.asin(Math.max(-1,Math.min(1,sa)));
 var ca=(Math.sin(decl)-Math.sin(lat)*sa)/(Math.cos(lat)*Math.cos(alt)+1e-9),az=Math.acos(Math.max(-1,Math.min(1,ca)));if(H>0)az=2*Math.PI-az;
 return {alt:alt,az:az};}  // az clockwise from north
function sunPolarAz(hour){var s=sunAltAz(hour);return {az:s.az*180/Math.PI,polar:Math.max(4,90-s.alt*180/Math.PI)};}
function updateSunLight(){var s=sunPolarAz(D.hours[hidx()]);map.setLight({anchor:'map',position:[1.5,s.az,s.polar],color:'#fff4e0',intensity:sunShadow?0.5:0.3});if(tree3jsAdded&&_treeLayer.setSun)_treeLayer.setSun(D.hours[hidx()]);if(typeof POV!=='undefined'&&POV.C&&POV.C.scene&&document.getElementById('pov').style.display!=='none')povFrameBoth();}
document.getElementById('bSun').onclick=function(){if(!view3d){tip('请先切到「3D 鸟瞰」再开太阳阴影');return;}sunShadow=!sunShadow;this.classList.toggle('on',sunShadow);
  ensureTree3js();if(sunShadow&&_treeLayer.rebuildView)_treeLayer.rebuildView();if(_treeLayer.setSun)_treeLayer.setSun(D.hours[hidx()]);if(_treeLayer.applyShadow)_treeLayer.applyShadow();updateSunLight();map.triggerRepaint();
  tip(sunShadow?'太阳阴影:three.js 真实投射(视野内建筑/骑楼/树,随时刻),缩放到街区看,拖时刻滑杆影子转':'太阳阴影已关');};
// north-reset now via the compass control (bottom-right, above the scale bar; click compass to reset north)
document.getElementById('cShade').onclick=function(){colorMode='shade';pick(this,['cShade','cFlow','cPlain','cNone']);redrawNet();cleanView();tip('已自动干净展示:Simple 底图 · 2D · 关太阳阴影 · Routes Hide(均可手动改回)');};
document.getElementById('cFlow').onclick=function(){colorMode='flow';pick(this,['cShade','cFlow','cPlain','cNone']);redrawNet();cleanView();tip('已自动干净展示:Simple 底图 · 2D · 关太阳阴影 · Routes Hide(均可手动改回)');};
document.getElementById('cPlain').onclick=function(){colorMode='plain';pick(this,['cShade','cFlow','cPlain','cNone']);redrawNet();};
document.getElementById('cNone').onclick=function(){colorMode='none';pick(this,['cShade','cFlow','cPlain','cNone']);redrawNet();};
document.getElementById('fAll').onclick=function(){facMode='all';pick(this,['fAll','fOrig','fHi','fHiM']);ensureNetVisible();redrawNet();};
document.getElementById('fOrig').onclick=function(){facMode='orig';pick(this,['fAll','fOrig','fHi','fHiM']);ensureNetVisible();redrawNet();};
document.getElementById('fHi').onclick=function(){facMode='hi';pick(this,['fAll','fOrig','fHi','fHiM']);ensureNetVisible();cleanView();redrawNet();tip('已切到「分类高亮」:红=骑楼 蓝=连廊 青=过街天桥 橙=楼宇连廊 紫=接入桥 · 已自动干净展示:Simple 底图 · 2D · 关太阳阴影 · Routes Hide(均可手动改回)');};
document.getElementById('fHiM').onclick=function(){facMode='hiM';pick(this,['fAll','fOrig','fHi','fHiM']);ensureNetVisible();cleanView();redrawNet();tip('已切到「统一高亮」,设施统一洋红 · 已自动干净展示:Simple 底图 · 2D · 关太阳阴影 · Routes Hide(均可手动改回)');};
function setRM(m,el){routeMode=m;pick(el,['rBoth','rCool','rShort','rHide']);map.setLayoutProperty('rc','visibility',(m==='both'||m==='cool')?'visible':'none');map.setLayoutProperty('rs','visibility',(m==='both'||m==='short')?'visible':'none');}
document.getElementById('rBoth').onclick=function(){setRM('both',this);};document.getElementById('rCool').onclick=function(){setRM('cool',this);};document.getElementById('rShort').onclick=function(){setRM('short',this);};document.getElementById('rHide').onclick=function(){setRM('hide',this);};
var DTH=[1,1.2,1.5,2];document.getElementById('detTh').oninput=function(){DETOUR=DTH[+this.value];document.getElementById('detThV').textContent=DETOUR+'×';if(colorMode==='flow')redrawNet();if(O>=0&&Dst>=0)route();};
function setBM(m,el){baseMap=m;pick(el,['bmOne','bmOsm','bmGray','bmGsat','bmNone']);setBase();}
function cleanView(){ // one-click clean display for network/facility layers: clean basemap + 2D + shadow off + routes hidden
 if(baseMap!=='gray'&&baseMap!=='none')setBM('gray',document.getElementById('bmGray'));
 if(view3d){view3d=false;pick(document.getElementById('v2d'),['v2d','v3d']);toggle3D();}
 if(sunShadow){sunShadow=false;document.getElementById('bSun').classList.remove('on');if(_treeLayer.applyShadow)_treeLayer.applyShadow();updateSunLight();map.triggerRepaint();}
 setRM('hide',document.getElementById('rHide'));}
function ensureNetVisible(){if(colorMode==='none'){colorMode='plain';pick(document.getElementById('cPlain'),['cShade','cFlow','cPlain','cNone']);}}  // facility tabs need the net layer pipeline: Hide -> auto-switch to Plain
document.getElementById('bmOne').onclick=function(){setBM('onemap',this);};document.getElementById('bmOsm').onclick=function(){setBM('osm',this);};document.getElementById('bmGray').onclick=function(){setBM('gray',this);};document.getElementById('bmGsat').onclick=function(){setBM('gsat',this);};document.getElementById('bmNone').onclick=function(){setBM('none',this);};
function tip(t){let e=document.getElementById('tip');if(!t){e.style.display='none';return;}e.textContent=t;e.style.display='block';}
function odUpd(){let f=[],af=[];
 if(O>=0&&OA){f.push({type:'Feature',properties:{c:'#0F6E56',lab:'O'},geometry:{type:'Point',coordinates:[OA.clng,OA.clat]}});if(OA.acc>0.5)af.push({type:'Feature',properties:{},geometry:{type:'LineString',coordinates:[[OA.clng,OA.clat],[OA.flng,OA.flat]]}});}
 if(Dst>=0&&DA){f.push({type:'Feature',properties:{c:'#c0392b',lab:'D'},geometry:{type:'Point',coordinates:[DA.clng,DA.clat]}});if(DA.acc>0.5)af.push({type:'Feature',properties:{},geometry:{type:'LineString',coordinates:[[DA.clng,DA.clat],[DA.flng,DA.flat]]}});}
 map.getSource('od').setData({type:'FeatureCollection',features:f});if(map.getSource('acc'))map.getSource('acc').setData({type:'FeatureCollection',features:af});}
let hadRoute=false,FITRT=false;  // auto-hide network colouring once when the first OD route appears; FITRT = fit view to the full route extent after computing (set by demo/import flows)
function route(){if(O<0||Dst<0)return;tip('计算中…');
 setTimeout(()=>{let s=dij(O,Dst,'short'),c=coolDetour(O,Dst,s);tip(0);
  if(!c||!s){map.getSource('rc').setData({type:'FeatureCollection',features:[]});map.getSource('rs').setData({type:'FeatureCollection',features:[]});document.getElementById('results').innerHTML='<div class="card">两点不在同一连通区域。</div>';profile(null,null);return;}
  let oA=(OA&&OA.acc>0.5)?[[OA.clng,OA.clat],[OA.flng,OA.flat],ll(D.nx[O],D.ny[O])]:null,dA=(DA&&DA.acc>0.5)?[ll(D.nx[Dst],D.ny[Dst]),[DA.flng,DA.flat],[DA.clng,DA.clat]]:null;
  let wa=r=>{let co=r.coords.slice();if(oA)co.unshift(oA);if(dA)co.push(dA);return co;};
  let wc=wa(c),ws=wa(s);
  map.getSource('rc').setData({type:'Feature',geometry:{type:'MultiLineString',coordinates:wc},properties:{}});
  map.getSource('rs').setData({type:'Feature',geometry:{type:'MultiLineString',coordinates:ws},properties:{}});
  if(FITRT){FITRT=false;let mn=[1e9,1e9],mx=[-1e9,-1e9];wc.concat(ws).forEach(function(seg){seg.forEach(function(p){if(p[0]<mn[0])mn[0]=p[0];if(p[1]<mn[1])mn[1]=p[1];if(p[0]>mx[0])mx[0]=p[0];if(p[1]>mx[1])mx[1]=p[1];});});if(mx[0]>mn[0])map.fitBounds([mn,mx],{padding:90,duration:650});}
  let acc=((OA&&OA.acc)||0)+((DA&&DA.acc)||0),cF=c.len+acc,sF=s.len+acc,mn=x=>Math.round(x/80),det=c.len/s.len;
  let accNote=acc>0.5?'<div class="card" style="background:#eef1f4"><span class="mut">接入步行 '+Math.round(acc)+' m(灰虚线,从点击点垂直接入最近路段);下列为含接入的总行程。</span></div>':'';
  document.getElementById('results').innerHTML=accNote+'<div class="card cool"><div style="display:flex;justify-content:space-between"><b style="color:#0F6E56">最遮荫路</b><span class="mut">'+mn(cF)+' 分钟</span></div><div class="big" style="color:#0F6E56">'+Math.round(c.shade*100)+'% <span class="mut" style="font-size:12px">遮荫</span></div><div class="mut">'+(cF/1000).toFixed(2)+' km · 日晒 '+Math.round(c.len*(1-c.shade))+' m · '+pathFlow(c)+' ppl</div>'+facLine(c)+'</div>'+
   '<div class="card short"><div style="display:flex;justify-content:space-between"><b style="color:#985316">最短路</b><span class="mut">'+mn(sF)+' 分钟</span></div><div class="big" style="color:#985316">'+Math.round(s.shade*100)+'% <span class="mut" style="font-size:12px">遮荫</span></div><div class="mut">'+(sF/1000).toFixed(2)+' km · 日晒 '+Math.round(s.len*(1-s.shade))+' m · '+pathFlow(s)+' ppl</div>'+facLine(s)+'</div>'+
   '<div class="card" style="background:#f4f6f4"><span class="mut">避热选最遮荫路:遮荫 +'+Math.round((c.shade-s.shade)*100)+' 个百分点,绕行 '+Math.round((det-1)*100)+'%(多走 '+Math.round(c.len-s.len)+' m)。</span></div>';if(!hadRoute){hadRoute=true;if(colorMode!=='none'){colorMode='none';pick(document.getElementById('cNone'),['cShade','cFlow','cPlain','cNone']);redrawNet();tip('已自动隐藏路网着色以突出 OD 路线(可在 Network view 重新开启)');}}profile(c,s);POV.RC=c;POV.RS=s;if(document.getElementById('pov').style.display!=='none')povBuild();if(topoOpen())topoBuild(true);
 },20);}
map.on('click',e=>{let lng=e.lngLat.lng,lat=e.lngLat.lat,isPt=false;
 let ql=['odSta','odBld'].filter(l=>map.getLayer(l));
 if(ql.length){let qf=map.queryRenderedFeatures(e.point,{layers:ql});if(qf.length&&qf[0].geometry.type==='Point'){let c=qf[0].geometry.coordinates;lng=c[0];lat=c[1];isPt=true;}}
 let mx=lng/180*Math.PI*Rm-OX, my=Rm*Math.log(Math.tan(Math.PI/4+lat/180*Math.PI/2))-OY,snap;
 if(isPt){let n=nearest(mx,my);if(n<0)return;snap={node:n,clng:lng,clat:lat,flng:lng,flat:lat,acc:0};}
 else{let se=snapEdge(mx,my);if(se)snap={node:se.node,clng:lng,clat:lat,flng:se.flng,flat:se.flat,acc:se.acc};else{let n=nearest(mx,my);if(n<0)return;let nl=ll(D.nx[n],D.ny[n]);snap={node:n,clng:lng,clat:lat,flng:nl[0],flat:nl[1],acc:haversine([lng,lat],nl)};}}
 if(O<0||(O>=0&&Dst>=0)){O=snap.node;OA=snap;Dst=-1;DA=null;map.getSource('rc').setData({type:'FeatureCollection',features:[]});map.getSource('rs').setData({type:'FeatureCollection',features:[]});document.getElementById('results').innerHTML='';document.getElementById('oTxt').textContent='起点:已选';document.getElementById('dTxt').textContent='终点:再次点击地图';}
 else{Dst=snap.node;DA=snap;document.getElementById('dTxt').textContent='终点:已选';route();}odUpd();});
document.getElementById('bClear').onclick=()=>{O=-1;Dst=-1;OA=null;DA=null;hadRoute=false;map.getSource('rc').setData({type:'FeatureCollection',features:[]});map.getSource('rs').setData({type:'FeatureCollection',features:[]});odUpd();document.getElementById('results').innerHTML='';document.getElementById('oTxt').textContent='起点:点击地图选择';document.getElementById('dTxt').textContent='终点:再次点击地图';document.getElementById('prof').style.display='none';document.getElementById('topo').style.display='none';document.getElementById('bTopo').classList.remove('on');if(map.getSource('rtopo'))map.getSource('rtopo').setData({type:'FeatureCollection',features:[]});povMarkClear();};
document.getElementById('profX').onclick=()=>document.getElementById('prof').style.display='none';
(function(){let P=document.getElementById('prof'),h=P.firstElementChild,dx=0,dy=0,drag=false;h.style.cursor='move';
 h.addEventListener('mousedown',e=>{if(e.target.id==='profX')return;drag=true;let r=P.getBoundingClientRect();dx=e.clientX-r.left;dy=e.clientY-r.top;P.style.bottom='auto';e.preventDefault();});
 document.addEventListener('mousemove',e=>{if(!drag)return;let pr=P.parentElement.getBoundingClientRect();P.style.left=Math.max(0,e.clientX-pr.left-dx)+'px';P.style.top=Math.max(0,e.clientY-pr.top-dy)+'px';});
 document.addEventListener('mouseup',()=>{drag=false;});})();
// === POV: two first-person three.js scenes (Coolest top + Shortest bottom) walked at the SAME speed via a shared POV.d in metres; Shortest finishes first and waits at its end. coords: x=east,y=up,z=-north ===
var POV={C:{},S:{},d:0,maxTotal:1,playing:false,RC:null,RS:null};
function povSunVec(hour){var s=sunAltAz(hour),a=Math.max(0.04,s.alt);return {e:Math.sin(s.az)*Math.cos(a),u:Math.sin(a),n:Math.cos(s.az)*Math.cos(a)};}
function povInitOne(k,cvId){var W=POV[k];if(W.r)return;var cv=document.getElementById(cvId);
 W.r=new THREE.WebGLRenderer({canvas:cv,antialias:true});W.r.setSize(440,168,false);W.r.shadowMap.enabled=true;W.r.shadowMap.type=THREE.PCFSoftShadowMap;
 W.scene=new THREE.Scene();W.scene.background=new THREE.Color(0xc2d8ea);W.scene.fog=new THREE.Fog(0xc2d8ea,150,470);
 W.cam=new THREE.PerspectiveCamera(74,440/168,0.4,1600);
 W.hemi=new THREE.HemisphereLight(0xffffff,0x9aa488,0.8);W.scene.add(W.hemi);
 W.sun=new THREE.DirectionalLight(0xfff2d6,1.0);W.sun.castShadow=true;W.sun.shadow.mapSize.set(1536,1536);W.sun.shadow.bias=-0.0004;
 var sc=W.sun.shadow.camera;sc.near=1;sc.far=1000;sc.left=-240;sc.right=240;sc.top=240;sc.bottom=-240;sc.updateProjectionMatrix();
 W.scene.add(W.sun);W.scene.add(W.sun.target);W.world=new THREE.Group();W.scene.add(W.world);}
function povClearOne(W){if(!W.world)return;for(var i=W.world.children.length-1;i>=0;i--){var c=W.world.children[i];if(c.geometry)c.geometry.dispose();W.world.remove(c);}}
function povFlat(rt){if(!rt||!rt.coords||!rt.coords.length)return null;  // flatten route edges O->D with per-chunk orientation (edge geometry direction is arbitrary)
 var chunks=rt.coords.slice().reverse(),oPos=(O>=0)?ll(D.nx[O],D.ny[O]):null,pts=[];
 chunks.forEach(function(seg){var s=seg.slice(),ref=pts.length?pts[pts.length-1]:oPos;
  if(ref){var d0=Math.hypot(s[0][0]-ref[0],s[0][1]-ref[1]),d1=Math.hypot(s[s.length-1][0]-ref[0],s[s.length-1][1]-ref[1]);if(d1<d0)s.reverse();}
  s.forEach(function(L){if(!pts.length||Math.abs(L[0]-pts[pts.length-1][0])>1e-7||Math.abs(L[1]-pts[pts.length-1][1])>1e-7)pts.push(L);});});
 return pts.length>1?pts:null;}
function povBuildOne(k,cvId,rt){povInitOne(k,cvId);var W=POV[k];povClearOne(W);
 var pts=povFlat(rt);
 if(!pts){W.path=null;return;}
 var o=pts[Math.floor(pts.length/2)],mLat=111320,mLng=111320*Math.cos(o[1]*Math.PI/180);W.o=o;
 W.path=pts.map(function(L){return [(L[0]-o[0])*mLng,(L[1]-o[1])*mLat];});
 W.cum=[0];for(var i=1;i<W.path.length;i++)W.cum.push(W.cum[i-1]+Math.hypot(W.path[i][0]-W.path[i-1][0],W.path[i][1]-W.path[i-1][1]));
 W.arc=W.cum[W.cum.length-1]||1;W.total=(rt.len&&rt.len>0)?rt.len:W.arc;W.d=0;W.playing=false;W.hdg=null;W.lastAd=undefined;W.hT=undefined;  // total = dij route length (matches profile x-axis); arc = simplified coords arc-length used for camera position
 var gnd=new THREE.Mesh(new THREE.PlaneGeometry(1800,1800),new THREE.MeshLambertMaterial({color:0xe9e5db}));gnd.rotation.x=-Math.PI/2;gnd.receiveShadow=true;W.world.add(gnd);
 var xs=W.path.map(function(p){return p[0];}),zs=W.path.map(function(p){return p[1];});
 var bx0=Math.min.apply(null,xs)-100,bx1=Math.max.apply(null,xs)+100,bz0=Math.min.apply(null,zs)-100,bz1=Math.max.apply(null,zs)+100;
 function E(c){return (c[0]-o[0])*mLng;} function Nf(c){return (c[1]-o[1])*mLat;}
 function inB(ext){if(!ext||ext.length<2)return false;var c=ll(ext[0],ext[1]),e=E(c),n=Nf(c);return e>=bx0&&e<=bx1&&n>=bz0&&n<=bz1;}
 function addPoly(ext,base,top,color){if(!ext||ext.length<6||top<=base)return;var shp=new THREE.Shape();for(var m=0;m<ext.length;m+=2){var c=ll(ext[m],ext[m+1]);if(m===0)shp.moveTo(E(c),Nf(c));else shp.lineTo(E(c),Nf(c));}
  var g=new THREE.ExtrudeGeometry(shp,{depth:top-base,bevelEnabled:false});g.rotateX(-Math.PI/2);g.translate(0,base,0);var mm=new THREE.Mesh(g,new THREE.MeshLambertMaterial({color:color}));mm.castShadow=true;mm.receiveShadow=true;W.world.add(mm);}
 function addArcCols(ext){if(!ext||ext.length<6)return;var CR=0.15,CS=4.5,ac=0;for(var mc=0;mc<ext.length-2;mc+=2){var q1=ll(ext[mc],ext[mc+1]),q2=ll(ext[mc+2],ext[mc+3]),e1=E(q1),n1=Nf(q1),dxx=E(q2)-e1,dzz=Nf(q2)-n1,sg=Math.hypot(dxx,dzz);if(sg<0.1)continue;for(var dd=(CS-ac%CS)%CS;dd<sg;dd+=CS){var tt=dd/sg,col=new THREE.Mesh(new THREE.CylinderGeometry(CR,CR,3.6,6),new THREE.MeshLambertMaterial({color:0x9a5238}));col.position.set(e1+dxx*tt,1.8,-(n1+dzz*tt));col.castShadow=true;W.world.add(col);}ac+=sg;}}
 var nb=0;for(var bi=0;bi<D.bld.length&&nb<600;bi++){if(inB(D.bld[bi][0])){addPoly(D.bld[bi][0],0,D.bldH[bi]||10,0xcfccc4);nb++;}}
 for(var ai=0;ai<D.arc.length;ai++){if(inB(D.arc[ai][0])){addPoly(D.arc[ai][0],3.6,Math.max(D.arcBH[ai]||6,4.2),0xc0603a);if(typeof arcCol!=='undefined'&&arcCol)addArcCols(D.arc[ai][0]);}}
 for(var li=0;li<D.lkw.length;li++){if(inB(D.lkw[li][0]))addPoly(D.lkw[li][0],3.0,3.4,0x3b7fb0);}
 var T=D.tree3d||{x:[]},GH={0:5,1:9,2:14,3:18},GR={0:1.7,1:2.8,2:4.5,3:6.5},tc=0;
 for(var ti=0;ti<T.x.length&&tc<800;ti++){var c2=ll(T.x[ti],T.y[ti]),e=E(c2),n=Nf(c2);if(e<bx0||e>bx1||n<bz0||n>bz1)continue;
  var h=T.h[ti]||GH[T.g[ti]]||8,r=GR[T.g[ti]]||3,th=h*0.45;
  var tk=new THREE.Mesh(new THREE.CylinderGeometry(Math.max(0.12,r*0.09),Math.max(0.15,r*0.11),th,6),new THREE.MeshLambertMaterial({color:0x6e4d30}));tk.position.set(e,th/2,-n);tk.castShadow=true;W.world.add(tk);
  var cn=new THREE.Mesh(new THREE.SphereGeometry(r,8,6),new THREE.MeshLambertMaterial({color:0x5aa54a}));cn.position.set(e,th+r*0.6,-n);cn.scale.y=1.1;cn.castShadow=true;W.world.add(cn);tc++;}
 function addRib(P,color,y,op){if(!P||P.length<2)return;var pos=[],lc=P.map(function(c){return [E(c),Nf(c)];});
  for(var ri=0;ri<lc.length;ri++){var a=lc[Math.max(0,ri-1)],b=lc[Math.min(lc.length-1,ri+1)],dx=b[0]-a[0],dz=b[1]-a[1],L2=Math.hypot(dx,dz)||1,px=-dz/L2*0.6,pz=dx/L2*0.6;
   pos.push(lc[ri][0]+px,y,-(lc[ri][1]+pz),lc[ri][0]-px,y,-(lc[ri][1]-pz));}
  var idx=[];for(var qi=0;qi<lc.length-1;qi++){var a2=2*qi;idx.push(a2,a2+1,a2+2,a2+1,a2+3,a2+2);}
  var g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setIndex(idx);
  var mm=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:color,transparent:true,opacity:op,depthWrite:false}));mm.renderOrder=2;W.world.add(mm);}
 addRib(pts,k==='C'?0x0F9E75:0xE0791E,0.07,0.95);
 var orr=(k==='C')?POV.RS:POV.RC,opts=povFlat(orr);if(opts)addRib(opts,k==='C'?0xE0791E:0x0F9E75,0.05,0.6);}  // both Routes (own solid + the other translucent for comparison); no street network is drawn in the POV scene
function povPt(W,a){var cum=W.cum,pth=W.path,n=cum.length;if(a<=0)return pth[0];if(a>=cum[n-1])return pth[n-1];var i=1;while(i<n&&cum[i]<a)i++;var s=cum[i]-cum[i-1],f=s>0?(a-cum[i-1])/s:0;var p0=pth[i-1],p1=pth[i];return [p0[0]+(p1[0]-p0[0])*f,p0[1]+(p1[1]-p0[1])*f];}
function povFrameOne(k){var W=POV[k];if(!W.scene||!W.path)return;var d=Math.min(W.d||0,W.total),ad=d/(W.total||1)*(W.arc||1);
 var P=povPt(W,ad),x=P[0],z=P[1],T=povPt(W,Math.min(ad+12,W.arc)),tx=T[0]-x,tz=T[1]-z,TL=Math.hypot(tx,tz);
 if(TL>0.05){tx/=TL;tz/=TL;}else if(W.hdg){tx=W.hdg[0];tz=W.hdg[1];}else{tx=1;tz=0;}
 var now=performance.now();
 if(!W.hdg||Math.abs(ad-(W.lastAd===undefined?ad:W.lastAd))>30){W.hdg=[tx,tz];}
 else{var dt=Math.min((now-(W.hT||now))/1000,0.1),aa=1-Math.exp(-dt*3.5),hx=W.hdg[0]+(tx-W.hdg[0])*aa,hz=W.hdg[1]+(tz-W.hdg[1])*aa,HL=Math.hypot(hx,hz)||1;W.hdg=[hx/HL,hz/HL];}
 W.hT=now;W.lastAd=ad;
 W.cam.position.set(x,1.6,-z);W.cam.lookAt(x+W.hdg[0]*14,1.45,-(z+W.hdg[1]*14));
 var sv=povSunVec(D.hours[hidx()]);W.sun.position.set(x+sv.e*300,sv.u*300+40,-z-sv.n*300);W.sun.target.position.set(x,0,-z);W.sun.target.updateMatrixWorld();
 W.hemi.intensity=0.45+0.42*sv.u;W.sun.intensity=0.35+0.8*sv.u;povMark(k,x,z);W.r.render(W.scene,W.cam);}
var POVMK={C:null,S:null,tick:false};  // live position markers: pulsing halo + white-cased heading dart per walker
function povMark(k,x,z){var W=POV[k];if(!W.o||!W.hdg)return;
 POVMK[k]={x:x,z:z,o:W.o.slice(),hdg:W.hdg.slice(),col:k==='C'?'#0F9E75':'#E0791E'};
 povMarkFlush();povMarkTickStart();}
function povMarkFeats(mk){var mLat=111320,mLng=111320*Math.cos(mk.o[1]*Math.PI/180),lng=mk.o[0]+mk.x/mLng,lat=mk.o[1]+mk.z/mLat,hx=mk.hdg[0],hz=mk.hdg[1];
 function pt(fw,sd){var ex=hx*fw-hz*sd,ny=hz*fw+hx*sd;return [lng+ex/mLng,lat+ny/mLat];}
 function dart(sc){return [pt(13*sc,0),pt(-6*sc,5.5*sc),pt(-2.5*sc,0),pt(-6*sc,-5.5*sc),pt(13*sc,0)];}
 var ph=(performance.now()%1300)/1300,R=8+11*ph,ring=[];
 for(var i2=0;i2<=16;i2++){var a2=i2/16*2*Math.PI;ring.push([lng+R*Math.cos(a2)/mLng,lat+R*Math.sin(a2)/mLat]);}
 return [
  {type:'Feature',properties:{c:mk.col,o:0.38*(1-ph)},geometry:{type:'Polygon',coordinates:[ring]}},
  {type:'Feature',properties:{c:'#ffffff',o:0.95},geometry:{type:'Polygon',coordinates:[dart(1.4)]}},
  {type:'Feature',properties:{c:mk.col,o:1.0},geometry:{type:'Polygon',coordinates:[dart(1.0)]}}];}
function povMarkFlush(){if(!map.getSource('povpos')){map.addSource('povpos',{type:'geojson',data:{type:'FeatureCollection',features:[]}});
  map.addLayer({id:'povpos',type:'fill',source:'povpos',paint:{'fill-color':['get','c'],'fill-opacity':['get','o']}});}
 var f=[];['C','S'].forEach(function(k){if(POVMK[k])f=f.concat(povMarkFeats(POVMK[k]));});
 map.getSource('povpos').setData({type:'FeatureCollection',features:f});}
function povMarkTickStart(){if(POVMK.tick)return;POVMK.tick=true;
 (function loop(){if(!POVMK.tick)return;
  if((!POVMK.C&&!POVMK.S)||document.getElementById('pov').style.display==='none'){POVMK.tick=false;return;}
  povMarkFlush();requestAnimationFrame(loop);})();}
function povMarkClear(){POVMK.C=null;POVMK.S=null;POVMK.tick=false;if(map.getSource('povpos'))map.getSource('povpos').setData({type:'FeatureCollection',features:[]});}
function profCursor(){var P=document.getElementById('prof');if(!P||P.style.display==='none'||!window.PROF)return;  // live cursors on the shade-over-distance profile (x = walked metres on the shared axis)
 [['C','pcurC'],['S','pcurS']].forEach(function(a){var W=POV[a[0]],el=document.getElementById(a[1]);if(!el)return;
  if(!W||!W.path||W.d===undefined){el.setAttribute('x1',-10);el.setAttribute('x2',-10);return;}
  var x=PROF.pad+Math.min(W.d,W.total)/PROF.TOT*PROF.gx;el.setAttribute('x1',x.toFixed(1));el.setAttribute('x2',x.toFixed(1));});}
function profCursorClear(){['pcurC','pcurS'].forEach(function(id){var el=document.getElementById(id);if(el){el.setAttribute('x1',-10);el.setAttribute('x2',-10);}});}
function povTxt(k){var W=POV[k];var tx=document.getElementById('povTxt'+k);if(tx)tx.textContent=Math.round(Math.min(W.d||0,W.total||0))+'/'+Math.round(W.total||0)+'m';var sk=document.getElementById('povSeek'+k);if(sk&&W.total)sk.value=(W.d||0)/W.total*1000;profCursor();}
function povFrameBoth(){povFrameOne('C');povFrameOne('S');povTxt('C');povTxt('S');}
function povBuild(){if(!POV.RC||!POV.RS||typeof THREE==='undefined')return;povBuildOne('C','povCvC',POV.RC);povBuildOne('S','povCvS',POV.RS);povFrameBoth();povSyncLabel();}
function povSyncLabel(){var b=document.getElementById('povBoth');if(b)b.textContent=(POV.C.playing||POV.S.playing)?'⏸ 全部暂停':'⏯ 同步播放';}
function povAnimOne(k,ts){var W=POV[k];if(!W.playing)return;var now=(ts!==undefined)?ts:performance.now();var dt=(W.tPrev!==undefined)?Math.min((now-W.tPrev)/1000,0.1):0.0167;W.tPrev=now;W.d=(W.d||0)+1.4*POVSPD*dt;if(W.d>=W.total){W.d=W.total;W.playing=false;document.getElementById('povPlay'+k).textContent='▶';povSyncLabel();}povFrameOne(k);povTxt(k);if(W.playing)requestAnimationFrame(function(t2){povAnimOne(k,t2);});}
function povPlayOne(k){var W=POV[k];if(!W.path)return;W.playing=!W.playing;document.getElementById('povPlay'+k).textContent=W.playing?'⏸':'▶';if(W.playing){if(W.d>=W.total)W.d=0;W.tPrev=undefined;povAnimOne(k);}povSyncLabel();}
document.getElementById('bPov').onclick=function(){if(typeof THREE==='undefined'){tip('three.js 未加载(需联网),行人视角不可用');return;}if(O<0||Dst<0||!POV.RC){tip('请先点出起终点(或用 Demo)算出路径,再开行人视角');return;}document.getElementById('pov').style.display='block';povBuild();tip('行人视角:上=最遮荫 下=最短;各自 ▶ 单独播,或「同步播放」一起走(同速率,短路先到);拖时刻滑杆变太阳');};
document.getElementById('povX').onclick=function(){document.getElementById('pov').style.display='none';POV.C.playing=false;POV.S.playing=false;povMarkClear();profCursorClear();};
document.getElementById('povPlayC').onclick=function(){povPlayOne('C');};
document.getElementById('povPlayS').onclick=function(){povPlayOne('S');};
document.getElementById('povBoth').onclick=function(){var pl=!(POV.C.playing||POV.S.playing);['C','S'].forEach(function(k){var W=POV[k];if(!W.path)return;W.playing=pl;document.getElementById('povPlay'+k).textContent=pl?'⏸':'▶';if(pl){if(W.d>=W.total)W.d=0;W.tPrev=undefined;povAnimOne(k);}});povSyncLabel();};
document.getElementById('povSeekC').oninput=function(){POV.C.playing=false;document.getElementById('povPlayC').textContent='▶';POV.C.d=+this.value/1000*(POV.C.total||1);povFrameOne('C');povTxt('C');povSyncLabel();};
document.getElementById('povSeekS').oninput=function(){POV.S.playing=false;document.getElementById('povPlayS').textContent='▶';POV.S.d=+this.value/1000*(POV.S.total||1);povFrameOne('S');povTxt('S');povSyncLabel();};
let POVSPD=4;  // playback speed in walking-speed multiples (walk = 1.4 m/s); time-based so it is refresh-rate independent
[['povSp4',4],['povSp9',9],['povSp18',18],['povSp36',36]].forEach(function(a){document.getElementById(a[0]).onclick=function(){POVSPD=a[1];['povSp4','povSp9','povSp18','povSp36'].forEach(function(id){document.getElementById(id).classList.remove('on');});this.classList.add('on');};});
(function(){var P=document.getElementById('pov'),h=P.firstElementChild,dx=0,dy=0,drag=false;
 h.addEventListener('mousedown',function(e){if(e.target.id==='povX')return;drag=true;var r=P.getBoundingClientRect();dx=e.clientX-r.left;dy=e.clientY-r.top;P.style.bottom='auto';e.preventDefault();});
 document.addEventListener('mousemove',function(e){if(!drag)return;var pr=P.parentElement.getBoundingClientRect();P.style.left=Math.max(0,e.clientX-pr.left-dx)+'px';P.style.top=Math.max(0,e.clientY-pr.top-dy)+'px';});
 document.addEventListener('mouseup',function(){drag=false;});})();
// === Topology graphs: feasible OD paths under the detour cap (lambda sweep + iterative penalty diversification). Graph1 = path-level (nodes O/D, edges = paths, W1=0.2crowd+0.3(1-shade)+0.3detour+0.2sunrun), Graph2 = street-level (nodes = OD+junctions, W2=0.4crowd+0.6(1-shade)). Wider stroke = lower resistance. All metrics use esf() so the Arcades/Linkways scenario propagates. ===
const TOPOW={crowd:0.20,shade:0.30,detour:0.30,sunrun:0.20,s_crowd:0.40,s_shade:0.60};  // expert-set weights (no canonical 4-factor weights in literature; shade-related dominates at 50%/60%, cf. CoolWalks single-alpha paradigm)
function topoOpen(){return document.getElementById('topo').style.display!=='none';}
function dijPen(s,t,pen,f,cool){ // penalized dijkstra for diversification (cool=false: length cost; cool=true: shade cost at LAM); respects eban
 const dist=new Float64Array(NN).fill(1e18),pe=new Int32Array(NN).fill(-1),pn=new Int32Array(NN).fill(-1);dist[s]=0;const hd=[s],hk=[0];
 function up(i){while(i>0){let p=(i-1)>>1;if(hk[p]<=hk[i])break;[hk[p],hk[i]]=[hk[i],hk[p]];[hd[p],hd[i]]=[hd[i],hd[p]];i=p;}}
 function dn(i){let n=hd.length;for(;;){let l=2*i+1,r=l+1,m=i;if(l<n&&hk[l]<hk[m])m=l;if(r<n&&hk[r]<hk[m])m=r;if(m===i)break;[hk[m],hk[i]]=[hk[i],hk[m]];[hd[m],hd[i]]=[hd[i],hd[m]];i=m;}}
 while(hd.length){let d=hk[0],x=hd[0],L=hd.length-1;hd[0]=hd[L];hk[0]=hk[L];hd.pop();hk.pop();if(hd.length)dn(0);if(d>dist[x])continue;if(x===t)break;
  for(let p=off[x];p<off[x+1];p++){let y=aN[p],ei=aE[p];if(eban(ei))continue;let c=(cool?EL[ei]*((1-esf(ei)/100)+LAM):EL[ei])*(pen.has(ei)?f:1);let n2=d+c;if(n2<dist[y]){dist[y]=n2;pe[y]=ei;pn[y]=x;hd.push(y);hk.push(n2);up(hd.length-1);}}}
 if(dist[t]>=1e18)return null;let coords=[],eis=[],len=0,x=t;
 while(pn[x]>=0){let ei=pe[x];eis.push(ei);len+=EL[ei];coords.push(edgeLL(D.egeom[ei]));x=pn[x];}
 eis.reverse();return {coords:coords,len:len,eis:eis};}
function topoGen(){ // candidate feasible paths within detour cap DETOUR
 let s=dij(O,Dst,'short');if(!s)return null;let cap=DETOUR*s.len+1,cands=[];
 function sim(a,b){let A=new Set(a.eis),i2=0;for(let k=0;k<b.eis.length;k++)if(A.has(b.eis[k]))i2++;return i2/(a.eis.length+b.eis.length-i2);}
 function add(r){if(!r||!r.eis||!r.eis.length||r.len>cap)return;for(let q=0;q<cands.length;q++)if(sim(cands[q],r)>0.9)return;cands.push(r);}
 add(s);
 let r0=dij(O,Dst,'cool',0),spec=(r0&&s.len>0)?(r0.len/s.len):1;  // pure most-shaded (lam=0) = feasible-spectrum upper bound: caps above it stop binding
 add(r0);add(coolDetour(O,Dst,s));                                 // include the app's displayed Coolest (adaptive lam finds equal-length substitutes the fixed sweep misses)
 [0.03,0.08,0.15,0.35,0.7,1.4].forEach(function(l){add(dij(O,Dst,'cool',l));});
 for(let rd=0;rd<3&&cands.length<8;rd++){let pen=new Set();cands.forEach(function(r){r.eis.forEach(function(e2){pen.add(e2);});});add(dijPen(O,Dst,pen,1.7,false));add(dijPen(O,Dst,pen,1.7,true));}
 return {s:s,cands:cands.slice(0,8),spec:spec};}
function topoMet(g){ // per-path metrics + composite resistance W1 (crowding uses the current Detour-limit flow field)
 let F=flowCur(),ms=g.cands.map(function(r){let cr=0,shm=0,run=0,mx=0;
  for(let k=0;k<r.eis.length;k++){let ei=r.eis[k];cr+=F[ei];let sv=esf(ei);shm+=EL[ei]*sv/100;if(sv<50){run+=EL[ei];if(run>mx)mx=run;}else run=0;}
  return {r:r,len:r.len,shade:shm/(r.len||1),crowd:cr/(r.eis.length||1),det:r.len/(g.s.len||1),sunrun:mx/(r.len||1)};});
 let cmax=1;ms.forEach(function(m){if(m.crowd>cmax)cmax=m.crowd;});
 ms.forEach(function(m){m.crowdN=m.crowd/cmax;m.detN=Math.max(0,Math.min(1,(m.det-1)/Math.max(DETOUR-1,0.05)));
  m.W1=TOPOW.crowd*m.crowdN+TOPOW.shade*(1-m.shade)+TOPOW.detour*m.detN+TOPOW.sunrun*m.sunrun;});
 ms.sort(function(a,b){return a.W1-b.W1;});return ms;}
function topoStreets(g){ // contract the union of path edges into street segments between junctions
 let ed=new Set();g.cands.forEach(function(r){r.eis.forEach(function(e2){ed.add(e2);});});
 let dg=new Map(),ad2=new Map();
 ed.forEach(function(ei){[U[ei],V[ei]].forEach(function(n){dg.set(n,(dg.get(n)||0)+1);if(!ad2.has(n))ad2.set(n,[]);ad2.get(n).push(ei);});});
 let J=new Set([O,Dst]);dg.forEach(function(d2,n){if(d2!==2)J.add(n);});
 let used=new Set(),st2=[];
 J.forEach(function(j){(ad2.get(j)||[]).forEach(function(e0){if(used.has(e0))return;
  let node=j,ei=e0,chain=[],nds=[j];
  for(;;){used.add(ei);chain.push(ei);let nx=(U[ei]===node)?V[ei]:U[ei];nds.push(nx);if(J.has(nx))break;
   let cn=(ad2.get(nx)||[]).filter(function(e2){return !used.has(e2);});if(!cn.length)break;node=nx;ei=cn[0];}
  let len=0,shm=0,cr=0,F=flowCur();chain.forEach(function(e2){len+=EL[e2];shm+=EL[e2]*esf(e2)/100;cr+=F[e2];});
  st2.push({nds:nds,len:len,shade:shm/(len||1),crowd:cr/(chain.length||1)});});});
 let cmax=1;st2.forEach(function(s2){if(s2.crowd>cmax)cmax=s2.crowd;});
 st2.forEach(function(s2){s2.W2=TOPOW.s_crowd*(s2.crowd/cmax)+TOPOW.s_shade*(1-s2.shade);});
 return {streets:st2,J:J};}
function shCol(sh){let a=[198,152,74],b=[15,110,86],c=a.map(function(v,i){return Math.round(v+(b[i]-v)*Math.max(0,Math.min(1,sh)));});return 'rgb('+c.join(',')+')';}
let TOPO={ms:null,stJ:null,sel:-1,base:null};
function g1svg(ms,sel){let Wd=486,Hg=Math.max(150,46+ms.length*24),ox=42,dx2=Wd-42,cy=Hg/2;
 let mn=1e9,mx=-1e9;ms.forEach(function(m){if(m.W1<mn)mn=m.W1;if(m.W1>mx)mx=m.W1;});let sp=(mx-mn)||1;
 let sv='<svg viewBox="0 0 '+Wd+' '+Hg+'" width="100%" style="display:block">';
 ms.forEach(function(m,i){let off2=(i-(ms.length-1)/2)*((Hg-52)/Math.max(ms.length-1,1))*2;
  let wpx=(1.2+6.5*(1-(m.W1-mn)/sp)).toFixed(1),col=shCol(m.shade),op=(sel<0||sel===i)?0.92:0.25;
  sv+='<path d="M '+ox+' '+cy+' Q '+(Wd/2)+' '+(cy+off2)+' '+dx2+' '+cy+'" fill="none" stroke="'+col+'" stroke-width="'+wpx+'" opacity="'+op+'" style="cursor:pointer" onclick="topoSel('+i+')"><title>P'+(i+1)+' · '+Math.round(m.len)+'m · 遮荫 '+Math.round(m.shade*100)+'% · 拥挤 '+Math.round(m.crowd)+' · 绕行 '+m.det.toFixed(2)+'x · 最长连续日晒 '+Math.round(m.sunrun*m.len)+'m · W='+m.W1.toFixed(3)+'</title></path>';
  sv+='<text x="'+(Wd/2)+'" y="'+(cy+off2/2-3)+'" font-size="8" text-anchor="middle" fill="#444" style="pointer-events:none">P'+(i+1)+' '+m.W1.toFixed(2)+'</text>';});
 sv+='<circle cx="'+ox+'" cy="'+cy+'" r="8" fill="#0F6E56"/><text x="'+ox+'" y="'+(cy+3.5)+'" font-size="10" fill="#fff" text-anchor="middle" font-weight="600">O</text>';
 sv+='<circle cx="'+dx2+'" cy="'+cy+'" r="8" fill="#c0392b"/><text x="'+dx2+'" y="'+(cy+3.5)+'" font-size="10" fill="#fff" text-anchor="middle" font-weight="600">D</text></svg>';
 return sv;}
function g2svg(stJ){let st2=stJ.streets,J=stJ.J;if(!st2.length)return '';
 let xs=[],ys=[];st2.forEach(function(s2){s2.nds.forEach(function(n){xs.push(D.nx[n]);ys.push(D.ny[n]);});});
 let x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs),y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys);
 let Wd=486,pad2=16,sc=Math.min((Wd-2*pad2)/Math.max(x1-x0,1),300/Math.max(y1-y0,1));
 let Hg=Math.max(120,Math.ceil((y1-y0)*sc)+2*pad2);
 function px(x){return pad2+(x-x0)*sc;} function py(y){return Hg-pad2-(y-y0)*sc;}
 let mn=1e9,mx=-1e9;st2.forEach(function(s2){if(s2.W2<mn)mn=s2.W2;if(s2.W2>mx)mx=s2.W2;});let sp=(mx-mn)||1;
 let sv='<svg viewBox="0 0 '+Wd+' '+Hg+'" width="100%" style="display:block">';
 st2.forEach(function(s2){let d='';s2.nds.forEach(function(n,i){d+=(i?'L':'M')+px(D.nx[n]).toFixed(1)+' '+py(D.ny[n]).toFixed(1);});
  let wpx=(1+5.5*(1-(s2.W2-mn)/sp)).toFixed(1);
  sv+='<path d="'+d+'" fill="none" stroke="'+shCol(s2.shade)+'" stroke-width="'+wpx+'" stroke-linecap="round" opacity="0.9"><title>街道 '+Math.round(s2.len)+'m · 遮荫 '+Math.round(s2.shade*100)+'% · 拥挤 '+Math.round(s2.crowd)+' · W='+s2.W2.toFixed(3)+'</title></path>';});
 J.forEach(function(n){if(n===O||n===Dst)return;sv+='<circle cx="'+px(D.nx[n]).toFixed(1)+'" cy="'+py(D.ny[n]).toFixed(1)+'" r="2.2" fill="#555" opacity="0.85"/>';});
 [[O,'#0F6E56','O'],[Dst,'#c0392b','D']].forEach(function(a2){sv+='<circle cx="'+px(D.nx[a2[0]]).toFixed(1)+'" cy="'+py(D.ny[a2[0]]).toFixed(1)+'" r="7" fill="'+a2[1]+'"/><text x="'+px(D.nx[a2[0]]).toFixed(1)+'" y="'+(py(D.ny[a2[0]])+3.2).toFixed(1)+'" font-size="9" fill="#fff" text-anchor="middle" font-weight="600">'+a2[2]+'</text>';});
 sv+='</svg>';return sv;}
function topoSummary(ms){return {W:ms[0].W1,shade:ms[0].shade,len:ms[0].len,K:ms.length};}
window.topoSel=function(i){TOPO.sel=(TOPO.sel===i)?-1:i;topoRender();
 if(!map.getSource('rtopo')){map.addSource('rtopo',{type:'geojson',data:{type:'FeatureCollection',features:[]}});map.addLayer({id:'rtopo',type:'line',source:'rtopo',layout:{'line-cap':'round'},paint:{'line-color':'#8e44ad','line-width':4.5,'line-dasharray':[1.6,1.1],'line-opacity':0.95}});}
 let m=(TOPO.sel>=0&&TOPO.ms)?TOPO.ms[TOPO.sel]:null;
 map.getSource('rtopo').setData(m?{type:'Feature',geometry:{type:'MultiLineString',coordinates:m.r.coords},properties:{}}:{type:'FeatureCollection',features:[]});};
function topoRender(){let ms=TOPO.ms;if(!ms||!ms.length)return;let el=document.getElementById('topoBody');
 let scn=(SCN.arc&&SCN.lkw)?'全设施':((!SCN.arc&&!SCN.lkw)?'已移除 骑楼+连廊':(!SCN.arc?'已移除 骑楼':'已移除 连廊'));
 let h='<div class="sec" style="margin:2px 0 4px">场景:<b>'+scn+'</b> · 绕行率上限 '+DETOUR+'× · <span title="纯最遮荫路(λ=0)的绕行率 = 可行谱系上界;Detour limit 高于它后再调大,图形不会变化">谱系上界 '+(TOPO.spec||1).toFixed(2)+'×</span> · 可行路径 '+ms.length+' 条(勾/取消左侧 Arcades / Covered linkways 图层对比)</div>';
 if(TOPO.base&&!(SCN.arc&&SCN.lkw)){let b=TOPO.base,c=topoSummary(ms);
  h+='<div class="card" style="background:#f7f4ef;font-size:11px;padding:6px 8px">对比全设施基线:最优路阻力 W '+b.W.toFixed(3)+' → <b>'+c.W.toFixed(3)+'</b> ('+(c.W>b.W?'+':'')+(c.W-b.W).toFixed(3)+') · 遮荫 '+Math.round(b.shade*100)+'% → <b>'+Math.round(c.shade*100)+'%</b> · 可行路径 '+b.K+' → '+c.K+' 条</div>';}
 h+='<div class="sec" style="margin:4px 0 1px">图1 · 路径级:点=OD,边=可行路径(互不交叉;越宽阻力越小;W=0.2拥挤+0.3(1−遮荫)+0.3绕行+0.2连续日晒)</div>'+g1svg(ms,TOPO.sel);
 h+='<div class="sec" style="margin:6px 0 1px">图2 · 街道级:点=OD+道路交叉口,边=街道(越宽阻力越小;W=0.4拥挤+0.6(1−遮荫))</div>'+g2svg(TOPO.stJ);
 h+='<div style="font-size:10px;color:#555;margin-top:3px">';
 ms.forEach(function(m,i){h+='<div style="padding:1px 2px;cursor:pointer;'+(TOPO.sel===i?'background:#eee9df;border-radius:4px;':'')+'" onclick="topoSel('+i+')"><b style="color:'+shCol(m.shade)+'">P'+(i+1)+'</b> '+Math.round(m.len)+'m · 遮荫'+Math.round(m.shade*100)+'% · 拥挤'+Math.round(m.crowd)+' · 绕行'+m.det.toFixed(2)+'× · 连续日晒'+Math.round(m.sunrun*100)+'% · <b>W='+m.W1.toFixed(3)+'</b>'+(m.det<=1.0001?' <span style="color:#985316">(最短)</span>':'')+'</div>';});
 h+='</div><div style="font-size:9px;color:#888;margin-top:3px">候选=λ扫描+迭代惩罚多样化(≤8 条代表性可行路径,穷举为指数级不可行);拥挤度=沿路人流均值(按候选集内最大值归一);连续日晒=最长连续日晒段长度占比(边遮荫<50% 记日晒);点击弧线或行可在地图高亮(紫虚线)。权重为专家设定值(文献无公认四因子权重;遮荫相关合计 50%/60%,参照 CoolWalks 单参数避晒范式)。</div>';
 el.innerHTML=h;}
function topoBuild(quiet){if(O<0||Dst<0)return;
 if(!quiet)tip('拓扑计算中…');
 setTimeout(function(){let g=topoGen();if(!g){tip('两点不在同一连通区域');return;}
  TOPO.ms=topoMet(g);TOPO.stJ=topoStreets(g);TOPO.sel=-1;TOPO.spec=g.spec;
  if(SCN.arc&&SCN.lkw){TOPO.base=topoSummary(TOPO.ms);}
  else{let sv={arc:SCN.arc,lkw:SCN.lkw};SCN.arc=true;SCN.lkw=true;let gb=topoGen();TOPO.base=gb?topoSummary(topoMet(gb)):null;SCN.arc=sv.arc;SCN.lkw=sv.lkw;}
  topoRender();if(!quiet)tip(0);},20);}
document.getElementById('bTopo').onclick=function(){let T2=document.getElementById('topo');
 if(T2.style.display!=='none'){T2.style.display='none';this.classList.remove('on');TOPO.sel=-1;if(map.getSource('rtopo'))map.getSource('rtopo').setData({type:'FeatureCollection',features:[]});return;}
 if(O<0||Dst<0){tip('请先点出起终点(或用 Demo)算出路径,再开拓扑图');return;}
 T2.style.display='block';this.classList.add('on');topoBuild();};
document.getElementById('topoX').onclick=function(){document.getElementById('topo').style.display='none';document.getElementById('bTopo').classList.remove('on');TOPO.sel=-1;if(map.getSource('rtopo'))map.getSource('rtopo').setData({type:'FeatureCollection',features:[]});};
(function(){var P=document.getElementById('topo'),h=document.getElementById('topoHdr'),dx=0,dy=0,drag=false;
 h.addEventListener('mousedown',function(e){if(e.target.id==='topoX')return;drag=true;var r=P.getBoundingClientRect();dx=e.clientX-r.left;dy=e.clientY-r.top;e.preventDefault();});
 document.addEventListener('mousemove',function(e){if(!drag)return;var pr=P.parentElement.getBoundingClientRect();P.style.right='auto';P.style.left=Math.max(0,e.clientX-pr.left-dx)+'px';P.style.top=Math.max(0,e.clientY-pr.top-dy)+'px';});
 document.addEventListener('mouseup',function(){drag=false;});})();
function demoPair(){var dist,hd,hk;
 function up(i){while(i>0){var p=(i-1)>>1;if(hk[p]<=hk[i])break;var t=hk[p];hk[p]=hk[i];hk[i]=t;t=hd[p];hd[p]=hd[i];hd[i]=t;i=p;}}
 function dn(i){var n=hd.length;for(;;){var l=2*i+1,r=l+1,m=i;if(l<n&&hk[l]<hk[m])m=l;if(r<n&&hk[r]<hk[m])m=r;if(m===i)break;var t=hk[m];hk[m]=hk[i];hk[i]=t;t=hd[m];hd[m]=hd[i];hd[i]=t;i=m;}}
 for(var att=0;att<25;att++){var s=(Math.random()*NN)|0;dist=new Float64Array(NN).fill(1e18);dist[s]=0;hd=[s];hk=[0];var cand=[];
  while(hd.length){var d=hk[0],x=hd[0],L=hd.length-1;hd[0]=hd[L];hk[0]=hk[L];hd.pop();hk.pop();if(hd.length)dn(0);if(d>dist[x])continue;if(d>1000)break;if(d>=500)cand.push(x);for(var p=off[x];p<off[x+1];p++){var y=aN[p],ei=aE[p],n2=d+EL[ei];if(n2<dist[y]){dist[y]=n2;hd.push(y);hk.push(n2);up(hd.length-1);}}}
  if(cand.length)return [s,cand[(Math.random()*cand.length)|0]];}
 return null;}
document.getElementById('bDemo').onclick=()=>{var pr=demoPair();if(!pr){tip('没找到 500m–1km 的示例,再点一次');return;}O=pr[0];Dst=pr[1];let a=ll(D.nx[O],D.ny[O]),b=ll(D.nx[Dst],D.ny[Dst]);OA={node:O,clng:a[0],clat:a[1],flng:a[0],flat:a[1],acc:0};DA={node:Dst,clng:b[0],clat:b[1],flng:b[0],flat:b[1],acc:0};document.getElementById('oTxt').textContent='起点:示例';document.getElementById('dTxt').textContent='终点:示例';odUpd();FITRT=true;route();};
document.getElementById('bSwap').onclick=()=>{if(O<0||Dst<0)return;let t=O;O=Dst;Dst=t;let ta=OA;OA=DA;DA=ta;odUpd();document.getElementById('oTxt').textContent='起点:已选';document.getElementById('dTxt').textContent='终点:已选';route();};
let demoI={};
function snapPt(lng,lat){let mx=lng/180*Math.PI*Rm-OX,my=Rm*Math.log(Math.tan(Math.PI/4+lat/180*Math.PI/2))-OY;let se=snapEdge(mx,my);if(se)return {node:se.node,clng:lng,clat:lat,flng:se.flng,flat:se.flat,acc:se.acc};let n=nearest(mx,my);if(n<0)return null;let nl=ll(D.nx[n],D.ny[n]);return {node:n,clng:lng,clat:lat,flng:nl[0],flat:nl[1],acc:haversine([lng,lat],nl)};}
function setDemo(scene,nm){let arr=(D.demoOD||{})[scene];if(!arr||!arr.length){tip('该场景暂无示例');return;}let i=(demoI[scene]||0)%arr.length;demoI[scene]=i+1;let p=arr[i];let oa=snapPt(p[0],p[1]),da=snapPt(p[2],p[3]);if(!oa||!da)return;O=oa.node;OA=oa;Dst=da.node;DA=da;document.getElementById('oTxt').textContent='起点:'+nm+' ('+(i+1)+'/'+arr.length+')';document.getElementById('dTxt').textContent='终点:'+nm.split('→')[1];odUpd();FITRT=true;route();}
document.getElementById('dmHt').onclick=()=>setDemo('hdb_transit','组屋→公交');
document.getElementById('dmOt').onclick=()=>setDemo('office_transit','办公→公交');
document.getElementById('dmOf').onclick=()=>setDemo('office_food','办公→餐饮');
document.getElementById('dmHm').onclick=()=>setDemo('hdb_mart','组屋→商超');
function svList(){try{return JSON.parse(localStorage.getItem('svDemo')||'[]')}catch(e){return [];}}
let importedOD=[];
function poolList(){return svList().concat(((D.demoOD||{}).my_saved)||[]).concat(importedOD);}
function svUpd(){var n=document.getElementById('svN');if(n)n.textContent=poolList().length;}
document.getElementById('dmSave').onclick=()=>{if(O<0||Dst<0||!OA||!DA){tip('请先点出起讫点(或选个示例)再保存');return;}var a=svList();a.push([+OA.clng.toFixed(6),+OA.clat.toFixed(6),+DA.clng.toFixed(6),+DA.clat.toFixed(6)]);localStorage.setItem('svDemo',JSON.stringify(a));svUpd();tip('已存第 '+a.length+' 条到收藏(可“我的收藏”回看 / “导出”发我永久内置)');};
let myI=0;
document.getElementById('dmMine').onclick=()=>{var arr=poolList();if(!arr.length){tip('轮播池为空,先“保存当前”或“导入”json');return;}var i=myI%arr.length;myI=i+1;var p=arr[i];var oa=snapPt(p[0],p[1]),da=snapPt(p[2],p[3]);if(!oa||!da){tip('该点不在路网附近');return;}O=oa.node;OA=oa;Dst=da.node;DA=da;document.getElementById('oTxt').textContent='起点:轮播 ('+(i+1)+'/'+arr.length+')';document.getElementById('dTxt').textContent='终点:轮播';odUpd();FITRT=true;route();};
document.getElementById('dmExport').onclick=()=>{var a=svList();if(!a.length){tip('收藏夹为空,无可导出');return;}var blob=new Blob([JSON.stringify(a,null,1)],{type:'application/json'});var u=URL.createObjectURL(blob);var el=document.createElement('a');el.href=u;el.download='demo_saved.json';document.body.appendChild(el);el.click();el.remove();URL.revokeObjectURL(u);tip('已导出 '+a.length+' 条 → demo_saved.json(发我合并进后台)');};
document.getElementById('dmClear').onclick=()=>{var a=svList();if(!a.length){tip('浏览器收藏本来就是空的');return;}if(confirm('清空浏览器收藏的 '+a.length+' 条?(内置 Demo 与本次导入不受影响)')){localStorage.removeItem('svDemo');myI=0;svUpd();tip('已清空浏览器收藏');}};
document.getElementById('dmImport').onclick=()=>document.getElementById('dmFile').click();
document.getElementById('dmFile').onchange=(e)=>{var f=e.target.files[0];if(!f)return;var r=new FileReader();r.onload=()=>{try{var d=JSON.parse(r.result);var arr=Array.isArray(d)?d:((d&&typeof d==='object')?Object.keys(d).reduce(function(x,k){return x.concat(Array.isArray(d[k])?d[k]:[]);},[]):null);if(!arr){tip('导入失败:需 OD 数组或场景对象 json');return;}arr=arr.filter(function(p){return Array.isArray(p)&&p.length>=4&&typeof p[0]==='number'&&typeof p[3]==='number';});if(!arr.length){tip('导入失败:没有有效 OD');return;}importedOD=arr;myI=svList().length+((((D.demoOD||{}).my_saved)||[]).length);svUpd();tip('已导入 '+arr.length+' 条(临时,不入收藏)');document.getElementById('dmMine').click();}catch(ex){tip('JSON 解析失败');}e.target.value='';};r.readAsText(f);};
svUpd();
function updLeg(){let l=document.getElementById('leg');if(colorMode==='shade'){let _e=[0,13,25,38,50,63,75,88,100],_bar='',_tk='';for(let b=0;b<8;b++)_bar+='<span style="width:15px;height:10px;display:inline-block;background:'+SHC[b]+'"></span>';for(let _i=0;_i<_e.length;_i++)_tk+='<span>'+_e[_i]+(_i===_e.length-1?'%':'')+'</span>';l.innerHTML='<b style="font-weight:500">14:00 路段遮荫率(蓝=遮荫,红=日晒)</b><div style="display:flex;margin-top:3px">'+_bar+'</div><div style="display:flex;justify-content:space-between;width:120px;font-size:9px;color:#555;margin-top:1px">'+_tk+'</div>';}else if(colorMode==='flow')l.innerHTML='<b style="font-weight:500">'+(facMode==='orig'?'步行人流量(原始网络)':'步行人流量(避热 τ='+DETOUR+'×)')+'</b><br><span class="sw" style="background:#7a3d18"></span>高 &nbsp;<span class="sw" style="background:#f4cf86"></span>低';else if(colorMode==='plain')l.innerHTML='<b style="font-weight:500">路网(无色)</b><br><span class="sw" style="background:#8c8c8c"></span>路网线';else l.innerHTML='<b style="font-weight:500">路网已隐藏</b>';l.innerHTML+='<br><span class="sw" style="background:#0F9E75"></span>最遮荫路 &nbsp;<span class="sw" style="background:#E0791E"></span>最短路';if(colorMode!=='none'){if(facMode==='hi')l.innerHTML+='<br><span class="sw" style="background:#d6336c"></span>骑楼 &nbsp;<span class="sw" style="background:#2b6cb0"></span>连廊 &nbsp;<span class="sw" style="background:#12a5b0"></span>过街天桥 &nbsp;<span class="sw" style="background:#e07b2a"></span>楼宇连廊 &nbsp;<span class="sw" style="background:#9b59b6"></span>接入桥';else if(facMode==='hiM')l.innerHTML+='<br><span class="sw" style="background:#d6336c"></span>人工遮荫设施(连廊/骑楼)';}}
;(function(){function foldPanel(id){var p=document.getElementById(id);if(!p)return;var btn=document.getElementById(id+'F');if(!btn)return;var folded=p.getAttribute('data-fold')==='1';var kids=Array.from(p.children);kids.forEach(function(c,i){if(i>0)c.style.display=folded?'':'none';});p.setAttribute('data-fold',folded?'0':'1');btn.textContent=folded?'▾':'▸';}['prof','pov','topo'].forEach(function(id){var b=document.getElementById(id+'F');if(b)b.onclick=function(e){e.stopPropagation();foldPanel(id);};});})();
</script></body></html>'''
HTML=HTML.replace('__DATA__',dj)
open(f"{WEB}\\nav_app.html","w",encoding="utf-8").write(HTML)
print(f"已写 nav_app.html (MapLibre, {len(HTML)/1e6:.1f} MB) | {time.time()-t0:.0f}s",flush=True)
