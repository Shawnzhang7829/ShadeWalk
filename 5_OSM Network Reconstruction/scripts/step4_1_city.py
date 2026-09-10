# -*- coding: utf-8 -*-
"""Step4-1 city-wide version: shade-facility polygons (arcade + covered_linkway) -> Voronoi medial-axis centreline network. Whole island, no clipping.
Parameters identical to the validated GLSZ04 run (STEP=1.0, PRUNE=4.0, SIMP=0.5). pyenv."""
import numpy as np, geopandas as gpd, time, math
from scipy.spatial import Voronoi
from shapely.geometry import LineString
from shapely.ops import unary_union, linemerge
from shapely import prepared, force_2d
import networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ARC=r"D:\Claude\SVI_FFW\Shp\SG\step2_arcade_sg.gpkg"                       # arcade polygons (used to decide the offset direction)
RUNS=r"D:\Claude\SVI_FFW\output\step2_projection\step2b_runs_sg.gpkg"      # arcade facade edges (new centreline source)
LKW=r"D:\Claude\SVI_FFW\Shp\SG\covered_linkway_SG_island_tv_pednet_bridged.gpkg"
OUT=r"D:\Claude\SVI_FFW\output\step4_network"
STEP=1.0; PRUNE=4.0; SIMP=3.0; ARC_OFFSET=1.0   # arcade centreline = runs facade edge offset 1 m into the arcade (along the long axis; fixes the Voronoi short-axis problem); linkway keeps Voronoi

def load(p):
    g=gpd.read_file(p)
    try: g=g.to_crs(3414)
    except Exception: g=g.set_crs(3414,allow_override=True)
    g['geometry']=g.geometry.apply(force_2d)
    return g[~g.geometry.is_empty]
arc=load(ARC); lkw=load(LKW); runs=load(RUNS)
print(f"City-wide | arcade polygons {len(arc)} | runs facade edges {len(runs)} | linkway {len(lkw)}",flush=True)

def densify_ring(ring,step):
    n=max(int(ring.length/step),8); return [ring.interpolate(i/n,normalized=True).coords[0][:2] for i in range(n)]
def medial(part,step):
    pts=[]
    for ring in [part.exterior]+list(part.interiors): pts+=densify_ring(ring,step)
    pts=np.array(pts)
    if len(pts)<4: return []
    try: vor=Voronoi(pts)
    except Exception: return []
    pg=prepared.prep(part); segs=[]
    for i,j in vor.ridge_vertices:
        if i<0 or j<0: continue
        s=LineString([vor.vertices[i],vor.vertices[j]])
        if pg.contains(s): segs.append(s)
    return segs
def prune_graph(segs,prune):
    G=nx.Graph()
    for s in segs:
        a,b=s.coords[0],s.coords[-1]; na=(round(a[0],2),round(a[1],2)); nb=(round(b[0],2),round(b[1],2))
        if na!=nb: G.add_edge(na,nb,length=s.length)
    changed=True
    while changed:
        changed=False
        for leaf in [n for n in G.nodes if G.degree(n)==1]:
            if leaf not in G or G.degree(leaf)!=1: continue
            path=[leaf]; length=0.0; prev=leaf; cur=next(iter(G[leaf]))
            while True:
                length+=G[prev][cur]['length']; path.append(cur)
                if G.degree(cur)!=2: break
                nxt=[x for x in G[cur] if x!=prev][0]; prev,cur=cur,nxt
            if G.degree(path[-1])>=3 and length<prune:
                for k in range(len(path)-1):
                    if G.has_edge(path[k],path[k+1]): G.remove_edge(path[k],path[k+1])
                changed=True
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n)==0])
    return [LineString([u,v]) for u,v in G.edges()]
def _ang(p,a,b):
    """Angle (deg) at point p between vectors p->a and p->b: 180 = collinear straight continuation, 90 = perpendicular sharp turn."""
    v1=(a[0]-p[0],a[1]-p[1]); v2=(b[0]-p[0],b[1]-p[1]); n1=math.hypot(*v1); n2=math.hypot(*v2)
    if n1<1e-9 or n2<1e-9: return 0.0
    c=max(-1.0,min(1.0,(v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2))); return math.degrees(math.acos(c))
def main_skeleton(segs,ang=120.0):
    """Deflection-angle main skeleton: (1) break cycles with a maximum spanning tree (long edges kept by length); (2) remove sharp-turn spur ends by deflection angle -- trace from a degree-1
    end along the degree-2 chain to the junction J; if the maximum angle between the chain and the other segments at J is < ang (it can only join by a sharp turn, e.g. a perpendicular
    rib) delete the whole end chain; if >= ang (a straight continuation exists, e.g. main-axis end / long branch arm) keep it. Iterate until stable. Same-facility connectivity is guaranteed by the tree structure."""
    G=nx.Graph()
    for s in segs:
        a,b=s.coords[0][:2],s.coords[-1][:2]; na=(round(a[0],2),round(a[1],2)); nb=(round(b[0],2),round(b[1],2))
        if na!=nb: G.add_edge(na,nb)
    if G.number_of_edges()==0: return []
    if G.number_of_edges()>=G.number_of_nodes():
        for u,v in G.edges(): G[u][v]['w']=math.hypot(u[0]-v[0],u[1]-v[1])
        G=nx.maximum_spanning_tree(G,weight='w')
    changed=True
    while changed:
        changed=False
        for leaf in [n for n in list(G.nodes) if G.degree(n)==1]:
            if leaf not in G or G.degree(leaf)!=1: continue
            prev=leaf; cur=next(iter(G[leaf])); ce=[(leaf,cur)]
            while G.degree(cur)==2:
                nx_=[x for x in G[cur] if x!=prev]
                if not nx_: break
                ce.append((cur,nx_[0])); prev,cur=cur,nx_[0]
            J=cur; pin=prev; others=[nb for nb in G[J] if nb!=pin]
            if not others: continue   # J is itself an endpoint (whole isolated line), keep
            if max(_ang(J,pin,o) for o in others)<ang:   # can only join by a sharp turn -> delete the whole end chain
                for e in ce:
                    if G.has_edge(*e): G.remove_edge(*e)
                changed=True
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n)==0])
    return [LineString([u,v]) for u,v in G.edges()]
def mrr_centerline(poly):
    """Long-axis centreline of the minimum rotated rectangle: joins the midpoints of the two short sides (through the centre along the long axis), clipped to poly. Fixes short, wide arcades that Voronoi cuts along the short axis."""
    try: mrr=poly.minimum_rotated_rectangle
    except Exception: return None
    if mrr.geom_type!='Polygon': return None
    xy=list(mrr.exterior.coords)[:4]
    edges=[(xy[k],xy[(k+1)%4]) for k in range(4)]
    lens=[math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in edges]
    li=int(np.argmax(lens)); s1=edges[(li+1)%4]; s2=edges[(li+3)%4]
    m1=((s1[0][0]+s1[1][0])/2,(s1[0][1]+s1[1][1])/2); m2=((s2[0][0]+s2[1][0])/2,(s2[0][1]+s2[1][1])/2)
    cl=LineString([m1,m2]).intersection(poly.buffer(0))
    if cl.is_empty: return None
    if cl.geom_type=='MultiLineString': cl=max(cl.geoms,key=lambda gg:gg.length)
    return cl if (cl.geom_type=='LineString' and cl.length>0.5) else None
def _axis_ang(poly):
    try: mrr=poly.minimum_rotated_rectangle
    except Exception: return None
    if mrr.geom_type!='Polygon': return None
    xy=list(mrr.exterior.coords)[:4]
    edges=[(xy[k],xy[(k+1)%4]) for k in range(4)]
    lens=[math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in edges]
    li=int(np.argmax(lens)); a,b=edges[li]
    return math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))%180
def _line_ang(geom):
    pts=[]; gs=geom.geoms if geom.geom_type=='MultiLineString' else [geom]
    for gg in gs: pts+=[c[:2] for c in gg.coords]
    pts=np.array(pts)
    if len(pts)<2: return None
    c=pts-pts.mean(0)
    try: _,_,vt=np.linalg.svd(c); return math.degrees(math.atan2(vt[0][1],vt[0][0]))%180
    except Exception: return None
def centerlines(gdf,tag,mrr_fb=False):
    """mrr_fb=True (arcade): if the Voronoi centreline direction deviates from the long axis by > 30 deg or none was generated -> fall back to the MRR long-axis centreline (fixes short, wide arcades picking the short axis / being missed); linkway keeps pure Voronoi."""
    out=[]; t=time.time(); N=len(gdf); n_mrr=0
    for i,geom in enumerate(gdf.geometry):
        parts=geom.geoms if geom.geom_type=='MultiPolygon' else [geom]
        plines=[]
        for part in parts:
            if part.geom_type!='Polygon' or part.area<1: continue
            segs=medial(part,STEP); vm=None
            if segs:
                segs=prune_graph(segs,PRUNE)
                if segs: segs=main_skeleton(segs)
                if segs: vm=linemerge(unary_union(segs))
            if mrr_fb:
                use_mrr=(vm is None); axang=_axis_ang(part)
                if vm is not None and axang is not None:
                    vang=_line_ang(vm)
                    if vang is not None:
                        d=abs(axang-vang); d=min(d,180-d)
                        if d>30: use_mrr=True            # Voronoi direction deviates from the long axis -> MRR long-axis fallback
                if use_mrr:
                    ml=mrr_centerline(part)
                    if ml is not None: plines.append(ml); n_mrr+=1; continue
            if vm is not None: plines.append(vm)
        if plines:
            m=unary_union(plines).simplify(SIMP); out.append((m, f"{tag}_{i}"))
        if (i+1)%2000==0: print(f"  {tag} {i+1}/{N} (MRR fallback {n_mrr}) ({time.time()-t:.0f}s)",flush=True)
    print(f"  {tag} done {N} (MRR fallback {n_mrr}) ({time.time()-t:.0f}s)",flush=True)
    return out

# arcade centreline: runs (facade edges) offset 1 m into the arcade -- follows the facade long axis, never picks the short axis, follows curves too (replaces Voronoi/MRR)
from shapely import STRtree
arc_geoms=list(arc.geometry.values); arc_tree=STRtree(arc_geoms)
def round_corners(line, r=0.8, n=6):
    """Smooth vertex joins: each vertex is replaced by a quadratic Bezier arc (control point = original vertex b, start/end = tangent points p1/p2 at distance r on either side),
    so adjacent segments meet smoothly at the vertex and sharp-corner triangles are removed. Straight segments (2 vertices) unchanged."""
    c=list(line.coords)
    if len(c)<3: return line
    out=[c[0]]
    for i in range(1,len(c)-1):
        a,b,d=c[i-1],c[i],c[i+1]
        v1=(a[0]-b[0],a[1]-b[1]); v2=(d[0]-b[0],d[1]-b[1])
        n1=math.hypot(*v1); n2=math.hypot(*v2)
        if n1<1e-6 or n2<1e-6: out.append(b); continue
        t=min(r, n1*0.45, n2*0.45)
        p1=(b[0]+v1[0]/n1*t, b[1]+v1[1]/n1*t); p2=(b[0]+v2[0]/n2*t, b[1]+v2[1]/n2*t)
        for j in range(n+1):                       # sample the quadratic Bezier arc p1->b->p2
            s=j/n; w=1.0-s
            out.append((w*w*p1[0]+2*w*s*b[0]+s*s*p2[0], w*w*p1[1]+2*w*s*b[1]+s*s*p2[1]))
    out.append(c[-1])
    try: return LineString(out)
    except Exception: return line
def offset_in(g, off):
    """Offset the run by off to both sides (round-join fillets), pick the side with the longer length inside the arcade polygon = pedestrian centreline; then round_corners trims sharp-corner triangles."""
    cand=[arc_geoms[int(i)] for i in arc_tree.query(g.buffer(off+1.5))]
    au=unary_union(cand) if cand else None
    best=None; best_in=-1.0
    for d in (off,-off):
        try: o=g.offset_curve(d, join_style='round', quad_segs=12)   # round join: arcs at convex corners, no sharp spikes
        except Exception:
            try: o=g.offset_curve(d)
            except Exception: continue
        if o.is_empty: continue
        for oo in (list(o.geoms) if o.geom_type=='MultiLineString' else [o]):
            if oo.length<0.5: continue
            il=oo.intersection(au).length if au is not None else 0.0
            if il>best_in: best_in=il; best=oo
    return round_corners(best) if best is not None else None   # fillet again, double safeguard against triangles
def is_closed(line, tol=0.5):
    c=list(line.coords); return len(c)>3 and math.hypot(c[0][0]-c[-1][0],c[0][1]-c[-1][1])<tol
def inner_ring(line, off):
    """Closed-loop path (continuous facade around a whole building) -> shrink inward by off to form an inner ring (negative buffer, round joins built in), then Bezier smoothing."""
    from shapely.geometry import Polygon
    try:
        poly=Polygon(line)
        if not poly.is_valid: poly=poly.buffer(0)
        inner=poly.buffer(-off, join_style='round')
        if inner.is_empty: return None
        if inner.geom_type=='MultiPolygon': inner=max(inner.geoms, key=lambda p:p.area)
        return round_corners(LineString(inner.exterior.coords))
    except Exception: return None
# First linemerge adjacent runs (end-to-end continuous segments of the same facade) -> continuous lines, then offset: same-facade centrelines are continuous, corners are naturally rounded, no more facconn triangles
_merged=linemerge(unary_union([force_2d(g) for g in runs.geometry]))
_mlines=[g for g in (list(_merged.geoms) if _merged.geom_type=='MultiLineString' else [_merged]) if g.geom_type=='LineString' and g.length>0.5]
print(f"  runs linemerge: {len(runs)} -> {len(_mlines)} continuous lines (removes facconn triangles at same-facade corners)",flush=True)
arc_cl=[]; _t0=time.time(); _ncl=0
for ri,mline in enumerate(_mlines):
    if is_closed(mline):
        ml=inner_ring(mline,ARC_OFFSET); _ncl+=1      # closed loop -> shrink 1 m into the building to form an inner ring
    else:
        ml=offset_in(mline,ARC_OFFSET)                # open -> offset 1 m into the arcade polygon
    if ml is not None: arc_cl.append((ml, f"arcade_{ri}"))
    if (ri+1)%2000==0: print(f"  arcade offset {ri+1}/{len(_mlines)} ({time.time()-_t0:.0f}s)",flush=True)
print(f"  arcade offset done {len(arc_cl)}/{len(_mlines)} (closed-loop inner rings {_ncl}) ({time.time()-_t0:.0f}s)",flush=True)
lkw_cl=centerlines(lkw,'linkway',mrr_fb=False)
rows=[{'src':'arcade','fac_id':fid,'geometry':g} for g,fid in arc_cl]+[{'src':'linkway','fac_id':fid,'geometry':g} for g,fid in lkw_cl]
cl=gpd.GeoDataFrame(rows,crs=3414).explode(index_parts=False).reset_index(drop=True)
cl=cl[cl.geometry.length>0.5]
cl.to_file(f"{OUT}\\step4_1_shade_centerline_SG.gpkg",driver="GPKG")
print(f"Centreline segments {len(cl)} | total length {cl.geometry.length.sum()/1000:,.1f} km | arcade {cl[cl.src=='arcade'].geometry.length.sum()/1000:,.1f} km | linkway {cl[cl.src=='linkway'].geometry.length.sum()/1000:,.1f} km",flush=True)

fig,ax=plt.subplots(figsize=(16,11))
cl[cl.src=='arcade'].plot(ax=ax,color='#b00000',lw=0.3); cl[cl.src=='linkway'].plot(ax=ax,color='#c06000',lw=0.3)
ax.set_title("Step4-1 city-wide shade-facility centrelines (red=arcade, orange=linkway)",fontsize=13); ax.set_aspect('equal'); ax.axis('off')
plt.savefig(f"{OUT}\\step4_1_centerline_SG.png",dpi=140,bbox_inches='tight'); print("preview saved",flush=True)
