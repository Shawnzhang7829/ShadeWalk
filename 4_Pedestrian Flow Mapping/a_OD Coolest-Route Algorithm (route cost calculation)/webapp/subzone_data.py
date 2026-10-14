# -*- coding: utf-8 -*-
"""Subzone overlay data for the web tool (Basemap > Subzone: boundaries and names of Singapore's planning subzones).
Source: Shp/SG/SG_Subzone/SG_subzone boundary 2019_SVY21.shp (URA Master Plan 2019, 332 subzones, EPSG:3414).
Output (embedded as D.subz, same page-metre convention as the other rings: EPSG:3857 minus ox/oy, rounded to 1 m,
flat [x1,y1,x2,y2,...] rings without the closing vertex; the page's ringLL() closes them):
  n  = SUBZONE_N (official upper-case names), p = PLN_AREA_N,
  g  = per subzone a list of polygons, each a list of rings (exterior first),
  lx, ly = label point per subzone (polylabel of its largest part, 5 m tolerance).
Boundaries simplified with a 2 m tolerance (58,917 -> ~19,500 vertices)."""
import numpy as np

SHP = r"D:\Claude\SVI_FFW\Shp\SG\SG_Subzone\SG_subzone boundary 2019_SVY21.shp"


def build_subz(ox, oy, simp=2.0):
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.ops import polylabel
    g = gpd.read_file(SHP)
    if g.crs is None or g.crs.to_epsg() != 3414:
        g = g.set_crs(3414, allow_override=True) if g.crs is None else g.to_crs(3414)
    TO = Transformer.from_crs(3414, 3857, always_xy=True)

    def ring(coords):
        a = np.asarray(coords, float)[:, :2]
        mx, my = TO.transform(a[:, 0], a[:, 1])
        r = np.round(np.c_[np.asarray(mx) - ox, np.asarray(my) - oy]).astype(np.int64)[:-1]   # drop the closing vertex
        keep = np.r_[True, np.any(np.diff(r, axis=0) != 0, axis=1)]                          # drop repeats created by 1 m rounding
        return r[keep].ravel().tolist()

    n, p, gg, lx, ly = [], [], [], [], []
    for name, plan, geom, simp_geom in zip(g['SUBZONE_N'], g['PLN_AREA_N'], g.geometry, g.geometry.simplify(simp, preserve_topology=True)):
        polys = list(simp_geom.geoms) if simp_geom.geom_type == 'MultiPolygon' else [simp_geom]
        rings = []
        for q in polys:
            rr = [ring(q.exterior.coords)] + [ring(i.coords) for i in q.interiors]
            rr = [x for x in rr if len(x) >= 6]
            if rr: rings.append(rr)
        parts = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
        lp = polylabel(max(parts, key=lambda q: q.area), tolerance=5.0)
        mx, my = TO.transform(lp.x, lp.y)
        n.append(str(name)); p.append(str(plan)); gg.append(rings)
        lx.append(int(round(mx - ox))); ly.append(int(round(my - oy)))
    return dict(n=n, p=p, g=gg, lx=lx, ly=ly)
