"""
Build tiles_fishnet.gpkg labelled by the v3 dataset split:
  train  = tile present in v3  images/train
  val    = tile present in v3  images/val   (test set)
  blank  = tile in the grid but NOT used by v3 (no annotation)

Geometry from tiles_meta.csv (xmin/ymin/xmax/ymax, EPSG:3414).
Any v3 tile missing from the csv is rebuilt from its filename offsets
+ the source-TIF geotransform so nothing is lost.
"""
from __future__ import annotations
import os, glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rasterio

META  = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\grid\tiles_meta.csv"
V3IMG = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\images"
SRC   = r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\SG_google_map_03m_SVY21.tif"
OUTDIR = r"C:\Users\City Syntax Lab\Desktop\Covered Linkways\1-output\0-SG Google map\output_files"
TILE = 1024
CRS  = "EPSG:3414"

os.makedirs(OUTDIR, exist_ok=True)

# v3 split membership (by filename)
v3 = {}
for s in ("train", "val"):
    for p in glob.glob(os.path.join(V3IMG, s, "*.png")):
        v3[os.path.basename(p)] = s
print(f"v3 labelled tiles: {sum(v=='train' for v in v3.values())} train, "
      f"{sum(v=='val' for v in v3.values())} val  (total {len(v3)})")

df = pd.read_csv(META)
print(f"tiles_meta.csv: {len(df)} grid tiles")

with rasterio.open(SRC) as srx:
    tf = srx.transform

rows = []
seen = set()
for _, r in df.iterrows():
    fn = r["filename"]; seen.add(fn)
    split = v3.get(fn, "blank")          # train / val / blank
    rows.append(dict(filename=fn, split=split,
                     col_off=int(r["col_off"]), row_off=int(r["row_off"]),
                     geometry=box(r["xmin"], r["ymin"], r["xmax"], r["ymax"])))

# add any v3 tile not in the csv (rebuild bbox from offsets + transform)
added = 0
for fn, s in v3.items():
    if fn in seen:
        continue
    stem = os.path.splitext(fn)[0]            # tile_x{col}_y{row}
    _, xs, ys = stem.split("_")
    c = int(xs[1:]); rr = int(ys[1:])
    x0 = tf.c + c * tf.a
    y0 = tf.f + rr * tf.e
    x1 = x0 + TILE * tf.a
    y1 = y0 + TILE * tf.e
    rows.append(dict(filename=fn, split=s, col_off=c, row_off=rr,
                      geometry=box(min(x0, x1), min(y0, y1),
                                   max(x0, x1), max(y0, y1))))
    added += 1
if added:
    print(f"added {added} v3 tiles missing from csv")

gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
vc = gdf["split"].value_counts().to_dict()
print(f"fishnet: total {len(gdf)}  ->  {vc}")

main = os.path.join(OUTDIR, "tiles_fishnet.gpkg")
gdf.to_file(main, driver="GPKG")
print(f"saved {main}")
for s in ("train", "val", "blank"):
    sub = gdf[gdf["split"] == s]
    if len(sub):
        op = os.path.join(OUTDIR, f"tiles_fishnet_{s}.gpkg")
        sub.to_file(op, driver="GPKG")
        print(f"saved {op}  ({len(sub)} tiles)")
