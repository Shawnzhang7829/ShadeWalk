"""
GeoSAM-TopoLoRA Linkway pipeline (gsltl_pipeline.py)
=====================================================

Wraps the proven Meta-SAM2 `cl_pipeline.py` engine (SAM2 image encoder +
LoRA on QKV + UNet decoder + clDice topology loss = "TopoLoRA") and adds
the GeoSAM-TopoLoRA-specific helpers requested for the SG Covered Linkway
extraction job:

  - regenerate train/val fishnet grid files (GPKG) from the labelled image list
  - stitch the labelled train+val mask PNGs into one big binary GeoTIFF
    (overlaps OR-merged so the white covered-linkway regions accumulate)
  - inference wrapper that REQUIRES the SG_Building footprint as an
    exclusion mask and OPTIONALLY constrains output to a footpath buffer
  - mosaic inference dispatcher that picks GPU-parallel or CPU-parallel
  - progress reporting that follows the requested rule
    (every 10 tiles if total<1000, else every 100 tiles)

Only this module needs to live in the GeoSAM-TopoLoRA project.  Heavy
training / inference primitives are imported from `cl_pipeline`.
"""

from __future__ import annotations

import os
import sys
import glob
import time
import threading
from pathlib import Path
from queue import Queue
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

import cv2
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box as shapely_box, shape as shapely_shape
from scipy.ndimage import label as scipy_label


# ---------------------------------------------------------------------------
# 0. Bring in the SAM2-UNet-LoRA engine from Meta-SAM2
# ---------------------------------------------------------------------------
# The proven pipeline lives next to this repo. Add its parent to sys.path so
# `import cl_pipeline` resolves.

_CL_PIPELINE_DIR = r"D:\Claude\Meta-SAM2"
if _CL_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _CL_PIPELINE_DIR)

import cl_pipeline as _clp                                       # noqa: E402

# Re-export the public API so the notebook can do `import gsltl_pipeline as g`
generate_masks_from_json        = _clp.generate_masks_from_json
SAM2UNetLoRA                    = _clp.SAM2UNetLoRA
count_trainable_params          = _clp.count_trainable_params
get_dataloaders                 = _clp.get_dataloaders
train                           = _clp.train
list_checkpoints                = _clp.list_checkpoints
load_checkpoint                 = _clp.load_checkpoint
run_tile_inference              = _clp.run_tile_inference
run_island_full_inference       = _clp.run_island_full_inference
run_island_full_inference_parallel = _clp.run_island_full_inference_parallel
save_tile_predictions           = _clp.save_tile_predictions
stitch_tiles_to_mosaic          = _clp.stitch_tiles_to_mosaic
connectivity_postprocess        = _clp.connectivity_postprocess
apply_gis_constraints           = _clp.apply_gis_constraints
save_mask_tif                   = _clp.save_mask_tif
mask_to_polygons                = _clp.mask_to_polygons
save_vector                     = _clp.save_vector


def _report_every(n_total: int) -> int:
    """Progress reporting cadence per the project spec."""
    return 10 if n_total < 1000 else 100


# ===========================================================================
# 1. Regenerate train / val grid GPKG + tiles_meta.csv
# ===========================================================================

def generate_grids_from_labelled_tiles(
    input_tif: str,
    images_dir: str,
    grid_dir: str,
    splits: Iterable[str] = ("train", "val"),
    tile_size: int = 1024,
    overlap: int = 128,
    crs_epsg: int = 3414,
    verbose: bool = True,
) -> dict:
    """
    Rebuild the fishnet grid files for the labelled train + val tiles.

    The tile PNGs are named `tile_x{col_off}_y{row_off}.png` so the pixel
    offsets are recovered directly from the filename. The world coordinates
    of each tile are derived from the input TIF's geotransform.

    Outputs
    -------
    grid_dir/tiles_fishnet.gpkg        (all tiles, with split attr)
    grid_dir/tiles_fishnet_train.gpkg  (train subset)
    grid_dir/tiles_fishnet_val.gpkg    (val subset)
    grid_dir/tiles_meta.csv            (filename, split, col_off, row_off,
                                        width, height, transform.a..f,
                                        xmin/ymin/xmax/ymax, crs)
    """
    os.makedirs(grid_dir, exist_ok=True)

    with rasterio.open(input_tif) as src:
        tif_tf  = src.transform
        tif_crs = src.crs
        tif_w, tif_h = src.width, src.height

    rows = []
    for split in splits:
        sdir = os.path.join(images_dir, split)
        if not os.path.isdir(sdir):
            if verbose:
                print(f"[warn] {sdir} does not exist, skipping {split}")
            continue
        pngs = sorted(glob.glob(os.path.join(sdir, "*.png")))
        if verbose:
            print(f"[{split}] {len(pngs)} labelled tiles")

        for p in pngs:
            stem = Path(p).stem                         # tile_x100352_y49280
            try:
                _, xs, ys = stem.split("_")
                col_off = int(xs[1:])
                row_off = int(ys[1:])
            except Exception:
                continue

            x_min = tif_tf.c + col_off * tif_tf.a
            y_max = tif_tf.f + row_off * tif_tf.e      # e is negative
            x_max = x_min + tile_size * tif_tf.a
            y_min = y_max + tile_size * tif_tf.e

            rows.append({
                "filename":   stem + ".png",
                "split":      split,
                "col_off":    col_off,
                "row_off":    row_off,
                "width":      tile_size,
                "height":     tile_size,
                "transform_a": tif_tf.a,
                "transform_b": tif_tf.b,
                "transform_c": x_min,
                "transform_d": tif_tf.d,
                "transform_e": tif_tf.e,
                "transform_f": y_max,
                "xmin": x_min, "ymin": y_min,
                "xmax": x_max, "ymax": y_max,
                "crs":  f"EPSG:{crs_epsg}",
                "geometry": shapely_box(x_min, y_min, x_max, y_max),
            })

    if not rows:
        raise RuntimeError(f"No labelled tiles found under {images_dir}")

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=f"EPSG:{crs_epsg}")

    all_gpkg   = os.path.join(grid_dir, "tiles_fishnet.gpkg")
    train_gpkg = os.path.join(grid_dir, "tiles_fishnet_train.gpkg")
    val_gpkg   = os.path.join(grid_dir, "tiles_fishnet_val.gpkg")
    csv_path   = os.path.join(grid_dir, "tiles_meta.csv")

    gdf.to_file(all_gpkg, driver="GPKG")
    gdf[gdf["split"] == "train"].to_file(train_gpkg, driver="GPKG")
    gdf[gdf["split"] == "val"  ].to_file(val_gpkg,   driver="GPKG")
    df.drop(columns=["geometry"]).to_csv(csv_path, index=False)

    if verbose:
        print(f"\nGrid files written to {grid_dir}")
        print(f"  all  : {all_gpkg}   ({len(gdf):,} tiles)")
        print(f"  train: {train_gpkg} ({(gdf['split']=='train').sum():,})")
        print(f"  val  : {val_gpkg}   ({(gdf['split']=='val'  ).sum():,})")
        print(f"  csv  : {csv_path}")
        print(f"  tile_size={tile_size}, overlap={overlap}")

    return dict(all=all_gpkg, train=train_gpkg, val=val_gpkg, csv=csv_path,
                gdf=gdf, transform=tif_tf, crs=tif_crs,
                tif_size=(tif_w, tif_h))


# ===========================================================================
# 2. Stitch labelled mask PNGs into one big GeoTIFF (OR-union)
# ===========================================================================

def stitch_labeled_masks_to_big_mask(
    masks_dir: str,
    input_tif: str,
    output_tif: str,
    splits: Iterable[str] = ("train", "val"),
    verbose: bool = True,
) -> dict:
    """
    Combine all labelled train+val mask PNGs into a single binary GeoTIFF
    sized to match the input TIF. Overlapping white pixels are OR-merged.

    Tile placement uses the `tile_x{col}_y{row}.png` naming convention.
    """
    with rasterio.open(input_tif) as src:
        H, W       = src.height, src.width
        tif_tf     = src.transform
        tif_crs    = src.crs

    mosaic = np.zeros((H, W), dtype=np.uint8)
    files: list[tuple[str, str]] = []
    for split in splits:
        for p in sorted(glob.glob(os.path.join(masks_dir, split, "*.png"))):
            files.append((split, p))

    n_total = len(files)
    cadence = _report_every(n_total)
    if verbose:
        print(f"Stitching {n_total} labelled masks into {H}x{W} big mask...")

    n_placed = 0
    for idx, (split, p) in enumerate(files, start=1):
        stem = Path(p).stem
        try:
            _, xs, ys = stem.split("_")
            col_off, row_off = int(xs[1:]), int(ys[1:])
        except Exception:
            continue

        arr = np.array(Image.open(p).convert("L"))
        bin_arr = (arr > 128).astype(np.uint8)
        th, tw = bin_arr.shape

        r0, c0 = row_off, col_off
        r1 = min(r0 + th, H); c1 = min(c0 + tw, W)
        if r1 <= r0 or c1 <= c0:
            continue
        mosaic[r0:r1, c0:c1] = np.maximum(
            mosaic[r0:r1, c0:c1], bin_arr[: r1 - r0, : c1 - c0]
        )
        n_placed += 1
        if verbose and idx % cadence == 0:
            print(f"  {idx}/{n_total} masks merged...")

    profile = {
        "driver":   "GTiff", "dtype": "uint8",
        "width":    W, "height": H, "count": 1,
        "crs":      tif_crs, "transform": tif_tf,
        "compress": "lzw", "tiled": True,
        "blockxsize": 512, "blockysize": 512,
        "nodata":   0, "BIGTIFF": "YES",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_tif)), exist_ok=True)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(mosaic, 1)

    if verbose:
        print(f"Big mask saved: {output_tif}  "
              f"(positive px = {mosaic.sum():,})")

    return dict(big_mask=output_tif, mask=mosaic,
                transform=tif_tf, crs=tif_crs, profile=profile,
                placed=n_placed, total=n_total)


# ===========================================================================
# 3. Mosaic inference with mandatory building + optional footpath constraint
# ===========================================================================

def run_mosaic_inference(
    input_tif: str,
    model,
    output_tif: str,
    boundary_shp: str,
    building_gpkg: str,                         # MANDATORY for this project
    footpath_gpkg: Optional[str] = None,        # OPTIONAL; if given enforce
    footpath_buffer_m: float = 2.0,
    tile_size: int = 1024,
    overlap:   int = 128,
    threshold: float = 0.5,
    rgb_bands: tuple = (1, 2, 3),
    use_tta: bool = True,
    device: str = "cuda",
    batch_size: int = 8,
    use_parallel: bool = True,
    use_amp_inference: bool = True,
    verbose: bool = True,
) -> dict:
    """
    GPU- or CPU-parallel sliding-window inference on the full TIF.

    - Building footprint is loaded once and applied per-tile (mandatory).
    - If `footpath_gpkg` is provided, predictions are kept only where they
      intersect the footpath buffer (post-processing step).
    - Progress is reported every 10 tiles when total<1000, else every 100.

    Returns the dict from the underlying inference runner, augmented with
    `output_tif`.
    """
    assert building_gpkg and os.path.exists(building_gpkg), \
        "SG_Building footprint is required."

    runner = (run_island_full_inference_parallel if use_parallel
              else run_island_full_inference)

    if verbose:
        print(f"Mosaic inference: parallel={'GPU' if use_parallel else 'CPU/sequential'}, "
              f"device={device}, tile={tile_size}, overlap={overlap}, "
              f"TTA={use_tta}, batch={batch_size}")

    kwargs = dict(
        input_tif       = input_tif,
        boundary_shp    = boundary_shp,
        model           = model,
        output_tif      = output_tif,
        tile_size       = tile_size,
        overlap         = overlap,
        threshold       = threshold,
        use_tta         = use_tta,
        filter_canopy   = True,
        building_gpkg   = building_gpkg,
        rgb_bands       = rgb_bands,
        device          = device,
        verbose         = verbose,
    )
    if use_parallel:
        kwargs.update(batch_size=batch_size,
                      use_amp_inference=use_amp_inference)

    info = runner(**kwargs)

    # ── Optional footpath constraint (post-processing on the written TIF) ──
    if footpath_gpkg and os.path.exists(footpath_gpkg):
        if verbose:
            print(f"\nApplying footpath constraint "
                  f"(buffer={footpath_buffer_m} m)...")
        _apply_footpath_filter_inplace(
            mosaic_tif=output_tif,
            footpath_gpkg=footpath_gpkg,
            buffer_m=footpath_buffer_m,
            verbose=verbose,
        )

    info["output_tif"] = output_tif
    return info


def _apply_footpath_filter_inplace(
    mosaic_tif: str,
    footpath_gpkg: str,
    buffer_m: float = 2.0,
    block: int = 4096,
    verbose: bool = True,
) -> None:
    """
    Block-wise multiply the mosaic TIF in-place by a rasterised footpath
    buffer.  Streams to keep memory bounded on a country-sized raster.
    """
    fp = gpd.read_file(footpath_gpkg)
    with rasterio.open(mosaic_tif, "r+") as ds:
        if fp.crs != ds.crs:
            fp = fp.to_crs(ds.crs)
        fp_buf = fp.copy()
        fp_buf["geometry"] = fp.geometry.buffer(buffer_m)

        H, W   = ds.height, ds.width
        tf     = ds.transform
        n_total = ((H + block - 1) // block) * ((W + block - 1) // block)
        cadence = _report_every(n_total)
        idx = 0
        for r0 in range(0, H, block):
            for c0 in range(0, W, block):
                idx += 1
                rh = min(block, H - r0); cw = min(block, W - c0)
                win = Window(c0, r0, cw, rh)
                win_tf = rasterio.windows.transform(win, tf)

                arr = ds.read(1, window=win)
                if not arr.any():
                    continue

                fp_block = rasterize(
                    [(g, 1) for g in fp_buf.geometry],
                    out_shape=(rh, cw), transform=win_tf,
                    fill=0, dtype="uint8",
                )
                if fp_block.any():
                    ds.write((arr * fp_block).astype(np.uint8), 1, window=win)
                else:
                    ds.write(np.zeros((rh, cw), dtype=np.uint8), 1, window=win)

                if verbose and idx % cadence == 0:
                    print(f"  footpath block {idx}/{n_total} done")
    if verbose:
        print("Footpath filter applied in place.")


# ===========================================================================
# 4. Final vectorisation / saving (thin wrapper)
# ===========================================================================

def vectorise_mosaic(
    mosaic_tif: str,
    output_dir: str,
    name: str = "covered_linkway",
    min_area_m2: float = 4.0,
    verbose: bool = True,
) -> tuple[str, str]:
    """
    Polygonise the final mosaic TIF and save GPKG + SHP.
    """
    with rasterio.open(mosaic_tif) as ds:
        mask    = ds.read(1)
        tf      = ds.transform
        crs     = ds.crs
        pix     = abs(tf.a)

    gdf = mask_to_polygons(
        mask=mask, transform=tf, crs=crs,
        min_area_m2=min_area_m2, pixel_size_m=pix,
    )
    gpkg, shp = save_vector(gdf, output_dir, name=name)
    if verbose:
        print(f"Vectorised {len(gdf):,} polygons "
              f"(min_area_m2={min_area_m2})")
    return gpkg, shp
