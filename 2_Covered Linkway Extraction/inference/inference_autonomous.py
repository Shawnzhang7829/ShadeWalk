"""
GeoSAM + TopoLoRA AUTONOMOUS inference on a full GeoTIFF.

Loads a lean checkpoint (LoRA + mask_decoder + projection only),
runs sliding-window inference with text-only prompts (no point prompts),
applies building exclusion + boundary clip, writes GeoTIFF + GPKG.

All writes to C: (mosaic on C:); reads from D: (TIF + GIS layers).
"""

from __future__ import annotations

import os, sys, time, argparse, threading
from queue import Queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import rasterio
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import shape as shapely_shape

from segment_anything import sam_model_registry

# Reuse helpers from training scripts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))  # repo layout: training modules

from train_geosam_topolora_linkway import inject_lora_into_qkv, ensure_text_embedding
from train_autonomous_v2 import forward_autonomous, _MEAN, _STD


def build_and_load(sam_ckpt, trained_ckpt, device="cuda"):
    """Rebuild SAM + LoRA + projection and load a LEAN checkpoint."""
    print(f"[load] SAM ViT-H base : {sam_ckpt}")
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device)
    for p in sam.parameters():
        p.requires_grad_(False)

    state = torch.load(trained_ckpt, map_location=device, weights_only=False)
    lora_rank  = state.get("lora_rank",  16)
    lora_alpha = state.get("lora_alpha", 32.0)
    print(f"[load] injecting LoRA rank={lora_rank} alpha={lora_alpha}")
    inject_lora_into_qkv(sam.image_encoder, rank=lora_rank, alpha=lora_alpha)
    sam.image_encoder.to(device)

    # Load lean trainable subset
    sam_state = state["sam_trainable"]
    missing, unexpected = sam.load_state_dict(sam_state, strict=False)
    print(f"[load] sam_trainable: {len(sam_state)} keys "
          f"(missing/frozen={len(missing)}, unexpected={len(unexpected)})")

    proj = nn.Linear(512, 256).to(device)
    proj.load_state_dict(state["projection"])

    sam.eval(); proj.eval()
    print(f"[load] checkpoint epoch={state.get('epoch')} val_dice={state.get('val_dice'):.4f}")
    return sam, proj


def run_mosaic(input_tif, sam_ckpt, trained_ckpt, text_emb_path,
               output_tif, output_gpkg,
               building_gpkg=None, boundary_shp=None,
               tile_size=1024, overlap=128, threshold=0.5,
               rgb_bands=(1, 2, 3), min_area_m2=4.0,
               device="cuda", verbose=True):
    sam, proj = build_and_load(sam_ckpt, trained_ckpt, device)
    text_emb = ensure_text_embedding(text_emb_path,
                                     class_name="Covered Linkway",
                                     device=device)

    # ── Open input ──
    with rasterio.open(input_tif) as src:
        H, W = src.height, src.width
        transform = src.transform; crs = src.crs
    pix = abs(transform.a)
    if verbose:
        print(f"[geom] input {W}x{H} px @ {pix} m  CRS={crs}")

    # ── Rasterise GIS layers (small TIF: keep in RAM) ──
    bld_mask = None
    if building_gpkg and os.path.exists(building_gpkg):
        if verbose: print(f"[gis] loading buildings...")
        bld = gpd.read_file(building_gpkg).to_crs(crs)
        l, b, r, t = (transform.c, transform.f + H*transform.e,
                      transform.c + W*transform.a, transform.f)
        bld = bld.cx[l:r, b:t]
        if verbose: print(f"[gis] {len(bld):,} buildings in extent")
        bld_mask = rasterize([(g, 1) for g in bld.geometry],
                             out_shape=(H, W), transform=transform,
                             fill=0, dtype=np.uint8)
        if verbose: print(f"[gis] building mask covers {100*bld_mask.mean():.2f}%")

    bnd_mask = None
    if boundary_shp and os.path.exists(boundary_shp):
        bnd = gpd.read_file(boundary_shp).to_crs(crs)
        bnd_mask = rasterize([(g, 1) for g in bnd.geometry],
                             out_shape=(H, W), transform=transform,
                             fill=0, dtype=np.uint8)
        if verbose: print(f"[gis] boundary covers {100*bnd_mask.mean():.2f}%")

    # ── Tile grid ──
    step = tile_size - overlap
    rows = list(range(0, max(H-tile_size+1, 1), step))
    cols = list(range(0, max(W-tile_size+1, 1), step))
    if not rows or rows[-1] + tile_size < H: rows.append(max(H-tile_size, 0))
    if not cols or cols[-1] + tile_size < W: cols.append(max(W-tile_size, 0))
    tiles = [(r, c) for r in rows for c in cols]
    n_total = len(tiles)
    cadence = 10 if n_total < 1000 else 100
    if verbose: print(f"[grid] {len(rows)}x{len(cols)}={n_total} tiles")

    # ── Blending weight ramp ──
    ramp = np.ones(tile_size, np.float32)
    ramp[:overlap]  = np.linspace(0, 1, overlap)
    ramp[-overlap:] = np.linspace(1, 0, overlap)
    w2d = np.outer(ramp, ramp)

    accum  = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.float32)
    t0 = time.time()

    # ── Pipelined reader + GPU ──
    Q = Queue(maxsize=8)
    def reader_fn():
        try:
            with rasterio.open(input_tif) as src:
                for idx, (r, c) in enumerate(tiles):
                    win = Window(c, r, tile_size, tile_size)
                    data = src.read(list(rgb_bands), window=win,
                                    out_shape=(len(rgb_bands), tile_size, tile_size),
                                    boundless=True, fill_value=0)
                    tile_rgb = data.transpose(1, 2, 0).astype(np.uint8)
                    if tile_rgb.max() < 5:        # black tile, skip
                        Q.put(dict(idx=idx, r=r, c=c, prob=None))
                        continue
                    Q.put(dict(idx=idx, r=r, c=c, tile_rgb=tile_rgb))
        finally:
            Q.put(None)

    threading.Thread(target=reader_fn, daemon=False).start()

    while True:
        item = Q.get()
        if item is None: break
        if item.get("prob", "missing") is None and "tile_rgb" not in item:
            continue
        # Prepare image tensor (ImageNet normalised) and forward
        img = item["tile_rgb"].astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img_t = torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                logits = forward_autonomous(sam, proj, text_emb, img_t, device,
                                            use_text=True)
            prob = torch.sigmoid(logits).squeeze().float().cpu().numpy()

        r, c, idx = item["r"], item["c"], item["idx"]
        re_, ce_ = r + tile_size, c + tile_size
        accum [r:re_, c:ce_] += prob * w2d
        weight[r:re_, c:ce_] += w2d
        if verbose and (idx+1) % cadence == 0:
            el = time.time()-t0; eta = el/(idx+1)*(n_total-idx-1)
            print(f"  {idx+1}/{n_total} tiles  elapsed={el/60:.1f}min  ETA={eta/60:.1f}min",
                  flush=True)

    weight = np.where(weight == 0, 1, weight)
    prob_map = accum / weight
    pred = (prob_map > threshold).astype(np.uint8)
    if verbose: print(f"[infer] raw positive px: {pred.sum():,} ({100*pred.mean():.3f}%)")

    if bnd_mask is not None:
        pred = pred * bnd_mask
        if verbose: print(f"[gis] after boundary clip: {pred.sum():,}")
    if bld_mask is not None:
        pred = pred * (1 - bld_mask)
        if verbose: print(f"[gis] after building exclusion: {pred.sum():,}")

    # ── Save GeoTIFF ──
    profile = {"driver":"GTiff", "dtype":"uint8", "width":W, "height":H,
               "count":1, "crs":crs, "transform":transform,
               "compress":"lzw", "tiled":True,
               "blockxsize":512, "blockysize":512, "nodata":0}
    os.makedirs(os.path.dirname(os.path.abspath(output_tif)), exist_ok=True)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(pred, 1)
    print(f"[save] {output_tif}")

    # ── Vectorise ──
    rows_out = []
    for geom_dict, val in rio_shapes(pred, transform=transform):
        if val != 1: continue
        s = shapely_shape(geom_dict)
        if s.area >= min_area_m2:
            rows_out.append({"geometry": s, "area_m2": round(s.area, 2)})
    if rows_out:
        gdf = gpd.GeoDataFrame(rows_out, crs=crs)
        gdf = gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)
        gdf.to_file(output_gpkg, driver="GPKG")
        print(f"[save] {output_gpkg}  ({len(gdf):,} polygons)")
    else:
        print("[save] no polygons above min_area_m2")

    return dict(output_tif=output_tif, output_gpkg=output_gpkg,
                positive_px=int(pred.sum()),
                n_polygons=len(rows_out),
                elapsed_sec=time.time()-t0)


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--input_tif",
        default=r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\SG_google_map_4000_03m.tif")
    P.add_argument("--sam_ckpt",
        default=r"C:\GeoSAM-backup\sam_vit_h_4b8939.pth")
    P.add_argument("--trained_ckpt", required=True)
    P.add_argument("--text_emb",
        default=r"C:\GeoSAM-backup\clip_linkway_emb.pth")
    P.add_argument("--output_tif",
        default=r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_autonomous.tif")
    P.add_argument("--output_gpkg",
        default=r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_autonomous.gpkg")
    P.add_argument("--building_gpkg",
        default=r"C:\Users\City Syntax Lab\Desktop\SOLWEIG_GPU\data\1-SG data\Shp\SG_Building\SG_Building_SVY21.gpkg")
    P.add_argument("--boundary_shp",
        default=r"D:\Claude\GeoSAM-TopoLoRA\shp\SUB_SG_polygon\SUB_SG_polygon.shp")
    P.add_argument("--threshold", type=float, default=0.5)
    P.add_argument("--min_area_m2", type=float, default=4.0)
    args = P.parse_args()

    info = run_mosaic(
        input_tif=args.input_tif, sam_ckpt=args.sam_ckpt,
        trained_ckpt=args.trained_ckpt, text_emb_path=args.text_emb,
        output_tif=args.output_tif, output_gpkg=args.output_gpkg,
        building_gpkg=args.building_gpkg, boundary_shp=args.boundary_shp,
        threshold=args.threshold, min_area_m2=args.min_area_m2,
    )
    print(info)
