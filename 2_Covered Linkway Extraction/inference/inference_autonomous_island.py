"""
GeoSAM + TopoLoRA AUTONOMOUS — FULL ISLAND streaming inference.

Designed for very large GeoTIFFs (179k x 115k px Singapore island).
NO in-memory mosaic — writes directly to a tiled GeoTIFF.

Workflow per tile:
  Reader thread  : read RGB + downsampled boundary/building masks
        ↓
  GPU thread     : SAM ViT-H + LoRA + text-only prompt -> prob map
        ↓
  Main loop      : threshold + boundary clip + building exclusion
                   + write center crop to output GeoTIFF
                   + progress every 100 tiles
"""

from __future__ import annotations

import os, sys, time, argparse, threading
from queue import Queue
import numpy as np
import torch
import torch.nn as nn
import cv2
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
from segment_anything import sam_model_registry

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))  # repo layout: training modules

from train_geosam_topolora_linkway import inject_lora_into_qkv, ensure_text_embedding
from train_autonomous_v2 import forward_autonomous, _MEAN, _STD


def build_and_load(sam_ckpt, trained_ckpt, device="cuda"):
    print(f"[load] SAM ViT-H base: {sam_ckpt}")
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device)
    for p in sam.parameters():
        p.requires_grad_(False)
    state = torch.load(trained_ckpt, map_location=device, weights_only=False)
    inject_lora_into_qkv(sam.image_encoder,
                         rank=state.get("lora_rank", 16),
                         alpha=state.get("lora_alpha", 32.0))
    sam.image_encoder.to(device)
    sam.load_state_dict(state["sam_trainable"], strict=False)
    proj = nn.Linear(512, 256).to(device)
    proj.load_state_dict(state["projection"])
    sam.eval(); proj.eval()
    print(f"[load] OK  epoch={state.get('epoch')}  val_dice={state.get('val_dice'):.4f}")
    return sam, proj


def run(input_tif, sam_ckpt, trained_ckpt, text_emb_path,
        output_tif,
        boundary_shp, building_gpkg=None,
        tile_size=1024, overlap=128, threshold=0.5,
        rgb_bands=(1, 2, 3),
        device="cuda", verbose=True):

    sam, proj = build_and_load(sam_ckpt, trained_ckpt, device)
    text_emb = ensure_text_embedding(text_emb_path,
                                     class_name="Covered Linkway",
                                     device=device)

    # ── 1. Open input ──
    with rasterio.open(input_tif) as src:
        H, W      = src.height, src.width
        transform = src.transform
        crs       = src.crs
    pix = abs(transform.a)
    if verbose:
        print(f"[geom] input {W:,}x{H:,} px @ {pix} m  CRS={crs}")

    # ── 2. Downsampled boundary mask (1/16) for fast tile filtering ──
    DOWN = 16
    H_d, W_d = H // DOWN, W // DOWN
    transform_d = rasterio.transform.from_origin(
        transform.c, transform.f, pix * DOWN, pix * DOWN
    )
    if verbose: print(f"[gis] loading boundary {boundary_shp}")
    bnd_gdf = gpd.read_file(boundary_shp).to_crs(crs)
    bnd_d = rasterize([(g, 1) for g in bnd_gdf.geometry],
                      out_shape=(H_d, W_d), transform=transform_d,
                      fill=0, dtype=np.uint8)
    if verbose: print(f"[gis] boundary covers {100*bnd_d.mean():.2f}% of raster")

    # ── 3. Downsampled building mask (1/16) ──
    bld_d = None
    if building_gpkg and os.path.exists(building_gpkg):
        if verbose: print(f"[gis] loading buildings {building_gpkg}")
        bld_gdf = gpd.read_file(building_gpkg).to_crs(crs)
        if verbose: print(f"[gis] {len(bld_gdf):,} building polygons; rasterising 1/{DOWN}...")
        bld_d = rasterize([(g, 1) for g in bld_gdf.geometry],
                          out_shape=(H_d, W_d), transform=transform_d,
                          fill=0, dtype=np.uint8)
        del bld_gdf
        if verbose: print(f"[gis] building mask covers {100*bld_d.mean():.2f}%")

    # ── 4. Sliding window + filter to tiles overlapping the boundary ──
    step = tile_size - overlap
    rows = list(range(0, max(H - tile_size + 1, 1), step))
    cols = list(range(0, max(W - tile_size + 1, 1), step))
    if not rows or rows[-1] + tile_size < H: rows.append(max(H - tile_size, 0))
    if not cols or cols[-1] + tile_size < W: cols.append(max(W - tile_size, 0))

    valid = []
    for r in rows:
        for c in cols:
            r_d0, c_d0 = r // DOWN, c // DOWN
            r_d1 = (r + tile_size + DOWN - 1) // DOWN
            c_d1 = (c + tile_size + DOWN - 1) // DOWN
            if bnd_d[r_d0:r_d1, c_d0:c_d1].sum() > 0:
                valid.append((r, c))
    n_total = len(rows) * len(cols)
    n_valid = len(valid)
    cadence = 10 if n_valid < 1000 else 100
    if verbose:
        print(f"[grid] total tiles : {n_total:,} ({len(rows)} rows x {len(cols)} cols)")
        print(f"[grid] valid tiles : {n_valid:,} ({100*n_valid/n_total:.1f}%, rest is ocean)")

    # ── 5. Output GeoTIFF profile (tiled, LZW, BIGTIFF) ──
    pad = overlap // 2
    out_profile = {
        "driver":"GTiff", "dtype":"uint8",
        "width":W, "height":H, "count":1,
        "crs":crs, "transform":transform,
        "compress":"lzw", "tiled":True,
        "blockxsize":512, "blockysize":512,
        "nodata":0, "BIGTIFF":"YES",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_tif)), exist_ok=True)
    if verbose: print(f"[out] writing to {output_tif}")

    # ── 6. Streaming pipeline ──
    Q = Queue(maxsize=8)

    def reader_fn():
        try:
            with rasterio.open(input_tif) as src:
                for idx, (r, c) in enumerate(valid):
                    win = Window(c, r, tile_size, tile_size)
                    data = src.read(list(rgb_bands), window=win,
                                    out_shape=(len(rgb_bands), tile_size, tile_size),
                                    boundless=True, fill_value=0)
                    tile_rgb = data.transpose(1, 2, 0).astype(np.uint8)
                    if tile_rgb.max() < 5:                  # all-black tile
                        Q.put(dict(idx=idx, r=r, c=c, skip=True))
                        continue

                    r_d0, c_d0 = r // DOWN, c // DOWN
                    r_d1 = (r + tile_size + DOWN - 1) // DOWN
                    c_d1 = (c + tile_size + DOWN - 1) // DOWN
                    bnd_tile = cv2.resize(bnd_d[r_d0:r_d1, c_d0:c_d1],
                                          (tile_size, tile_size),
                                          interpolation=cv2.INTER_NEAREST)
                    bld_tile = None
                    if bld_d is not None:
                        bd = bld_d[r_d0:r_d1, c_d0:c_d1]
                        if bd.sum() > 0:
                            bld_tile = cv2.resize(bd, (tile_size, tile_size),
                                                  interpolation=cv2.INTER_NEAREST)
                    Q.put(dict(idx=idx, r=r, c=c,
                               tile_rgb=tile_rgb,
                               bnd_tile=bnd_tile,
                               bld_tile=bld_tile))
        finally:
            Q.put(None)

    threading.Thread(target=reader_fn, daemon=False).start()

    t0 = time.time()
    pos_total = 0
    with rasterio.open(output_tif, "w", **out_profile) as dst:
        while True:
            item = Q.get()
            if item is None: break
            r, c, idx = item["r"], item["c"], item["idx"]

            if item.get("skip"):
                continue

            # GPU forward
            img = item["tile_rgb"].astype(np.float32) / 255.0
            img = (img - _MEAN) / _STD
            img_t = torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    logits = forward_autonomous(sam, proj, text_emb,
                                                img_t, device, use_text=True)
                prob = torch.sigmoid(logits).squeeze().float().cpu().numpy()
            pred = (prob > threshold).astype(np.uint8)
            pred = pred * item["bnd_tile"]
            if item["bld_tile"] is not None:
                pred = pred * (1 - item["bld_tile"])

            # Write center crop only (avoid edge double-writes from overlap)
            wr_top = 0           if r == rows[0]  else pad
            wr_bot = tile_size   if r == rows[-1] else (tile_size - pad)
            wr_lft = 0           if c == cols[0]  else pad
            wr_rht = tile_size   if c == cols[-1] else (tile_size - pad)

            crop = pred[wr_top:wr_bot, wr_lft:wr_rht]
            r_off, c_off = r + wr_top, c + wr_lft
            ow = min(wr_rht - wr_lft, W - c_off)
            oh = min(wr_bot - wr_top, H - r_off)
            if ow > 0 and oh > 0:
                dst.write(crop[:oh, :ow], 1,
                          window=Window(c_off, r_off, ow, oh))
                pos_total += int(crop[:oh, :ow].sum())

            if verbose and (idx + 1) % cadence == 0:
                el = time.time() - t0
                eta = el / (idx + 1) * (n_valid - idx - 1)
                print(f"  {idx+1:,}/{n_valid:,}  ({100*(idx+1)/n_valid:.1f}%)  "
                      f"elapsed={el/60:.1f}min  ETA={eta/60:.1f}min", flush=True)

    el = time.time() - t0
    print(f"\n[done] {n_valid:,} tiles in {el/60:.1f} min  ({el/n_valid:.2f}s/tile)")
    print(f"[done] positive px: {pos_total:,}  ({100*pos_total/(H*W):.3f}%)")
    print(f"[done] output: {output_tif}")
    return dict(output_tif=output_tif, total_tiles=n_total,
                valid_tiles=n_valid, positive_px=pos_total,
                elapsed_sec=el)


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--input_tif",
        default=r"D:\Claude\GeoSAM-TopoLoRA\covered Linkway\SG_google_map_03m_SVY21.tif")
    P.add_argument("--sam_ckpt",
        default=r"C:\GeoSAM-backup\sam_vit_h_4b8939.pth")
    P.add_argument("--trained_ckpt",
        default=r"C:\GeoSAM-backup\runs\autonomous\geosam_topolora_autonomous_lean_dice0.5242.pth")
    P.add_argument("--text_emb",
        default=r"C:\GeoSAM-backup\clip_linkway_emb.pth")
    P.add_argument("--output_tif",
        default=r"C:\GeoSAM-backup\runs\autonomous\covered_linkway_SG_island.tif")
    P.add_argument("--boundary_shp",
        default=r"D:\Claude\GeoSAM-TopoLoRA\shp\Island_boarder\Island_boarder.shp")
    P.add_argument("--building_gpkg",
        default=r"C:\Users\City Syntax Lab\Desktop\SOLWEIG_GPU\data\1-SG data\Shp\SG_Building\SG_Building_SVY21.gpkg")
    P.add_argument("--threshold", type=float, default=0.5)
    args = P.parse_args()

    run(input_tif=args.input_tif,
        sam_ckpt=args.sam_ckpt,
        trained_ckpt=args.trained_ckpt,
        text_emb_path=args.text_emb,
        output_tif=args.output_tif,
        boundary_shp=args.boundary_shp,
        building_gpkg=args.building_gpkg,
        threshold=args.threshold)
