# -*- coding: utf-8 -*-
"""Tile-wise driver: reuses the 77 preprocessed tiles; each tile runs compute_utci in a subprocess with a timeout (hung tiles are killed/logged/skipped),
 and tiles with existing output are skipped. Fixes the problem that the citywide solweig loop has no timeout, so a single hanging tile stalls the whole run.
 Run: python-qgis-ltr.bat driver_tilewise.py [single tile key for testing]
"""
import sys, os, time
import multiprocessing as mp
import numpy as np
sys.path.insert(0, r'D:\Claude\SVI_FFW\Module\SOLWEIG-GPU')
from solweig_gpu.utci_process import map_files_by_key

BASE = r"D:\Claude\SVI_FFW\TIF_shadow_newarcade"
PRE  = os.path.join(BASE, "processed_inputs")
OUT  = os.path.join(BASE, "output_folder")
DATE = '2026-03-01'
TIMEOUT = 600   # 10 min per tile

def _key(k):
    try: x, y = k.split("_"); return (int(x), int(y))
    except Exception: return (10**18, 10**18)

def run_tile(kwargs):
    sys.path.insert(0, r'D:\Claude\SVI_FFW\Module\SOLWEIG-GPU')
    from solweig_gpu.utci_process import compute_utci
    compute_utci(**kwargs)

def build_maps():
    m = lambda d, e=".tif": map_files_by_key(os.path.join(PRE, d), e)
    return dict(bdsm=m("Building_DSM"), tree=m("Trees"), ldsm=m("LDSM"), adsm=m("ADSM"),
                adsmb=m("ADSMB"), brem=m("BREMAIN"), dem=m("DEM"), walls=m("walls"),
                aspect=m("aspect"), met=m("metfiles", ".txt"))

def kwargs_for(M, key, outdir):
    return dict(
        building_dsm_path=M["bdsm"][key], tree_path=M["tree"][key], dem_path=M["dem"][key],
        walls_path=M["walls"][key], aspect_path=M["aspect"][key], landcover_path=None,
        met_file_data=np.loadtxt(M["met"][key], skiprows=1),
        output_path=outdir, number=key, selected_date_str=DATE,
        save_tmrt=False, save_svf=False, save_kup=False, save_kdown=False,
        save_lup=False, save_ldown=False, save_shadow=True, shadow_category=True, only_shadow=True,
        canopy_height_ratio=0.23, ldsm_bottom_height_ratio=0.90, shelter_transmittance=0.0,
        ldsm_path=M["ldsm"][key], adsm_path=M["adsm"][key], adsm_base_path=M["adsmb"][key],
        arcade_transmittance=0.0, bremain_path=M["brem"][key],
    )

def run_one(M, key, force=False):
    outdir = os.path.join(OUT, key); os.makedirs(outdir, exist_ok=True)
    shp = os.path.join(outdir, f"Shadow_{key}.tif")
    if os.path.exists(shp) and not force:
        return "skip", 0
    t = time.time()
    p = mp.Process(target=run_tile, args=(kwargs_for(M, key, outdir),))
    p.start(); p.join(timeout=TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join(); return "STUCK", time.time()-t
    if p.exitcode != 0:
        return f"ERR(exit{p.exitcode})", time.time()-t
    return "done", time.time()-t

def main():
    M = build_maps()
    keys = sorted(set(M["bdsm"]) & set(M["tree"]) & set(M["dem"]) & set(M["met"]) &
                  set(M["ldsm"]) & set(M["adsm"]) & set(M["adsmb"]) & set(M["brem"]), key=_key)
    print(f"total tiles: {len(keys)}", flush=True)
    if len(sys.argv) > 1:   # single-tile test (forced recompute)
        k = sys.argv[1]; st, dt = run_one(M, k, force=True)
        print(f"[TEST {k}] {st} {dt:.0f}s", flush=True); return
    done = skip = 0; bad = []
    for i, key in enumerate(keys):
        st, dt = run_one(M, key)
        if st == "skip": skip += 1; continue
        if st == "done": done += 1; print(f"[done] {key} {dt:.0f}s [{i+1}/{len(keys)}]", flush=True)
        else: bad.append((key, st)); print(f"[{st}] {key} {dt:.0f}s [{i+1}/{len(keys)}]", flush=True)
    print(f"\n[ALL] done={done} skip={skip} bad={len(bad)}", flush=True)
    if bad: print("bad tiles:", bad, flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
