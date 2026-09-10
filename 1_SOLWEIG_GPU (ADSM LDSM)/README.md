# Module 1 - SOLWEIG-GPU with LDSM and ADSM shadow layers

This module is the shadow engine of ShadeWalk. It is a fork of **SOLWEIG-GPU v1.2.21**
(Kamath et al., 2026, *JOSS* 11(118):9535, GPL-3.0, https://github.com/nvnsudharsan/solweig-gpu)
extended with two additional opaque shading layers and a 13-class shadow-category output:

| Layer | Raster | What it represents | Shadow class family |
|---|---|---|---|
| **LDSM** - linkway DSM | `LDSM` (height above ground, 3 m in Singapore) | roofs of covered linkways (pedestrian shelters), modelled as a thin opaque canopy with a bottom at `ldsm_bottom_height_ratio` x top (default 0.90) | 4-7 |
| **ADSM / ADSMB** - arcade DSM top / base | `ADSM` (= building height over the arcade strip), `ADSMB` (= arcade ceiling height `arc_h`, capped to top - 0.5 m, min 0.5 m) | the upper floors of a shophouse that float above a five-foot way / arcade walkway; the walkway strip itself is cut out of the building DSM (`DSMremain`) so the ground under the box stays walkable | 8-11 |
| **BREMAIN** - building-footprint mask | 0/1 | remaining building footprints; pixels inside are labelled class 12 (not a shadow class) | 12 |

The whole implementation lives in the `solweig_gpu/` package of this folder (the *ADSM/LDSM module*). The exact
changes against upstream SOLWEIG-GPU are listed in `CHANGES_vs_upstream_v1.2.21.patch` (1,330 changed lines in
five files); the touched files are `preprocessor.py` (tiling of the new rasters, met-file time window),
`solweig_gpu.py` (new `thermal_comfort` arguments, sparse-tile skipping, tile reuse, shadow-only mode),
`utci_process.py` (per-tile driver `compute_utci` with the new layers and the category output),
`solweig.py` (the `_opaque_layer_shadow` function used for both LDSM and ADSM, and the 13-class category rules)
and `walls_aspect.py` (vectorised wall detection).

Verification: the package in this folder re-ran the 4000 x 3974 px test area (24 hours, shadow-only, 115 s on an
RTX 6000 Ada) and reproduced the reference `Category_0_0.tif` and `Shadow_0_0.tif` of the paper bit-for-bit in all
24 bands.

## 1. Shadow-category semantics (final, 2026-06-13)

| Class | Name | Rule |
|---|---|---|
| 0 | sunlit | no shadow |
| 1 | building | shadow cast by buildings only |
| 2 | vegetation | tree-canopy shadow only |
| 3 | building + vegetation | overlap of 1 and 2 |
| 4 | linkway (LDSM) | covered-linkway roof shadow only |
| 5 | building + linkway | |
| 6 | vegetation + linkway | |
| 7 | building + vegetation + linkway | |
| 8 | arcade | walkway strip pixel shaded by the building body (the floating box base or the main building) |
| 9 | vegetation + arcade | |
| 10 | linkway + arcade | |
| 11 | vegetation + linkway + arcade | |
| 12 | building footprint | inside `BREMAIN`; excluded from shadow statistics |

Rules applied in `solweig.py` after the per-layer shadow casting:

1. `BREMAIN > 0` -> class 12.
2. Arcade classes 8-11 are restricted to the arcade strip (`ADSM > 0`): off-strip pixels that the box shades are
   building shadow (8->1, 9->3, 10->5, 11->7), because the box is part of the building.
3. Building shadow that falls on the strip is merged into the arcade family (1->8, 3->9, 5->10, 7->11) so that a
   walkway pixel sheltered by any building body counts as arcade shelter; combinations with vegetation / linkway are kept.
4. Pure vegetation / linkway shadow on the strip keeps classes 2 / 4 / 6; unshaded strip pixels stay 0.

The `Shadow_*.tif` values are physical (1 = sunlit, 0 = shaded, fractional values for vegetation transmissivity)
and are not affected by the category remapping (verified bit-for-bit).

## 2. New `thermal_comfort` arguments

```python
from solweig_gpu import thermal_comfort
thermal_comfort(
    base_path, selected_date_str,
    building_dsm_filename='SUB_SG_Polygon_DSMremain_1m.tif',   # DEM + building_remain height
    dem_filename='SUB_SG_Polygon_DEM_1m.tif',
    trees_filename='SUB_SG_Polygon_CDSMclean_1m.tif',          # canopy height above ground, nodata cleaned to 0
    ldsm_filename='SUB_SG_Polygon_LDSMpednet_1m.tif',          # NEW  covered-linkway roof height (3 m)
    arcade_filename='SUB_SG_Polygon_ADSM_1m.tif',              # NEW  arcade box top (= building height)
    arcade_base_filename='SUB_SG_Polygon_ADSMB_1m.tif',        # NEW  arcade box base (= arcade ceiling)
    building_remain_filename='SUB_SG_Polygon_BREMAIN_1m.tif',  # NEW  footprint mask -> class 12
    wallheight_filename='SUB_SG_Polygon_WALLS0_1m.tif',        # zero rasters are sufficient in shadow-only mode
    wallaspect_filename='SUB_SG_Polygon_ASPECT0_1m.tif',
    landcover_filename=None,
    tile_size=4000, overlap=100,
    use_own_met=True, own_met_file='Forcing_data/S50_Clementi Road.txt',
    start_time='2026-03-01 00:00:00', end_time='2026-03-01 23:00:00',
    save_shadow=True, shadow_category=True,   # NEW  write Shadow_*.tif and Category_*.tif
    only_shadow=True,                          # NEW  skip SVF / radiation / Tmrt (saves ~7-8 h city-wide)
    reuse_tiles=True,                          # NEW  reuse processed_inputs/ tiles when rerunning another date
    skip_sparse_tiles=True, min_building_fraction=0.01, min_tree_fraction=0.01, min_ldsm_fraction=0.0,
    canopy_height_ratio=0.23, ldsm_bottom_height_ratio=0.90,
    shelter_transmittance=0.0, arcade_transmittance=0.0,
)
```

`arcade_filename` and `arcade_base_filename` must be given together. Wall height / aspect rasters do not influence
ground shadow, so all-zero rasters can be passed in shadow-only mode; real walls are only needed for Tmrt / UTCI
(`scripts/01_prepare_rasters/step3b_walls_parallel.py` recomputes them on 12 cores, bit-identical to the package).

## 3. Pipeline

All rasters share one grid: 1 m, EPSG:3414 (SVY21), 44,000 x 27,000 px for the whole of Singapore
(bounds 2938.6, 23953.5 - 46938.6, 50953.5). Environment B (QGIS Python with torch + GDAL) for the model run,
environment A for raster preparation and merging (see `docs/ENVIRONMENTS.md`).

| Step | Script | Purpose |
|---|---|---|
| 1 | `scripts/01_prepare_rasters/prepare_newarcade.py` | one-shot builder: rasterises `building_remain` (height) -> `DSMremain` (= DEM + height) and `BREMAIN`; arcade polygons (`bld_h`, `arc_h`) -> `ADSM` / `ADSMB` (thin-box protection); covered linkways -> `LDSMpednet` (3 m); cleans the canopy DSM -> `CDSMclean`; writes zero `WALLS0` / `ASPECT0` |
| 1b | `step3n_city_dsm_remain.py`, `step3o_ldsm_pednet.py`, `step3e_prepare_city_sg.py`, `step3i_rasterize_bremain.py`, `step3k_build_dsm_remain.py`, `step3t_rebuild_aligned.py` | the individual builders that `prepare_newarcade.py` consolidates (kept for the test-area workflow) |
| 2 | `scripts/02_run_shadow/run_newarcade.py` | whole-city shadow + category run (`thermal_comfort`, 77 tiles of 4000 px with a 100 px halo, sparse sea tiles skipped) |
| 2b | `scripts/02_run_shadow/driver_tilewise.py` / `driver_tilewise_dates.py` | tile-wise driver that reuses `processed_inputs/`, runs each tile in a subprocess with a 10 min timeout and skips finished tiles; the dated version takes `SW_DATE`, `SW_OUT`, `SW_MET` from the environment |
| 2c | `scripts/02_run_shadow/orchestrate_dates.ps1` | PowerShell orchestrator for the four equinox / solstice runs (driver -> merge -> clean-up per date) |
| 2d | `scripts/02_run_shadow/step3c_b10_final.py` | test-area run (4000 x 3974 px) with the same configuration |
| 3 | `scripts/03_merge_and_stats/merge_newarcade.py`, `merge_dates.py`, `step3y_merge_tiles.py`, `step3z_extract_2pm.py`, `extract_h11_h15.py` | mosaic the tile cores (halo dropped) into 24-band city rasters and extract single-hour rasters (11:00-15:00, palette embedded in the category raster) |
| 3b | `step3w_city_stats.py`, `step3x_land_stats.py`, `step3aa_composition.py`, `step3ac_share_with_pies.py`, `step3g_*` | hourly class counts (land mask = DEM > 0, denominator excludes class 12), shadow-share curves and previews |

Typical commands (Windows, from this folder):

```bat
:: 1. build the input rasters (environment A)
python scripts\01_prepare_rasters\prepare_newarcade.py

:: 2. run the shadow model (environment B)
scripts\02_run_shadow\run_qgis.bat scripts\02_run_shadow\run_newarcade.py

:: 2b. or tile-wise with timeout / resume
set SW_DATE=2026-06-21
set SW_OUT=D:\data\output_folder_2026-06-21
set SW_MET=D:\data\Forcing_data\S50_Clementi Road_2026-06-21_summer_solstice_doy172.txt
scripts\02_run_shadow\run_qgis.bat scripts\02_run_shadow\driver_tilewise_dates.py

:: 3. mosaic + hourly extracts + statistics (environment A)
python scripts\03_merge_and_stats\merge_dates.py D:\data\output_folder_2026-06-21 D:\data\merge_images\2026-06-21
python scripts\03_merge_and_stats\step3x_land_stats.py
```

Runtime on one RTX 6000 Ada: 98.9 min for the 77 Singapore tiles in shadow-only mode (2026-03-01 run);
about 165 s per tile in the dated tile-wise runs (about 3.6 h per date including preprocessing reuse and merging).

## 4. Notebook

`notebooks/ADSM_Arcade_Shadow_Standalone.ipynb` is the stand-alone version of the whole workflow
(raster preparation -> `thermal_comfort` -> class statistics -> preview) that was used for the production run.
It initialises the QGIS / UMEP Python environment in its first cells and expects the `solweig_gpu` package of this
folder on `sys.path` (`local_repo` variable).

## 5. Outputs

| Output | Format | Notes |
|---|---|---|
| `output_folder/<x>_<y>/Shadow_<x>_<y>.tif` | float32, 24 bands (band = hour + 1) | 1 = sunlit, 0 = shaded |
| `output_folder/<x>_<y>/Category_<x>_<y>.tif` | uint8, 24 bands | classes 0-12 |
| `merge_images/Shadow/Shadow_merged.tif`, `merge_images/Category/Category_merged.tif` | 44,000 x 27,000 x 24 | tile cores only, no seams |
| `merge_images/Category_2pm_h14.tif`, `Shadow_2pm_h14.tif` (and 11:00-15:00) | single band | used by Modules 4a and 5 |
| `step3_city_class_counts_land.csv` | CSV | hourly pixel counts per class over land (DEM > 0) |

Whole-city result for 2026-03-01 at 14:00 (land, denominator classes 0-11): 38.3 % of the ground is shaded;
class 8 (arcade-sheltered walkway) covers 132,170 px and the linkway family (4-7) about 1.58 million px.

## 6. Path configuration

The scripts keep the absolute paths of the original workstation (`D:\Claude\SVI_FFW\TIF_shadow_newarcade`,
`D:\Claude\SVI_FFW\Shp\SG\...`). Each script defines its paths in the constants at the top of the file
(`BASE`, `TIF`, `NEW`, `MET`, `GPKG`, ...); edit those lines before running. `run_qgis.bat` expects QGIS 3.40 LTR
in `C:\Program Files\QGIS 3.40.15`.

## 7. Inputs

See `INPUT_DATA.md` for the complete input inventory (rasters, vectors, meteorological forcing) and the
`sample_data/forcing` folder for the five UMEP-format forcing files used in the paper.

## 8. Licence

GPL-3.0, inherited from SOLWEIG-GPU (see `LICENSE`). Please cite SOLWEIG-GPU (Kamath et al., 2026) and the original
SOLWEIG model (Lindberg et al., 2008) when using this module; upstream documentation is kept in `upstream_docs/`.
