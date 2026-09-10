# Provenance

Every file of this repository was copied from the working folders of the original workstation on 2026-09-10.
The table gives the original location of each file (or file group). Python scripts were translated to English
(comments, docstrings and message strings only) and verified to be identical in logic to the originals by an
abstract-syntax-tree comparison; the few files that were edited beyond that are listed first.

## Intentional edits beyond translation

| Repository file | Edit |
|---|---|
| `2_Covered Linkway Extraction/train/train_autonomous_v2.py` | sys.path insert made repo-relative (imports the shared module from the same folder) |
| `2_Covered Linkway Extraction/train/train_autonomous_tversky.py` | sys.path insert made repo-relative |
| `2_Covered Linkway Extraction/inference/inference_autonomous_island.py` | two machine-specific sys.path inserts replaced by one repo-relative insert of ../train |
| `2_Covered Linkway Extraction/inference/inference_autonomous.py` | two machine-specific sys.path inserts replaced by one repo-relative insert of ../train |
| `2_Covered Linkway Extraction/evaluation/eval_ckpt_paper.py` | two machine-specific sys.path inserts replaced by one repo-relative insert of ../train |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/extract_h11_h15.py` | removed a two-line consistency check against a paper-figure statistics file of the workstation |

## Files kept verbatim with their original Chinese UI strings

The three web-tool generator scripts contain the Chinese user-interface strings of the bilingual application
inside string literals; only their comments and docstrings were translated:

- `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/make_nav_app_maplibre.py`
- `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/make_paper_version.py`
- `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/make_en_navapp.py`

## Origin of the repository files

### 1_SOLWEIG_GPU (ADSM LDSM)

| Repository file | Original location | Note |
|---|---|---|
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/Tgmaps_v1.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\Tgmaps_v1.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/__init__.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\__init__.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/calculate_utci.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\calculate_utci.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/cli.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\cli.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/preprocessor.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\preprocessor.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/shadow.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\shadow.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/solweig.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\solweig.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/solweig_gpu.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\solweig_gpu.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/solweig_gpu_gui.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\solweig_gpu_gui.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/sun_position.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\sun_position.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/utci_process.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\utci_process.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/walls_aspect.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\walls_aspect.py` | modified SOLWEIG-GPU package (LDSM + ADSM + 13-class shadow category) |
| `1_SOLWEIG_GPU (ADSM LDSM)/solweig_gpu/landcoverclasses_2016a.txt` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\solweig_gpu\landcoverclasses_2016a.txt` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/setup.py` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\setup.py` | upstream packaging files (SOLWEIG-GPU v1.2.21) |
| `1_SOLWEIG_GPU (ADSM LDSM)/pyproject.toml` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\pyproject.toml` | upstream packaging files (SOLWEIG-GPU v1.2.21) |
| `1_SOLWEIG_GPU (ADSM LDSM)/requirements.txt` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\requirements.txt` | upstream packaging files (SOLWEIG-GPU v1.2.21) |
| `1_SOLWEIG_GPU (ADSM LDSM)/MANIFEST.in` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\MANIFEST.in` | upstream packaging files (SOLWEIG-GPU v1.2.21) |
| `1_SOLWEIG_GPU (ADSM LDSM)/LICENSE` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\LICENSE` | upstream packaging files (SOLWEIG-GPU v1.2.21) |
| `1_SOLWEIG_GPU (ADSM LDSM)/upstream_docs/README_upstream_SOLWEIG-GPU.md` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\README.md` | upstream SOLWEIG-GPU README |
| `1_SOLWEIG_GPU (ADSM LDSM)/upstream_docs/input_data.md` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\docs\input_data.md` | upstream SOLWEIG-GPU documentation |
| `1_SOLWEIG_GPU (ADSM LDSM)/upstream_docs/outputs.md` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\docs\outputs.md` | upstream SOLWEIG-GPU documentation |
| `1_SOLWEIG_GPU (ADSM LDSM)/upstream_docs/configuration.md` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\docs\configuration.md` | upstream SOLWEIG-GPU documentation |
| `1_SOLWEIG_GPU (ADSM LDSM)/upstream_docs/quickstart.md` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\docs\quickstart.md` | upstream SOLWEIG-GPU documentation |
| `1_SOLWEIG_GPU (ADSM LDSM)/upstream_docs/installation.md` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\docs\installation.md` | upstream SOLWEIG-GPU documentation |
| `1_SOLWEIG_GPU (ADSM LDSM)/upstream_docs/api_reference.md` | `D:\Claude\SVI_FFW\Module\SOLWEIG-GPU\docs\api_reference.md` | upstream SOLWEIG-GPU documentation |
| `1_SOLWEIG_GPU (ADSM LDSM)/notebooks/ADSM_Arcade_Shadow_Standalone.ipynb` | `D:\Claude\SVI_FFW\Module\01-SG_ADSM_Arcade.ipynb` | stand-alone ADSM workflow notebook |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/prepare_newarcade.py` | `D:\Claude\SVI_FFW\output\step3_adsm\prepare_newarcade.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/step3e_prepare_city_sg.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3e_prepare_city_sg.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/step3n_city_dsm_remain.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3n_city_dsm_remain.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/step3o_ldsm_pednet.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3o_ldsm_pednet.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/step3k_build_dsm_remain.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3k_build_dsm_remain.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/step3i_rasterize_bremain.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3i_rasterize_bremain.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/step3t_rebuild_aligned.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3t_rebuild_aligned.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/step3b_walls_parallel.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3b_walls_parallel.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/step3a_rasterize_adsm.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3a_rasterize_adsm.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/01_prepare_rasters/step3a_v2_prepare_arcade.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3a_v2_prepare_arcade.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/02_run_shadow/run_newarcade.py` | `D:\Claude\SVI_FFW\output\step3_adsm\run_newarcade.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/02_run_shadow/step3f_run_city_sg.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3f_run_city_sg.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/02_run_shadow/step3c_b10_final.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3c_b10_final.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/02_run_shadow/driver_tilewise.py` | `D:\Claude\SVI_FFW\output\step3_adsm\driver_tilewise.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/02_run_shadow/driver_tilewise_dates.py` | `D:\Claude\SVI_FFW\output\step3_adsm\driver_tilewise_dates.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/02_run_shadow/run_qgis.bat` | `D:\Claude\SVI_FFW\output\step3_adsm\run_qgis.bat` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/02_run_shadow/orchestrate_dates.ps1` | `D:\Claude\SVI_FFW\output\step3_adsm\_orchestrate_dates.ps1` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/merge_newarcade.py` | `D:\Claude\SVI_FFW\output\step3_adsm\merge_newarcade.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/merge_dates.py` | `D:\Claude\SVI_FFW\output\step3_adsm\merge_dates.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/step3y_merge_tiles.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3y_merge_tiles.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/step3z_extract_2pm.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3z_extract_2pm.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/step3w_city_stats.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3w_city_stats.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/step3x_land_stats.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3x_land_stats.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/step3aa_composition.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3aa_composition.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/step3ac_share_with_pies.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3ac_share_with_pies.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/step3g_c_shadow_previews.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3g_c_shadow_previews.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/step3g_make_preview.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3g_make_preview.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/step3g_b_preview_13cls.py` | `D:\Claude\SVI_FFW\output\step3_adsm\step3g_b_preview_13cls.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/scripts/03_merge_and_stats/extract_h11_h15.py` | `D:\Claude\SVI_FFW\TIF_shadow_newarcade\merge_images\_extract_h12_h13\extract_h11_h15.py` |  |
| `1_SOLWEIG_GPU (ADSM LDSM)/sample_data/forcing/S50_Clementi Road.txt` | `D:\Claude\SVI_FFW\TIF_shadow_newarcade\Forcing_data\S50_Clementi Road.txt` | UMEP-format meteorological forcing (S50 Clementi Road) |
| `1_SOLWEIG_GPU (ADSM LDSM)/sample_data/forcing/S50_Clementi Road_2026-03-20_spring_equinox_doy79.txt` | `D:\Claude\SVI_FFW\TIF_shadow_newarcade\Forcing_data\S50_Clementi Road_2026-03-20_spring_equinox_doy79.txt` | UMEP-format meteorological forcing (S50 Clementi Road) |
| `1_SOLWEIG_GPU (ADSM LDSM)/sample_data/forcing/S50_Clementi Road_2026-06-21_summer_solstice_doy172.txt` | `D:\Claude\SVI_FFW\TIF_shadow_newarcade\Forcing_data\S50_Clementi Road_2026-06-21_summer_solstice_doy172.txt` | UMEP-format meteorological forcing (S50 Clementi Road) |
| `1_SOLWEIG_GPU (ADSM LDSM)/sample_data/forcing/S50_Clementi Road_2026-09-23_autumn_equinox_doy266.txt` | `D:\Claude\SVI_FFW\TIF_shadow_newarcade\Forcing_data\S50_Clementi Road_2026-09-23_autumn_equinox_doy266.txt` | UMEP-format meteorological forcing (S50 Clementi Road) |
| `1_SOLWEIG_GPU (ADSM LDSM)/sample_data/forcing/S50_Clementi Road_2026-12-22_winter_solstice_doy356.txt` | `D:\Claude\SVI_FFW\TIF_shadow_newarcade\Forcing_data\S50_Clementi Road_2026-12-22_winter_solstice_doy356.txt` | UMEP-format meteorological forcing (S50 Clementi Road) |

### 2_Covered Linkway Extraction

| Repository file | Original location | Note |
|---|---|---|
| `2_Covered Linkway Extraction/train/train_geosam_topolora_linkway.py` | `D:\Claude\GeoSAM-TopoLoRA\train_geosam_topolora_linkway.py` | shared module: LoRA injection, CLIP text prompt, point-prompted stage |
| `2_Covered Linkway Extraction/train/train_autonomous_v2.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\train_autonomous_v2.py` |  |
| `2_Covered Linkway Extraction/train/train_autonomous_tversky.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\train_autonomous_tversky.py` |  |
| `2_Covered Linkway Extraction/train/thermal_watchdog.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\thermal_watchdog.py` |  |
| `2_Covered Linkway Extraction/inference/inference_autonomous_island.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\inference_autonomous_island.py` |  |
| `2_Covered Linkway Extraction/inference/inference_autonomous.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\inference_autonomous.py` |  |
| `2_Covered Linkway Extraction/postprocess/postprocess_island.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\postprocess_island.py` |  |
| `2_Covered Linkway Extraction/postprocess/vectorise_generic.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\vectorise_generic.py` |  |
| `2_Covered Linkway Extraction/postprocess/apply_pednet_filter.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\apply_pednet_filter.py` |  |
| `2_Covered Linkway Extraction/postprocess/pednet_filter_generic.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\pednet_filter_generic.py` |  |
| `2_Covered Linkway Extraction/postprocess/footpath_bridge_island.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\footpath_bridge_island.py` |  |
| `2_Covered Linkway Extraction/postprocess/postprocess.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\postprocess.py` |  |
| `2_Covered Linkway Extraction/postprocess/postprocess_close_v3.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\postprocess_close_v3.py` |  |
| `2_Covered Linkway Extraction/postprocess/material_grow.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\material_grow.py` |  |
| `2_Covered Linkway Extraction/evaluation/eval_island_vs_val.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\eval_island_vs_val.py` |  |
| `2_Covered Linkway Extraction/evaluation/eval_paper_metrics.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\eval_paper_metrics.py` |  |
| `2_Covered Linkway Extraction/evaluation/eval_ckpt_paper.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\eval_ckpt_paper.py` |  |
| `2_Covered Linkway Extraction/evaluation/eval_sam2unet_paper.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\eval_sam2unet_paper.py` |  |
| `2_Covered Linkway Extraction/evaluation/sweep_fp_filters.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\sweep_fp_filters.py` |  |
| `2_Covered Linkway Extraction/evaluation/footpath_bridge_eval.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\footpath_bridge_eval.py` |  |
| `2_Covered Linkway Extraction/evaluation/footpath_bridge_eval_lta.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\footpath_bridge_eval_lta.py` |  |
| `2_Covered Linkway Extraction/data_prep/make_fishnet_v3.py` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\make_fishnet_v3.py` |  |
| `2_Covered Linkway Extraction/baseline_sam2unet/cl_pipeline.py` | `D:\Claude\Meta-SAM2\cl_pipeline.py` | SAM2-UNet + LoRA baseline engine |
| `2_Covered Linkway Extraction/baseline_sam2unet/run_training.py` | `D:\Claude\Meta-SAM2\run_training.py` | SAM2-UNet + LoRA baseline engine |
| `2_Covered Linkway Extraction/baseline_sam2unet/run_inference_mosaic.py` | `D:\Claude\Meta-SAM2\run_inference_mosaic.py` | SAM2-UNet + LoRA baseline engine |
| `2_Covered Linkway Extraction/baseline_sam2unet/generate_masks.py` | `D:\Claude\Meta-SAM2\generate_masks.py` | SAM2-UNet + LoRA baseline engine |
| `2_Covered Linkway Extraction/baseline_sam2unet/gsltl_pipeline.py` | `D:\Claude\GeoSAM-TopoLoRA\gsltl_pipeline.py` |  |
| `2_Covered Linkway Extraction/baseline_sam2unet/covered_linkway_sam2unet_lora.ipynb` | `D:\Claude\GeoSAM-TopoLoRA\covered_linkway_geosam_topolora.ipynb` |  |
| `2_Covered Linkway Extraction/training_data/grid/tiles_fishnet.gpkg` | `D:\Claude\GeoSAM-TopoLoRA\covered Linkway\grid_v1\tiles_fishnet.gpkg` | tile grid (train/val split) |
| `2_Covered Linkway Extraction/training_data/grid/tiles_fishnet_train.gpkg` | `D:\Claude\GeoSAM-TopoLoRA\covered Linkway\grid_v1\tiles_fishnet_train.gpkg` | tile grid (train/val split) |
| `2_Covered Linkway Extraction/training_data/grid/tiles_fishnet_val.gpkg` | `D:\Claude\GeoSAM-TopoLoRA\covered Linkway\grid_v1\tiles_fishnet_val.gpkg` | tile grid (train/val split) |
| `2_Covered Linkway Extraction/training_data/grid/tiles_meta.csv` | `D:\Claude\GeoSAM-TopoLoRA\covered Linkway\grid_v1\tiles_meta.csv` | tile grid (train/val split) |
| `2_Covered Linkway Extraction/checkpoints/geosam_topolora_autonomous_lean_dice0.7138.pth` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\runs\autonomous_tversky\geosam_topolora_autonomous_lean_dice0.7138.pth` | final lean checkpoint (LoRA + mask decoder + projection) |
| `2_Covered Linkway Extraction/checkpoints/clip_linkway_emb.pth` | `D:\Claude\GeoSAM-TopoLoRA\checkpoints\clip_linkway_emb.pth` | cached CLIP text embedding for 'Covered Linkway' |
| `2_Covered Linkway Extraction/checkpoints/train_log_focal_tversky.json` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\runs\autonomous_tversky\train_log.json` |  |
| `2_Covered Linkway Extraction/checkpoints/train_log_focal_tversky_15ep.json` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\runs\autonomous_tversky_15ep\train_log.json` |  |
| `2_Covered Linkway Extraction/checkpoints/train_log_v3_structure.json` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\runs\autonomous\train_log.json` |  |
| `2_Covered Linkway Extraction/checkpoints/train_log_ablation_no_cldice.json` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\runs\autonomous_nocldice\train_log.json` |  |
| `2_Covered Linkway Extraction/report/performance_table.csv` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\FINAL_REPORT\performance_table.csv` |  |
| `2_Covered Linkway Extraction/report/ablation_cldice.txt` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\FINAL_REPORT\ablation_cldice.txt` |  |
| `2_Covered Linkway Extraction/report/sam2unet_val_metrics.txt` | `D:\Claude\GeoSAM-TopoLoRA\outputs\GeoSAM-backup\FINAL_REPORT\sam2unet_val_metrics.txt` |  |
| `2_Covered Linkway Extraction/training_data/masks/train/` (808 files) | `D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks\train` | binary mask (0/255) |
| `2_Covered Linkway Extraction/training_data/masks/val/` (303 files) | `D:\Claude\GeoSAM-TopoLoRA\covered Linkway\masks\val` | binary mask (0/255) |
| `2_Covered Linkway Extraction/training_data/labels_json/train/` (808 files) | `D:\Claude\GeoSAM-TopoLoRA\covered Linkway\json\train` | LabelMe polygon annotation |
| `2_Covered Linkway Extraction/training_data/labels_json/val/` (303 files) | `D:\Claude\GeoSAM-TopoLoRA\covered Linkway\json\val` | LabelMe polygon annotation |

### 3_Arcade Extraction and Projection

| Repository file | Original location | Note |
|---|---|---|
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/detection/assign_tiers.py` | `D:\Claude\SVI_FFW\output\step1_SVI\detection\assign_tiers.py` | unified reference implementation |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/detection/detect_arcade.py` | `D:\Claude\SVI_FFW\output\step1_SVI\detection\detect_arcade.py` | unified reference implementation |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/train_probes.py` | `D:\Claude\SVI_FFW\output\step1_SVI\stageA_l14\train_probes.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/validate_broad.py` | `D:\Claude\SVI_FFW\output\step1_SVI\stageA_l14\validate_broad.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/validate_human_gt.py` | `D:\Claude\SVI_FFW\output\step1_SVI\stageA_l14\validate_human_gt.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/train_strict_probe.py` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\train_strict_probe.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/train_final_colonnade.py` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\train_final_colonnade.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/prelabel_shophouse.py` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\prelabel_shophouse.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/prelabel_negatives.py` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\prelabel_negatives.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/gen_facing_neg_depth.py` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\gen_facing_neg_depth.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/build_sg_validation.py` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1c_sg_val\build_sg_validation.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/step_a_bearing.py` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1c_sg_val\step_a_bearing.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/step_b_clip.py` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1c_sg_val\step_b_clip.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/eval_clip_on_official.py` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1c_sg_val\eval_clip_on_official.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/build_clean_gt.py` | `D:\Claude\SVI_FFW\output\step1_SVI\sg_improve\build_clean_gt.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/benchmark_methods.py` | `D:\Claude\SVI_FFW\output\step1_SVI\sg_improve\benchmark_methods.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/bench_compare.py` | `D:\Claude\SVI_FFW\output\_bench_compare.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/phase0_build_samples.py` | `D:\Claude\SVI_FFW\output\phase0_build_samples.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/view_selection.py` | `D:\Claude\SVI_FFW\output\view_selection.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/probe_training/tier_breakdown.py` | `D:\Claude\SVI_FFW\output\tier_breakdown.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/geometry_first/run_facing_precise.py` | `D:\Claude\SVI_FFW\output\step2_detection_geomfirst\run_facing_precise.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/geometry_first/detect_precise.py` | `D:\Claude\SVI_FFW\output\step2_detection_geomfirst\detect_precise.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/geometry_first/project_precise.py` | `D:\Claude\SVI_FFW\output\step2_detection_geomfirst\project_precise.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/geometry_first/compare_match.py` | `D:\Claude\SVI_FFW\output\step2_detection_geomfirst\compare_match.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/geometry_first/viz_added_map.py` | `D:\Claude\SVI_FFW\output\step2_detection_geomfirst\viz_added_map.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/geometry_first/make_figs_v2.py` | `D:\Claude\SVI_FFW\output\step2_detection_geomfirst\make_figs_v2.py` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/models/sg_probe.joblib` | `D:\Claude\SVI_FFW\output\step1_SVI\stageA_l14\sg_probe.joblib` | broad-screen probes + per-tier thresholds |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/models/bo_probe.joblib` | `D:\Claude\SVI_FFW\output\step1_SVI\stageA_l14\bo_probe.joblib` | broad-screen probes + per-tier thresholds |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/models/thresholds.json` | `D:\Claude\SVI_FFW\output\step1_SVI\stageA_l14\thresholds.json` | broad-screen probes + per-tier thresholds |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/models/sg_colonnade_final.joblib` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\sg_colonnade_final.joblib` | Singapore colonnade probes |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/models/sg_strict_probe.joblib` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\sg_strict_probe.joblib` | Singapore colonnade probes |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/models/sg_colonnade_probe.joblib` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\sg_colonnade_probe.joblib` | Singapore colonnade probes |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/models/colonnade_threshold.json` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\colonnade_threshold.json` | Singapore colonnade probes |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/strict_gt/prelabel_500.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\prelabel_500.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/strict_gt/prelabel_neg.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\prelabel_neg.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/strict_gt/prelabel_neg2.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\prelabel_neg2.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/strict_gt/prelabel_neg3.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\prelabel_neg3.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/strict_gt/prelabel_neg4.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\prelabel_neg4.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/strict_gt/strict_train_scored.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\strict_gt\strict_train_scored.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase1b_sg/human_gt.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1b_sg\human_gt.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase1b_sg/human_gt_broad.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1b_sg\human_gt_broad.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase1b_sg/gt_labels.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1b_sg\gt_labels.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase1b_sg/compare_results.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1b_sg\compare_results.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/human_gt_validation.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\stageA_l14\human_gt_validation.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/gt_candidates.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\sg_improve\gt_candidates.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase1c_sg_val/sg_val_positive.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1c_sg_val\sg_val_positive.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase1c_sg_val/sg_val_negative.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1c_sg_val\sg_val_negative.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase1c_sg_val/overlap_stats.txt` | `D:\Claude\SVI_FFW\output\step1_SVI\phase1c_sg_val\overlap_stats.txt` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase0_samples/bo_positive.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\phase0_samples\bo_positive.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase0_samples/bo_negative.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\phase0_samples\bo_negative.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/phase0_samples/sg_test_pts.csv` | `D:\Claude\SVI_FFW\output\step1_SVI\phase0_samples\sg_test_pts.csv` |  |
| `3_Arcade Extraction and Projection/a_GSV Image Arcade Detection/labels/bench_results.csv` | `D:\Claude\SVI_FFW\output\_bench_results.csv` |  |
| `3_Arcade Extraction and Projection/b_Arcade Projection to Building Footprint/scripts/step2_project.py` | `D:\Claude\SVI_FFW\output\step2_projection\step2_project.py` |  |
| `3_Arcade Extraction and Projection/b_Arcade Projection to Building Footprint/scripts/step2_edge_ffw.py` | `D:\Claude\SVI_FFW\output\step2_projection\step2_edge_ffw.py` |  |
| `3_Arcade Extraction and Projection/b_Arcade Projection to Building Footprint/scripts/step2_buffer_split.py` | `D:\Claude\SVI_FFW\output\step2_projection\step2_buffer_split.py` |  |
| `3_Arcade Extraction and Projection/b_Arcade Projection to Building Footprint/scripts/step2_merge.py` | `D:\Claude\SVI_FFW\output\step2_projection\step2_merge.py` |  |
| `3_Arcade Extraction and Projection/b_Arcade Projection to Building Footprint/scripts/eval_thresholds.py` | `D:\Claude\SVI_FFW\output\step2_projection\eval_thresholds.py` |  |

### 4_Pedestrian Flow Mapping

| Repository file | Original location | Note |
|---|---|---|
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/01_prep/make_prep.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\_make_prep.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/02_edge_shade/step4_4a_city_building_shadow.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4a_city_building_shadow.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/02_edge_shade/step4_4a_bldtree_shadow.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4a_bldtree_shadow.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/02_edge_shade/step4_4b_city_edge_shade.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4b_city_edge_shade.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/02_edge_shade/step4_4b_esn.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4b_esn.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/02_edge_shade/step4_edge_facility.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_edge_facility.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/02_edge_shade/step4_edge_class_1m.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_edge_class_1m.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/02_edge_shade/step4_network_split_1m.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_network_split_1m.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/02_edge_shade/edge_px_permetre.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\edge_px_permetre.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/03_routing_flow/step4_4c_city_routing.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4c_city_routing.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/03_routing_flow/step4_4c_orig_flow.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4c_orig_flow.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/03_routing_flow/step4_4c_orig_coolflow.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4c_orig_coolflow.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/03_routing_flow/step4_4e_flow_detour.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4e_flow_detour.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/03_routing_flow/step4_4f_flow_lam.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4f_flow_lam.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/scripts/03_routing_flow/step4_4d_city_viz.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\step4_4d_city_viz.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/make_nav_app_maplibre.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\make_nav_app_maplibre.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/make_nav_app_mobile.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\make_nav_app_mobile.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/make_paper_version.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\_make_paper_version.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/make_en_navapp.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\_make_en_navapp.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/make_hourly_cache.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\_make_hourly_cache.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/make_tiles.py` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\_make_tiles.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/tile_rasters.py` | `D:\Claude\SVI_FFW\output\step4_network\tile_rasters.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/step4_demo_scenarios.py` | `D:\Claude\SVI_FFW\output\step4_network\step4_demo_scenarios.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/step4_merge_saved_demo.py` | `D:\Claude\SVI_FFW\output\step4_network\step4_merge_saved_demo.py` |  |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/dist/nav_app_paper_en.html.gz` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\webapp\nav_app_paper_en.html` | ShadeWalk web tool, paper routing build, English (gzip-compressed copy) |
| `4_Pedestrian Flow Mapping/a_OD Coolest-Route Algorithm (Route Cost Calculation)/webapp/dist/demo_scenarios.json` | `D:\Claude\SVI_FFW\output\step5_nav_webapp\_demo_scenarios.json` |  |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/Constants.py` | `D:\Claude\UNA\Patronage_Flow\Constants.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/Flow_Computation.py` | `D:\Claude\UNA\Patronage_Flow\Flow_Computation.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/Main.py` | `D:\Claude\UNA\Patronage_Flow\Main.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/Main_dualpass.py` | `D:\Claude\UNA\Patronage_Flow\Main_dualpass.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/Network.py` | `D:\Claude\UNA\Patronage_Flow\Network.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/__init__.py` | `D:\Claude\UNA\Patronage_Flow\__init__.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/build_occupancy_table.py` | `D:\Claude\UNA\Patronage_Flow\build_occupancy_table.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/export_building_hourly.py` | `D:\Claude\UNA\Patronage_Flow\export_building_hourly.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/export_station_hourly.py` | `D:\Claude\UNA\Patronage_Flow\export_station_hourly.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/run_2025_annual_total.py` | `D:\Claude\UNA\Patronage_Flow\run_2025_annual_total.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/run_2025_quarterly.py` | `D:\Claude\UNA\Patronage_Flow\run_2025_quarterly.py` | Patronage_Flow package |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/lookup/mrt_code_to_name.csv` | `D:\Claude\UNA\Patronage_Flow\lookup\mrt_code_to_name.csv` | lookup tables (MRT codes, occupancy density) |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/lookup/occupancy_density.csv` | `D:\Claude\UNA\Patronage_Flow\lookup\occupancy_density.csv` | lookup tables (MRT codes, occupancy density) |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/Patronage_Flow/lookup/occupancy_density.parquet` | `D:\Claude\UNA\Patronage_Flow\lookup\occupancy_density.parquet` | lookup tables (MRT codes, occupancy density) |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/notebooks/Patronage_Flow_Pipeline.ipynb` | `D:\Claude\UNA\Patronage_Flow_Pipeline.ipynb` |  |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/tools/patch_madina.py` | `D:\Claude\UNA\_patch_madina.py` | pandas 3.0 compatibility patch for madina |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/tools/madina_smoke.py` | `D:\Claude\UNA\_madina_smoke.py` |  |
| `4_Pedestrian Flow Mapping/b_Pedestrian OD Flow Assignment (Gravity and Huff model)/docs/DEVELOPMENT_NOTES.md` | `D:\Claude\UNA\CLAUDE.md` | development notes (module architecture, gotchas) |

### 5_OSM Network Reconstruction

| Repository file | Original location | Note |
|---|---|---|
| `5_OSM Network Reconstruction/scripts/step4_1_city.py` | `D:\Claude\SVI_FFW\output\step4_network\step4_1_city.py` |  |
| `5_OSM Network Reconstruction/scripts/step4_2a_fac.py` | `D:\Claude\SVI_FFW\output\step4_network\step4_2a_fac.py` |  |
| `5_OSM Network Reconstruction/scripts/step4_2b_osm.py` | `D:\Claude\SVI_FFW\output\step4_network\step4_2b_osm.py` |  |
| `5_OSM Network Reconstruction/scripts/step4_3a_cut_osm.py` | `D:\Claude\SVI_FFW\output\step4_network\step4_3a_cut_osm.py` |  |
| `5_OSM Network Reconstruction/scripts/step4_3b_link.py` | `D:\Claude\SVI_FFW\output\step4_network\step4_3b_link.py` |  |
| `5_OSM Network Reconstruction/scripts/step4_4_global.py` | `D:\Claude\SVI_FFW\output\step4_network\step4_4_global.py` |  |
| `5_OSM Network Reconstruction/scripts/step4_analyze.py` | `D:\Claude\SVI_FFW\output\step4_network\step4_analyze.py` |  |

### docs

| Repository file | Original location | Note |
|---|---|---|
| `docs/core_workflow.png` | `C:\Users\City Syntax Lab\AppData\Local\Temp\claude\D--Claude-ShadeWalk\836ce0db-5a38-4784-ae86-72e025ad2066\scratchpad\fig\core_workflow_4col_hi.png` | rendered from paper_figures core_workflow_4col.emf (2026-09-07) |
| `docs/core_workflow.svg` | `C:\Users\City Syntax Lab\AppData\Local\Temp\claude\D--Claude-ShadeWalk\836ce0db-5a38-4784-ae86-72e025ad2066\scratchpad\fig\core_workflow_4col_from_emf.svg` | rendered from paper_figures core_workflow_4col.emf (2026-09-07) |

