# Module 2 - Covered-linkway extraction with GeoSAM + TopoLoRA

Extracts the roofs of covered linkways (pedestrian shelters) for the whole of Singapore from 0.3 m satellite
imagery and delivers a binary raster plus polygons that feed the LDSM layer of Module 1 and the facility
centrelines of Module 5.

## 1. Method

| Component | Choice | Origin |
|---|---|---|
| image encoder | SAM ViT-H (`sam_vit_h_4b8939.pth`), frozen | Segment Anything (Meta, Apache-2.0) |
| parameter-efficient adaptation | LoRA rank 16, alpha 32 injected into every `attn.qkv` of the 32 encoder blocks | TopoLoRA recipe (LoRA + clDice) |
| prompt | one CLIP text embedding of "Covered Linkway" (`Linear(512 -> 256)` dense prompt token); **no point / box prompts** ("autonomous" variant) | GeoSAM text prompting (Rafi et al., 2024, MIT) |
| decoder | SAM prompt encoder (frozen, called without prompts) + SAM mask decoder (trainable) | |
| loss (final) | Focal-Tversky (alpha 0.3 / beta 0.7 / gamma 4/3) + 0.5 BCE + 0.3 clDice | clDice: Shit et al., CVPR 2021 (MIT) |
| trainable parameters | 6.8 M of 643.8 M (1.06 %) | |
| tiles | 1024 x 1024 px at 0.3 m, sliding window with 128 px overlap | |

Training stages and the effect of each change (all metrics on the same 303 validation tiles, threshold 0.5):

| Stage | Change | Model F1 | Model IoU | clDice |
|---|---|---|---|---|
| SAM2-UNet + LoRA baseline (`baseline_sam2unet/`) | SAM2.1 Hiera-L + LoRA + UNet decoder, BCE + IoU | 0.457 | 0.296 | 0.627 |
| GeoSAM + LoRA, no clDice (ablation) | `--cldice_weight 0` | 0.416 | 0.263 | 0.503 |
| v3 "structure" (`train_autonomous_v2.py`) | text-only prompting, weighted BCE + IoU + 0.3 clDice | 0.501 | 0.334 | 0.635 |
| **Focal-Tversky (final, `train_autonomous_tversky.py`)** | FP-penalising loss | **0.714** | **0.555** | **0.782** |

Deployment on the island (post-processed products, sampled at the validation windows):

| Product | P | R | F1 | IoU | clDice | Betti-0 | polygons | km2 |
|---|---|---|---|---|---|---|---|---|
| Tversky raw island | 0.694 | 0.692 | 0.693 | 0.530 | 0.753 | 5.83 | 10,064 | 2.38 |
| + pedestrian-network filter (5 m) | 0.719 | 0.655 | 0.686 | 0.522 | 0.745 | 1.38 | 6,323 | 1.64 |
| **+ footpath-corridor bridging (FINAL)** | 0.715 | 0.659 | 0.686 | 0.522 | 0.744 | 1.43 | 6,148 | 1.66 |

The full report with the ablation and the metric definitions is in `report/REPORT.md`
(`report/performance_table.csv` is the machine-readable version).

## 2. Folder layout

| Folder | Content |
|---|---|
| `train/` | `train_geosam_topolora_linkway.py` (shared module: LoRA injection, CLIP text-embedding cache, point-prompted stage-1 trainer), `train_autonomous_v2.py` (stage 2, structure loss, lean checkpoints), `train_autonomous_tversky.py` (final trainer, `--loss tversky`), `thermal_watchdog.py` (GPU temperature guard used on the workstation) |
| `inference/` | `inference_autonomous_island.py` (streaming sliding-window inference for very large GeoTIFFs, writes tiles directly), `inference_autonomous.py` (in-memory version for an AOI) |
| `postprocess/` | `postprocess_island.py` / `vectorise_generic.py` (streaming vectorisation, min area 25 m2), `apply_pednet_filter.py` / `pednet_filter_generic.py` (keep polygons within 5 m of the pedestrian network), `footpath_bridge_island.py` (F1-safe gap bridging inside the LTA footpath corridor), `postprocess.py`, `postprocess_close_v3.py`, `material_grow.py` (morphological closing and material-guided growing; tested and rejected, see report) |
| `evaluation/` | `eval_ckpt_paper.py` (model metrics on the val tiles), `eval_island_vs_val.py`, `eval_paper_metrics.py` (island products vs val tiles: P/R/F1/IoU, clDice, relaxed F1, Betti-0), `eval_sam2unet_paper.py` (baseline), `sweep_fp_filters.py`, `footpath_bridge_eval*.py` |
| `data_prep/` | `make_fishnet_v3.py` (tile grid with train / val / blank labels) |
| `training_data/` | tile grid, LabelMe annotations and binary masks of the 808 train + 303 val tiles (see `training_data/README.md`) |
| `checkpoints/` | final lean checkpoint (`geosam_topolora_autonomous_lean_dice0.7138.pth`, 27 MB: LoRA + mask decoder + projection), cached CLIP text embedding, per-epoch training logs |
| `baseline_sam2unet/` | SAM2-UNet + LoRA baseline engine (`cl_pipeline.py`), runners and notebook |
| `report/` | final report, performance table, ablation output |

## 3. Running

Environment C (`sam2`, see `docs/ENVIRONMENTS.md`). Download `sam_vit_h_4b8939.pth` first.

```bat
:: train the final model (about 9 min per epoch on an RTX 6000 Ada; 20 epochs by default)
python train\train_autonomous_tversky.py --loss tversky --tv_alpha 0.3 --tv_beta 0.7 --tv_gamma 1.3333 ^
    --cldice_weight 0.3 --pos_weight 30 --lr 5e-5 --batch_size 2 --epochs 20 ^
    --images_dir <tiles>\images --masks_dir <tiles>\masks --sam_ckpt <path>\sam_vit_h_4b8939.pth ^
    --text_cache checkpoints\clip_linkway_emb.pth --save_dir runs\tversky

:: full-island inference (about 3.5 h for 179,096 x 115,025 px)
python inference\inference_autonomous_island.py --input_tif <imagery>.tif --sam_ckpt <path>\sam_vit_h_4b8939.pth ^
    --trained_ckpt checkpoints\geosam_topolora_autonomous_lean_dice0.7138.pth --text_emb checkpoints\clip_linkway_emb.pth ^
    --output_tif out\covered_linkway_SG_island_tv.tif --boundary_shp <Island_boarder.shp> --building_gpkg <SG_Building_SVY21.gpkg> --threshold 0.5

:: vectorise (min area 25 m2), pedestrian-network soft constraint (5 m), footpath-corridor bridging
python postprocess\vectorise_generic.py --src_tif out\covered_linkway_SG_island_tv.tif --out_gpkg out\covered_linkway_SG_island_tv.gpkg
python postprocess\pednet_filter_generic.py --src_gpkg out\covered_linkway_SG_island_tv.gpkg --src_tif out\covered_linkway_SG_island_tv.tif ^
    --out_gpkg out\covered_linkway_SG_island_tv_pednet.gpkg --out_tif out\covered_linkway_SG_island_tv_pednet.tif --buffer 5 --pednet <pedestrian_network_filtered.gpkg>
python postprocess\footpath_bridge_island.py        (edit SRC_TIF / PEDN / OUT_* constants at the top)

:: validate against the 303 validation tiles
python evaluation\eval_paper_metrics.py
```

`run_full_island.bat` chains the three deployment steps with the arguments used for the paper.

Inference details: sliding window 1024 px / 128 px overlap, sigmoid threshold 0.5, clip to the island boundary,
exclude building footprints (a linkway is never inside a building), write the tile centre crop; progress is
reported every 100 tiles. Post-processing: drop polygons < 25 m2; keep polygons touching the 5 m buffer of the
pedestrian network (`pednet_overlap_ratio` attribute kept); bridge gaps <= 9 m with a 15 px closing restricted
to a 1.5 m half-width corridor around the LTA footpath layer (validated to leave F1 and IoU unchanged).

Morphological closing with radius 7 / 17 px and material-guided region growing were tested and abandoned
(they cannot bridge true tree-canopy gaps and add lateral false positives); the code is kept for transparency.

Verification of the released scripts (2026-09-10, RTX 6000 Ada): `inference_autonomous_island.py` with the
shipped checkpoint on the 4 km AOI image (13,331 x 13,242 px) ran in 0.6 min (225 tiles), `vectorise_generic.py`
and `pednet_filter_generic.py` completed the chain (352 -> 272 polygons), and the raw prediction agrees with the
published island product in that window on 99.8 % of the pixels (thin-object IoU 0.68; the island run used
different sliding-window positions). `train_autonomous_tversky.py --loss tversky --epochs 1` trained one epoch
in 9 min (val Dice 0.59 after a single epoch) and saved a lean checkpoint.

## 4. Path configuration

Training and inference scripts take all paths as command-line arguments. The post-processing / evaluation scripts
written for the island run (`postprocess_island.py`, `apply_pednet_filter.py`, `footpath_bridge_island.py`,
`eval_*.py`) keep the workstation paths in the constants at the top of each file; edit them before running.
The evaluation scripts import the training modules from `../train` (repo-relative `sys.path` insert).

## 5. Licences and credits

Code in this folder: MIT. Third-party: Segment Anything (Apache-2.0), GeoSAM (MIT, Rafi Ibn Sultan 2025),
clDice loss (MIT, Paetzold and Shit 2021), OpenAI CLIP (MIT), SAM 2 (Apache-2.0, baseline only).
