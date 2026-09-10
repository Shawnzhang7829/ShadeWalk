# Covered Linkway - final report

GeoSAM (SAM ViT-H) + TopoLoRA (LoRA on QKV + clDice) + Focal-Tversky loss, benchmarked against a
SAM2-UNet + LoRA baseline.

**Evaluation protocol.** Two protocols appear in this report; every row is labelled with one of them:

- **(model)** - checkpoint forward on the 303 validation tiles, threshold 0.5, micro-averaged, prompt-free,
  leakage-free. Apples-to-apples between trained models (SAM2-UNet + LoRA, GeoSAM-TopoLoRA v3, Focal-Tversky).
- **(island)** - full-Singapore deployment GeoTIFF (after post-processing: building exclusion / pedestrian-network
  filter / bridging) sampled at the same 303 validation-tile windows. Reflects the shipped product, not the raw model.

## 1. Training / optimisation milestones

| # | Stage | Key change | Effect |
|---|-------|-----------|--------|
| 0 | SAM2-UNet + LoRA baseline | SAM2.1 Hiera-L + LoRA, BCE + IoU | model F1 0.457 (reference baseline) |
| 0b | GeoSAM + LoRA, no clDice | ablation control (see 2.1) | model F1 0.416 - lower bound of this framework |
| 1 | Point-prompted GeoSAM + TopoLoRA | SAM ViT-H + LoRA + CLIP + GT-sampled points | val Dice 0.93 but GT leakage; blind = 0 -> unusable |
| 2 | Autonomous v3 (structure loss) | remove ALL point prompts, text-only | first deployable, model F1 0.501 |
| 3 | Full island + pedestrian network | min area 25 m2 + 5 m pedestrian-network soft constraint | island F1 0.501 -> 0.525 |
| 4 | Post-processing experiments | closing R7 / R17, material grow, FP sweep | none beat the baseline (Pareto-optimal negative result) |
| 5 | Focal-Tversky loss (decisive) | beta 0.7 > alpha 0.3 penalises FP, + 0.3 clDice | model F1 0.501 -> 0.714 (+43 %), precision doubled |
| 6 | 15-epoch convergence check | full 15 epochs from 0.7138 | peak 0.7157 ~ 0.7138 -> converged |
| 7 | Tversky full-island regeneration | new model -> island + pedestrian network | island 6,323 segments / 1.64 km2, P 0.719 |
| 8 | F1-safe footpath bridge | LTA footpath + 15 px / 1.5 m corridor closing | dF1 = 0; bridges <= 9 m gaps, connectivity up |

## 2. Complete performance table

| Stage | P | R | F1 | IoU | clDice | Betti-0 | polygons | km2 |
|-------|---|---|----|-----|--------|---------|----------|-----|
| SAM2-UNet + LoRA (model) | 0.301 | 0.946 | 0.457 | 0.296 | 0.627 | 5.40 | - | - |
| v3 structure (model) | 0.342 | 0.932 | 0.501 | 0.334 | 0.635 | 10.35 | - | - |
| v3 structure + pednet (island) | 0.382 | 0.839 | 0.525 | 0.356 | 0.640 | 2.78 | 17,209 | 6.28 |
| **Focal-Tversky (model)** | 0.709 | 0.718 | **0.714** | **0.555** | **0.782** | 4.33 | - | - |
| Tversky raw (island) | 0.694 | 0.692 | 0.693 | 0.530 | 0.753 | 5.83 | 10,064 | 2.38 |
| Tversky + pednet (island) | 0.719 | 0.655 | 0.686 | 0.522 | 0.745 | 1.38 | 6,323 | 1.64 |
| **Tversky + pednet + bridge (FINAL)** | 0.715 | 0.659 | 0.686 | 0.522 | 0.744 | 1.43 | 6,148 | 1.66 |

- vs SAM2-UNet + LoRA baseline (model): F1 0.457 -> 0.714 = **+0.257 (+56 %)**, IoU 0.296 -> 0.555 (+88 %),
  clDice 0.627 -> 0.782.
- GeoSAM-TopoLoRA internal gain (v3 structure -> Focal-Tversky, model): F1 +0.213 (+43 %), IoU +0.221 (+66 %),
  precision doubled, clDice +0.147.
- Decisive lever = stage 5 (Focal-Tversky); a pure loss-function change.
- SAM2-UNet + LoRA = high recall (0.946) / low precision (0.301): it over-predicts, hence the low F1 / IoU
  despite a decent clDice.

### 2.1 clDice ablation (controlled)

Single evaluation run, identical protocol; the ONLY difference is `--cldice_weight 0` vs `0.3`
(loss = structure (pw = 30), lr 5e-5, batch 2, LoRA r16 / a32, same 808 / 303 split). Raw output: `ablation_cldice.txt`.

| Config | P | R | F1 | IoU | clDice | Betti-0 |
|--------|---|---|----|-----|--------|---------|
| GeoSAM + LoRA, **no clDice** (best, ep 11) | 0.267 | 0.946 | 0.416 | 0.263 | 0.503 | 16.66 |
| GeoSAM + LoRA, **no clDice** (ep 20 end) | 0.261 | 0.951 | 0.410 | 0.258 | 0.505 | 14.88 |
| GeoSAM + LoRA + **clDice** (v3) | 0.342 | 0.932 | **0.501** | **0.334** | **0.635** | **10.35** |

clDice contributes across the board, not just topology: F1 +0.084 (+20 %), IoU +0.071 (+27 %), clDice +0.132
(+26 %), Betti-0 -6.31 (38 % fewer fragmentation errors). Mechanism: skeleton supervision suppresses false
positives - precision 0.267 -> 0.342 at essentially unchanged recall.

Not an under-training artefact: the no-clDice run got 15 from-scratch epochs (best val Dice 0.4320 at ep 11)
plus 5 resumed epochs, and no epoch beat 0.4320; its clDice metric stays at 0.503-0.505 at both 15 and 20 epochs,
while the +clDice run reached val Dice 0.5210 within 15 epochs.

Protocol sanity check: re-evaluating the v3 checkpoint in this run reproduces the published row exactly
(F1 0.5006, IoU 0.3339, clDice 0.6352, Betti-0 10.35).

Note on loss composition: `--loss` is a choice, not an addition - Focal-Tversky *replaces* the structure loss
(and carries its own 0.5 BCE term). So the FINAL objective is `FocalTversky + 0.5 BCE + 0.3 clDice`, not
structure + FocalTversky + clDice.

## 3. Files in this folder

| File | Content |
|------|---------|
| `performance_table.csv` | machine-readable version of table 2 |
| `sam2unet_val_metrics.txt` | raw SAM2-UNet + LoRA (model) evaluation line |
| `ablation_cldice.txt` | raw clDice-ablation evaluation output and provenance |

Figures (metric progression, island / local comparisons, training curves, gallery) are part of the paper and its
supplement and are not stored in the code repository.

## 4. Final model and deliverables

- Model: `checkpoints/geosam_topolora_autonomous_lean_dice0.7138.pth` (this repository)
- FINAL product: `covered_linkway_SG_island_tv_pednet_bridged.tif` / `.gpkg` (data repository)
- Alternative (F1-max, no pedestrian-network filter): `covered_linkway_SG_island_tv.tif` / `.gpkg`
- SAM2-UNet checkpoint: `best_model_dice0.4856.pth` (869 MB, data repository)
- Reproduce: `run_full_island.bat` and the module README

## 5. Metric definitions

Binary segmentation, foreground = covered linkway. TP / FP / FN are pixel counts.

- **Precision** = TP / (TP + FP) - of predicted linkway pixels, the fraction that is correct
  (low = over-prediction / false positives).
- **Recall** = TP / (TP + FN) - of true linkway pixels, the fraction recovered (low = misses / fragmentation).
- **F1** = 2PR / (P + R) - harmonic mean; numerically equals the Dice coefficient for binary masks.
  Primary ranking metric.
- **IoU** (Jaccard) = TP / (TP + FP + FN) - stricter than F1 (always <= F1), more sensitive to boundary offset.
- **clDice** - Dice on the skeletons (centrelines) of prediction and ground truth; measures topology /
  connectivity rather than area. Well suited to thin linkways.
- **Betti-0** - mean |#connected components (pred) - #cc (GT)|. **Lower is better**: large = broken into
  fragments or speckle; about 0 = connectivity matches the ground truth. Post-processing (pedestrian-network filter,
  bridging) drives this down.

Rule of thumb: P / R / F1 / IoU = "is it drawn accurately"; clDice / Betti-0 = "is it connected". For linkway
networks the latter two often matter more.
