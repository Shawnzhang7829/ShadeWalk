# Module 3a - Arcade detection in Google Street View images

Detects whether a street-view image shows a covered arcade / five-foot way (Singapore) or a portico (Bologna)
with a frozen CLIP ViT-L/14 image encoder and small city-specific linear probes (logistic regression on the
768-d image embedding). The output is the list of arcade-positive views that Module 3b projects onto the
building footprints.

## 1. Data and naming convention

Every panorama point (`pid`) has four perspective views taken at the vehicle heading + 0 / 90 / 180 / 270 degrees.
File name: `<pid>_<lon>_<lat>_<YYYYMM>_baseheading<deg>_viewheading<deg>_<view>.jpg`, e.g.
`2_11.3128997_44.4919039_202506_baseheading265.63_viewheading265.63_1.jpg` (view 1 = front, 2 = right,
3 = back, 4 = left; headings are clockwise from north). Singapore: 595,408 images (about 148,850 panoramas at
20 m spacing along the road network); Bologna: 287,724 images. The images themselves are Google content and are
not part of this repository.

## 2. Pipeline (reference implementation, `detection/`)

```
for every panorama (all 4 views):
    feats = CLIP ViT-L/14(view 1..4)                         # one embedding per image
    broad = broad_probe(feats)                                # 1. coarse screen (covered street / open street)
    candidate = any(broad[v] >= THRESH[region_tier])          #    threshold depends on the region tier
    if candidate:
        arc = arcade_probe(feats)                             # 2. arcade / portico probe
        is_arcade[v] = arc[v] >= ARC_THR                      #    SG 0.45 / BO 0.40
# whether the view faces a street-facing facade is decided later, geometrically, in Module 3b
```

| Step | Script | Environment | Output |
|---|---|---|---|
| 0 - region tier | `detection/assign_tiers.py sg|bo` | A (geopandas) | `pano_tier_<city>.csv` (`pid, lon, lat, region_tier`) |
| 1 + 2 - detection | `detection/detect_arcade.py sg|bo [smoke_N]` | B (QGIS Python + torch + transformers) | `detect_<city>.csv` (every view: `broad_score, is_candidate, arcade_prob, is_arcade`) and `arcade_<city>_positive.csv` (positive views only, the input of Module 3b) |

Region tiers and thresholds (`models/thresholds.json`, `models/colonnade_threshold.json`):

| Tier | Definition (Singapore) | Definition (Bologna) | broad threshold SG / BO |
|---|---|---|---|
| 1 | within 12 m of a CoveredLinkWay polygon that overlaps a building footprint | within 10 m of the UNESCO portici lines | 0.35 / 0.325 |
| 2 | inside the Master Plan 2025 conservation areas (+30 m buffer) | inside the historic-centre zone (`Bologna_Arcade.gpkg`, layer `zone`) | 0.42 / 0.395 |
| 3 | elsewhere | elsewhere | 0.58 / 0.555 |

The arcade probe thresholds are 0.45 (SG, `sg_colonnade_final.joblib`) and 0.40 (BO, `bo_probe.joblib`; in Bologna
the broad probe and the arcade probe are the same model with different thresholds). Detection is streamed with
incremental checkpointing and can be resumed per panorama. Throughput: about 12 panoramas (48 images) per second
on an RTX 6000 Ada.

The production run of the paper used the older multi-script chain (broad screening -> backfill of views 1/3 ->
colonnade probe, kept in the working folders). The two scripts in `detection/` reproduce that output exactly
(validated on 150 Tier-1 panoramas: Jaccard 1.000, max probability difference 0.00000) and are the reference
implementation.

Verification of the released scripts (2026-09-10): `detect_arcade.py sg 150` (150 Tier-1 panoramas, 600 images)
ran in 0.3 min at 9 panoramas per second and reproduced the production detections exactly (407 positive views on
134 panoramas, Jaccard 1.000, maximum probability difference 0.0).

## 3. Probes and training data (`probe_training/`, `labels/`, `models/`)

| Probe | File | Training set | Cross-validation |
|---|---|---|---|
| SG broad probe | `models/sg_probe.joblib` | positives = views facing CoveredLinkWay-building overlaps, negatives = views far from any covered walkway (`labels/gt_candidates.csv`) | GroupKFold(5) by panorama |
| SG colonnade probe (final) | `models/sg_colonnade_final.joblib` | 241 human-verified positives (100 shophouse five-foot ways + 141 colonnade views) and 159 verified negatives (including 82 clean facade-facing hard negatives) plus 80 random open-scene negatives; labels in `labels/strict_gt/` | GroupKFold(5) by panorama |
| BO probe | `models/bo_probe.joblib` | 1,500 positives within 10 m of the UNESCO portici and 1,500 negatives beyond 50 m (`labels/phase0_samples/`) | StratifiedKFold(5) |

Scripts: `train_probes.py` (broad probes), `prelabel_shophouse.py` / `prelabel_negatives.py` /
`gen_facing_neg_depth.py` (candidate sheets for manual verification), `train_strict_probe.py` and
`train_final_colonnade.py` (colonnade probes), `build_sg_validation.py` + `step_a_bearing.py` + `step_b_clip.py` +
`eval_clip_on_official.py` (validation against the official CoveredLinkWay layer), `validate_broad.py` /
`validate_human_gt.py` (human ground truth), `bench_compare.py` (zero-shot vs probe, B/32 vs L/14),
`phase0_build_samples.py` and `view_selection.py` (sampling and view rules of the early phase).

Benchmark on the curated ground truth (`labels/bench_results.csv`):

| Method | SG F1 | BO F1 |
|---|---|---|
| CLIP zero-shot, 3 prompts, ViT-B/32 | 0.742 | 0.800 |
| CLIP zero-shot, ViT-L/14 | 0.704 | 0.812 |
| linear probe, ViT-B/32 | 0.852 | 0.947 |
| linear probe, ViT-L/14 (used) | 0.837 | 0.962 |

Production probes, 5-fold cross-validation: Singapore accuracy 0.812, precision 0.749, recall 0.942, F1 0.835,
AUC 0.920; Bologna 0.955 / 0.934 / 0.979 / 0.956 / 0.991.

## 4. Geometry-first route (`geometry_first/`)

An alternative ordering was validated: first select the views that face a street-facing facade using the
building-footprint geometry of Module 3b (`run_facing_precise.py`), then run the colonnade probe only on those views
(`detect_precise.py`), then project (`project_precise.py`). On the common domain it reproduces the production
detections exactly (Bologna Jaccard 1.000; Singapore 0.9994) and recovers views that the broad screen had
missed (+883 SG, +3,568 BO views), with 62 % fewer CLIP encodings. See `geometry_first/REPORT.md`.

## 5. Path configuration

The scripts use the workstation layout (`ROOT = D:\Claude\SVI_FFW`, images under `SVI/<City>/<City> SVI`,
vectors under `Shp/<City>`, outputs under `output/...`). Edit the `ROOT` / `CFG` constants at the top of each script.

## 6. Licences

Code: MIT. CLIP (OpenAI) is MIT; `openai/clip-vit-large-patch14` weights are downloaded by Hugging Face
`transformers` on first use.
