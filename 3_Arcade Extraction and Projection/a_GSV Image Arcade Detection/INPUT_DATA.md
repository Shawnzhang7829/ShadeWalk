# Module 3a - input data inventory

Data record (input layers referred to below): figshare, https://doi.org/10.6084/m9.figshare.33549025

| Item | Files | Specification | Source | Used by |
|---|---|---|---|---|
| Street-view images, Singapore | `SVI/Singapore/Singapore SVI/*.jpg` (595,408 files) | 4 perspective views per panorama, JPEG, named `<pid>_<lon>_<lat>_<YYYYMM>_baseheading<deg>_viewheading<deg>_<view>.jpg`; WGS84 lon/lat in the name | Google Street View, downloaded at the sample points (nearest panorama to each point; not redistributable) | detection, probe training |
| Street-view images, Bologna | `SVI/Bologna/Bologna SVI/*.jpg` (287,724 files) | same convention | same | detection, probe training |
| Sample points and roads | `Shp/SG/Singapore_street view/Singapore_sample_points.shp` (167,466 points, EPSG:4326), `Singapore_roads.shp`; `Shp/Bologna/Bologna_street view/Italy_City_sample_points.shp`, `Italy_City_roads.shp` | 20 m sampling points along the road network | derived from OSM roads | `phase0_build_samples.py` |
| Building footprints, Singapore | `Shp/SG/SG_Building/SG_Building_SVY21_TH.shp` | 103,113 polygons, EPSG:3414, field `height` (m) | City Syntax Lab building dataset (OpenStreetMap footprints with heights and functions compiled by the lab; released as `SG_buildings_footprint_height_function.zip` in the data record) | `assign_tiers.py` (Tier 1), validation set |
| Building footprints, Bologna | `Shp/Bologna/c_a944ctc_edifici_pl.geojson` | polygons, EPSG:32632, field `altezza_gr` | Comune di Bologna open data (edifici) | Module 3b |
| Covered linkways (official) | `Shp/SG/CoveredLinkWay_Mar2026/CoveredLinkWay.shp` | 7,012 polygons, SVY21 | LTA DataMall (CoveredLinkWay, March 2026) | Tier 1 definition; official validation set (`build_sg_validation.py`) |
| Conservation areas | `Shp/SG/SG ConservationArea2025/MasterPlan2025ConservationAreaBoundaryLayer.geojson` | 295 polygons, EPSG:4326 | URA Master Plan 2025 | Tier 2 |
| UNESCO porticoes | `Shp/Bologna/origini-di-bologna-portici/origini-di-bologna-portici.shp` | 12 portico groups (lines) | Comune di Bologna open data | Tier 1 (BO), probe positives |
| Bologna historic centre | `Shp/Bologna/Bologna_Arcade.gpkg`, layer `zone` | 4 zones, EPSG:25832 | Comune di Bologna | Tier 2 (BO) |
| Test polygon | `Shp/SG/4000pixel_polygon_SVY21/` | 4000 px window | project | smoke tests |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | Hugging Face transformers | OpenAI | embeddings |
| Probes | `models/*.joblib` | scikit-learn LogisticRegression on 768-d CLIP embeddings | this module | detection |
| Thresholds | `models/thresholds.json`, `models/colonnade_threshold.json` | per-tier broad thresholds; arcade threshold | this module | detection |

## Label files shipped in `labels/`

| File | Content |
|---|---|
| `strict_gt/prelabel_500.csv` | 500 pre-ranked shophouse candidates (`rank, path, pid, viewheading, probe_score, shophouse_score`); ranks 1-100 were verified as positives |
| `strict_gt/prelabel_neg*.csv` | hard-negative candidate sheets (covered but not shophouse; facade-facing negatives); verified subsets are referenced by rank in `train_final_colonnade.py` |
| `strict_gt/strict_train_scored.csv` | final training set of the colonnade probe with cross-validated probabilities (`path, pid, y, src, cv_proba`) |
| `phase1b_sg/human_gt.csv`, `human_gt_broad.csv`, `gt_labels.csv`, `compare_results.csv` | 32 human-labelled images used to compare model variants |
| `human_gt_validation.csv` | human validation of the L/14 broad probe |
| `gt_candidates.csv` | broad-probe training candidates (CoveredLinkWay bearing filter) |
| `phase1c_sg_val/sg_val_positive.csv`, `sg_val_negative.csv`, `overlap_stats.txt` | official validation set built from CoveredLinkWay-building overlaps |
| `phase0_samples/bo_positive.csv`, `bo_negative.csv`, `sg_test_pts.csv` | Bologna portici samples and Singapore test points |
| `bench_results.csv` | benchmark table (method, city, threshold, F1, P, R, accuracy) |

The `path` columns contain the original absolute image paths of the workstation; only the file name part is
needed to locate an image.

## Outputs

The derived products below are not part of the data record; they are regenerated from the released inputs with the scripts of this module and are available from the corresponding author on request.

`detect_<city>.csv`, `arcade_<city>_positive.csv` (reference implementation) and the production files
`colonnade_<city>_v2.csv` / `colonnade_<city>_v2_positive.csv` (28,908 positive views in Singapore, 35,442 in
Bologna) and `pano_tier_<city>.csv`. The positive CSV has the columns
`pid, view, viewheading, lon, lat, probe_prob, is_colonnade`.
