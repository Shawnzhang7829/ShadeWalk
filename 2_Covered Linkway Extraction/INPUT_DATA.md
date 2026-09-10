# Module 2 - input data inventory

| Item | File | Specification | Source | Used by |
|---|---|---|---|---|
| Satellite imagery, whole island | `SG_google_map_03m_SVY21.tif` | RGBA GeoTIFF, 179,096 x 115,025 px, 0.3 m, EPSG:3414, bounds 2667.5 / 15748.7 / 56396.3 / 50256.2 | 0.3 m Google Earth imagery mosaicked and reprojected to SVY21 (the island mosaic is not redistributed; the 1,111 training tiles are part of the data record) | training-tile extraction, `inference_autonomous_island.py`, `make_fishnet_v3.py`, `sweep_fp_filters.py` |
| Satellite imagery, 4 km AOI | `SG_google_map_4000_03m.tif` | RGBA, 13,331 x 13,242 px, 0.3 m, EPSG:3414 (the Module 1 test window) | same | `inference_autonomous.py`, quick tests |
| Training tiles | `images/{train,val}/tile_x<col>_y<row>.png` | 1024 x 1024 RGB PNG cut from the island imagery at pixel offsets (col, row); 808 train / 303 val (7:3 split) | cut with `data_prep` / grid | training and validation |
| Annotations | `training_data/labels_json/{train,val}/*.json` | LabelMe 6.0 polygons, label `linkway` | manual annotation | mask generation |
| Masks | `training_data/masks/{train,val}/*.png` | 1024 x 1024 uint8, 0 / 255 | rasterised from the JSON polygons | `--masks_dir` |
| Tile grid | `training_data/grid/tiles_fishnet*.gpkg`, `tiles_meta.csv` | 2,082 tiles, EPSG:3414, `split` = train / val / blank, pixel offsets and geotransform per tile | `make_fishnet_v3.py` | reproducing the tile cut |
| Island boundary | `Island_boarder.shp` | 14 polygons (mainland + offshore islands), EPSG:3414 | City Syntax Lab (released as `SG_island_boundary.zip` in the data record) | `--boundary_shp` |
| Building footprints | `SG_Building_SVY21.gpkg` | building polygons, EPSG:3414 | City Syntax Lab building dataset (OpenStreetMap footprints with heights and functions compiled by the lab; same source as Module 3b) | `--building_gpkg` (exclusion mask) |
| Pedestrian network | `pedestrian_network_filtered.gpkg` | 142,095 line segments, EPSG:3414 | Module 4b (`Patronage_Flow.Main` export of the OSM pedestrian-passable highway subset; the cleaned OSM lines are released as `SG_osm_lines.gpkg` in the data record) | `apply_pednet_filter.py` (5 m buffer) |
| LTA footpath layer | `Footpath_Mar2026/Footpath.gpkg` | footpath polygons / lines, EPSG:3414 | LTA DataMall (Footpath, March 2026 release) | `footpath_bridge_island.py` (1.5 m corridor) |
| SAM ViT-H weights | `sam_vit_h_4b8939.pth` | 2.56 GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth | training and inference |
| CLIP text embedding | `checkpoints/clip_linkway_emb.pth` | 512-d embedding of "Covered Linkway" from CLIP ViT-B/32 | cached by `ensure_text_embedding()` | prompt token |
| Trained checkpoint | `checkpoints/geosam_topolora_autonomous_lean_dice0.7138.pth` | 27 MB lean state dict (LoRA A/B, mask decoder, text projection) | `train_autonomous_tversky.py` | `--trained_ckpt` |

## Products

| Product | Specification |
|---|---|
| `covered_linkway_SG_island_tv.tif` | uint8 binary raster on the imagery grid (0.3 m, 179,096 x 115,025 px), nodata 0 |
| `covered_linkway_SG_island_tv.gpkg` | polygons >= 25 m2 (10,064) |
| `covered_linkway_SG_island_tv_pednet.gpkg/.tif` | after the 5 m pedestrian-network filter (6,323 polygons, `pednet_overlap_ratio`) |
| `covered_linkway_SG_island_tv_pednet_bridged.gpkg/.tif` | FINAL: 6,148 polygons, 1.66 km2, field `area_m2`; input of Module 1 (LDSM) and Module 5 (facility centrelines) |

The derived products below are not part of the data record; they are regenerated from the released inputs with the scripts of this module and are available from the corresponding author on request.
