# Module 3b - input data inventory

Data record (input layers referred to below): figshare, https://doi.org/10.6084/m9.figshare.33549025

| Item | File | Specification | Source | Used by |
|---|---|---|---|---|
| Arcade-positive views, Singapore | `colonnade_sg_v2_positive.csv` (or `arcade_sg_positive.csv` from the reference detector) | `pid, view, viewheading, lon, lat, probe_prob, is_colonnade`; 28,908 rows; lon/lat WGS84 | Module 3a | `step2_project.py sg` |
| Arcade-positive views, Bologna | `colonnade_bo_v2_positive.csv` | 35,442 rows | Module 3a | `step2_project.py bo` |
| Building footprints, Singapore | `Shp/SG/SG_Building/SG_Building_SVY21_TH.shp` | 103,113 polygons (Polygon Z), EPSG:3414; height field `height` (m above ground); OSM attributes (`osm_id`, `building`, `levels`, `floorarea_`, ...) | City Syntax Lab building dataset (OpenStreetMap footprints with heights and functions compiled by the lab), earlier extract limited to footprints with a height. The current version of the dataset is released as `SG_buildings_footprint_height_function.zip` in the data record (118,782 footprints, EPSG:4326): 96.9 % of the 103,113 polygons lie inside a released footprint and their `height` equals the released `h_src_a` for 98.6 % of them; to run with the released file, reproject it to EPSG:3414 and use `h_src_a` (or `building_h` where `h_src_a` is empty) as `height` | all three scripts (`CFG["sg"]["bld"]`, `hcol="height"`) |
| Building footprints, Bologna | `Shp/Bologna/c_a944ctc_edifici_pl.geojson` | polygons, EPSG:32632, height field `altezza_gr` | Comune di Bologna open data | `CFG["bo"]` |
| Building footprints, Guangzhou (optional) | `Shp/GZ/GZ_core_building/GZ_core_building.shp` | EPSG:32649, height field `Height` | Guangzhou core-area buildings | `CFG["gz"]` (Guangzhou was not carried through to the paper) |
| Conservation areas (legacy merge only) | `MasterPlan2025ConservationAreaBoundaryLayer.geojson` | EPSG:4326 | Master Plan 2025 conservation-area boundary layer, data.gov.sg open data (not part of the data record) | `step2_merge.py` |

Parameters (defined at the top of the scripts):

| Parameter | Value | Script |
|---|---|---|
| `RAY_LEN` | 50 m (a facade further than 50 m is a building across a junction, not the facing facade) | `step2_project.py` |
| `SEG_HALF` | 10 m (projected facade segment = 20 m) | `step2_project.py` |
| `FAN` | -28, -16, -8, 0, 8, 16, 28 deg | `step2_project.py` |
| `clear` | 3.0 m outward clearance (facade must face open street) | `step2_project.py` |
| `CORNER_ANG` | 30 deg (ring split) | `step2_edge_ffw.py` |
| `NEAR`, `PARA_COS` | 1.5 m, cos 30 deg | `step2_edge_ffw.py` |
| `COV_PCT`, `COV_LEN`, `MINEDGE` | 20 %, 20 m, 2 m | `step2_edge_ffw.py` |
| `buf` | 2.0 m (SG, GZ), 3.0 m (BO) | `step2_buffer_split.py` |
| `arc_h` | 3.6 m default, capped by the building height | `step2_buffer_split.py` |

## Products

The derived products below are not part of the data record; they are regenerated from the released inputs with the scripts of this module and are available from the corresponding author on request.

| File | Geometry | Fields |
|---|---|---|
| `step2a_points_<city>.gpkg` | points (panorama positions) | `pid, view, viewheading, projected (0/1), dist` |
| `step2a_segments_<city>.gpkg` | 20 m facade segments | `bld_idx, bld_h, pid, view, viewheading, dist` |
| `step2b_runs_<city>.gpkg` | facade edges labelled as arcade | `run_id, length, n_seg, bld_h_med` |
| `step2_arcade_<city>.gpkg` | arcade strips (polygons) | `bld_h, arc_h` |
| `step2_building_remain_<city>.gpkg` | footprints minus strips | original attributes + `height`, `_bh` |
