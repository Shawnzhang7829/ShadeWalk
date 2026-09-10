# Module 3b - Projection of arcade detections onto building footprints

Turns the arcade-positive street-view views of Module 3a into vector products on the building footprints:
facade runs (lines), arcade strips (polygons with building height `bld_h` and arcade ceiling height `arc_h`)
and the remaining building footprints. Environment A (geopandas / shapely); scripts take the city code
(`sg`, `bo`, `gz`) as argument.

## 1. Chain (final version, 2026-06-24)

| Step | Script | What it does | Output |
|---|---|---|---|
| 2a | `scripts/step2_project.py <city>` | For every positive view: cast a fan of 7 rays (-28 .. +28 deg around `viewheading`, max 50 m) from the panorama point; the first building wall hit that (1) faces open street (3 m clearance outward, no other building) and (2) is not occluded is the target facade; a 20 m facade segment centred at the hit point is written. Panorama points inside a footprint are discarded. | `step2a_points_<city>.gpkg` (`pid, view, viewheading, projected, dist`), `step2a_segments_<city>.gpkg` (`bld_idx, bld_h, pid, view, viewheading, dist`) |
| 2b | `scripts/step2_edge_ffw.py <city>` | Edge-based facade decision: dissolve touching footprints into blocks, take the outer and inner rings, split each ring at corners (> 30 deg), and label an edge as arcade when the projected segments that are parallel (< 30 deg) and close (< 1.5 m) cover more than 20 % of its length or more than 20 m. | `step2b_runs_<city>.gpkg` (`run_id, length, n_seg, bld_h_med`) |
| 2c | `scripts/step2_buffer_split.py <city>` | Buffer the runs into the building by 2.0 m (Singapore, Guangzhou) or 3.0 m (Bologna) with flat caps and mitre joins; intersect with the footprints -> arcade strips (area >= 2 m2) with `arc_h = min(3.6 m, building height)`; subtract the strips from the footprints -> remaining buildings (all original attributes and `height` kept). | `step2_arcade_<city>.gpkg` (`bld_h, arc_h`), `step2_building_remain_<city>.gpkg` |

Earlier gap-bridging version (kept for reference): `scripts/step2_merge.py` groups collinear segments
(< 30 deg, offset < 3 m), bridges gaps < 50 m along the footprint outline (stops at corners > 30 deg, requires 1 m of
air on the street side), drops runs < 5 m. `scripts/eval_thresholds.py` is the data-driven analysis of the gap and
minimum-length thresholds (panorama spacing, gap distribution).

## 2. Results

| City | positive views | projected segments | arcade runs | arcade strips | remaining buildings |
|---|---|---|---|---|---|
| Singapore | 28,908 views (28,415 panorama points) | 18,032 | 7,219 | 10,076 | 103,112 |
| Bologna | 35,442 views (34,918 points) | 24,999 | 7,888 | 9,043 | (footprints minus strips) |

Runtime: about 8 min for the projection and 8 min for the edge decision per city (whole Singapore).

Verification of the released scripts (2026-09-10): the three-step chain was re-run for Singapore from the
production positive views and reproduced the published vectors exactly (28,415 points, 18,032 segments,
7,219 runs with 216.0 km, 10,076 arcade strips with 428,292 m2, 103,112 remaining buildings; sorted strip areas
and run lengths identical).

## 3. Path configuration

`ROOT = Path(r"D:\Claude\SVI_FFW")` and the per-city `CFG` / `BLD` dictionaries at the top of each script define the
building layers, the positive-view CSV, the buffer width and the default arcade height. Outputs go to
`ROOT/output/step2_projection`.

## 4. Products used downstream

`step2_arcade_sg.gpkg` and `step2_building_remain_sg.gpkg` are rasterised by Module 1 (ADSM / ADSMB / DSMremain /
BREMAIN); `step2b_runs_sg.gpkg` and `step2_arcade_sg.gpkg` provide the arcade centrelines of Module 5 and the arcade
layer of the web tool (Module 4a). See `INPUT_DATA.md`.
