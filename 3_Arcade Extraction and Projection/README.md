# Module 3 - Arcade (five-foot way / portico) extraction and projection

Arcades are covered walkways inside the ground floor of buildings (Singapore shophouse five-foot ways, Bologna
porticoes). They cannot be seen from above, so they are detected in Google Street View (GSV) imagery and then
projected onto the building footprints. The module has two parts:

| Part | Folder | Input | Output |
|---|---|---|---|
| a | `a_GSV Image Arcade Detection/` | GSV perspective images (4 views per panorama), building footprints, conservation-area and reference layers | list of arcade-positive views (`pid, view, viewheading, lon, lat, probe_prob`) |
| b | `b_Arcade Projection to Building Footprint/` | positive views + building footprints with heights | arcade facade runs (lines), arcade strips (polygons with `bld_h`, `arc_h`) and the remaining building footprints |

The products of part b (`step2_arcade_<city>.gpkg`, `step2b_runs_<city>.gpkg`, `step2_building_remain_<city>.gpkg`)
are the arcade inputs of Module 1 (ADSM / ADSMB / DSMremain rasters) and Module 5 (arcade centrelines).
Each part has its own README and `INPUT_DATA.md`.
