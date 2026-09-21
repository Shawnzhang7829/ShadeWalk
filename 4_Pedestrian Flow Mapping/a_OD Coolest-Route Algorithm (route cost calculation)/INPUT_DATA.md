# Module 4a - input data inventory

Data record (input layers referred to below): figshare, https://doi.org/10.6084/m9.figshare.33549025

| Item | File | Specification | Source | Used by |
|---|---|---|---|---|
| Reconstructed pedestrian network | `step4_network_final.gpkg` | 469,434 line segments, EPSG:3414, fields `src`, `length` | Module 5 | `make_prep.py` |
| 14:00 shadow raster, full system | `TIF_shadow_newarcade/merge_images/Shadow_2pm_h14.tif` | float32, 44,000 x 27,000 px, 1 m, EPSG:3414 (1 = sunlit, 0 = shaded) | Module 1 (buildings + trees + linkways + arcades) | `step4_4b_city_edge_shade.py` (`shade_full`) |
| 14:00 shadow category raster | `merge_images/Category_2pm_h14.tif` | uint8, classes 0-12 | Module 1 | `step4_edge_facility.py`, `step4_edge_class_1m.py`, `step4_network_split_1m.py`, `edge_px_permetre.py` |
| 24-band shadow / category rasters | `merge_images/Shadow/Shadow_merged.tif`, `Category/Category_merged.tif` | 24 bands (band = hour + 1) | Module 1 | `make_hourly_cache.py`, `make_tiles.py` (web tool layers) |
| Building DSM, DEM, canopy | `SUB_SG_Polygon_DSMremain_1m.tif`, `SUB_SG_Polygon_DEM_1m.tif`, `SUB_SG_Polygon_CDSMclean_1m.tif` | float32, same grid | Module 1 | `step4_4a_city_building_shadow.py`, `step4_4a_bldtree_shadow.py` (baselines) |
| Building-footprint mask | `SUB_SG_Polygon_BREMAIN_1m.tif` | uint8 0/1 | Module 1 | edges through buildings are treated as indoor (fully shaded, excluded from facility attribution) |
| Arcade strips | `step2_arcade_sg.gpkg` | 10,076 polygons, `bld_h`, `arc_h` | Module 3b | web tool layers, arcade DSM top for the 3D view |
| Covered linkways | `covered_linkway_SG_island_tv_pednet_bridged.gpkg` | 6,148 polygons | Module 2 | web tool layers |
| Remaining buildings | `step2_building_remain_sg.gpkg` | 103,112 polygons with `height` | Module 3b | web tool 3D buildings |
| Subzone boundaries | `SG_Subzone/SG_subzone boundary 2019_SVY21.shp` | Master Plan 2019 subzone boundaries, EPSG:3414 | data.gov.sg open data (Master Plan 2019 Subzone Boundary; not part of the data record) | maps, sub-zone experiments |
| Road section lines | `RoadSectionLine_Mar2026/RoadSectionLine.shp` | LTA road centre lines | LTA DataMall | web tool basemap layer |
| Tree points | `SG point tree/Point tree.shp` | about 697,000 tree points with size attributes | trees.sg open tree map (web tool visualisation only) | web tool 3D trees |
| Station ridership, building weights, per-edge flows | `station_hourly_ridership_v4.gpkg`, `building_hourly_weight_gfa.gpkg`, `step4_4_edges_flow_SG.gpkg` | see Module 4b | Module 4b | web tool (flow layers, HDB footprints, route cards) |
| Demo OD pairs | `webapp/dist/demo_scenarios.json` | precomputed demonstration origin-destination pairs (`step4_demo_scenarios.py`) | this module | web tool |

Parameters: 2 m sampling step along edges, sun position of 13:30 for the half-hour-centred 14:00 shadow, 0.5 m
endpoint clustering. The demand and routing parameters (lambda, distance decay, catchments, snapping) are listed in
the `INPUT_DATA.md` of Module 4b.

## Products

The derived products below are not part of the data record; they are regenerated from the released inputs with the scripts of this module and are available from the corresponding author on request. The flow products (`step4_4_edges_flow_SG.gpkg`, `flow_*.npy`, the OD metrics and sensitivity summaries) are listed in Module 4b.

| File | Content |
|---|---|
| `step4_network_final_prep.gpkg` | network with `comp` and mapped `src` |
| `step4_4_edges_SG.gpkg`, `step4_4_nodes_SG.gpkg` | graph edges (`u`, `v`, `shade_full`, `shade_bld`) and nodes |
| `edge_facility_SG.npy`, `edge_shade_nofac_SG.npy`, `edge_px_SG.npz`, `edge_class_1m_totals.csv` | per-edge facility class, no-facility shade, per-metre shade attribution |
| `edge_shade_hourly_SG.npz` | per-edge shade fraction for every hour 08-18 (keys `h08` ... `h18`, full row order of `step4_4_edges_SG.gpkg`; `h14` = `shade_full`) |
| `SUB_SG_BUILDING_SHADOW_h14_1m.tif`, `SUB_SG_BLDTREE_SHADOW_h14_1m.tif` | baseline shadow rasters (uint8) |
