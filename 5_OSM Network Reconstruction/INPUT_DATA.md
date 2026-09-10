# Module 5 - input data inventory

| Item | File | Specification | Source | Used by |
|---|---|---|---|---|
| OSM pedestrian network | `Shp/SG/Pedestrian route/OSM/Highway_OSM.gpkg` | 266,977 line features, EPSG:3414, OSM `highway` tags and all other OSM tags (427 fields) | OpenStreetMap extract of Singapore (`highway` layer, June 2026), reprojected to SVY21 | `step4_2b_osm.py` (motorway / trunk classes excluded) |
| Arcade facade runs | `step2b_runs_sg.gpkg` | 7,219 lines, EPSG:3414 | Module 3b | `step4_1_city.py` (arcade centrelines, offset 1 m into the strip) |
| Arcade strips | `step2_arcade_sg.gpkg` | 10,076 polygons | Module 3b | `step4_1_city.py` (offset direction), `step4_3b_link.py` (landing test) |
| Covered linkways | `covered_linkway_SG_island_tv_pednet_bridged.gpkg` | 6,148 polygons | Module 2 | `step4_1_city.py` (Voronoi centrelines), `step4_3b_link.py` |
| Remaining building footprints | `step2_building_remain_sg.gpkg` | 103,112 polygons, `height` | Module 3b | `step4_3b_link.py`, `step4_4_global.py` (through-building test) |
| Alternative pedestrian layers (comparison only) | `SG-Footpath/Footpath.shp`, `SG-Road/RoadSectionLine.shp`, `pedestrian_network_filtered.gpkg` | LTA footpath polygons, LTA road section lines, madina-filtered OSM network of Module 4b | LTA DataMall; Module 4b | the earlier sub-zone experiments (not shipped) compared these sources before OSM was selected |

Parameters are listed in the README (5 m facility joins, 3 m spur removal, 15 m OSM and facility-to-OSM joins,
30 m arcade-corner links, 8 m duplicate band with 80 % overlap and 30 deg parallel test, 0.5 m endpoint clustering).

## Products (data repository)

| File | Content |
|---|---|
| `step4_1_shade_centerline_SG.gpkg` | facility centrelines (`src`, `fac_id`) |
| `step4_fac_lines.gpkg` | cleaned facility lines (`src`, `fac_id`, `line_id`, `length`) |
| `step4_osm_lines.gpkg`, `step4_osm_kept.gpkg` | cleaned OSM lines before / after duplicate removal |
| `step4_connected.gpkg` | facilities connected to OSM |
| `step4_network_final.gpkg` | final network, 469,434 segments (`src`, `length`) - input of Module 4a |
