# Module 4b - input data inventory

Data record (input layers referred to below): figshare, https://doi.org/10.6084/m9.figshare.33549025

## Demand inputs (`demand_inputs/`)

All inputs are placed under `Constants.ROOT` (this module folder); the file names below are the ones expected by
`Constants.py`.

| Item | File | Specification | Source | Notes |
|---|---|---|---|---|
| OSM highway network | `Highway_OSM.gpkg` (layer `highway`) | 266,977 lines, EPSG:3414, all OSM tags | OpenStreetMap extract of Singapore (June 2026) | filtered to `PEDESTRIAN_HIGHWAYS` = footway, pedestrian, path, steps, living_street, corridor, track, cycleway, service, residential; `foot`/`access` = no/private, vehicle-only service sub-types and `expressway=yes` removed (`export_pedestrian_subset.py`) |
| Bus stops | `BusStopLocation_Aug2025/BusStop.shp` | 5,000+ points, field `BUS_STOP_N` | LTA DataMall (Bus Stop Location, August 2025) | origins, 400 m catchment |
| MRT / LRT exits | `Train/TrainStationExit_Feb2025/Train_Station_Exit_Layer.shp` | 595 exit points, `stn_name`, `exit_code` | LTA DataMall (Train Station Exit, February 2025) | origins, 800 m catchment; station ridership divided over its real exit points |
| Station code lookup | `demand_inputs/lookup/mrt_code_to_name.csv` | `PT_CODE` -> station name | compiled from LTA station lists | included |
| Ridership by node | `Station flow/<YYYY-MM>/node/transport_node_bus_<YYYYMM>.csv`, `transport_node_train_<YYYYMM>.csv` | monthly tap-in / tap-out volumes per node, day type and hour (`PT_CODE`, `DAY_TYPE`, `TIME_PER_HOUR`, `TOTAL_TAP_IN_VOLUME`, `TOTAL_TAP_OUT_VOLUME`) | LTA DataMall "Passenger Volume by Bus Stops" / "by Train Stations" (2025-01 to 2025-12 and 2026-01) | interchange codes with `/` are one station; monthly totals divided by 22 weekdays / 9 weekend days |
| Building footprints with archetypes | `building/sg_buildings_v5.geojson` | 118,782 polygons, EPSG:4326 (reprojected to EPSG:3414 on load), `gross_floor_area`, `building_archetype` (21 classes: hdb, office, retail, mixed_development, industrial, hotel, hospital, ...) | City Syntax Lab building dataset (OpenStreetMap footprints with heights, gross floor area and an archetype classification compiled by the lab; released as `SG_buildings_footprint_height_function.zip` in the data record: the same 118,782 footprints, with `archetype` = `building_archetype` and `gfa_orig` = `gross_floor_area`) | destinations (geometry, archetype) |
| Released building dataset | `5_base_data/SG_buildings_footprint_height_function/SG_buildings_footprint_height_function.shp` | 118,782 polygons, `id`, `gfa_orig`, `gfa_corr` (storeys x footprint), height, storeys, archetype | data record, folder `5_base_data` | the gross floor area used for the destination weights is `gfa_corr` (`building_weights.py`, joined on the OSM id) |
| EnergyPlus archetypes | `AllArhcetypes_SGP_2025_V5/*.idf` (20 files: 01_HDB ... 20_Supermarket) | `People per Floor Area` and `Schedule: Building Occupancy` per archetype | Singapore building-archetype models (SGP 2025 V5) | parsed by `build_occupancy_table.py`; result included as `lookup/occupancy_density.*` |
| Island boundary | `Island_boarder/Island_boarder.shp` | 14 polygons | City Syntax Lab (released as `SG_island_boundary.zip` in the data record) | network and stations clipped to it |

Parameters (`Constants.py`): `PEDESTRIAN_HIGHWAYS` and the exclusion tags above, `WEEKDAYS_PER_MONTH` 22,
`WEEKEND_HOLIDAYS_PER_MONTH` 9, `OCCUPANCY_DEFAULT_DENSITY` 0, `SMOKE_BBOX` None (full island).

## Routing and flow inputs (`routing_flow/`)

| Item | File | Specification | Source | Used by |
|---|---|---|---|---|
| Graph edges and nodes with shade | `step4_4_edges_SG.gpkg`, `step4_4_nodes_SG.gpkg` | 469,434 edges (`u`, `v`, `length`, `src`, `comp`, `shade_full`, `shade_bld`) and their nodes | Module 4a (`step4_4b_city_edge_shade.py`) | all routing scripts (`OUT`) |
| Dominant facility class per edge | `edge_facility_SG.npy` | per-edge class (arcade > linkway > tree > building) | Module 4a (`step4_edge_facility.py`) | `step4_4e_flow_detour.py`, `step4_4f_flow_lam.py` (facility share of shaded metres) |
| Station ridership | `station_hourly_ridership_v4.gpkg` | 5,758 points = 591 MRT / LRT exit points + 5,167 bus stops (one record per real exit point), `PT_CODE` (first line code), `LTA_CODE`, `CODES` (all line codes sharing the exit), `source`, `n_exits` (real exit points of the station), hourly `in_/out_/tot_/inj_weekday_HH` and weekend columns, totals | `demand_inputs/export_station_hourly.py` + `station_table_real_exits.py` | demand origins (`inj_weekday_14` = station total / real exit points; an interchange is one station; `tot_*` is the whole-station value and is not used per record) |
| Building weights | `building_hourly_weight_gfa.gpkg` | 118,782 polygons, `building_archetype`, `gross_floor_area` (= `gfa_corr` of the released building dataset), `weight_weekday_HH` (48 hourly columns), peaks and totals | `demand_inputs/building_weights.py` | demand destinations (`weight_weekday_14`), building types in the metrics |
| Subzone boundaries | `SG_Subzone/SG_subzone boundary 2019_SVY21.shp` | Master Plan 2019 subzone boundaries, EPSG:3414 | data.gov.sg open data (Master Plan 2019 Subzone Boundary; not part of the data record) | `step4_4d_city_viz.py` (maps) |

Parameters (top of the scripts): `LAM` 0.15 (`step4_4c_city_routing.py`) / 0.2 (paper, lambda sweep), `BETA` 350 m
(distance decay), `D_MRT` 800 m, `D_BUS` 400 m, `SNAP_MAX` 120 m, minimum shortest length 20 m; main-component snapping.

The LTA DataMall datasets are available under the Singapore Open Data Licence (https://datamall.lta.gov.sg);
the station and exit locations and the archetype IDF files are released in the data record (folder `4_OD_flow`);
the derived products (`pedestrian_network_filtered.gpkg`, `station_hourly_ridership.gpkg` and its per-exit-point form
`station_hourly_ridership_v4.gpkg`, `building_hourly_weight_gfa.gpkg`, the per-edge flows) are regenerated with this
module and are available from the corresponding author on request.
