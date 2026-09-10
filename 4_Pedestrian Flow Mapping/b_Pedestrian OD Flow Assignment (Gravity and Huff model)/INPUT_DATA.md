# Module 4b - input data inventory

Data record (input layers referred to below): figshare, https://doi.org/10.6084/m9.figshare.33549025

All inputs are placed under `Constants.ROOT` (the parent folder of the `Patronage_Flow` package); the file names
below are the ones expected by `Constants.py`.

| Item | File | Specification | Source | Notes |
|---|---|---|---|---|
| OSM highway network | `Highway_OSM.gpkg` (layer `highway`) | 266,977 lines, EPSG:3414, all OSM tags | OpenStreetMap extract of Singapore (June 2026) | filtered to `PEDESTRIAN_HIGHWAYS` = footway, pedestrian, path, steps, living_street, corridor, track, cycleway, service, residential; `foot`/`access` = no/private, vehicle-only service sub-types and `expressway=yes` removed |
| Bus stops | `BusStopLocation_Aug2025/BusStop.shp` | 5,000+ points, field `BUS_STOP_N` | LTA DataMall (Bus Stop Location, August 2025) | origins, 400 m catchment |
| MRT / LRT stations | `Train/TrainStation_Aug2025/RapidTransitSystemStation.shp` | station polygons | LTA DataMall (Train Station, August 2025) | station names |
| MRT / LRT exits | `Train/TrainStationExit_Feb2025/Train_Station_Exit_Layer.shp` | 595 exit points, `stn_name`, `exit_code` | LTA DataMall (Train Station Exit, February 2025) | origins, 800 m catchment; ridership divided by the number of exits |
| Station code lookup | `Patronage_Flow/lookup/mrt_code_to_name.csv` | `PT_CODE` -> station name | compiled from LTA station lists | included |
| Ridership by node | `Station flow/<YYYY-MM>/node/transport_node_bus_<YYYYMM>.csv`, `transport_node_train_<YYYYMM>.csv` | monthly tap-in / tap-out volumes per node, day type and hour (`PT_CODE`, `DAY_TYPE`, `TIME_PER_HOUR`, `TOTAL_TAP_IN_VOLUME`, `TOTAL_TAP_OUT_VOLUME`) | LTA DataMall "Passenger Volume by Bus Stops" / "by Train Stations" (2025-01 to 2025-12 and 2026-01) | interchange codes with `/` are split and divided |
| Building footprints with archetypes | `building/sg_buildings_v5.geojson` | 118,782 polygons, EPSG:4326 (reprojected to EPSG:3414 on load), `gross_floor_area`, `building_archetype` (21 classes: hdb, office, retail, mixed_development, industrial, hotel, hospital, ...) | City Syntax Lab building dataset (OpenStreetMap footprints with heights, gross floor area and an archetype classification compiled by the lab; released as `SG_buildings_footprint_height_function.zip` in the data record: the same 118,782 footprints, with `archetype` = `building_archetype` and `gfa_orig` = `gross_floor_area`) | destinations |
| EnergyPlus archetypes | `AllArhcetypes_SGP_2025_V5/*.idf` (20 files: 01_HDB ... 20_Supermarket) | `People per Floor Area` and `Schedule: Building Occupancy` per archetype | Singapore building-archetype models (SGP 2025 V5) | parsed by `build_occupancy_table.py`; result included as `lookup/occupancy_density.*` |
| Island boundary | `Island_boarder/Island_boarder.shp` | 14 polygons | City Syntax Lab (released as `SG_island_boundary.zip` in the data record) | network and stations clipped to it |

Parameters (`Constants.py`): `NODE_SNAPPING_TOLERANCE_M` 1.0, `REDUNDANT_EDGE_TREATMENT` discard,
`TURN_PENALTY_M` 0, `STATION_SNAP_MAX_M` 60, `BUS_RADIUS_M` 400, `RAIL_RADIUS_M` 800, `BETA` 0.003,
`DETOUR_RATIO` 1.05, `CLOSEST_DESTINATION` False (Huff competition), `RIDERSHIP_MODE` both,
`WEEKDAYS_PER_MONTH` 22, `WEEKEND_HOLIDAYS_PER_MONTH` 9, `HOURS_WEEKDAY` [8, 10, 12, 14, 16] by default
(24 hours for the full run), `N_WORKERS` 12.

The LTA DataMall datasets are available under the Singapore Open Data Licence (https://datamall.lta.gov.sg);
the station and exit locations and the archetype IDF files are released in the data record (folder `4_OD_flow`);
the derived products (`pedestrian_network_filtered.gpkg`, `station_hourly_ridership.gpkg`, `building_hourly_weight.gpkg`,
`flow_weekday_HH.gpkg`) are regenerated with this package and are available from the corresponding author on request.
