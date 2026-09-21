# Module 4b - Pedestrian OD flow assignment (gravity distance decay and Huff destination choice)

Builds the demand inputs of the flow model (station ridership, building weights, occupancy density, the OSM
pedestrian subset), distributes the 14:00 transit ridership of Singapore (bus stops and MRT / LRT exits) to the
surrounding buildings and assigns every origin-destination pair to the reconstructed network under the shortest-path
and the coolest-path rule (edge costs of Module 4a). Writes the per-edge flows, the OD metrics and the lambda / tau
sensitivity products used in the paper and by the ShadeWalk web tool. Environment A.

## 1. Model

Every trip has one end at a transit station (MRT / LRT exit or bus stop) and the other at a building.

**Origins.** Station demand = 14:00 weekday ridership of each origin record (`inj_weekday_14` of
`station_hourly_ridership_v4.gpkg`: the station's tap-in + tap-out divided over its real exit points; an interchange is
one station whose line codes share the same exits; `tot_weekday_14` holds the whole-station value and must not be used
per record). LTA volumes are monthly totals and are divided by 22 weekdays / 9 weekend days; interchange codes such as
`NS24/NE6/CC1` are one station; the few bus-stop codes that occur twice in the stop shapefile share their volume over
their rows.

**Destinations.** Building weight `W_b = GFA_b x occupancy_density(archetype_b, day type, hour)` (people-equivalent),
with the gross floor area `gfa_corr` of the released building dataset and the occupancy density of the 20 Singapore
EnergyPlus archetype models (`People per Floor Area` x `Schedule: Building Occupancy`, parsed by
`build_occupancy_table.py` into `lookup/occupancy_density.parquet`, 21 archetypes x 2 day types x 24 hours);
the 14:00 weekday column `weight_weekday_14` of `building_hourly_weight_gfa.gpkg` is used.

**Distribution and assignment.** The demand of a station is distributed to the buildings within its walking
catchment (network distance <= 800 m for MRT / LRT, <= 400 m for bus, cut-off Dijkstra) in proportion to

```
W_b * exp(-d(s, b) / 350 m)          Huff competition among the reachable buildings, exponential distance decay
```

Stations and buildings snap to the nearest node of the main connected component of the network within 120 m; pairs
closer than 20 m are dropped. Each pair is routed twice: shortest path (`sum l_i`) and coolest path
(`sum l_i x ((1 - sigma_i) + lambda)`, Module 4a); flow on an edge = sum of the demand of all OD pairs whose route uses
the edge. `step4_4f_flow_lam.py` sweeps 18 values of lambda (paper setting 0.2 included); `step4_4e_flow_detour.py`
is the alternative formulation with a hard detour cap tau.

## 2. Layout

| File | Role |
|---|---|
| `demand_inputs/Constants.py` | paths and parameters of the demand inputs (single source of truth) |
| `demand_inputs/loaders.py` | readers: OSM pedestrian subset, bus stops and MRT / LRT exits, LTA passenger volumes by node |
| `demand_inputs/build_occupancy_table.py` | EnergyPlus IDF parser -> `lookup/occupancy_density.parquet` |
| `demand_inputs/export_pedestrian_subset.py` | `output/pedestrian_network_filtered.gpkg`: the pedestrian-passable OSM subset (142,095 segments) used by Modules 2 and 5 |
| `demand_inputs/export_station_hourly.py` | `output/station_hourly_ridership.gpkg`: hourly tap-in / tap-out per station row (MRT / LRT exits x line codes, bus stops) |
| `demand_inputs/station_table_real_exits.py` | station table v4 = the export collapsed to one record per REAL exit point / bus stop, values rebuilt from the raw LTA month files (an interchange is one station, `inj_* = station total / real exit points`); the origin-weight table (`station_hourly_ridership_v4.gpkg`) |
| `demand_inputs/building_weights.py` | `building_hourly_weight_gfa.gpkg`: the hourly building weights with the gross floor area taken from `gfa_corr` of the released building dataset (`SG_buildings_footprint_height_function.shp`, joined on the OSM id); the destination-weight table |
| `demand_inputs/lookup/` | `mrt_code_to_name.csv`, `occupancy_density.csv` / `.parquet` |
| `routing_flow/step4_4c_city_routing.py` | shortest and coolest (lambda = 0.15) routing for all station-building pairs, per-edge flows `flow_short`, `flow_cool`, metrics per building type |
| `routing_flow/step4_4c_orig_flow.py`, `step4_4c_orig_coolflow.py` | the same on the original footpath-only network (`flow_orig`, `flow_cool_orig`) to separate the network effect from the behaviour effect |
| `routing_flow/step4_4f_flow_lam.py` | flows for the lambda sweep (paper setting lambda = 0.2 included), summary of shade, detour and facility share per lambda |
| `routing_flow/step4_4e_flow_detour.py` | tau-capped variant (tau = 1.0, 1.2, 1.5, 2.0) |
| `routing_flow/step4_4d_city_viz.py` | city-wide maps of edge shade and flow |

## 3. Running

```bash
# demand inputs (from this folder; data folders under Constants.ROOT, see INPUT_DATA.md)
python -m demand_inputs.build_occupancy_table            # rebuild lookup/occupancy_density.parquet from the IDF files
python -m demand_inputs.export_pedestrian_subset         # output/pedestrian_network_filtered.gpkg
python -m demand_inputs.export_station_hourly            # output/station_hourly_ridership.gpkg
python demand_inputs/station_table_real_exits.py --export output/station_hourly_ridership.gpkg --raw-dir "Station flow/2026-01/node" --tag 202601 --out output/station_hourly_ridership_v4.gpkg
python demand_inputs/building_weights.py                 # output/building_hourly_weight_gfa.gpkg (GFA = gfa_corr of the released building dataset)

# routing and flows (inputs of Module 4a: step4_4_edges_SG.gpkg, step4_4_nodes_SG.gpkg, edge_facility_SG.npy; paths at the top of each script)
python routing_flow/step4_4c_city_routing.py             # flow_short, flow_cool, OD metrics
python routing_flow/step4_4c_orig_flow.py                # flow_orig (original network)
python routing_flow/step4_4c_orig_coolflow.py            # flow_cool_orig
python routing_flow/step4_4f_flow_lam.py                 # lambda sweep
python routing_flow/step4_4e_flow_detour.py              # tau-capped variant
python routing_flow/step4_4d_city_viz.py                 # maps
```

Set `SMOKE_BBOX = (27000, 28000, 33000, 34500)` (SVY21 metres, CBD) in `Constants.py` to run the exports on a
small window. Runtime: one lambda value of the city-wide routing about 1-2 h (one cut-off Dijkstra per station,
5,921 stations); the exports take seconds to minutes.

Verification of the released code (2026-09-21, on the original workstation): the four demand-input exports were
re-run with the repository package and compared with the products in use: `pedestrian_network_filtered.gpkg`
(142,095 segments, identical geometry and length), `station_hourly_ridership_v4.gpkg` (5,758 records, every value
identical; only the row order differs) and `building_hourly_weight_gfa.gpkg` (118,782 buildings, 55 columns identical)
are reproduced, and `occupancy_density.parquet` is reproduced from the IDF files (21 x 48 values, max difference 0). The routing / flow scripts are the scripts that produced the flow products of the paper (main-component
snapping, station table v4, weights on `gfa_corr`), moved here from Module 4a (code unchanged; two comment lines name the tools of this module). End-to-end run of the released chain (2026-09-21, scratch folder, original workstation; Module 4a graph and edge
shade -> `step4_4c_city_routing.py` -> `step4_4c_orig_flow.py` -> `step4_4c_orig_coolflow.py`, `step4_4e_flow_detour.py`
(3 station subsets + merge), `step4_4f_flow_lam.py` (18 lambda values, 25 min) -> `step4_4d_city_viz.py`): `flow_short`,
`flow_cool`, `flow_orig`, `flow_cool_orig`, the 4 tau and 18 lambda arrays and the OD metrics are identical, on every edge, to
the products in use.

## 4. Results (Singapore, 14:00, all trips; `step4_4f_flow_lam_summary.csv` of 2026-09-21)

| Routing | mean length (m) | mean shade | detour | facility share of shaded metres |
|---|---|---|---|---|
| shortest | 355.0 | 0.333 | 1.000 | 0.148 |
| coolest, lambda = 1.0 | 361.3 | 0.400 | 1.014 | 0.196 |
| coolest, lambda = 0.2 (paper) | 382.8 | 0.460 | 1.069 | 0.236 |
| coolest, lambda = 0.15 | 387.4 | 0.468 | 1.080 | 0.241 |
| coolest, lambda = 0.001 | 413.0 | 0.505 | 1.155 | 0.265 |

Trip-weighted means over all station-building pairs (station table v4, main-component snapping, weights on `gfa_corr`).

## 5. Outputs

| File | Content |
|---|---|
| `output/pedestrian_network_filtered.gpkg` | the pedestrian-passable OSM subset (142,095 segments) used by Modules 2 and 5 |
| `output/station_hourly_ridership.gpkg`, `station_hourly_ridership_v4.gpkg` | station ridership per row of the export and per real exit point (the v4 table is the origin-weight table) |
| `output/building_hourly_weight_gfa.gpkg` | hourly destination weights (gross floor area = `gfa_corr` of the released building dataset) |
| `step4_4_edges_flow_SG.gpkg` | the edge layer of Module 4a + `flow_short`, `flow_cool`, `flow_orig` |
| `flow_lam_<lambda>.npy`, `flow_cooltau_<tau>.npy`, `flow_cool_orig_SG.npy` | per-edge flows (main component order) for the sensitivity runs |
| `step4_4_od_metrics_SG.csv`, `step4_4f_flow_lam_summary.csv`, `step4_4e_flow_detour_summary.csv` | aggregate metrics by building type / lambda / tau |

The derived products are not part of the data record; they are regenerated from the released inputs with these
scripts and are available from the corresponding author on request.

## 6. Path configuration

`Constants.ROOT` is this module folder; the data folders (`Highway_OSM.gpkg`, `BusStopLocation_Aug2025/`, `Train/`,
`Station flow/<YYYY-MM>/node/`, `building/sg_buildings_v5.geojson`, `5_base_data/SG_buildings_footprint_height_function/`,
`AllArhcetypes_SGP_2025_V5/`, `Island_boarder/`) are expected under it, the outputs go to `output/`. The routing / flow
scripts keep the absolute paths of the original workstation in the constants at the top of each file (`OUT`, `BW`, `ST`);
edit them before running. See `INPUT_DATA.md`.
