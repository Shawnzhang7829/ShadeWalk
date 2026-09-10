# Module 4b - Pedestrian OD flow assignment (gravity distance decay and Huff destination choice)

`Patronage_Flow` distributes the hourly transit ridership of Singapore (bus stops and MRT / LRT exits) over the
pedestrian street network towards buildings weighted by gross floor area and hourly occupancy. It runs the
*patronage betweenness* of the madina package (Alhassan and Sevtsuk, 2024; the Python implementation of the Urban
Network Analysis toolbox) and writes per-edge hourly pedestrian flows as GeoPackages. Environment A with madina 0.0.15.

## 1. Model

For each transit origin *s* with weight `W_s` = ridership(s, day type, hour) and every building *b* within the walking
catchment along the network (`BUS_RADIUS_M` 400 m for bus stops, `RAIL_RADIUS_M` 800 m for MRT / LRT exits):

```
share of W_s going to b  ~  W_b * exp(-beta * d(s, b))          Huff competition among all reachable destinations
W_b = gross_floor_area_b * occupancy_density(archetype_b, day type, hour)     people-equivalent destination weight
```

with `beta = 0.003` (half-distance 231 m, madina's recommended pedestrian value). The trip volume of every pair is
added to the edges of its shortest path (alternatives up to `DETOUR_RATIO = 1.05` are admitted and share the
volume), and the sums over all pairs give the edge flow. One betweenness call per (day type, hour) slot;
bus and rail are run as two sub-passes with their own radius and added.

Ridership handling: LTA volumes are monthly totals and are divided by 22 weekdays / 9 weekend days; interchange
codes such as `NS24/NE6/CC1` are split and their volume divided by the number of parts; MRT ridership is divided
equally over the physical exits of the station (`n_exits`). `RIDERSHIP_MODE = "both"` injects tap-in + tap-out;
`Main_dualpass.py` runs the true dual pass instead (stations -> buildings with tap-out, buildings -> stations with
tap-in, summed). Origins that are not in the active sub-pass are deactivated before each betweenness call
(about 3x faster).

Occupancy density per archetype and hour comes from the 20 Singapore EnergyPlus archetype models
(`People per Floor Area` x `Schedule: Building Occupancy`, parsed by `build_occupancy_table.py` into
`lookup/occupancy_density.parquet`, 21 archetypes x 2 day types x 24 hours).

## 2. Package layout

| File | Role |
|---|---|
| `Patronage_Flow/Constants.py` | all paths and parameters (single source of truth) |
| `Patronage_Flow/Network.py` | load the OSM pedestrian highway subset, stations (bus stops, MRT exits), buildings -> madina `Zonal` |
| `Patronage_Flow/Flow_Computation.py` | ridership loading, weight management, hourly betweenness loop |
| `Patronage_Flow/Main.py` | single-pass orchestrator (`python -m Patronage_Flow.Main`) |
| `Patronage_Flow/Main_dualpass.py` | dual-pass ingress + egress model |
| `Patronage_Flow/run_2025_quarterly.py`, `run_2025_annual_total.py` | seasonal snapshots (March / June / September / December 2025, 12:00 and 14:00) and the 12-month annual total (betweenness is linear in the origin weights, so the monthly sums are run once) |
| `Patronage_Flow/build_occupancy_table.py` | EnergyPlus IDF parser -> occupancy density table |
| `Patronage_Flow/export_building_hourly.py`, `export_station_hourly.py` | QA exports: `building_hourly_weight.gpkg`, `station_hourly_ridership.gpkg` (the inputs of Module 4a) |
| `Patronage_Flow/lookup/` | `mrt_code_to_name.csv`, `occupancy_density.csv` / `.parquet` |
| `notebooks/Patronage_Flow_Pipeline.ipynb` | step-by-step notebook (smoke bounding box first, then full island) |
| `tools/patch_madina.py` | removes the `fastpath=True` argument that madina 0.0.15 passes to pandas (incompatible with pandas 3) |
| `tools/madina_smoke.py` | minimal end-to-end madina test on a CBD bounding box |
| `docs/DEVELOPMENT_NOTES.md` | architecture and implementation notes written during development |

## 3. Running

```bash
# install madina (editable) and patch it for pandas 3
git clone https://github.com/cityform-lab/madina.git   # or the PyPI release 0.0.15
pip install -e madina
python tools/patch_madina.py                              # edit the path inside first

# from the folder that contains Patronage_Flow/
python -m Patronage_Flow.build_occupancy_table            # rebuild lookup/occupancy_density.parquet from the IDF files
python -m Patronage_Flow.Main                             # single pass, hours in Constants.HOURS_WEEKDAY
python -m Patronage_Flow.Main_dualpass                    # dual pass (about 2x compute)
python -m Patronage_Flow.run_2025_quarterly
python -m Patronage_Flow.run_2025_annual_total
python -m Patronage_Flow.export_building_hourly
python -m Patronage_Flow.export_station_hourly
```

Set `SMOKE_BBOX = (27000, 28000, 33000, 34500)` (SVY21 metres, CBD) in `Constants.py` for a quick test.
Full island: 171 k nodes, 139 k edges, 5.9 k origins, 118 k destinations; one betweenness call takes roughly
1 h on 12 cores (`N_WORKERS = 12`).

Verification of the released package (2026-09-10): `Patronage_Flow.Main` was run from this folder on the CBD
smoke bounding box (SVY21 27000-33000 E, 28000-34500 N) for the 14:00 weekday slot with the January 2026
ridership files: 15,582 madina edges, 756 origins, 13,400 destinations; the bus and rail sub-passes completed
in 46 s on 4 cores and wrote `flow_weekday_14.gpkg` (flow on 2,738 edges, p99 298, maximum 2,390 pedestrians)
and `flow_long.parquet`. On Windows the entry point must be run as a script or module (madina spawns worker
processes), which is why every runner keeps its `if __name__ == "__main__":` guard.

## 4. Outputs

| File | Content |
|---|---|
| `output/pedestrian_network_filtered.gpkg` | the pedestrian-passable OSM subset (142,095 segments) used by Modules 2 and 5 |
| `output/edges.gpkg`, `station_snap.gpkg` | madina edges (`parent_street_id`) and snapped origins (QA) |
| `output/flow_weekday_HH.gpkg` | per-slot pedestrian flow (`betweenness`, `flow`) - main deliverable |
| `output/flow_long.parquet` | long table `edge_id x DAY_TYPE x HOUR -> flow` |
| `output/2025_MM/`, `2025_annual/`, `2025_total/` | seasonal snapshots, four-quarter mean and true annual total |
| `output/building_hourly_weight.gpkg`, `station_hourly_ridership.gpkg` | hourly destination weights and station ridership (used by Module 4a) |

## 5. Path configuration

`Constants.ROOT` is the parent folder of the package; data folders (`Highway_OSM.gpkg`, `BusStopLocation_Aug2025/`,
`Train/`, `Station flow/<YYYY-MM>/node/`, `building/sg_buildings_v5.geojson`, `Island_boarder/`) are expected
under it. See `INPUT_DATA.md`.
