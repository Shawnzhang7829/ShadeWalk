# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Codebase Does

`Patronage_Flow` is a Singapore pedestrian-flow modelling pipeline. It distributes LTA transit ridership (bus stops + MRT exits) across the pedestrian street network using **madina betweenness centrality**, weighted by building GFA × EnergyPlus occupancy schedules. Output is per-edge hourly pedestrian flow as GeoPackage files for QGIS.

Working directory: `D:\Claude\UNA`. All scripts are run as modules from there.

---

## Running the Pipeline

```bash
# Standard single-pass (most common)
python -m Patronage_Flow.Main

# True dual-pass (egress + ingress; ~2× compute)
python -m Patronage_Flow.Main_dualpass

# 2025 quarterly snapshots (Mar/Jun/Sep/Dec, hours 12+14 only)
python -m Patronage_Flow.run_2025_quarterly

# 2025 annual total (all 12 months summed, then one betweenness run)
python -m Patronage_Flow.run_2025_annual_total

# Rebuild occupancy density table from EnergyPlus IDF files
python -m Patronage_Flow.build_occupancy_table

# QA exports (can run anytime, no madina dependency)
python -m Patronage_Flow.export_building_hourly
python -m Patronage_Flow.export_station_hourly

# Data inspection
python -m Patronage_Flow._inspect [command]
```

**Always launch Python via the pyenv executable directly** when using `Start-Process` in PowerShell — the shim (`python.bat`) breaks stdout redirection:
```powershell
$py = "C:\Users\City Syntax Lab\.pyenv\pyenv-win\versions\3.11.9\python.exe"
Start-Process $py -ArgumentList "-m", "Patronage_Flow.Main" -RedirectStandardOutput "out.log" -NoNewWindow
```

---

## All Tunable Parameters Live in `Constants.py`

Key knobs — no other file needs editing for normal runs:

| Parameter | Default | Notes |
|---|---|---|
| `SMOKE_BBOX` | `None` | Set to SVY21 tuple for quick test; `None` = full island |
| `HOURS_WEEKDAY` | `[8,10,12,14,16]` | Set to `list(range(24))` for full 24h |
| `HOURS_WEEKEND` | `[]` | |
| `N_WORKERS` | `12` | CPU cores for madina; ~12 is optimal on this Windows host |
| `BUS_RADIUS_M` | `400` | 5-min walk catchment |
| `RAIL_RADIUS_M` | `800` | 10-min walk catchment |
| `BETA` | `0.003` | Exponential decay (half-life ≈ 231 m) |
| `DETOUR_RATIO` | `1.05` | Admits near-shortest alternatives |
| `RIDERSHIP_MODE` | `"both"` | `"tap_in"`, `"tap_out"`, or `"both"` |
| `BUS_CSV` / `TRAIN_CSV` | `Station flow/2026-01/node/...` | Change for other months |

---

## Architecture

### Module Responsibilities

```
Constants.py          All paths and parameters — single source of truth
Network.py            Load highway/stations/buildings → madina Zonal
Flow_Computation.py   Ridership loading, weight management, betweenness loop
Main.py               Single-pass orchestrator (tap_in + tap_out combined)
Main_dualpass.py      Dual-pass: forward Zonal (egress) + reverse Zonal (ingress)
run_2025_quarterly.py Seasonal analysis: 4 months, 2 hours, reuses network
run_2025_annual_total.py 12-month sum → single betweenness run (fast)
build_occupancy_table.py EnergyPlus IDF parser → lookup/occupancy_density.parquet
export_building_hourly.py  Building destination weights as GPKG (QA)
export_station_hourly.py   Per-station ridership as GPKG (QA)
```

### Data Flow

```
OSM highway GPKG + Station SHPs + Building GeoJSON
        ↓  Network.py
madina Zonal  (171k nodes, 139k edges, 5.9k origins, 118k destinations)
        ↓
LTA ridership CSVs  →  load_ridership_table()  →  wide DF (PT_CODE × (DAY_TYPE, HOUR))
EnergyPlus parquet  →  load_occupancy_density() →  wide DF (archetype × (DAY_TYPE, HOUR))
        ↓  compute_hourly_flow()  — for each (DAY_TYPE, HOUR):
        │   1. set destination weights = building_GFA × occupancy_density[slot]
        │   2. BUS sub-pass:  400m radius → betweenness() → flow_A
        │   3. MRT sub-pass:  800m radius → betweenness() → flow_B
        │   4. flow = flow_A + flow_B  → write flow_<tag>.gpkg
        ↓
flow_long.parquet  [edge_id, DAY_TYPE, HOUR, flow]
```

### Critical Implementation Details

**Interchange station (slash-code) fix** — LTA CSVs encode interchanges as `"NS24/NE6/CC1"` (one row, combined ridership). `load_ridership_table()` divides volumes by the number of parts before exploding:
```python
n_parts = split["PT_CODE"].str.count("/") + 1
split["TOTAL_TAP_IN_VOLUME"]  /= n_parts
split["TOTAL_TAP_OUT_VOLUME"] /= n_parts
```
Omitting this inflates interchange stations by n× (3× for Dhoby Ghaut).

**MRT exit weighting** — Each station has N physical exits (`weight_divisor = N`). Ridership is divided equally per exit so the total injected weight is preserved: `per-exit weight = ridership[PT_CODE] / N_exits`.

**Origin masking speedup** — madina runs BFS for every `type == "origin"` node even if `weight == 0`. Before each sub-pass, non-target origins are renamed to `"origin_inactive"` (unrecognised type → skipped), giving ~3× speedup:
```python
_activate_origin_subset(zonal, stations_gdf, keep_source="BUS")   # before betweenness()
_restore_all_origins(zonal)                                         # after
```

**Annual total = single betweenness call** — Because betweenness is linear in origin weights: `betweenness(Σ months) = Σ betweenness(month)`. `run_2025_annual_total.py` sums raw monthly CSV volumes (no per-day normalisation) across all 12 months and runs once.

**Dual-pass** — `Main_dualpass.py` builds two Zonals from the same network data:
- `zA`: stations (origins, weight=`tap_out`) → buildings (destinations, weight=occupancy)  — models post-alighting egress
- `zB`: buildings (origins, weight=occupancy) → stations (destinations, weight=`tap_in`) — models pre-boarding ingress

### Output Directory Layout

```
Patronage_Flow/output/
  edges.gpkg                     madina edge geometries (QA)
  station_snap.gpkg              snapped origin positions (QA)
  flow_weekday_HH.gpkg           per-slot pedestrian flow (main deliverable)
  flow_long.parquet              long table: edge_id × DAY_TYPE × HOUR → flow
  2025_03/ … 2025_12/            quarterly snapshots (hours 12+14)
  2025_annual/                   4-quarter mean (from run_2025_quarterly)
  2025_total/                    true 12-month cumulative total
    flow_weekday_12.gpkg
    flow_weekday_14.gpkg
    annual_ridership_table.parquet
    node_all_months_2025.csv      raw 12-month combined node data
```

---

## Known Issues / Gotchas

- **Windows cp1252 encoding**: avoid Chinese characters in `print()` statements inside scripts run with stdout redirected to a file — they produce `UnicodeEncodeError` and exit code 1 even if all outputs have already been written.
- **`SMOKE_BBOX` units**: must be in SVY21 metres (EPSG:3414), not lat/lon. Example CBD bbox: `(27000, 28000, 33000, 34500)`.
- **madina pandas 3.0 compatibility**: madina was written for pandas < 2.0. Patches are applied at import time (see `project_madina_fixes.md` in memory/).
- **`N_WORKERS = 12`** is the empirical sweet spot on this Windows host. Higher values cause UI lag without proportional speedup (memory-bandwidth bound).
- **LTA data path**: `BUS_CSV` / `TRAIN_CSV` must be updated in `Constants.py` when switching data months. The path `ROOT / "Station flow" / "YYYY-MM" / "node" / ...` is correct; an old Windows directory junction at `ROOT / "2026-01"` may mask path errors.
