"""
Patronage_Flow / Constants.py    (v2 — madina + buildings)

Paths and run parameters. Mirrors the role of UNA's Constants.py.
Every tunable knob lives here so the rest of the pipeline stays declarative.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # D:/Claude/UNA

# pedestrian network: OSM highway gpkg (replaces the old Footpath shapefile,
# which was a non-routable collection of pavement runs)
HIGHWAY_GPKG  = ROOT / "Highway_OSM.gpkg"
HIGHWAY_LAYER = "highway"

# OSM `highway` tags that we treat as pedestrian-passable.
# Conservative inclusion -- adjust to taste:
#   strict pedestrian-only : footway pedestrian path steps living_street corridor track
#   shared / sidewalked    : cycleway service residential
# In SG, primary/secondary/tertiary roads usually have sidewalks too. Add them
# here if the research needs the full street-front pedestrian network.
PEDESTRIAN_HIGHWAYS = [
    "footway", "pedestrian", "path", "steps", "living_street",
    "corridor", "track", "cycleway",
    "service", "residential",
    # "tertiary", "secondary", "primary",  # uncomment for full sidewalk network
]
# OSM tags that, when set, mean "no pedestrians": filtered out of the network.
EXCLUDE_FOOT_VALUES   = {"no", "private"}
EXCLUDE_ACCESS_VALUES = {"no", "private"}

# `service` sub-types that are vehicle-only in practice and should be removed
# even though their highway tag (`service`) is in PEDESTRIAN_HIGHWAYS.
#   parking_aisle    : lanes inside parking lots
#   driveway         : private property access drives
#   drive-through    : fast-food / bank drive-thrus
#   emergency_access : fire lanes, etc.
EXCLUDE_SERVICE_VALUES = {"parking_aisle", "driveway", "drive-through", "emergency_access"}

# Drop expressway=yes (PIE / CTE / KPE etc.) — these are limited-access
# motorways often mis-tagged with non-motorway highway values.
EXCLUDE_EXPRESSWAY_VALUES = {"yes"}

# stations and ridership
BUS_SHP   = ROOT / "BusStopLocation_Aug2025" / "BusStop.shp"
MRT_SHP   = ROOT / "Train" / "TrainStation_Aug2025" / "RapidTransitSystemStation.shp"
# 595 individual MRT/LRT exits with stn_name + exit_code. v3 uses these as
# origin points instead of one centroid per station -- correctly distributes
# station ridership across all real boarding/alighting points.
MRT_EXIT_SHP = ROOT / "Train" / "TrainStationExit_Feb2025" / "Train_Station_Exit_Layer.shp"
BUS_CSV   = ROOT / "Station flow" / "2026-01" / "node" / "transport_node_bus_202601.csv"
TRAIN_CSV = ROOT / "Station flow" / "2026-01" / "node" / "transport_node_train_202601.csv"
MRT_LOOKUP_CSV = ROOT / "Patronage_Flow" / "lookup" / "mrt_code_to_name.csv"

# building footprints (destinations in the trip-distribution model)
# v3 used the URA shapefile with only `floorarea_`. v4 uses sg_buildings_v5
# which adds `building_archetype` so we can apply per-use-class multipliers.
BUILDING_GEOJSON       = ROOT / "building" / "sg_buildings_v5.geojson"
BUILDING_GFA_COL       = "gross_floor_area"
BUILDING_ARCHETYPE_COL = "building_archetype"

# Per-archetype, per-hour, per-day_type occupancy density table (people / m^2).
# Built from AllArhcetypes_SGP_2025_V5/*.idf via build_occupancy_table.py:
# density(arch, day_type, hour) = People per Floor Area * Schedule:Compact value
# Final destination weight = gross_floor_area * density(arch, day_type, hour).
OCCUPANCY_DENSITY_PARQUET = ROOT / "Patronage_Flow" / "lookup" / "occupancy_density.parquet"

# Density assumed for archetypes missing from the lookup (= empty building).
OCCUPANCY_DEFAULT_DENSITY = 0.0

# Singapore island boundary (mainland + offshore islands, 14 polygons).
# Highway edges and stations are filtered to this polygon; anything outside
# the territorial limit is dropped.
ISLAND_BORDER_SHP = ROOT / "Island_boarder" / "Island_boarder.shp"

# outputs
OUTPUT_DIR = ROOT / "Patronage_Flow" / "output"

# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------
TARGET_CRS = "EPSG:3414"   # SVY21, units in metres

# ---------------------------------------------------------------------------
# Network parameters (madina Zonal.create_street_network)
# ---------------------------------------------------------------------------
# tolerance for fusing coincident endpoints during network construction.
# OSM Highway is properly noded, so a tiny tolerance suffices.
NODE_SNAPPING_TOLERANCE_M = 1.0

# how to handle parallel edges between the same pair of nodes
# 'discard' keeps the shortest, 'split' splits at midpoint, 'keep' is unsafe
REDUNDANT_EDGE_TREATMENT = "discard"

# turn penalty (madina supports it). Set to 0 to disable.
TURN_THRESHOLD_DEG = 45
TURN_PENALTY_M     = 0   # was 30 in madina docs; we disable for v1 simplicity

# stations farther than this from any network node will be dropped with a warning.
STATION_SNAP_MAX_M = 60.0

# ---------------------------------------------------------------------------
# Flow / betweenness parameters
# ---------------------------------------------------------------------------
# walking catchment (m). v5 differentiates by transit type:
#   bus  -> 400 m  (~5-min walk, frequent service catchment)
#   MRT/LRT -> 800 m (~10-min walk, heavy/light rail TOD catchment)
# Used as two sub-passes per slot in compute_hourly_flow.
RADIUS_M       = 800.0   # legacy / fallback (used if no per-source split)
BUS_RADIUS_M   = 400.0
RAIL_RADIUS_M  = 800.0

# decay coefficient for the exponential kernel  k(d) = exp(-beta * d).
# beta = 0.003 is madina's recommended pedestrian-walking value.
# (beta = 1/d_half * ln 2  -> beta = 0.003 means d_half = 231 m.)
BETA = 0.003

# Should pedestrians take only the shortest path, or distribute across
# alternative paths up to detour_ratio? 1.0 = shortest only.
# 1.05 admits a small set of near-shortest alternatives -> visible smoothing
# at moderate compute cost.
DETOUR_RATIO = 1.05

# closest_destination=True  : people go only to nearest building (simplistic)
# closest_destination=False : Huff-style competition across all reachable
#                              destinations within radius (recommended)
CLOSEST_DESTINATION = False

# ridership emitted from each station per record:
#   "tap_out"  -- people walking AWAY from the station (egress, post-arrival)
#   "tap_in"   -- people walking TOWARD the station (ingress, pre-departure)
#   "both"     -- tap_in + tap_out (any pedestrian associated with station)
RIDERSHIP_MODE = "both"

# divide LTA's monthly sum to get a per-day mean.
WEEKDAYS_PER_MONTH         = 22.0
WEEKEND_HOLIDAYS_PER_MONTH = 9.0

# ---------------------------------------------------------------------------
# Hours to run
# ---------------------------------------------------------------------------
# Each (DAY_TYPE, HOUR) combination triggers ONE betweenness call.
# Full coverage = 24 * 2 = 48 calls. Override here for faster iteration.
# HOURS_WEEKDAY  = list(range(24))
# HOURS_WEEKEND  = list(range(24))
# Full coverage: 24h × 2 day types = 48 betweenness calls
# HOURS_WEEKDAY = list(range(24))
# HOURS_WEEKEND = list(range(24))

# Quick 5-slot test: 8, 10, 12, 14, 16 weekday only
HOURS_WEEKDAY = [8, 10, 12, 14, 16]
HOURS_WEEKEND = []

# ---------------------------------------------------------------------------
# Run-mode flags
# ---------------------------------------------------------------------------
# smoke-test bbox in SVY21 metres. Set to None for full Singapore.
SMOKE_BBOX = None
# SMOKE_BBOX = (27000, 28000, 33000, 34500)   # CBD / Marina (smoke)

# parallelism for madina betweenness. 12 cores empirically the sweet spot
# on this Windows host -- 16 caused excessive UI lag without proportional
# speedup (likely memory-bandwidth bound).
N_WORKERS = 12
