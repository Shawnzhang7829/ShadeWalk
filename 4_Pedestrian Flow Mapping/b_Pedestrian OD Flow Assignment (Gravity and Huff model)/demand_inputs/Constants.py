"""
demand_inputs / Constants.py

Paths and parameters of the demand-input tables (station ridership, building weights, occupancy density,
OSM pedestrian subset).  Every tunable knob lives here so the rest of the package stays declarative.
The routing / flow scripts of `routing_flow/` keep their own constants at the top of each file.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent   # the module folder (the parent of demand_inputs/); data folders live under it
LOOKUP_DIR = Path(__file__).resolve().parent / "lookup"

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
# 595 individual MRT/LRT exits with stn_name + exit_code. v3 uses these as
# origin points instead of one centroid per station -- correctly distributes
# station ridership across all real boarding/alighting points.
MRT_EXIT_SHP = ROOT / "Train" / "TrainStationExit_Feb2025" / "Train_Station_Exit_Layer.shp"
BUS_CSV   = ROOT / "Station flow" / "2026-01" / "node" / "transport_node_bus_202601.csv"
TRAIN_CSV = ROOT / "Station flow" / "2026-01" / "node" / "transport_node_train_202601.csv"
MRT_LOOKUP_CSV = LOOKUP_DIR / "mrt_code_to_name.csv"

# building footprints (destinations in the trip-distribution model)
# v3 used the URA shapefile with only `floorarea_`. v4 uses sg_buildings_v5
# which adds `building_archetype` so we can apply per-use-class multipliers.
BUILDING_GEOJSON       = ROOT / "building" / "sg_buildings_v5.geojson"
BUILDING_GFA_COL       = "gross_floor_area"
BUILDING_ARCHETYPE_COL = "building_archetype"
# released building dataset (data record, folder 5_base_data): its `gfa_corr` is the gross floor area of the weights
BUILDING_RELEASED_SHP  = ROOT / "5_base_data" / "SG_buildings_footprint_height_function" / "SG_buildings_footprint_height_function.shp"

# Per-archetype, per-hour, per-day_type occupancy density table (people / m^2).
# Built from AllArhcetypes_SGP_2025_V5/*.idf via build_occupancy_table.py:
# density(arch, day_type, hour) = People per Floor Area * Schedule:Compact value
# Final destination weight = gross_floor_area * density(arch, day_type, hour).
IDF_DIR = ROOT / "AllArhcetypes_SGP_2025_V5"
OCCUPANCY_DENSITY_PARQUET = LOOKUP_DIR / "occupancy_density.parquet"

# Density assumed for archetypes missing from the lookup (= empty building).
OCCUPANCY_DEFAULT_DENSITY = 0.0

# Singapore island boundary (mainland + offshore islands, 14 polygons).
# Highway edges and stations are filtered to this polygon; anything outside
# the territorial limit is dropped.
ISLAND_BORDER_SHP = ROOT / "Island_boarder" / "Island_boarder.shp"

# outputs
OUTPUT_DIR = ROOT / "output"

# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------
TARGET_CRS = "EPSG:3414"   # SVY21, units in metres

# ---------------------------------------------------------------------------
# Ridership
# ---------------------------------------------------------------------------
# divide LTA's monthly sum to get a per-day mean.
WEEKDAYS_PER_MONTH         = 22.0
WEEKEND_HOLIDAYS_PER_MONTH = 9.0

# ---------------------------------------------------------------------------
# Run-mode flags
# ---------------------------------------------------------------------------
# smoke-test bbox in SVY21 metres. Set to None for full Singapore.
SMOKE_BBOX = None
# SMOKE_BBOX = (27000, 28000, 33000, 34500)   # CBD / Marina (smoke)
