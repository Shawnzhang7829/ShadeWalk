"""
Export the pedestrian-passable subset of the OSM highway layer.

Reads Highway_OSM.gpkg, keeps the pedestrian-passable features (loaders.load_pedestrian_highway: `highway` tag in
PEDESTRIAN_HIGHWAYS, no foot / access = no or private, vehicle-only service sub-types and expressway = yes removed,
multi-part lines exploded, clipped to the island boundary) and writes them with their length.

Output: output/pedestrian_network_filtered.gpkg
  geometry : LineString (SVY21)
  length_m : segment length in metres
Used by Module 2 (pedestrian-network filter of the covered-linkway product) and Module 5 (comparison layer).

Run:
    python -m demand_inputs.export_pedestrian_subset      (from the folder that contains demand_inputs/)
"""
from __future__ import annotations
import warnings
warnings.simplefilter("ignore")

from demand_inputs import Constants as C
from demand_inputs.loaders import load_pedestrian_highway


def main():
    bbox = tuple(C.SMOKE_BBOX) if C.SMOKE_BBOX is not None else None
    if bbox:
        print(f"[export] SMOKE BBOX active = {bbox}")
    hw = load_pedestrian_highway(bbox=bbox)
    hw_export = hw.copy()
    hw_export["length_m"] = hw_export.geometry.length
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hw_export_path = C.OUTPUT_DIR / "pedestrian_network_filtered.gpkg"
    hw_export.to_file(hw_export_path, driver="GPKG")
    print(f"[export] wrote {hw_export_path.name} ({len(hw_export):,} edges, "
          f"total length = {hw_export['length_m'].sum()/1000:.1f} km)")


if __name__ == "__main__":
    main()
