"""Tiny end-to-end test of madina on a CBD bbox before refactoring the framework."""
import geopandas as gpd
import pandas as pd
import warnings
warnings.simplefilter("ignore")

from madina.zonal import Zonal
from madina.una.tools import betweenness   # the proper, current API


def main():
    BBOX = (27000, 28000, 33000, 34500)
    PED = ["footway", "pedestrian", "path", "steps", "living_street",
           "corridor", "track", "cycleway", "service", "residential"]

    print("[1] loading highway gpkg ...")
    hw = gpd.read_file(r"D:/Claude/UNA/Highway_OSM.gpkg",
                       layer="highway", bbox=BBOX,
                       columns=["highway", "foot", "access", "geometry"])
    print(f"    bbox features         : {len(hw):,}")
    hw = hw[hw["highway"].isin(PED)].copy()
    hw = hw[~hw["foot"].isin(["no", "private"])]
    hw = hw[~hw["access"].isin(["no", "private"])]
    print(f"    after foot/access fix : {len(hw):,}")
    hw = hw[["geometry"]].reset_index(drop=True).set_crs("EPSG:3414", allow_override=True)

    print("[2] loading stations ...")
    bus = gpd.read_file(r"D:/Claude/UNA/BusStopLocation_Aug2025/BusStop.shp", bbox=BBOX)
    bus = bus[["BUS_STOP_N", "geometry"]].rename(columns={"BUS_STOP_N": "PT_CODE"})
    bus["PT_CODE"] = bus["PT_CODE"].astype(str)
    bus["W"] = 1.0
    bus = bus.set_crs("EPSG:3414", allow_override=True)
    print(f"    bus stops in bbox     : {len(bus)}")

    print("[3] loading buildings ...")
    b = gpd.read_file(r"D:/Claude/UNA/SG_Building_SVY21_TH/SG_Building_SVY21_TH.shp",
                      bbox=BBOX, columns=["floorarea_", "geometry"])
    b = b.dropna(subset=["floorarea_"])
    b["geometry"] = b.geometry.centroid
    b = b.rename(columns={"floorarea_": "W"})
    b = b.set_crs("EPSG:3414", allow_override=True)
    print(f"    building points       : {len(b)}")

    print("[4] building Zonal ...")
    z = Zonal()
    z.load_layer("streets", hw)
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=1.0,
                            redundant_edge_treatment="discard")
    print(f"    network nodes/edges   : {len(z.network.nodes)} / {len(z.network.edges)}")
    z.load_layer("stations", bus)
    z.insert_node("stations", label="origin", weight_attribute="W")
    z.load_layer("buildings", b)
    z.insert_node("buildings", label="destination", weight_attribute="W")
    z.create_graph(light_graph=True, d_graph=True)
    print("    node types            :",
          z.network.nodes["type"].value_counts().to_dict())

    print("[5] running betweenness (single-core for smoke) ...")
    import time
    t0 = time.perf_counter()
    betweenness(
        zonal=z,
        search_radius=800,
        detour_ratio=1.0,
        decay=True, decay_method="exponent", beta=0.003,
        num_cores=1,
        closest_destination=False,
        save_betweenness_as="flow",
    )
    print(f"    done in {time.perf_counter()-t0:.1f} s")

    ed = z.network.edges
    print("\nbetweenness summary on network edges:")
    print(ed["betweenness"].describe(percentiles=[0.5, 0.9, 0.99]).round(3).to_string())
    print("non-zero edges :", (ed["betweenness"] > 0).sum(), "/", len(ed))


if __name__ == "__main__":
    main()
